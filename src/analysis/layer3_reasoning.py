from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from src.models.llm_wrapper import QwenWrapper


class ReasoningLayer:
    """
    Layer3: hybrid retrieval + LLM reasoning.

    RAG retrieval strategy:
    1) semantic recall over historical embeddings
    2) recent-window recall for dialogue continuity
    3) same-user recall for persona continuity
    4) keyword-overlap recall for topic continuity
    """

    def __init__(
        self,
        max_memory_size: int = 3000,
        recent_window: int = 16,
        semantic_top_k: int = 12,
        context_top_k: int = 6,
        max_context_chars: int = 1400,
    ):
        from src.models.embedding import EmbeddingEngine

        self.embedder = EmbeddingEngine()
        self.llm = QwenWrapper()

        self.max_memory_size = max_memory_size
        self.recent_window = recent_window
        self.semantic_top_k = semantic_top_k
        self.context_top_k = context_top_k
        self.max_context_chars = max_context_chars

        self.memory_pool: List[Dict] = []
        self.user_index: Dict[str, List[int]] = defaultdict(list)
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)

    @staticmethod
    def _normalize(v: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if v is None:
            return None
        if not isinstance(v, torch.Tensor):
            return None
        denom = torch.norm(v)
        if denom.item() == 0:
            return None
        return v / denom

    @staticmethod
    def _safe_lower(x) -> str:
        return str(x or "").strip().lower()

    @staticmethod
    def _is_first_person(text: str) -> bool:
        return any(p in (text or "") for p in ["我", "本人", "咱", "俺"])

    def _risk_prior(self, msg: Dict) -> float:
        l1 = float(msg.get("l1_risk_score", 0) or 0)
        l2 = float(msg.get("l2_risk_score", 0) or 0)
        return min((l1 + l2) / 200.0, 1.0)

    def _extract_keywords(self, msg: Dict) -> List[str]:
        kws = []
        kws.extend(msg.get("nlp_keywords", []) or [])
        kws.extend(msg.get("preliminary_topics", []) or [])
        return [str(k).strip() for k in kws if str(k).strip()]

    def _calc_keyword_overlap(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa = set(a)
        sb = set(b)
        union_n = len(sa | sb)
        if union_n == 0:
            return 0.0
        return len(sa & sb) / union_n

    def _build_context(self, current_msg: Dict, current_vec: Optional[torch.Tensor]) -> str:
        if not self.memory_pool:
            return "无历史上下文"

        cur_user = current_msg.get("username", "unknown")
        cur_keywords = self._extract_keywords(current_msg)

        candidate_idx = set()

        # A. recency recall
        start = max(0, len(self.memory_pool) - self.recent_window)
        candidate_idx.update(range(start, len(self.memory_pool)))

        # B. same-user recall
        user_hist = self.user_index.get(cur_user, [])
        candidate_idx.update(user_hist[-4:])

        # C. keyword-linked recall
        for kw in cur_keywords[:8]:
            linked = self.keyword_index.get(kw, [])
            candidate_idx.update(linked[-6:])

        # D. semantic recall
        normalized_current = self._normalize(current_vec)
        if normalized_current is not None:
            sem_scores = []
            for idx, item in enumerate(self.memory_pool):
                vec = item.get("norm_vec")
                if vec is None:
                    continue
                sim = torch.dot(normalized_current, vec).item()
                sem_scores.append((idx, sim))
            sem_scores.sort(key=lambda x: x[1], reverse=True)
            candidate_idx.update(idx for idx, _ in sem_scores[: self.semantic_top_k])

        if not candidate_idx:
            return "无可用上下文"

        scored = []
        for idx in candidate_idx:
            if idx < 0 or idx >= len(self.memory_pool):
                continue
            item = self.memory_pool[idx]

            sem = 0.0
            if normalized_current is not None and item.get("norm_vec") is not None:
                sem = torch.dot(normalized_current, item["norm_vec"]).item()

            age = len(self.memory_pool) - idx
            recency = math.exp(-age / 45.0)
            kw_overlap = self._calc_keyword_overlap(cur_keywords, item.get("keywords", []))
            risk_prior = item.get("risk_prior", 0.0)
            same_user = 1.0 if item.get("username") == cur_user else 0.0

            score = (
                0.56 * sem
                + 0.20 * recency
                + 0.12 * kw_overlap
                + 0.08 * risk_prior
                + 0.04 * same_user
            )
            scored.append((idx, score, sem, item))

        if not scored:
            return "无可用上下文"

        scored.sort(key=lambda x: x[1], reverse=True)

        chosen = []
        per_user_cap = 2
        per_user_count = defaultdict(int)

        for idx, score, sem, item in scored:
            user = item.get("username", "unknown")
            if per_user_count[user] >= per_user_cap:
                continue
            chosen.append((idx, score, sem, item))
            per_user_count[user] += 1
            if len(chosen) >= self.context_top_k:
                break

        if not chosen:
            return "无可用上下文"

        chosen.sort(key=lambda x: x[0])

        lines = []
        total_chars = 0
        for idx, score, sem, item in chosen:
            risk_val = int(round(item.get("risk_prior", 0.0) * 100))
            txt = str(item.get("text", "")).replace("\n", " ").strip()
            line = f"[{idx:04d}][{item.get('username', 'unknown')}][sim={sem:.2f}][r={risk_val}] {txt}"
            if total_chars + len(line) > self.max_context_chars:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines) if lines else "上下文超长，已截断"

    def _append_memory(self, msg: Dict, vec: Optional[torch.Tensor]):
        text = msg.get("text", "")
        if not text:
            return

        username = msg.get("username", "unknown")
        keywords = self._extract_keywords(msg)

        record = {
            "username": username,
            "text": text,
            "vec": vec,
            "norm_vec": self._normalize(vec),
            "risk_prior": self._risk_prior(msg),
            "keywords": keywords,
        }

        idx = len(self.memory_pool)
        self.memory_pool.append(record)
        self.user_index[username].append(idx)
        for kw in keywords:
            self.keyword_index[kw].append(idx)

        if len(self.memory_pool) > self.max_memory_size:
            self._prune_memory()

    def _prune_memory(self):
        self.memory_pool = self.memory_pool[-self.max_memory_size :]

        new_user_index = defaultdict(list)
        new_keyword_index = defaultdict(list)
        for i, item in enumerate(self.memory_pool):
            user = item.get("username", "unknown")
            new_user_index[user].append(i)
            for kw in item.get("keywords", []):
                new_keyword_index[kw].append(i)

        self.user_index = new_user_index
        self.keyword_index = new_keyword_index

    def _parse_result(self, raw: str) -> Dict[str, str]:
        txt = str(raw or "").strip()
        low = txt.lower()

        role = "other"
        if any(x in low for x in ["role: scammer", "角色: 嫌疑", "角色:scammer", "scammer", "嫌疑人"]):
            role = "scammer"
        elif any(x in low for x in ["role: victim", "角色: 受害", "角色:victim", "victim", "受害者"]):
            role = "victim"

        risk = "low"
        if any(x in low for x in ["risk: high", "风险: 高", "风险:high", " high", "|high"]):
            risk = "high"
        elif any(x in low for x in ["risk: medium", "风险: 中", "风险:medium", " medium", "|medium"]):
            risk = "medium"

        intent_match = re.search(r"(?:intent|意图)\s*[:：]\s*([^\n|]+)", txt, re.I)
        intent = intent_match.group(1).strip() if intent_match else txt

        return {"risk": risk, "role": role, "intent": intent}

    def _apply_hard_rules(self, current_msg: Dict, parsed: Dict[str, str]) -> Dict[str, str]:
        text = str(current_msg.get("text", "") or "")
        lower_text = text.lower()
        has_pii = bool(current_msg.get("has_pii"))
        first_person = self._is_first_person(text)

        noise_tokens = {"签到", "滴滴", "打卡", "路过", "冒泡"}
        if text.strip() in noise_tokens:
            return {"risk": "low", "role": "other", "intent": "签到/噪声消息"}

        casual_tokens = ["天气", "羡慕", "看看", "围观", "聊天", "专业啊"]
        if any(k in text for k in casual_tokens) and not has_pii:
            parsed["role"] = "other"
            parsed["risk"] = "low"
            parsed["intent"] = "闲聊"
            return parsed

        op_keywords = [
            "卡号",
            "户名",
            "下发",
            "车队",
            "待命",
            "保证金",
            "开后台",
            "进二群",
            "内部通道",
            "换卡",
            "新卡",
            "收U",
            "四件套",
            "白户",
            "踢",
            "滚",
        ]
        threat_patterns = [
            "你自己也是违法",
            "看警察抓谁",
            "报警？",
            "别影响别人",
            "懂吗",
            "让你消失",
        ]
        victim_keywords = [
            "被骗",
            "骗子",
            "还钱",
            "没到账",
            "求助",
            "救命",
            "报警",
            "报案",
            "拉黑",
            "退钱",
            "兼职",
            "刷单",
            "待审核",
            "审核好了吗",
            "不理我",
            "充值",
            "提现",
            "还要钱",
            "我转了",
        ]
        self_pii_cues = ["我的电话", "身份证号", "住址", "我这就拍", "照片都发了", "我是学生"]

        op_signal = any(k in text for k in op_keywords)
        threat_signal = any(k in text for k in threat_patterns)
        victim_signal = any(k in text for k in victim_keywords) and (first_person or "你们" in text)
        self_pii_signal = has_pii and first_person and any(k in text for k in self_pii_cues)

        has_long_number = bool(re.search(r"(?<!\d)\d{15,19}(?!\d)", lower_text))

        if threat_signal or (op_signal and (has_long_number or "卡号" in text or "保证金" in text) and not self_pii_signal):
            parsed["role"] = "scammer"
            if parsed["risk"] == "low":
                parsed["risk"] = "high" if has_pii or has_long_number else "medium"
            if not parsed.get("intent") or parsed["intent"] == "other":
                parsed["intent"] = "指令/威胁/资金操作"
            return parsed

        if self_pii_signal or (victim_signal and not op_signal):
            parsed["role"] = "victim"
            if parsed["risk"] == "low":
                parsed["risk"] = "medium"
            if not parsed.get("intent") or parsed["intent"] == "other":
                parsed["intent"] = "求助/维权/隐私泄露"
            return parsed

        if has_long_number and op_signal:
            parsed["role"] = "scammer"
            if parsed["risk"] == "low":
                parsed["risk"] = "medium"

        username = self._safe_lower(current_msg.get("username"))
        if (username.startswith("unknown") or "bot" in username) and not (victim_signal or op_signal):
            parsed["role"] = "other"
            parsed["risk"] = "low"

        return parsed

    def quick_analyze(self, current_msg: Dict) -> Dict[str, str]:
        """
        Low-cost inference path (no LLM call), used for token saving on low-risk messages.
        """
        text = str(current_msg.get("text", "") or "").strip()
        username = self._safe_lower(current_msg.get("username"))

        if not text or current_msg.get("is_system_msg") or username == "system":
            return {"risk": "low", "role": "other", "intent": "system_message"}

        base = {"risk": "low", "role": "other", "intent": "rule_fast_path"}
        parsed = self._apply_hard_rules(current_msg, base)

        if parsed.get("role") == "other":
            score = (float(current_msg.get("l1_risk_score", 0) or 0) + float(current_msg.get("l2_risk_score", 0) or 0)) / 2
            if score >= 65:
                parsed["risk"] = "high"
            elif score >= 30:
                parsed["risk"] = "medium"
            else:
                parsed["risk"] = "low"

        current_vec = self.embedder.get_embedding(text)
        self._append_memory(current_msg, current_vec)
        return parsed

    def analyze(self, current_msg: Dict) -> Dict[str, str]:
        text = str(current_msg.get("text", "") or "").strip()
        username = self._safe_lower(current_msg.get("username"))

        if not text or current_msg.get("is_system_msg") or username == "system":
            return {"risk": "low", "role": "other", "intent": "system_message"}

        current_vec = self.embedder.get_embedding(text)
        context_str = self._build_context(current_msg, current_vec)

        prompt = f"""
你是群聊风控分析助手。请根据消息、检索上下文与规则线索判断角色和风险。

[当前消息]
用户: {current_msg.get('username', 'unknown')}
内容: {text}

[规则线索]
L1风险分: {current_msg.get('l1_risk_score', 0)}
L2风险分: {current_msg.get('l2_risk_score', 0)}
L1证据: {current_msg.get('l1_evidence', [])}
L2证据: {current_msg.get('l2_evidence', [])}

[检索上下文]
{context_str}

请严格按以下格式输出一行:
Role: scammer/victim/other | Risk: high/medium/low | Intent: <一句话>
""".strip()

        raw_result = self.llm.generate_response(prompt)
        parsed = self._parse_result(raw_result)
        parsed = self._apply_hard_rules(current_msg, parsed)

        self._append_memory(current_msg, current_vec)
        return parsed

    def generate_comprehensive_report(self, group_stats, top_kols):
        suspect_assets = group_stats.get("suspect_assets", []) or []
        victim_leaks = group_stats.get("victim_leaks", []) or []
        victim_names = group_stats.get("victim_list", []) or []
        suspect_names = group_stats.get("suspect_list", []) or []
        irrelevant_names = set(group_stats.get("irrelevant_list", []) or [])

        core_users = []
        for name in suspect_names[:4]:
            if name not in core_users:
                core_users.append(name)
        for u in top_kols:
            name = u.get("username", "unknown")
            if name not in core_users:
                core_users.append(name)
            if len(core_users) >= 8:
                break

        if not victim_names:
            victim_names = ["暂无明确受害用户"]

        def format_evidence(items: List[Dict]) -> str:
            if not items:
                return "- 暂无"
            lines = []
            for e in items:
                lines.append(
                    f"- **{e.get('type', '其他')}**：{e.get('content', '')}"
                    f"（发布者：{e.get('owner', 'unknown')}，语境：{e.get('context', '')}）"
                )
            return "\n".join(lines)

        risk_level = "高" if (len(suspect_assets) >= 2 and len(victim_names) >= 1) else "中"

        role_map = {u.get("username", "unknown"): str(u.get("predicted_role", "other")) for u in top_kols}
        core_profiles = []
        for name in core_users[:8]:
            role = role_map.get(name, "other")
            if name in irrelevant_names:
                desc = "无关人士/水军，主要为签到或闲聊噪声。"
            elif name in victim_names:
                desc = "疑似受害者，其信息可能被用于非法资金活动。"
            elif name in suspect_names or role == "scammer":
                desc = "疑似嫌疑人，具备资金或指令操控特征。"
            else:
                desc = "群内活跃成员，需结合更多证据研判。"
            core_profiles.append(f"- **{name}**：{desc}")

        report = f"""**非法网络活动研判报告**

---

### 1. 案件定性
本案件属于**非法网络活动**，涉及**隐私信息泄露**与**资金相关操作**。核心人物包括**{', '.join(core_users[:4]) if core_users else '暂无'}**，其中疑似受害者为**{', '.join(victim_names)}**。

---

### 2. 核心成员画像
{chr(10).join(core_profiles) if core_profiles else '- 暂无'}

---

### 3. 隐私泄露与作案账号明细表

{format_evidence(victim_leaks)}

{format_evidence(suspect_assets)}

---

### 4. 打击建议
- **立即冻结涉案账户**，追查资金流向，锁定资金链。
- **对核心嫌疑人实施全链条打击**，包括指令发布、资金分发与执行节点。
- **对受害者开展保护与取证协助**，避免二次伤害。
- **持续监控群内关键词与账户复用行为**，防止同类活动扩散。

---
**风险等级：{risk_level}**"""

        return report
