from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch


class ReasoningContextMixin:
    """封装 Layer3 的上下文构建逻辑。

    这一层只负责把“当前消息可参考的历史证据”组织出来：
    1. 短期记忆池召回
    2. 同用户历史召回
    3. 关键词关联召回
    4. 向量语义召回
    5. 外部向量库 RAG 召回

    这样拆分后，`layer3_reasoning.py` 会保留为总入口文件，
    仍然符合原有三层架构，只是把实现细节下沉到了内部模块。
    """

    @staticmethod
    def _normalize(v: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """将向量做 L2 归一化，便于后续余弦相似度计算。"""
        if v is None:
            return None
        if not isinstance(v, torch.Tensor):
            return None
        denom = torch.norm(v)
        if denom.item() == 0:
            return None
        return v / denom

    def _risk_prior(self, msg: Dict) -> float:
        """把 L1/L2 风险分压缩到 0~1，作为历史消息的重要性先验。"""
        l1 = float(msg.get("l1_risk_score", 0) or 0)
        l2 = float(msg.get("l2_risk_score", 0) or 0)
        return min((l1 + l2) / 200.0, 1.0)

    def _extract_keywords(self, msg: Dict) -> List[str]:
        """统一抽取后续检索要用到的关键词来源。"""
        kws = []
        kws.extend(msg.get("nlp_keywords", []) or [])
        kws.extend(msg.get("preliminary_topics", []) or [])
        return [str(k).strip() for k in kws if str(k).strip()]

    def _calc_keyword_overlap(self, a: List[str], b: List[str]) -> float:
        """用交并比衡量两条消息的主题重叠程度。"""
        if not a or not b:
            return 0.0
        sa = set(a)
        sb = set(b)
        union_n = len(sa | sb)
        if union_n == 0:
            return 0.0
        return len(sa & sb) / union_n

    def _build_memory_context_lines(self, current_msg: Dict, current_vec: Optional[torch.Tensor]) -> List[str]:
        """从本轮运行中的记忆池提取最值得给 LLM 参考的历史消息。"""
        if not self.memory_pool:
            return []

        cur_user = current_msg.get("username", "unknown")
        cur_keywords = self._extract_keywords(current_msg)

        candidate_idx = set()

        # A. 最近窗口召回，保证对话连贯性。
        start = max(0, len(self.memory_pool) - self.recent_window)
        candidate_idx.update(range(start, len(self.memory_pool)))

        # B. 同一用户历史召回，保持人物画像连续。
        user_hist = self.user_index.get(cur_user, [])
        candidate_idx.update(user_hist[-4:])

        # C. 关键词关联召回，补足主题连续性。
        for kw in cur_keywords[:8]:
            linked = self.keyword_index.get(kw, [])
            candidate_idx.update(linked[-6:])

        # D. 语义向量召回，补足规则难以覆盖的近义表达。
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
            return []

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
            return []

        scored.sort(key=lambda x: x[1], reverse=True)

        chosen = []
        per_user_cap = 2
        per_user_count = defaultdict(int)

        # 限制单用户占比，避免上下文被同一人刷屏。
        for idx, score, sem, item in scored:
            user = item.get("username", "unknown")
            if per_user_count[user] >= per_user_cap:
                continue
            chosen.append((idx, score, sem, item))
            per_user_count[user] += 1
            if len(chosen) >= self.context_top_k:
                break

        if not chosen:
            return []

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
        return lines

    def _build_rag_context_lines(self, current_msg: Dict, current_vec: Optional[torch.Tensor]) -> List[str]:
        """从持久化向量库中召回跨批次可复用的历史线索。"""
        if current_vec is None:
            return []

        hits = self.vector_store.query(current_vec, top_k=self.rag_top_k)
        if not hits:
            return []

        seen_text = set()
        lines = []
        for h in hits:
            text = str(h.get("text", "") or "").replace("\n", " ").strip()
            if not text or text in seen_text:
                continue
            seen_text.add(text)

            meta = h.get("metadata", {}) or {}
            user = meta.get("username", "unknown")
            group = meta.get("source_group", "unknown")
            sim = float(h.get("score", 0.0) or 0.0)

            if self.rag_max_doc_chars and len(text) > self.rag_max_doc_chars:
                text = text[: self.rag_max_doc_chars] + "…"

            line = f"[RAG][{group}][{user}][sim={sim:.2f}] {text}"
            lines.append(line)
        return lines

    def _build_context_bundle(self, current_msg: Dict, current_vec: Optional[torch.Tensor]) -> Dict[str, object]:
        """统一输出给决策层使用的上下文包。"""
        memory_lines = self._build_memory_context_lines(current_msg, current_vec)
        rag_lines = self._build_rag_context_lines(current_msg, current_vec)

        if not memory_lines and not rag_lines:
            return {
                "memory_lines": memory_lines,
                "rag_lines": rag_lines,
                "memory_hit_count": 0,
                "rag_hit_count": 0,
                "context_str": "无可用上下文",
            }

        lines = []
        total_chars = 0
        for line in memory_lines + rag_lines:
            if total_chars + len(line) > self.max_context_chars:
                break
            lines.append(line)
            total_chars += len(line)

        context_str = "\n".join(lines) if lines else "上下文超长，已截断"
        return {
            "memory_lines": memory_lines,
            "rag_lines": rag_lines,
            "memory_hit_count": len(memory_lines),
            "rag_hit_count": len(rag_lines),
            "context_str": context_str,
        }

    def _build_context(self, current_msg: Dict, current_vec: Optional[torch.Tensor]) -> str:
        return str(self._build_context_bundle(current_msg, current_vec).get("context_str", "无可用上下文"))

    @staticmethod
    def _make_doc_id(msg: Dict) -> str:
        """保持向量库文档 ID 的生成规则不变，避免重复写入。"""
        group = msg.get("source_group", "unknown")
        file_name = msg.get("source_file", "unknown")
        idx = msg.get("msg_index", 0)
        user = msg.get("username", "unknown")
        base = f"{group}|{file_name}|{idx}|{user}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    def _append_memory(self, msg: Dict, vec: Optional[torch.Tensor]):
        """把当前消息同时写入内存记忆池和持久化向量库。"""
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

        if vec is None:
            return
        if msg.get("is_system_msg"):
            return

        metadata = {
            "username": msg.get("username", "unknown"),
            "source_group": msg.get("source_group", "unknown"),
            "source_file": msg.get("source_file", "unknown"),
            "msg_index": int(msg.get("msg_index", 0) or 0),
            "risk_prior": self._risk_prior(msg),
        }
        self.vector_store.add(self._make_doc_id(msg), text, vec, metadata=metadata)

    def _prune_memory(self):
        """裁剪记忆池后，顺带重建倒排索引，避免索引指针失效。"""
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
