from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch

from src.config import AGENT_ENABLE_LIGHT_REACT, AGENT_ENABLE_REFLECTION, AGENT_MAX_INTENT_CHARS


class ReasoningDecisionMixin:
    """封装 Layer3 的规则判定、LLM 判定与反思修正逻辑。"""

    @staticmethod
    def _safe_lower(x) -> str:
        return str(x or "").strip().lower()

    @staticmethod
    def _is_first_person(text: str) -> bool:
        return any(p in (text or "") for p in ["我", "本人", "咱", "俺"])

    @staticmethod
    def _risk_rank(risk: str) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(str(risk or "low"), 1)

    @classmethod
    def _max_risk(cls, *risks: str) -> str:
        valid = [str(r or "low") for r in risks]
        return max(valid, key=cls._risk_rank) if valid else "low"

    def _infer_intent_from_signals(self, current_msg: Dict, role_hint: str = "other") -> str:
        """当 LLM 意图字段不可靠时，用规则信号回退一个稳定短语。"""
        text = str(current_msg.get("text", "") or "")
        first_person = self._is_first_person(text)

        if role_hint == "scammer":
            if any(x in text for x in ["卡号", "户名", "下发", "保证金", "换卡", "新卡", "收U", "开后台"]):
                return "资金操作/指令"
            if any(x in text for x in ["报警？", "别影响别人", "看警察抓谁", "让你消失"]):
                return "威胁/施压"
        if role_hint == "victim":
            if bool(current_msg.get("has_pii")) and first_person:
                return "求助/隐私泄露"
            if any(x in text for x in ["被骗", "没到账", "拉黑", "退钱", "报警", "报案", "还钱"]):
                return "求助/维权"

        if any(x in text for x in ["签到", "滴滴", "打卡", "路过", "冒泡"]):
            return "签到/噪声消息"
        if any(x in text for x in ["天气", "羡慕", "看看", "围观", "聊天", "专业啊"]):
            return "闲聊"
        if bool(current_msg.get("has_pii")):
            return "敏感信息传播"
        return "一般讨论"

    def _sanitize_intent(self, intent: str, current_msg: Dict, role_hint: str = "other") -> str:
        """清洗 LLM 输出的意图字段，确保长度、格式和语义都可控。"""
        cleaned = re.sub(r"\s+", " ", str(intent or "").replace("\n", " ")).strip()
        low = cleaned.lower()

        if cleaned in {
            "rule_fast_path",
            "system_message",
            "签到/噪声消息",
            "闲聊",
            "指令/威胁/资金操作",
            "求助/维权/隐私泄露",
        }:
            return cleaned

        if not cleaned or any(x in low for x in ["role:", "risk:", "intent:", "角色:", "风险:", "意图:"]):
            return self._infer_intent_from_signals(current_msg, role_hint)

        if len(cleaned) > AGENT_MAX_INTENT_CHARS:
            return self._infer_intent_from_signals(current_msg, role_hint)
        return cleaned

    def _build_rule_candidate(self, current_msg: Dict) -> Dict[str, str]:
        """先走纯规则候选路径，给后续 LLM 和反思步骤提供锚点。"""
        parsed = {"risk": "low", "role": "other", "intent": "rule_fast_path"}
        parsed = self._apply_hard_rules(current_msg, parsed)
        parsed = self._self_check(current_msg, parsed)

        if parsed.get("role") == "other":
            score = (
                float(current_msg.get("l1_risk_score", 0) or 0)
                + float(current_msg.get("l2_risk_score", 0) or 0)
            ) / 2
            if score >= 65:
                parsed["risk"] = "high"
            elif score >= 30:
                parsed["risk"] = "medium"
            else:
                parsed["risk"] = "low"

        parsed["intent"] = self._sanitize_intent(parsed.get("intent", ""), current_msg, parsed.get("role", "other"))
        return parsed

    def _compose_prompt(self, current_msg: Dict, context_str: str) -> str:
        """保持原有提示词模板不变，确保报告与分类风格连续。"""
        text = str(current_msg.get("text", "") or "").strip()
        return f"""
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

    def _estimate_decision_quality(
        self,
        current_msg: Dict,
        parsed: Dict[str, str],
        rule_candidate: Dict[str, str],
        context_bundle: Dict[str, object],
        llm_used: bool,
    ) -> Dict[str, object]:
        """对当前判定做置信度估算，供反思步骤决定是否回退到规则。"""
        text = str(current_msg.get("text", "") or "")
        first_person = self._is_first_person(text)
        has_pii = bool(current_msg.get("has_pii"))
        op_signal = any(k in text for k in ["卡号", "户名", "下发", "车队", "保证金", "换卡", "新卡", "收U"])
        victim_signal = any(k in text for k in ["被骗", "骗子", "没到账", "还钱", "求助", "报警", "报案"]) and (
            first_person or "你们" in text
        )

        issues: List[str] = []
        score = 0.58 if llm_used else 0.66

        role = str(parsed.get("role", "other"))
        risk = str(parsed.get("risk", "low"))
        intent = str(parsed.get("intent", "") or "").strip()

        if role in {"scammer", "victim", "other"}:
            score += 0.10
        else:
            issues.append("invalid_role")

        if risk in {"low", "medium", "high"}:
            score += 0.08
        else:
            issues.append("invalid_risk")

        if intent:
            score += 0.08
        else:
            issues.append("empty_intent")

        if role == rule_candidate.get("role"):
            score += 0.08
        elif rule_candidate.get("role") != "other":
            score -= 0.14
            issues.append("rule_conflict")

        if self._risk_rank(risk) >= self._risk_rank(rule_candidate.get("risk", "low")):
            score += 0.05
        elif self._risk_rank(rule_candidate.get("risk", "low")) > self._risk_rank(risk):
            score -= 0.07
            issues.append("risk_understated")

        if has_pii and risk == "low":
            score -= 0.10
            issues.append("pii_low_risk")

        if op_signal and role == "victim" and rule_candidate.get("role") == "scammer":
            score -= 0.12
            issues.append("op_signal_mismatch")

        if victim_signal and role == "scammer" and rule_candidate.get("role") == "victim":
            score -= 0.12
            issues.append("victim_signal_mismatch")

        if int(context_bundle.get("memory_hit_count", 0) or 0) + int(context_bundle.get("rag_hit_count", 0) or 0) > 0:
            score += 0.03

        score = max(0.0, min(0.99, score))
        level = "high" if score >= 0.82 else "medium" if score >= 0.62 else "low"
        return {
            "score": round(score, 3),
            "level": level,
            "issues": issues,
            "llm_used": llm_used,
            "rule_role": rule_candidate.get("role", "other"),
        }

    def _reflect_decision(
        self,
        current_msg: Dict,
        parsed: Dict[str, str],
        rule_candidate: Dict[str, str],
        quality: Dict[str, object],
    ) -> Dict[str, str]:
        """反思层只做“纠偏”，不改原始规则和输出格式。"""
        refined = {
            "risk": str(parsed.get("risk", "low") or "low"),
            "role": str(parsed.get("role", "other") or "other"),
            "intent": str(parsed.get("intent", "") or ""),
        }
        changed_fields: List[str] = []

        if refined["role"] not in {"scammer", "victim", "other"}:
            refined["role"] = rule_candidate.get("role", "other")
            changed_fields.append("role")
        if refined["risk"] not in {"low", "medium", "high"}:
            refined["risk"] = rule_candidate.get("risk", "low")
            changed_fields.append("risk")

        if AGENT_ENABLE_REFLECTION:
            issues = set(quality.get("issues", []) or [])
            should_follow_rules = (
                quality.get("level") == "low"
                or "rule_conflict" in issues
                or "op_signal_mismatch" in issues
                or "victim_signal_mismatch" in issues
            )

            if should_follow_rules and rule_candidate.get("role") != "other":
                if refined["role"] != rule_candidate.get("role"):
                    refined["role"] = rule_candidate.get("role", "other")
                    changed_fields.append("role")
                stronger_risk = self._max_risk(refined["risk"], rule_candidate.get("risk", "low"))
                if stronger_risk != refined["risk"]:
                    refined["risk"] = stronger_risk
                    changed_fields.append("risk")

            if "risk_understated" in issues:
                stronger_risk = self._max_risk(refined["risk"], rule_candidate.get("risk", "low"))
                if stronger_risk != refined["risk"]:
                    refined["risk"] = stronger_risk
                    changed_fields.append("risk")

        sanitized_intent = self._sanitize_intent(refined.get("intent", ""), current_msg, refined.get("role", "other"))
        if sanitized_intent != refined.get("intent", ""):
            refined["intent"] = sanitized_intent
            changed_fields.append("intent")

        rule_checked = self._apply_hard_rules(current_msg, dict(refined))
        rule_checked = self._self_check(current_msg, rule_checked)
        rule_checked["intent"] = self._sanitize_intent(
            rule_checked.get("intent", ""),
            current_msg,
            rule_checked.get("role", "other"),
        )
        rule_checked["reflection"] = {
            "revised": bool(changed_fields),
            "changed_fields": changed_fields,
            "issues_before": list(quality.get("issues", []) or []),
        }
        return rule_checked

    def _react_analyze(self, current_msg: Dict, current_vec: Optional[torch.Tensor]) -> Dict[str, str]:
        """轻量 ReAct 路径：规则锚定 -> 上下文召回 -> LLM -> 反思修正。"""
        context_bundle = self._build_context_bundle(current_msg, current_vec)
        rule_candidate = self._build_rule_candidate(current_msg)

        steps = [
            {
                "action": "rules",
                "observation": (
                    f"rule_role={rule_candidate.get('role', 'other')}, "
                    f"rule_risk={rule_candidate.get('risk', 'low')}"
                ),
            },
            {
                "action": "retrieve_context",
                "observation": (
                    f"memory_hits={context_bundle.get('memory_hit_count', 0)}, "
                    f"rag_hits={context_bundle.get('rag_hit_count', 0)}"
                ),
            },
        ]

        prompt = self._compose_prompt(current_msg, str(context_bundle.get("context_str", "无可用上下文")))
        raw_result = self.llm.generate_response(prompt)
        steps.append({"action": "llm", "observation": (raw_result or "")[:96]})

        parsed = self._parse_result(raw_result)
        parsed = self._apply_hard_rules(current_msg, parsed)
        parsed = self._self_check(current_msg, parsed)
        parsed["intent"] = self._sanitize_intent(parsed.get("intent", ""), current_msg, parsed.get("role", "other"))

        quality_before = self._estimate_decision_quality(current_msg, parsed, rule_candidate, context_bundle, llm_used=True)
        final_decision = self._reflect_decision(current_msg, parsed, rule_candidate, quality_before)
        final_quality = self._estimate_decision_quality(
            current_msg,
            final_decision,
            rule_candidate,
            context_bundle,
            llm_used=True,
        )

        steps.append(
            {
                "action": "reflect",
                "observation": (
                    f"quality={final_quality.get('score', 0)}, "
                    f"revised={final_decision.get('reflection', {}).get('revised', False)}"
                ),
            }
        )

        final_decision["quality"] = final_quality
        final_decision["agent_trace"] = {
            "mode": "light_react_llm",
            "tools_used": ["rules", "memory", "rag", "llm", "reflection"],
            "steps": steps,
            "memory_hits": context_bundle.get("memory_hit_count", 0),
            "rag_hits": context_bundle.get("rag_hit_count", 0),
        }
        return final_decision

    def _self_check(self, current_msg: Dict, parsed: Dict[str, str]) -> Dict[str, str]:
        """对初步判定做一轮规则自检，尽量避免明显错判。"""
        text = str(current_msg.get("text", "") or "")
        first_person = self._is_first_person(text)
        has_pii = bool(current_msg.get("has_pii"))

        op_keywords = ["卡号", "户名", "下发", "车队", "保证金", "开后台", "换卡", "新卡", "收U"]
        victim_keywords = ["被骗", "骗子", "还钱", "没到账", "求助", "报警", "报案", "拉黑", "退钱", "维权"]

        op_signal = any(k in text for k in op_keywords)
        victim_signal = any(k in text for k in victim_keywords) and (first_person or "你们" in text)
        self_pii_signal = has_pii and first_person
        has_long_number = bool(re.search(r"(?<!\d)\d{15,19}(?!\d)", text))

        suggestion = None
        if parsed.get("role") == "other":
            if op_signal and (has_long_number or has_pii) and not self_pii_signal:
                suggestion = "scammer"
            elif victim_signal and (self_pii_signal or has_pii):
                suggestion = "victim"

        result = {
            "flags": {
                "op_signal": op_signal,
                "victim_signal": victim_signal,
                "self_pii_signal": self_pii_signal,
                "long_number": has_long_number,
            },
            "suggested_role": suggestion,
            "applied": False,
        }

        if suggestion and parsed.get("role") == "other":
            parsed["role"] = suggestion
            if self._risk_rank(parsed.get("risk")) < 2:
                parsed["risk"] = "medium"
            result["applied"] = True

        parsed["self_check"] = result
        return parsed

    def _parse_result(self, raw: str) -> Dict[str, str]:
        """兼容多种 LLM 输出格式，但最终仍压缩成固定三元组。"""
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
        """保留原有硬规则优先级，确保关键高危语句不会被 LLM 稀释。"""
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
        """低成本路径：不调用 LLM，只保留规则 + 反思闭环。"""
        text = str(current_msg.get("text", "") or "").strip()
        username = self._safe_lower(current_msg.get("username"))

        if not text or current_msg.get("is_system_msg") or username == "system":
            return {"risk": "low", "role": "other", "intent": "system_message"}

        parsed = self._build_rule_candidate(current_msg)
        quality = self._estimate_decision_quality(current_msg, parsed, parsed, {}, llm_used=False)
        parsed = self._reflect_decision(current_msg, parsed, parsed, quality)
        parsed["quality"] = self._estimate_decision_quality(current_msg, parsed, parsed, {}, llm_used=False)
        parsed["agent_trace"] = {
            "mode": "fast_path",
            "tools_used": ["rules", "heuristic_risk", "reflection"],
            "steps": [
                {
                    "action": "rules",
                    "observation": f"role={parsed.get('role', 'other')}, risk={parsed.get('risk', 'low')}",
                },
                {
                    "action": "reflect",
                    "observation": f"quality={parsed.get('quality', {}).get('score', 0)}",
                },
            ],
        }

        current_vec = self.embedder.get_embedding(text)
        self._append_memory(current_msg, current_vec)
        return parsed

    def analyze(self, current_msg: Dict) -> Dict[str, str]:
        """标准路径：按原顺序执行嵌入、检索、推理和记忆写回。"""
        text = str(current_msg.get("text", "") or "").strip()
        username = self._safe_lower(current_msg.get("username"))

        if not text or current_msg.get("is_system_msg") or username == "system":
            return {"risk": "low", "role": "other", "intent": "system_message"}

        current_vec = self.embedder.get_embedding(text)
        if AGENT_ENABLE_LIGHT_REACT:
            parsed = self._react_analyze(current_msg, current_vec)
        else:
            context_str = self._build_context(current_msg, current_vec)
            prompt = self._compose_prompt(current_msg, context_str)
            raw_result = self.llm.generate_response(prompt)
            parsed = self._parse_result(raw_result)
            parsed = self._apply_hard_rules(current_msg, parsed)
            parsed = self._self_check(current_msg, parsed)
            parsed["intent"] = self._sanitize_intent(parsed.get("intent", ""), current_msg, parsed.get("role", "other"))

        self._append_memory(current_msg, current_vec)
        return parsed
