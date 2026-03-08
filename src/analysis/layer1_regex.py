import re
from typing import Dict, List

from src.config import KEYWORD_RULES, REGEX_PATTERNS


class RegexAnalyzer:
    def __init__(self):
        self.patterns = {k: re.compile(v) for k, v in REGEX_PATTERNS.items()}

    @staticmethod
    def _normalize_matches(matches) -> List[str]:
        normalized = []
        for m in matches:
            if isinstance(m, tuple):
                m = next((x for x in m if x), "")
            m = str(m).strip()
            if m:
                normalized.append(m)
        return normalized

    @staticmethod
    def _clean_name_candidates(names: List[str]) -> List[str]:
        bad_tokens = {
            "客服", "系统", "银行卡", "身份证", "手机号", "地址", "本人", "学生", "兼职", "家人", "老婆", "兄弟"
        }
        bad_substrings = ["转给", "那个", "工行卡", "银行卡", "截图", "私聊", "就是", "那个群"]

        out = []
        for n in names:
            if n in bad_tokens:
                continue
            if any(bs in n for bs in bad_substrings):
                continue
            if re.search(r"\d", n):
                continue
            if len(n) < 2 or len(n) > 6:
                continue
            out.append(n)
        return out

    @staticmethod
    def _clean_address_candidates(addrs: List[str]) -> List[str]:
        out = []
        for a in addrs:
            if len(a) < 6:
                continue
            if not any(k in a for k in ["省", "市", "区", "县", "路", "街", "号", "地库", "广场", "大厦"]):
                continue
            out.append(a)
        return out

    def scan_pii(self, text: str) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}

        for name, pattern in self.patterns.items():
            matches = self._normalize_matches(pattern.findall(text or ""))
            if not matches:
                continue

            if name == "name_cn":
                matches = self._clean_name_candidates(matches)
            elif name == "address_cn":
                matches = self._clean_address_candidates(matches)

            if matches:
                results[f"extracted_{name}"] = sorted(set(matches))

        # Deduplicate overlapping regex results.
        id_values = set(results.get("extracted_id_card", []))
        if "extracted_bank_card" in results and id_values:
            cleaned_bank = [x for x in results["extracted_bank_card"] if x not in id_values]
            if cleaned_bank:
                results["extracted_bank_card"] = cleaned_bank
            else:
                del results["extracted_bank_card"]

        return results

    def match_keywords(self, text: str) -> List[str]:
        detected_topics = []
        for topic, keywords in KEYWORD_RULES.items():
            if any(k in (text or "") for k in keywords):
                detected_topics.append(topic)
        return detected_topics

    def detect_role_clues(self, text: str) -> str:
        victim_patterns = [
            r"被骗", r"报警", r"报案", r"没到账", r"还钱", r"拉黑", r"客服", r"求助", r"没钱了", r"提现"
        ]
        aggressor_patterns = [
            r"卡号", r"户名", r"下发", r"保证金", r"车队", r"进场", r"通道", r"踢了", r"收U", r"换卡"
        ]

        v_score = sum(1 for p in victim_patterns if re.search(p, text or ""))
        a_score = sum(1 for p in aggressor_patterns if re.search(p, text or ""))

        if v_score > a_score:
            return "potential_victim"
        if a_score > v_score:
            return "potential_aggressor"
        return "neutral"

    def process_single_message(self, message: Dict) -> Dict:
        text = message.get("text", "")
        if not text:
            return message

        pii_info = self.scan_pii(text)
        topics = self.match_keywords(text)

        l1_risk_score = 0
        l1_evidence = []

        if pii_info:
            pii_count = sum(len(v) for v in pii_info.values())
            pii_score = min(pii_count * 15, 50)
            l1_risk_score += pii_score
            l1_evidence.append(f"命中隐私信息({pii_count}): {list(pii_info.keys())}")

        topic_weights = {
            "fraud": 45,
            "gambling": 35,
            "trade": 25,
            "social": 10,
        }
        for topic in topics:
            l1_risk_score += topic_weights.get(topic, 10)
            l1_evidence.append(f"命中敏感话题: {topic}")

        role_clue = self.detect_role_clues(text)
        if role_clue != "neutral":
            l1_evidence.append(f"规则身份倾向: {role_clue}")

        l1_risk_score = min(l1_risk_score, 100)

        message.update(
            {
                "has_pii": bool(pii_info),
                "pii_details": pii_info,
                "preliminary_topics": topics,
                "l1_risk_score": l1_risk_score,
                "l1_evidence": l1_evidence,
                "l1_role_clue": role_clue,
                "risk_level": "high" if l1_risk_score > 60 else ("medium" if l1_risk_score > 20 else "low"),
            }
        )
        return message
