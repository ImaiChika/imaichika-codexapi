import re
from typing import Dict, List

from src.config import KEYWORD_RULES, REGEX_PATTERNS


class RegexAnalyzer:
    def __init__(self):
        self.patterns = {k: re.compile(v) for k, v in REGEX_PATTERNS.items()}
        # 这些补充规则专门覆盖真实群聊里常见的“无固定标签”写法。
        self.qq_context_pattern = re.compile(r"(?:QQ|qq|Q号|q号|企鹅号)[^\n]{0,24}?([1-9]\d{4,11})")
        self.explicit_name_cn_pattern = re.compile(r"(?:Name|name)\s*[:：]\s*([\u4e00-\u9fa5·]{2,6})")
        self.explicit_name_en_pattern = re.compile(r"(?:Name|name)\s*[:：]\s*([A-Z][a-z]+(?:[A-Z][a-z]+){1,3})")
        self.leading_name_en_pattern = re.compile(r"^\s*([A-Z][a-z]+(?:[A-Z][a-z]+){1,3})\b")
        self.standalone_name_cn_pattern = re.compile(r"^[\u4e00-\u9fa5·]{2,4}$")
        self.inline_address_patterns = [
            re.compile(
                r"([\u4e00-\u9fa5]{2,4}省[\u4e00-\u9fa5]{2,6}市[\u4e00-\u9fa5]{1,4}(?:区|县)"
                r"[^\n，。；]{0,24}(?:路|街|大道|巷|道|号|基地|广场|大厦|园区|公寓)[^\n，。；]{0,16})"
            ),
            re.compile(
                r"([\u4e00-\u9fa5]{2,6}市[\u4e00-\u9fa5]{1,4}(?:区|县)"
                r"[^\n，。；]{0,24}(?:路|街|大道|巷|道|号|基地|广场|大厦|园区|公寓)[^\n，。；]{0,16})"
            ),
            re.compile(
                r"([\u4e00-\u9fa5]{1,4}(?:区|县)"
                r"[^\n，。；]{0,24}(?:路|街|大道|巷|道|号|基地|广场|大厦|园区|公寓)[^\n，。；]{0,16})"
            ),
        ]
        self.inline_region_patterns = [
            re.compile(r"([\u4e00-\u9fa5]{2,4}省[\u4e00-\u9fa5]{2,6}市(?:[\u4e00-\u9fa5]{1,4}(?:区|县))?)"),
            re.compile(r"([\u4e00-\u9fa5]{2,6}市(?:[\u4e00-\u9fa5]{1,4}(?:区|县))?)"),
        ]

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
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def _clean_name_candidates(names: List[str]) -> List[str]:
        bad_tokens = {
            "客服", "系统", "银行卡", "身份证", "手机号", "地址", "本人", "学生", "兼职", "家人", "老婆", "兄弟",
            "来一个", "公群老板",
        }
        bad_substrings = ["转给", "那个", "工行卡", "银行卡", "截图", "私聊", "就是", "那个群"]

        out = []
        for n in names:
            n = re.sub(r"^[\s:：,/|]+|[\s:：,/|]+$", "", str(n or ""))
            if not n:
                continue
            if n in bad_tokens:
                continue
            if any(bs in n for bs in bad_substrings):
                continue
            if re.search(r"\d", n):
                continue
            if not re.fullmatch(r"[\u4e00-\u9fa5·]{2,6}", n):
                continue
            if len(n) < 2 or len(n) > 6:
                continue
            out.append(n)
        return out

    @staticmethod
    def _clean_english_name_candidates(names: List[str]) -> List[str]:
        out = []
        for n in names:
            n = re.sub(r"^[\s:：,/|]+|[\s:：,/|]+$", "", str(n or ""))
            if not n:
                continue
            if not re.fullmatch(r"[A-Z][a-z]+(?:[A-Z][a-z]+){1,3}", n):
                continue
            if any(flag in n for flag in ["Admin", "System", "Unknown", "Bot"]):
                continue
            out.append(n)
        return out

    @staticmethod
    def _clean_address_candidates(addrs: List[str]) -> List[str]:
        noisy_prefixes = [
            "对方给我", "给我", "我湖北省", "我先发", "先发", "这条住址我先发", "住址能补到", "能补到", "一成", "成",
        ]
        province_pattern = re.compile(
            r"(北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)(?:省|市|自治区|特别行政区)?"
        )
        out = []
        for a in addrs:
            a = re.sub(r"\s+", " ", str(a or "")).strip(" ，。；;:：")
            for prefix in noisy_prefixes:
                if a.startswith(prefix):
                    a = a[len(prefix):].strip(" ，。；;:：")
            province_match = province_pattern.search(a)
            if province_match:
                a = a[province_match.start():]
            a = re.sub(r"(?:的住址|常驻地|口径统一成).*$", "", a).strip(" ，。；;:：")
            if re.match(r"^[\u4e00-\u9fa5]市", a):
                continue
            if len(a) < 6:
                continue
            geo_hits = sum(1 for k in ["省", "市", "区", "县", "路", "街", "号", "地库", "广场", "大厦", "大道", "基地", "园区"] if k in a)
            if geo_hits < 2:
                continue
            out.append(a)
        return out

    def _extract_contextual_qq(self, text: str) -> List[str]:
        return [m.group(1) for m in self.qq_context_pattern.finditer(text or "")]

    def _extract_contextual_names(self, text: str) -> Dict[str, List[str]]:
        names_cn: List[str] = []
        names_en: List[str] = []
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]

        for idx, line in enumerate(lines):
            names_cn.extend(self.explicit_name_cn_pattern.findall(line))
            names_en.extend(self.explicit_name_en_pattern.findall(line))

            # 真实样本里经常出现 “WangKang 139... 320...” 这种前置实体格式。
            has_strong_pii = bool(re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", line) or re.search(r"\d{17}[0-9Xx]", line))
            has_person_context = any(x in line for x in ["查", "查询", "机主", "三要素", "户籍", "家谱", "近照", "照片", "住址"])

            if has_strong_pii or has_person_context:
                match = self.leading_name_en_pattern.search(line)
                if match:
                    names_en.append(match.group(1))

            # 兼容 “王康 / 320... / 139...” 这种多行排布，避免放宽全局正则带来误报。
            if self.standalone_name_cn_pattern.fullmatch(line):
                next_window = " ".join(lines[idx + 1: idx + 3])
                if re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", next_window) or re.search(r"\d{17}[0-9Xx]", next_window):
                    names_cn.append(line)

        return {
            "name_cn": self._clean_name_candidates(names_cn),
            "name_en": self._clean_english_name_candidates(names_en),
        }

    def _extract_inline_addresses(self, text: str, pii_results: Dict[str, List[str]]) -> List[str]:
        candidates: List[str] = []
        strong_pii_present = any(
            pii_results.get(key)
            for key in [
                "extracted_mobile_cn",
                "extracted_id_card",
                "extracted_qq_number",
                "extracted_name_cn",
                "extracted_name_en",
            ]
        )

        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue

            for pattern in self.inline_address_patterns:
                candidates.extend(pattern.findall(line))

            # 对 “QQ + 手机号 + 地区” 这类短位置写法，只在同一行有其他强线索时补抓。
            if strong_pii_present or any(x in line for x in ["住址", "地址", "常驻地", "定位"]):
                for pattern in self.inline_region_patterns:
                    candidates.extend(pattern.findall(line))

        return self._clean_address_candidates(candidates)

    def scan_pii(self, text: str) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}

        for name, pattern in self.patterns.items():
            matches = self._normalize_matches(pattern.findall(text or ""))
            if not matches:
                continue

            if name == "name_cn":
                matches = self._clean_name_candidates(matches)
            elif name == "name_en":
                matches = self._clean_english_name_candidates(matches)
            elif name == "address_cn":
                matches = self._clean_address_candidates(matches)

            if matches:
                results[f"extracted_{name}"] = self._dedupe_keep_order(matches)

        # 下面这些补充提取不改变原流程，只把真实样本里更隐式的线索补上。
        contextual_qq = self._dedupe_keep_order(self._extract_contextual_qq(text))
        if contextual_qq:
            merged_qq = list(results.get("extracted_qq_number", [])) + contextual_qq
            results["extracted_qq_number"] = self._dedupe_keep_order(merged_qq)

        contextual_names = self._extract_contextual_names(text)
        if contextual_names["name_en"]:
            merged_name_en = list(results.get("extracted_name_en", [])) + contextual_names["name_en"]
            results["extracted_name_en"] = self._dedupe_keep_order(merged_name_en)
        if contextual_names["name_cn"]:
            merged_name_cn = list(results.get("extracted_name_cn", [])) + contextual_names["name_cn"]
            results["extracted_name_cn"] = self._dedupe_keep_order(merged_name_cn)

        inline_addresses = self._extract_inline_addresses(text, results)
        if inline_addresses:
            merged_addrs = list(results.get("extracted_address_cn", [])) + inline_addresses
            results["extracted_address_cn"] = self._dedupe_keep_order(merged_addrs)


        id_values = set(results.get("extracted_id_card", []))
        if "extracted_bank_card" in results and id_values:
            cleaned_bank = [x for x in results["extracted_bank_card"] if x not in id_values]
            if cleaned_bank:
                results["extracted_bank_card"] = cleaned_bank
            else:
                del results["extracted_bank_card"]

        wallet_values = set(results.get("extracted_usdt_address", []))
        if "extracted_payment_address" in results and wallet_values:
            cleaned_payment = [x for x in results["extracted_payment_address"] if x not in wallet_values]
            if cleaned_payment:
                results["extracted_payment_address"] = cleaned_payment
            else:
                del results["extracted_payment_address"]

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
