# src/analysis/layer1_regex.py
import re
from src.config import REGEX_PATTERNS, KEYWORD_RULES


class RegexAnalyzer:
    def __init__(self):
        # 预编译正则，提高处理速度
        self.patterns = {k: re.compile(v) for k, v in REGEX_PATTERNS.items()}

    def scan_pii(self, text):
        """扫描隐私信息 (PII)"""
        results = {}
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                # 去重并存入列表
                results[f"extracted_{name}"] = list(set(matches))
        return results

    def match_keywords(self, text):
        """基于关键词的初步话题分类"""
        detected_topics = []
        for topic, keywords in KEYWORD_RULES.items():
            # 只要命中一个关键词，就认为涉及该话题
            if any(k in text for k in keywords):
                detected_topics.append(topic)
        return detected_topics

    def detect_role_clues(self, text):
        """基于正则的身份线索检测"""
        # 受害者特征：提及损失、报警、被拉黑、求助
        victim_patterns = [r"被骗", r"报警", r"警察", r"钱.*没", r"还.*钱", r"没到账", r"生活费", r"拉黑", r"喂", r"客服"]
        # 施害者特征：指令、卡号下发、踢人、威胁
        aggressor_patterns = [r"踢了", r"拉黑他", r"滚", r"交钱", r"押金", r"下发", r"卡号", r"回款", r"速度", r"备好"]
        
        v_score = sum(1 for p in victim_patterns if re.search(p, text))
        a_score = sum(1 for p in aggressor_patterns if re.search(p, text))
        
        if v_score > a_score: return "potential_victim"
        if a_score > v_score: return "potential_aggressor"
        return "neutral"

    def process_single_message(self, message):
        """
        处理单条消息的主入口
        :param message: 原始消息字典 {'username':..., 'text':...}
        :return: 增加字段后的消息字典
        """
        text = message.get("text", "")
        if not text:
            return message

        # 1. 提取隐私信息
        pii_info = self.scan_pii(text)

        # 2. 关键词分类
        topics = self.match_keywords(text)

        # 3. 计算 L1 规则风险评分与证据
        l1_risk_score = 0
        l1_evidence = []

        # PII 评分逻辑
        if pii_info:
            pii_count = sum(len(v) for v in pii_info.values())
            # 每项 PII 加 20 分，最高 40 分（初步规则）
            pii_score = min(pii_count * 20, 40)
            l1_risk_score += pii_score
            l1_evidence.append(f"命中隐私信息({pii_count}项): {list(pii_info.keys())}")

        # 关键词评分逻辑 (根据不同话题赋予不同权重)
        topic_weights = {
            "fraud": 50,    # 诈骗类权重最高
            "gambling": 40, # 博彩类
            "trade": 30,    # 交易类
            "social": 10    # 社交类
        }
        
        for topic in topics:
            weight = topic_weights.get(topic, 10)
            l1_risk_score += weight
            l1_evidence.append(f"命中敏感话题: {topic}")

        # 4. 身份初步线索
        role_clue = self.detect_role_clues(text)
        if role_clue != "neutral":
            l1_evidence.append(f"正则身份倾向: {role_clue}")

        # 归一化/封顶 (0-100)
        l1_risk_score = min(l1_risk_score, 100)

        # 5. 构造增强数据
        analysis_result = {
            "has_pii": len(pii_info) > 0,
            "pii_details": pii_info,
            "preliminary_topics": topics,
            "l1_risk_score": l1_risk_score,
            "l1_evidence": l1_evidence,
            "l1_role_clue": role_clue,
            "risk_level": "high" if l1_risk_score > 60 else ("medium" if l1_risk_score > 20 else "low")
        }

        # 合并结果
        message.update(analysis_result)
        return message