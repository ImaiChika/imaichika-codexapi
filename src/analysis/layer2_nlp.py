# src/analysis/layer2_nlp.py
import jieba
import jieba.analyse
import networkx as nx
import re
from collections import Counter
from src.config import STOPWORDS, SYSTEM_MSG_KEYWORDS


class TextMiner:
    """对应PPT ：NLP技术提取基础特征"""

    def __init__(self):
        # 注意：这里不再调用 jieba.analyse.set_stop_words
        self.stopwords = STOPWORDS

    def is_system_message(self, text):
        """简单判断是否为系统垃圾消息"""
        if not text:
            return True
        for kw in SYSTEM_MSG_KEYWORDS:
            if kw in text:
                return True
        return False

    def extract_keywords(self, text, top_k=5):
        """提取单条消息的关键词 (基于TF-IDF)"""
        if not text or self.is_system_message(text):
            return []

        # 1. 使用jieba提取关键词 (不指定停用词路径)
        # allowPOS=('n', 'v', 'vn') 限制只提取名词、动词、名动词
        tags = jieba.analyse.extract_tags(text, topK=top_k * 2, withWeight=False, allowPOS=('n', 'v', 'vn'))

        # 2. 手动过滤掉停用词，并限制长度大于1（过滤掉“我”、“有”等单字）
        filtered_tags = [
            t for t in tags
            if t not in self.stopwords and len(t) > 1
        ]

        # 3. 返回前 top_k 个
        return filtered_tags[:top_k]

    def process(self, message):
        """处理单条消息"""
        text = message.get("text", "")

        # 1. 判断是否有效用户发言
        is_sys = self.is_system_message(text)

        # 2. 提取关键词
        keywords = self.extract_keywords(text) if not is_sys else []

        # 3. 计算 L2 风险评分与证据
        l2_risk_score = 0
        l2_evidence = []

        if is_sys:
            l2_evidence.append("系统垃圾/广告消息过滤")
            l2_risk_score = 0 
        else:
            # 关键词深度分析 (简单演示：根据命中高危词的数量计算)
            # 这里可以引入更复杂的语义评分模型
            high_risk_kws = ["骗子", "黑号", "禁言", "点位", "码商", "通道", "实名", "收号", "博彩", "注单", "提现"]
            hits = [k for k in keywords if k in high_risk_kws]
            
            if hits:
                l2_risk_score += min(len(hits) * 30, 90) # 每个高危词 30 分，封顶 90
                l2_evidence.append(f"NLP识别高危词汇: {hits}")

            # 语气分析
            if "?" in text or "？" in text or "怎么" in text:
                l2_evidence.append("带有疑问/质询语气 (Victim特征)")
            if any(w in text for w in ["踢", "滚", "速度", "赶紧", "查收"]):
                l2_evidence.append("带有指令/管理语气 (Aggressor特征)")

            # 活跃度与句长加权 (短句/低活跃度通常风险低，长句/特定语气加权)
            if len(text) > 50:
                l2_risk_score += 10
                l2_evidence.append("文本较长，含较多信息熵")

        return {
            "is_system_msg": is_sys,
            "nlp_keywords": keywords,
            "token_count": len(list(jieba.cut(text))) if text else 0,
            "l2_risk_score": l2_risk_score,
            "l2_evidence": l2_evidence
        }


class InteractionNetwork:
    """对应PPT ：互动图与核心人物识别"""

    def __init__(self):
        self.graph = nx.DiGraph()  # 有向图
        self.user_activity = Counter()
        # 修正正则：匹配 @ 后跟数字、字母、下划线的用户名
        self.mention_pattern = re.compile(r"@([\w\_]+)")

    def build_from_data(self, all_messages):
        """
        改进后的构建逻辑：
        1. 显式提及 (@mention) -> 强权重 (权重为 5)
        2. 上下文邻近 (A说完B说) -> 弱权重 (权重为 1)
        """
        last_user = None

        for msg in all_messages:
            user = msg.get("username", "unknown")
            text = msg.get("text", "")

            # 排除系统消息和未知用户
            if user == "unknown" or msg.get("is_system_msg", False):
                continue

            # 1. 记录活跃度 (节点权重的基础)
            self.user_activity[user] += 1

            # --- 逻辑 1：显式提及 (@) ---
            mentions = self.mention_pattern.findall(text)
            for target in mentions:
                if target != user:
                    # @ 某人是强互动
                    self._add_edge_weight(user, target, weight=5)

            # --- 逻辑 2：上下文邻近 (解决 PageRank 为 0 的关键) ---
            # 如果上一个人不是自己，认为当前用户在回应上一个人
            if last_user and last_user != user:
                # 这种邻近关系虽然不如 @ 准确，但能勾勒出对话流
                self._add_edge_weight(user, last_user, weight=1)

            last_user = user

    def _add_edge_weight(self, source, target, weight=1):
        """辅助函数：安全地添加或增加边的权重"""
        if self.graph.has_edge(source, target):
            self.graph[source][target]['weight'] += weight
        else:
            self.graph.add_edge(source, target, weight=weight)

    def analyze_centrality(self):
        """计算核心指标：PageRank 和 度中心性"""
        # 如果整个图完全没有边（甚至邻近关系都没有）
        if self.graph.number_of_edges() == 0:
            stats = {}
            for user, count in self.user_activity.items():
                stats[user] = {
                    "msg_count": count,
                    "pagerank": 0.0,
                    "mentioned_count": 0,
                    "is_kol": False
                }
            return stats

        # 使用 networkx 计算 PageRank
        # alpha=0.85 是标准阻尼系数
        pagerank = nx.pagerank(self.graph, weight='weight', alpha=0.85)

        # 被他人提及或回应的加权总数
        in_degree = dict(self.graph.in_degree(weight='weight'))

        network_stats = {}
        all_users = set(self.user_activity.keys()) | set(self.graph.nodes)

        for user in all_users:
            network_stats[user] = {
                "msg_count": self.user_activity.get(user, 0),
                "pagerank": pagerank.get(user, 0.0),
                "mentioned_count": in_degree.get(user, 0),
                "is_kol": False
            }

        return network_stats

    def identify_kols(self, stats, top_n=5):
        """识别核心人物 (KOL)"""
        if not stats:
            return [], {}

        # 排序策略：PageRank 代表影响力，msg_count 代表活跃度
        sorted_users = sorted(
            stats.items(),
            key=lambda x: (x[1]['pagerank'], x[1]['msg_count']),
            reverse=True
        )

        kol_list = []
        for i, (user, data) in enumerate(sorted_users[:top_n]):
            stats[user]['is_kol'] = True
            kol_list.append(user)

        return kol_list, stats