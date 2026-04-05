import re
from collections import Counter

import networkx as nx

try:
    import jieba
    import jieba.analyse
except Exception: 
    jieba = None

from src.config import STOPWORDS, SYSTEM_MSG_KEYWORDS


class TextMiner:
    """NLP feature extraction for single messages."""

    def __init__(self):
        self.stopwords = STOPWORDS
        self.high_risk_kws = {
            "骗子",
            "黑卡",
            "卡号",
            "户名",
            "保证金",
            "车队",
            "通道",
            "实名",
            "跑分",
            "收U",
            "换卡",
            "下发",
            "提现",
            "充值",
            "QQ",
            "收款地址",
            "三要素",
            "户籍",
            "家谱",
            "近照",
            "尾号",
            "开头",
            "结尾",
            "前六",
            "尾四",
            "半套",
            "全量",
            "别发全",
            "模糊定位",
            "统一口径",
            "重复卖",
            "拆开卖",
            "假地址",
            "售后",
        }
        self.implicit_trade_kws = {
            "尾号",
            "开头",
            "结尾",
            "前六",
            "尾四",
            "半套",
            "全量",
            "别发全",
            "别全名",
            "模糊定位",
            "统一口径",
            "重复卖",
            "拆开卖",
            "主推",
            "后补",
            "只发到县",
            "不发到门牌",
            "家里人号",
            "备注",
        }
        self.complaint_kws = {
            "投诉",
            "维权",
            "假地址",
            "不给全",
            "只给",
            "怀疑",
            "重复卖",
            "拉黑",
            "不发到门牌",
            "售后",
        }

    def is_system_message(self, text: str) -> bool:
        if not text:
            return True
        t = str(text)
        if any(kw in t for kw in SYSTEM_MSG_KEYWORDS):
            return True
        if re.search(r"已(加入|移出)群组", t):
            return True
        return False

    def _fallback_keywords(self, text: str, top_k: int = 5):
        candidates = [x for x in re.split(r"[\s,，。！？!?:：;；()\[\]{}<>\-_/\\|]+", text) if x]
        candidates = [c for c in candidates if len(c) > 1 and c not in self.stopwords]
        return candidates[:top_k]

    def extract_keywords(self, text: str, top_k: int = 5):
        if not text or self.is_system_message(text):
            return []

        if jieba is None:
            return self._fallback_keywords(text, top_k=top_k)

        tags = jieba.analyse.extract_tags(
            text,
            topK=top_k * 2,
            withWeight=False,
            allowPOS=("n", "v", "vn", "ns", "nr"),
        )
        filtered = [t for t in tags if t not in self.stopwords and len(t) > 1]
        return filtered[:top_k]

    def process(self, message):
        text = str(message.get("text", "") or "")
        is_sys = self.is_system_message(text)
        keywords = self.extract_keywords(text) if not is_sys else []

        l2_risk_score = 0
        l2_evidence = []

        if is_sys:
            l2_evidence.append("系统/低价值消息")
        else:
            hits = [k for k in keywords if k in self.high_risk_kws]
            if hits:
                l2_risk_score += min(len(hits) * 25, 80)
                l2_evidence.append(f"命中高风险词: {hits}")

            direct_hits = [k for k in self.high_risk_kws if k in text]
            if direct_hits and not hits:
                l2_risk_score += min(12 + len(direct_hits) * 5, 70)
                l2_evidence.append(f"命中高风险短语: {direct_hits[:8]}")

            implicit_trade_hits = [k for k in self.implicit_trade_kws if k in text]
            if implicit_trade_hits:
                l2_risk_score += min(10 + len(implicit_trade_hits) * 4, 35)
                l2_evidence.append(f"命中隐式交易线索: {implicit_trade_hits[:8]}")

            complaint_hits = [k for k in self.complaint_kws if k in text]
            if complaint_hits:
                l2_risk_score += min(8 + len(complaint_hits) * 4, 28)
                l2_evidence.append(f"命中投诉/售后线索: {complaint_hits[:8]}")

            if any(x in text for x in ["?", "？", "怎么", "为何", "为什么"]):
                l2_evidence.append("疑问/求助语气")
            if any(x in text for x in ["卡号", "户名", "下发", "速度", "查收", "进场", "收款地址", "三要素", "查档", "尾号", "开头", "全量", "半套"]):
                l2_evidence.append("操作/指令语气")

            if len(text) > 50:
                l2_risk_score += 10
                l2_evidence.append("文本较长，信息密度较高")

        token_count = 0
        if text:
            if jieba is None:
                token_count = len(text)
            else:
                token_count = len(list(jieba.cut(text)))

        return {
            "is_system_msg": is_sys,
            "nlp_keywords": keywords,
            "token_count": token_count,
            "l2_risk_score": min(l2_risk_score, 100),
            "l2_evidence": l2_evidence,
        }


class InteractionNetwork:
    """Directed interaction graph with mention + adjacency edges."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.user_activity = Counter()
        self.mention_pattern = re.compile(r"@([\w\_]+)")

    def build_from_data(self, all_messages):
        last_user = None
        for msg in all_messages:
            user = msg.get("username", "unknown")
            text = str(msg.get("text", "") or "")

            if user == "unknown" or msg.get("is_system_msg", False):
                continue

            self.user_activity[user] += 1

            mentions = self.mention_pattern.findall(text)
            for target in mentions:
                if target != user:
                    self._add_edge_weight(user, target, weight=5)

            if last_user and last_user != user:
                self._add_edge_weight(user, last_user, weight=1)

            last_user = user

    def _add_edge_weight(self, source, target, weight=1):
        if self.graph.has_edge(source, target):
            self.graph[source][target]["weight"] += weight
        else:
            self.graph.add_edge(source, target, weight=weight)

    def analyze_centrality(self):
        if self.graph.number_of_edges() == 0:
            return {
                u: {"msg_count": c, "pagerank": 0.0, "mentioned_count": 0, "is_kol": False}
                for u, c in self.user_activity.items()
            }

        pagerank = nx.pagerank(self.graph, weight="weight", alpha=0.85)
        in_degree = dict(self.graph.in_degree(weight="weight"))

        stats = {}
        all_users = set(self.user_activity.keys()) | set(self.graph.nodes)
        for user in all_users:
            stats[user] = {
                "msg_count": self.user_activity.get(user, 0),
                "pagerank": pagerank.get(user, 0.0),
                "mentioned_count": in_degree.get(user, 0),
                "is_kol": False,
            }
        return stats

    def identify_kols(self, stats, top_n=5):
        if not stats:
            return [], {}

        sorted_users = sorted(
            stats.items(),
            key=lambda x: (x[1]["pagerank"], x[1]["msg_count"]),
            reverse=True,
        )

        kol_list = []
        for user, _ in sorted_users[:top_n]:
            stats[user]["is_kol"] = True
            kol_list.append(user)

        return kol_list, stats
