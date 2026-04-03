from collections import Counter


class UserProfiler:
    """把消息级结果聚合成用户级画像摘要。"""

    def __init__(self):
        self.profiles = {}

    def aggregate(self, msg_list):
        """累计发言次数、关键词、PageRank 和角色分布。"""
        for msg in msg_list:
            u = msg.get("username")
            if not u or u == "system" or msg.get("is_system_msg"):
                continue

            if u not in self.profiles:
                self.profiles[u] = {
                    "count": 0,
                    "risks": [],
                    "keywords": set(),
                    "pagerank": 0.0,
                    "roles_detected": Counter(),
                }

            self.profiles[u]["count"] += 1
            if "nlp_keywords" in msg:
                self.profiles[u]["keywords"].update(msg.get("nlp_keywords", []))

            # 从全局交互网络中回填该用户的影响力得分。
            if "user_profile" in msg:
                self.profiles[u]["pagerank"] = msg["user_profile"].get("pagerank", 0.0)

            if "llm_decision" in msg and isinstance(msg["llm_decision"], dict):
                decision = msg["llm_decision"]
                self.profiles[u]["roles_detected"][decision.get("role", "other")] += 1
                self.profiles[u]["risks"].append(decision.get("risk", "low"))

    def finalize(self):
        """输出供报告使用的用户画像列表。"""
        summary = []
        sorted_users = sorted(self.profiles.items(), key=lambda x: x[1]["pagerank"], reverse=True)
        for u, data in sorted_users:
            main_role = data["roles_detected"].most_common(1)[0][0] if data["roles_detected"] else "unknown"
            summary.append(
                {
                    "username": u,
                    "influence": round(data["pagerank"] * 1000, 2),
                    "msg_count": data["count"],
                    "predicted_role": main_role,
                }
            )
        return summary
