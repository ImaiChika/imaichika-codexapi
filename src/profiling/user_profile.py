# # src/profiling/user_profile.py
# class UserProfiler:
#     def __init__(self):
#         self.profiles = {}

#     def aggregate(self, msg_list):
#         """汇总单条消息到用户画像"""
#         for msg in msg_list:
#             u = msg.get("username")
#             if not u or msg.get("is_system_msg"): continue

#             if u not in self.profiles:
#                 self.profiles[u] = {
#                     "count": 0,
#                     "risks": [],
#                     "keywords": set(),
#                     "pagerank": msg.get("user_profile", {}).get("pagerank", 0)
#                 }

#             self.profiles[u]["count"] += 1
#             self.profiles[u]["keywords"].update(msg.get("nlp_keywords", []))
#             if "llm_decision" in msg:
#                 self.profiles[u]["risks"].append(msg["llm_decision"])

#     def finalize(self):
#         """生成最终的高危用户列表"""
#         summary = []
#         for u, data in self.profiles.items():
#             # 综合逻辑：影响力 > 0 且 LLM 判定过风险
#             if data['pagerank'] > 0:
#                 summary.append({
#                     "username": u,
#                     "influence": round(data['pagerank'] * 1000, 4),
#                     "tags": list(data['keywords'])[:5],
#                     "risk_summary": data['risks'][-1] if data['risks'] else "正常"
#                 })
#         return sorted(summary, key=lambda x: x['influence'], reverse=True)

# src/profiling/user_profile.py
from collections import Counter

class UserProfiler:
    def __init__(self):
        self.profiles = {}

    def aggregate(self, msg_list):
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
                    "roles_detected": Counter()
                }

            self.profiles[u]["count"] += 1
            if "nlp_keywords" in msg:
                self.profiles[u]["keywords"].update(msg.get("nlp_keywords", []))
            
            # 从消息中获取之前存入的 PageRank
            if "user_profile" in msg:
                self.profiles[u]["pagerank"] = msg["user_profile"].get("pagerank", 0.0)

            if "llm_decision" in msg and isinstance(msg["llm_decision"], dict):
                decision = msg["llm_decision"]
                self.profiles[u]["roles_detected"][decision.get("role", "other")] += 1
                self.profiles[u]["risks"].append(decision.get("risk", "low"))

    def finalize(self):
        summary = []
        sorted_users = sorted(self.profiles.items(), key=lambda x: x[1]['pagerank'], reverse=True)
        for u, data in sorted_users:
            main_role = data["roles_detected"].most_common(1)[0][0] if data["roles_detected"] else "unknown"
            summary.append({
                "username": u,
                "influence": round(data['pagerank'] * 1000, 2),
                "msg_count": data['count'],
                "predicted_role": main_role
            })
        return summary