from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple


class GroupProfiler:
    """
    Aggregate message-level decisions into group-level role lists and evidence.

    Roles:
    - suspect: core malicious operators
    - victim: victim / near-victim users
    - irrelevant: noisy sign-in / casual chatter users
    """

    def __init__(self):
        self.stats = {
            "total_msgs": 0,
            "all_evidence": [],
        }
        self._seen_evidence = set()

        self.user_stats: Dict[str, Dict] = defaultdict(
            lambda: {
                "msg_count": 0,
                "votes": {"scammer": 0, "victim": 0, "other": 0},
                "victim_signal": 0,
                "loss_signal": 0,
                "scammer_signal": 0,
                "manager_signal": 0,
                "noise_signal": 0,
                "casual_signal": 0,
                "pii_count": 0,
                "bank_card_posts": 0,
                "id_posts": 0,
                "self_pii_leak": 0,
            }
        )

        self.victim_keywords = [
            "被骗",
            "骗子",
            "还钱",
            "没到账",
            "没到帐",
            "求助",
            "救命",
            "报警",
            "报案",
            "拉黑",
            "退钱",
            "投诉",
            "维权",
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
        self.loss_keywords = [
            "没到账",
            "没到帐",
            "提现",
            "还要钱",
            "拉黑",
            "不理我",
            "被骗了",
            "退钱",
            "投诉",
            "维权",
            "骗子",
            "报案",
            "报警",
        ]
        self.scammer_keywords = [
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
            "实名",
            "高仿",
            "包邮",
            "踢",
            "滚",
            "反诈",
        ]
        self.manager_keywords = [
            "车队",
            "待命",
            "保证金",
            "开后台",
            "新卡",
            "换卡",
            "下发",
            "卡号",
            "户名",
            "内部通道",
            "进二群",
            "收U",
            "四件套",
            "白户",
        ]
        self.noise_keywords = {"签到", "滴滴", "打卡", "路过", "冒泡"}
        self.casual_keywords = ["天气", "专业", "羡慕", "看看", "围观", "聊天"]

    @staticmethod
    def _normalize_role(role: str) -> str:
        role = str(role or "other").strip().lower()
        if role in {"scammer", "victim", "other"}:
            return role
        return "other"

    @staticmethod
    def _is_system_user(username: str) -> bool:
        return str(username or "").strip().lower() == "system"

    @staticmethod
    def _is_noise_user(username: str) -> bool:
        u = str(username or "").strip().lower()
        return (
            u in {"unknown", "unknown_user", "unknownuser"}
            or u.startswith("unknown")
            or "bot" in u
            or "sign_in" in u
        )

    @staticmethod
    def _has_long_number(text: str) -> bool:
        return bool(re.search(r"(?<!\d)\d{15,19}(?!\d)", text or ""))

    @staticmethod
    def _is_first_person_text(text: str) -> bool:
        return any(p in (text or "") for p in ["我", "本人", "咱", "俺"])

    def _guess_pii_label(self, key: str, text: str, content: str) -> str:
        k = str(key or "").lower()
        t = str(text or "")

        if "phone" in k or "mobile" in k:
            return "手机号"
        if "id" in k:
            return "身份证"
        if "mail" in k:
            return "邮箱"

        # Bank regex can over-hit ID fragments; use context to fix.
        if "bank" in k:
            if any(x in t for x in ["身份证", "住址", "手持照", "实名注册"]):
                return "身份证"
            if len(content) < 16 and any(x in t for x in ["身份证", "住址"]):
                return "身份证"
            return "银行卡"

        return "其他"

    def _update_behavior_signals(self, username: str, text: str):
        rec = self.user_stats[username]
        t = str(text or "").strip()
        if not t:
            return

        if t in self.noise_keywords:
            rec["noise_signal"] += 1
            return

        first_person = self._is_first_person_text(t)
        victim_hit = any(k in t for k in self.victim_keywords)
        loss_hit = any(k in t for k in self.loss_keywords) and first_person
        scammer_hit = any(k in t for k in self.scammer_keywords)
        manager_hit = any(k in t for k in self.manager_keywords)

        if self._has_long_number(t) and any(x in t for x in ["卡号", "户名", "银行", "下发", "新卡", "换卡"]):
            scammer_hit = True
            manager_hit = True

        if first_person and any(x in t for x in ["我的电话", "身份证号", "住址", "照片都发了", "我这就拍"]):
            rec["victim_signal"] += 1
            rec["self_pii_leak"] += 1

        if victim_hit and (first_person or "你们" in t):
            rec["victim_signal"] += 1
        if loss_hit:
            rec["loss_signal"] += 1
        if scammer_hit:
            rec["scammer_signal"] += 1
        if manager_hit:
            rec["manager_signal"] += 1

        if not victim_hit and not scammer_hit and any(k in t for k in self.casual_keywords):
            rec["casual_signal"] += 1

    def update(self, message: Dict):
        self.stats["total_msgs"] += 1

        username = message.get("username")
        text = str(message.get("text", "") or "")

        if not username or self._is_system_user(username):
            return

        rec = self.user_stats[username]
        rec["msg_count"] += 1

        llm_res = message.get("llm_decision", {}) if isinstance(message.get("llm_decision", {}), dict) else {}
        role = self._normalize_role(llm_res.get("role", "other"))
        rec["votes"][role] = rec["votes"].get(role, 0) + 1

        self._update_behavior_signals(username, text)

        pii_details = message.get("pii_details", {})
        if not isinstance(pii_details, dict):
            pii_details = {}

        first_person = self._is_first_person_text(text)
        self_leak_context = first_person and any(x in text for x in ["我的", "身份证", "住址", "照片", "我这就拍", "我是学生"])
        op_context = any(x in text for x in ["卡号", "户名", "银行", "下发", "新卡", "换卡", "收款"])

        for key, values in pii_details.items():
            if not isinstance(values, list):
                continue

            for val in values:
                content = str(val or "").strip()
                if not content:
                    continue

                label = self._guess_pii_label(key, text, content)

                rec["pii_count"] += 1
                if label == "银行卡" and op_context and not self_leak_context:
                    rec["bank_card_posts"] += 1
                if label == "身份证":
                    rec["id_posts"] += 1
                if self_leak_context:
                    rec["self_pii_leak"] += 1
                    rec["victim_signal"] += 1

                dedupe_key = (username, label, content)
                if dedupe_key in self._seen_evidence:
                    continue
                self._seen_evidence.add(dedupe_key)

                self.stats["all_evidence"].append(
                    {
                        "type": label,
                        "content": content,
                        "owner": username,
                        "context": text[:80],
                    }
                )

    def _user_scores(self, username: str, network_stats: Dict) -> Tuple[float, float, float]:
        rec = self.user_stats.get(username, {})
        votes = rec.get("votes", {})

        v_votes = votes.get("victim", 0)
        s_votes = votes.get("scammer", 0)
        o_votes = votes.get("other", 0)

        victim_signal = rec.get("victim_signal", 0)
        loss_signal = rec.get("loss_signal", 0)
        scammer_signal = rec.get("scammer_signal", 0)
        manager_signal = rec.get("manager_signal", 0)
        noise_signal = rec.get("noise_signal", 0)
        casual_signal = rec.get("casual_signal", 0)

        bank_posts = rec.get("bank_card_posts", 0)
        id_posts = rec.get("id_posts", 0)
        self_pii_leak = rec.get("self_pii_leak", 0)

        pr = float(network_stats.get(username, {}).get("pagerank", 0.0) or 0.0)

        victim_score = (
            1.5 * loss_signal
            + 1.5 * victim_signal
            + 0.8 * v_votes
            + 1.2 * self_pii_leak
            + 0.5 * id_posts
            - 0.8 * s_votes
            - 1.0 * manager_signal
            - 1.0 * bank_posts
            - 0.4 * scammer_signal
            + (0.2 if pr < 0.06 else 0.0)
        )

        scammer_score = (
            1.0 * s_votes
            + 1.6 * manager_signal
            + 1.2 * bank_posts
            + 0.9 * scammer_signal
            - 1.2 * loss_signal
            - 1.0 * victim_signal
            - 0.8 * self_pii_leak
            - 0.5 * noise_signal
            + (0.2 if pr >= 0.08 else 0.0)
        )

        irrelevant_score = (
            1.4 * noise_signal
            + 1.1 * casual_signal
            + 0.3 * o_votes
            - 1.0 * victim_signal
            - 1.2 * loss_signal
            - 1.0 * manager_signal
            - 0.8 * bank_posts
            - 0.6 * self_pii_leak
        )

        return victim_score, scammer_score, irrelevant_score

    def _classify_users(self, network_stats: Dict):
        victim_scores = {}
        scammer_scores = {}

        victims = set()
        suspects = set()
        irrelevant = set()

        for username in self.user_stats.keys():
            victim_score, scammer_score, irrelevant_score = self._user_scores(username, network_stats)
            victim_scores[username] = victim_score
            scammer_scores[username] = scammer_score

            rec = self.user_stats[username]
            v_votes = rec["votes"].get("victim", 0)

            manager_signal = rec.get("manager_signal", 0)
            bank_posts = rec.get("bank_card_posts", 0)
            victim_signal = rec.get("victim_signal", 0)
            loss_signal = rec.get("loss_signal", 0)
            self_pii_leak = rec.get("self_pii_leak", 0)
            noise_signal = rec.get("noise_signal", 0)
            casual_signal = rec.get("casual_signal", 0)
            msg_count = max(rec.get("msg_count", 0), 1)

            is_water = (
                (noise_signal + casual_signal) >= max(2, int(msg_count * 0.6))
                and victim_signal == 0
                and loss_signal == 0
                and manager_signal == 0
                and bank_posts == 0
                and self_pii_leak == 0
                and rec["votes"].get("scammer", 0) == 0
            )
            if is_water or self._is_noise_user(username):
                irrelevant.add(username)
                continue

            has_hard_scammer_behavior = manager_signal >= 2 or bank_posts > 0 or rec.get("scammer_signal", 0) >= 3
            if has_hard_scammer_behavior and scammer_score >= max(2.6, victim_score + 1.2):
                suspects.add(username)
                continue

            has_victim_cue = (
                loss_signal > 0
                or self_pii_leak > 0
                or victim_signal >= 2
                or (v_votes >= 4 and msg_count >= 5)
            )
            if has_victim_cue and victim_score >= max(1.2, scammer_score + 0.2):
                victims.add(username)
                continue

            if irrelevant_score >= 1.6 and victim_signal == 0 and loss_signal == 0 and manager_signal == 0:
                irrelevant.add(username)

        cleaned_victims = set()
        for username in victims:
            rec = self.user_stats[username]
            if username in suspects or username in irrelevant:
                continue
            if rec.get("bank_card_posts", 0) > 0 and rec.get("self_pii_leak", 0) == 0 and rec.get("loss_signal", 0) == 0:
                continue
            cleaned_victims.add(username)

        cleaned_suspects = set()
        for username in suspects:
            rec = self.user_stats[username]
            if username in irrelevant:
                continue
            if rec.get("self_pii_leak", 0) > 0 and rec.get("manager_signal", 0) == 0:
                continue
            cleaned_suspects.add(username)

        return cleaned_victims, cleaned_suspects, irrelevant, victim_scores, scammer_scores

    @staticmethod
    def _format_evidence(evidence_list: List[Dict]) -> str:
        if not evidence_list:
            return "未发现相关数据"
        lines = []
        for e in evidence_list:
            lines.append(f"- [{e['type']}] {e['content']}（发布者: {e['owner']}，语境: {e['context']}）")
        return "\n".join(lines)

    def get_summary_context(self, network_stats, influence_threshold=0.1):
        victims, suspects, irrelevant, victim_scores, scammer_scores = self._classify_users(network_stats)

        suspect_assets = []
        victim_leaks = []

        for e in self.stats["all_evidence"]:
            owner = e["owner"]
            if owner in suspects:
                suspect_assets.append(e)
                continue
            if owner in victims:
                victim_leaks.append(e)
                continue
            if owner in irrelevant:
                continue

            rec = self.user_stats.get(owner, {})
            if e["type"] == "银行卡" and rec.get("manager_signal", 0) > 0:
                suspect_assets.append(e)
            elif rec.get("self_pii_leak", 0) > 0:
                victim_leaks.append(e)
            else:
                victim_leaks.append(e)

        ranked_victims = sorted(
            list(victims),
            key=lambda u: (
                victim_scores.get(u, 0.0),
                -float(network_stats.get(u, {}).get("pagerank", 0.0) or 0.0),
                self.user_stats.get(u, {}).get("msg_count", 0),
            ),
            reverse=True,
        )

        ranked_suspects = sorted(
            list(suspects),
            key=lambda u: (
                scammer_scores.get(u, 0.0),
                float(network_stats.get(u, {}).get("pagerank", 0.0) or 0.0),
                self.user_stats.get(u, {}).get("msg_count", 0),
            ),
            reverse=True,
        )

        ranked_irrelevant = sorted(
            list(irrelevant),
            key=lambda u: (
                self.user_stats.get(u, {}).get("noise_signal", 0) + self.user_stats.get(u, {}).get("casual_signal", 0),
                self.user_stats.get(u, {}).get("msg_count", 0),
            ),
            reverse=True,
        )

        final_victims = []
        for u in ranked_victims:
            pr = float(network_stats.get(u, {}).get("pagerank", 0.0) or 0.0)
            has_explicit_victim_cue = (
                self.user_stats[u].get("loss_signal", 0) > 0
                or self.user_stats[u].get("self_pii_leak", 0) > 0
                or self.user_stats[u].get("victim_signal", 0) >= 2
            )
            if pr < influence_threshold or has_explicit_victim_cue:
                final_victims.append(u)

        return {
            "msg_count": self.stats["total_msgs"],
            "suspect_assets": suspect_assets,
            "victim_leaks": victim_leaks,
            "suspect_assets_str": self._format_evidence(suspect_assets),
            "victim_leaks_str": self._format_evidence(victim_leaks),
            "victim_list": final_victims,
            "suspect_list": ranked_suspects,
            "irrelevant_list": ranked_irrelevant,
        }
