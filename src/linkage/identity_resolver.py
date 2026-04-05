import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class _DSU:
    parent: Dict[str, str]

    def __init__(self):
        self.parent = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        pa = self.find(a)
        pb = self.find(b)
        if pa != pb:
            self.parent[pb] = pa


class IdentityResolver:
    """
    Cross-group, cross-account identity linkage with conservative hard-merging:
    - hard merge: same username across groups
    - hard merge: strong PII (phone/id) only when self-claimed by >=2 nodes
    - all other links are exported as soft clue chains (no forced identity assertion)
    """

    EVENT_KEYWORDS = {
        "ORDER_PUBLISH": ["下发", "新卡", "换卡", "车队", "保证金", "开后台", "收U", "四件套", "白户", "尾号", "开头", "结尾", "前六", "尾四", "半套", "全量"],
        "ORDER_RECEIVE": ["收到", "进场", "准备", "查收"],
        "VICTIM_COMPLAINT": ["被骗", "没到账", "拉黑", "报警", "报案", "还要钱", "退钱", "骗子", "投诉", "维权", "不给全", "假地址", "不发到门牌"],
        "PII_LEAK": ["身份证", "手机号", "QQ", "住址", "收款地址", "我的电话", "照片都发了", "我这就拍", "尾号", "开头", "结尾", "前六", "尾四"],
        "OFFLINE_MEET": ["接头地点", "见面", "地库", "地址", "延安中路"],
        "THREAT": ["你自己也是违法", "让你消失", "滚", "踢了"],
    }

    @staticmethod
    def _node_id(msg: Dict) -> str:
        return f"{msg.get('source_group','unknown')}::{msg.get('username','unknown')}"

    @staticmethod
    def _normalize_username(u: str) -> str:
        return str(u or "unknown").strip().lower()

    @staticmethod
    def _is_noise_username(u: str) -> bool:
        u = str(u or "").strip().lower()
        return u.startswith("system") or "bot" in u or u.startswith("unknown")

    @staticmethod
    def _is_first_person(text: str) -> bool:
        return any(x in (text or "") for x in ["我", "本人", "咱", "俺"])

    def _is_self_claim(self, text: str, ptype: str, token: str) -> bool:
        t = str(text or "")
        first = self._is_first_person(t)

        if ptype == "phone":
            if re.search(r"我.{0,8}1[3-9]\d{9}", t):
                return True
            return first and any(x in t for x in ["我的电话", "我电话", "手机号", "联系我", "我是"])

        if ptype == "id":
            if re.search(r"我.{0,8}\d{17}[0-9Xx]", t):
                return True
            return first and any(x in t for x in ["身份证", "身份证号", "我证件"])

        if ptype == "bank":
            if re.search(r"我.{0,12}\d{15,19}", t):
                return True
            return first and any(x in t for x in ["我卡", "我的卡", "卡号", "银行卡"])

        if ptype == "name":
            return any(x in t for x in ["我叫", "我是", "姓名"]) and token in t

        if ptype == "address":
            return first and any(x in t for x in ["我住", "我在", "我家", "住址", "地址"])

        if ptype == "qq":
            return first and any(x in t for x in ["我的QQ", "我QQ", "QQ号"]) and token in t

        return False

    def _iter_pii_tokens(self, msg: Dict) -> Iterable[Tuple[str, str]]:
        pii_details = msg.get("pii_details", {})
        if not isinstance(pii_details, dict):
            return []

        out: List[Tuple[str, str]] = []
        for k, vals in pii_details.items():
            key = str(k).lower()
            ptype = None
            if "mobile" in key or "phone" in key:
                ptype = "phone_fragment" if "fragment" in key or "masked" in key else "phone"
            elif "qq" in key:
                ptype = "qq"
            elif "id" in key:
                ptype = "id_fragment" if "fragment" in key or "masked" in key else "id"
            elif "bank" in key:
                ptype = "bank"
            elif "payment" in key or "wallet" in key or "usdt" in key:
                ptype = "wallet"
            elif "alias" in key:
                ptype = "alias"
            elif "name" in key:
                ptype = "name"
            elif "address" in key or "location" in key:
                ptype = "address_hint" if "hint" in key else "address"

            if not ptype:
                continue

            if isinstance(vals, list):
                for v in vals:
                    vv = str(v or "").strip()
                    if vv:
                        out.append((ptype, vv))

        return out

    @staticmethod
    def _node_is_suspect_like(node_signal: Dict) -> bool:
        return node_signal.get("scammer", 0) >= node_signal.get("victim", 0)

    def resolve(self, messages: List[Dict]) -> Dict:
        dsu = _DSU()

        node_meta: Dict[str, Dict] = {}
        node_signal: Dict[str, Dict] = defaultdict(lambda: {"scammer": 0, "victim": 0, "other": 0})

        username_nodes: Dict[str, List[str]] = defaultdict(list)
        token_mentions: Dict[str, List[Dict]] = defaultdict(list)

        for msg in messages:
            node = self._node_id(msg)
            user = msg.get("username", "unknown")
            group = msg.get("source_group", "unknown")
            text = str(msg.get("text", "") or "")

            if node not in node_meta:
                node_meta[node] = {
                    "username": user,
                    "source_group": group,
                    "msg_count": 0,
                }
            node_meta[node]["msg_count"] += 1

            role = str((msg.get("llm_decision", {}) or {}).get("role", "other"))
            node_signal[node][role if role in {"scammer", "victim", "other"} else "other"] += 1

            if not self._is_noise_username(user):
                username_nodes[self._normalize_username(user)].append(node)

            for ptype, pval in self._iter_pii_tokens(msg):
                token = f"{ptype}:{pval}"
                token_mentions[token].append(
                    {
                        "node": node,
                        "username": user,
                        "source_group": group,
                        "self_claim": self._is_self_claim(text, ptype, pval),
                    }
                )

        # hard merge: same username across groups
        for nodes in username_nodes.values():
            uniq_nodes = list(dict.fromkeys(nodes))
            if len(uniq_nodes) < 2:
                continue
            anchor = uniq_nodes[0]
            for n in uniq_nodes[1:]:
                dsu.union(anchor, n)

        # hard merge: strong pii (phone/id) must be self-claimed by >=2 nodes.
        for token, mentions in token_mentions.items():
            if len(mentions) < 2:
                continue
            ptype, _ = token.split(":", 1)
            uniq_nodes = list(dict.fromkeys(m["node"] for m in mentions))
            if len(uniq_nodes) < 2:
                continue

            if ptype not in {"phone", "id"}:
                continue

            self_nodes = list(dict.fromkeys(m["node"] for m in mentions if m.get("self_claim")))
            if len(self_nodes) >= 2:
                anchor = self_nodes[0]
                for n in self_nodes[1:]:
                    dsu.union(anchor, n)
                continue

            # fallback: same normalized username sharing same strong token across groups
            by_user = defaultdict(list)
            for m in mentions:
                by_user[self._normalize_username(m["username"])].append(m["node"])
            for user, nodes in by_user.items():
                uniq = list(dict.fromkeys(nodes))
                if user != "unknown" and len(uniq) >= 2:
                    anchor = uniq[0]
                    for n in uniq[1:]:
                        dsu.union(anchor, n)

        clusters_nodes: Dict[str, Set[str]] = defaultdict(set)
        for node in node_meta.keys():
            clusters_nodes[dsu.find(node)].add(node)

        clusters: List[Dict] = []
        node_to_cluster: Dict[str, str] = {}

        cid = 1
        for _, nodes in clusters_nodes.items():
            members = []
            aliases = set()
            groups = set()
            max_msgs = -1
            canonical_name = "unknown"

            shared_pii: Dict[str, Set[str]] = defaultdict(set)

            for node in nodes:
                meta = node_meta[node]
                members.append({"source_group": meta["source_group"], "username": meta["username"]})
                aliases.add(meta["username"])
                groups.add(meta["source_group"])

                if meta["msg_count"] > max_msgs:
                    max_msgs = meta["msg_count"]
                    canonical_name = meta["username"]

            # shared pii in cluster: only add token if linked by >=2 distinct nodes in this cluster
            for token, mentions in token_mentions.items():
                touched = {m["node"] for m in mentions if m["node"] in nodes}
                if len(touched) < 2:
                    continue
                ptype, pval = token.split(":", 1)
                shared_pii[ptype].add(pval)

            cluster_id = f"C{cid:03d}"
            cid += 1
            for node in nodes:
                node_to_cluster[node] = cluster_id

            confidence = "low"
            if len(groups) >= 2 and (
                len(shared_pii.get("phone", [])) > 0
                or len(shared_pii.get("id", [])) > 0
                or len(aliases) > 1
            ):
                confidence = "high"
            elif len(groups) >= 2 or len(shared_pii) > 0:
                confidence = "medium"

            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "canonical_name": canonical_name,
                    "aliases": sorted(aliases),
                    "groups": sorted(groups),
                    "members": sorted(members, key=lambda x: (x["source_group"], x["username"])),
                    "shared_pii": {k: sorted(v) for k, v in shared_pii.items()},
                    "confidence": confidence,
                }
            )

        clusters.sort(key=lambda x: (len(x["members"]), len(x["shared_pii"])), reverse=True)

        return {
            "clusters": clusters,
            "node_to_cluster": node_to_cluster,
        }

    def build_trace_events(self, messages: List[Dict], node_to_cluster: Dict[str, str]) -> List[Dict]:
        events: List[Dict] = []
        for msg in messages:
            node = self._node_id(msg)
            cluster_id = node_to_cluster.get(node, "")
            text = str(msg.get("text", "") or "")
            if not text:
                continue

            for etype, kws in self.EVENT_KEYWORDS.items():
                if any(k in text for k in kws):
                    events.append(
                        {
                            "cluster_id": cluster_id,
                            "source_group": msg.get("source_group", "unknown"),
                            "username": msg.get("username", "unknown"),
                            "msg_index": int(msg.get("msg_index", 0)),
                            "event_type": etype,
                            "detail": text,
                        }
                    )
                    break

        return events

    def attach_cluster_labels(self, messages: List[Dict], node_to_cluster: Dict[str, str]):
        for msg in messages:
            msg["identity_cluster"] = node_to_cluster.get(self._node_id(msg), "")

    def build_clue_chains(self, messages: List[Dict], node_to_cluster: Dict[str, str], group_summary: Dict) -> List[Dict]:
        suspects = set(group_summary.get("suspect_list", []) or [])
        victims = set(group_summary.get("victim_list", []) or [])
        irrelevant = set(group_summary.get("irrelevant_list", []) or [])

        token_mentions: Dict[str, List[Dict]] = defaultdict(list)
        for msg in messages:
            text = str(msg.get("text", "") or "")
            if not text:
                continue
            for ptype, pval in self._iter_pii_tokens(msg):
                token = f"{ptype}:{pval}"
                username = msg.get("username", "unknown")

                if username in suspects:
                    role_hint = "suspect"
                elif username in victims:
                    role_hint = "victim"
                elif username in irrelevant:
                    role_hint = "irrelevant"
                else:
                    role_hint = str((msg.get("llm_decision", {}) or {}).get("role", "other"))

                token_mentions[token].append(
                    {
                        "source_group": msg.get("source_group", "unknown"),
                        "username": username,
                        "msg_index": int(msg.get("msg_index", 0)),
                        "text_excerpt": text[:100],
                        "self_claim": self._is_self_claim(text, ptype, pval),
                        "role_hint": role_hint,
                        "cluster_id": node_to_cluster.get(self._node_id(msg), ""),
                    }
                )

        conf_rank = {"high": 3, "medium": 2, "low": 1}
        chains: List[Dict] = []

        for token, mentions in token_mentions.items():
            uniq_users = sorted({m["username"] for m in mentions})
            uniq_groups = sorted({m["source_group"] for m in mentions})
            if len(uniq_users) < 2 and len(uniq_groups) < 2:
                continue

            ptype, pval = token.split(":", 1)
            mentions_sorted = sorted(mentions, key=lambda x: (x["source_group"], x["msg_index"], x["username"]))

            suspect_users = sorted({m["username"] for m in mentions_sorted if m["role_hint"] == "suspect"})
            victim_users = sorted({m["username"] for m in mentions_sorted if m["role_hint"] == "victim"})
            irrelevant_users = sorted({m["username"] for m in mentions_sorted if m["role_hint"] == "irrelevant"})
            self_claim_users = sorted({m["username"] for m in mentions_sorted if m.get("self_claim")})

            confidence = "low"
            chain_type = "cross_group_correlation"
            inference = "同一线索在多个账号/群出现，建议继续人工核验。"

            if ptype in {"phone", "id"} and len(self_claim_users) >= 2:
                confidence = "high"
                chain_type = "alt_account_candidate"
                inference = "多个账号分别自述同一强隐私标识，存在大小号/换号可能（软结论）。"
            elif suspect_users and victim_users:
                confidence = "medium"
                chain_type = "suspect_victim_link"
                inference = "嫌疑人与受害者围绕同一隐私标识交叉出现，形成作案-受害线索链。"
            elif len(suspect_users) >= 2:
                confidence = "medium"
                chain_type = "suspect_internal_asset"
                inference = "多名嫌疑人复用同一线索，疑似内部协作或分工流转。"
            elif len(self_claim_users) >= 2:
                confidence = "medium"
                chain_type = "identity_soft_link"
                inference = "多个账号的自述信息重合，建议作为同人候选链跟踪。"

            candidate_pairs = []
            if len(self_claim_users) >= 2:
                for a, b in list(combinations(self_claim_users, 2))[:6]:
                    candidate_pairs.append({"account_a": a, "account_b": b, "basis": f"{ptype}:{pval}"})

            chains.append(
                {
                    "clue_type": ptype,
                    "clue_value": pval,
                    "chain_type": chain_type,
                    "confidence": confidence,
                    "inference": inference,
                    "groups": uniq_groups,
                    "mentions_count": len(mentions_sorted),
                    "points_to": {
                        "victims": victim_users,
                        "suspects": suspect_users,
                        "irrelevant": irrelevant_users,
                    },
                    "soft_identity_candidates": candidate_pairs,
                    "mentions": mentions_sorted[:12],
                }
            )

        chains.sort(
            key=lambda x: (
                conf_rank.get(x.get("confidence", "low"), 1),
                x.get("mentions_count", 0),
                len(x.get("groups", [])),
            ),
            reverse=True,
        )

        for i, c in enumerate(chains, start=1):
            c["chain_id"] = f"L{i:03d}"

        return chains

    @staticmethod
    def summarize(clusters: List[Dict], events: List[Dict]) -> Dict:
        cross_group_clusters = [c for c in clusters if len(c.get("groups", [])) >= 2]
        event_counts = defaultdict(int)
        for e in events:
            event_counts[e.get("event_type", "OTHER")] += 1

        return {
            "cluster_count": len(clusters),
            "cross_group_cluster_count": len(cross_group_clusters),
            "cross_group_clusters": cross_group_clusters,
            "event_counts": dict(event_counts),
        }
