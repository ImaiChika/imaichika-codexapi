from collections import defaultdict
from dataclasses import dataclass
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
    Cross-group, cross-account identity linkage based on:
    - same username across groups
    - shared strong PII (phone/id)
    - shared weak PII (bank/name/address) under strict constraints
    """

    EVENT_KEYWORDS = {
        "ORDER_PUBLISH": ["下发", "新卡", "换卡", "车队", "保证金", "开后台", "收U", "四件套", "白户"],
        "ORDER_RECEIVE": ["收到", "进场", "准备", "查收"],
        "VICTIM_COMPLAINT": ["被骗", "没到账", "拉黑", "报警", "报案", "还要钱", "退钱", "骗子"],
        "PII_LEAK": ["身份证", "手机号", "住址", "我的电话", "照片都发了", "我这就拍"],
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

    def _iter_pii_tokens(self, msg: Dict) -> Iterable[Tuple[str, str]]:
        pii_details = msg.get("pii_details", {})
        if not isinstance(pii_details, dict):
            return []

        out: List[Tuple[str, str]] = []
        for k, vals in pii_details.items():
            key = str(k).lower()
            ptype = None
            if "mobile" in key or "phone" in key:
                ptype = "phone"
            elif "id" in key:
                ptype = "id"
            elif "bank" in key:
                ptype = "bank"
            elif "name" in key:
                ptype = "name"
            elif "address" in key or "location" in key:
                ptype = "address"

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
        pii_nodes: Dict[str, List[str]] = defaultdict(list)

        for msg in messages:
            node = self._node_id(msg)
            user = msg.get("username", "unknown")
            group = msg.get("source_group", "unknown")

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
                pii_nodes[f"{ptype}:{pval}"].append(node)

        # same username across groups
        for nodes in username_nodes.values():
            if len(nodes) < 2:
                continue
            anchor = nodes[0]
            for n in nodes[1:]:
                dsu.union(anchor, n)

        # pii-based unions with confidence tiers
        for token, nodes in pii_nodes.items():
            if len(nodes) < 2:
                continue

            ptype, pval = token.split(":", 1)
            uniq_nodes = list(dict.fromkeys(nodes))
            if len(uniq_nodes) < 2:
                continue

            if ptype in {"phone", "id"}:
                anchor = uniq_nodes[0]
                for n in uniq_nodes[1:]:
                    dsu.union(anchor, n)
                continue

            if ptype == "name":
                # avoid merging by generic short names only
                if len(pval) < 2 or len(pval) > 6:
                    continue
                anchor = uniq_nodes[0]
                for n in uniq_nodes[1:]:
                    dsu.union(anchor, n)
                continue

            if ptype == "address":
                # address is useful for meetup linkage
                if len(pval) < 8:
                    continue
                anchor = uniq_nodes[0]
                for n in uniq_nodes[1:]:
                    dsu.union(anchor, n)
                continue

            if ptype == "bank":
                # bank card is weak evidence by itself: victims can repeat suspect account.
                # only merge if both sides are suspect-like or usernames are identical.
                for i in range(len(uniq_nodes)):
                    for j in range(i + 1, len(uniq_nodes)):
                        a = uniq_nodes[i]
                        b = uniq_nodes[j]
                        ua = self._normalize_username(node_meta[a]["username"])
                        ub = self._normalize_username(node_meta[b]["username"])
                        same_user = ua == ub and ua != "unknown"
                        both_suspect_like = self._node_is_suspect_like(node_signal[a]) and self._node_is_suspect_like(node_signal[b])
                        if same_user or both_suspect_like:
                            dsu.union(a, b)

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

            for token, t_nodes in pii_nodes.items():
                if any(n in nodes for n in t_nodes):
                    ptype, pval = token.split(":", 1)
                    shared_pii[ptype].add(pval)

            cluster_id = f"C{cid:03d}"
            cid += 1
            for node in nodes:
                node_to_cluster[node] = cluster_id

            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "canonical_name": canonical_name,
                    "aliases": sorted(aliases),
                    "groups": sorted(groups),
                    "members": sorted(members, key=lambda x: (x["source_group"], x["username"])),
                    "shared_pii": {k: sorted(v) for k, v in shared_pii.items()},
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
