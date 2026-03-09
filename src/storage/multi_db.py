import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


class MultiDBManager:
    """
    Global + per-account sqlite storage.

    - global db keeps normalized, cross-group searchable data
    - per-account db simulates separate backend stores for each account
    """

    def __init__(self, db_root: Path):
        self.db_root = Path(db_root)
        self.account_root = self.db_root / "accounts"
        self.db_root.mkdir(parents=True, exist_ok=True)
        self.account_root.mkdir(parents=True, exist_ok=True)

        self.global_db_path = self.db_root / "global_index.db"
        self.global_conn = sqlite3.connect(self.global_db_path)
        self.global_conn.row_factory = sqlite3.Row
        self.account_conns: Dict[str, sqlite3.Connection] = {}

        self._init_global_schema()

    def _init_global_schema(self):
        cur = self.global_conn.cursor() # 创建游标对象用于执行指令

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_group TEXT,
                source_file TEXT,
                msg_index INTEGER,
                username TEXT,
                text TEXT,
                is_system_msg INTEGER,
                has_pii INTEGER,
                l1_risk_score REAL,
                l2_risk_score REAL,
                role TEXT,
                risk TEXT,
                intent TEXT,
                created_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pii_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                source_group TEXT,
                username TEXT,
                pii_type TEXT,
                pii_value TEXT,
                FOREIGN KEY(message_id) REFERENCES messages(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_clusters (
                cluster_id TEXT PRIMARY KEY,
                canonical_name TEXT,
                aliases_json TEXT,
                groups_json TEXT,
                shared_pii_json TEXT,
                created_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id TEXT,
                source_group TEXT,
                username TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id TEXT,
                source_group TEXT,
                username TEXT,
                msg_index INTEGER,
                event_type TEXT,
                detail TEXT
            )
            """
        )

        self.global_conn.commit()

    @staticmethod
    def _safe_filename(name: str) -> str:
        name = str(name or "unknown").strip()
        if not name:
            name = "unknown"
        return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

    def _get_account_conn(self, username: str) -> sqlite3.Connection:
        key = self._safe_filename(username)
        if key in self.account_conns:
            return self.account_conns[key]

        path = self.account_root / f"{key}.db"
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_group TEXT,
                source_file TEXT,
                msg_index INTEGER,
                text TEXT,
                has_pii INTEGER,
                l1_risk_score REAL,
                l2_risk_score REAL,
                role TEXT,
                risk TEXT,
                intent TEXT,
                created_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pii_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                pii_type TEXT,
                pii_value TEXT
            )
            """
        )

        conn.commit()
        self.account_conns[key] = conn
        return conn

    @staticmethod
    def _iter_pii_items(message: Dict) -> Iterable[Tuple[str, str]]:
        pii_details = message.get("pii_details", {})
        if not isinstance(pii_details, dict):
            return []

        rows: List[Tuple[str, str]] = []
        for k, vals in pii_details.items():
            if not isinstance(vals, list):
                continue
            for val in vals:
                vv = str(val or "").strip()
                if vv:
                    rows.append((str(k), vv))
        return rows

    def store_message(self, message: Dict) -> int:
        llm = message.get("llm_decision", {}) if isinstance(message.get("llm_decision", {}), dict) else {}

        cur = self.global_conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (
                source_group, source_file, msg_index, username, text,
                is_system_msg, has_pii, l1_risk_score, l2_risk_score,
                role, risk, intent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.get("source_group", ""),
                message.get("source_file", ""),
                int(message.get("msg_index", 0)),
                message.get("username", "unknown"),
                message.get("text", ""),
                1 if message.get("is_system_msg") else 0,
                1 if message.get("has_pii") else 0,
                float(message.get("l1_risk_score", 0) or 0),
                float(message.get("l2_risk_score", 0) or 0),
                llm.get("role", "other"),
                llm.get("risk", "low"),
                llm.get("intent", ""),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        msg_id = int(cur.lastrowid)

        for pii_type, pii_value in self._iter_pii_items(message):
            cur.execute(
                """
                INSERT INTO pii_evidence (message_id, source_group, username, pii_type, pii_value)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    message.get("source_group", ""),
                    message.get("username", "unknown"),
                    pii_type,
                    pii_value,
                ),
            )

        self.global_conn.commit()

        # Per-account backend DB
        acc_conn = self._get_account_conn(message.get("username", "unknown"))
        acc_cur = acc_conn.cursor()
        acc_cur.execute(
            """
            INSERT INTO messages (
                source_group, source_file, msg_index, text,
                has_pii, l1_risk_score, l2_risk_score, role, risk, intent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.get("source_group", ""),
                message.get("source_file", ""),
                int(message.get("msg_index", 0)),
                message.get("text", ""),
                1 if message.get("has_pii") else 0,
                float(message.get("l1_risk_score", 0) or 0),
                float(message.get("l2_risk_score", 0) or 0),
                llm.get("role", "other"),
                llm.get("risk", "low"),
                llm.get("intent", ""),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        acc_msg_id = int(acc_cur.lastrowid)
        for pii_type, pii_value in self._iter_pii_items(message):
            acc_cur.execute(
                "INSERT INTO pii_evidence (message_id, pii_type, pii_value) VALUES (?, ?, ?)",
                (acc_msg_id, pii_type, pii_value),
            )
        acc_conn.commit()

        return msg_id

    def store_identity_clusters(self, clusters: List[Dict]):
        cur = self.global_conn.cursor()
        cur.execute("DELETE FROM identity_members")
        cur.execute("DELETE FROM identity_clusters")

        now = datetime.now().isoformat(timespec="seconds")
        for c in clusters:
            cur.execute(
                """
                INSERT INTO identity_clusters (
                    cluster_id, canonical_name, aliases_json, groups_json, shared_pii_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    c.get("cluster_id"),
                    c.get("canonical_name", "unknown"),
                    json.dumps(c.get("aliases", []), ensure_ascii=False),
                    json.dumps(c.get("groups", []), ensure_ascii=False),
                    json.dumps(c.get("shared_pii", {}), ensure_ascii=False),
                    now,
                ),
            )

            for m in c.get("members", []):
                cur.execute(
                    "INSERT INTO identity_members (cluster_id, source_group, username) VALUES (?, ?, ?)",
                    (c.get("cluster_id"), m.get("source_group", ""), m.get("username", "unknown")),
                )

        self.global_conn.commit()

    def store_trace_events(self, events: List[Dict]):
        cur = self.global_conn.cursor()
        cur.execute("DELETE FROM trace_events")
        for e in events:
            cur.execute(
                """
                INSERT INTO trace_events (
                    cluster_id, source_group, username, msg_index, event_type, detail
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    e.get("cluster_id", ""),
                    e.get("source_group", ""),
                    e.get("username", "unknown"),
                    int(e.get("msg_index", 0)),
                    e.get("event_type", ""),
                    e.get("detail", ""),
                ),
            )
        self.global_conn.commit()

    def close(self):
        try:
            self.global_conn.commit()
            self.global_conn.close()
        except Exception:
            pass

        for conn in self.account_conns.values():
            try:
                conn.commit()
                conn.close()
            except Exception:
                pass
        self.account_conns.clear()
