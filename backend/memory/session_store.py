from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from shutil import move
from threading import Lock
from typing import Any

from backend.config.settings import AppSettings, LEGACY_SQLITE_PATH, SQLITE_PATH, settings


@dataclass(frozen=True)
class SessionTurn:
    session_id: str
    request_id: str
    user_message: str
    assistant_answer: str
    retrieval_snippets: list[dict[str, Any]]
    timestamp: str


class SQLiteSessionStore:
    def __init__(
        self,
        app_settings: AppSettings | None = None,
        sqlite_path: Path | None = None,
    ) -> None:
        """初始化 SQLite 会话存储并确保表结构存在。"""
        resolved_settings = app_settings or settings
        self._sqlite_path = sqlite_path or resolved_settings.session.sqlite_path
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_sqlite_files(sqlite_path)
        self._lock = Lock()
        self._ensure_schema()

    def append_turn(
        self,
        session_id: str,
        request_id: str,
        user_message: str,
        assistant_answer: str,
        retrieval_snippets: list[dict[str, Any]],
        timestamp: str,
    ) -> None:
        """写入一轮对话记录。"""
        payload = json.dumps(retrieval_snippets, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_turns (
                    session_id,
                    request_id,
                    user_message,
                    assistant_answer,
                    retrieval_snippets,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    request_id,
                    user_message,
                    assistant_answer,
                    payload,
                    timestamp,
                ),
            )
            conn.commit()

    def get_recent_turns(self, session_id: str, limit: int) -> list[SessionTurn]:
        """读取指定会话最近 N 轮对话（按时间正序返回）。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    session_id,
                    request_id,
                    user_message,
                    assistant_answer,
                    retrieval_snippets,
                    created_at
                FROM chat_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        ordered_rows = list(reversed(rows))
        return [
            SessionTurn(
                session_id=str(row["session_id"]),
                request_id=str(row["request_id"]),
                user_message=str(row["user_message"]),
                assistant_answer=str(row["assistant_answer"]),
                retrieval_snippets=self._parse_retrieval_snippets(row["retrieval_snippets"]),
                timestamp=str(row["created_at"]),
            )
            for row in ordered_rows
        ]

    def count_turns(self, session_id: str) -> int:
        """统计会话累计轮次。"""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS turn_count
                FROM chat_turns
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return 0
        return int(row["turn_count"])

    def get_session_detail(
        self, session_id: str, limit: int
    ) -> tuple[list[SessionTurn], int]:
        """获取会话详情：最近轮次列表及总轮次。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM chat_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            total_turns = int(row["cnt"]) if row else 0

            rows = conn.execute(
                """
                SELECT
                    session_id,
                    request_id,
                    user_message,
                    assistant_answer,
                    retrieval_snippets,
                    created_at
                FROM chat_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        ordered_rows = list(reversed(rows))
        turns = [
            SessionTurn(
                session_id=str(r["session_id"]),
                request_id=str(r["request_id"]),
                user_message=str(r["user_message"]),
                assistant_answer=str(r["assistant_answer"]),
                retrieval_snippets=self._parse_retrieval_snippets(r["retrieval_snippets"]),
                timestamp=str(r["created_at"]),
            )
            for r in ordered_rows
        ]
        return turns, total_turns

    def delete_session(self, session_id: str) -> int:
        """删除会话全部记录并返回删除条数。"""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM chat_turns
                WHERE session_id = ?
                """,
                (session_id,),
            )
            conn.commit()
        return max(int(cursor.rowcount), 0)

    def _ensure_schema(self) -> None:
        """创建会话表和索引（若不存在）。"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_answer TEXT NOT NULL,
                    retrieval_snippets TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_turns_session_id_id
                ON chat_turns(session_id, id)
                """
            )
            conn.commit()

    def _migrate_legacy_sqlite_files(self, explicit_sqlite_path: Path | None) -> None:
        """Move the legacy session database into backend/data on first startup."""
        if explicit_sqlite_path is not None or self._sqlite_path != SQLITE_PATH:
            return
        if self._sqlite_path.exists() or not LEGACY_SQLITE_PATH.exists():
            return

        for suffix in ("", "-wal", "-shm", "-journal"):
            legacy_path = Path(f"{LEGACY_SQLITE_PATH}{suffix}")
            target_path = Path(f"{self._sqlite_path}{suffix}")
            if legacy_path.exists() and not target_path.exists():
                move(str(legacy_path), str(target_path))

    def _connect(self) -> sqlite3.Connection:
        """创建 SQLite 连接并应用基础 PRAGMA。"""
        connection = sqlite3.connect(
            str(self._sqlite_path),
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _parse_retrieval_snippets(self, payload: Any) -> list[dict[str, Any]]:
        """将 JSON 字符串解析为检索片段列表。"""
        if not isinstance(payload, str) or not payload:
            return []
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []
