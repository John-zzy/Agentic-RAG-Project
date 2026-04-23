from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from backend.config.settings import AppSettings, settings


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
        resolved_settings = app_settings or settings
        self._sqlite_path = sqlite_path or resolved_settings.session.sqlite_path
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _ensure_schema(self) -> None:
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._sqlite_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _parse_retrieval_snippets(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, str) or not payload:
            return []
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

