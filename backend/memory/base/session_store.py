from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from shutil import move
from threading import Lock
from typing import Any, Literal

from backend.config.settings import AppSettings, LEGACY_SQLITE_PATH, SQLITE_PATH, settings


SessionStatus = Literal["active", "expired"]


@dataclass(frozen=True)
class SessionRecord:
    """描述会话主记录的持久化状态。"""

    session_id: str
    status: SessionStatus
    created_at: str
    updated_at: str
    last_active_at: str
    expired_at: str | None


@dataclass(frozen=True)
class SessionTurn:
    """描述单轮对话的持久化内容。"""

    session_id: str
    request_id: str
    user_message: str
    assistant_answer: str
    retrieval_snippets: list[dict[str, Any]]
    timestamp: str


class SQLiteSessionStore:
    """基于 SQLite 的会话与对话轮次存储实现。"""

    def __init__(
        self,
        app_settings: AppSettings | None = None,
        sqlite_path: Path | None = None,
    ) -> None:
        """初始化 SQLite 会话存储并确保表结构存在。"""
        resolved_settings = app_settings or settings
        self._settings = resolved_settings
        self._sqlite_path = sqlite_path or resolved_settings.session.sqlite_path
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_sqlite_files(sqlite_path)
        self._lock = Lock()
        self._ensure_schema()

    def create_session(
        self,
        session_id: str,
        now: datetime | str | None = None,
    ) -> SessionRecord:
        """创建会话主记录；若已存在则直接返回当前状态。"""
        existing = self.get_session(session_id)
        if existing is not None:
            return existing

        timestamp = self._normalize_timestamp(now)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    status,
                    created_at,
                    updated_at,
                    last_active_at,
                    expired_at
                ) VALUES (?, 'active', ?, ?, ?, NULL)
                """,
                (session_id, timestamp, timestamp, timestamp),
            )
            conn.commit()

        created = self.get_session(session_id)
        if created is None:
            raise RuntimeError(f"Failed to create session: {session_id}")
        return created

    def get_session(self, session_id: str) -> SessionRecord | None:
        """读取会话主记录，不存在时返回 None。"""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    session_id,
                    status,
                    created_at,
                    updated_at,
                    last_active_at,
                    expired_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return self._parse_session_record(row)

    def touch_session(
        self,
        session_id: str,
        now: datetime | str | None = None,
    ) -> SessionRecord | None:
        """更新会话最后活跃时间；若会话不存在则返回 None。"""
        current = self.get_session(session_id)
        if current is None:
            return None

        timestamp = self._normalize_timestamp(now)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET
                    updated_at = ?,
                    last_active_at = ?,
                    expired_at = CASE WHEN status = 'active' THEN NULL ELSE expired_at END
                WHERE session_id = ?
                """,
                (timestamp, timestamp, session_id),
            )
            conn.commit()

        return self.get_session(session_id)

    def cleanup_expired_sessions(
        self,
        now: datetime | str | None = None,
        timeout_minutes: int | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """标记超时会话为过期，返回本次过期的 session_id 列表。"""
        resolved_now = self._normalize_datetime(now)
        resolved_timeout_minutes = timeout_minutes or self._settings.session.timeout_minutes
        resolved_limit = limit or self._settings.session.cleanup_batch_size
        cutoff = (resolved_now - timedelta(minutes=resolved_timeout_minutes)).isoformat()
        expired_at = resolved_now.isoformat()

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE status = 'active'
                  AND last_active_at <= ?
                ORDER BY last_active_at ASC
                LIMIT ?
                """,
                (cutoff, resolved_limit),
            ).fetchall()
            expired_session_ids = [str(row["session_id"]) for row in rows]

            if not expired_session_ids:
                return []

            placeholders = ", ".join("?" for _ in expired_session_ids)
            conn.execute(
                f"""
                UPDATE sessions
                SET
                    status = 'expired',
                    updated_at = ?,
                    expired_at = ?
                WHERE session_id IN ({placeholders})
                """,
                (expired_at, expired_at, *expired_session_ids),
            )
            conn.commit()

        return expired_session_ids

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
        self.create_session(session_id=session_id, now=timestamp)
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
            conn.execute(
                """
                UPDATE sessions
                SET
                    updated_at = ?,
                    last_active_at = ?
                WHERE session_id = ?
                """,
                (timestamp, timestamp, session_id),
            )
            conn.commit()

    def get_recent_turns(self, session_id: str, limit: int) -> list[SessionTurn]:
        """读取指定会话最近 N 轮对话，按时间正序返回。"""
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
            conn.execute(
                """
                DELETE FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
            conn.commit()
        return max(int(cursor.rowcount), 0)

    def _ensure_schema(self) -> None:
        """创建会话表和索引。"""
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
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_active_at TEXT NOT NULL,
                    expired_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_turns_session_id_id
                ON chat_turns(session_id, id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_status_last_active
                ON sessions(status, last_active_at)
                """
            )
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    status,
                    created_at,
                    updated_at,
                    last_active_at,
                    expired_at
                )
                SELECT
                    chat_turns.session_id,
                    'active',
                    MIN(chat_turns.created_at),
                    MAX(chat_turns.created_at),
                    MAX(chat_turns.created_at),
                    NULL
                FROM chat_turns
                LEFT JOIN sessions
                    ON sessions.session_id = chat_turns.session_id
                WHERE sessions.session_id IS NULL
                GROUP BY chat_turns.session_id
                """
            )
            conn.commit()

    def _migrate_legacy_sqlite_files(self, explicit_sqlite_path: Path | None) -> None:
        """首次启动时将旧会话数据库迁移到 backend/data。"""
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

    def _parse_session_record(self, row: sqlite3.Row | None) -> SessionRecord | None:
        """将 SQLite 行解析为 SessionRecord。"""
        if row is None:
            return None
        return SessionRecord(
            session_id=str(row["session_id"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_active_at=str(row["last_active_at"]),
            expired_at=str(row["expired_at"]) if row["expired_at"] else None,
        )

    def _normalize_timestamp(self, value: datetime | str | None) -> str:
        """将输入统一转换为 ISO 8601 时间戳。"""
        return self._normalize_datetime(value).isoformat()

    def _normalize_datetime(self, value: datetime | str | None) -> datetime:
        """将输入统一转换为 UTC datetime。"""
        if value is None:
            return datetime.now(UTC)
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
