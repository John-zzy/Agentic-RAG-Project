from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from shutil import move
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_from_dict
from langchain_core.messages.base import message_to_dict

from backend.platform.config.settings import AppSettings, LEGACY_SQLITE_PATH, SQLITE_PATH, settings
from backend.platform.knowledge.sources import (
    parse_mounted_knowledge_sources,
    serialize_mounted_knowledge_sources,
)


SessionStatus = Literal["active", "expired"]


@dataclass(frozen=True)
class SessionRecord:
    """描述会话主记录的持久化状态。"""

    session_id: str
    scene: str
    mounted_knowledge_sources: tuple[str, ...]
    status: SessionStatus
    created_at: str
    updated_at: str
    last_active_at: str
    expired_at: str | None


@dataclass(frozen=True)
class SessionTurn:
    """描述单轮问答及其引用片段。"""

    session_id: str
    request_id: str
    user_message: str
    assistant_answer: str
    retrieval_snippets: list[dict[str, Any]]
    timestamp: str

    def to_messages(self) -> list[BaseMessage]:
        """将当前轮次转换为 LangChain message 列表。"""
        return [
            HumanMessage(content=self.user_message),
            AIMessage(content=self.assistant_answer),
        ]


class SQLiteSessionStore:
    """基于 SQLite 的会话、轮次与 LangChain message 持久化实现。"""

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
        scene: str = "generic_assistant",
        mounted_knowledge_sources: Sequence[str] | None = None,
        now: datetime | str | None = None,
    ) -> SessionRecord:
        """创建会话主记录；若已存在则直接返回当前状态。"""
        existing = self.get_session(session_id)
        if existing is not None:
            return existing

        timestamp = self._normalize_timestamp(now)
        serialized_sources = serialize_mounted_knowledge_sources(mounted_knowledge_sources)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    scene,
                    mounted_knowledge_sources,
                    status,
                    created_at,
                    updated_at,
                    last_active_at,
                    expired_at
                ) VALUES (?, ?, ?, 'active', ?, ?, ?, NULL)
                """,
                (session_id, scene, serialized_sources, timestamp, timestamp, timestamp),
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
                    scene,
                    mounted_knowledge_sources,
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
        """写入一轮问答记录，并同步写入 LangChain message 历史。"""
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
            self._insert_messages(
                conn=conn,
                session_id=session_id,
                request_id=request_id,
                messages=[
                    HumanMessage(content=user_message),
                    AIMessage(content=assistant_answer),
                ],
                timestamp=timestamp,
            )
            self._touch_session_record(conn, session_id=session_id, timestamp=timestamp)
            conn.commit()

    def append_messages(
        self,
        session_id: str,
        messages: Sequence[BaseMessage],
        *,
        timestamp: datetime | str | None = None,
        request_id: str | None = None,
    ) -> str:
        """追加 LangChain message 序列，供 BaseChatMessageHistory 适配层复用。"""
        if not messages:
            return request_id or uuid4().hex

        resolved_timestamp = self._normalize_timestamp(timestamp)
        resolved_request_id = request_id or uuid4().hex
        self.create_session(session_id=session_id, now=resolved_timestamp)
        with self._lock, self._connect() as conn:
            self._insert_messages(
                conn=conn,
                session_id=session_id,
                request_id=resolved_request_id,
                messages=messages,
                timestamp=resolved_timestamp,
            )
            self._touch_session_record(conn, session_id=session_id, timestamp=resolved_timestamp)
            conn.commit()
        return resolved_request_id

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
        return [self._row_to_session_turn(row) for row in ordered_rows]

    def get_messages(self, session_id: str, limit: int | None = None) -> list[BaseMessage]:
        """读取会话消息历史，兼容 LangChain BaseChatMessageHistory。"""
        query = """
            SELECT message_payload
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
        """
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            query = """
                SELECT message_payload
                FROM (
                    SELECT id, message_payload
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
            """
            params = (session_id, limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return self._parse_messages(rows)

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
        turns = [self._row_to_session_turn(row) for row in ordered_rows]
        return turns, total_turns

    def delete_session(self, session_id: str) -> int:
        """删除会话全部记录并返回删除条数。"""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = ?
                """,
                (session_id,),
            )
            turn_cursor = conn.execute(
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
        return max(int(turn_cursor.rowcount), 0)

    def _ensure_schema(self) -> None:
        """创建会话、轮次与消息历史表。"""
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
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    message_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sequence_index INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    scene TEXT NOT NULL DEFAULT 'generic_assistant',
                    mounted_knowledge_sources TEXT NOT NULL DEFAULT '["documents"]',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_active_at TEXT NOT NULL,
                    expired_at TEXT
                )
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "scene" not in columns:
                conn.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN scene TEXT NOT NULL DEFAULT 'generic_assistant'
                    """
                )
            if "mounted_knowledge_sources" not in columns:
                conn.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN mounted_knowledge_sources TEXT NOT NULL DEFAULT '["documents"]'
                    """
                )
            conn.execute(
                """
                UPDATE sessions
                SET mounted_knowledge_sources = '["documents"]'
                WHERE mounted_knowledge_sources IS NULL OR mounted_knowledge_sources = ''
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
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_id
                ON chat_messages(session_id, id)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_messages_request_sequence
                ON chat_messages(session_id, request_id, sequence_index)
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
                    scene,
                    status,
                    created_at,
                    updated_at,
                    last_active_at,
                    expired_at
                )
                SELECT
                    chat_turns.session_id,
                    'generic_assistant',
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
            self._backfill_messages(conn)
            conn.commit()

    def _backfill_messages(self, conn: sqlite3.Connection) -> None:
        """将旧 chat_turns 中尚未同步的轮次补写到消息历史表。"""
        rows = conn.execute(
            """
            SELECT
                session_id,
                request_id,
                user_message,
                assistant_answer,
                created_at
            FROM chat_turns
            WHERE NOT EXISTS (
                SELECT 1
                FROM chat_messages
                WHERE chat_messages.session_id = chat_turns.session_id
                  AND chat_messages.request_id = chat_turns.request_id
            )
            ORDER BY id ASC
            """
        ).fetchall()

        for row in rows:
            self._insert_messages(
                conn=conn,
                session_id=str(row["session_id"]),
                request_id=str(row["request_id"]),
                messages=[
                    HumanMessage(content=str(row["user_message"])),
                    AIMessage(content=str(row["assistant_answer"])),
                ],
                timestamp=str(row["created_at"]),
            )

    def _insert_messages(
        self,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        request_id: str,
        messages: Sequence[BaseMessage],
        timestamp: str,
    ) -> None:
        """批量写入 LangChain message，并保持消息顺序。"""
        if not messages:
            return

        rows = []
        for index, message in enumerate(messages):
            serialized = message_to_dict(message)
            rows.append(
                (
                    session_id,
                    request_id,
                    serialized["type"],
                    json.dumps(serialized, ensure_ascii=False),
                    timestamp,
                    index,
                )
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO chat_messages (
                session_id,
                request_id,
                message_type,
                message_payload,
                created_at,
                sequence_index
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _touch_session_record(
        self,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        timestamp: str,
    ) -> None:
        """在当前连接里更新会话活跃时间，避免重复开关连接。"""
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
        """将 JSON 字符串解析为检索片段列表，并兼容旧版引用结构。"""
        if not isinstance(payload, str) or not payload:
            return []
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(value, list):
            return []
        return [self._normalize_retrieval_snippet(item, index) for index, item in enumerate(value, start=1)]

    def _normalize_retrieval_snippet(self, item: Any, index: int) -> dict[str, Any]:
        """把历史 retrieval_snippet 规范化为当前 citation 契约。"""
        if not isinstance(item, dict):
            return {}

        namespace = self._coerce_str(item.get("namespace")) or "knowledge"
        citation_id = self._coerce_str(item.get("citation_id")) or f"{namespace}:{index}"
        snippet = self._coerce_str(item.get("snippet")) or ""
        source_kind = self._coerce_str(item.get("source_kind")) or namespace
        source_name = self._coerce_str(item.get("source_name")) or (
            self._coerce_str(item.get("source_path"))
            or self._coerce_str(item.get("document_id"))
            or citation_id
        )

        normalized: dict[str, Any] = {
            "index": self._coerce_int(item.get("index")) or index,
            "citation_id": citation_id,
            "namespace": namespace,
            "source_kind": source_kind,
            "source_name": source_name,
            "source_path": self._coerce_str(item.get("source_path")),
            "document_id": self._coerce_str(item.get("document_id")),
            "chunk_id": self._coerce_str(item.get("chunk_id")) or citation_id,
            "chunk_index": self._coerce_int(item.get("chunk_index")),
            "snippet": snippet,
            "score": self._coerce_float(item.get("score")),
            "rank": self._coerce_int(item.get("rank")) or index,
        }
        return normalized

    def _parse_messages(self, rows: Sequence[sqlite3.Row]) -> list[BaseMessage]:
        """将消息表记录解析为 LangChain message 列表。"""
        messages: list[BaseMessage] = []
        for row in rows:
            payload = row["message_payload"]
            if not isinstance(payload, str) or not payload:
                continue
            try:
                serialized = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(serialized, dict):
                continue
            parsed = messages_from_dict([serialized])
            if parsed:
                messages.extend(parsed)
        return messages

    def _row_to_session_turn(self, row: sqlite3.Row) -> SessionTurn:
        """将 SQLite 轮次记录转换为 SessionTurn。"""
        return SessionTurn(
            session_id=str(row["session_id"]),
            request_id=str(row["request_id"]),
            user_message=str(row["user_message"]),
            assistant_answer=str(row["assistant_answer"]),
            retrieval_snippets=self._parse_retrieval_snippets(row["retrieval_snippets"]),
            timestamp=str(row["created_at"]),
        )

    def _parse_session_record(self, row: sqlite3.Row | None) -> SessionRecord | None:
        """将 SQLite 行解析为 SessionRecord。"""
        if row is None:
            return None
        return SessionRecord(
            session_id=str(row["session_id"]),
            scene=str(row["scene"]),
            mounted_knowledge_sources=parse_mounted_knowledge_sources(
                row["mounted_knowledge_sources"]
                if "mounted_knowledge_sources" in row.keys()
                else None
            ),
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

    def _coerce_str(self, value: Any) -> str | None:
        """把常见标量安全转成字符串。"""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, int | float):
            return str(value)
        return None

    def _coerce_int(self, value: Any) -> int | None:
        """把整数样式的值安全转成 int。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _coerce_float(self, value: Any) -> float | None:
        """把分值安全转成 float。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None
