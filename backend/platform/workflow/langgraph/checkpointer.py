from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class SQLiteLangGraphCheckpointer(BaseCheckpointSaver[int]):
    """基于 SQLite 的 LangGraph checkpoint 持久化适配层。"""

    def __init__(self, sqlite_path: Path) -> None:
        super().__init__()
        self._sqlite_path = sqlite_path
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._ensure_schema()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, parent_checkpoint_id = self._checkpoint_scope(config)
        checkpoint_id = str(checkpoint["id"])
        checkpoint_payload = checkpoint.copy()
        channel_values = checkpoint_payload.pop("channel_values")
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint_payload)
        metadata_type, metadata_blob = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO langgraph_checkpoints (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob,
                ),
            )
            self._persist_blobs(
                conn=conn,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                channel_values=channel_values,
                new_versions=new_versions,
            )
            conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = get_checkpoint_id(config)

        with self._connect() as conn:
            row = self._select_checkpoint_row(
                conn=conn,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )
            if row is None:
                return None
            return self._row_to_checkpoint_tuple(conn, row, config=config if checkpoint_id else None)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        query, params = self._build_list_query(config=config, before=before)
        yielded = 0
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            for row in rows:
                metadata = self.serde.loads_typed(
                    (row["metadata_type"], row["metadata_blob"])
                )
                if filter and not all(
                    metadata.get(key) == value for key, value in filter.items()
                ):
                    continue
                if limit is not None and yielded >= limit:
                    break
                yielded += 1
                yield self._row_to_checkpoint_tuple(
                    conn,
                    row,
                    metadata=metadata,
                )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns, checkpoint_id = self._write_scope(config)
        with self._lock, self._connect() as conn:
            for index, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, index)
                value_type, value_blob = self.serde.dumps_typed(value)
                statement = "INSERT OR IGNORE" if write_index >= 0 else "INSERT OR REPLACE"
                conn.execute(
                    f"""
                    {statement} INTO langgraph_writes (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_index,
                        channel,
                        value_type,
                        value_blob,
                        task_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_index,
                        channel,
                        value_type,
                        value_blob,
                        task_path,
                    ),
                )
            conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM langgraph_writes WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM langgraph_blobs WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM langgraph_checkpoints WHERE thread_id = ?", (thread_id,))
            conn.commit()

    def _ensure_schema(self) -> None:
        """创建 checkpoint、pending writes 与 channel blob 表结构。"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_type TEXT NOT NULL,
                    checkpoint_blob BLOB NOT NULL,
                    metadata_type TEXT NOT NULL,
                    metadata_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS langgraph_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    write_index INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_index
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_langgraph_checkpoints_thread_latest
                ON langgraph_checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_langgraph_writes_checkpoint
                ON langgraph_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, write_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_langgraph_blobs_channel_version
                ON langgraph_blobs(thread_id, checkpoint_ns, channel, version)
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _persist_blobs(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        channel_values: dict[str, Any],
        new_versions: ChannelVersions,
    ) -> None:
        # checkpoint 主体不保存 channel_values，按 LangGraph saver 约定拆到 blob 表。
        for channel, version in new_versions.items():
            if channel in channel_values:
                value_type, value_blob = self.serde.dumps_typed(channel_values[channel])
            else:
                value_type, value_blob = "empty", b""
            conn.execute(
                """
                INSERT OR REPLACE INTO langgraph_blobs (
                    thread_id,
                    checkpoint_ns,
                    channel,
                    version,
                    value_type,
                    value_blob
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    channel,
                    self._serialize_version(version),
                    value_type,
                    value_blob,
                ),
            )

    def _select_checkpoint_row(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
    ) -> sqlite3.Row | None:
        if checkpoint_id:
            return conn.execute(
                """
                SELECT *
                FROM langgraph_checkpoints
                WHERE thread_id = ?
                  AND checkpoint_ns = ?
                  AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()
        return conn.execute(
            """
            SELECT *
            FROM langgraph_checkpoints
            WHERE thread_id = ?
              AND checkpoint_ns = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (thread_id, checkpoint_ns),
        ).fetchone()

    def _build_list_query(
        self,
        *,
        config: RunnableConfig | None,
        before: RunnableConfig | None,
    ) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        if config is not None:
            configurable = config["configurable"]
            clauses.append("thread_id = ?")
            params.append(str(configurable["thread_id"]))
            if "checkpoint_ns" in configurable:
                clauses.append("checkpoint_ns = ?")
                params.append(str(configurable.get("checkpoint_ns", "")))
            if checkpoint_id := get_checkpoint_id(config):
                clauses.append("checkpoint_id = ?")
                params.append(checkpoint_id)
        before_checkpoint_id = get_checkpoint_id(before) if before is not None else None
        if before_checkpoint_id:
            clauses.append("checkpoint_id < ?")
            params.append(before_checkpoint_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return (
            f"""
            SELECT *
            FROM langgraph_checkpoints
            {where}
            ORDER BY checkpoint_id DESC
            """,
            tuple(params),
        )

    def _row_to_checkpoint_tuple(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        config: RunnableConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointTuple:
        thread_id = str(row["thread_id"])
        checkpoint_ns = str(row["checkpoint_ns"])
        checkpoint_id = str(row["checkpoint_id"])
        checkpoint = self.serde.loads_typed(
            (row["checkpoint_type"], row["checkpoint_blob"])
        )
        resolved_metadata = metadata or self.serde.loads_typed(
            (row["metadata_type"], row["metadata_blob"])
        )
        checkpoint["channel_values"] = self._load_blobs(
            conn=conn,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            versions=checkpoint["channel_versions"],
        )
        return CheckpointTuple(
            config=config
            or {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=resolved_metadata,
            parent_config=self._build_parent_config(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                parent_checkpoint_id=row["parent_checkpoint_id"],
            ),
            pending_writes=self._load_pending_writes(
                conn=conn,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            ),
        )

    def _load_blobs(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for channel, version in versions.items():
            row = conn.execute(
                """
                SELECT value_type, value_blob
                FROM langgraph_blobs
                WHERE thread_id = ?
                  AND checkpoint_ns = ?
                  AND channel = ?
                  AND version = ?
                """,
                (thread_id, checkpoint_ns, channel, self._serialize_version(version)),
            ).fetchone()
            if row is None or row["value_type"] == "empty":
                continue
            values[channel] = self.serde.loads_typed((row["value_type"], row["value_blob"]))
        return values

    def _load_pending_writes(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        rows = conn.execute(
            """
            SELECT task_id, channel, value_type, value_blob
            FROM langgraph_writes
            WHERE thread_id = ?
              AND checkpoint_ns = ?
              AND checkpoint_id = ?
            ORDER BY task_id ASC, write_index ASC
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        return [
            (
                str(row["task_id"]),
                str(row["channel"]),
                self.serde.loads_typed((row["value_type"], row["value_blob"])),
            )
            for row in rows
        ]

    def _build_parent_config(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: Any,
    ) -> RunnableConfig | None:
        if not parent_checkpoint_id:
            return None
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": str(parent_checkpoint_id),
            }
        }

    def _checkpoint_scope(
        self,
        config: RunnableConfig,
    ) -> tuple[str, str, str | None]:
        configurable = config["configurable"]
        metadata = dict(config.get("metadata") or {})
        return (
            str(configurable["thread_id"]),
            str(configurable.get("checkpoint_ns") or metadata.get("checkpoint_ns") or ""),
            str(configurable["checkpoint_id"]) if "checkpoint_id" in configurable else None,
        )

    def _write_scope(self, config: RunnableConfig) -> tuple[str, str, str]:
        configurable = config["configurable"]
        metadata = dict(config.get("metadata") or {})
        return (
            str(configurable["thread_id"]),
            str(configurable.get("checkpoint_ns") or metadata.get("checkpoint_ns") or ""),
            str(configurable["checkpoint_id"]),
        )

    def _serialize_version(self, version: Any) -> str:
        return f"{type(version).__name__}:{version!r}"
