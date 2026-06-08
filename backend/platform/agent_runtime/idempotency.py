from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol, runtime_checkable

from backend.platform.agent_runtime.contracts import ToolObservation


ToolInvocationStatus = Literal["pending", "succeeded", "failed"]
CompensationStatus = Literal["none", "supported", "unsupported", "completed", "failed"]


@dataclass(frozen=True)
class ToolExecutionContext:
    """生成工具幂等 key 的稳定上下文，避免副作用工具因重试重复执行。"""

    session_id: str
    request_id: str
    run_id: str
    node_name: str
    turn_id: str | None = None
    step_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    def identity_parts(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "node_name": self.node_name,
            "turn_id": self.turn_id,
            "step_id": self.step_id,
        }


@dataclass(frozen=True)
class ToolInvocationRecord:
    """SQLite 中保存的一次工具调用事实。"""

    idempotency_key: str
    status: ToolInvocationStatus
    tool_name: str
    observation: ToolObservation | None = None
    compensation_status: CompensationStatus = "none"
    metadata: dict[str, Any] | None = None


@runtime_checkable
class CompensatableTool(Protocol):
    """副作用工具可选实现的补偿协议。"""

    def compensate(
        self,
        *,
        observation: ToolObservation,
        context: ToolExecutionContext,
    ) -> Mapping[str, Any] | None:
        """撤销或修正已执行副作用，并返回补偿审计信息。"""
        ...


class ToolIdempotencyStore(Protocol):
    """工具幂等仓储协议，便于测试和未来替换持久实现。"""

    def begin_invocation(
        self,
        *,
        key: str,
        tool_name: str,
        input_hash: str,
        context: ToolExecutionContext,
        compensation_status: CompensationStatus,
    ) -> ToolInvocationRecord | None:
        """不存在时写入 pending；存在时返回已有事实。"""
        ...

    def mark_succeeded(
        self,
        *,
        key: str,
        observation: ToolObservation,
        compensation_status: CompensationStatus,
    ) -> None:
        """工具成功后保存 observation，供重复调用复用。"""
        ...

    def mark_failed(
        self,
        *,
        key: str,
        observation: ToolObservation,
        compensation_status: CompensationStatus,
    ) -> None:
        """工具失败后保存 observation，供恢复时判断起点。"""
        ...

    def get(self, key: str) -> ToolInvocationRecord | None:
        """按幂等 key 读取工具调用事实。"""
        ...

    def list_by_session(self, session_id: str) -> tuple[ToolInvocationRecord, ...]:
        """读取某会话下的工具调用事实，恢复执行时使用。"""
        ...

    def list_by_request(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> tuple[ToolInvocationRecord, ...]:
        """读取某会话单次请求下的工具调用事实，恢复执行时使用。"""
        ...


class SQLiteToolIdempotencyStore:
    """基于 SQLite 的工具调用幂等仓储。"""

    def __init__(self, sqlite_path: Path) -> None:
        self._sqlite_path = sqlite_path
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._ensure_schema()

    def begin_invocation(
        self,
        *,
        key: str,
        tool_name: str,
        input_hash: str,
        context: ToolExecutionContext,
        compensation_status: CompensationStatus,
    ) -> ToolInvocationRecord | None:
        payload = _json_dumps(context.identity_parts())
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_tool_invocations (
                    idempotency_key,
                    status,
                    tool_name,
                    session_id,
                    request_id,
                    run_id,
                    node_name,
                    turn_id,
                    step_id,
                    input_hash,
                    context_json,
                    compensation_status,
                    created_at,
                    updated_at
                ) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    tool_name,
                    context.session_id,
                    context.request_id,
                    context.run_id,
                    context.node_name,
                    context.turn_id,
                    context.step_id,
                    input_hash,
                    payload,
                    compensation_status,
                    now,
                    now,
                ),
            )
            conn.commit()
            return None if conn.total_changes > 0 else self._select(conn, key)

    def mark_succeeded(
        self,
        *,
        key: str,
        observation: ToolObservation,
        compensation_status: CompensationStatus,
    ) -> None:
        self._mark_done(
            key=key,
            status="succeeded",
            observation=observation,
            compensation_status=compensation_status,
        )

    def mark_failed(
        self,
        *,
        key: str,
        observation: ToolObservation,
        compensation_status: CompensationStatus,
    ) -> None:
        self._mark_done(
            key=key,
            status="failed",
            observation=observation,
            compensation_status=compensation_status,
        )

    def get(self, key: str) -> ToolInvocationRecord | None:
        with self._connect() as conn:
            return self._select(conn, key)

    def list_by_session(self, session_id: str) -> tuple[ToolInvocationRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM agent_tool_invocations
                WHERE session_id = ?
                ORDER BY updated_at ASC, idempotency_key ASC
                """,
                (session_id,),
            ).fetchall()
            return tuple(_row_to_record(row) for row in rows)

    def list_by_request(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> tuple[ToolInvocationRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM agent_tool_invocations
                WHERE session_id = ?
                  AND request_id = ?
                ORDER BY updated_at ASC, idempotency_key ASC
                """,
                (session_id, request_id),
            ).fetchall()
            return tuple(_row_to_record(row) for row in rows)

    def _mark_done(
        self,
        *,
        key: str,
        status: ToolInvocationStatus,
        observation: ToolObservation,
        compensation_status: CompensationStatus,
    ) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_tool_invocations
                SET status = ?,
                    observation_json = ?,
                    compensation_status = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    status,
                    observation.model_dump_json(),
                    compensation_status,
                    now,
                    key,
                ),
            )
            conn.commit()

    def _ensure_schema(self) -> None:
        """创建工具幂等事实表；该表独立于 LangGraph checkpoint。"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_tool_invocations (
                    idempotency_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    turn_id TEXT,
                    step_id TEXT,
                    input_hash TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    observation_json TEXT,
                    compensation_status TEXT NOT NULL DEFAULT 'none',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_tool_invocations_session
                ON agent_tool_invocations(session_id, request_id, run_id)
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _select(
        self,
        conn: sqlite3.Connection,
        key: str,
    ) -> ToolInvocationRecord | None:
        row = conn.execute(
            """
            SELECT *
            FROM agent_tool_invocations
            WHERE idempotency_key = ?
            """,
            (key,),
        ).fetchone()
        return _row_to_record(row) if row is not None else None


def build_tool_idempotency_key(
    *,
    context: ToolExecutionContext,
    tool_name: str,
    input_payload: Mapping[str, Any] | None,
) -> str:
    """基于运行上下文、工具名和规范化入参生成稳定 key。"""
    raw = {
        **context.identity_parts(),
        "tool_name": tool_name,
        "input": _normalize_input(input_payload),
    }
    digest = hashlib.sha256(_json_dumps(raw).encode("utf-8")).hexdigest()
    return f"tool:{digest}"


def build_input_hash(input_payload: Mapping[str, Any] | None) -> str:
    normalized = _normalize_input(input_payload)
    return hashlib.sha256(_json_dumps(normalized).encode("utf-8")).hexdigest()


def is_side_effect_tool(tool: Any) -> bool:
    """判断工具是否需要幂等保护，避免只读检索工具被额外持久化。"""
    if bool(getattr(tool, "side_effect", False)):
        return True
    return getattr(tool, "capability_type", None) == "action"


def compensation_status_for_tool(tool: Any) -> CompensationStatus:
    if not is_side_effect_tool(tool):
        return "none"
    return "supported" if isinstance(tool, CompensatableTool) else "unsupported"


def attach_idempotency_trace(
    *,
    observation: ToolObservation,
    key: str,
    status: ToolInvocationStatus,
    reused: bool,
    compensation_status: CompensationStatus,
) -> ToolObservation:
    metadata = dict(observation.metadata)
    trace = dict(observation.trace)
    idempotency_trace = {
        "idempotency_key": key,
        "status": status,
        "reused": reused,
        "compensation_status": compensation_status,
    }
    metadata["idempotency"] = idempotency_trace
    metadata["compensation_status"] = compensation_status
    trace["idempotency"] = idempotency_trace
    trace["compensation"] = {"status": compensation_status}
    execution = observation.execution
    if execution is not None:
        execution = execution.model_copy(update={"idempotency_key": key})
    return observation.model_copy(
        update={
            "metadata": metadata,
            "trace": trace,
            "execution": execution,
        }
    )


def _row_to_record(row: sqlite3.Row) -> ToolInvocationRecord:
    observation = None
    if row["observation_json"]:
        observation = ToolObservation.model_validate_json(str(row["observation_json"]))
    metadata = json.loads(str(row["context_json"]))
    return ToolInvocationRecord(
        idempotency_key=str(row["idempotency_key"]),
        status=str(row["status"]),  # type: ignore[arg-type]
        tool_name=str(row["tool_name"]),
        observation=observation,
        compensation_status=str(row["compensation_status"]),  # type: ignore[arg-type]
        metadata=metadata,
    )


def _normalize_input(input_payload: Mapping[str, Any] | None) -> Any:
    if input_payload is None:
        return {}
    return json.loads(_json_dumps(dict(input_payload)))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
