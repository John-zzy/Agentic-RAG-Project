from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from backend.platform.workflow.langgraph.config import build_runtime_graph_config
from backend.platform.workflow.langgraph.state import RuntimeGraphState
from backend.platform.workflow.state_machine import is_terminal, validate_transition


RecoverMode = Literal["in_place", "fork"]


class RuntimeRecoveryError(ValueError):
    """内部恢复入口拒绝恢复时抛出的错误。"""


@dataclass(frozen=True)
class RuntimeRecoveryResult:
    """一次 recover_run 的恢复结果和审计快照。"""

    mode: RecoverMode
    state: RuntimeGraphState
    config: dict[str, Any]
    run_id: str
    recovered_from_run_id: str | None
    checkpoint: dict[str, Any]
    pending_writes: tuple[tuple[str, str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    idempotency_facts: tuple[dict[str, Any], ...]


class RuntimeRecoveryMixin:
    def recover_run(
        self,
        *,
        session_id: str,
        request_id: str,
        fork_failed: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeRecoveryResult:
        """内部恢复入口：retrying 原地继续，failed 只能 fork 新 run。"""
        config = build_runtime_graph_config(
            session_id=session_id,
            request_id=request_id,
            metadata=dict(metadata or {}),
        )
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        if checkpoint_tuple is None:
            raise RuntimeRecoveryError("No checkpoint found for recovery.")
        state = self._load_or_build_thread_state(
            session_id=session_id,
            request_id=request_id,
            config=config,
            require_checkpoint=True,
        )
        status = str(state.get("status") or "running")
        snapshot = _RecoverySnapshot(
            checkpoint=dict(checkpoint_tuple.checkpoint),
            pending_writes=tuple(checkpoint_tuple.pending_writes or ()),
            failures=_collect_failure_records(state),
            idempotency_facts=self._load_idempotency_facts(
                session_id=session_id,
                request_id=request_id,
            ),
        )
        if status == "retrying":
            return self._recover_retrying_run(
                state=state,
                config=config,
                snapshot=snapshot,
                metadata=metadata,
            )
        if status == "failed":
            if not fork_failed:
                raise RuntimeRecoveryError("failed run cannot recover in place.")
            return self._recover_failed_run(
                state=state,
                config=config,
                snapshot=snapshot,
                metadata=metadata,
            )
        if is_terminal(status):
            raise RuntimeRecoveryError(f"terminal run cannot be recovered: {status}.")
        raise RuntimeRecoveryError(f"run status is not recoverable: {status}.")

    def _recover_retrying_run(
        self,
        *,
        state: RuntimeGraphState,
        config: dict[str, Any],
        snapshot: "_RecoverySnapshot",
        metadata: Mapping[str, Any] | None,
    ) -> RuntimeRecoveryResult:
        run_id = str(state.get("run_id") or "")
        if not run_id:
            raise RuntimeRecoveryError("retrying recovery requires run_id.")
        run = self.lifecycle.create_run(
            thread_id=str(state["session_id"]),
            request_id=str(state["request_id"]),
            run_id=run_id,
            metadata={
                **dict(metadata or {}),
                "recovery": "retrying_in_place",
            },
        )
        self.lifecycle.mark_running(run)
        self.lifecycle.mark_retrying(run)
        self.lifecycle.mark_running(run)
        next_status = validate_transition("retrying", "retry")
        output = self._persist_state_update(
            state=state,
            config=config,
            update={
                "status": next_status,
                "state_event": "retry",
                "final_state": None,
                "metadata": _merge_recovery_metadata(
                    state=state,
                    mode="in_place",
                    recovered_from_run_id=None,
                    snapshot=snapshot,
                    extra=metadata,
                ),
            },
        )
        return _build_recovery_result(
            mode="in_place",
            state=output,
            config=config,
            run_id=run_id,
            recovered_from_run_id=None,
            snapshot=snapshot,
        )

    def _recover_failed_run(
        self,
        *,
        state: RuntimeGraphState,
        config: dict[str, Any],
        snapshot: "_RecoverySnapshot",
        metadata: Mapping[str, Any] | None,
    ) -> RuntimeRecoveryResult:
        recovered_from_run_id = str(state.get("run_id") or "")
        new_run_id = uuid4().hex
        run = self.lifecycle.create_run(
            thread_id=str(state["session_id"]),
            request_id=str(state["request_id"]),
            run_id=new_run_id,
            metadata={
                **dict(metadata or {}),
                "recovery": "failed_fork",
                "recovered_from_run_id": recovered_from_run_id,
            },
        )
        self.lifecycle.mark_running(run)
        output = self._persist_state_update(
            state=state,
            config=config,
            update={
                "status": "running",
                "run_id": new_run_id,
                "state_event": "run_start",
                "final_state": None,
                "metadata": _merge_recovery_metadata(
                    state=state,
                    mode="fork",
                    recovered_from_run_id=recovered_from_run_id,
                    snapshot=snapshot,
                    extra=metadata,
                ),
            },
        )
        return _build_recovery_result(
            mode="fork",
            state=output,
            config=config,
            run_id=new_run_id,
            recovered_from_run_id=recovered_from_run_id,
            snapshot=snapshot,
        )

    def _load_idempotency_facts(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> tuple[dict[str, Any], ...]:
        if self.tool_idempotency_store is None:
            return tuple()
        facts: list[dict[str, Any]] = []
        for record in self.tool_idempotency_store.list_by_request(
            session_id=session_id,
            request_id=request_id,
        ):
            facts.append(
                {
                    "idempotency_key": record.idempotency_key,
                    "status": record.status,
                    "tool_name": record.tool_name,
                    "compensation_status": record.compensation_status,
                }
            )
        return tuple(facts)


@dataclass(frozen=True)
class _RecoverySnapshot:
    checkpoint: dict[str, Any]
    pending_writes: tuple[tuple[str, str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    idempotency_facts: tuple[dict[str, Any], ...]


def _build_recovery_result(
    *,
    mode: RecoverMode,
    state: RuntimeGraphState,
    config: dict[str, Any],
    run_id: str,
    recovered_from_run_id: str | None,
    snapshot: _RecoverySnapshot,
) -> RuntimeRecoveryResult:
    return RuntimeRecoveryResult(
        mode=mode,
        state=state,
        config=config,
        run_id=run_id,
        recovered_from_run_id=recovered_from_run_id,
        checkpoint=snapshot.checkpoint,
        pending_writes=snapshot.pending_writes,
        failures=snapshot.failures,
        idempotency_facts=snapshot.idempotency_facts,
    )


def _merge_recovery_metadata(
    *,
    state: RuntimeGraphState,
    mode: RecoverMode,
    recovered_from_run_id: str | None,
    snapshot: _RecoverySnapshot,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    recovery = {
        "mode": mode,
        "recovered_from_run_id": recovered_from_run_id,
        "pending_write_count": len(snapshot.pending_writes),
        "failure_count": len(snapshot.failures),
        "idempotency_fact_count": len(snapshot.idempotency_facts),
    }
    return {
        **dict(state.get("metadata") or {}),
        **dict(extra or {}),
        "recovery": recovery,
        "idempotency_facts": list(snapshot.idempotency_facts),
    }


def _collect_failure_records(state: RuntimeGraphState) -> tuple[dict[str, Any], ...]:
    metadata = dict(state.get("metadata") or {})
    values: list[Any] = []
    for key in ("failures", "failure_records"):
        raw = metadata.get(key)
        if isinstance(raw, list):
            values.extend(raw)
    for run_key in ("react_run", "plan_run"):
        run = state.get(run_key)
        if isinstance(run, Mapping):
            run_metadata = run.get("metadata")
            if isinstance(run_metadata, Mapping):
                raw = run_metadata.get("failures")
                if isinstance(raw, list):
                    values.extend(raw)
    return tuple(dict(value) for value in values if isinstance(value, Mapping))
