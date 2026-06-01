from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4


GraphRunStatus = Literal["created", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class GraphRunRef:
    """标识一次 graph run，供 lifecycle 后续状态变更复用。"""

    run_id: str
    thread_id: str
    request_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRunLifecycleEvent:
    """记录一次 graph run lifecycle 状态变更。"""

    run_id: str
    thread_id: str
    request_id: str
    status: GraphRunStatus
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class GraphRunLifecycleRecorder:
    """内存态 lifecycle 记录器，后续可由 application facade 替换为持久实现。"""

    def __init__(self) -> None:
        self._events: list[GraphRunLifecycleEvent] = []

    def create_run(
        self,
        *,
        thread_id: str,
        request_id: str,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> GraphRunRef:
        if not thread_id:
            raise ValueError("thread_id is required for graph run lifecycle.")
        if not request_id:
            raise ValueError("request_id is required for graph run lifecycle.")

        run = GraphRunRef(
            run_id=run_id or uuid4().hex,
            thread_id=thread_id,
            request_id=request_id,
            metadata=dict(metadata or {}),
        )
        self._append(run, "created")
        return run

    def mark_running(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        return self._append(run, "running")

    def mark_succeeded(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        return self._append(run, "succeeded")

    def mark_failed(
        self,
        run: GraphRunRef,
        error: BaseException | str,
    ) -> GraphRunLifecycleEvent:
        return self._append(run, "failed", error=self._summarize_error(error))

    def events(self, run_id: str | None = None) -> tuple[GraphRunLifecycleEvent, ...]:
        if run_id is None:
            return tuple(self._events)
        return tuple(event for event in self._events if event.run_id == run_id)

    def events_for_thread(
        self,
        *,
        thread_id: str,
        request_id: str | None = None,
    ) -> tuple[GraphRunLifecycleEvent, ...]:
        return tuple(
            event
            for event in self._events
            if event.thread_id == thread_id
            and (request_id is None or event.request_id == request_id)
        )

    def events_for_request(self, request_id: str) -> tuple[GraphRunLifecycleEvent, ...]:
        return tuple(event for event in self._events if event.request_id == request_id)

    def statuses(self, run: GraphRunRef) -> tuple[GraphRunStatus, ...]:
        return tuple(event.status for event in self.events(run.run_id))

    def latest(self, run: GraphRunRef) -> GraphRunLifecycleEvent | None:
        events = self.events(run.run_id)
        return events[-1] if events else None

    def _append(
        self,
        run: GraphRunRef,
        status: GraphRunStatus,
        *,
        error: str | None = None,
    ) -> GraphRunLifecycleEvent:
        event = GraphRunLifecycleEvent(
            run_id=run.run_id,
            thread_id=run.thread_id,
            request_id=run.request_id,
            status=status,
            timestamp=datetime.now(UTC),
            metadata=dict(run.metadata),
            error=error,
        )
        self._events.append(event)
        return event

    def _summarize_error(self, error: BaseException | str) -> str:
        if isinstance(error, str):
            return error
        message = str(error)
        return f"{type(error).__name__}: {message}" if message else type(error).__name__
