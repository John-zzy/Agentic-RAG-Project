from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from backend.platform.workflow.state_machine import (
    WorkflowRunEvent,
    WorkflowRunState,
    ensure_workflow_state,
    validate_transition,
)

GraphRunStatus = WorkflowRunState


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
        self._append_initial(run)
        return run

    def mark_planning(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        """记录进入规划态；当前简单 graph 可跳过该状态。"""
        return self._transition(run, "plan_start")

    def mark_running(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        """记录进入执行态；从 retrying 回来时使用 retry 事件。"""
        latest = self.latest(run)
        event: WorkflowRunEvent = "retry" if latest and latest.status == "retrying" else "run_start"
        return self._transition(run, event)

    def mark_waiting_user(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        """记录 graph interrupt 后等待人工输入。"""
        return self._transition(run, "interrupt")

    def mark_retrying(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        """记录一次可重试错误，表示 runtime 正准备下一次尝试。"""
        return self._transition(run, "tool_error_retryable")

    def mark_succeeded(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        """记录正常完成终态。"""
        return self._transition(run, "success")

    def mark_failed(
        self,
        run: GraphRunRef,
        error: BaseException | str,
    ) -> GraphRunLifecycleEvent:
        """记录异常失败终态；失败只用于模型、工具或 runtime 错误。"""
        return self._transition(run, "fail", error=self._summarize_error(error))

    def mark_cancelled(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        """记录人工拒绝或系统取消终态，和 failed 明确区分。"""
        return self._transition(run, "cancel")

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

    def _append_initial(self, run: GraphRunRef) -> GraphRunLifecycleEvent:
        """创建 run 时只能写入 created，后续状态必须走转移表。"""
        return self._append(run, ensure_workflow_state("created"))

    def _transition(
        self,
        run: GraphRunRef,
        event: WorkflowRunEvent,
        *,
        error: str | None = None,
    ) -> GraphRunLifecycleEvent:
        """读取当前 run 最新状态，校验事件后再追加 lifecycle 事件。"""
        latest = self.latest(run)
        if latest is None:
            raise ValueError("graph run has no created lifecycle event.")
        next_status = validate_transition(latest.status, event)
        return self._append(run, next_status, error=error)

    def _summarize_error(self, error: BaseException | str) -> str:
        if isinstance(error, str):
            return error
        message = str(error)
        return f"{type(error).__name__}: {message}" if message else type(error).__name__
