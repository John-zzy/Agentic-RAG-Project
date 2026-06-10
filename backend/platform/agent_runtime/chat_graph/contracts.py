from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from backend.platform.workflow.langgraph.state import RuntimeGraphState

_HITL_PENDING_ACTIONS = {"tool_approval", "external_api_approval", "clarification"}
_HITL_ALLOWED_ACTIONS = {"approve", "edit", "reject", "respond"}


class PreparedChatTurn(Protocol):
    """chat_graph 运行时所需的最小准备上下文。"""

    session_id: str
    request_id: str
    user_message: str
    answer_mode: str
    final_decision: str | None
    knowledge_used: bool
    citations: list[Any]
    retrieval_trace: Any
    scene_metadata: Any
    agent_mode: str
    agent_mode_reason: str
    agent_mode_signals: dict[str, Any] | None
    react_run: dict[str, Any] | None
    plan_run: dict[str, Any] | None
    current_turn_id: str | None
    current_step_id: str | None
    current_tool_call: dict[str, Any] | None


PreparedGraphTurn = PreparedChatTurn
AnswerBuilder = Callable[[PreparedChatTurn], tuple[str, list[Any]]]
HistoryLoader = Callable[[PreparedChatTurn], Sequence[BaseMessage]]
HitlApproveExecutor = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
HitlRespondHandler = Callable[[Mapping[str, Any], RuntimeGraphState], Mapping[str, Any] | None]
HitlPlanToolExecutor = Any


class ChatGraphDependencies(Protocol):
    """chat_graph 节点的业务依赖协议。"""

    prepared: PreparedChatTurn
    answer_builder: AnswerBuilder
    build_agent_runtime_success_update: Callable[
        [RuntimeGraphState, str, Sequence[Any], bool],
        dict[str, Any],
    ]
    select_agent_mode: Callable[[PreparedChatTurn], dict[str, Any]] | None
    build_react_deps: Callable[[PreparedChatTurn, RuntimeGraphState], Any] | None
    build_plan_graph_deps: Callable[[PreparedChatTurn, RuntimeGraphState], Any] | None
    build_prepared_from_state: Callable[
        [PreparedChatTurn, RuntimeGraphState],
        PreparedChatTurn,
    ] | None
    build_hitl_wait_update: Callable[
        [PreparedChatTurn, RuntimeGraphState],
        dict[str, Any],
    ] | None


class HitlResumeError(ValueError):
    """用户恢复等待任务失败时抛出的错误，例如等待点过期或动作不允许。"""


@dataclass(frozen=True)
class HitlWaitInput:
    """让当前会话暂停并等待用户处理时需要的信息。"""

    session_id: str
    request_id: str
    reason: str
    pending_action: str
    allowed_actions: Sequence[str]
    proposed_tool_call: Mapping[str, Any] | None = None
    suggested_responses: Sequence[Mapping[str, Any]] | None = None
    allow_freeform_response: bool = False
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HitlResumeInput:
    """用户点击批准、拒绝或补充信息后，恢复会话时带上的信息。"""

    session_id: str
    request_id: str
    interrupt_id: str
    action: str
    payload: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HitlRuntimeResult:
    """HITL 等待或恢复后，返回最新的 graph 状态和本次运行信息。"""

    state: RuntimeGraphState
    config: dict[str, Any]
    run_id: str
    tool_result: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeGraphResult:
    """一次普通 graph 执行完成后的回答、引用和运行信息。"""

    answer: str
    citations: list[Any]
    state: RuntimeGraphState
    config: dict[str, Any]
    run_id: str
