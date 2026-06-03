from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from backend.application.runtime.api.chat.schemas import Citation
from backend.platform.workflow.langgraph.lifecycle import GraphRunRef
from backend.platform.workflow.langgraph.state import RuntimeGraphState
class PreparedGraphTurn(Protocol):
    """ChatService 准备好一轮聊天后，交给 graph runtime 使用的数据。"""

    session_id: str
    request_id: str
    user_message: str
    answer_mode: str
    final_decision: str | None
    knowledge_used: bool
    citations: list[Citation]
    retrieval_trace: Any
    scene_metadata: Any
    agent_mode: str
    react_run: dict[str, Any] | None
    plan_run: dict[str, Any] | None
    current_turn_id: str | None
    current_step_id: str | None
    current_tool_call: dict[str, Any] | None


AnswerBuilder = Callable[[PreparedGraphTurn], tuple[str, list[Citation]]]
HistoryLoader = Callable[[PreparedGraphTurn], Sequence[BaseMessage]]
HitlApproveExecutor = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
HitlRespondHandler = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
_HITL_PENDING_ACTIONS = {"tool_approval", "external_api_approval", "clarification"}
_HITL_ALLOWED_ACTIONS = {"approve", "edit", "reject", "respond"}


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
    citations: list[Citation]
    state: RuntimeGraphState
    config: dict[str, Any]
    run_id: str


@dataclass(frozen=True)
class RuntimeGraphRunHandle:
    """流式路径持有的 graph run 句柄，用于后续完成或失败状态写入。"""

    run: GraphRunRef
    state: RuntimeGraphState
    config: dict[str, Any]

    @property
    def run_id(self) -> str:
        """返回当前 graph run ID，方便 API/SSE 关联。"""
        return self.run.run_id


