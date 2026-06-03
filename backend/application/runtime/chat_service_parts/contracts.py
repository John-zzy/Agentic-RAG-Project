from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.documents import Document

from backend.application.runtime.api.chat.schemas import Citation, RetrievalTrace
from backend.platform.agent_runtime.contracts import ReActAction
from backend.platform.agent_runtime.react import ReActActionContext
from backend.platform.models.base.router import TaskComplexity

RuntimeFinalDecision = Literal[
    "answer_with_evidence",
    "ask_user",
    "max_rounds_reached",
    "no_evidence",
    "retrieval_failed",
]
AnswerMode = Literal["evidence_answer", "follow_up", "fallback"]
class RetrievalChainModel(Protocol):
    """定义运行时依赖的最小模型构建协议。"""

    def get_runnable(
            self,
            complexity: TaskComplexity = "simple",
            prompt_template: Any | None = None,
            *,
            output_parser: Any | None = None,
    ) -> Any:
        """返回可供 runtime 执行的 LCEL runnable。"""
        ...

    def invoke_runnable(
            self,
            runnable: Any,
            input: Any,
            *,
            config: Any | None = None,
    ) -> Any:
        """同步执行 runnable。"""
        ...

    def stream_runnable(
            self,
            runnable: Any,
            input: Any,
            *,
            config: Any | None = None,
    ) -> Iterator[Any]:
        """流式执行 runnable。"""
        ...


class ChatServiceError(RuntimeError):
    """封装可返回给 API 层的业务错误。"""

    def __init__(self, *, status_code: int, code: str, message: str, request_id: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(message)


@dataclass(frozen=True)
class SceneMetadata:
    """描述统一聊天响应需要返回的场景元数据。"""

    scene: str
    agent: str | None = None


@dataclass(frozen=True)
class PreparedChatTurn:
    """封装同步和流式路径共享的聊天准备结果。"""

    session_id: str
    request_id: str
    timestamp: str
    user_message: str
    documents: list[Document]
    tool_event: dict[str, Any]
    retrieval_trace: RetrievalTrace
    citations: list[Citation]
    knowledge_used: bool
    scene_metadata: SceneMetadata
    complexity: TaskComplexity | None
    # 同步与 SSE 后续共用的回答分支元数据
    final_decision: RuntimeFinalDecision | None = None
    follow_up_question: str | None = None
    answer_mode: AnswerMode = "fallback"
    hitl_clarification_enabled: bool = False
    agent_mode: str = "react"
    agent_mode_reason: str = "default_simple_react"
    agent_mode_signals: dict[str, Any] | None = None
    react_run: dict[str, Any] | None = None
    plan_run: dict[str, Any] | None = None
    current_turn_id: str | None = None
    current_step_id: str | None = None
    current_tool_call: dict[str, Any] | None = None
    tool_observation: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentRuntimeExecutionResult:
    """Agent Runtime 工具执行后的 application 层归一化结果。"""

    documents: list[Document]
    tool_event: dict[str, Any]
    retrieval_trace: RetrievalTrace
    citations: list[Citation]
    knowledge_used: bool
    final_decision: RuntimeFinalDecision | None
    follow_up_question: str | None
    answer_mode: AnswerMode
    react_run: dict[str, Any] | None = None
    plan_run: dict[str, Any] | None = None
    current_turn_id: str | None = None
    current_step_id: str | None = None
    current_tool_call: dict[str, Any] | None = None
    tool_observation: dict[str, Any] | None = None


class _SingleToolReActActionSelector:
    """最小 ReAct selector：第一轮调用工具，第二轮交给 final synthesis。"""

    def __init__(self, *, tool_name: str, input_payload: Mapping[str, Any]) -> None:
        self._tool_name = tool_name
        self._input_payload = dict(input_payload)

    def select_action(self, context: ReActActionContext) -> ReActAction:
        if context.previous_turns:
            return ReActAction(
                action_type="final_answer",
                rationale_summary="Tool observation is ready for final synthesis.",
            )
        return ReActAction(
            action_type="tool_call",
            tool_name=self._tool_name,
            input=self._input_payload,
            rationale_summary="Top-level ReAct selected the RAG tool through ToolExecutor.",
        )



