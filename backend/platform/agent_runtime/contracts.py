from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.platform.workflow.state_machine import WorkflowRunState


AgentMode = Literal["react", "plan"]
ReActActionType = Literal["tool_call", "ask_user", "final_answer", "stop"]
ReActTurnStatus = Literal[
    "pending",
    "running",
    "waiting_user",
    "retrying",
    "succeeded",
    "failed",
    "cancelled",
]
PlanStepStatus = Literal[
    "pending",
    "running",
    "waiting_user",
    "retrying",
    "succeeded",
    "failed",
    "cancelled",
    "skipped",
]


class AgentRuntimeModel(BaseModel):
    """Agent Runtime 的可序列化模型基类，避免 checkpoint 混入未知字段。"""

    model_config = ConfigDict(extra="forbid")


class RetryMetadata(AgentRuntimeModel):
    """描述一次 turn/step/tool 的重试预算和当前进度。"""

    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=2, ge=0)
    retryable: bool = True
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionMetadata(AgentRuntimeModel):
    """ToolExecutor 调用前后需要保留的最小审计信息。"""

    tool_name: str
    tool_call_id: str | None = None
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=2, ge=0)
    retryable: bool = True
    timeout_ms: int | None = Field(default=None, ge=1)
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(AgentRuntimeModel):
    """统一工具观察结果，供 ReAct turn 或 Plan step 消费。"""

    tool_name: str
    success: bool
    tool_call_id: str | None = None
    output: Any = None
    result_summary: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
    requires_user: bool = False
    user_prompt: str | None = None
    error: str | None = None
    execution: ToolExecutionMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRun(AgentRuntimeModel):
    """顶层 Agent run 的通用快照；具体模式使用 ReActRun 或 PlanRun 承载细节。"""

    agent_run_id: str
    session_id: str
    request_id: str
    mode: AgentMode
    user_goal: str
    workflow_status: WorkflowRunState = "created"
    current_tool_call: ToolExecutionMetadata | None = None
    final_answer: str | None = None
    result_summary: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReActAction(AgentRuntimeModel):
    """顶层 ReAct 动作，只保存可审计摘要，不保存隐藏推理链。"""

    action_type: ReActActionType = "tool_call"
    tool_name: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    instruction: str | None = None
    rationale_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_tool_name_for_tool_call(self) -> "ReActAction":
        if self.action_type == "tool_call" and not self.tool_name:
            raise ValueError("tool_name is required when action_type is tool_call.")
        return self


class ReActTurn(AgentRuntimeModel):
    """ReAct 模式下的一轮顶层 action/observation。"""

    turn_id: str
    round_index: int = Field(ge=1)
    goal: str
    action: ReActAction
    status: ReActTurnStatus = "pending"
    input: dict[str, Any] = Field(default_factory=dict)
    tool_name: str | None = None
    observation: ToolObservation | None = None
    observation_summary: str = ""
    result_summary: str = ""
    retry_metadata: RetryMetadata = Field(default_factory=RetryMetadata)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def align_tool_name_with_action(self) -> "ReActTurn":
        if self.action.action_type == "tool_call":
            action_tool_name = self.action.tool_name
            if self.tool_name and self.tool_name != action_tool_name:
                raise ValueError("tool_name must match action.tool_name.")
            self.tool_name = action_tool_name
        return self


class ReActRun(AgentRuntimeModel):
    """简单任务的顶层 ReAct 运行记录。"""

    react_run_id: str
    session_id: str
    request_id: str
    mode: Literal["react"] = "react"
    user_goal: str
    workflow_status: WorkflowRunState = "running"
    max_turns: int = Field(default=5, ge=1)
    turns: list[ReActTurn] = Field(default_factory=list)
    current_turn_id: str | None = None
    current_tool_call: ToolExecutionMetadata | None = None
    final_answer: str | None = None
    result_summary: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(AgentRuntimeModel):
    """复杂任务计划中的一个可执行步骤。"""

    step_id: str
    goal: str
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: PlanStepStatus = "pending"
    observation: ToolObservation | None = None
    output: Any = None
    result_summary: str = ""
    retry_metadata: RetryMetadata = Field(default_factory=RetryMetadata)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanRun(AgentRuntimeModel):
    """复杂任务的显式计划运行记录。"""

    plan_run_id: str
    session_id: str
    request_id: str
    mode: Literal["plan"] = "plan"
    user_goal: str
    workflow_status: WorkflowRunState = "planning"
    steps: list[PlanStep] = Field(default_factory=list)
    current_step_id: str | None = None
    current_tool_call: ToolExecutionMetadata | None = None
    final_answer: str | None = None
    result_summary: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
