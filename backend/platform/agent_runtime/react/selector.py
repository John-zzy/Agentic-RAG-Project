from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, Literal, Protocol

from langchain_core.prompts import PromptTemplate
from pydantic import Field, ValidationError, model_validator

from backend.platform.agent_runtime.contracts import (
    AgentRuntimeModel,
    ReActAction,
    ReActRun,
    ReActTurn,
    ToolObservation,
)
from backend.platform.agent_runtime.react.policy import (
    ReActScenePolicy,
    public_scene_policy,
)
from backend.platform.agent_runtime.react.state import attempted_tools, observation_final_decision
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.agent_runtime.validation import (
    ToolAccessValidationError,
    ToolInputValidationError,
)


class ReActActionContext(AgentRuntimeModel):
    """ReAct action selector 可见的审计上下文，不包含隐藏推理链。"""

    react_run_id: str
    session_id: str
    request_id: str
    user_goal: str
    round_index: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    allowed_tools: list[str] = Field(default_factory=list)
    previous_turns: list[ReActTurn] = Field(default_factory=list)
    run_observations: list[ToolObservation] = Field(default_factory=list)
    attempted_tools: list[str] = Field(default_factory=list)
    latest_final_decision: str | None = None
    scene_policy: ReActScenePolicy = Field(default_factory=ReActScenePolicy)
    resume_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMReActActionOutput(AgentRuntimeModel):
    """LLM 结构化输出 schema，只允许声明可审计动作。"""

    action_type: Literal["tool_call", "ask_user", "final_answer", "stop"]
    tool_name: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    instruction: str | None = None
    rationale_summary: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "LLMReActActionOutput":
        if self.action_type == "tool_call" and not self.tool_name:
            raise ValueError("tool_name is required when action_type is tool_call.")
        if self.action_type != "tool_call" and self.tool_name is not None:
            raise ValueError("tool_name is only allowed when action_type is tool_call.")
        if self.action_type != "tool_call" and self.input:
            raise ValueError("input is only allowed when action_type is tool_call.")
        return self


class ReActSelectorError(ValueError):
    """ReAct selector 的统一错误类型。"""


class ReActSelectorOutputError(ReActSelectorError):
    """模型输出无法被解析为合法结构化 action。"""


class ReActSelectorActionValidationError(ReActSelectorError):
    """模型动作违反工具 allowlist 或 input schema。"""


class ReActTurnBudgetError(ReActSelectorActionValidationError):
    """模型在 turn budget 耗尽后仍要求继续执行工具。"""


class ReActActionSelectionModel(Protocol):
    """LLM selector 依赖的最小模型协议。"""

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ) -> Any:
        """返回可执行的模型 runnable。"""

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> Any:
        """执行模型 runnable 并返回原始输出。"""


class ReActActionSelector(Protocol):
    """选择下一步 ReAct 顶层动作的中立协议。"""

    def select_action(self, context: ReActActionContext) -> ReActAction:
        """Return the next auditable ReAct action."""


REACT_ACTION_SELECTION_PROMPT = PromptTemplate.from_template(
    """REACT_SELECTOR
你是顶层 Agent Runtime 的 ReAct 调度器，只负责决定下一步动作。

你只能看到下面这些公开上下文：
- 用户目标
- 当前轮次和最大轮次
- 当前允许工具列表
- 已尝试工具
- scene 策略
- 历史 turn 摘要
- run observations 摘要
- 上一轮 final decision 与 resume metadata 摘要

你看不到原始工具结果、完整 checkpoint、内部系统提示词，也不要输出隐藏推理链。

输出要求：
- 只返回一个 JSON object，不要输出 Markdown 或解释。
- action_type 只能是 "tool_call"、"ask_user"、"final_answer" 或 "stop"。
- 如果 action_type="tool_call"，必须提供 tool_name；input 只填写本轮必要输入。
- tool_name 必须来自允许工具列表，不要编造新工具。
- 不要输出 schema 之外的字段。
- rationale_summary 只写简短可审计摘要，不要暴露隐藏推理链。

当前用户目标：
{react_user_goal}

当前轮次：
{react_round_index}/{react_max_turns}

允许工具：
{react_allowed_tools_json}

已尝试工具：
{react_attempted_tools_json}

上一轮 final_decision：
{react_latest_final_decision}

scene 策略：
{react_scene_policy_json}

resume metadata：
{react_resume_metadata_json}

历史 turn 摘要：
{react_previous_turns_json}

run observations 摘要：
{react_run_observations_json}
"""
)


class LLMReActActionSelector:
    """调用模型生成结构化 ReAct action，不直接执行工具。"""

    def __init__(
        self,
        *,
        model_client: ReActActionSelectionModel,
        model_complexity: str = "simple",
    ) -> None:
        self._model_client = model_client
        self._model_complexity = model_complexity

    def select_action(self, context: ReActActionContext) -> ReActAction:
        runnable = self._model_client.get_runnable(
            complexity=self._model_complexity,
            prompt_template=REACT_ACTION_SELECTION_PROMPT,
        )
        raw_output = self._model_client.invoke_runnable(
            runnable,
            build_selector_prompt_variables(context),
        )
        payload = _load_selector_payload(raw_output)
        try:
            action_output = LLMReActActionOutput.model_validate(payload)
        except ValidationError as exc:
            raise ReActSelectorOutputError(_selector_output_validation_message(exc)) from exc
        return _to_react_action(action_output=action_output, scene_policy=context.scene_policy)


class ReActActionValidator:
    """校验 selector 输出的顶层 action；不执行工具。"""

    def __init__(self, *, tool_executor: ToolExecutor) -> None:
        self._tool_executor = tool_executor

    def validate(
        self,
        *,
        action: ReActAction,
        run: ReActRun,
        round_index: int,
    ) -> ReActAction:
        if round_index > run.max_turns:
            raise ReActTurnBudgetError(
                f"ReAct turn budget exhausted: {round_index}/{run.max_turns}."
            )
        if action.action_type != "tool_call":
            return action
        try:
            validated_input = self._tool_executor.validate_call(
                tool_name=action.tool_name or "",
                input_payload=action.input,
            )
        except (ToolAccessValidationError, ToolInputValidationError) as exc:
            raise ReActSelectorActionValidationError(str(exc)) from exc
        return action.model_copy(update={"input": validated_input})


class ReActActionSelectionCoordinator:
    """调度 action、校验 action，并记录可审计 selector 结果。"""

    def __init__(
        self,
        *,
        action_selector: ReActActionSelector,
        action_validator: ReActActionValidator,
        scene_policy: ReActScenePolicy,
        selector_retry_budget: int,
    ) -> None:
        self._action_selector = action_selector
        self._action_validator = action_validator
        self._scene_policy = scene_policy
        self._selector_retry_budget = selector_retry_budget

    def select_next_action(self, *, run: ReActRun, context: ReActActionContext) -> ReActAction | None:
        max_attempts = self._selector_retry_budget + 1
        for selector_attempt in range(1, max_attempts + 1):
            try:
                selected = self._action_selector.select_action(context)
                action = self._coerce_action(selected)
                action = self._action_validator.validate(
                    action=action,
                    run=run,
                    round_index=context.round_index,
                )
                record_action_selection_audit(
                    run=run,
                    round_index=context.round_index,
                    selector_attempt=selector_attempt,
                    status="validated",
                    action=action,
                )
                return action
            except Exception as exc:
                error = selector_error_summary(exc)
                retryable = selector_failure_retryable(exc)
                allow_ask_user = selector_failure_can_ask_user(exc)
                record_action_selection_audit(
                    run=run,
                    round_index=context.round_index,
                    selector_attempt=selector_attempt,
                    status="failed",
                    error=error,
                )
                if retryable and selector_attempt < max_attempts:
                    continue
                return self._handle_selector_failure(
                    run=run,
                    round_index=context.round_index,
                    attempts=selector_attempt,
                    error=error,
                    allow_ask_user=allow_ask_user,
                )
        return None

    def validate_action(
        self,
        *,
        action: ReActAction,
        run: ReActRun,
        round_index: int,
    ) -> ReActAction:
        """公开动作校验边界，供 graph 节点直接复用。"""
        return self._action_validator.validate(action=action, run=run, round_index=round_index)

    def _coerce_action(self, selected: Any) -> ReActAction:
        if isinstance(selected, ReActAction):
            return selected
        try:
            return ReActAction.model_validate(selected)
        except ValidationError as exc:
            raise ReActSelectorOutputError(_selector_output_validation_message(exc)) from exc

    def _handle_selector_failure(
        self,
        *,
        run: ReActRun,
        round_index: int,
        attempts: int,
        error: str,
        allow_ask_user: bool,
    ) -> ReActAction | None:
        failure_summary = {
            "round_index": round_index,
            "attempts": attempts,
            "error": error,
            "retry_budget": self._selector_retry_budget,
        }
        run.metadata["latest_selector_failure"] = failure_summary
        if allow_ask_user and self._scene_policy.no_evidence_action == "ask_user":
            return ReActAction(
                action_type="ask_user",
                instruction="当前调度结果无效，请补充更具体的信息后继续。",
                rationale_summary="调度输出连续无效，转人工补充。",
                metadata={"selector_failure": failure_summary},
            )
        return None


def build_selector_prompt_variables(context: ReActActionContext) -> dict[str, str]:
    return {
        "user_message": context.user_goal,
        "react_user_goal": context.user_goal,
        "react_round_index": str(context.round_index),
        "react_max_turns": str(context.max_turns),
        "react_allowed_tools_json": _to_json(context.allowed_tools),
        "react_attempted_tools_json": _to_json(context.attempted_tools),
        "react_latest_final_decision": context.latest_final_decision or "none",
        "react_scene_policy_json": _to_json(public_scene_policy(context.scene_policy)),
        "react_resume_metadata_json": _to_json(context.resume_metadata),
        "react_previous_turns_json": _to_json(_summarize_turns(context.previous_turns)),
        "react_run_observations_json": _to_json(_summarize_observations(context.run_observations)),
    }


def selector_error_summary(exc: Exception) -> str:
    if isinstance(exc, ReActSelectorError):
        return str(exc)
    return f"ReAct selector failed: {exc}"


def selector_failure_retryable(exc: Exception) -> bool:
    return not isinstance(exc, ReActSelectorActionValidationError)


def selector_failure_can_ask_user(exc: Exception) -> bool:
    return not isinstance(exc, ReActSelectorActionValidationError)


def record_action_selection_audit(
    *,
    run: ReActRun,
    round_index: int,
    selector_attempt: int,
    status: Literal["validated", "failed"],
    action: ReActAction | None = None,
    error: str | None = None,
) -> None:
    audits = list(run.metadata.get("action_selection_audits") or [])
    audits.append(
        {
            "round_index": round_index,
            "selector_attempt": selector_attempt,
            "status": status,
            "action_type": action.action_type if action is not None else None,
            "tool_name": action.tool_name if action is not None else None,
            "rationale_summary": action.rationale_summary if action is not None else None,
            "error": error,
        }
    )
    run.metadata["action_selection_audits"] = audits
    run.metadata["latest_action_selection"] = {
        "round_index": round_index,
        "selector_attempt": selector_attempt,
        "status": status,
        "action_type": action.action_type if action is not None else None,
        "tool_name": action.tool_name if action is not None else None,
        "rationale_summary": action.rationale_summary if action is not None else None,
        "validation_result": "passed" if status == "validated" else "failed",
        "error": error,
    }
    run.metadata["attempted_tools"] = _attempted_tool_snapshot(run=run, action=action)


def _attempted_tool_snapshot(*, run: ReActRun, action: ReActAction | None) -> list[str]:
    tools = attempted_tools(run)
    if action is None or action.action_type != "tool_call" or not action.tool_name:
        return tools
    if action.tool_name not in tools:
        tools.append(action.tool_name)
    return tools


def _summarize_turns(turns: list[ReActTurn]) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": turn.turn_id,
            "round_index": turn.round_index,
            "action_type": turn.action.action_type,
            "tool_name": turn.tool_name,
            "status": turn.status,
            "result_summary": turn.result_summary,
            "error": turn.error,
        }
        for turn in turns
    ]


def _summarize_observations(observations: list[ToolObservation]) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": observation.tool_name,
            "success": observation.success,
            "result_summary": observation.result_summary,
            "retryable": observation.retryable,
            "requires_user": observation.requires_user,
            "final_decision": observation_final_decision(observation),
            "error": observation.error,
        }
        for observation in observations
    ]


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_selector_payload(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, Mapping):
        return dict(raw_output)
    if hasattr(raw_output, "content"):
        return _load_selector_payload(getattr(raw_output, "content"))
    text = str(raw_output or "").strip()
    if not text:
        raise ReActSelectorOutputError("selector returned empty output.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReActSelectorOutputError("selector output must be a valid JSON object.") from exc
    if not isinstance(payload, dict):
        raise ReActSelectorOutputError("selector output must be a JSON object.")
    return payload


def _to_react_action(
    *,
    action_output: LLMReActActionOutput,
    scene_policy: ReActScenePolicy,
) -> ReActAction:
    if action_output.action_type == "tool_call":
        tool_name = action_output.tool_name or ""
        merged_input = dict(scene_policy.tool_input_hints.get(tool_name) or {})
        merged_input.update(action_output.input)
        return ReActAction(
            action_type="tool_call",
            tool_name=tool_name,
            input=merged_input,
            rationale_summary=_sanitize_rationale_summary(action_output.rationale_summary),
        )
    return ReActAction(
        action_type=action_output.action_type,
        instruction=action_output.instruction,
        rationale_summary=_sanitize_rationale_summary(action_output.rationale_summary),
    )


def _sanitize_rationale_summary(value: str) -> str:
    return " ".join(value.split())[:200]


def _selector_output_validation_message(exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors():
        field = ".".join(str(item) for item in error.get("loc", ())) or "selector_output"
        messages.append(f"{field}: {error.get('msg', 'invalid value')}")
    detail = "; ".join(messages) if messages else "invalid selector output"
    return f"ReAct selector output is invalid: {detail}."
