from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from backend.platform.agent_runtime.contracts import (
    AgentRuntimeModel,
    PlanRun,
    PlanStep,
    ReActRun,
    ToolObservation,
)


class CorrectionAction(StrEnum):
    """self-check 失败后的受控修正动作。"""

    NONE = "none"
    RETRY_TOOL = "retry_tool"
    REWRITE_QUERY = "rewrite_query"
    ASK_USER = "ask_user"
    REVISE_ANSWER = "revise_answer"
    FAIL_FINAL = "fail_final"


class ResultValidationReport(AgentRuntimeModel):
    """结果自检报告，可直接写入 run metadata 或 checkpoint metadata。"""

    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    correction_action: CorrectionAction = CorrectionAction.NONE
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """输出 checkpoint 友好的 JSON payload。"""

        return self.model_dump(mode="json")


class ResultValidationContext(AgentRuntimeModel):
    """规则校验只依赖运行事实，不依赖 application 或 scene。"""

    answer_mode: str | None = None
    final_decision: str | None = None
    status: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    react_run: ReActRun | None = None
    plan_run: PlanRun | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultValidationIssue(AgentRuntimeModel):
    """单条规则命中结果，由汇总器合成最终 correction action。"""

    rule_id: str
    category: str
    reason: str
    correction_action: CorrectionAction
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultValidationRule(Protocol):
    """规则协议：每条规则只处理一种失败语义。"""

    rule_id: str

    def validate(self, context: ResultValidationContext) -> ResultValidationIssue | None:
        """Return an issue when the context violates this rule."""


class EvidenceCitationRule:
    """证据回答必须带有可采用 citations。"""

    rule_id = "evidence_answer_requires_citations"

    def validate(self, context: ResultValidationContext) -> ResultValidationIssue | None:
        if not _is_evidence_answer(context):
            return None
        if context.citations:
            return None
        return ResultValidationIssue(
            rule_id=self.rule_id,
            category="missing_citations",
            reason="Evidence answer cannot be returned without citations.",
            correction_action=CorrectionAction.FAIL_FINAL,
        )


class PlanRequiredStepRule:
    """Plan 必需步骤失败、阻塞或取消时不能合成为成功。"""

    rule_id = "failed_plan_step_blocks_success"
    _blocked_statuses = {"failed", "blocked", "cancelled"}

    def validate(self, context: ResultValidationContext) -> ResultValidationIssue | None:
        plan_run = context.plan_run
        if plan_run is None:
            return None
        blocked_steps = [
            step for step in plan_run.steps
            if _is_required_plan_step(step) and step.status in self._blocked_statuses
        ]
        if not blocked_steps:
            return None
        return ResultValidationIssue(
            rule_id=self.rule_id,
            category="plan_step_not_successful",
            reason="Plan run cannot succeed while required steps are failed, blocked or cancelled.",
            correction_action=CorrectionAction.FAIL_FINAL,
            metadata={"step_ids": [step.step_id for step in blocked_steps]},
        )


class SuccessfulObservationRule:
    """证据回答必须来自至少一次成功 observation。"""

    rule_id = "evidence_answer_requires_successful_observation"

    def validate(self, context: ResultValidationContext) -> ResultValidationIssue | None:
        if not _is_evidence_answer(context):
            return None
        run = context.react_run or context.plan_run
        if run is None:
            return None
        if any(observation.success for observation in run.observations):
            return None
        return ResultValidationIssue(
            rule_id=self.rule_id,
            category="missing_successful_observation",
            reason="Evidence answer cannot be returned without a successful observation.",
            correction_action=CorrectionAction.FAIL_FINAL,
            metadata={"run_id": _run_id(run)},
        )


class RequiresUserStateRule:
    """requires_user observation 必须把 run 推进到 waiting_user。"""

    rule_id = "requires_user_must_wait_user"

    def validate(self, context: ResultValidationContext) -> ResultValidationIssue | None:
        if context.status == "waiting_user":
            return None
        run = context.react_run or context.plan_run
        if run is None:
            return None
        observation = _first_requires_user_observation(run.observations)
        if observation is None:
            return None
        if _is_follow_up_without_hitl_wait(context):
            return None
        action = (
            CorrectionAction.ASK_USER
            if _observation_user_prompt(observation)
            else CorrectionAction.FAIL_FINAL
        )
        return ResultValidationIssue(
            rule_id=self.rule_id,
            category="requires_user_not_waiting",
            reason="Observation requires user input but runtime is not waiting_user.",
            correction_action=action,
            metadata={
                "run_id": _run_id(run),
                "tool_name": observation.tool_name,
                "user_prompt": _observation_user_prompt(observation),
            },
        )


class ResultValidator:
    """规则型结果自检器，后续可被 LangChain after-agent middleware 复用。"""

    def __init__(self, rules: Sequence[ResultValidationRule] | None = None) -> None:
        self._rules = tuple(rules or _default_rules())

    def validate(self, context: ResultValidationContext) -> ResultValidationReport:
        issues = [
            issue
            for rule in self._rules
            if (issue := rule.validate(context)) is not None
        ]
        if not issues:
            return ResultValidationReport(
                passed=True,
                metadata={
                    "rule_ids": [rule.rule_id for rule in self._rules],
                    "correction_round": 0,
                    "max_self_check_rounds": 1,
                },
            )
        action = _select_correction_action(issues)
        return ResultValidationReport(
            passed=False,
            failure_reasons=[issue.reason for issue in issues],
            categories=_unique(issue.category for issue in issues),
            correction_action=action,
            metadata={
                "rule_ids": [issue.rule_id for issue in issues],
                "issues": [issue.model_dump(mode="json") for issue in issues],
                "correction_round": 0,
                "max_self_check_rounds": 1,
            },
        )


def build_result_validation_context(
    *,
    answer_mode: str | None,
    final_decision: str | None,
    status: str | None,
    citations: Sequence[Mapping[str, Any]] | None = None,
    react_run: ReActRun | Mapping[str, Any] | None = None,
    plan_run: PlanRun | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ResultValidationContext:
    """从 graph state/run payload 构造自检上下文。"""

    return ResultValidationContext(
        answer_mode=answer_mode,
        final_decision=final_decision,
        status=status,
        citations=[dict(citation) for citation in citations or ()],
        react_run=_coerce_react_run(react_run),
        plan_run=_coerce_plan_run(plan_run),
        metadata=dict(metadata or {}),
    )


def _default_rules() -> tuple[ResultValidationRule, ...]:
    return (
        EvidenceCitationRule(),
        PlanRequiredStepRule(),
        SuccessfulObservationRule(),
        RequiresUserStateRule(),
    )


def _is_evidence_answer(context: ResultValidationContext) -> bool:
    return (
        context.answer_mode == "evidence_answer"
        or context.final_decision == "answer_with_evidence"
    )


def _is_follow_up_without_hitl_wait(context: ResultValidationContext) -> bool:
    # 普通检索无命中的追问不进入 HITL 等待；self-check 只拦截异常的 requires_user 状态投影。
    if context.metadata.get("hitl_wait_enabled") is not False:
        return False
    return (
        context.answer_mode == "follow_up"
        and context.final_decision == "ask_user"
    )


def _is_required_plan_step(step: PlanStep) -> bool:
    return step.status != "skipped"


def _first_requires_user_observation(
    observations: Sequence[ToolObservation],
) -> ToolObservation | None:
    for observation in observations:
        if observation.requires_user:
            return observation
    return None


def _observation_user_prompt(observation: ToolObservation) -> str | None:
    if observation.user_prompt and observation.user_prompt.strip():
        return observation.user_prompt.strip()
    if observation.result_summary and observation.result_summary.strip():
        return observation.result_summary.strip()
    return None


def _select_correction_action(
    issues: Sequence[ResultValidationIssue],
) -> CorrectionAction:
    priority = (
        CorrectionAction.ASK_USER,
        CorrectionAction.FAIL_FINAL,
        CorrectionAction.RETRY_TOOL,
        CorrectionAction.REWRITE_QUERY,
        CorrectionAction.REVISE_ANSWER,
    )
    actions = {issue.correction_action for issue in issues}
    for action in priority:
        if action in actions:
            return action
    return CorrectionAction.NONE


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _run_id(run: ReActRun | PlanRun) -> str:
    if isinstance(run, ReActRun):
        return run.react_run_id
    return run.plan_run_id


def _coerce_react_run(value: ReActRun | Mapping[str, Any] | None) -> ReActRun | None:
    if value is None or isinstance(value, ReActRun):
        return value
    return ReActRun.model_validate(dict(value))


def _coerce_plan_run(value: PlanRun | Mapping[str, Any] | None) -> PlanRun | None:
    if value is None or isinstance(value, PlanRun):
        return value
    return PlanRun.model_validate(dict(value))
