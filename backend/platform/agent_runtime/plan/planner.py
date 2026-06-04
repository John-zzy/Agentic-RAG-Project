from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field

from backend.platform.agent_runtime.contracts import AgentRuntimeModel, PlanRun, PlanStep
from backend.platform.agent_runtime.rag_tools import (
    AGENTIC_RAG_TOOL_NAME,
    NATIVE_RAG_TOOL_NAME,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.agent_runtime.validation import (
    validate_plan_dependencies,
    validate_plan_tool_allowlist,
)
from backend.platform.workflow.state_machine import validate_transition


class PlannerContext(AgentRuntimeModel):
    """Planner 可见的最小上下文，不绑定 application 或 scene 具体类型。"""

    session_id: str
    request_id: str
    user_goal: str
    mounted_knowledge_sources: list[str] = Field(default_factory=list)
    candidate_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    scene_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStepSelector(Protocol):
    """可替换的步骤选择协议；默认 Planner 仍负责统一校验。"""

    def select_steps(self, context: PlannerContext) -> Sequence[PlanStep | Mapping[str, Any]]:
        """Return proposed steps for a plan context."""


class MinimalPlanStepSelector:
    """根据 policy preferred tools 生成最小步骤草案。"""

    def select_steps(self, context: PlannerContext) -> Sequence[Mapping[str, Any]]:
        tools = _extract_policy_tools(context.scene_policy)
        allowed = set(context.allowed_tools)
        selected = [tool_name for tool_name in tools if tool_name in allowed][:3]
        steps: list[dict[str, Any]] = []
        for index, tool_name in enumerate(selected, start=1):
            steps.append(
                {
                    "step_id": f"step-{index}",
                    "goal": context.user_goal if index == 1 else f"Continue: {context.user_goal}",
                    "tool_name": tool_name,
                    "input": {"query": context.user_goal},
                    "depends_on": [] if index == 1 else [f"step-{index - 1}"],
                }
            )
        return steps


@dataclass(frozen=True)
class _PlanDraft:
    steps: list[PlanStep]
    source: str


class MinimalPlanner:
    """生成 1-3 个可执行 PlanStep，并在返回前完成平台层校验。"""

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor,
        max_steps: int = 3,
        plan_run_id_factory: Callable[[], str] | None = None,
        step_id_factory: Callable[[int], str] | None = None,
    ) -> None:
        if max_steps < 1 or max_steps > 3:
            raise ValueError("max_steps must be between 1 and 3.")
        self._tool_executor = tool_executor
        self._max_steps = max_steps
        self._plan_run_id_factory = plan_run_id_factory
        self._step_id_factory = step_id_factory

    def create_plan(
        self,
        *,
        session_id: str,
        request_id: str,
        user_goal: str,
        mounted_knowledge_sources: Sequence[str] = (),
        candidate_tools: Sequence[str] = (),
        scene_policy: Mapping[str, Any] | Any | None = None,
        plan_run_id: str | None = None,
        proposed_steps: Sequence[PlanStep | Mapping[str, Any]] | None = None,
    ) -> PlanRun:
        context = PlannerContext(
            session_id=session_id,
            request_id=request_id,
            user_goal=user_goal,
            mounted_knowledge_sources=list(mounted_knowledge_sources),
            candidate_tools=list(candidate_tools),
            allowed_tools=sorted(self._tool_executor.allowed_tools),
            scene_policy=_policy_to_mapping(scene_policy),
        )
        draft = self._build_steps(context=context, proposed_steps=proposed_steps)
        steps = draft.steps
        self._validate_steps(steps)
        workflow_status = validate_transition("created", "plan_start")
        context_summary = _build_context_summary(context=context, steps=steps)
        return PlanRun(
            plan_run_id=plan_run_id or self._new_plan_run_id(),
            session_id=session_id,
            request_id=request_id,
            user_goal=user_goal,
            context_summary=context_summary,
            workflow_status=workflow_status,
            steps=steps,
            metadata={
                "planner": {
                    "name": "minimal_planner",
                    "step_source": draft.source,
                    "context_summary": context_summary,
                    "mounted_knowledge_sources": list(mounted_knowledge_sources),
                    "candidate_tools": list(candidate_tools),
                    "allowed_tools": sorted(self._tool_executor.allowed_tools),
                    "selected_tools": [step.tool_name for step in steps],
                    "workflow_transitions": [
                        {
                            "from": "created",
                            "event": "plan_start",
                            "to": workflow_status,
                        }
                    ],
                }
            },
        )

    def _build_steps(
        self,
        *,
        context: PlannerContext,
        proposed_steps: Sequence[PlanStep | Mapping[str, Any]] | None,
    ) -> _PlanDraft:
        if proposed_steps is not None:
            return _PlanDraft(
                steps=self._coerce_steps(proposed_steps),
                source="proposed_steps",
            )

        policy_steps = _extract_policy_steps(context.scene_policy)
        if policy_steps:
            return _PlanDraft(
                steps=self._coerce_steps(policy_steps),
                source="scene_policy.plan_steps",
            )

        selected_tools = self._select_policy_tools(context)
        source = "scene_policy.plan_tools"
        if not selected_tools:
            selected_tools = self._select_candidate_tools(context)
            source = "candidate_tools"
        if not selected_tools:
            selected_tools = [self._select_default_tool(context)]
            source = "default_tool"

        steps: list[PlanStep] = []
        for index, tool_name in enumerate(selected_tools[: self._max_steps], start=1):
            dependency = [] if index == 1 else [steps[-1].step_id]
            steps.append(
                PlanStep(
                    step_id=self._new_step_id(index),
                    goal=context.user_goal if index == 1 else f"Continue: {context.user_goal}",
                    tool_name=tool_name,
                    input=_input_for_tool(context=context, tool_name=tool_name),
                    depends_on=dependency,
                )
            )
        return _PlanDraft(steps=steps, source=source)

    def _coerce_steps(self, proposed_steps: Sequence[PlanStep | Mapping[str, Any]]) -> list[PlanStep]:
        if not proposed_steps:
            raise ValueError("Planner must create at least one plan step.")
        if len(proposed_steps) > self._max_steps:
            raise ValueError(f"Planner supports at most {self._max_steps} plan steps.")
        steps: list[PlanStep] = []
        for index, proposed_step in enumerate(proposed_steps, start=1):
            step = (
                proposed_step
                if isinstance(proposed_step, PlanStep)
                else PlanStep.model_validate(proposed_step)
            )
            if not step.step_id:
                step = step.model_copy(update={"step_id": self._new_step_id(index)})
            steps.append(step)
        return steps

    def _validate_steps(self, steps: list[PlanStep]) -> None:
        if not steps:
            raise ValueError("Planner must create at least one plan step.")
        if len(steps) > self._max_steps:
            raise ValueError(f"Planner supports at most {self._max_steps} plan steps.")

        validate_plan_tool_allowlist(steps, self._tool_executor.allowed_tools)
        validate_plan_dependencies(steps)
        for index, step in enumerate(steps):
            validated_input = self._tool_executor.validate_call(
                tool_name=step.tool_name,
                input_payload=step.input,
            )
            steps[index] = step.model_copy(update={"input": validated_input})

    def _select_policy_tools(self, context: PlannerContext) -> list[str]:
        tools = _extract_policy_tools(context.scene_policy)
        allowed = self._tool_executor.allowed_tools
        # 首选工具列表属于策略建议；不可用项跳过，显式 step 计划则会被严格拒绝。
        return [tool_name for tool_name in tools if tool_name in allowed][: self._max_steps]

    def _select_candidate_tools(self, context: PlannerContext) -> list[str]:
        allowed = self._tool_executor.allowed_tools
        candidates = _deduplicate(context.candidate_tools or _extract_candidate_tools(context.scene_policy))
        return [tool_name for tool_name in candidates if tool_name in allowed][: self._max_steps]

    def _select_default_tool(self, context: PlannerContext) -> str:
        allowed = self._tool_executor.allowed_tools
        if not allowed:
            raise ValueError("Planner cannot create a plan without allowed tools.")

        if "documents" in set(context.mounted_knowledge_sources):
            for rag_tool_name in (AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME):
                if rag_tool_name in allowed:
                    return rag_tool_name

        for tool_name in sorted(allowed):
            if "rag" in tool_name or "search" in tool_name:
                return tool_name
        return sorted(allowed)[0]

    def _new_plan_run_id(self) -> str:
        if self._plan_run_id_factory is not None:
            return self._plan_run_id_factory()
        return str(uuid4())

    def _new_step_id(self, index: int) -> str:
        if self._step_id_factory is not None:
            return self._step_id_factory(index)
        return f"step-{index}"


def _policy_to_mapping(scene_policy: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if scene_policy is None:
        return {}
    if isinstance(scene_policy, Mapping):
        return dict(scene_policy)
    policy: dict[str, Any] = {}
    for attribute in (
        "plan_steps",
        "preferred_plan_tools",
        "default_plan_tools",
        "plan_tools",
        "candidate_tools",
        "plan_tool_inputs",
        "tool_input",
    ):
        value = getattr(scene_policy, attribute, None)
        if value is not None:
            policy[attribute] = value
    return policy


def _extract_policy_steps(
    scene_policy: Mapping[str, Any],
) -> list[PlanStep | Mapping[str, Any]] | None:
    value = scene_policy.get("plan_steps")
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("scene_policy.plan_steps must be a sequence.")
    steps: list[PlanStep | Mapping[str, Any]] = []
    for step in value:
        if isinstance(step, PlanStep):
            steps.append(step)
        elif isinstance(step, Mapping):
            steps.append(dict(step))
        else:
            raise ValueError("scene_policy.plan_steps entries must be mappings or PlanStep.")
    return steps


def _extract_policy_tools(scene_policy: Mapping[str, Any]) -> list[str]:
    for key in ("preferred_plan_tools", "default_plan_tools", "plan_tools"):
        value = scene_policy.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [str(tool_name) for tool_name in value]
        raise ValueError(f"scene_policy.{key} must be a tool name or sequence.")
    return []


def _extract_candidate_tools(scene_policy: Mapping[str, Any]) -> list[str]:
    value = scene_policy.get("candidate_tools")
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(tool_name) for tool_name in value]
    raise ValueError("scene_policy.candidate_tools must be a tool name or sequence.")


def _input_for_tool(*, context: PlannerContext, tool_name: str) -> dict[str, Any]:
    tool_input = _tool_specific_input(context.scene_policy, tool_name)
    tool_input.setdefault("query", context.user_goal)
    return tool_input


def _tool_specific_input(scene_policy: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    tool_inputs = scene_policy.get("plan_tool_inputs")
    if isinstance(tool_inputs, Mapping):
        value = tool_inputs.get(tool_name)
        if isinstance(value, Mapping):
            return dict(value)
    value = scene_policy.get("tool_input")
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _build_context_summary(*, context: PlannerContext, steps: Sequence[PlanStep]) -> str:
    tools = ", ".join(step.tool_name for step in steps)
    sources = ", ".join(context.mounted_knowledge_sources) or "none"
    return (
        f"Plan for goal={context.user_goal!r}; "
        f"mounted_sources={sources}; "
        f"steps={len(steps)}; "
        f"tools={tools}."
    )


def _deduplicate(values: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        resolved.append(item)
    return resolved


PlanContext = PlannerContext
Planner = MinimalPlanner
