from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field

from backend.platform.agent_runtime.core.contracts import AgentRuntimeModel, PlanRun, PlanStep
from backend.platform.agent_runtime.core.validation import (
    validate_plan_dependencies,
    validate_plan_tool_allowlist,
)
from backend.platform.agent_runtime.middleware.model_call import (
    SharedModelCallGuard,
    default_model_call_context,
)
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.models.base.router import TaskComplexity
from backend.platform.workflow.state_machine import validate_transition


class PlanStepDraft(AgentRuntimeModel):
    """LLM planner 只负责生成草案；真正执行前还会走平台校验。"""

    step_id: str = ""
    goal: str
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class PlanDraft(AgentRuntimeModel):
    """LangChain structured output 的顶层返回结构。"""

    steps: list[PlanStepDraft] = Field(default_factory=list)
    rationale_summary: str = ""


class PlanPlannerContext(AgentRuntimeModel):
    """LLM 创建计划时能看到的中立上下文。"""

    session_id: str
    request_id: str
    user_goal: str
    mounted_knowledge_sources: list[str] = Field(default_factory=list)
    candidate_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    scene_policy: dict[str, Any] = Field(default_factory=dict)
    default_tool_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    complexity: str = "moderate"
    max_plan_steps: int = 8


@dataclass(frozen=True)
class LangChainPlanPlanner:
    """用 LangChain structured output 让 LLM 生成 PlanRun。"""

    model_provider: Callable[[TaskComplexity], BaseChatModel]
    tool_executor: ToolExecutor
    model_call_guard: SharedModelCallGuard | None = None
    plan_run_id_factory: Callable[[], str] | None = None
    step_id_factory: Callable[[int], str] | None = None
    system_prompt: str = (
        "You are a planning node in a workflow. Create a concise executable "
        "plan using only the allowed tools. Each step must include one tool, "
        "valid tool input, and dependencies only on earlier step ids."
    )

    def create_plan(
        self,
        *,
        session_id: str,
        request_id: str,
        user_goal: str,
        mounted_knowledge_sources: Sequence[str] = (),
        candidate_tools: Sequence[str] = (),
        scene_policy: Mapping[str, Any] | None = None,
        default_tool_inputs: Mapping[str, Mapping[str, Any]] | None = None,
        complexity: TaskComplexity = "moderate",
        max_plan_steps: int = 8,
        plan_run_id: str | None = None,
    ) -> PlanRun:
        context = PlanPlannerContext(
            session_id=session_id,
            request_id=request_id,
            user_goal=user_goal,
            mounted_knowledge_sources=list(mounted_knowledge_sources),
            candidate_tools=list(candidate_tools),
            allowed_tools=sorted(self.tool_executor.allowed_tools),
            scene_policy=dict(scene_policy or {}),
            default_tool_inputs={
                str(tool_name): dict(tool_input)
                for tool_name, tool_input in dict(default_tool_inputs or {}).items()
            },
            complexity=str(complexity),
            max_plan_steps=max_plan_steps,
        )
        draft = self._invoke_planner(context)
        steps = self._coerce_steps(draft.steps)
        self._validate_steps(steps=steps, max_plan_steps=max_plan_steps)
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
                    "name": "langchain_plan_planner",
                    "step_source": "llm_structured_output",
                    "rationale_summary": draft.rationale_summary,
                    "context_summary": context_summary,
                    "mounted_knowledge_sources": list(mounted_knowledge_sources),
                    "candidate_tools": list(candidate_tools),
                    "allowed_tools": sorted(self.tool_executor.allowed_tools),
                    "selected_tools": [step.tool_name for step in steps],
                    "max_plan_steps": max_plan_steps,
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

    def _invoke_planner(self, context: PlanPlannerContext) -> PlanDraft:
        operation = lambda: self._structured_model(context).invoke(
            _planner_messages(context=context, system_prompt=self.system_prompt)
        )
        if self.model_call_guard is None:
            result = operation()
        else:
            runtime_context = default_model_call_context(
                session_id=context.session_id,
                request_id=context.request_id,
                scene=str(context.scene_policy.get("scene") or "platform.plan"),
                mounted_knowledge_sources=tuple(context.mounted_knowledge_sources),
                complexity=context.complexity,
                workflow_metadata={
                    "run_id": None,
                    "checkpoint_ns": "plan_graph",
                    "metadata": {"node": "create_plan"},
                },
                request_metadata={
                    "operation": "plan.create_plan",
                    "candidate_tools": context.candidate_tools,
                    "allowed_tools": context.allowed_tools,
                },
            )
            result = self.model_call_guard.invoke(
                operation,
                context=runtime_context,
                metadata={"operation": "plan.create_plan"},
                output_type=PlanDraft,
            )
        if isinstance(result, PlanDraft):
            return result
        if isinstance(result, Mapping):
            return PlanDraft.model_validate(dict(result))
        raise TypeError("LLM planner returned an unexpected structured output.")

    def _structured_model(self, context: PlanPlannerContext) -> Any:
        model = self.model_provider(context.complexity)  # type: ignore[arg-type]
        return model.with_structured_output(PlanDraft)

    def _coerce_steps(self, drafts: Sequence[PlanStepDraft | Mapping[str, Any]]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for index, draft in enumerate(drafts, start=1):
            item = draft if isinstance(draft, PlanStepDraft) else PlanStepDraft.model_validate(draft)
            step_id = item.step_id.strip() or self._new_step_id(index)
            steps.append(
                PlanStep(
                    step_id=step_id,
                    goal=item.goal,
                    tool_name=item.tool_name,
                    input=dict(item.input),
                    depends_on=list(item.depends_on),
                )
            )
        return steps

    def _validate_steps(self, *, steps: list[PlanStep], max_plan_steps: int) -> None:
        if not steps:
            raise ValueError("LLM planner must create at least one plan step.")
        if len(steps) > max_plan_steps:
            raise ValueError(
                f"LLM planner created {len(steps)} steps; max_plan_steps is {max_plan_steps}."
            )
        validate_plan_tool_allowlist(steps, self.tool_executor.allowed_tools)
        validate_plan_dependencies(steps)
        for index, step in enumerate(steps):
            validated_input = self.tool_executor.validate_call(
                tool_name=step.tool_name,
                input_payload=step.input,
            )
            steps[index] = step.model_copy(update={"input": validated_input})

    def _new_plan_run_id(self) -> str:
        if self.plan_run_id_factory is not None:
            return self.plan_run_id_factory()
        return str(uuid4())

    def _new_step_id(self, index: int) -> str:
        if self.step_id_factory is not None:
            return self.step_id_factory(index)
        return f"step-{index}"


def _planner_messages(*, context: PlanPlannerContext, system_prompt: str) -> list[dict[str, str]]:
    tools = [
        {
            "name": tool_name,
            "default_input": context.default_tool_inputs.get(tool_name, {}),
        }
        for tool_name in context.allowed_tools
    ]
    user_prompt = (
        f"User goal: {context.user_goal}\n"
        f"Mounted knowledge sources: {context.mounted_knowledge_sources}\n"
        f"Candidate tools: {context.candidate_tools}\n"
        f"Allowed tools and default inputs: {tools}\n"
        f"Scene policy: {context.scene_policy}\n"
        f"Maximum steps: {context.max_plan_steps}\n"
        "Return only the structured plan."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_context_summary(*, context: PlanPlannerContext, steps: Sequence[PlanStep]) -> str:
    tools = ", ".join(step.tool_name for step in steps)
    sources = ", ".join(context.mounted_knowledge_sources) or "none"
    return (
        f"Plan for goal={context.user_goal!r}; "
        f"mounted_sources={sources}; "
        f"steps={len(steps)}; "
        f"tools={tools}."
    )
