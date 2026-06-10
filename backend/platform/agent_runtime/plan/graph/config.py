from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.platform.agent_runtime.core.contracts import PlanRun
from backend.platform.agent_runtime.middleware.model_call import SharedModelCallGuard
from backend.platform.agent_runtime.plan.planner import LangChainPlanPlanner
from backend.platform.agent_runtime.plan.synthesis import PlanFinalSynthesizer, StepSummarySynthesizer
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.models.base.router import TaskComplexity


@dataclass(frozen=True)
class PlanGraphDependencies:
    """Plan 图节点依赖，保持与业务实现分离。"""

    tool_executor: ToolExecutor
    planner: LangChainPlanPlanner | None = None
    session_id: str = ""
    request_id: str = ""
    user_goal: str = ""
    mounted_knowledge_sources: Sequence[str] = ()
    candidate_tools: Sequence[str] = ()
    scene_policy: Mapping[str, Any] | None = None
    default_tool_inputs: Mapping[str, Mapping[str, Any]] | None = None
    complexity: TaskComplexity = "moderate"
    max_plan_steps: int = 8
    project_result: Callable[[PlanRun], Mapping[str, Any]] | None = None
    final_synthesizer: PlanFinalSynthesizer | None = None
    model_call_guard: SharedModelCallGuard | None = None
    checkpointer: Any | None = None

    def resolved_final_synthesizer(self) -> PlanFinalSynthesizer:
        return self.final_synthesizer or StepSummarySynthesizer()
