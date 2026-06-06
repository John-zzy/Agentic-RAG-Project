from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from backend.platform.agent_runtime.contracts import PlanRun
from backend.platform.agent_runtime.plan.executor import PlanExecutor
from backend.platform.agent_runtime.plan.planner import MinimalPlanner
from backend.platform.agent_runtime.plan.synthesis import PlanFinalSynthesizer, StepSummarySynthesizer
from backend.platform.agent_runtime.tool_executor import ToolExecutor


@dataclass(frozen=True)
class PlanGraphDependencies:
    """Plan 图节点依赖，保持与业务实现分离。"""

    tool_executor: ToolExecutor
    session_id: str = ""
    request_id: str = ""
    user_goal: str = ""
    mounted_knowledge_sources: Sequence[str] = ()
    candidate_tools: Sequence[str] = ()
    scene_policy: Mapping[str, Any] | None = None
    planner: MinimalPlanner | None = None
    project_result: Callable[[PlanRun], Mapping[str, Any]] | None = None
    final_synthesizer: Any | None = None

    def build_planner(self) -> MinimalPlanner:
        """构建或复用 Planner，让 create_plan 节点拥有计划创建职责。"""
        return self.planner or MinimalPlanner(tool_executor=self.tool_executor)

    def build_executor(self) -> PlanExecutor:
        """复用现有 PlanExecutor 作为图节点的业务执行器。"""
        return PlanExecutor(
            tool_executor=self.tool_executor,
            final_synthesizer=self.final_synthesizer or StepSummarySynthesizer(),
        )
