from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.platform.agent_runtime.plan.executor import PlanExecutor
from backend.platform.agent_runtime.plan.synthesis import PlanFinalSynthesizer, StepSummarySynthesizer
from backend.platform.agent_runtime.tool_executor import ToolExecutor


@dataclass(frozen=True)
class PlanGraphDependencies:
    """Plan 图节点依赖，保持与业务实现分离。"""

    tool_executor: ToolExecutor
    final_synthesizer: Any | None = None

    def build_executor(self) -> PlanExecutor:
        """复用现有 PlanExecutor 作为图节点的业务执行器。"""
        return PlanExecutor(
            tool_executor=self.tool_executor,
            final_synthesizer=self.final_synthesizer or StepSummarySynthesizer(),
        )
