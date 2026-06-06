from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Any

from backend.platform.agent_runtime.contracts import ReActRun
from backend.platform.agent_runtime.react.policy import ReActScenePolicy
from backend.platform.agent_runtime.react.runtime import ReActRuntime
from backend.platform.agent_runtime.react.selector import ReActActionSelector
from backend.platform.agent_runtime.react.synthesis import ObservationSummarySynthesizer
from backend.platform.agent_runtime.tool_executor import ToolExecutor


@dataclass(frozen=True)
class ReActGraphDependencies:
    """ReAct 图节点依赖，保持与业务实现分离。"""

    tool_executor: ToolExecutor
    action_selector: ReActActionSelector
    session_id: str = ""
    request_id: str = ""
    user_goal: str = ""
    react_run_id: str | None = None
    initial_run: ReActRun | None = None
    project_result: Callable[[ReActRun], Mapping[str, Any]] | None = None
    final_synthesizer: Any | None = None
    max_turns: int = 5
    scene_policy: ReActScenePolicy | None = None
    selector_retry_budget: int = 1
    turn_id_factory: Callable[[int], str] | None = None

    def build_runtime(self) -> ReActRuntime:
        """复用现有 ReActRuntime 作为图节点的业务执行器。"""
        return ReActRuntime(
            tool_executor=self.tool_executor,
            action_selector=self.action_selector,
            final_synthesizer=self.final_synthesizer or ObservationSummarySynthesizer(),
            max_turns=self.max_turns,
            scene_policy=self.scene_policy,
            selector_retry_budget=self.selector_retry_budget,
            turn_id_factory=self.turn_id_factory,
        )
