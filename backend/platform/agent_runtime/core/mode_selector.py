from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.platform.agent_runtime.core.contracts import AgentMode


@dataclass(frozen=True)
class ModeSelectionContext:
    """ModeSelector 可见的最小任务上下文，不依赖 application 或 scene 类型。"""

    user_message: str
    complexity: str | None = None
    mounted_knowledge_sources: Sequence[str] = ()
    scene_metadata: Mapping[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = False


@dataclass(frozen=True)
class ModeSelection:
    """记录模式选择结果，供 checkpoint、SSE 和测试审计。"""

    mode: AgentMode
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)


class ModeSelector:
    """选择顶层 Agent orchestration mode；默认保持轻量 ReAct。"""

    _EXPLICIT_PLAN_KEYWORDS = (
        "计划",
        "规划",
        "步骤",
        "分步",
        "逐步",
        "step by step",
        "step",
        "plan",
        "checklist",
        "审批",
        "确认",
    )
    _MULTI_GOAL_MARKERS = ("，然后", "然后", "再", "同时", "对比", "比较", "并且", "汇总")

    def select(
        self,
        *,
        message: str,
        complexity: str | None = None,
        mounted_knowledge_sources: Sequence[str] = (),
        scene_metadata: Mapping[str, Any] | None = None,
        requires_human_approval: bool = False,
    ) -> ModeSelection:
        context = ModeSelectionContext(
            user_message=message,
            complexity=complexity,
            mounted_knowledge_sources=tuple(mounted_knowledge_sources),
            scene_metadata=dict(scene_metadata or {}),
            requires_human_approval=requires_human_approval,
        )
        mode = self.select_mode(context)
        return ModeSelection(
            mode=mode,
            reason=self._reason_for(context=context, mode=mode),
            signals=self._signals(context),
        )

    def select_mode(self, context: ModeSelectionContext) -> AgentMode:
        """只返回 mode 的轻量入口，便于平台内部单元测试。"""
        message = context.user_message.strip().lower()
        if not message:
            return "react"
        policy_default = str(context.scene_metadata.get("default_agent_mode") or "").strip()
        if policy_default in {"react", "plan"}:
            return policy_default  # type: ignore[return-value]
        if context.requires_human_approval:
            return "plan"
        if context.complexity == "complex":
            return "plan"
        if len(set(context.mounted_knowledge_sources)) > 1:
            return "plan"
        if any(keyword in message for keyword in self._EXPLICIT_PLAN_KEYWORDS):
            return "plan"
        if sum(1 for marker in self._MULTI_GOAL_MARKERS if marker in message) >= 2:
            return "plan"
        return "react"

    def _reason_for(self, *, context: ModeSelectionContext, mode: AgentMode) -> str:
        if context.scene_metadata.get("default_agent_mode") in {"react", "plan"}:
            return "scene_policy"
        if mode == "plan":
            return "complex_or_explicit_plan"
        return "default_simple_react"

    def _signals(self, context: ModeSelectionContext) -> dict[str, Any]:
        message = context.user_message.strip().lower()
        return {
            "complexity": context.complexity,
            "mounted_knowledge_sources": list(context.mounted_knowledge_sources),
            "requires_human_approval": context.requires_human_approval,
            "keyword_hits": [
                keyword
                for keyword in self._EXPLICIT_PLAN_KEYWORDS
                if keyword in message
            ],
            "multi_goal_marker_count": sum(
                1 for marker in self._MULTI_GOAL_MARKERS if marker in message
            ),
        }


MinimalModeSelector = ModeSelector
