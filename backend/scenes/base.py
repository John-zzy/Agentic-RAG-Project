from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from backend.platform.models.base.router import TaskComplexity


@dataclass(frozen=True)
class SceneBootstrapResult:
    """描述场景启动预热结果，供运行时汇总展示。"""

    metrics: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneFallbackPolicy:
    """描述场景在未命中知识时的兜底回复策略。"""

    no_hit_message: str


@dataclass(frozen=True)
class SceneDefinition:
    """描述场景可挂载到 runtime 的最小装配协议。"""

    scene: str
    name: str
    description: str
    build_retriever: Callable[[], BaseRetriever]
    build_tools: Callable[[], tuple[BaseTool, ...]]
    system_prompt: str
    fallback_policy: SceneFallbackPolicy
    infer_complexity: Callable[[str], TaskComplexity]
    bootstrap: Callable[[], SceneBootstrapResult] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
