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
class SceneRetrievalPolicy:
    """描述场景级检索策略，作为 runtime 与 RAG 能力之间的配置边界。"""

    # 默认召回条数上限。
    top_k: int = 5
    # 最低相关性阈值；None 表示沿用检索服务默认判断。
    min_relevance_score: float | None = None
    # 召回策略名称，例如 semantic、keyword、hybrid。
    recall_strategy: str = "hybrid"
    # 无命中后的处理策略，例如 ask_user、fallback_answer。
    no_hit_strategy: str = "ask_user"
    # 是否启用 ReRank；当前仅作为 scene 级接入位。
    rerank_enabled: bool = False
    # ReRank 后保留条数；None 表示不覆盖召回条数。
    rerank_top_n: int | None = None


@dataclass(frozen=True)
class SceneDefinition:
    """描述场景可挂载到 runtime 的最小装配协议。"""

    scene: str
    name: str
    description: str
    build_retriever: Callable[[], BaseRetriever]
    build_tools: Callable[[], tuple[BaseTool, ...]]
    candidate_retrieval_tools_resolver: Callable[[tuple[str, ...]], tuple[str, ...]]
    system_prompt: str
    fallback_policy: SceneFallbackPolicy
    infer_complexity: Callable[[str], TaskComplexity]
    retrieval_policy: SceneRetrievalPolicy = field(default_factory=SceneRetrievalPolicy)
    bootstrap: Callable[[], SceneBootstrapResult] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_candidate_retrieval_tools(
        self,
        mounted_knowledge_sources: tuple[str, ...],
    ) -> tuple[str, ...]:
        """根据当前会话挂载知识源解析可用的候选检索工具。"""
        return self.candidate_retrieval_tools_resolver(mounted_knowledge_sources)
