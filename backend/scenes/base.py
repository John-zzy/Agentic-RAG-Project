from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from backend.platform.models.base.router import TaskComplexity
from backend.platform.rag.contracts import RecallStrategy


@dataclass(frozen=True)
class SceneBootstrapResult:
    """描述场景启动预热结果，供运行时汇总展示。"""

    metrics: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneFallbackPolicy:
    """描述场景在未命中知识时的兜底回复策略。"""

    no_hit_message: str
    neutral_no_hit_message: str | None = None

    def message_for_strategy(self, strategy: str) -> str:
        """按 no-hit 策略返回不携带引用的兜底文案。"""
        if strategy == "fallback_answer" and self.neutral_no_hit_message:
            return self.neutral_no_hit_message
        return self.no_hit_message


NoHitStrategy = Literal["ask_user", "fallback_answer"]


@dataclass(frozen=True)
class SceneRetrievalPolicy:
    """描述场景级检索策略，作为 runtime 与 RAG 能力之间的配置边界。"""

    # 默认召回条数上限。
    top_k: int = 5
    # 最低相关性阈值；None 表示沿用检索服务默认判断。
    min_relevance_score: float | None = None
    # 召回策略名称，例如 semantic、keyword、hybrid。
    recall_strategy: RecallStrategy = "hybrid"
    # 无命中后的处理策略，例如 ask_user、fallback_answer。
    no_hit_strategy: NoHitStrategy = "ask_user"
    # 是否启用 ReRank；当前仅作为 scene 级接入位。
    rerank_enabled: bool = False
    # ReRank 后保留条数；None 表示不覆盖召回条数。
    rerank_top_n: int | None = None

    def __post_init__(self) -> None:
        allowed_recall_strategies = ("semantic", "keyword", "hybrid")
        if self.recall_strategy not in allowed_recall_strategies:
            raise ValueError(
                "Unsupported scene retrieval recall_strategy "
                f"{self.recall_strategy!r}; expected one of {allowed_recall_strategies}."
            )
        allowed_no_hit_strategies = ("ask_user", "fallback_answer")
        if self.no_hit_strategy not in allowed_no_hit_strategies:
            raise ValueError(
                "Unsupported scene retrieval no_hit_strategy "
                f"{self.no_hit_strategy!r}; expected one of {allowed_no_hit_strategies}."
            )
        if self.rerank_top_n is not None and self.rerank_top_n <= 0:
            raise ValueError("Scene retrieval rerank_top_n must be None or a positive integer.")


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
