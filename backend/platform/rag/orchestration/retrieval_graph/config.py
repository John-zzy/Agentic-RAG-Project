from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig


AGENTIC_RAG_CONFIG_KEY = "agentic_rag"


@dataclass(frozen=True)
class AgenticRagGraphDependencies:
    """Agentic RAG 图运行时需要的最小依赖集合。"""

    retriever: Any
    run_manager: CallbackManagerForRetrieverRun | None = None


def build_agentic_rag_graph_config(
    *,
    retriever: Any,
    run_manager: CallbackManagerForRetrieverRun | None = None,
) -> RunnableConfig:
    """把依赖放进 LangGraph 的 configurable 区域，供节点按需解析。"""
    return {
        "configurable": {
            AGENTIC_RAG_CONFIG_KEY: AgenticRagGraphDependencies(
                retriever=retriever,
                run_manager=run_manager,
            )
        }
    }


def resolve_agentic_rag_graph_dependencies(
    config: RunnableConfig | dict[str, Any] | None,
) -> AgenticRagGraphDependencies:
    """从 LangGraph config 中取回节点依赖。"""
    if config is None:
        raise ValueError("Agentic RAG graph config is required.")
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if not isinstance(configurable, dict):
        raise ValueError("Agentic RAG graph config is missing configurable dependencies.")
    dependencies = configurable.get(AGENTIC_RAG_CONFIG_KEY)
    if not isinstance(dependencies, AgenticRagGraphDependencies):
        raise ValueError("Agentic RAG graph dependencies are missing.")
    return dependencies

