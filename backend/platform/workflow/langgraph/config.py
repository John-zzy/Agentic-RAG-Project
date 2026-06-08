from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

CHAT_GRAPH_CHECKPOINT_NS = "chat_graph"
REACT_PROVIDER_CHECKPOINT_NS = "react_provider"
PLAN_GRAPH_CHECKPOINT_NS = "plan_graph"
AGENTIC_RAG_CHECKPOINT_NS = "agentic_rag"

DEFAULT_RUNTIME_CHECKPOINT_NS = CHAT_GRAPH_CHECKPOINT_NS
RUNTIME_CHECKPOINT_NAMESPACES = {
    "chat_graph": CHAT_GRAPH_CHECKPOINT_NS,
    "react_provider": REACT_PROVIDER_CHECKPOINT_NS,
    "plan_graph": PLAN_GRAPH_CHECKPOINT_NS,
    "agentic_rag": AGENTIC_RAG_CHECKPOINT_NS,
}


def checkpoint_namespace_for(graph_name: str) -> str:
    """返回项目约定的确定性 checkpoint namespace。"""
    try:
        return RUNTIME_CHECKPOINT_NAMESPACES[graph_name]
    except KeyError as exc:
        raise ValueError(f"Unknown runtime graph checkpoint namespace: {graph_name}") from exc


def build_runtime_graph_config(
    *,
    session_id: str,
    request_id: str,
    checkpoint_ns: str = DEFAULT_RUNTIME_CHECKPOINT_NS,
    checkpoint_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RunnableConfig:
    """构造 LangGraph config：thread 绑定 session，run metadata 绑定 request。"""
    if not session_id:
        raise ValueError("session_id is required for runtime graph config.")
    if not request_id:
        raise ValueError("request_id is required for runtime graph config.")

    configurable: dict[str, Any] = {
        "thread_id": session_id,
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id

    # request/session 是 runtime 关联主键，显式覆盖外部 metadata 中的同名字段。
    resolved_metadata = dict(metadata or {})
    resolved_metadata["request_id"] = request_id
    resolved_metadata["session_id"] = session_id
    resolved_metadata["checkpoint_ns"] = checkpoint_ns

    return {
        "configurable": configurable,
        "metadata": resolved_metadata,
    }
