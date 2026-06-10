from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.model_call import SharedModelCallGuard
from backend.platform.agent_runtime.middleware.trace import RuntimeTraceMiddleware


AGENTIC_RAG_CONFIG_KEY = "agentic_rag"


@dataclass(frozen=True)
class AgenticRagGraphDependencies:
    """Agentic RAG 图运行时需要的最小依赖集合。"""

    retriever: Any
    run_manager: CallbackManagerForRetrieverRun | None = None
    model_call_guard: SharedModelCallGuard | None = None
    trace: RuntimeTraceMiddleware | None = None


class AgenticRagGraphContext(dict[str, Any]):
    """LangGraph context_schema 只读上下文，承载节点运行依赖和安全元数据。"""


def build_agentic_rag_graph_config(
    *,
    retriever: Any,
    run_manager: CallbackManagerForRetrieverRun | None = None,
    runtime_context: AgentRuntimeContext | None = None,
    model_call_guard: SharedModelCallGuard | None = None,
    trace: RuntimeTraceMiddleware | None = None,
    thread_id: str | None = None,
    checkpoint_ns: str | None = None,
) -> RunnableConfig:
    """构建 graph config；typed context 通过 invoke(context=...) 传入。"""
    configurable: dict[str, Any] = {}
    if thread_id:
        configurable["thread_id"] = thread_id
    if checkpoint_ns:
        configurable["checkpoint_ns"] = checkpoint_ns
    return {"configurable": configurable}


def build_agentic_rag_graph_context(
    *,
    retriever: Any,
    run_manager: CallbackManagerForRetrieverRun | None = None,
    runtime_context: AgentRuntimeContext | None = None,
    model_call_guard: SharedModelCallGuard | None = None,
    trace: RuntimeTraceMiddleware | None = None,
) -> AgenticRagGraphContext:
    """构建 LangGraph typed runtime context，供节点只读解析依赖。"""
    context: AgenticRagGraphContext = AgenticRagGraphContext(
        {
            AGENTIC_RAG_CONFIG_KEY: AgenticRagGraphDependencies(
                retriever=retriever,
                run_manager=run_manager,
                model_call_guard=model_call_guard,
                trace=trace,
            ),
        }
    )
    if runtime_context is not None:
        context["runtime_context"] = runtime_context
    return context


def resolve_agentic_rag_graph_dependencies(
    config: RunnableConfig | dict[str, Any] | Runtime[AgenticRagGraphContext] | None,
) -> AgenticRagGraphDependencies:
    """从 LangGraph context/config 中取回节点依赖。"""
    if config is None:
        raise ValueError("Agentic RAG graph config is required.")
    if isinstance(config, Runtime):
        dependencies = config.context.get(AGENTIC_RAG_CONFIG_KEY) if isinstance(config.context, dict) else None
        if isinstance(dependencies, AgenticRagGraphDependencies):
            return dependencies
    dependencies = _read_dependency(config, source="context")
    if dependencies is None:
        dependencies = _read_dependency(config, source="configurable")
    if not isinstance(dependencies, AgenticRagGraphDependencies):
        raise ValueError("Agentic RAG graph dependencies are missing.")
    return dependencies


def resolve_agentic_rag_runtime_context(
    config: RunnableConfig | dict[str, Any] | Runtime[AgenticRagGraphContext] | None,
) -> AgentRuntimeContext | None:
    if isinstance(config, Runtime):
        runtime_context = config.context.get("runtime_context") if isinstance(config.context, dict) else None
        return runtime_context if isinstance(runtime_context, AgentRuntimeContext) else None
    if not isinstance(config, dict):
        return None
    context = config.get("context")
    if not isinstance(context, dict):
        return None
    runtime_context = context.get("runtime_context")
    return runtime_context if isinstance(runtime_context, AgentRuntimeContext) else None


def _read_dependency(
    config: RunnableConfig | dict[str, Any],
    *,
    source: str,
) -> AgenticRagGraphDependencies | None:
    if not isinstance(config, dict):
        return None
    payload = config.get(source)
    if not isinstance(payload, dict):
        return None
    value = payload.get(AGENTIC_RAG_CONFIG_KEY)
    return value if isinstance(value, AgenticRagGraphDependencies) else None
