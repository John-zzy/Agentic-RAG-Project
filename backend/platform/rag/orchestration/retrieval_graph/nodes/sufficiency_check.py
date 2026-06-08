from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from backend.platform.rag.contracts import RetrievalContext
from backend.platform.rag.orchestration.retrieval_graph.config import (
    AgenticRagGraphDependencies,
    resolve_agentic_rag_graph_dependencies,
    resolve_agentic_rag_runtime_context,
)
from backend.platform.rag.orchestration.retrieval_graph.state import AgenticRagGraphState


def build_sufficiency_check_node(dependencies: AgenticRagGraphDependencies):
    """调用 judge，只写决策结果，不在这里做分支跳转。"""

    def sufficiency_check(
        state: AgenticRagGraphState,
        runtime: Runtime | RunnableConfig | None = None,
    ) -> dict[str, Any]:
        runtime_dependencies = _dependencies(runtime, dependencies)
        plan = state["plan"]
        retrieval_context = RetrievalContext(
            plan=plan,
            results=list(state["results"]),
            documents=list(state["documents"]),
        )
        decision = _invoke_judge(
            dependencies=runtime_dependencies,
            context=retrieval_context,
            runtime=runtime,
        )
        return {
            "current_decision": decision,
            "follow_up_question": decision.follow_up_question,
            "route_next_action": decision.next_action,
        }

    return sufficiency_check


def _invoke_judge(
    *,
    dependencies: AgenticRagGraphDependencies,
    context: RetrievalContext,
    runtime: Runtime | RunnableConfig | None,
):
    runtime_context = resolve_agentic_rag_runtime_context(runtime)
    if dependencies.model_call_guard is None or runtime_context is None:
        return dependencies.retriever._judge(context, dependencies.run_manager)
    return dependencies.model_call_guard.invoke(
        lambda: dependencies.retriever._judge(context, dependencies.run_manager),
        context=runtime_context,
        metadata={
            "operation": "agentic_rag.sufficiency_judge",
            "tool_name": context.plan.selected_tool,
            "round_index": context.plan.round_index,
        },
    )


def _dependencies(
    config: Runtime | RunnableConfig | None,
    fallback: AgenticRagGraphDependencies,
) -> AgenticRagGraphDependencies:
    if config is None:
        return fallback
    return resolve_agentic_rag_graph_dependencies(config)
