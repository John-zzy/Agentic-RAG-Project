from __future__ import annotations

from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from backend.knowledge.rag.core import (
    QueryRewrite,
    QueryRewriter,
    RetrievalContext,
    RetrievalPlan,
    RetrievalResult,
    RetrievalTool,
    SufficiencyDecision,
    SufficiencyJudge,
)


class RetrievalRound(RetrievalContext):
    """描述单轮检索执行轨迹，供编排器输出过程结果与调试信息。"""

    result: RetrievalResult
    decision: SufficiencyDecision
    rewrite: QueryRewrite | None = None


class AgenticRetrievalOutcome(RetrievalContext):
    """描述一次 Agentic Retrieval 会话的聚合结果与退出状态。"""

    success: bool
    rounds: list[RetrievalRound] = Field(default_factory=list)
    final_plan: RetrievalPlan
    final_decision: SufficiencyDecision
    exit_reason: str
    follow_up_question: str | None = None


class AgenticRetriever(BaseRetriever):
    """基于 LangChain BaseRetriever 编排多轮检索。"""

    tools: dict[str, RetrievalTool] = Field(default_factory=dict)
    sufficiency_judge: SufficiencyJudge = Field(exclude=True)
    query_rewriter: QueryRewriter | None = Field(default=None, exclude=True)
    default_tool: str | None = None
    max_rounds: int = Field(default=3, ge=1)
    attach_trace: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """适配 LangChain retriever 协议，对外返回最终聚合的 Document 列表。"""
        return self.retrieve_with_trace(query=query, run_manager=run_manager).documents

    def retrieve_with_trace(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        selected_tool: str | None = None,
        candidate_tools: tuple[str, ...] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AgenticRetrievalOutcome:
        """执行多轮检索并返回完整轨迹，供 Agent 或 LangGraph 复用。"""
        self._validate_tools()
        initial_tool = selected_tool or self.default_tool or next(iter(self.tools))
        current_plan = RetrievalPlan(
            user_query=query,
            active_query=query,
            selected_tool=initial_tool,
            max_rounds=self.max_rounds,
            candidate_tools=candidate_tools or tuple(self.tools.keys()),
            filters=filters or {},
        )
        rounds: list[RetrievalRound] = []
        results: list[RetrievalResult] = []
        documents: list[Document] = []

        while True:
            result = self._run_tool(current_plan, run_manager)
            results.append(result)
            documents = self._merge_documents(documents, result.documents)

            context = RetrievalContext(plan=current_plan, results=results, documents=documents)
            decision = self._judge(context, run_manager)
            round_trace = RetrievalRound(
                plan=current_plan,
                results=list(results),
                documents=list(documents),
                result=result,
                decision=decision,
            )

            if decision.is_sufficient or decision.next_action == "finish":
                rounds.append(round_trace)
                exit_reason = "sufficient" if decision.is_sufficient else "finished_by_judge"
                return AgenticRetrievalOutcome(
                    plan=current_plan,
                    results=results,
                    documents=self._finalize_documents(documents, rounds, exit_reason),
                    success=decision.is_sufficient and result.success,
                    rounds=rounds,
                    final_plan=current_plan,
                    final_decision=decision,
                    exit_reason=exit_reason,
                    follow_up_question=decision.follow_up_question,
                )

            if current_plan.round_index >= current_plan.max_rounds:
                bounded_decision = decision.model_copy(
                    update={
                        "is_sufficient": False,
                        "next_action": "ask_user",
                        "reason": (
                            f"{decision.reason} Reached max retrieval rounds "
                            f"({current_plan.max_rounds})."
                        ),
                    }
                )
                rounds.append(
                    RetrievalRound(
                        plan=current_plan,
                        results=list(results),
                        documents=list(documents),
                        result=result,
                        decision=bounded_decision,
                    )
                )
                return AgenticRetrievalOutcome(
                    plan=current_plan,
                    results=results,
                    documents=self._finalize_documents(documents, rounds, "max_rounds_reached"),
                    success=False,
                    rounds=rounds,
                    final_plan=current_plan,
                    final_decision=bounded_decision,
                    exit_reason="max_rounds_reached",
                    follow_up_question=bounded_decision.follow_up_question,
                )

            if decision.next_action == "ask_user":
                rounds.append(round_trace)
                return AgenticRetrievalOutcome(
                    plan=current_plan,
                    results=results,
                    documents=self._finalize_documents(documents, rounds, "ask_user"),
                    success=False,
                    rounds=rounds,
                    final_plan=current_plan,
                    final_decision=decision,
                    exit_reason="ask_user",
                    follow_up_question=decision.follow_up_question,
                )

            if decision.next_action == "rewrite":
                rewrite = self._rewrite_query(context, run_manager)
                round_trace.rewrite = rewrite
                next_plan = current_plan.create_followup(
                    active_query=rewrite.query,
                    metadata={
                        "rewrite_reason": rewrite.reason,
                        "rewrite_metadata": rewrite.metadata,
                    },
                )
            elif decision.next_action == "switch_tool":
                next_plan = current_plan.create_followup(
                    selected_tool=self._resolve_next_tool(current_plan, decision)
                )
            else:
                raise ValueError(f"Unsupported retrieval next action: {decision.next_action}")

            rounds.append(round_trace)
            current_plan = next_plan

    def _run_tool(
        self,
        plan: RetrievalPlan,
        run_manager: CallbackManagerForRetrieverRun | None,
    ) -> RetrievalResult:
        """执行当前轮次指定工具，并传递 LangChain 回调上下文。"""
        tool = self._get_tool(plan.selected_tool)
        child_manager = run_manager.get_child(tag=f"retrieval:{tool.name}") if run_manager else None
        return tool.retrieve(query=plan.active_query, run_manager=child_manager)

    def _rewrite_query(
        self,
        context: RetrievalContext,
        run_manager: CallbackManagerForRetrieverRun | None,
    ) -> QueryRewrite:
        """调用 LangChain runnable 风格的查询改写器。"""
        if self.query_rewriter is None:
            raise ValueError("Query rewriter is required when judge requests rewrite.")
        rewrite = self.query_rewriter.invoke(
            context,
            config=self._build_runnable_config(run_manager, tag="query_rewriter"),
        )
        if not rewrite.query.strip():
            raise ValueError("Query rewriter returned an empty query.")
        return rewrite

    def _judge(
        self,
        context: RetrievalContext,
        run_manager: CallbackManagerForRetrieverRun | None,
    ) -> SufficiencyDecision:
        """调用 LangChain runnable 风格的充分性判断器。"""
        return self.sufficiency_judge.invoke(
            context,
            config=self._build_runnable_config(run_manager, tag="sufficiency_judge"),
        )

    def _resolve_next_tool(self, plan: RetrievalPlan, decision: SufficiencyDecision) -> str:
        """解析下一轮工具；优先使用 judge 建议，其次选择尚未尝试的候选工具。"""
        if decision.suggested_tool:
            self._get_tool(decision.suggested_tool)
            return decision.suggested_tool

        attempted = set(plan.attempted_tools) | {plan.selected_tool}
        for tool_name in plan.candidate_tools:
            if tool_name not in attempted:
                self._get_tool(tool_name)
                return tool_name

        raise ValueError("No alternative retrieval tool available for switch_tool decision.")

    def _get_tool(self, tool_name: str) -> RetrievalTool:
        """按名称解析检索工具，不存在时抛出显式错误。"""
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Retrieval tool '{tool_name}' is not registered.")
        return tool

    def _validate_tools(self) -> None:
        """校验工具集合、默认工具和最大轮次配置是否有效。"""
        if not self.tools:
            raise ValueError("At least one retrieval tool must be provided.")
        if self.default_tool and self.default_tool not in self.tools:
            raise ValueError(f"Default retrieval tool '{self.default_tool}' is not registered.")

    def _merge_documents(
        self,
        existing: list[Document],
        incoming: list[Document],
    ) -> list[Document]:
        """按文档标识聚合去重，保留首次出现顺序。"""
        merged = list(existing)
        seen = {self._document_key(document) for document in existing}
        for document in incoming:
            key = self._document_key(document)
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)
        return merged

    def _finalize_documents(
        self,
        documents: list[Document],
        rounds: list[RetrievalRound],
        exit_reason: str,
    ) -> list[Document]:
        """按需将检索轨迹写入文档 metadata，便于 LangChain 下游链路调试。"""
        if not self.attach_trace:
            return documents

        trace = [
            {
                "round_index": round_trace.plan.round_index,
                "tool_name": round_trace.result.tool_name,
                "query": round_trace.result.query,
                "decision": round_trace.decision.next_action,
                "reason": round_trace.decision.reason,
            }
            for round_trace in rounds
        ]
        return [
            document.model_copy(
                update={
                    "metadata": {
                        **document.metadata,
                        "agentic_retrieval": {
                            "trace": trace,
                            "exit_reason": exit_reason,
                        },
                    }
                }
            )
            for document in documents
        ]

    def _build_runnable_config(
        self,
        run_manager: CallbackManagerForRetrieverRun | None,
        *,
        tag: str,
    ) -> dict[str, Any] | None:
        """为内部 runnable 构造 LangChain config，保证 tracing 不断链。"""
        if run_manager is None:
            return None
        return {"callbacks": [run_manager.get_child(tag=tag)]}

    def _document_key(self, document: Document) -> tuple[str, str]:
        """为去重计算稳定键，优先使用 metadata 中的 citation_id/source。"""
        citation_id = str(document.metadata.get("citation_id") or document.id or document.page_content)
        source = str(document.metadata.get("namespace") or document.metadata.get("source") or "knowledge")
        return source, citation_id
