from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.platform.agent_runtime.contracts import ToolObservation
from backend.platform.agent_runtime.idempotency import (
    SQLiteToolIdempotencyStore,
    ToolExecutionContext,
)
from backend.platform.agent_runtime.rag_tools import (
    AgenticRagToolAdapter,
    NativeRagToolAdapter,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.rag.contracts import (
    RetrievalPlan,
    RetrievalResult,
)
from backend.platform.rag.orchestration.agentic import (
    AgenticRetrievalOutcome,
    RetrievalRound,
)
from backend.platform.rag.orchestration.decisions import SufficiencyDecision
from backend.platform.tools.base import SceneTool, ToolResult
from backend.tests.test_support import make_test_runtime_dir


class _SearchArgs(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1)


class _LookupArgs(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=1, ge=1)


class _FakeNativeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> RetrievalResult:
        self.calls.append(dict(kwargs))
        return self.result


class _FakeAgenticRetriever:
    def __init__(self, outcome: AgenticRetrievalOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def retrieve_with_trace(self, **kwargs: Any) -> AgenticRetrievalOutcome:
        self.calls.append(dict(kwargs))
        return self.outcome


class _FakeSceneDefinition:
    def __init__(
        self,
        *tools: Any,
        candidate_tools: tuple[str, ...] = ("knowledge_document_search",),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._tools = tools
        self._candidate_tools = candidate_tools
        self.metadata = metadata or {}

    def build_tools(self) -> tuple[Any, ...]:
        return self._tools

    def resolve_candidate_retrieval_tools(
        self,
        mounted_knowledge_sources: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not mounted_knowledge_sources:
            return ()
        return self._candidate_tools


class _ActionWriteTool(SceneTool):
    name = "write_record"
    description = "Write a record."
    capability_type = "action"
    args_schema = _LookupArgs

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult.ok(
            tool_name=self.name,
            records=[dict(kwargs)],
            metadata={"side_effect": "local_write"},
        )


def test_native_rag_tool_success_preserves_evidence_metadata() -> None:
    retriever = _FakeNativeRetriever(
        RetrievalResult.ok(
            tool_name="knowledge_document_search",
            query="报销制度",
            records=[{"chunk_id": "chunk-1", "text": "差旅报销需提交发票。"}],
            documents=[
                Document(
                    page_content="差旅报销需提交发票。",
                    metadata={"citation_id": "chunk-1", "source": "policy.md"},
                )
            ],
            citations=[
                {
                    "citation_id": "chunk-1",
                    "snippet": "差旅报销需提交发票。",
                    "source_type": "document",
                }
            ],
            metadata={"retrieval_trace": {"raw_candidates_count": 1}},
        )
    )
    tool = NativeRagToolAdapter(
        name="native_rag_search",
        retriever=retriever,
        args_schema=_SearchArgs,
    )

    observation = tool.invoke({"query": "报销制度", "top_k": 1})

    assert isinstance(observation, ToolObservation)
    assert observation.tool_name == "native_rag_search"
    assert observation.success is True
    assert observation.output["query"] == "报销制度"
    assert observation.output["knowledge_used"] is True
    assert observation.citations == [
        {
            "citation_id": "chunk-1",
            "snippet": "差旅报销需提交发票。",
            "source_type": "document",
            "metadata": {},
        }
    ]
    assert observation.trace["retrieval_trace"]["final_decision"] == "answer_with_evidence"
    assert observation.trace["retrieval_trace"]["knowledge_used"] is True
    assert retriever.calls[0]["top_k"] == 1


def test_agentic_rag_tool_success_keeps_rounds_as_nested_trace_not_react_turns() -> None:
    round_trace = _agentic_round(
        round_index=1,
        result=RetrievalResult.ok(
            tool_name="knowledge_document_search",
            query="绩效制度",
            records=[{"chunk_id": "chunk-perf-1"}],
            documents=[Document(page_content="绩效每季度校准一次。")],
        ),
        decision=SufficiencyDecision(
            is_sufficient=True,
            next_action="finish",
            reason="证据足够。",
        ),
    )
    outcome = _agentic_outcome(
        success=True,
        rounds=[round_trace],
        final_decision=round_trace.decision,
        exit_reason="sufficient",
    )
    tool = AgenticRagToolAdapter(
        name="agentic_rag_search",
        retriever=_FakeAgenticRetriever(outcome),
        args_schema=_SearchArgs,
    )

    observation = tool.invoke({"query": "绩效制度", "top_k": 2})

    retrieval_trace = observation.trace["retrieval_trace"]
    assert observation.success is True
    assert retrieval_trace["final_decision"] == "answer_with_evidence"
    assert retrieval_trace["rounds"][0]["round_index"] == 1
    assert retrieval_trace["rounds"][0]["decision"] == "finish"
    assert "turn_id" not in retrieval_trace["rounds"][0]
    assert "rounds" not in observation.trace
    assert "react_turns" not in observation.trace


def test_agentic_rag_tool_uses_default_candidate_tools_when_input_omits_them() -> None:
    round_trace = _agentic_round(
        round_index=1,
        result=RetrievalResult.ok(
            tool_name="knowledge_document_search",
            query="制度",
            documents=[Document(page_content="制度证据。", metadata={"citation_id": "chunk-1"})],
        ),
        decision=SufficiencyDecision(
            is_sufficient=True,
            next_action="finish",
            reason="证据足够。",
        ),
    )
    retriever = _FakeAgenticRetriever(
        _agentic_outcome(
            success=True,
            rounds=[round_trace],
            final_decision=round_trace.decision,
            exit_reason="sufficient",
        )
    )
    tool = AgenticRagToolAdapter(
        name="agentic_rag_search",
        retriever=retriever,
        candidate_tools=("knowledge_document_search",),
    )

    observation = tool.invoke({"query": "制度"})

    assert observation.success is True
    assert retriever.calls[0]["candidate_tools"] == ("knowledge_document_search",)


def test_agentic_rag_tool_ask_user_observation_promotes_follow_up_signal() -> None:
    decision = SufficiencyDecision(
        is_sufficient=False,
        next_action="ask_user",
        reason="缺少范围。",
        follow_up_question="请确认要查询哪个制度范围？",
    )
    outcome = _agentic_outcome(
        success=False,
        rounds=[
            _agentic_round(
                round_index=1,
                result=RetrievalResult.ok(
                    tool_name="knowledge_document_search",
                    query="制度",
                ),
                decision=decision,
            )
        ],
        final_decision=decision,
        exit_reason="ask_user",
        follow_up_question="请确认要查询哪个制度范围？",
    )
    tool = AgenticRagToolAdapter(
        name="agentic_rag_search",
        retriever=_FakeAgenticRetriever(outcome),
        args_schema=_SearchArgs,
    )

    observation = tool.invoke({"query": "制度"})

    assert observation.success is True
    assert observation.requires_user is True
    assert observation.user_prompt == "请确认要查询哪个制度范围？"
    assert observation.trace["retrieval_trace"]["final_decision"] == "ask_user"
    assert observation.trace["retrieval_trace"]["follow_up_question"] == observation.user_prompt


def test_failed_retrieval_observation_keeps_no_evidence_boundary() -> None:
    tool = NativeRagToolAdapter(
        name="native_rag_search",
        retriever=_FakeNativeRetriever(
            RetrievalResult.fail(
                tool_name="knowledge_document_search",
                query="不存在的资料",
                error="retrieval backend unavailable",
            )
        ),
        args_schema=_SearchArgs,
    )

    observation = tool.invoke({"query": "不存在的资料"})

    assert observation.success is False
    assert observation.output["knowledge_used"] is False
    assert observation.citations == []
    assert observation.error == "retrieval backend unavailable"
    assert observation.trace["retrieval_trace"]["final_decision"] == "retrieval_failed"
    assert observation.trace["retrieval_trace"]["citations"] == []


def test_native_rag_records_without_citations_remain_no_evidence() -> None:
    tool = NativeRagToolAdapter(
        name="native_rag_search",
        retriever=_FakeNativeRetriever(
            RetrievalResult.ok(
                tool_name="knowledge_document_search",
                query="只有结构化记录",
                records=[{"id": "record-1"}],
            )
        ),
        args_schema=_SearchArgs,
    )

    observation = tool.invoke({"query": "只有结构化记录"})

    assert observation.success is True
    assert observation.output["knowledge_used"] is False
    assert observation.trace["retrieval_trace"]["final_decision"] == "no_evidence"
    assert observation.citations == []


def test_agentic_rag_failed_round_normalizes_to_retrieval_failed() -> None:
    failed_result = RetrievalResult.fail(
        tool_name="knowledge_document_search",
        query="失败查询",
        error="retrieval backend unavailable",
    )
    decision = SufficiencyDecision(
        is_sufficient=False,
        next_action="finish",
        reason="检索失败。",
    )
    tool = AgenticRagToolAdapter(
        name="agentic_rag_search",
        retriever=_FakeAgenticRetriever(
            _agentic_outcome(
                success=False,
                rounds=[
                    _agentic_round(
                        round_index=1,
                        result=failed_result,
                        decision=decision,
                    )
                ],
                final_decision=decision,
                exit_reason="retrieval_failed",
            )
        ),
        args_schema=_SearchArgs,
    )

    observation = tool.invoke({"query": "失败查询"})

    assert observation.success is False
    assert observation.retryable is True
    assert observation.error == "retrieval backend unavailable"
    assert observation.trace["retrieval_trace"]["final_decision"] == "retrieval_failed"


def test_tool_executor_runs_allowed_tool_and_normalizes_tool_result() -> None:
    tool = StructuredTool.from_function(
        func=_lookup_policy,
        name="lookup_policy",
        description="Lookup policy records.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor(
        tools={"lookup_policy": tool},
        allowed_tools={"lookup_policy"},
    )

    observation = executor.execute(
        tool_name="lookup_policy",
        input_payload={"query": "报销", "limit": 2},
    )

    assert isinstance(observation, ToolObservation)
    assert observation.success is True
    assert observation.tool_name == "lookup_policy"
    assert observation.output["records"] == [{"query": "报销", "limit": 2}]
    assert observation.result_summary == "lookup_policy succeeded with 1 record(s)."


def test_tool_executor_from_scene_resolves_scene_rag_and_internal_tools() -> None:
    scene_tool = StructuredTool.from_function(
        func=_lookup_policy,
        name="lookup_policy",
        description="Lookup policy records.",
        args_schema=_LookupArgs,
    )
    rag_tool = NativeRagToolAdapter(
        name="native_rag_search",
        retriever=_FakeNativeRetriever(
            RetrievalResult.ok(
                tool_name="knowledge_document_search",
                query="制度",
                records=[{"chunk_id": "chunk-1"}],
            )
        ),
        args_schema=_SearchArgs,
    )
    internal_tool = StructuredTool.from_function(
        func=_lookup_policy,
        name="final_synthesizer",
        description="Synthesize final answer.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor.from_scene(
        scene_definition=_FakeSceneDefinition(scene_tool, candidate_tools=("lookup_policy",)),
        mounted_knowledge_sources=("documents",),
        rag_tools={"native_rag_search": rag_tool},
        internal_tools={"final_synthesizer": internal_tool},
    )

    assert executor.allowed_tools == frozenset(
        {"lookup_policy", "native_rag_search", "final_synthesizer"}
    )
    assert executor.execute(
        tool_name="native_rag_search",
        input_payload={"query": "制度"},
    ).tool_name == "native_rag_search"


def test_tool_executor_from_scene_blocks_unresolved_scene_tools() -> None:
    scene_tool = StructuredTool.from_function(
        func=_lookup_policy,
        name="lookup_policy",
        description="Lookup policy records.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor.from_scene(
        scene_definition=_FakeSceneDefinition(scene_tool, candidate_tools=()),
        mounted_knowledge_sources=("documents",),
    )

    assert executor.allowed_tools == frozenset()
    observation = executor.execute(
        tool_name="lookup_policy",
        input_payload={"query": "报销"},
    )
    assert observation.success is False
    assert observation.retryable is False
    assert observation.error is not None
    assert "not allowed" in observation.error


def test_tool_executor_from_scene_allows_business_scene_tools_when_source_mounted() -> None:
    scene_tool = StructuredTool.from_function(
        func=_lookup_policy,
        name="lookup_policy",
        description="Lookup policy records.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor.from_scene(
        scene_definition=_FakeSceneDefinition(
            scene_tool,
            candidate_tools=(),
            metadata={"knowledge_sources": ("documents", "ecommerce")},
        ),
        mounted_knowledge_sources=("documents", "ecommerce"),
    )

    assert executor.allowed_tools == frozenset({"lookup_policy"})
    observation = executor.execute(
        tool_name="lookup_policy",
        input_payload={"query": "报销"},
    )
    assert observation.success is True


def test_tool_executor_rejects_unavailable_tool_without_invocation() -> None:
    tool = StructuredTool.from_function(
        func=_lookup_policy,
        name="lookup_policy",
        description="Lookup policy records.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor(
        tools={"lookup_policy": tool},
        allowed_tools={"lookup_policy"},
    )

    observation = executor.execute(
        tool_name="unsafe_tool",
        input_payload={"query": "报销"},
    )

    assert observation.success is False
    assert observation.tool_name == "unsafe_tool"
    assert observation.error is not None
    assert "not allowed" in observation.error


def test_tool_executor_rejects_invalid_input_before_invocation() -> None:
    called = False

    def never_called(query: str, limit: int = 1) -> ToolResult:
        nonlocal called
        called = True
        return _lookup_policy(query=query, limit=limit)

    tool = StructuredTool.from_function(
        func=never_called,
        name="lookup_policy",
        description="Lookup policy records.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor(
        tools={"lookup_policy": tool},
        allowed_tools={"lookup_policy"},
    )

    observation = executor.execute(
        tool_name="lookup_policy",
        input_payload={"query": "报销", "limit": 0},
    )

    assert called is False
    assert observation.success is False
    assert observation.retryable is False
    assert observation.error is not None
    assert "Invalid input" in observation.error


def test_tool_executor_normalizes_retryable_errors() -> None:
    def timeout_tool(query: str, limit: int = 1) -> ToolResult:
        raise TimeoutError("lookup timed out")

    tool = StructuredTool.from_function(
        func=timeout_tool,
        name="lookup_policy",
        description="Lookup policy records.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor(
        tools={"lookup_policy": tool},
        allowed_tools={"lookup_policy"},
    )

    observation = executor.execute(
        tool_name="lookup_policy",
        input_payload={"query": "报销"},
    )

    assert observation.success is False
    assert observation.retryable is True
    assert observation.error is not None
    assert "lookup timed out" in observation.error
    assert observation.execution is not None
    assert observation.execution.retryable is True


def test_tool_executor_preserves_hitl_required_observation() -> None:
    def needs_user(query: str, limit: int = 1) -> ToolObservation:
        return ToolObservation(
            tool_name="approval_tool",
            success=False,
            requires_user=True,
            user_prompt="是否批准查询外部系统？",
            result_summary="等待用户批准。",
            trace={"hitl": {"pending_action": "tool_approval"}},
        )

    tool = StructuredTool.from_function(
        func=needs_user,
        name="approval_tool",
        description="Ask for approval.",
        args_schema=_LookupArgs,
    )
    executor = ToolExecutor(
        tools={"approval_tool": tool},
        allowed_tools={"approval_tool"},
    )

    observation = executor.execute(
        tool_name="approval_tool",
        input_payload={"query": "订单"},
    )

    assert observation.success is False
    assert observation.requires_user is True
    assert observation.user_prompt == "是否批准查询外部系统？"
    assert observation.trace["hitl"]["pending_action"] == "tool_approval"


def test_tool_executor_reuses_successful_side_effect_observation_by_idempotency_key() -> None:
    runtime_dir = make_test_runtime_dir("tool-executor-idempotency-reuse")
    store = SQLiteToolIdempotencyStore(runtime_dir / "langgraph.db")
    tool = _ActionWriteTool()
    executor = ToolExecutor(
        tools={tool.name: tool},
        allowed_tools={tool.name},
        idempotency_store=store,
    )
    context = ToolExecutionContext(
        session_id="session-1",
        request_id="request-1",
        run_id="react-run-1",
        node_name="react.execute_tool",
        turn_id="turn-1",
    )

    first = executor.execute(
        tool_name=tool.name,
        input_payload={"query": "订单", "limit": 1},
        execution_context=context,
    )
    second = executor.execute(
        tool_name=tool.name,
        input_payload={"limit": 1, "query": "订单"},
        execution_context=context,
    )

    assert len(tool.calls) == 1
    assert first.success is True
    assert second.success is True
    assert second.metadata["idempotency"]["reused"] is True
    assert second.execution is not None
    assert second.execution.idempotency_key == first.execution.idempotency_key
    facts = store.list_by_session("session-1")
    assert len(facts) == 1
    assert facts[0].status == "succeeded"
    assert facts[0].compensation_status == "unsupported"


def test_tool_idempotency_store_duplicate_begin_returns_existing_pending() -> None:
    runtime_dir = make_test_runtime_dir("tool-idempotency-duplicate-begin")
    store = SQLiteToolIdempotencyStore(runtime_dir / "langgraph.db")
    context = ToolExecutionContext(
        session_id="session-duplicate",
        request_id="request-duplicate",
        run_id="run-duplicate",
        node_name="react.execute_tool",
        turn_id="turn-1",
    )

    first = store.begin_invocation(
        key="tool:duplicate",
        tool_name="write_record",
        input_hash="hash",
        context=context,
        compensation_status="unsupported",
    )
    second = store.begin_invocation(
        key="tool:duplicate",
        tool_name="write_record",
        input_hash="hash",
        context=context,
        compensation_status="unsupported",
    )

    assert first is None
    assert second is not None
    assert second.status == "pending"
    assert store.get("tool:duplicate").status == "pending"


def _lookup_policy(query: str, limit: int = 1) -> ToolResult:
    return ToolResult.ok(
        tool_name="lookup_policy",
        records=[{"query": query, "limit": limit}],
        citations=[{"citation_id": "policy-1"}],
    )


def _agentic_round(
    *,
    round_index: int,
    result: RetrievalResult,
    decision: SufficiencyDecision,
) -> RetrievalRound:
    plan = RetrievalPlan(
        user_query=result.query,
        active_query=result.query,
        selected_tool=result.tool_name,
        round_index=round_index,
    )
    return RetrievalRound(
        plan=plan,
        results=[result],
        documents=list(result.documents),
        result=result,
        decision=decision,
    )


def _agentic_outcome(
    *,
    success: bool,
    rounds: list[RetrievalRound],
    final_decision: SufficiencyDecision,
    exit_reason: str,
    follow_up_question: str | None = None,
) -> AgenticRetrievalOutcome:
    final_plan = rounds[-1].plan
    results = [round_trace.result for round_trace in rounds]
    documents = [
        document
        for round_trace in rounds
        for document in round_trace.documents
    ]
    return AgenticRetrievalOutcome(
        plan=final_plan,
        results=results,
        documents=documents,
        success=success,
        rounds=rounds,
        decision_log=[],
        final_plan=final_plan,
        final_decision=final_decision,
        exit_reason=exit_reason,
        follow_up_question=follow_up_question,
    )
