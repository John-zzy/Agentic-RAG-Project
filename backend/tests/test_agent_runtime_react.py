from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.platform.agent_runtime.contracts import (
    ReActAction,
    ToolObservation,
)
from backend.platform.agent_runtime.react import (
    ReActActionContext,
    ReActRuntime,
    ReActSynthesisContext,
    ReActSynthesisResult,
)
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.tools.base import ToolResult


class _QueryArgs(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=1, ge=1)


class _SequencedActionSelector:
    def __init__(self, *actions: ReActAction) -> None:
        self._actions = list(actions)
        self.contexts: list[ReActActionContext] = []

    def select_action(self, context: ReActActionContext) -> ReActAction:
        self.contexts.append(context)
        if not self._actions:
            raise AssertionError("No action configured for ReAct selector.")
        return self._actions.pop(0)


class _RecordingSynthesizer:
    def __init__(self, *, answer_prefix: str = "final") -> None:
        self.answer_prefix = answer_prefix
        self.contexts: list[ReActSynthesisContext] = []

    def synthesize(self, context: ReActSynthesisContext) -> ReActSynthesisResult:
        self.contexts.append(context)
        summaries = [
            observation.result_summary for observation in context.observations
        ]
        return ReActSynthesisResult(
            final_answer=f"{self.answer_prefix}: {' | '.join(summaries)}",
            result_summary=f"used {len(context.observations)} observation(s)",
            citations=list(context.citations),
            knowledge_used=bool(context.citations),
            metadata={
                "observation_count": len(context.observations),
                "max_turns_reached": context.metadata.get("max_turns_reached", False),
            },
        )


class _StaticObservationTool:
    name = "agentic_rag_search"
    description = "Return a static RAG observation."
    args_schema = _QueryArgs

    def __init__(self, observation: ToolObservation) -> None:
        self.observation = observation
        self.calls: list[dict[str, Any]] = []

    def invoke(self, input_payload: dict[str, Any]) -> ToolObservation:
        self.calls.append(dict(input_payload))
        return self.observation


def test_react_single_tool_success_synthesizes_final_answer() -> None:
    tool_calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=tool_calls,
                records=[{"policy": "travel"}],
                citations=[{"citation_id": "policy-1"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="lookup_policy",
            input={"query": "travel reimbursement", "limit": 1},
        ),
        ReActAction(action_type="final_answer", instruction="answer now"),
    )
    synthesizer = _RecordingSynthesizer()

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        final_synthesizer=synthesizer,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-1",
        request_id="request-1",
        user_goal="Find the travel reimbursement policy.",
        react_run_id="react-1",
    )

    assert run.workflow_status == "succeeded"
    assert run.current_turn_id is None
    assert run.current_tool_call is None
    assert run.final_answer == "final: lookup_policy succeeded with 1 record(s)."
    assert run.metadata["citations"] == [{"citation_id": "policy-1"}]
    assert run.metadata["knowledge_used"] is True
    assert [turn.status for turn in run.turns] == ["succeeded", "succeeded"]
    assert run.turns[0].turn_id == "turn-1"
    assert run.turns[0].tool_name == "lookup_policy"
    assert run.turns[0].input == {"query": "travel reimbursement", "limit": 1}
    assert run.turns[0].observation is not None
    assert run.turns[0].observation.citations == [{"citation_id": "policy-1"}]
    assert tool_calls == [{"query": "travel reimbursement", "limit": 1}]
    assert selector.contexts[0].allowed_tools == ["lookup_policy"]
    assert selector.contexts[1].previous_turns[0].turn_id == "turn-1"
    assert synthesizer.contexts[0].observations == [run.turns[0].observation]


def test_react_multi_tool_success_records_each_top_level_tool_turn() -> None:
    lookup_calls: list[dict[str, Any]] = []
    inventory_calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=lookup_calls,
                records=[{"policy": "return-window"}],
                citations=[{"citation_id": "policy-1"}],
            ),
            "lookup_inventory": _structured_tool(
                name="lookup_inventory",
                calls=inventory_calls,
                records=[{"sku": "sku-1", "available": True}],
                citations=[{"citation_id": "inventory-1"}],
            ),
        },
        allowed_tools={"lookup_inventory", "lookup_policy"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="lookup_policy",
            input={"query": "return window"},
        ),
        ReActAction(
            action_type="tool_call",
            tool_name="lookup_inventory",
            input={"query": "sku-1"},
        ),
        ReActAction(action_type="final_answer"),
    )
    synthesizer = _RecordingSynthesizer(answer_prefix="combined")

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        final_synthesizer=synthesizer,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-1",
        request_id="request-2",
        user_goal="Check return policy and item availability.",
        react_run_id="react-2",
    )

    assert run.workflow_status == "succeeded"
    assert [turn.tool_name for turn in run.turns] == [
        "lookup_policy",
        "lookup_inventory",
        None,
    ]
    assert [turn.round_index for turn in run.turns] == [1, 2, 3]
    assert lookup_calls == [{"query": "return window", "limit": 1}]
    assert inventory_calls == [{"query": "sku-1", "limit": 1}]
    assert run.final_answer == (
        "combined: lookup_policy succeeded with 1 record(s). | "
        "lookup_inventory succeeded with 1 record(s)."
    )
    assert run.metadata["citations"] == [
        {"citation_id": "policy-1"},
        {"citation_id": "inventory-1"},
    ]
    assert len(synthesizer.contexts[0].turns) == 3
    assert len(synthesizer.contexts[0].observations) == 2
    assert selector.contexts[2].previous_turns[1].tool_name == "lookup_inventory"


def test_react_rag_tool_keeps_internal_rounds_nested_under_observation_trace() -> None:
    rag_observation = ToolObservation(
        tool_name="agentic_rag_search",
        success=True,
        output={"knowledge_used": True},
        result_summary="agentic_rag_search returned 2 evidence record(s).",
        citations=[{"citation_id": "doc-1"}],
        trace={
            "retrieval_trace": {
                "final_decision": "answer_with_evidence",
                "rounds": [
                    {
                        "round_index": 1,
                        "tool_name": "knowledge_document_search",
                        "decision": "rewrite",
                    },
                    {
                        "round_index": 2,
                        "tool_name": "knowledge_document_search",
                        "decision": "finish",
                    },
                ],
            }
        },
        metadata={"knowledge_used": True},
    )
    rag_tool = _StaticObservationTool(rag_observation)
    executor = ToolExecutor(
        tools={"agentic_rag_search": rag_tool},
        allowed_tools={"agentic_rag_search"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="agentic_rag_search",
            input={"query": "policy evidence"},
        ),
        ReActAction(action_type="final_answer"),
    )

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-1",
        request_id="request-rag",
        user_goal="Find evidence.",
        react_run_id="react-rag",
    )

    assert run.workflow_status == "succeeded"
    assert len(run.turns) == 2
    assert run.turns[0].tool_name == "agentic_rag_search"
    assert run.turns[0].observation is not None
    retrieval_trace = run.turns[0].observation.trace["retrieval_trace"]
    assert [round_trace["round_index"] for round_trace in retrieval_trace["rounds"]] == [1, 2]
    assert "turn_id" not in retrieval_trace["rounds"][0]
    assert "react_turns" not in run.turns[0].observation.trace
    assert run.turns[1].action.action_type == "final_answer"
    assert rag_tool.calls == [{"query": "policy evidence", "limit": 1}]
    assert run.metadata["citations"] == [{"citation_id": "doc-1"}]


def test_react_stops_at_max_turns_and_synthesizes_available_observations() -> None:
    tool_calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=tool_calls,
                records=[{"policy": "expense"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="lookup_policy",
            input={"query": "expense policy"},
        ),
        ReActAction(
            action_type="tool_call",
            tool_name="lookup_policy",
            input={"query": "should not run"},
        ),
    )
    synthesizer = _RecordingSynthesizer()

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        final_synthesizer=synthesizer,
        max_turns=1,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-1",
        request_id="request-max",
        user_goal="Keep looking until stopped.",
        react_run_id="react-max",
    )

    assert run.workflow_status == "succeeded"
    assert len(run.turns) == 1
    assert len(selector.contexts) == 1
    assert tool_calls == [{"query": "expense policy", "limit": 1}]
    assert run.metadata["max_turns_reached"] is True
    assert run.metadata["final_synthesis"]["max_turns_reached"] is True
    assert synthesizer.contexts[0].metadata["max_turns_reached"] is True


def test_react_retryable_tool_failure_returns_retrying_run() -> None:
    def timeout_lookup(query: str, limit: int = 1) -> ToolResult:
        raise TimeoutError(f"timeout for {query}")

    executor = ToolExecutor(
        tools={
            "lookup_policy": StructuredTool.from_function(
                func=timeout_lookup,
                name="lookup_policy",
                description="Lookup policy records.",
                args_schema=_QueryArgs,
            )
        },
        allowed_tools={"lookup_policy"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="lookup_policy",
            input={"query": "expense policy"},
        ),
        ReActAction(action_type="final_answer"),
    )

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-1",
        request_id="request-retry",
        user_goal="Lookup policy.",
        react_run_id="react-retry",
    )

    assert run.workflow_status == "retrying"
    assert len(run.turns) == 1
    assert len(selector.contexts) == 1
    assert run.current_turn_id == "turn-1"
    assert run.current_tool_call is not None
    assert run.current_tool_call.tool_name == "lookup_policy"
    assert run.error == "timeout for expense policy"
    assert run.turns[0].status == "retrying"
    assert run.turns[0].retry_metadata.attempt == 1
    assert run.turns[0].retry_metadata.retryable is True
    assert run.turns[0].retry_metadata.last_error == "timeout for expense policy"
    assert run.turns[0].observation is not None
    assert run.turns[0].observation.retryable is True


def test_react_tool_observation_requires_user_creates_hitl_wait_metadata() -> None:
    observation = ToolObservation(
        tool_name="agentic_rag_search",
        success=False,
        requires_user=True,
        user_prompt="Please clarify the policy scope.",
        result_summary="RAG needs clarification.",
        trace={"retrieval_trace": {"final_decision": "ask_user"}},
    )
    rag_tool = _StaticObservationTool(observation)
    executor = ToolExecutor(
        tools={"agentic_rag_search": rag_tool},
        allowed_tools={"agentic_rag_search"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="agentic_rag_search",
            input={"query": "policy"},
        ),
        ReActAction(action_type="final_answer"),
    )

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-hitl-tool",
        request_id="request-hitl-tool",
        user_goal="Find policy evidence.",
        react_run_id="react-hitl-tool",
    )

    assert run.workflow_status == "waiting_user"
    assert len(run.turns) == 1
    assert run.current_turn_id == "turn-1"
    assert run.turns[0].status == "waiting_user"
    assert run.turns[0].observation is not None
    assert run.turns[0].observation.requires_user is True
    assert run.turns[0].observation.metadata["hitl"] == {
        "mode": "react",
        "react_run_id": "react-hitl-tool",
        "current_turn_id": "turn-1",
        "user_prompt": "Please clarify the policy scope.",
        "source": "tool_observation",
    }
    assert run.metadata["hitl"] == run.turns[0].observation.metadata["hitl"]
    assert rag_tool.calls == [{"query": "policy", "limit": 1}]


def test_react_ask_user_action_creates_hitl_wait_metadata() -> None:
    executor = ToolExecutor(tools={}, allowed_tools=set())
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="ask_user",
            instruction="Which policy year should I use?",
        ),
        ReActAction(action_type="final_answer"),
    )

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-hitl",
        request_id="request-hitl",
        user_goal="Lookup the policy.",
        react_run_id="react-hitl",
    )

    assert run.workflow_status == "waiting_user"
    assert len(run.turns) == 1
    assert len(selector.contexts) == 1
    assert run.current_turn_id == "turn-1"
    assert run.turns[0].status == "waiting_user"
    assert run.turns[0].observation is None
    assert run.turns[0].result_summary == "Which policy year should I use?"
    assert run.metadata["hitl"] == {
        "mode": "react",
        "react_run_id": "react-hitl",
        "current_turn_id": "turn-1",
        "user_prompt": "Which policy year should I use?",
        "source": "react_action",
    }
    assert run.turns[0].metadata["hitl"] == run.metadata["hitl"]


def _structured_tool(
    *,
    name: str,
    calls: list[dict[str, Any]],
    records: list[dict[str, Any]],
    citations: list[dict[str, Any]] | None = None,
) -> StructuredTool:
    def invoke_tool(query: str, limit: int = 1) -> ToolResult:
        calls.append({"query": query, "limit": limit})
        return ToolResult.ok(
            tool_name=name,
            records=records,
            citations=citations or [],
        )

    return StructuredTool.from_function(
        func=invoke_tool,
        name=name,
        description=f"Run {name}.",
        args_schema=_QueryArgs,
    )
