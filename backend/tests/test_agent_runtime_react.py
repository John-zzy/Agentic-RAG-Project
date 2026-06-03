from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.platform.agent_runtime.contracts import (
    ReActAction,
    ReActRun,
    ReActTurn,
    ToolObservation,
)
from backend.platform.agent_runtime.react import (
    LLMReActActionSelector,
    ReActActionContext,
    ReActScenePolicy,
    ReActSelectorOutputError,
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


class _RecordingSelectionModel:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.prompt_template: Any | None = None
        self.inputs: list[Any] = []

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ) -> object:
        del complexity, output_parser
        self.prompt_template = prompt_template
        return object()

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> Any:
        del runnable, config
        self.inputs.append(input)
        if callable(self.response):
            return self.response(input)
        return self.response


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


def test_llm_react_selector_uses_public_turn_and_observation_summaries() -> None:
    model = _RecordingSelectionModel(
        '{"action_type":"final_answer","rationale_summary":"已有足够公开证据。"}'
    )
    selector = LLMReActActionSelector(model_client=model)
    previous_turn = ReActTurn(
        turn_id="turn-1",
        round_index=1,
        goal="Find policy evidence.",
        action=ReActAction(
            action_type="tool_call",
            tool_name="lookup_policy",
            input={"query": "travel policy", "limit": 2},
            rationale_summary="先查制度。",
            metadata={"private_note": "do not expose"},
        ),
        status="succeeded",
        input={"query": "travel policy", "limit": 2},
        result_summary="lookup_policy succeeded with 2 record(s).",
        metadata={"private_note": "do not expose"},
    )
    observation = ToolObservation(
        tool_name="lookup_policy",
        success=True,
        output={"records": [{"id": "doc-1"}], "secret": "do not expose"},
        result_summary="lookup_policy succeeded with 2 record(s).",
        citations=[{"citation_id": "doc-1"}],
        trace={"retrieval_trace": {"final_decision": "answer_with_evidence"}},
        metadata={"internal": "do not expose"},
    )

    action = selector.select_action(
        ReActActionContext(
            react_run_id="react-1",
            session_id="session-1",
            request_id="request-1",
            user_goal="Find policy evidence.",
            round_index=2,
            max_turns=3,
            allowed_tools=["lookup_policy", "lookup_inventory"],
            previous_turns=[previous_turn],
            run_observations=[observation],
            attempted_tools=["lookup_policy"],
            latest_final_decision="answer_with_evidence",
            scene_policy=ReActScenePolicy(
                preferred_tools=["lookup_policy"],
                max_turns=3,
                no_evidence_action="ask_user",
            ),
            resume_metadata={"source": "user_reply"},
        )
    )

    assert action.action_type == "final_answer"
    assert "REACT_SELECTOR" in str(getattr(model.prompt_template, "template", ""))
    payload = model.inputs[0]
    previous_turns = json.loads(payload["react_previous_turns_json"])
    observations = json.loads(payload["react_run_observations_json"])
    assert previous_turns == [
        {
            "turn_id": "turn-1",
            "round_index": 1,
            "action_type": "tool_call",
            "tool_name": "lookup_policy",
            "status": "succeeded",
            "result_summary": "lookup_policy succeeded with 2 record(s).",
            "error": None,
        }
    ]
    assert observations == [
        {
            "tool_name": "lookup_policy",
            "success": True,
            "result_summary": "lookup_policy succeeded with 2 record(s).",
            "retryable": False,
            "requires_user": False,
            "final_decision": "answer_with_evidence",
            "error": None,
        }
    ]
    assert "secret" not in payload["react_run_observations_json"]
    assert "private_note" not in payload["react_previous_turns_json"]


def test_llm_react_selector_rejects_invalid_json_output() -> None:
    selector = LLMReActActionSelector(model_client=_RecordingSelectionModel("{invalid json"))

    with pytest.raises(ReActSelectorOutputError, match="valid JSON object"):
        selector.select_action(
            ReActActionContext(
                react_run_id="react-1",
                session_id="session-1",
                request_id="request-1",
                user_goal="Find policy evidence.",
                round_index=1,
                max_turns=2,
            )
        )


def test_llm_react_selector_rejects_hidden_reasoning_fields() -> None:
    selector = LLMReActActionSelector(
        model_client=_RecordingSelectionModel(
            json.dumps(
                {
                    "action_type": "final_answer",
                    "rationale_summary": "公开摘要。",
                    "thought": "private chain of thought",
                },
                ensure_ascii=False,
            )
        )
    )

    with pytest.raises(ReActSelectorOutputError, match="thought"):
        selector.select_action(
            ReActActionContext(
                react_run_id="react-1",
                session_id="session-1",
                request_id="request-1",
                user_goal="Find policy evidence.",
                round_index=1,
                max_turns=2,
            )
        )


def test_llm_react_selector_supports_tool_call_ask_user_and_multi_tool_progression() -> None:
    selector_model = _RecordingSelectionModel(
        lambda payload: json.dumps(
            {
                "action_type": "ask_user",
                "instruction": "请补充制度年份。",
                "rationale_summary": "信息不足，先询问用户。",
            },
            ensure_ascii=False,
        )
        if payload["react_round_index"] == "1"
        else json.dumps(
            {
                "action_type": "final_answer",
                "rationale_summary": "已有足够信息。",
            },
            ensure_ascii=False,
        )
    )
    selector = LLMReActActionSelector(model_client=selector_model)

    ask_user = selector.select_action(
        ReActActionContext(
            react_run_id="react-llm-ask-user",
            session_id="session-llm",
            request_id="request-llm",
            user_goal="查询制度。",
            round_index=1,
            max_turns=3,
            allowed_tools=["lookup_policy"],
        )
    )

    assert ask_user.action_type == "ask_user"
    assert ask_user.instruction == "请补充制度年份。"
    assert ask_user.rationale_summary == "信息不足，先询问用户。"

    policy_calls: list[dict[str, Any]] = []
    inventory_calls: list[dict[str, Any]] = []
    runtime_model = _RecordingSelectionModel(
        lambda payload: json.dumps(
            {
                "action_type": "tool_call",
                "tool_name": "lookup_policy",
                "input": {"query": "return policy"},
                "rationale_summary": "首轮先查制度。",
            },
            ensure_ascii=False,
        )
        if payload["react_round_index"] == "1"
        else json.dumps(
            {
                "action_type": "tool_call",
                "tool_name": "lookup_inventory",
                "input": {"query": "sku-1"},
                "rationale_summary": "制度已查，继续查库存。",
            },
            ensure_ascii=False,
        )
        if payload["react_round_index"] == "2"
        else json.dumps(
            {
                "action_type": "final_answer",
                "rationale_summary": "两类证据已经齐全。",
            },
            ensure_ascii=False,
        )
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(
            tools={
                "lookup_policy": _structured_tool(
                    name="lookup_policy",
                    calls=policy_calls,
                    records=[{"policy": "return-window"}],
                ),
                "lookup_inventory": _structured_tool(
                    name="lookup_inventory",
                    calls=inventory_calls,
                    records=[{"sku": "sku-1"}],
                ),
            },
            allowed_tools={"lookup_policy", "lookup_inventory"},
        ),
        action_selector=LLMReActActionSelector(model_client=runtime_model),
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )

    run = runtime.run(
        session_id="session-llm-runtime",
        request_id="request-llm-runtime",
        user_goal="同时检查退货制度和库存。",
        react_run_id="react-llm-runtime",
    )

    assert run.workflow_status == "succeeded"
    assert [turn.tool_name for turn in run.turns] == [
        "lookup_policy",
        "lookup_inventory",
        None,
    ]
    assert policy_calls == [{"query": "return policy", "limit": 1}]
    assert inventory_calls == [{"query": "sku-1", "limit": 1}]
    assert len(runtime_model.inputs) == 3
    assert runtime_model.inputs[1]["react_attempted_tools_json"] == '["lookup_policy"]'
    assert runtime_model.inputs[2]["react_attempted_tools_json"] == (
        '["lookup_policy", "lookup_inventory"]'
    )
    assert run.metadata["latest_action_selection"]["validation_result"] == "passed"
    assert run.metadata["attempted_tools"] == ["lookup_policy", "lookup_inventory"]


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
    assert run.observations == [run.turns[0].observation]
    assert tool_calls == [{"query": "travel reimbursement", "limit": 1}]
    assert selector.contexts[0].allowed_tools == ["lookup_policy"]
    assert selector.contexts[1].previous_turns[0].turn_id == "turn-1"
    assert selector.contexts[1].run_observations == run.observations
    assert selector.contexts[1].attempted_tools == ["lookup_policy"]
    assert selector.contexts[1].scene_policy.max_turns == run.max_turns
    assert synthesizer.contexts[0].observations == run.observations
    assert synthesizer.contexts[0].turn_order == ["turn-1", "turn-2"]
    assert [transition["event"] for transition in run.metadata["workflow_transitions"]] == [
        "run_start",
        "success",
    ]


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
    assert run.observations == [
        run.turns[0].observation,
        run.turns[1].observation,
    ]
    assert len(synthesizer.contexts[0].turns) == 3
    assert len(synthesizer.contexts[0].observations) == 2
    assert selector.contexts[2].previous_turns[1].tool_name == "lookup_inventory"
    assert selector.contexts[2].run_observations == run.observations
    assert selector.contexts[2].attempted_tools == ["lookup_policy", "lookup_inventory"]


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
    assert selector.contexts[1].latest_final_decision == "answer_with_evidence"
    retrieval_trace = run.turns[0].observation.trace["retrieval_trace"]
    assert [round_trace["round_index"] for round_trace in retrieval_trace["rounds"]] == [1, 2]
    assert "turn_id" not in retrieval_trace["rounds"][0]
    assert "react_turns" not in run.turns[0].observation.trace
    assert run.turns[1].action.action_type == "final_answer"
    assert rag_tool.calls == [{"query": "policy evidence", "limit": 1}]
    assert run.metadata["citations"] == [{"citation_id": "doc-1"}]
    assert run.observations == [run.turns[0].observation]


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


def test_react_retries_retryable_tool_failure_and_then_succeeds() -> None:
    calls: list[dict[str, Any]] = []

    def flaky_lookup(query: str, limit: int = 1) -> ToolResult:
        calls.append({"query": query, "limit": limit})
        if len(calls) == 1:
            raise TimeoutError(f"timeout for {query}")
        return ToolResult.ok(tool_name="lookup_policy", records=[{"policy": "ok"}])

    executor = ToolExecutor(
        tools={
            "lookup_policy": StructuredTool.from_function(
                func=flaky_lookup,
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
        request_id="request-retry-success",
        user_goal="Lookup policy.",
        react_run_id="react-retry-success",
    )

    assert run.workflow_status == "succeeded"
    assert calls == [
        {"query": "expense policy", "limit": 1},
        {"query": "expense policy", "limit": 1},
    ]
    assert run.turns[0].status == "succeeded"
    assert run.turns[0].retry_metadata.attempt == 2
    assert len(run.observations) == 2
    assert run.observations[0].retryable is True
    assert run.observations[1].success is True
    assert run.metadata["retry"] == {
        "attempt": 1,
        "max_attempts": 2,
        "latest_error": "timeout for expense policy",
        "current_turn_id": "turn-1",
        "tool_name": "lookup_policy",
        "tool_call_id": run.observations[0].tool_call_id,
        "retryable": True,
    }
    assert run.metadata["retry_history"] == [run.metadata["retry"]]
    assert [transition["event"] for transition in run.metadata["workflow_transitions"]] == [
        "run_start",
        "tool_error_retryable",
        "retry",
        "success",
    ]


def test_react_successful_retryable_observation_does_not_record_retry_metadata() -> None:
    observation = ToolObservation(
        tool_name="agentic_rag_search",
        success=True,
        retryable=True,
        result_summary="agentic_rag_search succeeded with retryable hint.",
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
        session_id="session-retryable-success",
        request_id="request-retryable-success",
        user_goal="Find evidence.",
        react_run_id="react-retryable-success",
    )

    assert run.workflow_status == "succeeded"
    assert run.observations[0].success is True
    assert run.observations[0].retryable is True
    assert "retry" not in run.metadata
    assert "retry_history" not in run.metadata


def test_react_fails_after_retry_exhaustion() -> None:
    calls: list[dict[str, Any]] = []

    def timeout_lookup(query: str, limit: int = 1) -> ToolResult:
        calls.append({"query": query, "limit": limit})
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

    assert run.workflow_status == "failed"
    assert len(run.turns) == 1
    assert len(selector.contexts) == 1
    assert run.current_turn_id == "turn-1"
    assert run.current_tool_call is None
    assert run.error == "timeout for expense policy"
    assert run.turns[0].status == "failed"
    assert run.turns[0].retry_metadata.attempt == 2
    assert run.turns[0].retry_metadata.retryable is True
    assert run.turns[0].retry_metadata.last_error == "timeout for expense policy"
    assert run.turns[0].observation is not None
    assert run.turns[0].observation.retryable is True
    assert len(run.observations) == 2
    assert run.observations[-1] == run.turns[0].observation
    assert run.error == run.observations[-1].error
    assert run.metadata["retry"] == {
        "attempt": 2,
        "max_attempts": 2,
        "latest_error": "timeout for expense policy",
        "current_turn_id": "turn-1",
        "tool_name": "lookup_policy",
        "tool_call_id": run.observations[-1].tool_call_id,
        "retryable": True,
    }
    assert [entry["attempt"] for entry in run.metadata["retry_history"]] == [1, 2]
    assert calls == [
        {"query": "expense policy", "limit": 1},
        {"query": "expense policy", "limit": 1},
    ]
    assert [transition["event"] for transition in run.metadata["workflow_transitions"]] == [
        "run_start",
        "tool_error_retryable",
        "retry",
        "tool_error_retryable",
        "tool_error_final",
    ]


def test_react_selector_tool_call_is_validated_before_tool_execution() -> None:
    calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "lookup_policy": _structured_tool(
                name="lookup_policy",
                calls=calls,
                records=[{"policy": "travel"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="lookup_policy",
            input={"query": ""},
        ),
    )

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-1",
        request_id="request-invalid-input",
        user_goal="Lookup policy.",
        react_run_id="react-invalid-input",
    )

    assert run.workflow_status == "failed"
    assert calls == []
    assert run.turns == []
    assert run.observations == []
    assert run.metadata["action_selection_audits"] == [
        {
            "round_index": 1,
            "selector_attempt": 1,
            "status": "failed",
            "action_type": None,
            "tool_name": None,
            "rationale_summary": None,
            "error": "Invalid input for tool: lookup_policy.",
        }
    ]
    assert "Invalid input for tool: lookup_policy." in run.error


def test_react_selector_rejects_disallowed_tool_before_execution() -> None:
    calls: list[dict[str, Any]] = []
    executor = ToolExecutor(
        tools={
            "unsafe_tool": _structured_tool(
                name="unsafe_tool",
                calls=calls,
                records=[{"policy": "unsafe"}],
            )
        },
        allowed_tools={"lookup_policy"},
    )
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="tool_call",
            tool_name="unsafe_tool",
            input={"query": "policy"},
        ),
    )

    run = ReActRuntime(
        tool_executor=executor,
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    ).run(
        session_id="session-1",
        request_id="request-disallowed-tool",
        user_goal="Lookup policy.",
        react_run_id="react-disallowed-tool",
    )

    assert run.workflow_status == "failed"
    assert calls == []
    assert run.turns == []
    assert run.observations == []
    assert run.metadata["action_selection_audits"] == [
        {
            "round_index": 1,
            "selector_attempt": 1,
            "status": "failed",
            "action_type": None,
            "tool_name": None,
            "rationale_summary": None,
            "error": "Tool is not allowed: unsafe_tool.",
        }
    ]
    assert run.error == "Tool is not allowed: unsafe_tool."


def test_react_runtime_retries_invalid_selector_output_then_waits_user() -> None:
    model = _RecordingSelectionModel("{invalid json")
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=LLMReActActionSelector(model_client=model),
        scene_policy=ReActScenePolicy(no_evidence_action="ask_user", max_turns=2),
        selector_retry_budget=1,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )

    run = runtime.run(
        session_id="session-invalid-selector",
        request_id="request-invalid-selector",
        user_goal="Lookup policy.",
        react_run_id="react-invalid-selector",
    )

    assert run.workflow_status == "waiting_user"
    assert len(model.inputs) == 2
    assert len(run.turns) == 1
    assert run.turns[0].action.action_type == "ask_user"
    assert run.turns[0].action.metadata["selector_failure"] == {
        "round_index": 1,
        "attempts": 2,
        "error": "selector output must be a valid JSON object.",
        "retry_budget": 1,
    }
    assert run.metadata["latest_selector_failure"] == run.turns[0].action.metadata["selector_failure"]
    assert [audit["status"] for audit in run.metadata["action_selection_audits"]] == [
        "failed",
        "failed",
    ]


def test_react_runtime_fails_after_invalid_selector_output_when_ask_user_disabled() -> None:
    model = _RecordingSelectionModel(
        json.dumps({"action_type": "unsupported", "rationale_summary": "bad"}, ensure_ascii=False)
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=LLMReActActionSelector(model_client=model),
        scene_policy=ReActScenePolicy(no_evidence_action="final_answer", max_turns=2),
        selector_retry_budget=0,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )

    run = runtime.run(
        session_id="session-invalid-selector-failed",
        request_id="request-invalid-selector-failed",
        user_goal="Lookup policy.",
        react_run_id="react-invalid-selector-failed",
    )

    assert run.workflow_status == "failed"
    assert run.turns == []
    assert run.observations == []
    assert len(model.inputs) == 1
    assert run.error == (
        "ReAct selector output is invalid: action_type: Input should be 'tool_call', "
        "'ask_user', 'final_answer' or 'stop'."
    )


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
    assert run.observations == [run.turns[0].observation]
    assert run.observations[0].metadata["hitl"] == run.metadata["hitl"]
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


def test_react_respond_continuation_resumes_same_run_with_user_context() -> None:
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="ask_user",
            instruction="Which policy year should I use?",
        ),
        ReActAction(action_type="final_answer"),
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )
    run = runtime.run(
        session_id="session-respond",
        request_id="request-respond",
        user_goal="Lookup the policy.",
        react_run_id="react-respond",
    )

    resumed = runtime.continue_after_respond(
        run=run,
        response="Use the 2026 policy.",
        source="suggested_response",
        suggestion_id="policy_2026",
        metadata={"operator": "user-1"},
    )

    assert resumed is run
    assert run.workflow_status == "succeeded"
    assert run.react_run_id == "react-respond"
    assert [turn.turn_id for turn in run.turns] == ["turn-1", "turn-2"]
    assert run.turns[0].status == "succeeded"
    assert run.turns[0].metadata["continuation"]["waiting_turn_id"] == "turn-1"
    assert run.metadata["resume"] == {
        "mode": "react",
        "action": "respond",
        "react_run_id": "react-respond",
        "waiting_turn_id": "turn-1",
        "continued_from_turn_id": "turn-1",
        "metadata": {"operator": "user-1"},
        "response": "Use the 2026 policy.",
        "source": "suggested_response",
        "suggestion_id": "policy_2026",
    }
    assert selector.contexts[1].resume_metadata == run.metadata["resume"]
    assert [transition["event"] for transition in run.metadata["workflow_transitions"]] == [
        "run_start",
        "interrupt",
        "resume_respond",
        "success",
    ]


def test_react_respond_continuation_extends_budget_when_waiting_on_last_turn() -> None:
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="ask_user",
            instruction="Which policy year should I use?",
        ),
        ReActAction(action_type="final_answer"),
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=selector,
        max_turns=1,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )
    run = runtime.run(
        session_id="session-respond-budget",
        request_id="request-respond-budget",
        user_goal="Lookup the policy.",
        react_run_id="react-respond-budget",
    )

    runtime.continue_after_respond(run=run, response="Use the 2026 policy.")

    assert run.workflow_status == "succeeded"
    assert run.max_turns == 2
    assert len(selector.contexts) == 2
    assert selector.contexts[1].resume_metadata["response"] == "Use the 2026 policy."
    assert selector.contexts[1].max_turns == 2
    assert run.metadata["resume"]["budget_extension"] == {
        "reason": "waiting_user_continuation",
        "previous_max_turns": 1,
        "extended_max_turns": 2,
    }


def test_react_approve_continuation_preserves_pending_tool_context() -> None:
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="ask_user",
            instruction="Approve this write operation?",
        ),
        ReActAction(action_type="final_answer"),
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )
    run = runtime.run(
        session_id="session-approve",
        request_id="request-approve",
        user_goal="Update the policy note.",
        react_run_id="react-approve",
    )
    pending_tool_call = {
        "tool_name": "write_policy_note",
        "args": {"item_id": "policy-1", "content": "approved"},
    }

    runtime.continue_after_approve(
        run=run,
        approval_result={"approved_by": "user-1"},
        pending_tool_call=pending_tool_call,
    )

    assert run.workflow_status == "succeeded"
    assert run.metadata["resume"] == {
        "mode": "react",
        "action": "approve",
        "react_run_id": "react-approve",
        "waiting_turn_id": "turn-1",
        "continued_from_turn_id": "turn-1",
        "metadata": {},
        "approved": True,
        "approval_result": {"approved_by": "user-1"},
        "pending_tool_call": pending_tool_call,
    }
    assert selector.contexts[1].resume_metadata["pending_tool_call"] == pending_tool_call
    assert run.turns[0].metadata["continuation"]["approved"] is True
    assert [transition["event"] for transition in run.metadata["workflow_transitions"]] == [
        "run_start",
        "interrupt",
        "resume_approve",
        "success",
    ]


def test_react_approve_continuation_extends_budget_when_waiting_on_last_turn() -> None:
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="ask_user",
            instruction="Approve this write operation?",
        ),
        ReActAction(action_type="final_answer"),
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=selector,
        max_turns=1,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )
    run = runtime.run(
        session_id="session-approve-budget",
        request_id="request-approve-budget",
        user_goal="Update the policy note.",
        react_run_id="react-approve-budget",
    )

    runtime.continue_after_approve(
        run=run,
        pending_tool_call={"tool_name": "write_policy_note", "args": {"item_id": "policy-1"}},
    )

    assert run.workflow_status == "succeeded"
    assert run.max_turns == 2
    assert len(selector.contexts) == 2
    assert selector.contexts[1].resume_metadata["approved"] is True
    assert run.metadata["resume"]["budget_extension"] == {
        "reason": "waiting_user_continuation",
        "previous_max_turns": 1,
        "extended_max_turns": 2,
    }


def test_react_reject_continuation_cancels_run_and_skips_pending_side_effect() -> None:
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="ask_user",
            instruction="Approve this external call?",
        ),
        ReActAction(action_type="final_answer"),
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )
    run = runtime.run(
        session_id="session-reject",
        request_id="request-reject",
        user_goal="Call external API.",
        react_run_id="react-reject",
    )
    pending_tool_call = {
        "tool_name": "external_api_call",
        "args": {"endpoint": "https://example.test/webhook"},
    }

    cancelled = runtime.continue_after_reject(
        run=run,
        reason="User rejected the external call.",
        pending_tool_call=pending_tool_call,
    )

    assert cancelled is run
    assert run.workflow_status == "cancelled"
    assert run.current_turn_id is None
    assert run.current_tool_call is None
    assert run.turns[0].status == "cancelled"
    assert run.result_summary == "User rejected the external call."
    assert run.metadata["resume"] == {
        "mode": "react",
        "action": "reject",
        "react_run_id": "react-reject",
        "waiting_turn_id": "turn-1",
        "continued_from_turn_id": "turn-1",
        "metadata": {},
        "reason": "User rejected the external call.",
        "pending_tool_call": pending_tool_call,
        "side_effect_executed": False,
    }
    assert len(selector.contexts) == 1
    assert [transition["event"] for transition in run.metadata["workflow_transitions"]] == [
        "run_start",
        "interrupt",
        "resume_reject",
    ]


def test_react_reject_continuation_uses_default_reason_for_blank_text() -> None:
    selector = _SequencedActionSelector(
        ReActAction(
            action_type="ask_user",
            instruction="Approve this external call?",
        ),
    )
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=selector,
        turn_id_factory=lambda round_index: f"turn-{round_index}",
    )
    run = runtime.run(
        session_id="session-reject-blank",
        request_id="request-reject-blank",
        user_goal="Call external API.",
        react_run_id="react-reject-blank",
    )

    runtime.continue_after_reject(run=run, reason="   ")

    assert run.workflow_status == "cancelled"
    assert run.result_summary == "User rejected the waiting ReAct turn."
    assert run.metadata["resume"]["reason"] == "User rejected the waiting ReAct turn."


def test_react_runtime_rejects_direct_continue_for_waiting_or_terminal_run() -> None:
    runtime = ReActRuntime(
        tool_executor=ToolExecutor(tools={}, allowed_tools=set()),
        action_selector=_SequencedActionSelector(ReActAction(action_type="final_answer")),
    )
    waiting_run = ReActRun(
        react_run_id="react-waiting",
        session_id="session-1",
        request_id="request-waiting",
        user_goal="waiting",
        workflow_status="waiting_user",
    )
    terminal_run = ReActRun(
        react_run_id="react-terminal",
        session_id="session-1",
        request_id="request-terminal",
        user_goal="terminal",
        workflow_status="succeeded",
    )

    with pytest.raises(ValueError, match="waiting for user"):
        runtime.continue_run(waiting_run)
    with pytest.raises(ValueError, match="already terminal"):
        runtime.continue_run(terminal_run)


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
