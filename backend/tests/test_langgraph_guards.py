from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from backend.platform.agent_runtime.quality.failures import (
    FailureCategory,
    build_failure_record,
    failure_record_from_payload,
)
from backend.platform.agent_runtime.core.contracts import PlanRun, ReActRun
from backend.platform.agent_runtime.chat_graph.graph import graph as chat_graph_module
from backend.platform.agent_runtime.plan.graph import graph as plan_graph_module
from backend.platform.rag.orchestration.retrieval_graph import graph as rag_graph_module
from backend.platform.workflow.langgraph.guards import (
    GuardedNodeFailureError,
    GuardTimeoutConfig,
    RetryPolicyConfig,
    build_guard_metadata,
    build_guarded_node_config,
    build_timeout_policy,
    extract_guard_failures,
    register_guarded_node,
)


class _GuardState(TypedDict, total=False):
    session_id: str
    request_id: str
    run_id: str
    current_turn_id: str
    current_step_id: str
    current_tool_call: dict[str, Any]
    metadata: dict[str, Any]
    value: str


class _ReActGuardState(TypedDict, total=False):
    run: ReActRun
    error: str


class _PlanGuardState(TypedDict, total=False):
    plan_run: PlanRun
    error: str


def test_guarded_node_retries_retryable_exception_and_clears_transient_failure() -> None:
    attempts = 0

    def flaky_node(state: _GuardState) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary timeout")
        return {"value": "ok"}

    builder = StateGraph(_GuardState)
    register_guarded_node(
        builder,
        "flaky",
        flaky_node,
        graph_name="test_graph",
        retry_config=RetryPolicyConfig(max_attempts=2, initial_interval=0.01),
    )
    builder.add_edge(START, "flaky")
    builder.add_edge("flaky", END)

    result = builder.compile().invoke(
        {"session_id": "session-1", "request_id": "req-1", "metadata": {}}
    )

    assert attempts == 2
    assert result["value"] == "ok"
    assert extract_guard_failures(result) == []
    assert "_guard_last_failure" not in result["metadata"]


def test_guarded_node_error_handler_raises_non_retryable_failure_payload() -> None:
    def broken_node(state: _GuardState) -> dict[str, Any]:
        del state
        raise ValueError("invalid action")

    builder = StateGraph(_GuardState)
    register_guarded_node(
        builder,
        "broken",
        broken_node,
        graph_name="react_graph",
        source="runtime",
        retry_config=RetryPolicyConfig(max_attempts=2, initial_interval=0.01),
        metadata={"phase": "select_action"},
    )
    builder.add_edge(START, "broken")
    builder.add_edge("broken", END)

    with pytest.raises(GuardedNodeFailureError) as exc_info:
        builder.compile().invoke(
            {
                "session_id": "session-2",
                "request_id": "req-2",
                "run_id": "run-2",
                "current_turn_id": "turn-1",
                "metadata": {},
            }
        )

    failure = failure_record_from_payload(exc_info.value.failure_payload)
    assert failure.category == FailureCategory.RUNTIME_ERROR
    assert failure.retryable is False
    assert failure.message == "invalid action"
    assert failure.graph_name == "react_graph"
    assert failure.node_name == "broken"
    assert failure.request_id == "req-2"
    assert failure.run_id == "run-2"
    assert failure.session_id == "session-2"
    assert failure.turn_id == "turn-1"
    assert failure.metadata == {"phase": "select_action"}


def test_guard_error_handler_raises_failure_for_react_run() -> None:
    def broken_node(state: _ReActGuardState) -> dict[str, Any]:
        del state
        raise RuntimeError("selector failed")

    builder = StateGraph(_ReActGuardState)
    register_guarded_node(
        builder,
        "select_action",
        broken_node,
        graph_name="react_graph",
        retry_config=RetryPolicyConfig(max_attempts=1),
    )
    builder.add_edge(START, "select_action")
    builder.add_edge("select_action", END)

    with pytest.raises(GuardedNodeFailureError) as exc_info:
        builder.compile().invoke(
            {
                "run": ReActRun(
                    react_run_id="react-1",
                    session_id="session-react",
                    request_id="req-react",
                    user_goal="查政策",
                )
            }
        )

    failure = failure_record_from_payload(exc_info.value.failure_payload)
    assert failure.graph_name == "react_graph"
    assert failure.node_name == "select_action"
    assert failure.category == FailureCategory.RUNTIME_ERROR


def test_chat_graph_guarded_node_raises_failure_for_runtime_path() -> None:
    builder = StateGraph(_GuardState)
    chat_graph_module._add_guarded_logged_node(  # noqa: SLF001
        builder,
        "final_synthesis",
        lambda state: (_ for _ in ()).throw(ValueError("answer failed")),
    )
    builder.add_edge(START, "final_synthesis")
    builder.add_edge("final_synthesis", END)

    with pytest.raises(GuardedNodeFailureError) as exc_info:
        builder.compile().invoke(
            {"session_id": "session-chat", "request_id": "req-chat", "metadata": {}}
        )

    failure = failure_record_from_payload(exc_info.value.failure_payload)
    assert failure.graph_name == chat_graph_module.CHAT_GRAPH_NAME
    assert failure.node_name == "final_synthesis"
    assert failure.category == FailureCategory.RUNTIME_ERROR
    assert failure.metadata == {"guard_scope": "chat_graph"}


def test_plan_graph_guarded_node_raises_failure_for_plan_run() -> None:
    builder = StateGraph(_PlanGuardState)
    plan_graph_module._add_guarded_logged_node(  # noqa: SLF001
        builder,
        plan_graph_module.EXECUTE_STEP,
        lambda state: (_ for _ in ()).throw(ValueError("tool turn failed")),
    )
    builder.add_edge(START, plan_graph_module.EXECUTE_STEP)
    builder.add_edge(plan_graph_module.EXECUTE_STEP, END)

    with pytest.raises(GuardedNodeFailureError) as exc_info:
        builder.compile().invoke(
            {
                "plan_run": PlanRun(
                    plan_run_id="plan-guard",
                    session_id="session-plan",
                    request_id="req-plan",
                    user_goal="查制度",
                )
            }
        )

    failure = failure_record_from_payload(exc_info.value.failure_payload)
    assert failure.graph_name == plan_graph_module.PLAN_GRAPH_NAME
    assert failure.node_name == plan_graph_module.EXECUTE_STEP
    assert failure.category == FailureCategory.TOOL_ERROR
    assert failure.run_id == "plan-guard"
    assert failure.session_id == "session-plan"
    assert failure.request_id == "req-plan"


def test_agentic_rag_guarded_node_raises_retrieval_failure() -> None:
    builder = StateGraph(_GuardState)
    rag_graph_module._add_guarded_node(  # noqa: SLF001
        builder,
        rag_graph_module.RETRIEVAL,
        lambda state: (_ for _ in ()).throw(ValueError("retrieval failed")),
    )
    builder.add_edge(START, rag_graph_module.RETRIEVAL)
    builder.add_edge(rag_graph_module.RETRIEVAL, END)

    with pytest.raises(GuardedNodeFailureError) as exc_info:
        builder.compile().invoke({"metadata": {}})

    failure = failure_record_from_payload(exc_info.value.failure_payload)
    assert failure.graph_name == rag_graph_module.AGENTIC_RAG_GRAPH_NAME
    assert failure.node_name == rag_graph_module.RETRIEVAL
    assert failure.category == FailureCategory.RETRIEVAL_ERROR


def test_required_graph_nodes_are_guarded_by_builder_helpers() -> None:
    assert set(chat_graph_module.GUARDED_CHAT_NODES) == {
        chat_graph_module.REACT_BRANCH,
        chat_graph_module.PLAN_BRANCH,
        "self_check_guard",
        "final_synthesis",
        "persist_turn",
    }
    assert set(plan_graph_module.GUARDED_PLAN_NODES) == {
        plan_graph_module.CREATE_PLAN,
        plan_graph_module.EXECUTE_STEP,
        plan_graph_module.HANDLE_RETRY,
        plan_graph_module.SYNTHESIZE_PLAN_RESULT,
    }
    assert set(rag_graph_module.GUARDED_AGENTIC_RAG_NODES) == {
        rag_graph_module.RETRIEVAL,
        rag_graph_module.QUERY_REWRITE,
        rag_graph_module.SUFFICIENCY_CHECK,
        rag_graph_module.FINAL_EVIDENCE_SYNTHESIS,
    }


def test_timeout_failure_record_and_timeout_policy_are_stable() -> None:
    record = build_failure_record(
        NodeTimeoutError("retrieval", 3.5, kind="run", run_timeout=3.0),
        source="retrieval",
        graph_name="rag_graph",
        node_name="retrieval",
    )
    timeout = build_timeout_policy(
        GuardTimeoutConfig(run_timeout_seconds=3.5, idle_timeout_seconds=1.0)
    )

    assert record.category == FailureCategory.RETRIEVAL_TIMEOUT
    assert record.retryable is True
    assert timeout is not None
    assert timeout.run_timeout.total_seconds() == 3.5
    assert timeout.idle_timeout.total_seconds() == 1.0
    assert timeout.refresh_on == "auto"


def test_guard_metadata_and_node_config_are_json_serializable() -> None:
    metadata = build_guard_metadata(
        graph_name="chat_graph",
        node_name="final_synthesis",
        source="model",
        request_id="req-meta",
        metadata={"answer_mode": "evidence_answer"},
    )
    config = build_guarded_node_config(
        graph_name="chat_graph",
        node_name="final_synthesis",
        node=lambda state: state,
        source="model",
        metadata={"answer_mode": "evidence_answer"},
    )

    encoded_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    encoded_node_metadata = json.dumps(
        config.metadata,
        ensure_ascii=False,
        sort_keys=True,
    )

    assert "\"graph_name\": \"chat_graph\"" in encoded_metadata
    assert "\"node_name\": \"final_synthesis\"" in encoded_node_metadata
    assert config.retry_policy.max_attempts == 2
    assert config.error_handler is not None
