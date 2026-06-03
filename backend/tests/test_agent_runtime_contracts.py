from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel, Field

from backend.platform import agent_runtime
from backend.platform.agent_runtime import (
    AgentRun,
    PlanDependencyValidationError,
    PlanRun,
    PlanStep,
    ReActAction,
    ReActRun,
    ReActTurn,
    ToolAccessValidationError,
    ToolInputValidationError,
    ToolObservation,
    build_retry_metadata,
    ensure_tool_allowed,
    validate_plan_dependencies,
    validate_plan_tool_allowlist,
    validate_tool_input,
)


class _SearchArgs(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1)


def test_agent_runtime_package_exports_neutral_contracts() -> None:
    assert "backend.scenes" not in inspect.getsource(agent_runtime)
    assert "backend.application" not in inspect.getsource(agent_runtime)
    assert set(agent_runtime.__all__) >= {
        "AgentRun",
        "ToolObservation",
        "ReActRun",
        "ReActTurn",
        "PlanRun",
        "PlanStep",
        "validate_plan_dependencies",
        "validate_tool_input",
    }


def test_agent_run_and_tool_observation_are_serializable_contracts() -> None:
    observation = ToolObservation(
        tool_name="agentic_rag_search",
        success=True,
        output={"answer": "命中证据"},
        result_summary="RAG 工具返回了可引用证据。",
        citations=[{"citation_id": "chunk-1"}],
        trace={"retrieval_trace": {"rounds": [{"round_index": 1}]}},
    )
    run = AgentRun(
        agent_run_id="agent-run-1",
        session_id="session-1",
        request_id="request-1",
        mode="react",
        user_goal="查询知识库",
        workflow_status="running",
    )

    assert observation.model_dump()["trace"]["retrieval_trace"]["rounds"][0]["round_index"] == 1
    assert observation.model_dump()["citations"] == [{"citation_id": "chunk-1"}]
    assert run.model_dump() == {
        "agent_run_id": "agent-run-1",
        "session_id": "session-1",
        "request_id": "request-1",
        "mode": "react",
        "user_goal": "查询知识库",
        "workflow_status": "running",
        "current_tool_call": None,
        "final_answer": None,
        "result_summary": "",
        "error": None,
        "metadata": {},
    }


def test_react_contract_records_top_level_turn_without_hidden_reasoning() -> None:
    turn = ReActTurn(
        turn_id="turn-1",
        round_index=1,
        goal="回答制度问题",
        action=ReActAction(
            action_type="tool_call",
            tool_name="native_rag_search",
            input={"query": "报销制度"},
            rationale_summary="需要先检索制度证据。",
        ),
        input={"query": "报销制度"},
    )
    run = ReActRun(
        react_run_id="react-run-1",
        session_id="session-react",
        request_id="request-react",
        user_goal="报销制度是什么",
        turns=[turn],
        current_turn_id="turn-1",
    )

    payload = run.model_dump()

    assert payload["mode"] == "react"
    assert payload["turns"][0]["tool_name"] == "native_rag_search"
    assert payload["turns"][0]["action"]["rationale_summary"] == "需要先检索制度证据。"
    assert "thought" not in payload["turns"][0]["action"]


def test_plan_contract_records_required_step_fields() -> None:
    step = PlanStep(
        step_id="step-1",
        goal="检索相关制度",
        tool_name="native_rag_search",
        input={"query": "报销制度"},
        depends_on=[],
    )
    run = PlanRun(
        plan_run_id="plan-run-1",
        session_id="session-plan",
        request_id="request-plan",
        user_goal="先查制度再总结",
        steps=[step],
        current_step_id="step-1",
    )

    payload = run.model_dump()

    assert payload["mode"] == "plan"
    assert set(payload["steps"][0]) >= {
        "step_id",
        "goal",
        "tool_name",
        "input",
        "depends_on",
        "status",
        "result_summary",
        "error",
    }
    assert payload["steps"][0]["status"] == "pending"


def test_tool_allowlist_and_input_schema_validation() -> None:
    assert ensure_tool_allowed("native_rag_search", {"native_rag_search"}) == "native_rag_search"
    assert validate_tool_input(
        tool_name="native_rag_search",
        input_payload={"query": "制度", "top_k": 2},
        args_schema=_SearchArgs,
    ) == {"query": "制度", "top_k": 2}

    with pytest.raises(ToolAccessValidationError, match="not allowed"):
        ensure_tool_allowed("unsafe_tool", {"native_rag_search"})
    with pytest.raises(ToolInputValidationError, match="Invalid input"):
        validate_tool_input(
            tool_name="native_rag_search",
            input_payload={"query": "制度", "top_k": 0},
            args_schema=_SearchArgs,
        )


def test_plan_dependency_and_retry_metadata_validation() -> None:
    steps = [
        PlanStep(
            step_id="step-1",
            goal="检索资料",
            tool_name="native_rag_search",
        ),
        PlanStep(
            step_id="step-2",
            goal="汇总结果",
            tool_name="final_synthesizer",
            depends_on=["step-1"],
        ),
    ]

    validate_plan_tool_allowlist(steps, {"native_rag_search", "final_synthesizer"})
    validate_plan_dependencies(steps)

    retry_metadata = build_retry_metadata(last_error="timeout")
    assert retry_metadata.model_dump() == {
        "attempt": 0,
        "max_attempts": 2,
        "retryable": True,
        "last_error": "timeout",
        "metadata": {},
    }

    with pytest.raises(PlanDependencyValidationError, match="Unknown dependencies"):
        validate_plan_dependencies(
            [
                PlanStep(
                    step_id="step-1",
                    goal="汇总",
                    tool_name="final_synthesizer",
                    depends_on=["missing-step"],
                )
            ]
        )

    with pytest.raises(PlanDependencyValidationError, match="cycle"):
        validate_plan_dependencies(
            [
                PlanStep(
                    step_id="step-1",
                    goal="第一步",
                    tool_name="native_rag_search",
                    depends_on=["step-2"],
                ),
                PlanStep(
                    step_id="step-2",
                    goal="第二步",
                    tool_name="final_synthesizer",
                    depends_on=["step-1"],
                ),
            ]
        )
