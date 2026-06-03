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
    collect_successful_tool_observations,
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
    assert payload["observations"] == []
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
    assert payload["context_summary"] == ""
    assert payload["observations"] == []
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


def test_agent_run_observations_are_serializable_and_restore_old_snapshots() -> None:
    observation = ToolObservation(
        tool_name="native_rag_search",
        success=True,
        result_summary="命中制度证据。",
    )
    react_payload = ReActRun(
        react_run_id="react-run-observation",
        session_id="session-react",
        request_id="request-react",
        user_goal="查询制度",
        observations=[observation],
    ).model_dump()
    plan_payload = PlanRun(
        plan_run_id="plan-run-observation",
        session_id="session-plan",
        request_id="request-plan",
        user_goal="拆解查询制度",
        context_summary="使用文档知识源生成计划。",
        observations=[observation],
    ).model_dump()

    assert react_payload["observations"][0]["result_summary"] == "命中制度证据。"
    assert plan_payload["context_summary"] == "使用文档知识源生成计划。"
    assert plan_payload["observations"][0]["tool_name"] == "native_rag_search"

    react_payload.pop("observations")
    plan_payload.pop("observations")
    plan_payload.pop("context_summary")

    assert ReActRun.model_validate(react_payload).observations == []
    restored_plan = PlanRun.model_validate(plan_payload)
    assert restored_plan.observations == []
    assert restored_plan.context_summary == ""


def test_collect_successful_tool_observations_prefers_run_level_results() -> None:
    failed_step_observation = ToolObservation(
        tool_name="native_rag_search",
        success=False,
        result_summary="步骤旧结果失败。",
    )
    successful_run_observation = ToolObservation(
        tool_name="native_rag_search",
        success=True,
        result_summary="run 级结果成功。",
    )
    run = PlanRun(
        plan_run_id="plan-run-collect",
        session_id="session-plan",
        request_id="request-plan",
        user_goal="汇总计划结果",
        steps=[
            PlanStep(
                step_id="step-1",
                goal="检索资料",
                tool_name="native_rag_search",
                observation=failed_step_observation,
            )
        ],
        observations=[successful_run_observation],
    )

    assert collect_successful_tool_observations(run) == [successful_run_observation]


def test_collect_successful_tool_observations_requires_run_level_results() -> None:
    successful_observation = ToolObservation(
        tool_name="native_rag_search",
        success=True,
        result_summary="turn 结果成功。",
    )
    failed_observation = ToolObservation(
        tool_name="native_rag_search",
        success=False,
        result_summary="turn 结果失败。",
    )
    run = ReActRun(
        react_run_id="react-run-collect",
        session_id="session-react",
        request_id="request-react",
        user_goal="汇总 ReAct 结果",
        turns=[
            ReActTurn(
                turn_id="turn-1",
                round_index=1,
                goal="检索资料",
                action=ReActAction(action_type="tool_call", tool_name="native_rag_search"),
                observation=failed_observation,
            ),
            ReActTurn(
                turn_id="turn-2",
                round_index=2,
                goal="检索资料",
                action=ReActAction(action_type="tool_call", tool_name="native_rag_search"),
                observation=successful_observation,
            ),
        ],
    )

    assert collect_successful_tool_observations(run) == []


def test_plan_step_blocked_status_is_a_serializable_contract() -> None:
    step = PlanStep(
        step_id="step-2",
        goal="汇总前置结果",
        tool_name="final_synthesizer",
        depends_on=["step-1"],
        status="blocked",
        error="依赖 step-1 未成功，当前步骤无法执行。",
    )

    payload = step.model_dump()

    assert payload["status"] == "blocked"
    assert payload["error"] == "依赖 step-1 未成功，当前步骤无法执行。"


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
