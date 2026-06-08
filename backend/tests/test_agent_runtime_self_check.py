from __future__ import annotations

from backend.platform.agent_runtime.contracts import (
    PlanRun,
    PlanStep,
    ReActAction,
    ReActRun,
    ReActTurn,
    ToolObservation,
)
from backend.platform.agent_runtime.self_check import (
    CorrectionAction,
    ResultValidator,
    build_result_validation_context,
)


def test_self_check_blocks_evidence_answer_without_successful_observation() -> None:
    run = ReActRun(
        react_run_id="react-no-observation",
        session_id="session-1",
        request_id="request-1",
        user_goal="查制度",
        workflow_status="succeeded",
        final_answer="这是证据回答。",
        metadata={"citations": [{"citation_id": "doc-1"}]},
    )

    report = ResultValidator().validate(
        build_result_validation_context(
            answer_mode="evidence_answer",
            final_decision="answer_with_evidence",
            status="running",
            citations=[{"citation_id": "doc-1"}],
            react_run=run,
        )
    )

    assert report.passed is False
    assert report.correction_action == CorrectionAction.FAIL_FINAL
    assert "missing_successful_observation" in report.categories


def test_self_check_blocks_failed_or_blocked_plan_steps() -> None:
    plan = PlanRun(
        plan_run_id="plan-blocked",
        session_id="session-1",
        request_id="request-1",
        user_goal="先查规则再查库存",
        workflow_status="succeeded",
        steps=[
            PlanStep(
                step_id="step-1",
                goal="查规则",
                tool_name="lookup_policy",
                status="succeeded",
                observation=ToolObservation(
                    tool_name="lookup_policy",
                    success=True,
                    result_summary="查到规则。",
                ),
            ),
            PlanStep(
                step_id="step-2",
                goal="查库存",
                tool_name="lookup_inventory",
                status="blocked",
                error="Plan step step-2 is blocked.",
            ),
        ],
        observations=[
            ToolObservation(
                tool_name="lookup_policy",
                success=True,
                result_summary="查到规则。",
            )
        ],
    )

    report = ResultValidator().validate(
        build_result_validation_context(
            answer_mode="fallback",
            final_decision="no_evidence",
            status="running",
            plan_run=plan,
        )
    )

    assert report.passed is False
    assert report.correction_action == CorrectionAction.FAIL_FINAL
    assert report.categories == ["plan_step_not_successful"]
    assert report.metadata["issues"][0]["metadata"]["step_ids"] == ["step-2"]


def test_self_check_routes_requires_user_observation_to_ask_user() -> None:
    observation = ToolObservation(
        tool_name="agentic_rag_search",
        success=False,
        requires_user=True,
        user_prompt="请补充查询范围。",
        result_summary="需要用户补充。",
    )
    run = ReActRun(
        react_run_id="react-ask-user",
        session_id="session-1",
        request_id="request-1",
        user_goal="查制度",
        workflow_status="succeeded",
        turns=[
            ReActTurn(
                turn_id="turn-1",
                round_index=1,
                goal="查制度",
                action=ReActAction(
                    action_type="tool_call",
                    tool_name="agentic_rag_search",
                ),
                status="succeeded",
                observation=observation,
            )
        ],
        observations=[observation],
    )

    report = ResultValidator().validate(
        build_result_validation_context(
            answer_mode="fallback",
            final_decision="no_evidence",
            status="running",
            react_run=run,
        )
    )

    assert report.passed is False
    assert report.correction_action == CorrectionAction.ASK_USER
    assert report.categories == ["requires_user_not_waiting"]
    assert report.metadata["max_self_check_rounds"] == 1
