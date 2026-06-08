from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.platform.agent_runtime.chat_graph.graph.config import ChatGraphDependencies
from backend.platform.agent_runtime.self_check import (
    CorrectionAction,
    ResultValidationReport,
    ResultValidator,
    build_result_validation_context,
)
from backend.platform.workflow.langgraph.state import RuntimeGraphState


def build_self_check_guard_node(dependencies: ChatGraphDependencies):
    """在最终回答前执行 after-agent self-check。"""

    del dependencies
    validator = ResultValidator()

    def self_check_guard(state: RuntimeGraphState) -> dict[str, Any]:
        report = validator.validate(_build_context(state))
        return _build_state_update(state=state, report=report)

    return self_check_guard


def _build_context(state: RuntimeGraphState):
    return build_result_validation_context(
        answer_mode=state.get("answer_mode"),
        final_decision=state.get("final_decision"),
        status=state.get("status"),
        citations=list(state.get("citations") or []),
        react_run=state.get("react_run"),
        plan_run=state.get("plan_run"),
        metadata=dict(state.get("metadata") or {}),
    )


def _build_state_update(
    *,
    state: RuntimeGraphState,
    report: ResultValidationReport,
) -> dict[str, Any]:
    payload = report.to_payload()
    metadata = _metadata_with_report(state.get("metadata"), payload)
    update: dict[str, Any] = {
        "metadata": metadata,
        "react_run": _run_with_report(state.get("react_run"), payload),
        "plan_run": _run_with_report(state.get("plan_run"), payload),
    }
    if report.passed:
        return update
    if report.correction_action == CorrectionAction.ASK_USER:
        return {
            **update,
            "status": "waiting_user",
            "state_event": "interrupt",
            "final_state": "waiting_user",
            "answer": "",
            "answer_mode": "follow_up",
            "final_decision": "ask_user",
            "follow_up_question": _follow_up_question(report),
        }
    return {
        **update,
        "status": "failed",
        "state_event": "fail",
        "final_state": "failed",
        "answer": "",
        "knowledge_used": False,
        "citations": [],
        "final_decision": "retrieval_failed",
    }


def _metadata_with_report(
    metadata: Mapping[str, Any] | None,
    report_payload: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(metadata or {})
    payload["self_check"] = dict(report_payload)
    return payload


def _run_with_report(
    run_payload: Mapping[str, Any] | None,
    report_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(run_payload, Mapping):
        return None
    payload = dict(run_payload)
    metadata = dict(payload.get("metadata") or {})
    metadata["self_check"] = dict(report_payload)
    payload["metadata"] = metadata
    if report_payload.get("passed") is False:
        _apply_failed_report_to_run(payload=payload, report_payload=report_payload)
    return payload


def _apply_failed_report_to_run(
    *,
    payload: dict[str, Any],
    report_payload: Mapping[str, Any],
) -> None:
    if report_payload.get("correction_action") == CorrectionAction.ASK_USER:
        payload["workflow_status"] = "waiting_user"
        payload["result_summary"] = _report_summary(report_payload)
        payload["error"] = None
        return
    payload["workflow_status"] = "failed"
    payload["final_answer"] = None
    payload["result_summary"] = _report_summary(report_payload)
    payload["error"] = payload["result_summary"]


def _follow_up_question(report: ResultValidationReport) -> str:
    for issue in report.metadata.get("issues", []):
        if not isinstance(issue, Mapping):
            continue
        metadata = issue.get("metadata")
        if isinstance(metadata, Mapping):
            prompt = metadata.get("user_prompt")
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
    return "请补充必要信息后继续。"


def _report_summary(report_payload: Mapping[str, Any]) -> str:
    reasons = report_payload.get("failure_reasons")
    if isinstance(reasons, list) and reasons:
        return str(reasons[0])
    return "Result self-check failed."
