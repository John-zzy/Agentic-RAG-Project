from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
import logging
from typing import Any

logger = logging.getLogger("backend.platform.agent_runtime.graph")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_MAX_TEXT_LENGTH = 500


def wrap_graph_node(
    *,
    graph_name: str,
    node_name: str,
    node: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """只记录关键节点调用，避免把完整 graph state 打到控制台。"""

    def logged_node(state: Any, runtime: Any = None) -> Any:
        logger.info(
            "AgentGraph[%s] node=%s context=%s",
            graph_name,
            node_name,
            _context_summary(state),
        )
        try:
            return _invoke_node(node=node, state=state, runtime=runtime)
        except Exception as exc:
            logger.exception(
                "AgentGraph[%s] node=%s error=%s",
                graph_name,
                node_name,
                exc,
            )
            raise

    return logged_node


def wrap_graph_route(
    *,
    graph_name: str,
    route_name: str,
    route: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """记录条件边最终选择了哪个分支。"""

    def logged_route(state: Any) -> Any:
        selected = route(state)
        logger.info(
            "AgentGraph[%s] route=%s selected=%s context=%s",
            graph_name,
            route_name,
            selected,
            _context_summary(state),
        )
        return selected

    return logged_route


def log_graph_invoke_start(*, graph_name: str, payload: Any) -> None:
    logger.info("AgentGraph[%s] invoke_start context=%s", graph_name, _context_summary(payload))


def log_graph_invoke_end(*, graph_name: str, payload: Any) -> None:
    logger.info("AgentGraph[%s] invoke_end context=%s", graph_name, _context_summary(payload))


def log_graph_invoke_error(*, graph_name: str, error: BaseException) -> None:
    logger.exception("AgentGraph[%s] invoke_error error=%s", graph_name, error)


def log_llm_output(
    *,
    source: str,
    request_id: str | None = None,
    output: Any,
) -> None:
    logger.info(
        "AgentGraph[llm] source=%s request_id=%s output=%s",
        source,
        request_id,
        _truncate(_output_text(output)),
    )


def _context_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    summary: dict[str, Any] = {}
    for key in (
        "session_id",
        "request_id",
        "agent_mode",
        "answer_mode",
        "final_decision",
        "status",
        "current_turn_id",
        "current_step_id",
    ):
        if key in payload and payload[key] is not None:
            summary[key] = payload[key]

    run = payload.get("run") or payload.get("react_run")
    if run is not None:
        summary["react_run"] = _run_summary(run, id_key="react_run_id", child_key="turns")

    plan_run = payload.get("plan_run")
    if plan_run is not None:
        summary["plan_run"] = _run_summary(plan_run, id_key="plan_run_id", child_key="steps")

    action = payload.get("action")
    action_summary = _action_summary(action)
    if action_summary:
        summary["action"] = action_summary

    step = payload.get("step")
    step_summary = _step_summary(step)
    if step_summary:
        summary["step"] = step_summary

    observation = payload.get("observation") or payload.get("tool_observation")
    observation_summary = _observation_summary(observation)
    if observation_summary:
        summary["observation"] = observation_summary

    retrieval_trace = payload.get("retrieval_trace")
    if isinstance(retrieval_trace, Mapping):
        summary["retrieval_trace"] = {
            "final_decision": retrieval_trace.get("final_decision"),
            "tool_call_count": retrieval_trace.get("tool_call_count"),
            "knowledge_used": retrieval_trace.get("knowledge_used"),
        }

    return summary


def _run_summary(value: Any, *, id_key: str, child_key: str) -> dict[str, Any]:
    payload = _model_or_mapping(value)
    return {
        "id": payload.get(id_key),
        "status": payload.get("workflow_status"),
        "child_count": _count(payload.get(child_key)),
        "observation_count": _count(payload.get("observations")),
    }


def _action_summary(value: Any) -> dict[str, Any] | None:
    payload = _model_or_mapping(value)
    if not payload:
        return None
    return {
        "action_type": payload.get("action_type"),
        "tool_name": payload.get("tool_name"),
        "rationale_summary": _truncate(str(payload.get("rationale_summary") or "")) or None,
    }


def _step_summary(value: Any) -> dict[str, Any] | None:
    payload = _model_or_mapping(value)
    if not payload:
        return None
    return {
        "step_id": payload.get("step_id"),
        "tool_name": payload.get("tool_name"),
        "status": payload.get("status"),
    }


def _observation_summary(value: Any) -> dict[str, Any] | None:
    payload = _model_or_mapping(value)
    if not payload:
        return None
    return {
        "tool_name": payload.get("tool_name"),
        "success": payload.get("success"),
        "requires_user": payload.get("requires_user"),
        "error": _truncate(str(payload.get("error") or "")) or None,
    }


def _model_or_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return {}


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list | tuple) else 0


def _output_text(output: Any) -> str:
    if hasattr(output, "content"):
        return str(getattr(output, "content") or "")
    return str(output or "")


def _truncate(value: str) -> str:
    text = value.strip()
    if len(text) <= _MAX_TEXT_LENGTH:
        return text
    return f"{text[:_MAX_TEXT_LENGTH]}..."


def _invoke_node(*, node: Callable[..., Any], state: Any, runtime: Any) -> Any:
    if runtime is None or len(inspect.signature(node).parameters) < 2:
        return node(state)
    return node(state, runtime)

