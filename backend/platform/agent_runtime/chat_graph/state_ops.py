from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.platform.agent_runtime.chat_graph.contracts import HitlWaitInput
from backend.platform.agent_runtime.core.contracts import AgentMode
from backend.platform.agent_runtime.tooling.rag import AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME
from backend.platform.workflow.langgraph.state import RuntimeGraphState, RuntimeHitlState
from backend.platform.workflow.state_machine import WorkflowRunState, is_terminal


class AgentRuntimeStateProjectionMixin:
    def _build_agent_runtime_success_update(
        self,
        *,
        state: RuntimeGraphState,
        answer: str,
        citations: Sequence[Any],
        knowledge_used: bool,
    ) -> dict[str, Any]:
        """把既有检索回答归档为顶层 Agent Runtime 审计结构。"""
        agent_mode = _coerce_agent_mode(state.get("agent_mode"))
        citation_payloads = [_dump_model_or_mapping(citation) for citation in citations]
        retrieval_trace = dict(state.get("retrieval_trace") or {})
        existing_run_update = _complete_existing_agent_run_payload(
            state=state,
            agent_mode=agent_mode,
            answer=answer,
            citations=citation_payloads,
            knowledge_used=knowledge_used,
        )
        if existing_run_update is not None:
            return existing_run_update
        tool_name = _resolve_rag_tool_name(retrieval_trace)
        observation = _build_tool_observation_payload(
            tool_name=tool_name,
            retrieval_trace=retrieval_trace,
            citations=citation_payloads,
            knowledge_used=knowledge_used,
        )
        if agent_mode == "plan":
            step_id = str(state.get("current_step_id") or "")
            plan_run = _build_plan_run_payload(
                state=state,
                answer=answer,
                observation=observation,
                citations=citation_payloads,
                step_id=step_id,
                status="succeeded",
            )
            return {
                "agent_mode": "plan",
                "plan_run": plan_run,
                "react_run": None,
                "current_turn_id": None,
                "current_step_id": None,
                "current_tool_call": None,
            }

        turn_id = str(state.get("current_turn_id") or "")
        react_run = _build_react_run_payload(
            state=state,
            answer=answer,
            observation=observation,
            citations=citation_payloads,
            turn_id=turn_id,
            status="succeeded",
        )
        return {
            "agent_mode": "react",
            "react_run": react_run,
            "plan_run": None,
            "current_turn_id": None,
            "current_step_id": None,
            "current_tool_call": None,
        }

    def _build_agent_runtime_wait_update(
        self,
        *,
        wait: HitlWaitInput,
        hitl: RuntimeHitlState,
    ) -> dict[str, Any]:
        """HITL wait 写入顶层 ReAct/Plan 恢复点。"""
        metadata = dict(wait.metadata or {})
        mode = _coerce_optional_agent_mode(metadata.get("mode"))
        if mode is None:
            return {}

        tool_call = (
            dict(wait.proposed_tool_call)
            if wait.proposed_tool_call is not None
            else None
        )
        if mode == "plan":
            step_id = str(metadata.get("current_step_id") or "")
            plan_run = _build_plan_wait_payload(
                wait=wait,
                hitl=hitl,
                step_id=step_id,
                tool_call=tool_call,
            )
            return {
                "agent_mode": "plan",
                "plan_run": plan_run,
                "react_run": None,
                "current_step_id": step_id,
                "current_turn_id": None,
                "current_tool_call": _tool_execution_payload(tool_call),
            }

        turn_id = str(metadata.get("current_turn_id") or "")
        react_run = _build_react_wait_payload(
            wait=wait,
            hitl=hitl,
            turn_id=turn_id,
            tool_call=tool_call,
        )
        return {
            "agent_mode": "react",
            "react_run": react_run,
            "plan_run": None,
            "current_turn_id": turn_id,
            "current_step_id": None,
            "current_tool_call": _tool_execution_payload(tool_call),
        }

    def _build_agent_runtime_resume_update(
        self,
        *,
        state: RuntimeGraphState,
        action: str,
        resume_payload: Mapping[str, Any],
        next_status: WorkflowRunState,
        final_answer: str | None = None,
        tool_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """resume 后同步顶层 turn/step 状态，reject 不执行副作用。"""
        agent_mode = _coerce_optional_agent_mode(state.get("agent_mode"))
        if agent_mode is None:
            return {}

        if agent_mode == "plan":
            plan_run = dict(state.get("plan_run") or {})
            _update_plan_run_status(
                plan_run=plan_run,
                next_status=next_status,
                action=action,
                current_step_id=state.get("current_step_id"),
                final_answer=final_answer,
            )
            clear_current = is_terminal(next_status)
            return {
                "plan_run": plan_run,
                "current_step_id": None if clear_current else state.get("current_step_id"),
                "current_tool_call": None if clear_current else state.get("current_tool_call"),
            }

        react_run = dict(state.get("react_run") or {})
        _record_react_resume_metadata(
            react_run=react_run,
            action=action,
            resume_payload=resume_payload,
            current_turn_id=state.get("current_turn_id"),
        )
        if action == "approve" and tool_result is not None:
            _append_approved_tool_observation(
                react_run=react_run,
                current_turn_id=state.get("current_turn_id"),
                tool_result=tool_result,
            )
        _update_react_run_status(
            react_run=react_run,
            next_status=next_status,
            action=action,
            current_turn_id=state.get("current_turn_id"),
            final_answer=final_answer,
        )
        clear_current = is_terminal(next_status)
        return {
            "react_run": react_run,
            "current_turn_id": None if clear_current else state.get("current_turn_id"),
            "current_tool_call": None if clear_current else state.get("current_tool_call"),
        }


def _coerce_agent_mode(value: Any) -> AgentMode:
    return "plan" if value == "plan" else "react"


def _coerce_optional_agent_mode(value: Any) -> AgentMode | None:
    if value in {"react", "plan"}:
        return value  # type: ignore[return-value]
    return None


def _dump_model_or_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _resolve_rag_tool_name(retrieval_trace: Mapping[str, Any]) -> str:
    rounds = retrieval_trace.get("rounds")
    if isinstance(rounds, Sequence) and not isinstance(rounds, (str, bytes)):
        if len(rounds) > 1:
            return AGENTIC_RAG_TOOL_NAME
    return NATIVE_RAG_TOOL_NAME


def _build_tool_observation_payload(
    *,
    tool_name: str,
    retrieval_trace: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    knowledge_used: bool,
) -> dict[str, Any]:
    success = bool(retrieval_trace.get("success", knowledge_used))
    final_decision = retrieval_trace.get("final_decision")
    return {
        "tool_name": tool_name,
        "success": success,
        "output": {
            "knowledge_used": knowledge_used,
            "final_decision": final_decision,
        },
        "result_summary": _summarize_retrieval_observation(
            final_decision=final_decision,
            knowledge_used=knowledge_used,
        ),
        "citations": [dict(citation) for citation in citations],
        "trace": {"retrieval_trace": dict(retrieval_trace)},
        "retryable": False,
        "requires_user": final_decision == "ask_user",
        "user_prompt": retrieval_trace.get("follow_up_question"),
        "error": None if success else str(final_decision or "retrieval_failed"),
        "metadata": {
            "knowledge_used": knowledge_used,
            "final_decision": final_decision,
        },
    }


def _complete_existing_agent_run_payload(
    *,
    state: RuntimeGraphState,
    agent_mode: AgentMode,
    answer: str,
    citations: Sequence[Mapping[str, Any]],
    knowledge_used: bool,
) -> dict[str, Any] | None:
    """优先使用 ChatService 真实执行出的 ReActRun/PlanRun，避免成功节点再合成审计。"""
    if agent_mode == "plan":
        plan_run = state.get("plan_run")
        if not isinstance(plan_run, Mapping):
            return None
        payload = dict(plan_run)
        payload["workflow_status"] = "succeeded"
        payload["final_answer"] = answer
        payload["error"] = None
        payload["current_step_id"] = None
        payload["current_tool_call"] = None
        steps = []
        for step in list(payload.get("steps") or []):
            if isinstance(step, Mapping):
                step_payload = dict(step)
                if step_payload.get("status") in {"running", "retrying", "waiting_user"}:
                    step_payload["status"] = "succeeded"
                steps.append(step_payload)
            else:
                steps.append(step)
        payload["steps"] = steps
        metadata = dict(payload.get("metadata") or {})
        metadata["citations"] = [dict(citation) for citation in citations]
        metadata["knowledge_used"] = knowledge_used
        payload["metadata"] = metadata
        return {
            "agent_mode": "plan",
            "plan_run": payload,
            "react_run": None,
            "current_turn_id": None,
            "current_step_id": None,
            "current_tool_call": None,
        }

    react_run = state.get("react_run")
    if not isinstance(react_run, Mapping):
        return None
    payload = dict(react_run)
    payload["workflow_status"] = "succeeded"
    payload["final_answer"] = answer
    payload["error"] = None
    payload["current_turn_id"] = None
    payload["current_tool_call"] = None
    turns = []
    for turn in list(payload.get("turns") or []):
        if isinstance(turn, Mapping):
            turn_payload = dict(turn)
            if turn_payload.get("status") in {"running", "retrying", "waiting_user"}:
                turn_payload["status"] = "succeeded"
            turns.append(turn_payload)
        else:
            turns.append(turn)
    payload["turns"] = turns
    metadata = dict(payload.get("metadata") or {})
    metadata["citations"] = [dict(citation) for citation in citations]
    metadata["knowledge_used"] = knowledge_used
    payload["metadata"] = metadata
    return {
        "agent_mode": "react",
        "react_run": payload,
        "plan_run": None,
        "current_turn_id": None,
        "current_step_id": None,
        "current_tool_call": None,
    }


def _summarize_retrieval_observation(
    *,
    final_decision: Any,
    knowledge_used: bool,
) -> str:
    if knowledge_used:
        return "RAG tool returned usable evidence."
    if final_decision == "ask_user":
        return "RAG tool requested user clarification."
    return "RAG tool completed without usable evidence."


def _build_react_run_payload(
    *,
    state: RuntimeGraphState,
    answer: str,
    observation: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    turn_id: str,
    status: WorkflowRunState,
) -> dict[str, Any]:
    tool_name = str(observation.get("tool_name") or NATIVE_RAG_TOOL_NAME)
    return {
        "react_run_id": _react_run_id(state),
        "session_id": state["session_id"],
        "request_id": state["request_id"],
        "mode": "react",
        "user_goal": _last_user_goal(state),
        "workflow_status": status,
        "max_turns": 5,
        "turns": [
            {
                "turn_id": turn_id,
                "round_index": 1,
                "goal": _last_user_goal(state),
                "action": {
                    "action_type": "tool_call",
                    "tool_name": tool_name,
                    "input": {"query": _last_user_goal(state)},
                    "rationale_summary": "Top-level ReAct selected the RAG tool.",
                    "metadata": {},
                },
                "status": status,
                "input": {"query": _last_user_goal(state)},
                "tool_name": tool_name,
                "observation": dict(observation),
                "observation_summary": str(observation.get("result_summary") or ""),
                "result_summary": str(observation.get("result_summary") or ""),
                "error": None,
                "metadata": {},
            }
        ],
        "observations": [dict(observation)],
        "current_turn_id": None,
        "current_tool_call": None,
        "final_answer": answer,
        "result_summary": "ReAct run synthesized final answer from collected observations.",
        "error": None,
        "metadata": {
            "citations": [dict(citation) for citation in citations],
            "knowledge_used": bool(citations),
        },
    }


def _build_plan_run_payload(
    *,
    state: RuntimeGraphState,
    answer: str,
    observation: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    step_id: str,
    status: WorkflowRunState,
) -> dict[str, Any]:
    tool_name = str(observation.get("tool_name") or NATIVE_RAG_TOOL_NAME)
    return {
        "plan_run_id": _plan_run_id(state),
        "session_id": state["session_id"],
        "request_id": state["request_id"],
        "mode": "plan",
        "user_goal": _last_user_goal(state),
        "context_summary": "Graph success projection from retrieval observation.",
        "workflow_status": status,
        "steps": [
            {
                "step_id": step_id,
                "goal": _last_user_goal(state),
                "tool_name": tool_name,
                "input": {"query": _last_user_goal(state)},
                "depends_on": [],
                "status": status,
                "observation": dict(observation),
                "output": dict(observation.get("output") or {}),
                "result_summary": str(observation.get("result_summary") or ""),
                "error": None,
                "metadata": {},
            }
        ],
        "observations": [dict(observation)],
        "current_step_id": None,
        "current_tool_call": None,
        "final_answer": answer,
        "result_summary": "Plan run synthesized final answer from successful steps.",
        "error": None,
        "metadata": {
            "workflow_transitions": [
                {"from": "created", "event": "plan_start", "to": "planning"},
                {"from": "planning", "event": "run_start", "to": "running"},
                {"from": "running", "event": "success", "to": status},
            ],
            "citations": [dict(citation) for citation in citations],
            "knowledge_used": bool(citations),
        },
    }


def _build_react_wait_payload(
    *,
    wait: HitlWaitInput,
    hitl: RuntimeHitlState,
    turn_id: str,
    tool_call: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(wait.metadata or {})
    user_goal = str(metadata.get("user_goal") or wait.reason)
    return {
        "react_run_id": str(metadata.get("react_run_id") or f"react-{wait.request_id}"),
        "session_id": wait.session_id,
        "request_id": wait.request_id,
        "mode": "react",
        "user_goal": user_goal,
        "workflow_status": "waiting_user",
        "max_turns": 5,
        "turns": [
            {
                "turn_id": turn_id,
                "round_index": 1,
                "goal": user_goal,
                "action": {
                    "action_type": "ask_user",
                    "tool_name": None,
                    "input": {},
                    "instruction": wait.reason,
                    "rationale_summary": "Top-level ReAct needs user clarification.",
                    "metadata": {},
                },
                "status": "waiting_user",
                "input": {},
                "tool_name": None,
                "observation": None,
                "observation_summary": wait.reason,
                "result_summary": wait.reason,
                "error": None,
                "metadata": {"hitl": dict(hitl)},
            }
        ],
        "observations": [],
        "current_turn_id": turn_id,
        "current_tool_call": _tool_execution_payload(tool_call),
        "final_answer": None,
        "result_summary": wait.reason,
        "error": None,
        "metadata": {"hitl": dict(hitl)},
    }


def _build_plan_wait_payload(
    *,
    wait: HitlWaitInput,
    hitl: RuntimeHitlState,
    step_id: str,
    tool_call: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(wait.metadata or {})
    user_goal = str(metadata.get("user_goal") or wait.reason)
    return {
        "plan_run_id": str(metadata.get("plan_run_id") or f"plan-{wait.request_id}"),
        "session_id": wait.session_id,
        "request_id": wait.request_id,
        "mode": "plan",
        "user_goal": user_goal,
        "context_summary": "HITL wait projection from graph runtime.",
        "workflow_status": "waiting_user",
        "steps": [
            {
                "step_id": step_id,
                "goal": user_goal,
                "tool_name": str((tool_call or {}).get("tool_name") or NATIVE_RAG_TOOL_NAME),
                "input": dict((tool_call or {}).get("args") or {"query": user_goal}),
                "depends_on": [],
                "status": "waiting_user",
                "observation": None,
                "output": None,
                "result_summary": wait.reason,
                "error": None,
                "metadata": {"hitl": dict(hitl)},
            }
        ],
        "observations": [],
        "current_step_id": step_id,
        "current_tool_call": _tool_execution_payload(tool_call),
        "final_answer": None,
        "result_summary": wait.reason,
        "error": None,
        "metadata": {
            "hitl": dict(hitl),
            "workflow_transitions": [
                {"from": "created", "event": "plan_start", "to": "planning"},
                {"from": "planning", "event": "run_start", "to": "running"},
                {"from": "running", "event": "interrupt", "to": "waiting_user"},
            ],
        },
    }


def _tool_execution_payload(tool_call: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """把 HITL proposed_tool_call 收敛为 ToolExecutionMetadata 可接受的字段。"""
    if not tool_call:
        return None
    payload: dict[str, Any] = {
        "tool_name": str(tool_call.get("tool_name") or ""),
        "metadata": {},
    }
    if tool_call.get("tool_call_id") is not None:
        payload["tool_call_id"] = str(tool_call["tool_call_id"])
    args = tool_call.get("args")
    if isinstance(args, Mapping):
        payload["metadata"] = {"args": dict(args)}
    return payload


def _update_react_run_status(
    *,
    react_run: dict[str, Any],
    next_status: WorkflowRunState,
    action: str,
    current_turn_id: Any,
    final_answer: str | None = None,
) -> None:
    react_run["workflow_status"] = next_status
    terminal_turn_status = _terminal_child_status(next_status)
    if terminal_turn_status is not None:
        for turn in react_run.get("turns") or []:
            if turn.get("turn_id") == current_turn_id:
                turn["status"] = terminal_turn_status
        react_run["current_turn_id"] = None
        react_run["current_tool_call"] = None
    _settle_run_result(run=react_run, status=next_status, final_answer=final_answer)


def _record_react_resume_metadata(
    *,
    react_run: dict[str, Any],
    action: str,
    resume_payload: Mapping[str, Any],
    current_turn_id: Any,
) -> None:
    """把 application resume 输入投影到 ReActRun，便于 checkpoint 审计。"""
    if not react_run:
        return
    metadata = dict(react_run.get("metadata") or {})
    continuation = {
        "mode": "react",
        "action": action,
        "react_run_id": react_run.get("react_run_id"),
        "waiting_turn_id": current_turn_id,
        "continued_from_turn_id": current_turn_id,
        "response": resume_payload.get("response"),
        "source": resume_payload.get("source"),
        "suggestion_id": resume_payload.get("suggestion_id"),
        "reason": resume_payload.get("reason"),
        "metadata": dict(resume_payload.get("metadata") or {}),
    }
    metadata["resume"] = continuation
    history = list(metadata.get("continuations") or [])
    history.append(continuation)
    metadata["continuations"] = history
    react_run["metadata"] = metadata
    for turn in react_run.get("turns") or []:
        if isinstance(turn, dict) and turn.get("turn_id") == current_turn_id:
            turn_metadata = dict(turn.get("metadata") or {})
            turn_metadata["continuation"] = continuation
            turn["metadata"] = turn_metadata
            break


def _append_approved_tool_observation(
    *,
    react_run: dict[str, Any],
    current_turn_id: Any,
    tool_result: Mapping[str, Any],
) -> None:
    """把 approve 后执行的工具结果写回等待 turn 和 run 级 observations。"""
    if not react_run or not tool_result:
        return
    observation = dict(tool_result)
    metadata = dict(observation.get("metadata") or {})
    metadata["continued_from_turn_id"] = current_turn_id
    metadata["resume_action"] = "approve"
    observation["metadata"] = metadata
    observations = list(react_run.get("observations") or [])
    observations.append(observation)
    react_run["observations"] = observations
    for turn in react_run.get("turns") or []:
        if isinstance(turn, dict) and turn.get("turn_id") == current_turn_id:
            turn["observation"] = observation
            turn["observation_summary"] = str(observation.get("result_summary") or "")
            turn["result_summary"] = str(observation.get("result_summary") or "")
            turn["error"] = observation.get("error")
            break


def _update_plan_run_status(
    *,
    plan_run: dict[str, Any],
    next_status: WorkflowRunState,
    action: str,
    current_step_id: Any,
    final_answer: str | None = None,
) -> None:
    plan_run["workflow_status"] = next_status
    terminal_step_status = _terminal_child_status(next_status)
    if terminal_step_status is not None:
        for step in plan_run.get("steps") or []:
            if step.get("step_id") == current_step_id:
                step["status"] = terminal_step_status
        plan_run["current_step_id"] = None
        plan_run["current_tool_call"] = None
    _settle_run_result(run=plan_run, status=next_status, final_answer=final_answer)


def _terminal_child_status(status: WorkflowRunState) -> str | None:
    if status == "succeeded":
        return "succeeded"
    if status == "failed":
        return "failed"
    if status == "cancelled":
        return "cancelled"
    return None


def _settle_run_result(
    *,
    run: dict[str, Any],
    status: WorkflowRunState,
    final_answer: str | None,
) -> None:
    """终态 resume 要把结果同步到 nested Agent run，避免 checkpoint 审计断层。"""
    if not is_terminal(status):
        return
    summary = str(final_answer or run.get("result_summary") or "")
    if status == "succeeded":
        run["final_answer"] = summary
        run["error"] = None
    elif status == "failed":
        run["error"] = summary or run.get("error")
    run["result_summary"] = summary


def _react_run_id(state: RuntimeGraphState) -> str:
    react_run = state.get("react_run")
    if isinstance(react_run, Mapping) and react_run.get("react_run_id"):
        return str(react_run["react_run_id"])
    return f"react-{state['request_id']}"


def _plan_run_id(state: RuntimeGraphState) -> str:
    plan_run = state.get("plan_run")
    if isinstance(plan_run, Mapping) and plan_run.get("plan_run_id"):
        return str(plan_run["plan_run_id"])
    return f"plan-{state['request_id']}"


def _last_user_goal(state: RuntimeGraphState) -> str:
    messages = list(state.get("messages") or ())
    for message in reversed(messages):
        content = getattr(message, "content", None)
        message_type = getattr(message, "type", None)
        if message_type == "human" and isinstance(content, str):
            return content
    return ""




