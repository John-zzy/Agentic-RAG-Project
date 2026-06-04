from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage

from backend.application.runtime.assembly.service_parts.contracts import PreparedChatTurn
from backend.platform.agent_runtime.rag_tools import AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME


class ChatStreamEventMixin:
    def _build_history_event(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造本轮模型调用前的历史消息快照事件。"""
        messages = self._get_session_history(
            prepared.session_id,
            request_id=prepared.request_id,
            timestamp=prepared.timestamp,
        ).messages
        return {
            "session_id": prepared.session_id,
            "request_id": prepared.request_id,
            "window_size": self.settings.session.window_size,
            "message_count": len(messages),
            "messages": [self._serialize_history_message(message) for message in messages],
        }

    def _build_tool_event(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造 retrieval/tool 阶段的结构化事件。"""
        return {
            **prepared.tool_event,
            "stage": "plan_step" if prepared.agent_mode == "plan" else "react_turn",
            "retrieval_stage": prepared.tool_event.get("stage"),
            "rounds": [round_trace.model_dump() for round_trace in prepared.retrieval_trace.rounds],
            "session_id": prepared.session_id,
            "request_id": prepared.request_id,
            "agent_mode": prepared.agent_mode,
            **self._build_agent_progress_event(prepared),
            "mode_selection": {
                "reason": prepared.agent_mode_reason,
                "signals": dict(prepared.agent_mode_signals or {}),
            },
            "final_decision": prepared.final_decision,
            "follow_up_question": prepared.follow_up_question,
            "answer_mode": prepared.answer_mode,
            "knowledge_used": prepared.knowledge_used,
            "citations": [citation.model_dump() for citation in prepared.citations],
            "retrieval_trace": prepared.retrieval_trace.model_dump(),
            "react_run": prepared.react_run,
            "plan_run": prepared.plan_run,
            "current_turn_id": prepared.current_turn_id,
            "current_step_id": prepared.current_step_id,
            "current_tool_call": prepared.current_tool_call,
            "tool_observation": prepared.tool_observation,
        }

    def _build_agent_progress_event(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造顶层 ReAct/Plan 进度字段，复用现有 tool SSE 事件名。"""
        if prepared.agent_mode == "plan":
            plan_run = dict(prepared.plan_run or {})
            step = self._event_plan_step(prepared)
            steps = self._event_plan_steps(plan_run)
            return {
                "plan_run_id": str(plan_run.get("plan_run_id") or f"plan-{prepared.request_id}"),
                "step_id": str((step or {}).get("step_id") or prepared.current_step_id or ""),
                "step_status": str((step or {}).get("status") or "running"),
                "workflow_status": str(plan_run.get("workflow_status") or "running"),
                "tool_name": str(
                    (step or {}).get("tool_name")
                    or self._resolve_tool_event_tool_name(prepared)
                ),
                "step_count": len(steps),
                "execution_order": self._event_plan_execution_order(steps),
            }
        react_run = dict(prepared.react_run or {})
        turn = self._event_react_turn(prepared)
        action = (turn or {}).get("action") if isinstance((turn or {}).get("action"), Mapping) else {}
        turns = self._event_react_turns(react_run)
        return {
            "react_run_id": str(react_run.get("react_run_id") or f"react-{prepared.request_id}"),
            "turn_id": str((turn or {}).get("turn_id") or prepared.current_turn_id or ""),
            "active_turn": turn,
            "turn_status": str((turn or {}).get("status") or "running"),
            "action": str(action.get("action_type") or "tool_call"),
            "action_type": str(action.get("action_type") or "tool_call"),
            "rationale_summary": str(action.get("rationale_summary") or ""),
            "workflow_status": str(react_run.get("workflow_status") or "running"),
            "tool_name": str(
                (turn or {}).get("tool_name")
                or action.get("tool_name")
                or self._resolve_tool_event_tool_name(prepared)
            ),
            "turn_count": len(turns),
            "max_turns": self._resolve_positive_int(react_run.get("max_turns")),
            "attempted_tools": self._event_attempted_tools(react_run=react_run, turn=turn),
            "latest_action_selection": self._event_latest_action_selection(react_run),
            "action_validation_result": self._event_action_validation_result(react_run),
        }

    def _resolve_tool_event_status(self, prepared: PreparedChatTurn) -> str:
        """避免 HITL 等待路径先向客户端报告顶层 turn/step 已成功。"""
        if prepared.agent_mode == "plan":
            step = self._event_plan_step(prepared)
            if step is not None:
                return str(step.get("status") or "running")
        turn = self._event_react_turn(prepared)
        if turn is not None:
            return str(turn.get("status") or "running")
        return "succeeded"

    def _resolve_tool_event_tool_name(self, prepared: PreparedChatTurn) -> str:
        """从 retrieval trace 推断顶层 RAG 工具名，RAG rounds 仍留在 nested trace。"""
        if prepared.tool_observation and prepared.tool_observation.get("tool_name"):
            return str(prepared.tool_observation["tool_name"])
        if prepared.tool_event.get("tool_name"):
            return str(prepared.tool_event["tool_name"])
        if len(prepared.retrieval_trace.rounds) > 1:
            return AGENTIC_RAG_TOOL_NAME
        return NATIVE_RAG_TOOL_NAME

    def _build_agent_hitl_metadata(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """为 HITL wait 标识顶层 Agent 恢复点。"""
        if prepared.agent_mode == "plan":
            plan_run = dict(prepared.plan_run or {})
            return {
                "mode": "plan",
                "plan_run_id": str(plan_run.get("plan_run_id") or f"plan-{prepared.request_id}"),
                "current_step_id": prepared.current_step_id or self._event_step_id_from_payload(prepared),
                "user_goal": prepared.user_message,
            }
        react_run = dict(prepared.react_run or {})
        return {
            "mode": "react",
            "react_run_id": str(react_run.get("react_run_id") or f"react-{prepared.request_id}"),
            "current_turn_id": prepared.current_turn_id or self._event_turn_id_from_payload(prepared),
            "user_prompt": prepared.follow_up_question or prepared.user_message,
            "source": "chat_hitl_clarification",
            "user_goal": prepared.user_message,
            "continuation": {
                "type": "waiting_user",
                "request_id": prepared.request_id,
            },
        }

    def _event_react_turn(self, prepared: PreparedChatTurn) -> dict[str, Any] | None:
        react_run = prepared.react_run
        if not isinstance(react_run, Mapping):
            return None
        turns = self._event_react_turns(react_run)
        if not turns:
            return None
        current_turn_id = prepared.current_turn_id
        if current_turn_id:
            for turn in turns:
                if turn.get("turn_id") == current_turn_id:
                    if turn.get("observation") is not None or prepared.tool_observation is None:
                        return turn
                    break
        for turn in reversed(turns):
            if turn.get("observation") is not None:
                return turn
        return turns[-1]

    def _event_plan_step(self, prepared: PreparedChatTurn) -> dict[str, Any] | None:
        plan_run = prepared.plan_run
        if not isinstance(plan_run, Mapping):
            return None
        steps = self._event_plan_steps(plan_run)
        if not steps:
            return None
        current_step_id = prepared.current_step_id
        if current_step_id:
            for step in steps:
                if step.get("step_id") == current_step_id:
                    return step
        for step in reversed(steps):
            if step.get("observation") is not None:
                return step
        return steps[-1]

    def _event_react_turns(self, react_run: Mapping[str, Any]) -> list[dict[str, Any]]:
        turns = react_run.get("turns")
        if not isinstance(turns, list):
            return []
        return [dict(turn) for turn in turns if isinstance(turn, Mapping)]

    def _event_plan_steps(self, plan_run: Mapping[str, Any]) -> list[dict[str, Any]]:
        steps = plan_run.get("steps")
        if not isinstance(steps, list):
            return []
        return [dict(step) for step in steps if isinstance(step, Mapping)]

    def _event_plan_execution_order(self, steps: list[dict[str, Any]]) -> list[str]:
        # steps 已按 executor 写入顺序保存，这里只投影前端需要的稳定执行序列。
        return [
            str(step.get("step_id"))
            for step in steps
            if step.get("status") in {"running", "waiting_user", "retrying", "succeeded"}
            and step.get("step_id") is not None
        ]

    def _event_attempted_tools(
            self,
            *,
            react_run: Mapping[str, Any],
            turn: Mapping[str, Any] | None,
    ) -> list[str]:
        metadata = react_run.get("metadata")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("attempted_tools"), list):
            return [str(tool_name) for tool_name in metadata["attempted_tools"]]
        attempted: list[str] = []
        seen: set[str] = set()
        for item in self._event_react_turns(react_run):
            tool_name = item.get("tool_name")
            if not tool_name and isinstance(item.get("action"), Mapping):
                tool_name = item["action"].get("tool_name")
            if not tool_name or str(tool_name) in seen:
                continue
            seen.add(str(tool_name))
            attempted.append(str(tool_name))
        if turn is not None:
            tool_name = turn.get("tool_name")
            if not tool_name and isinstance(turn.get("action"), Mapping):
                tool_name = turn["action"].get("tool_name")
            if tool_name and str(tool_name) not in seen:
                attempted.append(str(tool_name))
        return attempted

    def _event_latest_action_selection(self, react_run: Mapping[str, Any]) -> dict[str, Any] | None:
        metadata = react_run.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        latest = metadata.get("latest_action_selection")
        return dict(latest) if isinstance(latest, Mapping) else None

    def _event_action_validation_result(self, react_run: Mapping[str, Any]) -> str | None:
        latest = self._event_latest_action_selection(react_run)
        if latest is None:
            return None
        value = latest.get("validation_result")
        return str(value) if value is not None else None

    def _resolve_positive_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, float) and value.is_integer() and value > 0:
            return int(value)
        return None

    def _event_turn_id_from_payload(self, prepared: PreparedChatTurn) -> str:
        turn = self._event_react_turn(prepared)
        return str((turn or {}).get("turn_id") or "")

    def _event_step_id_from_payload(self, prepared: PreparedChatTurn) -> str:
        step = self._event_plan_step(prepared)
        return str((step or {}).get("step_id") or "")



