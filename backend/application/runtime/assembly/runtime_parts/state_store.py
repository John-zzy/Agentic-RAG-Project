from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from backend.platform.agent_runtime.chat_graph.contracts import (
    HitlResumeError,
    HistoryLoader,
    PreparedGraphTurn,
)
from backend.platform.workflow.langgraph.config import build_runtime_graph_config
from backend.platform.workflow.langgraph.lifecycle import GraphRunRef
from backend.platform.workflow.langgraph.state import RuntimeGraphState, build_runtime_graph_state


class RuntimeStateStoreMixin:
    def build_config(self, prepared: PreparedGraphTurn) -> dict[str, Any]:
        scene = getattr(prepared.scene_metadata, "scene", None)
        agent = getattr(prepared.scene_metadata, "agent", None)
        return build_runtime_graph_config(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            metadata={
                "scene": scene,
                "agent": agent,
                "agent_mode": getattr(prepared, "agent_mode", "react"),
                "answer_mode": prepared.answer_mode,
                "final_decision": prepared.final_decision,
            },
        )

    def _mark_agent_run_started(
        self,
        *,
        run: GraphRunRef,
        prepared: PreparedGraphTurn,
    ) -> None:
        """按顶层 Agent mode 记录 run 级 lifecycle；turn/step 状态只进 payload。"""
        if getattr(prepared, "agent_mode", "react") == "plan":
            self.lifecycle.mark_planning(run)
        self.lifecycle.mark_running(run)

    def delete_session_thread(self, session_id: str) -> None:
        """删除这个会话在 LangGraph 里的保存状态。"""
        self.checkpointer.delete_thread(session_id)

    def build_input_state(
        self,
        *,
        prepared: PreparedGraphTurn,
        history_loader: HistoryLoader,
        config: dict[str, Any],
        run_id: str,
    ) -> RuntimeGraphState:
        history_messages = self._history_seed(
            prepared=prepared,
            history_loader=history_loader,
            config=config,
        )
        return build_runtime_graph_state(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            scene=getattr(prepared.scene_metadata, "scene", None),
            messages=[
                *history_messages,
                HumanMessage(content=prepared.user_message),
            ],
            knowledge_used=prepared.knowledge_used,
            citations=[citation.model_dump() for citation in prepared.citations],
            retrieval_trace=prepared.retrieval_trace.model_dump(),
            metadata=dict(config["metadata"]),
            answer_mode=prepared.answer_mode,
            status="running",
            run_id=run_id,
            state_event="run_start",
            agent_mode=str(getattr(prepared, "agent_mode", "react")),
            agent_mode_reason=getattr(prepared, "agent_mode_reason", None),
            agent_mode_signals=getattr(prepared, "agent_mode_signals", None),
            react_run=getattr(prepared, "react_run", None),
            plan_run=getattr(prepared, "plan_run", None),
            current_turn_id=getattr(prepared, "current_turn_id", None),
            current_step_id=getattr(prepared, "current_step_id", None),
            current_tool_call=getattr(prepared, "current_tool_call", None),
        )

    def _persist_state_update(
        self,
        *,
        state: RuntimeGraphState,
        config: dict[str, Any],
        update: Mapping[str, Any],
    ) -> RuntimeGraphState:
        """用一个很小的 graph 写状态，确保仍走 LangGraph 的 checkpoint 保存流程。"""
        builder = StateGraph(RuntimeGraphState)

        def update_node(current_state: RuntimeGraphState) -> dict[str, Any]:
            del current_state
            return dict(update)

        builder.add_node("hitl_state_update", update_node)
        builder.add_edge(START, "hitl_state_update")
        builder.add_edge("hitl_state_update", END)
        graph = builder.compile(checkpointer=self.checkpointer)
        return graph.invoke(state, config)

    def _load_or_build_thread_state(
        self,
        *,
        session_id: str,
        request_id: str,
        config: dict[str, Any],
        require_checkpoint: bool = False,
    ) -> RuntimeGraphState:
        """读取会话最新保存状态；没有保存过时就创建一个空状态。"""
        checkpoint = self.checkpointer.get_tuple(config)
        if checkpoint is None:
            if require_checkpoint:
                raise HitlResumeError("No checkpoint found for HITL resume.")
            return build_runtime_graph_state(
                session_id=session_id,
                request_id=request_id,
                metadata=dict(config["metadata"]),
            )

        values = dict(checkpoint.checkpoint.get("channel_values") or {})
        return build_runtime_graph_state(
            session_id=str(values.get("session_id") or session_id),
            request_id=str(values.get("request_id") or request_id),
            scene=values.get("scene"),
            messages=list(values.get("messages") or ()),
            answer=str(values.get("answer") or ""),
            knowledge_used=bool(values.get("knowledge_used", False)),
            citations=list(values.get("citations") or ()),
            retrieval_trace=dict(values.get("retrieval_trace") or {}),
            metadata=dict(values.get("metadata") or {}),
            answer_mode=values.get("answer_mode"),
            status=values.get("status", "running"),
            run_id=values.get("run_id"),
            state_event=values.get("state_event"),
            final_state=values.get("final_state"),
            retry_attempt=int(values.get("retry_attempt") or 0),
            retry_metadata=dict(values.get("retry_metadata") or {}),
            hitl=values.get("hitl"),
            hitl_resume=values.get("hitl_resume"),
            agent_mode=values.get("agent_mode"),
            agent_mode_reason=values.get("agent_mode_reason"),
            agent_mode_signals=values.get("agent_mode_signals"),
            react_run=values.get("react_run"),
            plan_run=values.get("plan_run"),
            current_turn_id=values.get("current_turn_id"),
            current_step_id=values.get("current_step_id"),
            current_tool_call=values.get("current_tool_call"),
        )
    def _history_seed(
        self,
        *,
        prepared: PreparedGraphTurn,
        history_loader: HistoryLoader,
        config: dict[str, Any],
    ) -> list[BaseMessage]:
        checkpoint = self.checkpointer.get_tuple(config)
        if checkpoint is not None:
            return []
        # 只有第一次进入 LangGraph 时才带入旧历史，避免后续重复塞历史消息。
        return list(history_loader(prepared))




