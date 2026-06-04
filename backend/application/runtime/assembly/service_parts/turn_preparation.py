from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from backend.application.runtime.api.chat.schemas import ChatRequest
from backend.application.runtime.assembly.service_parts.contracts import (
    ChatServiceError,
    PreparedChatTurn,
    SceneMetadata,
)
from backend.platform.knowledge.sources import DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
from backend.platform.models.base.router import TaskComplexity
from backend.platform.agent_runtime.mode_selector import ModeSelection


class ChatTurnPreparationMixin:
    def _ensure_session_ready(
            self,
            *,
            session_id: str,
            timestamp: str,
            request_id: str,
            scene: str,
    ) -> None:
        """创建或续期当前会话。"""
        self.session_store.cleanup_expired_sessions(now=timestamp)
        session = self.session_store.get_session(session_id)
        if session is None:
            self.session_store.create_session(
                session_id=session_id,
                scene=scene,
                mounted_knowledge_sources=DEFAULT_MOUNTED_KNOWLEDGE_SOURCES,
                now=timestamp,
            )
            return
        if session.status == "expired":
            raise ChatServiceError(
                status_code=409,
                code="SESSION_EXPIRED",
                message="Session has expired. Please create a new session before continuing.",
                request_id=request_id,
            )
        if session.scene != scene:
            raise ChatServiceError(
                status_code=409,
                code="SCENE_SESSION_MISMATCH",
                message="Session is bound to a different scene. Please create a new session for this scene.",
                request_id=request_id,
            )
        self.session_store.touch_session(session_id=session_id, now=timestamp)

    def _ensure_resume_session_ready(
            self,
            *,
            session_id: str,
            timestamp: str,
            request_id: str,
            scene: str,
    ) -> None:
        """resume 只能恢复已有会话，不能静默创建新会话。"""
        self.session_store.cleanup_expired_sessions(now=timestamp)
        session = self.session_store.get_session(session_id)
        if session is None:
            raise ChatServiceError(
                status_code=404,
                code="SESSION_NOT_FOUND",
                message="Session was not found. Please create a new session before continuing.",
                request_id=request_id,
            )
        if session.status == "expired":
            raise ChatServiceError(
                status_code=409,
                code="SESSION_EXPIRED",
                message="Session has expired. Please create a new session before continuing.",
                request_id=request_id,
            )
        if session.scene != scene:
            raise ChatServiceError(
                status_code=409,
                code="SCENE_SESSION_MISMATCH",
                message="Session is bound to a different scene. Please create a new session for this scene.",
                request_id=request_id,
            )
        self.session_store.touch_session(session_id=session_id, now=timestamp)

    def _scene_metadata(self) -> SceneMetadata:
        """从场景定义中提取响应元数据。"""
        default_agent = self.scene_definition.metadata.get("default_agent")
        return SceneMetadata(
            scene=self.scene_definition.scene,
            agent=str(default_agent) if isinstance(default_agent, str) else None,
        )

    def _select_agent_mode(
            self,
            *,
            message: str,
            complexity: TaskComplexity | None,
            mounted_knowledge_sources: tuple[str, ...],
    ) -> ModeSelection:
        """选择顶层 Agent 模式；RAG 内部多轮不会直接决定顶层 ReAct/Plan。"""
        return self._mode_selector.select(
            message=message,
            complexity=complexity,
            mounted_knowledge_sources=mounted_knowledge_sources,
            scene_metadata=self.scene_definition.metadata,
        )

    def _prepare_chat_turn(self, payload: ChatRequest) -> PreparedChatTurn:
        """准备一次对话执行所需的共享上下文。"""
        request_id = uuid4().hex
        session_id = payload.session_id or uuid4().hex
        timestamp = datetime.now(UTC).isoformat()

        # 普通 /chat 可以创建新会话，也可以续期旧会话。
        self._ensure_session_ready(
            session_id=session_id,
            timestamp=timestamp,
            request_id=request_id,
            scene=self.scene_definition.scene,
        )
        return self._prepare_existing_session_turn(
            session_id=session_id,
            request_id=request_id,
            timestamp=timestamp,
            message=payload.message,
            hitl_clarification_enabled=bool(
                getattr(payload, "hitl_clarification_enabled", False)
            ),
        )

    def _prepare_existing_session_turn(
            self,
            *,
            session_id: str,
            request_id: str,
            timestamp: str,
            message: str,
            hitl_clarification_enabled: bool = False,
    ) -> PreparedChatTurn:
        """基于已存在的会话准备一轮 Agent Runtime 上下文，供 /chat 和 HITL respond 复用。"""
        # 会话里记录了本轮允许使用哪些知识源，例如只查 documents，或同时查 ecommerce。
        session = self.session_store.get_session(session_id)
        mounted_knowledge_sources = (
            session.mounted_knowledge_sources
            if session is not None
            else DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
        )
        complexity = self.scene_definition.infer_complexity(message)
        mode_selection = self._select_agent_mode(
            message=message,
            complexity=complexity,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
        agent_result = self._execute_agent_runtime(
            session_id=session_id,
            request_id=request_id,
            message=message,
            complexity=complexity,
            mounted_knowledge_sources=tuple(mounted_knowledge_sources),
            mode_selection=mode_selection,
        )
        # 把后续 JSON、SSE、LangGraph 和持久化都会用到的数据打成一个只读上下文对象。
        return PreparedChatTurn(
            session_id=session_id,
            request_id=request_id,
            timestamp=timestamp,
            user_message=message,
            documents=agent_result.documents,
            tool_event=agent_result.tool_event,
            retrieval_trace=agent_result.retrieval_trace,
            citations=agent_result.citations,
            knowledge_used=agent_result.knowledge_used,
            scene_metadata=self._scene_metadata(),
            complexity=complexity if agent_result.knowledge_used else None,
            final_decision=agent_result.final_decision,
            follow_up_question=agent_result.follow_up_question,
            answer_mode=agent_result.answer_mode,
            hitl_clarification_enabled=hitl_clarification_enabled,
            agent_mode=mode_selection.mode,
            agent_mode_reason=mode_selection.reason,
            agent_mode_signals=mode_selection.signals,
            react_run=agent_result.react_run,
            plan_run=agent_result.plan_run,
            current_turn_id=agent_result.current_turn_id,
            current_step_id=agent_result.current_step_id,
            current_tool_call=agent_result.current_tool_call,
            tool_observation=agent_result.tool_observation,
        )




