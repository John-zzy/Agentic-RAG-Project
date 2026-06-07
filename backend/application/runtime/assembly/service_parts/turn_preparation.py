from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from backend.application.runtime.api.chat.schemas import ChatRequest, RetrievalTrace
from backend.application.runtime.assembly.service_parts.contracts import (
    ChatServiceError,
    PreparedChatTurn,
    SceneMetadata,
)
from backend.platform.knowledge.sources import DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
from backend.platform.models.base.router import TaskComplexity


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

    def _empty_retrieval_trace(
            self,
            *,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
    ) -> RetrievalTrace:
        return RetrievalTrace(
            original_query=message,
            final_query=message,
            rewritten_query=None,
            tool_call_count=0,
            candidate_tools=list(
                self._resolve_runtime_candidate_retrieval_tools(
                    mounted_knowledge_sources=mounted_knowledge_sources,
                    request_id=request_id,
                )
            ),
            exit_reason="pending_chat_graph",
            final_decision=None,
            success=False,
            follow_up_question=None,
            raw_candidates_count=0,
            filtered_candidates_count=0,
            top_k_chunks=[],
            citations=[],
            knowledge_used=False,
            rounds=[],
        )

    def _prepare_chat_turn(self, payload: ChatRequest) -> PreparedChatTurn:
        """准备一次对话执行所需的共享上下文。

        执行流程：
        1. 生成请求追踪 ID（request_id）和会话标识（session_id）
        2. 确保会话存在且状态正常（创建新会话或续期旧会话）
        3. 调用 _prepare_existing_session_turn 完成：
           - 消息复杂度分析
           - Agent 执行模式选择（simple/react/plan）
           - 打包成 ChatGraph 可执行的 PreparedChatTurn 上下文对象

        注意：此方法执行完成后，RAG 尚未执行；mode selection、ReAct/Plan 和工具调用由 ChatGraph 分支完成。
        """

        # ── 第1步：生成请求标识 ──────────────────────────────────────
        # request_id 用于追踪整个请求链路，便于日志和调试
        # session_id 如果客户端未提供，则自动生成新的会话 ID
        request_id = uuid4().hex
        session_id = payload.session_id or uuid4().hex
        timestamp = datetime.now(UTC).isoformat()

        # ── 第2步：确保会话状态正常 ──────────────────────────────────
        # 普通 /chat 可以创建新会话，也可以续期旧会话
        # 如果会话已过期或场景不匹配，会抛出相应的业务异常
        self._ensure_session_ready(
            session_id=session_id,
            timestamp=timestamp,
            request_id=request_id,
            scene=self.scene_definition.scene,
        )

        # ── 第3步：准备完整的对话上下文 ──────────────────────────────
        # 此步骤只准备 ChatGraph 输入，不执行 RAG 检索或 Agent Runtime
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

        # ── 第1步：获取会话配置 ──────────────────────────────────────
        # 从会话存储中获取当前会话，提取允许使用的知识源列表
        # 如果会话不存在，使用默认的知识源配置
        session = self.session_store.get_session(session_id)
        mounted_knowledge_sources = (
            session.mounted_knowledge_sources
            if session is not None
            else DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
        )

        # ── 第2步：分析消息复杂度 ──────────────────────────────────────
        # 根据用户消息内容，判断任务复杂度（简单/中等/复杂）
        # 复杂度会影响后续的 Agent 模式选择和执行策略
        complexity = self.scene_definition.infer_complexity(message)

        # ── 第3步：构造 ChatGraph 输入上下文 ─────────────────────────────
        # Agent mode selection、ReAct/Plan 执行和 RAG 工具调用由 ChatGraph 分支节点完成。
        retrieval_trace = self._empty_retrieval_trace(
            message=message,
            mounted_knowledge_sources=tuple(mounted_knowledge_sources),
            request_id=request_id,
        )

        # ── 第4步：打包成只读上下文对象 ──────────────────────────────────
        # 把本轮对话需要的所有数据打包成 PreparedChatTurn
        # 这个对象会被后续的 JSON 序列化、SSE 流式推送、LangGraph 执行和持久化使用
        return PreparedChatTurn(
            session_id=session_id,                      # 会话 ID
            request_id=request_id,                      # 请求 ID（用于追踪和调试）
            timestamp=timestamp,                        # 时间戳
            user_message=message,                       # 用户原始消息
            documents=[],                               # graph 执行后回填检索文档
            tool_event={"stage": "agent_runtime_pending"},  # graph 执行后回填工具事件
            retrieval_trace=retrieval_trace,            # graph 执行后回填检索轨迹
            citations=[],                               # graph 执行后回填引用来源
            knowledge_used=False,                       # graph 执行后回填知识使用结果
            scene_metadata=self._scene_metadata(),      # 场景元数据（场景名、默认 Agent 等）
            complexity=complexity,                      # graph 内 mode selector 和模型路由使用
            final_decision=None,                        # graph 执行后回填 Agent 最终决策
            follow_up_question=None,                    # graph 执行后回填追问建议
            answer_mode="fallback",                    # graph 执行后回填答案模式
            hitl_clarification_enabled=hitl_clarification_enabled,  # 是否启用人工确认（HITL）
            agent_mode="",                              # graph select_mode 节点回填
            agent_mode_reason="pending_chat_graph",     # graph select_mode 节点回填
            agent_mode_signals={},                      # graph select_mode 节点回填
            react_run=None,                             # graph react_branch 节点回填
            plan_run=None,                              # graph plan_branch 节点回填
            current_turn_id=None,                       # graph 执行后回填当前轮次 ID
            current_step_id=None,                       # graph 执行后回填当前步骤 ID
            current_tool_call=None,                     # graph 执行后回填当前工具调用
            tool_observation=None,                      # graph 执行后回填工具观察结果
        )




