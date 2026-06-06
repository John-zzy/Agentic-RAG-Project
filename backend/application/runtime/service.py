from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from backend.application.runtime.api.chat.prompts import (
    build_direct_answer_prompt_template,
    build_rag_answer_prompt_template,
)
from backend.application.runtime.api.chat.schemas import (
    ChatRequest,
    ChatResumeRequest,
    ChatResumeResponse,
    ChatResponse,
)
from backend.application.runtime.assembly.service_parts.agent_runtime import ChatAgentRuntimeMixin
from backend.application.runtime.assembly.service_parts.answering import ChatAnsweringMixin
from backend.application.runtime.assembly.service_parts.citations import CitationMapper
from backend.application.runtime.assembly.service_parts.contracts import (
    AgentRuntimeExecutionResult,
    AnswerMode,
    ChatServiceError,
    PreparedChatTurn,
    RetrievalChainModel,
    RuntimeFinalDecision,
    SceneMetadata,
)
from backend.application.runtime.assembly.service_parts.events import ChatStreamEventMixin
from backend.application.runtime.assembly.service_parts.hitl import ChatHitlMixin
from backend.application.runtime.assembly.service_parts.responses import ChatResponseMixin
from backend.application.runtime.assembly.service_parts.turn_preparation import ChatTurnPreparationMixin
from backend.application.runtime.assembly.runtime_factory import ChatGraphRuntime
from backend.application.runtime.stream_events import ChatStreamEvent, GraphStreamEventMapper
from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.sources import (
    DEFAULT_MOUNTED_KNOWLEDGE_SOURCES,
    normalize_mounted_knowledge_sources,
)
from backend.platform.memory.base.session_store import SQLiteSessionStore, SessionRecord
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.agent_runtime.mode_selector import MinimalModeSelector
from backend.platform.models.base.router import TaskComplexity
from backend.platform.models.llm.client import ModelClient, model_client
from backend.platform.rag.retrieval.documents import DocumentRetrievalService
from backend.scenes.base import SceneDefinition
from backend.scenes.generic_assistant.definition import GenericAssistantBusinessExtension
from backend.scenes.generic_assistant.hitl import GenericAssistantHitlPlanner
from backend.scenes.registry import build_default_scene_definitions


class SceneRegistry:
    """维护可用场景定义，并解析当前激活场景。"""

    def __init__(self, definitions: list[SceneDefinition], default_scene: str) -> None:
        self._definitions = {definition.scene: definition for definition in definitions}
        self._default_scene = default_scene
        if default_scene not in self._definitions:
            supported = ", ".join(sorted(self._definitions))
            raise ValueError(
                f"Unknown active scene '{default_scene}'. Expected one of: {supported}."
            )

    @property
    def default_scene(self) -> str:
        """返回默认场景标识。"""
        return self._default_scene

    def list_definitions(self) -> tuple[SceneDefinition, ...]:
        """返回全部已注册场景定义。"""
        return tuple(self._definitions.values())

    def is_supported(self, scene: str) -> bool:
        """检查场景是否已注册。"""
        return scene in self._definitions

    def get_definition(self, scene: str) -> SceneDefinition:
        """按场景标识返回场景定义。"""
        return self._definitions[scene]

    def get_default_definition(self) -> SceneDefinition:
        """返回默认场景定义。"""
        return self.get_definition(self._default_scene)


class ChatService(
    ChatTurnPreparationMixin,
    ChatAgentRuntimeMixin,
    ChatHitlMixin,
    ChatAnsweringMixin,
    ChatStreamEventMixin,
    ChatResponseMixin,
):
    """执行单个场景下的检索、生成和会话持久化流程。"""

    def __init__(
            self,
            *,
            scene_definition: SceneDefinition,
            app_settings: AppSettings | None = None,
            session_store: SQLiteSessionStore | None = None,
            context_builder: PromptContextBuilder | None = None,
            model: RetrievalChainModel | None = None,
            graph_runtime: ChatGraphRuntime | None = None,
    ) -> None:
        """初始化场景聊天服务依赖。"""
        self.settings = app_settings or settings
        self.scene_definition = scene_definition
        self.session_store = session_store or SQLiteSessionStore(self.settings)
        self.context_builder = context_builder or PromptContextBuilder(
            window_size=self.settings.session.window_size
        )
        self.model = model or model_client
        self.graph_runtime = graph_runtime or ChatGraphRuntime.from_settings(self.settings)
        self._rag_answer_template = build_rag_answer_prompt_template(
            system_prompt=scene_definition.system_prompt
        )
        self._direct_answer_template = build_direct_answer_prompt_template(
            system_prompt=scene_definition.system_prompt
        )
        self._retriever = scene_definition.build_retriever()
        self._generic_hitl_planner = GenericAssistantHitlPlanner.from_scene_metadata(
            scene_definition.metadata,
            suggestion_model=self.model,
        )
        self._mode_selector = MinimalModeSelector()
        self._citation_mapper = CitationMapper()
        self._stream_event_mapper = GraphStreamEventMapper()
        self._answer_base_runnables: dict[TaskComplexity, Any] = {}

    def chat(self, payload: ChatRequest) -> ChatResponse:
        """执行一次完整对话流程，并返回统一结构。"""
        prepared = self._prepare_chat_turn(payload)
        prepared, answer, citations, run_id, graph_state = self._generate_answer(prepared)
        if graph_state.get("status") == "waiting_user":
            return self._build_hitl_wait_response_from_graph_state(
                prepared=prepared,
                result_state=graph_state,
            )
        self._persist_turn(prepared=prepared, answer=answer, citations=citations)
        return self._build_chat_response(
            prepared=prepared,
            answer=answer,
            citations=citations,
            run_id=run_id,
        )

    def resume(self, payload: ChatResumeRequest) -> ChatResumeResponse:
        """非流式恢复 HITL 等待点，并返回本次人工动作结果。"""
        request_id = uuid4().hex
        result = self._run_hitl_resume(payload=payload, request_id=request_id)
        return self._build_resume_response(
            payload=payload,
            request_id=request_id,
            result_state=result.state,
        )

    def chat_stream(self, payload: ChatRequest) -> Iterator[ChatStreamEvent]:
        """执行一次流式对话流程，并产出结构化 SSE 事件。

        流程概览：
        1. 准备回合上下文(session、场景、历史窗口)
        2. 提前捕获历史快照(必须在 ChatGraph 写入本轮消息之前)
        3. 通过 ChatGraph 完成完整的编排流程(模式选择、ReAct/Plan 执行、回答生成)
        4. 依次产出 SSE 事件推送给前端
        """
        # ── 第1步：准备回合上下文 ──
        # 将 API payload 解析为标准聊天回合，包含 session、场景配置、历史窗口等
        prepared = self._prepare_chat_turn(payload)

        # ── 第2步：提前捕获历史快照 ──
        # 记录的是"本轮 Agent 执行前，模型看到的历史消息"，
        # 必须在 ChatGraph 写入本轮 human/ai 消息之前拿到，否则会被污染
        history_event = self._build_history_event(prepared)

        # ── 第3步：通过 ChatGraph 执行完整的编排流程 ──
        # 图内依次完成：模式选择 → ReAct/Plan 分支执行 → 回答模式判定 → HITL 判断 → 生成最终回答
        prepared, answer, citations, run_id, graph_state = self._generate_answer(prepared)

        # ── 第4步：依次产出 SSE 事件 ──

        # 通知前端：graph run 已创建，附带本轮元信息（session、场景、agent 模式等）
        yield self._map_graph_stream_event(
            "graph_run_created",
            {
                "session_id": prepared.session_id,
                "request_id": prepared.request_id,
                "knowledge_used": prepared.knowledge_used,
                "scene": prepared.scene_metadata.scene,
                "agent": prepared.scene_metadata.agent,
                "agent_mode": prepared.agent_mode,
                "state": "running",
                "state_event": "run_start",
                "run_id": run_id,
            },
        )

        # 推送历史上下文快照，方便前端展示"模型看到了什么"
        yield self._map_graph_stream_event(
            "history_snapshot",
            history_event,
        )

        # 推送 Agent/RAG 工具的检索结果和审计信息
        # 即使 ReAct 没有调用工具，也会带上 direct_answer / no_evidence 等分支信息
        yield self._map_graph_stream_event(
            "retrieval_tool_result",
            self._build_tool_event(prepared),
        )

        # 如果 Agent 进入 HITL（人工确认）等待，SSE 流到此结束
        # 后续需要前端调用 resume_stream 恢复同一个 checkpoint
        if graph_state.get("status") == "waiting_user":
            hitl_response = self._build_hitl_wait_response_from_graph_state(
                prepared=prepared,
                result_state=graph_state,
            )
            yield self._map_graph_stream_event(
                "human_waiting",
                self._build_hitl_wait_event(hitl_response),
            )
            return

        # 推送最终回答（完整文本作为一个 chunk 发出）
        yield self._map_graph_stream_event("answer_chunk", {"delta": answer})

        # 持久化本轮对话记录（会话记忆，非 graph checkpoint）
        self._persist_turn(prepared=prepared, answer=answer, citations=citations)

        # 推送完成事件，携带完整的响应体
        response = self._build_chat_response(
            prepared=prepared,
            answer=answer,
            citations=citations,
            run_id=run_id,
        )
        yield self._map_graph_stream_event("graph_run_succeeded", response.model_dump())

    def resume_stream(self, payload: ChatResumeRequest) -> Iterator[ChatStreamEvent]:
        """流式恢复 HITL 等待点；只有恢复被接受后才发送 resume 事件。"""
        request_id = uuid4().hex
        try:
            result = self._run_hitl_resume(payload=payload, request_id=request_id)
        except ChatServiceError as exc:
            yield self._map_graph_stream_event(
                "graph_run_failed",
                {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": exc.request_id,
                    "final_state": "failed",
                },
            )
            return

        yield self._map_graph_stream_event(
            "human_resume",
            {
                "session_id": payload.session_id,
                "request_id": request_id,
                "interrupt_id": payload.interrupt_id,
                "action": payload.action,
                "state_event": f"resume_{payload.action}",
            },
        )
        response = self._build_resume_response(
            payload=payload,
            request_id=request_id,
            result_state=result.state,
        )
        yield self._map_graph_stream_event("graph_run_succeeded", response.model_dump())

    def delete_session(self, session_id: str) -> int:
        """清理 graph thread 后删除会话读模型，避免 memory 层依赖 workflow。"""
        self.graph_runtime.delete_session_thread(session_id)
        return self.session_store.delete_session(session_id=session_id)


class ActiveSceneChatService:
    """统一 `/chat` 入口，通过会话绑定场景分发请求。"""

    def __init__(
            self,
            *,
            scene_registry: SceneRegistry,
            app_settings: AppSettings | None = None,
            knowledge_service: object | None = None,
            session_store: SQLiteSessionStore | None = None,
            context_builder: PromptContextBuilder | None = None,
            model: RetrievalChainModel | None = None,
            graph_runtime: ChatGraphRuntime | None = None,
    ) -> None:
        """初始化运行时依赖，并缓存当前激活场景服务。"""
        del knowledge_service
        self.settings = app_settings or settings
        self.scene_registry = scene_registry
        self.session_store = session_store or SQLiteSessionStore(self.settings)
        self.context_builder = context_builder or PromptContextBuilder(
            window_size=self.settings.session.window_size
        )
        self.model = model or model_client
        self.graph_runtime = graph_runtime or ChatGraphRuntime.from_settings(self.settings)
        self._scene_services: dict[str, ChatService] = {}

    def chat(self, payload: ChatRequest) -> ChatResponse:
        """将请求转发给会话绑定的场景。"""
        scene = self.resolve_session_scene(payload.session_id)
        return self._get_scene_service(scene).chat(payload)

    def resume(self, payload: ChatResumeRequest) -> ChatResumeResponse:
        """将 HITL resume 请求转发给会话绑定的场景。"""
        scene = self.resolve_session_scene(payload.session_id)
        return self._get_scene_service(scene).resume(payload)

    def chat_stream(self, payload: ChatRequest) -> Iterator[ChatStreamEvent]:
        """将流式请求转发给会话绑定的场景。"""
        scene = self.resolve_session_scene(payload.session_id)
        yield from self._get_scene_service(scene).chat_stream(payload)

    def resume_stream(self, payload: ChatResumeRequest) -> Iterator[ChatStreamEvent]:
        """将流式 HITL resume 请求转发给会话绑定的场景。"""
        scene = self.resolve_session_scene(payload.session_id)
        yield from self._get_scene_service(scene).resume_stream(payload)

    def delete_session(self, session_id: str) -> int:
        """由 application facade 编排 session 与 LangGraph thread 的一致清理。"""
        self.graph_runtime.delete_session_thread(session_id)
        return self.session_store.delete_session(session_id=session_id)

    def list_scenes(self) -> tuple[SceneDefinition, ...]:
        """列出所有可用场景定义。"""
        return self.scene_registry.list_definitions()

    def default_scene(self) -> str:
        """返回默认场景标识。"""
        return self.scene_registry.default_scene

    def validate_scene(self, scene: str) -> str:
        """校验并返回合法场景标识。"""
        if not self.scene_registry.is_supported(scene):
            supported = ", ".join(
                definition.scene for definition in self.scene_registry.list_definitions()
            )
            raise ValueError(f"Unknown scene '{scene}'. Expected one of: {supported}.")
        return scene

    def create_session(
            self,
            scene: str | None = None,
            mounted_knowledge_sources: list[str] | tuple[str, ...] | None = None,
    ) -> SessionRecord:
        """创建绑定场景的新会话，并保存规范化后的挂载知识源。"""
        resolved_scene = self.validate_scene(scene or self.default_scene())
        resolved_sources = self.validate_mounted_knowledge_sources(mounted_knowledge_sources)
        session_id = uuid4().hex
        return self.session_store.create_session(
            session_id=session_id,
            scene=resolved_scene,
            mounted_knowledge_sources=resolved_sources,
        )

    def validate_mounted_knowledge_sources(
            self,
            mounted_knowledge_sources: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """校验并规范化会话挂载知识源。"""
        return normalize_mounted_knowledge_sources(mounted_knowledge_sources)

    def default_mounted_knowledge_sources(self) -> tuple[str, ...]:
        """返回系统默认挂载的知识源列表。"""
        return DEFAULT_MOUNTED_KNOWLEDGE_SOURCES

    def resolve_session_scene(self, session_id: str | None) -> str:
        """解析会话绑定场景；无会话时返回默认场景。"""
        if not session_id:
            return self.default_scene()
        session = self.session_store.get_session(session_id)
        if session is None:
            raise ChatServiceError(
                status_code=404,
                code="SESSION_NOT_FOUND",
                message="Session was not found. Please create a new session before continuing.",
                request_id="N/A",
            )
        return self.validate_scene(session.scene)

    def _get_scene_service(self, scene: str) -> ChatService:
        """按场景懒加载 ChatService。"""
        cached = self._scene_services.get(scene)
        if cached is not None:
            return cached

        service = ChatService(
            scene_definition=self.scene_registry.get_definition(scene),
            app_settings=self.settings,
            session_store=self.session_store,
            context_builder=self.context_builder,
            model=self.model,
            graph_runtime=self.graph_runtime,
        )
        self._scene_services[scene] = service
        return service


SceneChatService = ActiveSceneChatService


def build_default_scene_registry(
        *,
        app_settings: AppSettings | None = None,
        knowledge_service: object | None = None,
        document_retrieval_service: DocumentRetrievalService | None = None,
        generic_business_extensions: tuple[GenericAssistantBusinessExtension, ...] | None = None,
        include_default_business_extensions: bool = True,
) -> SceneRegistry:
    """构建默认场景注册表。"""
    resolved_settings = app_settings or settings
    definitions = list(
        build_default_scene_definitions(
            app_settings=resolved_settings,
            knowledge_service=knowledge_service,
            document_retrieval_service=document_retrieval_service,
            generic_business_extensions=generic_business_extensions,
            include_default_business_extensions=include_default_business_extensions,
        )
    )
    return SceneRegistry(definitions=definitions, default_scene=resolved_settings.app.active_scene)


def create_chat_service(
        app_settings: AppSettings | None = None,
        knowledge_service: object | None = None,
        document_retrieval_service: DocumentRetrievalService | None = None,
        generic_business_extensions: tuple[GenericAssistantBusinessExtension, ...] | None = None,
        include_default_business_extensions: bool = True,
        session_store: SQLiteSessionStore | None = None,
        context_builder: PromptContextBuilder | None = None,
        model: ModelClient | None = None,
        graph_runtime: ChatGraphRuntime | None = None,
) -> ActiveSceneChatService:
    """聊天服务工厂函数，返回统一场景运行时服务。"""
    resolved_settings = app_settings or settings
    scene_registry = build_default_scene_registry(
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        document_retrieval_service=document_retrieval_service,
        generic_business_extensions=generic_business_extensions,
        include_default_business_extensions=include_default_business_extensions,
    )
    return ActiveSceneChatService(
        scene_registry=scene_registry,
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        session_store=session_store,
        context_builder=context_builder,
        model=model,
        graph_runtime=graph_runtime,
    )
