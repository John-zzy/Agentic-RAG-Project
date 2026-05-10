from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.application.runtime.api.chat.prompts import build_rag_answer_prompt_template
from backend.application.runtime.api.chat.schemas import ChatRequest, ChatResponse, Citation
from backend.platform.config.settings import AppSettings, settings
from backend.scenes.base import SceneDefinition
from backend.scenes.ecommerce.definition import build_ecommerce_scene_definition
from backend.scenes.generic_assistant.definition import (
    build_generic_assistant_scene_definition,
)
from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.memory.base.session_store import SQLiteSessionStore, SessionTurn
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.models.base.router import TaskComplexity
from backend.platform.models.llm.client import ModelClient, model_client


class RetrievalChainModel(Protocol):
    """定义运行时依赖的最小模型构建协议。"""

    def build_chat_model_for_complexity(self, complexity: TaskComplexity) -> Any:
        """按复杂度构建可用于 RAG 链的聊天模型实例。"""
        ...


class ChatServiceError(RuntimeError):
    """封装可返回给 API 层的业务错误。"""

    def __init__(self, *, status_code: int, code: str, message: str, request_id: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(message)


@dataclass(frozen=True)
class SceneMetadata:
    """描述统一聊天响应需要返回的场景元数据。"""

    scene: str
    agent: str | None = None


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


class ChatService:
    """执行单个场景下的检索、生成和会话持久化流程。"""

    def __init__(
        self,
        *,
        scene_definition: SceneDefinition,
        app_settings: AppSettings | None = None,
        session_store: SQLiteSessionStore | None = None,
        context_builder: PromptContextBuilder | None = None,
        model: RetrievalChainModel | None = None,
    ) -> None:
        """初始化场景聊天服务依赖。"""
        self.settings = app_settings or settings
        self.scene_definition = scene_definition
        self.session_store = session_store or SQLiteSessionStore(self.settings)
        self.context_builder = context_builder or PromptContextBuilder(
            window_size=self.settings.session.window_size
        )
        self.model = model or model_client
        self._rag_answer_template = build_rag_answer_prompt_template(
            system_prompt=scene_definition.system_prompt
        )
        self._retriever = scene_definition.build_retriever()

    def chat(self, payload: ChatRequest) -> ChatResponse:
        """执行一次完整对话流程，并返回统一结构。"""
        request_id = uuid4().hex
        session_id = payload.session_id or uuid4().hex
        timestamp = datetime.now(UTC).isoformat()
        resolved_scene = self.scene_definition.scene

        if payload.stream:
            raise ChatServiceError(
                status_code=501,
                code="STREAM_NOT_SUPPORTED",
                message="Streaming mode is reserved but not enabled on this endpoint yet.",
                request_id=request_id,
            )

        self._ensure_session_ready(
            session_id=session_id,
            timestamp=timestamp,
            request_id=request_id,
            scene=resolved_scene,
        )
        history_turns = self.session_store.get_recent_turns(
            session_id=session_id,
            limit=self.settings.session.window_size,
        )
        history_text = self._format_history(history_turns)

        documents = self._retrieve_documents(payload.message)
        citations = self._citations_from_documents(documents)
        knowledge_used = len(citations) > 0
        scene_metadata = self._scene_metadata()

        if knowledge_used:
            complexity = self.scene_definition.infer_complexity(payload.message)
            answer, citations = self._invoke_chain_with_docs(
                documents=documents,
                user_message=payload.message,
                history_text=history_text,
                complexity=complexity,
                request_id=request_id,
                fallback_citations=citations,
            )
        else:
            answer = self.scene_definition.fallback_policy.no_hit_message

        self.session_store.append_turn(
            session_id=session_id,
            request_id=request_id,
            user_message=payload.message,
            assistant_answer=answer,
            retrieval_snippets=[citation.model_dump() for citation in citations],
            timestamp=timestamp,
        )

        return ChatResponse(
            session_id=session_id,
            request_id=request_id,
            answer=answer,
            knowledge_used=knowledge_used,
            scene=scene_metadata.scene,
            agent=scene_metadata.agent,
            citations=citations,
        )

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
            self.session_store.create_session(session_id=session_id, scene=scene, now=timestamp)
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

    def _retrieve_documents(self, message: str) -> list[Document]:
        """兼容普通 retriever 与 agentic retriever 的检索输出。"""
        if hasattr(self._retriever, "retrieve_with_trace"):
            outcome = self._retriever.retrieve_with_trace(message)  # type: ignore[attr-defined]
            return list(outcome.documents)
        if hasattr(self._retriever, "search"):
            return list(self._retriever.search(query=message))  # type: ignore[attr-defined]
        if isinstance(self._retriever, BaseRetriever):
            return list(self._retriever.invoke(message))
        raise TypeError("Retriever does not support document retrieval.")

    def _scene_metadata(self) -> SceneMetadata:
        """从场景定义中提取响应元数据。"""
        default_agent = self.scene_definition.metadata.get("default_agent")
        return SceneMetadata(
            scene=self.scene_definition.scene,
            agent=str(default_agent) if isinstance(default_agent, str) else None,
        )

    def _citations_from_documents(self, documents: list[Document]) -> list[Citation]:
        """从检索文档中提取并去重引用信息。"""
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()

        for doc in documents:
            namespace = str(doc.metadata.get("namespace", "knowledge"))
            citation_id = str(doc.metadata.get("citation_id", "unknown"))
            key = (namespace, citation_id)
            if key in seen:
                continue
            seen.add(key)

            score = doc.metadata.get("score")
            normalized_score = float(score) if isinstance(score, int | float) else None
            snippet = truncate_snippet(doc.page_content)
            if snippet:
                citations.append(
                    Citation(
                        citation_id=citation_id,
                        namespace=namespace,
                        snippet=snippet,
                        score=normalized_score,
                    )
                )

        return citations

    def _invoke_chain_with_docs(
        self,
        *,
        documents: list[Document],
        user_message: str,
        history_text: str,
        complexity: TaskComplexity,
        request_id: str,
        fallback_citations: list[Citation],
    ) -> tuple[str, list[Citation]]:
        """调用模型链生成答案，并返回答案与引用。"""
        try:
            llm = self.model.build_chat_model_for_complexity(complexity)
            combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=self._rag_answer_template)
            result = combine_docs_chain.invoke(
                {
                    "context": documents,
                    "input": user_message,
                    "history": history_text,
                }
            )
        except Exception as exc:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=request_id,
            ) from exc

        answer = str(result).strip() if isinstance(result, str) else str(result.get("answer", "")).strip()
        if not answer:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_EMPTY_RESPONSE",
                message="Model returned empty response.",
                request_id=request_id,
            )

        answer_citations = self._citations_from_documents(documents)
        return answer, answer_citations if answer_citations else fallback_citations

    def _format_history(self, history_turns: list[SessionTurn]) -> str:
        """将历史轮次格式化为模型可读文本。"""
        if not history_turns:
            return "(empty)"

        lines: list[str] = []
        for turn in history_turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.assistant_answer}")
        return "\n".join(lines)


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
        self._scene_services: dict[str, ChatService] = {}

    def chat(self, payload: ChatRequest) -> ChatResponse:
        """将请求转发给会话绑定的场景。"""
        scene = self.resolve_session_scene(payload.session_id)
        return self._get_scene_service(scene).chat(payload)

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

    def create_session(self, scene: str | None = None) -> str:
        """创建绑定场景的新会话。"""
        resolved_scene = self.validate_scene(scene or self.default_scene())
        session_id = uuid4().hex
        self.session_store.create_session(session_id=session_id, scene=resolved_scene)
        return session_id

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
        )
        self._scene_services[scene] = service
        return service


SceneChatService = ActiveSceneChatService


def build_default_scene_registry(
    *,
    app_settings: AppSettings | None = None,
    knowledge_service: object | None = None,
) -> SceneRegistry:
    """构建默认场景注册表。"""
    resolved_settings = app_settings or settings
    definitions = [
        build_generic_assistant_scene_definition(
            app_settings=resolved_settings,
            knowledge_service=knowledge_service,
        ),
        build_ecommerce_scene_definition(
            app_settings=resolved_settings,
            knowledge_service=knowledge_service,
        ),
    ]
    return SceneRegistry(definitions=definitions, default_scene=resolved_settings.app.active_scene)


def create_chat_service(
    app_settings: AppSettings | None = None,
    knowledge_service: object | None = None,
    session_store: SQLiteSessionStore | None = None,
    context_builder: PromptContextBuilder | None = None,
    model: ModelClient | None = None,
) -> ActiveSceneChatService:
    """聊天服务工厂函数，返回统一场景运行时服务。"""
    resolved_settings = app_settings or settings
    scene_registry = build_default_scene_registry(
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
    )
    return ActiveSceneChatService(
        scene_registry=scene_registry,
        app_settings=resolved_settings,
        knowledge_service=knowledge_service,
        session_store=session_store,
        context_builder=context_builder,
        model=model,
    )
