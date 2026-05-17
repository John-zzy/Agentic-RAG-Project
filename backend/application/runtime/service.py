from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Protocol
from uuid import uuid4

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.application.runtime.api.chat.prompts import build_rag_answer_prompt_template
from backend.application.runtime.api.chat.schemas import ChatRequest, ChatResponse, Citation
from backend.platform.config.settings import AppSettings, settings
from backend.platform.knowledge.sources import (
    DEFAULT_MOUNTED_KNOWLEDGE_SOURCES,
    normalize_mounted_knowledge_sources,
)
from backend.platform.rag.agentic import AgenticRetrievalOutcome
from backend.scenes.base import SceneDefinition
from backend.scenes.ecommerce.definition import build_ecommerce_scene_definition
from backend.scenes.generic_assistant.definition import (
    build_generic_assistant_scene_definition,
)
from backend.platform.knowledge.base.text import truncate_snippet
from backend.platform.memory.base.session_store import (
    SQLiteSessionStore,
    SessionRecord,
    SessionTurn,
)
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
        registered_retrieval_tools = getattr(self._retriever, "tools", None)
        if isinstance(registered_retrieval_tools, dict):
            self._retrieval_tool_names = set(registered_retrieval_tools.keys())
        else:
            self._retrieval_tool_names = {
                tool.name
                for tool in scene_definition.build_tools()
                if hasattr(tool, "name")
            }

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
        session = self.session_store.get_session(session_id)
        mounted_knowledge_sources = (
            session.mounted_knowledge_sources
            if session is not None
            else DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
        )
        history_turns = self.session_store.get_recent_turns(
            session_id=session_id,
            limit=self.settings.session.window_size,
        )
        history_text = self._format_history(history_turns)

        documents = self._retrieve_documents(
            payload.message,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
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

    def _retrieve_documents(
        self,
        message: str,
        *,
        mounted_knowledge_sources: tuple[str, ...],
    ) -> list[Document]:
        """兼容普通 retriever 与 agentic retriever 的检索输出。"""
        candidate_tools = self._resolve_candidate_tools(mounted_knowledge_sources)
        if hasattr(self._retriever, "retrieve_with_trace"):
            outcome: AgenticRetrievalOutcome = self._retriever.retrieve_with_trace(  # type: ignore[attr-defined]
                message,
                candidate_tools=candidate_tools,
            )
            return list(outcome.documents)
        if hasattr(self._retriever, "search"):
            return list(self._retriever.search(query=message))  # type: ignore[attr-defined]
        if isinstance(self._retriever, BaseRetriever):
            return list(self._retriever.invoke(message))
        raise TypeError("Retriever does not support document retrieval.")

    def _resolve_candidate_tools(
        self,
        mounted_knowledge_sources: tuple[str, ...],
    ) -> tuple[str, ...]:
        """按会话挂载源组装当前可用的候选检索工具。"""
        tool_names: list[str] = []
        seen: set[str] = set()

        if "documents" in mounted_knowledge_sources:
            self._append_candidate_tool(tool_names, seen, "knowledge_document_search")

        if "ecommerce" in mounted_knowledge_sources:
            for tool_name in (
                "product_semantic_search",
                "review_semantic_search",
                "order_semantic_search",
                "inventory_lookup",
                "product_detail_lookup",
            ):
                self._append_candidate_tool(tool_names, seen, tool_name)

        if not tool_names:
            raise ValueError("No retrieval tools available for mounted knowledge sources.")
        return tuple(tool_names)

    def _append_candidate_tool(
        self,
        tool_names: list[str],
        seen: set[str],
        tool_name: str,
    ) -> None:
        """只在当前场景已注册该工具时，才加入候选工具列表。"""
        if tool_name in seen or tool_name not in self._retrieval_tool_names:
            return
        seen.add(tool_name)
        tool_names.append(tool_name)

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

        for rank, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            namespace = str(metadata.get("namespace", "knowledge"))
            citation_id = str(metadata.get("citation_id") or metadata.get("chunk_id") or "unknown")
            key = self._build_citation_key(namespace=namespace, metadata=metadata, citation_id=citation_id)
            if key in seen:
                continue
            seen.add(key)

            score = metadata.get("score")
            normalized_score = float(score) if isinstance(score, int | float) else None
            snippet = truncate_snippet(doc.page_content)
            if snippet:
                citations.append(
                    self._build_citation(
                        index=len(citations) + 1,
                        rank=rank,
                        namespace=namespace,
                        citation_id=citation_id,
                        snippet=snippet,
                        score=normalized_score,
                        metadata=metadata,
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
            answer_documents = self._build_answer_documents(documents)
            result = combine_docs_chain.invoke(
                {
                    "context": answer_documents,
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
        final_citations = answer_citations if answer_citations else fallback_citations
        final_answer = self._ensure_answer_citation_markers(answer, final_citations)
        return final_answer, final_citations

    def _format_history(self, history_turns: list[SessionTurn]) -> str:
        """将历史轮次格式化为模型可读文本。"""
        if not history_turns:
            return "(empty)"

        lines: list[str] = []
        for turn in history_turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.assistant_answer}")
        return "\n".join(lines)

    def _build_answer_documents(self, documents: list[Document]) -> list[Document]:
        """把证据块改写为带编号的上下文，方便模型直接引用。"""
        citations = self._citations_from_documents(documents)
        citation_map = {
            self._build_citation_key(
                namespace=str(document.metadata.get("namespace", "knowledge")),
                metadata=document.metadata,
                citation_id=str(
                    document.metadata.get("citation_id") or document.metadata.get("chunk_id") or "unknown"
                ),
            ): citation
            for document, citation in zip(documents, citations, strict=False)
        }

        formatted_documents: list[Document] = []
        for document in documents:
            namespace = str(document.metadata.get("namespace", "knowledge"))
            citation_id = str(document.metadata.get("citation_id") or document.metadata.get("chunk_id") or "unknown")
            key = self._build_citation_key(
                namespace=namespace,
                metadata=document.metadata,
                citation_id=citation_id,
            )
            citation = citation_map.get(key)
            if citation is None:
                continue
            header = (
                f"[{citation.index}] "
                f"来源类型：{citation.source_kind}；"
                f"来源名称：{citation.source_name}；"
                f"来源路径：{citation.source_path or 'N/A'}；"
                f"分块：{citation.chunk_id or 'N/A'}"
            )
            formatted_documents.append(
                document.model_copy(
                    update={
                        "page_content": f"{header}\n{document.page_content}",
                    }
                )
            )
        return formatted_documents or documents

    def _build_citation(
        self,
        *,
        index: int,
        rank: int,
        namespace: str,
        citation_id: str,
        snippet: str,
        score: float | None,
        metadata: dict[str, Any],
    ) -> Citation:
        """把不同来源的 metadata 统一映射为 Citation。"""
        source_kind = self._resolve_source_kind(namespace=namespace, metadata=metadata)
        source_path = self._resolve_source_path(metadata)
        document_id = self._resolve_optional_str(metadata.get("document_id"))
        chunk_id = self._resolve_optional_str(metadata.get("chunk_id")) or (
            citation_id if source_kind == "document_chunk" else None
        )
        chunk_index = self._resolve_int(metadata.get("chunk_index"))
        source_name = self._resolve_source_name(
            source_kind=source_kind,
            citation_id=citation_id,
            source_path=source_path,
            document_id=document_id,
            metadata=metadata,
        )
        return Citation(
            index=index,
            citation_id=citation_id,
            namespace=namespace,
            source_kind=source_kind,
            source_name=source_name,
            source_path=source_path,
            document_id=document_id,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            snippet=snippet,
            score=score,
            rank=rank,
        )

    def _build_citation_key(
        self,
        *,
        namespace: str,
        metadata: dict[str, Any],
        citation_id: str,
    ) -> tuple[str, str]:
        """为 citation 去重生成稳定键。"""
        chunk_id = self._resolve_optional_str(metadata.get("chunk_id"))
        document_id = self._resolve_optional_str(metadata.get("document_id"))
        return namespace, chunk_id or citation_id or document_id or "unknown"

    def _resolve_source_kind(self, *, namespace: str, metadata: dict[str, Any]) -> str:
        """根据 metadata 判断引用来源类型。"""
        if metadata.get("chunk_id") is not None or metadata.get("document_id") is not None:
            return "document_chunk"
        source_kind_map = {
            "products": "product",
            "reviews": "review",
            "orders": "order",
            "inventory": "inventory",
            "product_detail": "product_detail",
            "documents": "document_chunk",
        }
        return source_kind_map.get(namespace, namespace)

    def _resolve_source_name(
        self,
        *,
        source_kind: str,
        citation_id: str,
        source_path: str | None,
        document_id: str | None,
        metadata: dict[str, Any],
    ) -> str:
        """生成适合前端展示的来源名称。"""
        if source_kind == "document_chunk":
            if source_path:
                return Path(source_path).name
            if document_id:
                return document_id
        for field_name in ("title", "name", "product_name", "order_id", "product_id", "review_id"):
            resolved = self._resolve_optional_str(metadata.get(field_name))
            if resolved:
                return resolved
        return citation_id

    def _resolve_source_path(self, metadata: dict[str, Any]) -> str | None:
        """提取来源路径。"""
        source_path = self._resolve_optional_str(metadata.get("source_path"))
        if source_path:
            return source_path
        for field_name in ("product_id", "review_id", "order_id"):
            resolved = self._resolve_optional_str(metadata.get(field_name))
            if resolved:
                return resolved
        return None

    def _ensure_answer_citation_markers(self, answer: str, citations: list[Citation]) -> str:
        """确保最终回答里能看到与 citations 对应的编号。"""
        if not citations:
            return answer
        if re.search(r"\[\d+\]", answer):
            return answer
        markers = "".join(f"[{citation.index}]" for citation in citations)
        return f"{answer}\n\n参考来源：{markers}"

    def _resolve_optional_str(self, value: Any) -> str | None:
        """把可选标量安全转成字符串。"""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, int | float):
            return str(value)
        return None

    def _resolve_int(self, value: Any) -> int | None:
        """把数字安全转成 int。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None



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
