from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from backend.api.prompts import build_rag_answer_prompt_template
from backend.api.schemas import ChatRequest, ChatResponse, Citation
from backend.config.settings import AppSettings, settings
from backend.knowledge._text_utils import truncate_snippet
from backend.knowledge.retriever import KnowledgeBaseRetriever
from backend.knowledge.service import KnowledgeService, create_knowledge_service
from backend.memory.prompt_context import PromptContextBuilder
from backend.memory.session_store import SessionTurn
from backend.memory.session_store import SQLiteSessionStore
from backend.models.client import ModelClient, model_client
from backend.models.router import TaskComplexity


class RetrievalChainModel(Protocol):
    def build_chat_model_for_complexity(self, complexity: TaskComplexity) -> Any:
        """按复杂度构建可用于 RAG 链的聊天模型实例。"""
        ...


class ChatServiceError(RuntimeError):
    def __init__(self, *, status_code: int, code: str, message: str, request_id: str) -> None:
        """封装可返回给 API 层的业务错误。"""
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(message)


class ChatService:
    def __init__(
        self,
        app_settings: AppSettings | None = None,
        knowledge_service: KnowledgeService | None = None,
        session_store: SQLiteSessionStore | None = None,
        context_builder: PromptContextBuilder | None = None,
        model: RetrievalChainModel | None = None,
    ) -> None:
        """初始化聊天服务的配置、知识库、会话存储与模型。"""
        self.settings = app_settings or settings
        self.knowledge_service = knowledge_service or create_knowledge_service(self.settings)
        self.session_store = session_store or SQLiteSessionStore(self.settings)
        self.context_builder = context_builder or PromptContextBuilder(
            window_size=self.settings.session.window_size
        )
        self.model = model or model_client
        self.minimum_relevance = 0.18
        self._rag_answer_template = build_rag_answer_prompt_template()

    def chat(self, payload: ChatRequest) -> ChatResponse:
        """执行一次完整对话流程：检索、生成、落库并返回响应。"""
        request_id = uuid4().hex
        session_id = payload.session_id or uuid4().hex

        if payload.stream:
            raise ChatServiceError(
                status_code=501,
                code="STREAM_NOT_SUPPORTED",
                message="Streaming mode is reserved but not enabled on this endpoint yet.",
                request_id=request_id,
            )

        top_k = payload.top_k or self.settings.vector_store.top_k
        history_turns = self.session_store.get_recent_turns(
            session_id=session_id,
            limit=self.settings.session.window_size,
        )
        history_text = self._format_history(history_turns)

        retriever = KnowledgeBaseRetriever(
            knowledge_service=self.knowledge_service,
            default_top_k=top_k,
            minimum_relevance=self.minimum_relevance,
        )
        preloaded_docs = retriever.search(query=payload.message, top_k=top_k)
        citations = self._citations_from_documents(preloaded_docs)
        knowledge_used = len(citations) > 0

        if knowledge_used:
            complexity = self._infer_complexity(payload.message)
            answer, citations = self._invoke_chain_with_docs(
                documents=preloaded_docs,
                user_message=payload.message,
                history_text=history_text,
                complexity=complexity,
                request_id=request_id,
                fallback_citations=citations,
            )
        else:
            answer = self._build_no_hit_answer(payload.message)

        timestamp = datetime.now(UTC).isoformat()
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
            citations=citations,
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

    def _build_no_hit_answer(self, message: str) -> str:
        """在无检索命中时返回兜底回答。"""
        return (
            "我暂时没有检索到足够相关的商品知识来直接回答该问题。"
            "你可以补充更具体的商品名称、预算范围或核心诉求，我再为你精确检索。"
            f"（当前问题：{message.strip()}）"
        )

    def _infer_complexity(self, message: str) -> TaskComplexity:
        """根据关键词和长度推断任务复杂度。"""
        normalized = message.lower()
        complex_keywords = ("退款", "退货", "投诉", "工单", "愤怒", "不满", "人工")
        moderate_keywords = ("推荐", "比较", "订单", "物流", "库存", "参数", "价格", "评价")

        if any(keyword in normalized for keyword in complex_keywords):
            return "complex"
        if any(keyword in normalized for keyword in moderate_keywords) or len(normalized) > 40:
            return "moderate"
        return "simple"


def create_chat_service(
    app_settings: AppSettings | None = None,
    knowledge_service: KnowledgeService | None = None,
    session_store: SQLiteSessionStore | None = None,
    context_builder: PromptContextBuilder | None = None,
    model: ModelClient | None = None,
) -> ChatService:
    """聊天服务工厂函数。"""
    return ChatService(
        app_settings=app_settings,
        knowledge_service=knowledge_service,
        session_store=session_store,
        context_builder=context_builder,
        model=model,
    )
