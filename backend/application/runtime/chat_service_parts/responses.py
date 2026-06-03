from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from backend.application.runtime.api.chat.schemas import ChatResponse, Citation
from backend.application.runtime.chat_service_parts.contracts import PreparedChatTurn
from backend.application.runtime.stream_events import (
    ChatStreamEvent,
    GraphRuntimeStreamEvent,
    GraphStreamEventName,
)


class ChatResponseMixin:
    def _persist_turn(
            self,
            *,
            prepared: PreparedChatTurn,
            answer: str,
            citations: list[Citation],
    ) -> None:
        """以既有语义写入最终对话轮次。"""
        self.session_store.append_turn(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            user_message=prepared.user_message,
            assistant_answer=answer,
            retrieval_snippets=[citation.model_dump() for citation in citations],
            timestamp=prepared.timestamp,
            persist_messages=not prepared.knowledge_used,
        )

    def _build_chat_response(
            self,
            *,
            prepared: PreparedChatTurn,
            answer: str,
            citations: list[Citation],
            run_id: str | None = None,
    ) -> ChatResponse:
        """统一构造聊天响应。"""
        return ChatResponse(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            answer=answer,
            knowledge_used=prepared.knowledge_used,
            scene=prepared.scene_metadata.scene,
            agent=prepared.scene_metadata.agent,
            state="succeeded",
            final_state="succeeded",
            run_id=run_id,
            state_event="success",
            citations=citations,
            retrieval_trace=prepared.retrieval_trace.model_copy(
                update={
                    "citations": citations,
                    "knowledge_used": prepared.knowledge_used,
                    "top_k_chunks": prepared.retrieval_trace.top_k_chunks
                    if prepared.knowledge_used
                    else [],
                }
            ),
        )

    def _serialize_history_message(self, message: BaseMessage) -> dict[str, Any]:
        """将 LangChain message 归一化为稳定的 SSE payload。"""
        return {
            "type": message.type,
            "content": message.content,
        }

    def _map_graph_stream_event(
            self,
            event: GraphStreamEventName,
            data: dict[str, Any],
    ) -> ChatStreamEvent:
        """统一映射 graph runtime 事件，避免 SSE 暴露底层事件名。"""
        return self._stream_event_mapper.map_event(
            GraphRuntimeStreamEvent(event=event, data=data)
        )



