from __future__ import annotations

from typing import Any

from backend.application.runtime.api.chat.schemas import ChatResponse, Citation
from backend.application.runtime.assembly.service_parts.contracts import PreparedChatTurn
from backend.application.runtime.stream_events import (
    ChatStreamEvent,
    TypedGraphStreamEvent,
    TypedGraphStreamEventName,
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

    def _map_graph_stream_event(
            self,
            event: TypedGraphStreamEventName,
            data: dict[str, Any],
    ) -> ChatStreamEvent:
        """统一映射 graph runtime 事件，避免 SSE 暴露底层事件名。"""
        return self._stream_event_mapper.map_event(
            TypedGraphStreamEvent(event=event, data=data)
        )

    def _build_stream_observability_metadata(
            self,
            *,
            prepared: PreparedChatTurn,
            graph_state: dict[str, Any],
            error_code: str | None = None,
    ) -> dict[str, Any]:
        """构造 SSE 专用安全观测摘要，不携带 prompt、历史或工具参数。"""
        metadata = {
            "provider": self._stream_provider_name(),
            "complexity": prepared.complexity or "simple",
            "retry_count": self._runtime_retry_count(graph_state),
            "tools": self._stream_tool_summaries(prepared=prepared, graph_state=graph_state),
        }
        latency_ms = self._stream_model_latency_ms(graph_state)
        if latency_ms is not None:
            metadata["model_latency_ms"] = latency_ms
        if error_code:
            metadata["error_classification"] = self._stream_error_classification(error_code)
        return metadata

    def _stream_provider_name(self) -> str:
        provider = getattr(self.model, "provider_name", None)
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
        return self.model.__class__.__name__

    def _runtime_retry_count(self, graph_state: dict[str, Any]) -> int:
        retry_counts = self._retry_counts_from_run(graph_state.get("react_run"))
        retry_counts.extend(self._retry_counts_from_run(graph_state.get("plan_run")))
        return max(retry_counts, default=0)

    def _retry_counts_from_run(self, run: Any) -> list[int]:
        if not isinstance(run, dict):
            return []
        items = list(run.get("turns") or []) + list(run.get("steps") or [])
        retry_counts: list[int] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            retry_metadata = item.get("retry_metadata")
            if not isinstance(retry_metadata, dict):
                continue
            attempt = retry_metadata.get("attempt")
            if isinstance(attempt, int):
                retry_counts.append(max(0, attempt - 1))
        return retry_counts

    def _stream_model_latency_ms(self, graph_state: dict[str, Any]) -> float | None:
        metadata = graph_state.get("metadata")
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("model_latency_ms")
        if isinstance(value, (int, float)):
            return round(float(value), 3)
        return None

    def _stream_tool_summaries(
            self,
            *,
            prepared: PreparedChatTurn,
            graph_state: dict[str, Any],
    ) -> list[dict[str, str]]:
        summaries = self._tool_summaries_from_run(graph_state.get("react_run"))
        summaries.extend(self._tool_summaries_from_run(graph_state.get("plan_run")))
        if not summaries:
            summaries.extend(self._tool_summary_from_observation(prepared.tool_observation))
            summaries.extend(self._tool_summary_from_observation(graph_state.get("tool_observation")))
        return _dedupe_stream_tool_summaries(summaries)

    def _tool_summaries_from_run(self, run: Any) -> list[dict[str, str]]:
        if not isinstance(run, dict):
            return []
        items = list(run.get("turns") or []) + list(run.get("steps") or [])
        summaries: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name")
            observation = item.get("observation")
            if not tool_name and isinstance(observation, dict):
                tool_name = observation.get("tool_name")
            if not tool_name:
                continue
            summaries.append(
                {
                    "tool_name": str(tool_name),
                    "tool_status": str(item.get("status") or self._observation_status(observation)),
                }
            )
        return summaries

    def _tool_summary_from_observation(self, observation: Any) -> list[dict[str, str]]:
        if not isinstance(observation, dict):
            return []
        tool_name = observation.get("tool_name")
        if not tool_name:
            return []
        return [
            {
                "tool_name": str(tool_name),
                "tool_status": self._observation_status(observation),
            }
        ]

    def _observation_status(self, observation: Any) -> str:
        if not isinstance(observation, dict):
            return "unknown"
        if observation.get("requires_user") is True:
            return "waiting_user"
        return "succeeded" if observation.get("success") is True else "failed"

    def _stream_error_classification(self, error_code: str) -> str:
        lowered = error_code.lower()
        if "timeout" in lowered:
            return "timeout"
        if "validation" in lowered or "interrupt" in lowered or "resume" in lowered:
            return "validation"
        if "model" in lowered or "provider" in lowered:
            return "provider_error"
        return "runtime_error"


def _dedupe_stream_tool_summaries(items: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["tool_name"], item["tool_status"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped




