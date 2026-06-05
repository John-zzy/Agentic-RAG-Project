from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from backend.application.runtime.api.chat.schemas import Citation
from backend.application.runtime.assembly.service_parts.contracts import (
    ChatServiceError,
    PreparedChatTurn,
)
from backend.platform.workflow.langgraph.state import RuntimeGraphState
from backend.platform.memory.base.chat_history import SQLiteChatMessageHistory
from backend.platform.models.base.router import TaskComplexity


class ChatAnsweringMixin:
    def _generate_answer(
        self,
        prepared: PreparedChatTurn,
    ) -> tuple[PreparedChatTurn, str, list[Citation], str, RuntimeGraphState]:
        """根据准备结果生成最终答案。"""
        result = self.graph_runtime.invoke(
            prepared=prepared,
            answer_builder=self._generate_answer_direct,
            history_loader=self._load_graph_seed_history,
            select_agent_mode=self._select_agent_mode_for_graph,
            run_agent_runtime=self._run_agent_runtime_for_graph,
            build_prepared_from_state=self._prepared_from_graph_state,
            build_hitl_wait_update=self._build_hitl_wait_update_for_graph,
        )
        resolved_prepared = self._prepared_from_graph_state(prepared, result.state)
        citations = [
            citation if isinstance(citation, Citation) else Citation.model_validate(citation)
            for citation in result.citations
        ]
        return resolved_prepared, result.answer, citations, result.run_id, result.state

    def _generate_answer_direct(self, prepared: PreparedChatTurn) -> tuple[str, list[Citation]]:
        """执行 graph answer node 内部的直接回答逻辑。"""
        if prepared.answer_mode != "evidence_answer":
            return self._build_non_evidence_answer(prepared)
        return self._invoke_answer_template(prepared=prepared)

    def _load_graph_seed_history(self, prepared: PreparedChatTurn) -> list[BaseMessage]:
        """读取首次 graph run 的旧会话消息种子。"""
        return self._get_session_history(
            prepared.session_id,
            request_id=prepared.request_id,
            timestamp=prepared.timestamp,
        ).messages

    def _invoke_answer_template(
            self,
            *,
            prepared: PreparedChatTurn,
    ) -> tuple[str, list[Citation]]:
        """调用模型链生成答案，并返回答案与引用。"""
        runnable = self._get_answer_runnable(prepared)
        try:
            answer = self.model.invoke_runnable(
                runnable,
                self._build_answer_variables(prepared),
                config=self._build_runnable_config(prepared.session_id),
            )
        except ValueError as exc:
            if str(exc) == "Model returned empty content":
                raise ChatServiceError(
                    status_code=502,
                    code="MODEL_EMPTY_RESPONSE",
                    message="Model returned empty response.",
                    request_id=prepared.request_id,
                ) from exc
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc
        except Exception as exc:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc

        return self._finalize_answer_text(answer, prepared.citations)

    def _stream_model_answer(self, prepared: PreparedChatTurn) -> Iterator[str]:
        """对最终答案生成阶段执行流式调用。"""
        runnable = self._get_answer_runnable(prepared)
        try:
            for chunk in self.model.stream_runnable(
                    runnable,
                    self._build_answer_variables(prepared),
                    config=self._build_runnable_config(prepared.session_id),
            ):
                yield str(chunk)
        except ValueError as exc:
            message = str(exc)
            if message == "Model returned empty streaming content":
                raise ChatServiceError(
                    status_code=502,
                    code="MODEL_EMPTY_RESPONSE",
                    message="Model returned empty response.",
                    request_id=prepared.request_id,
                ) from exc
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc
        except Exception as exc:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc

    def _finalize_streamed_answer(
            self,
            prepared: PreparedChatTurn,
            answer_parts: list[str],
    ) -> tuple[str, list[Citation]]:
        """将流式片段拼接为最终权威答案。"""
        joined_answer = "".join(answer_parts).strip()
        if not joined_answer:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_EMPTY_RESPONSE",
                message="Model returned empty response.",
                request_id=prepared.request_id,
            )
        return self._finalize_answer_text(joined_answer, prepared.citations)

    def _build_non_evidence_answer(self, prepared: PreparedChatTurn) -> tuple[str, list[Citation]]:
        """构造不携带引用的追问或降级回答。"""
        if prepared.answer_mode == "direct_answer":
            return self._invoke_direct_answer_template(prepared), []
        if prepared.answer_mode == "follow_up":
            return self._build_follow_up_answer(prepared), []
        return self._build_fallback_answer(prepared), []

    def _invoke_direct_answer_template(self, prepared: PreparedChatTurn) -> str:
        """不依赖知识证据时，走普通对话回答链，不生成 citations。"""
        runnable = self._get_direct_answer_runnable(prepared)
        try:
            answer = self.model.invoke_runnable(
                runnable,
                {"input": prepared.user_message},
                config=self._build_runnable_config(prepared.session_id),
            )
        except ValueError as exc:
            if str(exc) == "Model returned empty content":
                raise ChatServiceError(
                    status_code=502,
                    code="MODEL_EMPTY_RESPONSE",
                    message="Model returned empty response.",
                    request_id=prepared.request_id,
                ) from exc
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc
        except Exception as exc:
            raise ChatServiceError(
                status_code=502,
                code="MODEL_INVOCATION_FAILED",
                message="Model invocation failed. Please retry later.",
                request_id=prepared.request_id,
            ) from exc
        return str(answer).strip()

    def _build_follow_up_answer(self, prepared: PreparedChatTurn) -> str:
        """解析 ask_user 分支追问文案，缺失时回退到 scene no-hit 文案。"""
        if prepared.follow_up_question:
            return prepared.follow_up_question
        return self._build_fallback_answer(prepared)

    def _build_fallback_answer(self, prepared: PreparedChatTurn) -> str:
        """构造无命中或降级时的 fallback 回答。"""
        policy = self.scene_definition.retrieval_policy
        return self.scene_definition.fallback_policy.message_for_strategy(policy.no_hit_strategy)

    def _finalize_answer_text(
            self,
            answer: str,
            citations: list[Citation],
    ) -> tuple[str, list[Citation]]:
        """统一补齐 citation markers，并返回最终答案与引用。"""
        final_answer = self._citation_mapper.ensure_answer_citation_markers(answer.strip(), citations)
        return final_answer, citations

    def _build_answer_variables(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造最终回答模板需要的变量。"""
        return {
            "context": self._citation_mapper.build_answer_documents(prepared.documents),
            "input": prepared.user_message,
        }

    def _get_answer_base_runnable(self, complexity: TaskComplexity) -> Any:
        """为给定复杂度构建不携带请求上下文的基础回答 runnable。"""
        cached = self._answer_base_runnables.get(complexity)
        if cached is not None:
            return cached

        runnable = self.model.get_runnable(
            complexity=complexity,
            prompt_template=self._rag_answer_template,
        )
        self._answer_base_runnables[complexity] = runnable
        return runnable

    def _get_answer_runnable(self, prepared: PreparedChatTurn) -> RunnableWithMessageHistory:
        """为当前请求构建带消息历史的回答 runnable。"""
        base_runnable = self._get_answer_base_runnable(prepared.complexity or "simple")

        def history_factory(session_id: str) -> SQLiteChatMessageHistory:
            return self._get_session_history(
                session_id,
                request_id=prepared.request_id,
                timestamp=prepared.timestamp,
            )

        runnable = RunnableWithMessageHistory(
            base_runnable,
            history_factory,
            input_messages_key="input",
            history_messages_key="history",
        )
        return runnable

    def _get_direct_answer_runnable(self, prepared: PreparedChatTurn) -> RunnableWithMessageHistory:
        """为无需知识库的普通回答构建带历史的 runnable。"""
        base_runnable = self.model.get_runnable(complexity=prepared.complexity or "simple")

        def history_factory(session_id: str) -> SQLiteChatMessageHistory:
            return self._get_session_history(
                session_id,
                request_id=prepared.request_id,
                timestamp=prepared.timestamp,
            )

        return RunnableWithMessageHistory(
            base_runnable,
            history_factory,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_session_history(
            self,
            session_id: str,
            *,
            request_id: str,
            timestamp: str,
    ) -> SQLiteChatMessageHistory:
        """解析指定会话的 LangChain message history。"""
        return SQLiteChatMessageHistory(
            session_id,
            store=self.session_store,
            request_id=request_id,
            timestamp=timestamp,
            message_limit=self.settings.session.window_size * 2,
            message_transform=self.context_builder.trim_messages,
        )

    def _build_runnable_config(self, session_id: str) -> dict[str, Any]:
        """构造 RunnableWithMessageHistory 所需的 configurable config。"""
        return {"configurable": {"session_id": session_id}}



