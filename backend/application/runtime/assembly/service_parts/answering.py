from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import BaseMessage

from backend.application.runtime.api.chat.schemas import Citation
from backend.application.runtime.assembly.service_parts.contracts import (
    ChatServiceError,
    PreparedChatTurn,
)
from backend.platform.workflow.langgraph.state import RuntimeGraphState
from backend.platform.memory.base.chat_history import SQLiteChatMessageHistory
from backend.platform.models.base.router import TaskComplexity
from backend.platform.agent_runtime.observability.graph_logging import log_llm_output


class ChatAnsweringMixin:
    def _generate_answer(
        self,
        prepared: PreparedChatTurn,
    ) -> tuple[PreparedChatTurn, str, list[Citation], str, RuntimeGraphState]:
        """通过 ChatGraph 执行一次完整回答流程，并返回响应组装所需结果。

        流程概览：
        1. 把 application 层能力作为回调传给 ChatGraph
        2. 由 ChatGraph 完成状态流转、HITL、ReAct/Plan 分支和回答生成
        3. 将图运行后的 state 和 citations 转成 API 响应可以直接使用的结构
        """

        # ── 第1步：进入 platform 层 ChatGraph 主流程 ──
        # application 只提供“如何生成答案、如何加载历史、如何构建子图依赖”等具体能力，
        # 具体编排顺序由 ChatGraph 统一负责，避免主流程散落在 application 层。
        result = self.graph_runtime.invoke(
            prepared=prepared,
            answer_builder=self._generate_answer_direct,
            history_loader=self._load_graph_seed_history,
            select_agent_mode=self._select_agent_mode_for_graph,
            build_react_deps=self._build_react_deps,
            build_plan_graph_deps=self._build_plan_graph_deps,
            build_prepared_from_state=self._prepared_from_graph_state,
            build_hitl_wait_update=self._build_hitl_wait_update_for_graph,
        )
        
        # ── 第2步：用图运行后的最新 state 重建 prepared ──
        # state 可能已经被工具调用、人工节点或 ReAct/Plan 子图更新过，
        # 这里重新投影成 PreparedChatTurn，保证后续响应组装拿到的是最新上下文。
        resolved_prepared = self._prepared_from_graph_state(prepared, result.state)

        # ── 第3步：统一 citations 的数据形态 ──
        # platform 返回的 citations 可能是 dict，也可能已经是 Citation；
        # application 边界统一转成 API schema，避免响应层再判断多种格式。
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
        """读取当前会话的历史消息，作为 ChatGraph 本轮运行的初始上下文。"""
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
        try:
            runnable = self._get_answer_runnable(prepared)
            answer = self.model.invoke_runnable(
                runnable,
                self._build_answer_variables(prepared),
            )
            log_llm_output(
                source="evidence_answer",
                request_id=prepared.request_id,
                output=answer,
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
        try:
            runnable = self._get_answer_runnable(prepared)
            for chunk in self.model.stream_runnable(
                    runnable,
                    self._build_answer_variables(prepared),
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
        log_llm_output(
            source="stream_answer",
            request_id=prepared.request_id,
            output=joined_answer,
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
        try:
            runnable = self._get_direct_answer_runnable(prepared)
            answer = self.model.invoke_runnable(
                runnable,
                self._build_direct_answer_variables(prepared),
            )
            log_llm_output(
                source="direct_answer",
                request_id=prepared.request_id,
                output=answer,
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
            "history": self._load_prompt_history(prepared),
            "input": prepared.user_message,
        }

    def _build_direct_answer_variables(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        """构造普通回答模板变量。"""
        return {
            "history": self._load_prompt_history(prepared),
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

    def _get_answer_runnable(self, prepared: PreparedChatTurn) -> Any:
        """为当前请求解析最终回答 runnable。"""
        return self._get_answer_base_runnable(prepared.complexity or "simple")

    def _get_direct_answer_runnable(self, prepared: PreparedChatTurn) -> Any:
        """为无需知识库的普通回答构建 runnable。"""
        return self.model.get_runnable(
            complexity=prepared.complexity or "simple",
            prompt_template=self._direct_answer_template,
        )

    def _load_prompt_history(self, prepared: PreparedChatTurn) -> list[BaseMessage]:
        """从会话存储读取 prompt 历史，显式传入 LangChain prompt。"""
        return self._get_session_history(
            prepared.session_id,
            request_id=prepared.request_id,
            timestamp=prepared.timestamp,
        ).messages

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

