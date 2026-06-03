from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document

from backend.application.runtime.api.chat.schemas import (
    Citation,
    RetrievalTrace,
    RetrievalTraceRound,
    RetrievalTraceTopChunk,
)
from backend.application.runtime.chat_service_parts.contracts import (
    AgentRuntimeExecutionResult,
    AnswerMode,
    ChatServiceError,
    RuntimeFinalDecision,
    _SingleToolReActActionSelector,
)
from backend.platform.agent_runtime.contracts import PlanRun, ReActRun, ToolObservation
from backend.platform.agent_runtime.mode_selector import ModeSelection
from backend.platform.agent_runtime.plan_executor import PlanExecutor
from backend.platform.agent_runtime.planner import MinimalPlanner
from backend.platform.agent_runtime.rag_tools import (
    AGENTIC_RAG_TOOL_NAME,
    NATIVE_RAG_TOOL_NAME,
    build_rag_tool_adapters,
)
from backend.platform.agent_runtime.react import ReActRuntime
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.models.base.router import TaskComplexity
from backend.scenes.base import SceneRetrievalPolicy


class ChatAgentRuntimeMixin:
    def _execute_agent_runtime(
            self,
            *,
            session_id: str,
            request_id: str,
            message: str,
            complexity: TaskComplexity | None,
            mounted_knowledge_sources: tuple[str, ...],
            mode_selection: ModeSelection,
    ) -> AgentRuntimeExecutionResult:
        """通过顶层 ReAct/Plan Runtime 执行工具，而不是在 /chat 前置检索。"""
        tool_executor = self._build_agent_tool_executor(
            mounted_knowledge_sources=mounted_knowledge_sources
        )
        tool_name = self._select_rag_tool_name(
            tool_executor=tool_executor,
            request_id=request_id,
        )
        tool_input = self._build_rag_tool_input(
            message=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )

        if mode_selection.mode == "plan":
            plan_run = self._run_plan_agent(
                tool_executor=tool_executor,
                session_id=session_id,
                request_id=request_id,
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
                tool_name=tool_name,
                tool_input=tool_input,
            )
            observation = self._latest_plan_observation(plan_run)
            return self._agent_execution_result_from_observation(
                message=message,
                complexity=complexity,
                mounted_knowledge_sources=mounted_knowledge_sources,
                observation=observation,
                plan_run=plan_run,
            )

        react_run = self._run_react_agent(
            tool_executor=tool_executor,
            session_id=session_id,
            request_id=request_id,
            message=message,
            tool_name=tool_name,
            tool_input=tool_input,
        )
        observation = self._latest_react_observation(react_run)
        return self._agent_execution_result_from_observation(
            message=message,
            complexity=complexity,
            mounted_knowledge_sources=mounted_knowledge_sources,
            observation=observation,
            react_run=react_run,
        )

    def _build_agent_tool_executor(
            self,
            *,
            mounted_knowledge_sources: tuple[str, ...],
    ) -> ToolExecutor:
        candidate_tools = self.scene_definition.resolve_candidate_retrieval_tools(
            mounted_knowledge_sources
        )
        rag_tools = {
            tool.name: tool
            for tool in build_rag_tool_adapters(
                retriever=self._retriever,
                candidate_tools=candidate_tools,
            )
        }
        return ToolExecutor.from_scene(
            scene_definition=self.scene_definition,
            mounted_knowledge_sources=mounted_knowledge_sources,
            rag_tools=rag_tools,
        )

    def _select_rag_tool_name(self, *, tool_executor: ToolExecutor, request_id: str) -> str:
        """第一版保守选择 RAG 工具；是否多轮检索留在 RAG tool 内部。"""
        allowed = tool_executor.allowed_tools
        if AGENTIC_RAG_TOOL_NAME in allowed:
            return AGENTIC_RAG_TOOL_NAME
        if NATIVE_RAG_TOOL_NAME in allowed:
            return NATIVE_RAG_TOOL_NAME
        for tool_name in sorted(allowed):
            if "rag" in tool_name or "search" in tool_name:
                return tool_name
        raise ChatServiceError(
            status_code=500,
            code="AGENT_RUNTIME_TOOL_UNAVAILABLE",
            message="No RAG tool is available for current scene.",
            request_id=request_id,
        )

    def _build_rag_tool_input(
            self,
            *,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
    ) -> dict[str, Any]:
        policy = self.scene_definition.retrieval_policy
        return {
            "query": message,
            "candidate_tools": list(
                self.scene_definition.resolve_candidate_retrieval_tools(
                    mounted_knowledge_sources
                )
            ),
            "top_k": policy.top_k,
            "min_relevance_score": policy.min_relevance_score,
            "recall_strategy": policy.recall_strategy,
            "rerank_enabled": policy.rerank_enabled,
            "rerank_top_n": policy.rerank_top_n,
        }

    def _run_react_agent(
            self,
            *,
            tool_executor: ToolExecutor,
            session_id: str,
            request_id: str,
            message: str,
            tool_name: str,
            tool_input: Mapping[str, Any],
    ) -> ReActRun:
        runtime = ReActRuntime(
            tool_executor=tool_executor,
            action_selector=_SingleToolReActActionSelector(
                tool_name=tool_name,
                input_payload=tool_input,
            ),
            turn_id_factory=lambda index: f"turn-{index}",
            max_turns=2,
        )
        return runtime.run(
            session_id=session_id,
            request_id=request_id,
            user_goal=message,
            react_run_id=f"react-{request_id}",
        )

    def _run_plan_agent(
            self,
            *,
            tool_executor: ToolExecutor,
            session_id: str,
            request_id: str,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            tool_name: str,
            tool_input: Mapping[str, Any],
    ) -> PlanRun:
        planner = MinimalPlanner(
            tool_executor=tool_executor,
            plan_run_id_factory=lambda: f"plan-{request_id}",
            step_id_factory=lambda index: f"step-{index}",
        )
        plan_run = planner.create_plan(
            session_id=session_id,
            request_id=request_id,
            user_goal=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            scene_policy={
                "plan_steps": [
                    {
                        "step_id": "step-1",
                        "goal": message,
                        "tool_name": tool_name,
                        "input": dict(tool_input),
                        "depends_on": [],
                    }
                ]
            },
        )
        return PlanExecutor(tool_executor=tool_executor).execute(plan_run)

    def _latest_react_observation(self, react_run: ReActRun) -> ToolObservation:
        for turn in reversed(react_run.turns):
            if turn.observation is not None:
                return turn.observation
        raise ChatServiceError(
            status_code=500,
            code="AGENT_RUNTIME_OBSERVATION_MISSING",
            message="ReAct runtime did not produce a tool observation.",
            request_id=react_run.request_id,
        )

    def _latest_plan_observation(self, plan_run: PlanRun) -> ToolObservation:
        for step in reversed(plan_run.steps):
            if step.observation is not None:
                return step.observation
        raise ChatServiceError(
            status_code=500,
            code="AGENT_RUNTIME_OBSERVATION_MISSING",
            message="Plan runtime did not produce a tool observation.",
            request_id=plan_run.request_id,
        )

    def _agent_execution_result_from_observation(
            self,
            *,
            message: str,
            complexity: TaskComplexity | None,
            mounted_knowledge_sources: tuple[str, ...],
            observation: ToolObservation,
            react_run: ReActRun | None = None,
            plan_run: PlanRun | None = None,
    ) -> AgentRuntimeExecutionResult:
        del complexity
        documents = self._documents_from_observation(observation)
        candidate_citations = self._citation_mapper.citations_from_documents(documents)
        final_decision = self._final_decision_from_observation(observation)
        knowledge_used = self._can_answer_with_evidence(
            final_decision=final_decision,
            citations=candidate_citations,
        )
        final_decision = self._resolve_prepared_final_decision(
            final_decision=final_decision,
            knowledge_used=knowledge_used,
        )
        citations = candidate_citations if knowledge_used else []
        retrieval_trace = self._retrieval_trace_from_observation(
            message=message,
            observation=observation,
            mounted_knowledge_sources=mounted_knowledge_sources,
            citations=citations,
            knowledge_used=knowledge_used,
            final_decision=final_decision,
        )
        answer_mode = self._resolve_answer_mode(
            final_decision=final_decision,
            knowledge_used=knowledge_used,
        )
        adopted_documents = documents if knowledge_used else []
        tool_event = self._tool_event_from_observation(
            observation=observation,
            retrieval_trace=retrieval_trace,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
        return AgentRuntimeExecutionResult(
            documents=adopted_documents,
            tool_event=tool_event,
            retrieval_trace=retrieval_trace,
            citations=citations,
            knowledge_used=knowledge_used,
            final_decision=final_decision,
            follow_up_question=self._follow_up_question_from_observation(observation),
            answer_mode=answer_mode,
            react_run=react_run.model_dump() if react_run is not None else None,
            plan_run=plan_run.model_dump() if plan_run is not None else None,
            current_turn_id=self._event_turn_id(react_run) if react_run is not None else None,
            current_step_id=self._event_step_id(plan_run) if plan_run is not None else None,
            current_tool_call=(
                observation.execution.model_dump()
                if observation.execution is not None
                else None
            ),
            tool_observation=observation.model_dump(),
        )

    def _documents_from_observation(self, observation: ToolObservation) -> list[Document]:
        output = observation.output if isinstance(observation.output, Mapping) else {}
        raw_documents = output.get("documents") if isinstance(output, Mapping) else None
        documents: list[Document] = []
        if isinstance(raw_documents, list):
            for item in raw_documents:
                if not isinstance(item, Mapping):
                    continue
                page_content = str(item.get("page_content") or item.get("content") or "")
                if not page_content:
                    continue
                metadata = item.get("metadata")
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                    )
                )
        return documents

    def _final_decision_from_observation(
            self,
            observation: ToolObservation,
    ) -> RuntimeFinalDecision | None:
        for source in (
            observation.metadata,
            observation.output if isinstance(observation.output, Mapping) else {},
            self._raw_retrieval_trace(observation),
        ):
            value = source.get("final_decision") if isinstance(source, Mapping) else None
            if value in {
                "answer_with_evidence",
                "ask_user",
                "max_rounds_reached",
                "no_evidence",
                "retrieval_failed",
            }:
                return value  # type: ignore[return-value]
        if observation.success:
            return "no_evidence"
        return "retrieval_failed"

    def _follow_up_question_from_observation(self, observation: ToolObservation) -> str | None:
        if observation.user_prompt:
            return observation.user_prompt
        output = observation.output if isinstance(observation.output, Mapping) else {}
        value = output.get("follow_up_question") if isinstance(output, Mapping) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
        trace = self._raw_retrieval_trace(observation)
        value = trace.get("follow_up_question")
        return value if isinstance(value, str) and value.strip() else None

    def _retrieval_trace_from_observation(
            self,
            *,
            message: str,
            observation: ToolObservation,
            mounted_knowledge_sources: tuple[str, ...],
            citations: list[Citation],
            knowledge_used: bool,
            final_decision: RuntimeFinalDecision | None,
    ) -> RetrievalTrace:
        raw_trace = self._raw_retrieval_trace(observation)
        rounds = self._rounds_from_raw_trace(raw_trace)
        top_k_chunks = self._top_chunks_from_citations(citations)
        raw_candidates_count = self._resolve_trace_count(
            raw_trace.get("raw_candidates_count"),
            fallback=sum(round_trace.raw_candidates_count or 0 for round_trace in rounds),
        )
        filtered_candidates_count = self._resolve_trace_count(
            raw_trace.get("filtered_candidates_count"),
            fallback=sum(round_trace.filtered_candidates_count or 0 for round_trace in rounds),
        )
        final_query = str(raw_trace.get("final_query") or message)
        rewritten_query = raw_trace.get("rewritten_query")
        if rewritten_query is None:
            for round_trace in reversed(rounds):
                if round_trace.rewritten_query:
                    rewritten_query = round_trace.rewritten_query
                    break
        return RetrievalTrace(
            original_query=str(raw_trace.get("original_query") or message),
            final_query=final_query,
            rewritten_query=str(rewritten_query) if rewritten_query is not None else None,
            tool_call_count=self._resolve_trace_count(
                raw_trace.get("tool_call_count"),
                fallback=len(rounds),
            ),
            candidate_tools=list(
                raw_trace.get("candidate_tools")
                if isinstance(raw_trace.get("candidate_tools"), list)
                else self.scene_definition.resolve_candidate_retrieval_tools(
                    mounted_knowledge_sources
                )
            ),
            exit_reason=(
                str(raw_trace.get("exit_reason"))
                if raw_trace.get("exit_reason") is not None
                else None
            ),
            final_decision=final_decision,
            success=bool(raw_trace.get("success", observation.success)),
            follow_up_question=self._follow_up_question_from_observation(observation),
            raw_candidates_count=raw_candidates_count,
            filtered_candidates_count=filtered_candidates_count,
            top_k_chunks=top_k_chunks,
            citations=citations,
            knowledge_used=knowledge_used,
            rounds=rounds,
        )

    def _raw_retrieval_trace(self, observation: ToolObservation) -> dict[str, Any]:
        trace = observation.trace.get("retrieval_trace")
        return dict(trace) if isinstance(trace, Mapping) else {}

    def _rounds_from_raw_trace(self, raw_trace: Mapping[str, Any]) -> list[RetrievalTraceRound]:
        raw_rounds = raw_trace.get("rounds")
        if not isinstance(raw_rounds, list):
            return []
        rounds: list[RetrievalTraceRound] = []
        for index, item in enumerate(raw_rounds, start=1):
            if not isinstance(item, Mapping):
                continue
            try:
                rounds.append(
                    RetrievalTraceRound(
                        round_index=self._resolve_trace_count(
                            item.get("round_index"),
                            fallback=index,
                        ),
                        tool_name=str(item.get("tool_name") or "unknown"),
                        query=str(item.get("query") or raw_trace.get("original_query") or ""),
                        rewritten_query=(
                            str(item.get("rewritten_query"))
                            if item.get("rewritten_query") is not None
                            else None
                        ),
                        decision=str(item.get("decision") or "finish"),
                        is_sufficient=bool(item.get("is_sufficient", False)),
                        reason=(
                            str(item.get("reason"))
                            if item.get("reason") is not None
                            else None
                        ),
                        result_count=self._resolve_trace_count(
                            item.get("result_count"),
                            fallback=0,
                        ),
                        document_count=self._resolve_trace_count(
                            item.get("document_count"),
                            fallback=self._resolve_trace_count(item.get("result_count"), fallback=0),
                        ),
                        success=bool(item.get("success", item.get("result_success", False))),
                        error=(
                            str(item.get("error"))
                            if item.get("error") is not None
                            else None
                        ),
                        raw_candidates_count=self._resolve_trace_count(
                            item.get("raw_candidates_count"),
                            fallback=self._resolve_trace_count(item.get("result_count"), fallback=0),
                        ),
                        filtered_candidates_count=self._resolve_trace_count(
                            item.get("filtered_candidates_count"),
                            fallback=self._resolve_trace_count(item.get("result_count"), fallback=0),
                        ),
                        top_k_chunks=self._coerce_trace_top_chunks(item.get("top_k_chunks")),
                        rerank=dict(item.get("rerank")) if isinstance(item.get("rerank"), Mapping) else None,
                    )
                )
            except ValueError:
                continue
        return rounds

    def _tool_event_from_observation(
            self,
            *,
            observation: ToolObservation,
            retrieval_trace: RetrievalTrace,
            mounted_knowledge_sources: tuple[str, ...],
    ) -> dict[str, Any]:
        raw_trace = self._raw_retrieval_trace(observation)
        return {
            "stage": "retrieval",
            "mode": "agent_runtime_tool",
            "tool_name": observation.tool_name,
            "retrieval_policy": self._build_policy_summary(self.scene_definition.retrieval_policy),
            "candidate_tools": list(
                self.scene_definition.resolve_candidate_retrieval_tools(
                    mounted_knowledge_sources
                )
            ),
            "documents": len(self._documents_from_observation(observation)),
            "exit_reason": retrieval_trace.exit_reason,
            "success": observation.success,
            "final_decision": retrieval_trace.final_decision,
            "follow_up_question": retrieval_trace.follow_up_question,
            "rounds": [round_trace.model_dump() for round_trace in retrieval_trace.rounds],
            "nested_retrieval_trace": raw_trace,
        }

    def _event_turn_id(self, react_run: ReActRun | None) -> str | None:
        if react_run is None:
            return None
        if react_run.current_turn_id:
            return react_run.current_turn_id
        for turn in reversed(react_run.turns):
            if turn.observation is not None:
                return turn.turn_id
        if react_run.turns:
            return react_run.turns[-1].turn_id
        return None

    def _event_step_id(self, plan_run: PlanRun | None) -> str | None:
        if plan_run is None:
            return None
        if plan_run.current_step_id:
            return plan_run.current_step_id
        if plan_run.steps:
            return plan_run.steps[-1].step_id
        return None

    def _top_chunks_from_citations(self, citations: list[Citation]) -> list[RetrievalTraceTopChunk]:
        return [
            RetrievalTraceTopChunk(
                rank=citation.rank,
                citation_id=citation.citation_id,
                document_id=citation.document_id,
                chunk_id=citation.chunk_id,
                chunk_index=citation.chunk_index,
                source_name=citation.source_name,
                source_path=citation.source_path,
                score=citation.score,
                vector_score=citation.vector_score,
                keyword_score=citation.keyword_score,
                vector_rank=citation.vector_rank,
                keyword_rank=citation.keyword_rank,
                rerank_score=citation.rerank_score,
                matched_by=list(citation.matched_by),
            )
            for citation in citations
        ]

    def _coerce_trace_top_chunks(self, value: Any) -> list[RetrievalTraceTopChunk]:
        if not isinstance(value, list):
            return []
        chunks: list[RetrievalTraceTopChunk] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            try:
                chunks.append(RetrievalTraceTopChunk.model_validate(dict(item)))
            except ValueError:
                continue
        return chunks

    def _resolve_trace_count(self, value: Any, *, fallback: int) -> int:
        resolved = self._optional_trace_count(value)
        return fallback if resolved is None else resolved

    def _optional_trace_count(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, float) and value.is_integer() and value >= 0:
            return int(value)
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
        return None

    def _build_policy_summary(self, policy: SceneRetrievalPolicy) -> dict[str, Any]:
        return {
            "top_k": policy.top_k,
            "min_relevance_score": policy.min_relevance_score,
            "recall_strategy": policy.recall_strategy,
            "no_hit_strategy": policy.no_hit_strategy,
            "rerank_enabled": policy.rerank_enabled,
            "rerank_top_n": policy.rerank_top_n,
        }

    def _can_answer_with_evidence(
            self,
            *,
            final_decision: RuntimeFinalDecision | None,
            citations: list[Citation],
    ) -> bool:
        """只允许最终决策和有效引用同时满足时进入证据回答链。"""
        return final_decision == "answer_with_evidence" and len(citations) > 0

    def _build_prepared_retrieval_trace(
            self,
            *,
            retrieval_trace: RetrievalTrace,
            citations: list[Citation],
            knowledge_used: bool,
            final_decision: RuntimeFinalDecision | None,
    ) -> RetrievalTrace:
        """构造最终响应 trace，并保留轮次级诊断信息。"""
        return retrieval_trace.model_copy(
            update={
                "citations": citations,
                "knowledge_used": knowledge_used,
                "final_decision": final_decision,
                # 顶层 top_k_chunks 只代表最终采纳证据，非证据分支清空但保留 rounds。
                "top_k_chunks": retrieval_trace.top_k_chunks if knowledge_used else [],
            }
        )

    def _resolve_prepared_final_decision(
            self,
            *,
            final_decision: RuntimeFinalDecision | None,
            knowledge_used: bool,
    ) -> RuntimeFinalDecision | None:
        """将无有效 citation 的证据候选收敛为 no_evidence，保证 trace 解释最终分支。"""
        if final_decision == "answer_with_evidence" and not knowledge_used:
            return "no_evidence"
        return final_decision

    def _resolve_answer_mode(
            self,
            *,
            final_decision: RuntimeFinalDecision | None,
            knowledge_used: bool,
    ) -> AnswerMode:
        """按最终证据采纳结果决定 answer branch。"""
        if knowledge_used:
            return "evidence_answer"
        if final_decision == "ask_user":
            return "follow_up"
        return "fallback"


