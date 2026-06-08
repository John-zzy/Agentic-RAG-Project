from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from backend.application.runtime.api.chat.schemas import (
    Citation,
    RetrievalTrace,
    RetrievalTraceRound,
    RetrievalTraceTopChunk,
)
from backend.application.runtime.assembly.service_parts.contracts import (
    AgentRuntimeExecutionResult,
    AnswerMode,
    ChatServiceError,
    RuntimeFinalDecision,
    PreparedChatTurn,
)
from backend.platform.agent_runtime.contracts import (
    PlanRun,
    ReActRun,
    ToolObservation,
)
from backend.platform.agent_runtime.mode_selector import ModeSelection
from backend.platform.agent_runtime.plan.graph.config import PlanGraphDependencies
from backend.platform.agent_runtime.plan.planner import MinimalPlanner
from backend.platform.agent_runtime.projection import (
    RuntimeRunProjection,
    project_runtime_run,
    raw_retrieval_trace,
)
from backend.platform.agent_runtime.rag_tools import (
    AGENTIC_RAG_TOOL_NAME,
    NATIVE_RAG_TOOL_NAME,
    build_rag_tool_adapters,
)
from backend.platform.agent_runtime.react import (
    LLMReActActionSelector,
    ReActScenePolicy,
)
from backend.platform.agent_runtime.react.graph.config import ReActGraphDependencies
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.platform.agent_runtime.tool_policy import RuntimeToolPolicy
from backend.platform.knowledge.sources import DEFAULT_MOUNTED_KNOWLEDGE_SOURCES
from backend.platform.models.base.router import TaskComplexity
from backend.scenes.base import SceneRetrievalPolicy


class ChatAgentRuntimeMixin:
    def _select_agent_mode_for_graph(self, prepared: PreparedChatTurn) -> dict[str, Any]:
        selection = self._select_agent_mode(
            message=prepared.user_message,
            complexity=prepared.complexity,
            mounted_knowledge_sources=self._mounted_knowledge_sources_for_session(
                prepared.session_id
            ),
        )
        return {
            "agent_mode": selection.mode,
            "agent_mode_reason": selection.reason,
            "agent_mode_signals": dict(selection.signals),
        }

    def _mode_selection_from_graph_state(
            self,
            prepared: PreparedChatTurn,
            state: Mapping[str, Any],
    ) -> ModeSelection:
        mode = str(state.get("agent_mode") or prepared.agent_mode)
        return ModeSelection(
            mode="plan" if mode == "plan" else "react",
            reason=str(state.get("agent_mode_reason") or prepared.agent_mode_reason),
            signals=dict(state.get("agent_mode_signals") or prepared.agent_mode_signals or {}),
        )

    def _build_react_graph_deps(
            self,
            prepared: PreparedChatTurn,
            state: Mapping[str, Any],
    ) -> ReActGraphDependencies:
        mounted_knowledge_sources = self._mounted_knowledge_sources_for_session(
            prepared.session_id
        )
        mode_selection = self._mode_selection_from_graph_state(prepared, state)
        tool_executor = self._build_agent_tool_executor(
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=prepared.request_id,
        )
        tool_policy = self._build_runtime_tool_policy(
            tool_executor=tool_executor,
            message=prepared.user_message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=prepared.request_id,
        )
        scene_policy = self._build_react_scene_policy(
            tool_policy=tool_policy,
        )
        return ReActGraphDependencies(
            tool_executor=tool_executor,
            action_selector=LLMReActActionSelector(
                model_client=self.model,
                model_complexity=prepared.complexity or "simple",
            ),
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            user_goal=prepared.user_message,
            react_run_id=f"react-{prepared.request_id}",
            scene_policy=scene_policy,
            turn_id_factory=lambda index: f"turn-{index}",
            max_turns=scene_policy.max_turns,
            project_result=lambda run: self._project_react_graph_result(
                run=run,
                prepared=prepared,
                mounted_knowledge_sources=mounted_knowledge_sources,
                mode_selection=mode_selection,
            ),
        )

    def _build_plan_graph_deps(
            self,
            prepared: PreparedChatTurn,
            state: Mapping[str, Any],
    ) -> PlanGraphDependencies:
        mounted_knowledge_sources = self._mounted_knowledge_sources_for_session(
            prepared.session_id
        )
        mode_selection = self._mode_selection_from_graph_state(prepared, state)
        tool_executor = self._build_agent_tool_executor(
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=prepared.request_id,
        )
        tool_policy = self._build_runtime_tool_policy(
            tool_executor=tool_executor,
            message=prepared.user_message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=prepared.request_id,
        )
        tool_name = self._require_default_retrieval_tool(
            tool_policy=tool_policy,
            request_id=prepared.request_id,
        )
        scene_policy = self._build_plan_scene_policy(
            tool_policy=tool_policy,
        )
        return PlanGraphDependencies(
            tool_executor=tool_executor,
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            user_goal=prepared.user_message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            candidate_tools=self._plan_candidate_tools(tool_policy=tool_policy),
            scene_policy=scene_policy,
            planner=MinimalPlanner(
                tool_executor=tool_executor,
                plan_run_id_factory=lambda: f"plan-{prepared.request_id}",
                step_id_factory=lambda index: f"step-{index}",
            ),
            project_result=lambda run: self._project_plan_graph_result(
                run=run,
                prepared=prepared,
                mounted_knowledge_sources=mounted_knowledge_sources,
                mode_selection=mode_selection,
            ),
        )

    def _project_react_graph_result(
            self,
            *,
            run: ReActRun,
            prepared: PreparedChatTurn,
            mounted_knowledge_sources: tuple[str, ...],
            mode_selection: ModeSelection,
    ) -> dict[str, Any]:
        agent_result = self._agent_execution_result_from_run(
            message=prepared.user_message,
            complexity=prepared.complexity,
            mounted_knowledge_sources=mounted_knowledge_sources,
            react_run=run,
        )
        return self._graph_state_update_from_agent_result(
            agent_result=agent_result,
            mode_selection=mode_selection,
        )

    def _project_plan_graph_result(
            self,
            *,
            run: PlanRun,
            prepared: PreparedChatTurn,
            mounted_knowledge_sources: tuple[str, ...],
            mode_selection: ModeSelection,
    ) -> dict[str, Any]:
        agent_result = self._agent_execution_result_from_run(
            message=prepared.user_message,
            complexity=prepared.complexity,
            mounted_knowledge_sources=mounted_knowledge_sources,
            plan_run=run,
        )
        return self._graph_state_update_from_agent_result(
            agent_result=agent_result,
            mode_selection=mode_selection,
        )

    def _prepared_from_graph_state(
            self,
            prepared: PreparedChatTurn,
            state: Mapping[str, Any],
    ) -> PreparedChatTurn:
        citations = [
            citation if isinstance(citation, Citation) else Citation.model_validate(citation)
            for citation in list(state.get("citations") or prepared.citations)
        ]
        retrieval_trace_value = state.get("retrieval_trace") or prepared.retrieval_trace
        retrieval_trace = (
            retrieval_trace_value
            if isinstance(retrieval_trace_value, RetrievalTrace)
            else RetrievalTrace.model_validate(retrieval_trace_value)
        )
        return replace(
            prepared,
            documents=list(state.get("documents") or prepared.documents),
            tool_event=dict(state.get("tool_event") or prepared.tool_event),
            retrieval_trace=retrieval_trace,
            citations=citations,
            knowledge_used=bool(state.get("knowledge_used", prepared.knowledge_used)),
            final_decision=state.get("final_decision") or prepared.final_decision,
            follow_up_question=state.get("follow_up_question") or prepared.follow_up_question,
            answer_mode=str(state.get("answer_mode") or prepared.answer_mode),
            agent_mode=str(state.get("agent_mode") or prepared.agent_mode),
            agent_mode_reason=str(
                state.get("agent_mode_reason") or prepared.agent_mode_reason
            ),
            agent_mode_signals=dict(
                state.get("agent_mode_signals") or prepared.agent_mode_signals or {}
            ),
            react_run=state.get("react_run") or prepared.react_run,
            plan_run=state.get("plan_run") or prepared.plan_run,
            current_turn_id=state.get("current_turn_id") or prepared.current_turn_id,
            current_step_id=state.get("current_step_id") or prepared.current_step_id,
            current_tool_call=state.get("current_tool_call") or prepared.current_tool_call,
            tool_observation=state.get("tool_observation") or prepared.tool_observation,
        )

    def _mounted_knowledge_sources_for_session(self, session_id: str) -> tuple[str, ...]:
        session = self.session_store.get_session(session_id)
        if session is None:
            return tuple(DEFAULT_MOUNTED_KNOWLEDGE_SOURCES)
        return tuple(session.mounted_knowledge_sources)

    def _graph_state_update_from_agent_result(
            self,
            *,
            agent_result: AgentRuntimeExecutionResult,
            mode_selection: ModeSelection,
    ) -> dict[str, Any]:
        return {
            "documents": list(agent_result.documents),
            "tool_event": dict(agent_result.tool_event),
            "retrieval_trace": agent_result.retrieval_trace.model_dump(),
            "citations": [citation.model_dump() for citation in agent_result.citations],
            "knowledge_used": agent_result.knowledge_used,
            "final_decision": agent_result.final_decision,
            "follow_up_question": agent_result.follow_up_question,
            "answer_mode": agent_result.answer_mode,
            "agent_mode": mode_selection.mode,
            "agent_mode_reason": mode_selection.reason,
            "agent_mode_signals": dict(mode_selection.signals),
            "react_run": agent_result.react_run,
            "plan_run": agent_result.plan_run,
            "current_turn_id": agent_result.current_turn_id,
            "current_step_id": agent_result.current_step_id,
            "current_tool_call": agent_result.current_tool_call,
            "tool_observation": agent_result.tool_observation,
        }

    def _build_agent_tool_executor(
            self,
            *,
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
    ) -> ToolExecutor:
        candidate_tools = self._resolve_runtime_candidate_retrieval_tools(
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=request_id,
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
            candidate_retrieval_tools=candidate_tools,
            rag_tools=rag_tools,
            idempotency_store=self.graph_runtime.tool_idempotency_store,
        )

    def _resolve_runtime_candidate_retrieval_tools(
            self,
            *,
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
    ) -> tuple[str, ...]:
        """把 scene 的知识源解析错误收敛为统一 runtime 工具错误。"""
        try:
            return tuple(
                self.scene_definition.resolve_candidate_retrieval_tools(
                    mounted_knowledge_sources
                )
            )
        except ValueError as exc:
            raise ChatServiceError(
                status_code=500,
                code="AGENT_RUNTIME_TOOL_UNAVAILABLE",
                message="No RAG tool is available for current scene.",
                request_id=request_id,
            ) from exc

    def _build_runtime_tool_policy(
            self,
            *,
            tool_executor: ToolExecutor,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
    ) -> RuntimeToolPolicy:
        rag_input = self._build_rag_tool_input(
            message=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=request_id,
        )
        default_inputs: dict[str, dict[str, Any]] = {}
        for tool_name in (AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME):
            if tool_name in tool_executor.allowed_tools:
                default_inputs[tool_name] = dict(rag_input)
        return RuntimeToolPolicy.build(
            allowed_tools=sorted(tool_executor.allowed_tools),
            candidate_retrieval_tools=self._resolve_runtime_candidate_retrieval_tools(
                mounted_knowledge_sources=mounted_knowledge_sources,
                request_id=request_id,
            ),
            default_inputs=default_inputs,
            metadata=getattr(self.scene_definition, "metadata", {}) or {},
        )

    def _require_default_retrieval_tool(
            self,
            *,
            tool_policy: RuntimeToolPolicy,
            request_id: str,
    ) -> str:
        try:
            return tool_policy.require_default_retrieval_tool()
        except ValueError as exc:
            raise ChatServiceError(
                status_code=500,
                code="AGENT_RUNTIME_TOOL_UNAVAILABLE",
                message="No RAG tool is available for current scene.",
                request_id=request_id,
            ) from exc

    def _build_rag_tool_input(
            self,
            *,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
    ) -> dict[str, Any]:
        policy = self.scene_definition.retrieval_policy
        return {
            "query": message,
            "candidate_tools": list(
                self._resolve_runtime_candidate_retrieval_tools(
                    mounted_knowledge_sources=mounted_knowledge_sources,
                    request_id=request_id,
                )
            ),
            "top_k": policy.top_k,
            "min_relevance_score": policy.min_relevance_score,
            "recall_strategy": policy.recall_strategy,
            "rerank_enabled": policy.rerank_enabled,
            "rerank_top_n": policy.rerank_top_n,
        }

    def _build_react_scene_policy(
            self,
            *,
            tool_policy: RuntimeToolPolicy,
    ) -> ReActScenePolicy:
        """把 scene metadata 和 /chat 当前上下文整理成 LLM ReAct 调度策略。"""
        return ReActScenePolicy.from_metadata(
            getattr(self.scene_definition, "metadata", {}) or {},
            default_preferred_tools=list(tool_policy.preferred_tools),
            default_max_turns=2,
            default_no_evidence_action=self._react_no_evidence_action(),
            tool_input_hints=tool_policy.default_inputs,
        )

    def _react_no_evidence_action(self) -> str:
        if self.scene_definition.retrieval_policy.no_hit_strategy == "ask_user":
            return "ask_user"
        return "final_answer"

    def _build_plan_scene_policy(
            self,
            *,
            tool_policy: RuntimeToolPolicy,
    ) -> dict[str, Any]:
        """读取 scene 暴露的 plan 策略；没有策略时保持保守单工具计划。"""
        policy = self._agent_runtime_plan_policy()
        if not self._has_explicit_plan_policy(policy):
            policy["preferred_plan_tools"] = [tool_policy.require_default_retrieval_tool()]

        # RAG adapter 的输入由 application 统一注入检索策略，scene 可按工具覆盖。
        plan_tool_inputs = dict(policy.get("plan_tool_inputs") or {})
        for tool_name, tool_input in tool_policy.default_inputs.items():
            plan_tool_inputs.setdefault(tool_name, dict(tool_input))
        policy["candidate_tools"] = list(tool_policy.candidate_retrieval_tools)
        policy["allowed_tools"] = list(tool_policy.allowed_tools)
        policy["high_risk_tools"] = list(tool_policy.high_risk_tools)
        policy["plan_tool_inputs"] = plan_tool_inputs
        return policy

    def _agent_runtime_plan_policy(self) -> dict[str, Any]:
        metadata = getattr(self.scene_definition, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            return {}
        agent_runtime = metadata.get("agent_runtime")
        if isinstance(agent_runtime, Mapping):
            plan_policy = agent_runtime.get("plan")
            if isinstance(plan_policy, Mapping):
                return dict(plan_policy)
        plan_policy = metadata.get("plan_policy")
        return dict(plan_policy) if isinstance(plan_policy, Mapping) else {}

    def _has_explicit_plan_policy(self, policy: Mapping[str, Any]) -> bool:
        return any(
            key in policy
            for key in (
                "plan_steps",
                "preferred_plan_tools",
                "default_plan_tools",
                "plan_tools",
                "candidate_tools",
            )
        )

    def _plan_candidate_tools(
            self,
            *,
            tool_policy: RuntimeToolPolicy,
    ) -> tuple[str, ...]:
        selected = list(tool_policy.preferred_tools)
        default_tool_name = tool_policy.default_retrieval_tool
        if default_tool_name and default_tool_name not in selected:
            selected.append(default_tool_name)
        allowed = set(tool_policy.allowed_tools)
        return tuple(tool_name for tool_name in selected if tool_name in allowed)

    def _policy_step_tool_names(self, value: Any) -> list[str]:
        if not isinstance(value, list | tuple):
            return []
        tool_names: list[str] = []
        for step in value:
            if isinstance(step, Mapping) and isinstance(step.get("tool_name"), str):
                tool_names.append(str(step["tool_name"]))
        return tool_names

    def _policy_tool_names(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list | tuple):
            return [str(tool_name) for tool_name in value]
        return []

    def _agent_execution_result_from_run(
            self,
            *,
            message: str,
            complexity: TaskComplexity | None,
            mounted_knowledge_sources: tuple[str, ...],
            react_run: ReActRun | None = None,
            plan_run: PlanRun | None = None,
    ) -> AgentRuntimeExecutionResult:
        del complexity
        projection = project_runtime_run(react_run=react_run, plan_run=plan_run)
        run = projection.run
        if not projection.observations:
            return self._agent_execution_result_from_observationless_run(
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
                projection=projection,
            )
        observations = projection.observations
        current_observation = projection.current_observation
        if current_observation is None:
            raise ChatServiceError(
                status_code=500,
                code="AGENT_RUNTIME_DATA_INCOMPLETE",
                message="Agent runtime current observation is missing.",
                request_id=run.request_id,
            )
        documents = projection.documents
        candidate_citations = self._citation_mapper.citations_from_documents(documents)
        final_decision = projection.final_decision
        knowledge_used = self._can_answer_with_evidence(
            final_decision=final_decision,
            citations=candidate_citations,
        )
        final_decision = self._resolve_prepared_final_decision(
            final_decision=final_decision,
            knowledge_used=knowledge_used,
        )
        citations = candidate_citations if knowledge_used else []
        retrieval_trace = self._retrieval_trace_from_observations(
            message=message,
            observations=observations,
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=run.request_id,
            citations=citations,
            knowledge_used=knowledge_used,
            final_decision=final_decision,
            follow_up_question=projection.follow_up_question,
        )
        answer_mode = self._resolve_answer_mode(
            final_decision=final_decision,
            knowledge_used=knowledge_used,
        )
        adopted_documents = documents if knowledge_used else []
        tool_event = self._tool_event_from_observations(
            observations=observations,
            successful_observation_count=len(projection.successful_observations),
            document_count=len(documents),
            retrieval_trace=retrieval_trace,
            mounted_knowledge_sources=mounted_knowledge_sources,
            request_id=run.request_id,
        )
        return AgentRuntimeExecutionResult(
            documents=adopted_documents,
            tool_event=tool_event,
            retrieval_trace=retrieval_trace,
            citations=citations,
            knowledge_used=knowledge_used,
            final_decision=final_decision,
            follow_up_question=projection.follow_up_question,
            answer_mode=answer_mode,
            react_run=projection.react_run,
            plan_run=projection.plan_run,
            current_turn_id=projection.current_turn_id,
            current_step_id=projection.current_step_id,
            current_tool_call=(
                projection.current_tool_call.model_dump()
                if projection.current_tool_call is not None
                else None
            ),
            tool_observation=projection.tool_observation,
        )

    def _agent_execution_result_from_observationless_run(
            self,
            *,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            projection: RuntimeRunProjection,
    ) -> AgentRuntimeExecutionResult:
        run = projection.run
        if run.workflow_status in {"failed", "cancelled"}:
            raise ChatServiceError(
                status_code=500,
                code="AGENT_RUNTIME_RUN_FAILED",
                message=run.error or run.result_summary or "Agent runtime run failed.",
                request_id=run.request_id,
            )
        retrieval_trace = RetrievalTrace(
            original_query=message,
            final_query=message,
            rewritten_query=None,
            tool_call_count=0,
            candidate_tools=list(
                self._resolve_runtime_candidate_retrieval_tools(
                    mounted_knowledge_sources=mounted_knowledge_sources,
                    request_id=run.request_id,
                )
            ),
            exit_reason=projection.exit_reason,
            final_decision=projection.final_decision,
            success=False,
            follow_up_question=projection.follow_up_question,
            raw_candidates_count=0,
            filtered_candidates_count=0,
            top_k_chunks=[],
            citations=[],
            knowledge_used=False,
            rounds=[],
        )
        return AgentRuntimeExecutionResult(
            documents=[],
            tool_event=self._tool_event_from_observationless_run(
                run=run,
                retrieval_trace=retrieval_trace,
            ),
            retrieval_trace=retrieval_trace,
            citations=[],
            knowledge_used=False,
            final_decision=projection.final_decision,
            follow_up_question=projection.follow_up_question,
            answer_mode=self._resolve_answer_mode(
                final_decision=projection.final_decision,
                knowledge_used=False,
            ),
            react_run=projection.react_run,
            plan_run=projection.plan_run,
            current_turn_id=projection.current_turn_id,
            current_step_id=projection.current_step_id,
            current_tool_call=(
                projection.current_tool_call.model_dump()
                if projection.current_tool_call is not None
                else None
            ),
            tool_observation=projection.tool_observation,
        )

    def _tool_event_from_observationless_run(
            self,
            *,
            run: ReActRun | PlanRun,
            retrieval_trace: RetrievalTrace,
    ) -> dict[str, Any]:
        return {
            "stage": "agent_runtime",
            "mode": "agent_runtime_control",
            "tool_name": "agent_runtime_control",
            "tool_names": [],
            "documents": 0,
            "observation_count": 0,
            "successful_observation_count": 0,
            "exit_reason": retrieval_trace.exit_reason,
            "success": False,
            "final_decision": retrieval_trace.final_decision,
            "follow_up_question": retrieval_trace.follow_up_question,
            "rounds": [],
            "nested_retrieval_trace": {},
            "nested_retrieval_traces": [],
            "current_tool_call": (
                run.current_tool_call.model_dump()
                if run.current_tool_call is not None
                else None
            ),
            "tool_observation": None,
        }

    def _retrieval_trace_from_observations(
            self,
            *,
            message: str,
            observations: list[ToolObservation],
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
            citations: list[Citation],
            knowledge_used: bool,
            final_decision: RuntimeFinalDecision | None,
            follow_up_question: str | None,
    ) -> RetrievalTrace:
        raw_traces = [raw_retrieval_trace(observation) for observation in observations]
        rounds = self._rounds_from_observations(observations=observations)
        top_k_chunks = self._top_chunks_from_citations(citations)
        last_trace = self._last_non_empty_trace(raw_traces)
        raw_candidates_count = self._sum_trace_count(
            raw_traces,
            "raw_candidates_count",
            fallback=sum(round_trace.raw_candidates_count or 0 for round_trace in rounds),
        )
        filtered_candidates_count = self._sum_trace_count(
            raw_traces,
            "filtered_candidates_count",
            fallback=sum(round_trace.filtered_candidates_count or 0 for round_trace in rounds),
        )
        return RetrievalTrace(
            original_query=str(self._first_trace_value(raw_traces, "original_query") or message),
            final_query=str(last_trace.get("final_query") or message),
            rewritten_query=self._aggregate_rewritten_query(
                raw_traces=raw_traces,
                rounds=rounds,
                final_query=str(last_trace.get("final_query") or message),
            ),
            tool_call_count=self._aggregate_tool_call_count(
                observations=observations,
                raw_traces=raw_traces,
                round_count=len(rounds),
            ),
            candidate_tools=self._candidate_tools_from_traces(
                raw_traces=raw_traces,
                mounted_knowledge_sources=mounted_knowledge_sources,
                request_id=request_id,
            ),
            exit_reason=self._last_trace_text(raw_traces, "exit_reason"),
            final_decision=final_decision,
            success=any(observation.success for observation in observations),
            follow_up_question=follow_up_question,
            raw_candidates_count=raw_candidates_count,
            filtered_candidates_count=filtered_candidates_count,
            top_k_chunks=top_k_chunks,
            citations=citations,
            knowledge_used=knowledge_used,
            rounds=rounds,
        )

    def _rounds_from_observations(
            self,
            *,
            observations: list[ToolObservation],
    ) -> list[RetrievalTraceRound]:
        rounds: list[RetrievalTraceRound] = []
        for observation in observations:
            raw_trace = raw_retrieval_trace(observation)
            for round_trace in self._rounds_from_raw_trace(raw_trace):
                rounds.append(round_trace.model_copy(update={"round_index": len(rounds) + 1}))
        return rounds

    def _last_non_empty_trace(self, raw_traces: list[dict[str, Any]]) -> dict[str, Any]:
        for raw_trace in reversed(raw_traces):
            if raw_trace:
                return raw_trace
        return {}

    def _first_trace_value(self, raw_traces: list[dict[str, Any]], key: str) -> Any:
        for raw_trace in raw_traces:
            value = raw_trace.get(key)
            if value is not None:
                return value
        return None

    def _last_trace_text(self, raw_traces: list[dict[str, Any]], key: str) -> str | None:
        for raw_trace in reversed(raw_traces):
            value = raw_trace.get(key)
            if value is not None:
                return str(value)
        return None

    def _aggregate_rewritten_query(
            self,
            *,
            raw_traces: list[dict[str, Any]],
            rounds: list[RetrievalTraceRound],
            final_query: str,
    ) -> str | None:
        rewritten_query = self._last_trace_text(raw_traces, "rewritten_query")
        if rewritten_query is not None:
            return rewritten_query
        for round_trace in reversed(rounds):
            if round_trace.rewritten_query:
                return round_trace.rewritten_query
        # no-hit 场景需要说明实际检索查询；这仍来自 run-level trace，不回查 step/turn。
        no_hit_decision = self._last_trace_text(raw_traces, "final_decision")
        return final_query if no_hit_decision == "no_evidence" else None

    def _sum_trace_count(
            self,
            raw_traces: list[dict[str, Any]],
            key: str,
            *,
            fallback: int,
    ) -> int:
        values = [
            resolved
            for raw_trace in raw_traces
            if (resolved := self._optional_trace_count(raw_trace.get(key))) is not None
        ]
        return sum(values) if values else fallback

    def _aggregate_tool_call_count(
            self,
            *,
            observations: list[ToolObservation],
            raw_traces: list[dict[str, Any]],
            round_count: int,
    ) -> int:
        if observations:
            return len(observations)
        traced_count = self._sum_trace_count(
            raw_traces,
            "tool_call_count",
            fallback=0,
        )
        if traced_count > 0:
            return traced_count
        if round_count > 0:
            return round_count
        return len(observations)

    def _candidate_tools_from_traces(
            self,
            *,
            raw_traces: list[dict[str, Any]],
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
    ) -> list[str]:
        candidate_tools: list[str] = []
        seen: set[str] = set()
        for raw_trace in raw_traces:
            raw_tools = raw_trace.get("candidate_tools")
            if not isinstance(raw_tools, list):
                continue
            for tool_name in raw_tools:
                if not isinstance(tool_name, str) or tool_name in seen:
                    continue
                seen.add(tool_name)
                candidate_tools.append(tool_name)
        if candidate_tools:
            return candidate_tools
        return list(
            self._resolve_runtime_candidate_retrieval_tools(
                mounted_knowledge_sources=mounted_knowledge_sources,
                request_id=request_id,
            )
        )

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

    def _tool_event_from_observations(
            self,
            *,
            observations: list[ToolObservation],
            successful_observation_count: int,
            document_count: int,
            retrieval_trace: RetrievalTrace,
            mounted_knowledge_sources: tuple[str, ...],
            request_id: str,
    ) -> dict[str, Any]:
        current_observation = observations[-1]
        raw_trace = raw_retrieval_trace(current_observation)
        return {
            "stage": "retrieval",
            "mode": "agent_runtime_tool",
            "tool_name": current_observation.tool_name,
            "tool_names": self._tool_names_from_observations(observations),
            "retrieval_policy": self._build_policy_summary(self.scene_definition.retrieval_policy),
            "candidate_tools": list(
                self._resolve_runtime_candidate_retrieval_tools(
                    mounted_knowledge_sources=mounted_knowledge_sources,
                    request_id=request_id,
                )
            ),
            "documents": document_count,
            "observation_count": len(observations),
            "successful_observation_count": successful_observation_count,
            "exit_reason": retrieval_trace.exit_reason,
            "success": retrieval_trace.success,
            "final_decision": retrieval_trace.final_decision,
            "follow_up_question": retrieval_trace.follow_up_question,
            "rounds": [round_trace.model_dump() for round_trace in retrieval_trace.rounds],
            "nested_retrieval_trace": raw_trace,
            "nested_retrieval_traces": [
                raw_retrieval_trace(observation) for observation in observations
            ],
            "current_tool_call": (
                current_observation.execution.model_dump()
                if current_observation.execution is not None
                else None
            ),
            "tool_observation": current_observation.model_dump(),
        }

    def _tool_names_from_observations(self, observations: list[ToolObservation]) -> list[str]:
        tool_names: list[str] = []
        seen: set[str] = set()
        for observation in observations:
            if observation.tool_name in seen:
                continue
            seen.add(observation.tool_name)
            tool_names.append(observation.tool_name)
        return tool_names

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
        if final_decision == "direct_answer":
            return "direct_answer"
        if final_decision == "ask_user":
            return "follow_up"
        return "fallback"
