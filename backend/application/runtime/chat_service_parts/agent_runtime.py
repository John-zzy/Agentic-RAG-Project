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
)
from backend.platform.agent_runtime.contracts import (
    ReActAction,
    PlanRun,
    ReActRun,
    ReActTurn,
    ToolExecutionMetadata,
    ToolObservation,
)
from backend.platform.agent_runtime.mode_selector import ModeSelection
from backend.platform.agent_runtime.plan.executor import PlanExecutor
from backend.platform.agent_runtime.plan.planner import MinimalPlanner
from backend.platform.agent_runtime.rag_tools import (
    AGENTIC_RAG_TOOL_NAME,
    NATIVE_RAG_TOOL_NAME,
    build_rag_tool_adapters,
)
from backend.platform.agent_runtime.react import (
    LLMReActActionSelector,
    ReActRuntime,
    ReActScenePolicy,
)
from backend.platform.agent_runtime.react.selector import ReActSelectorError
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

        if mode_selection.mode == "plan":
            tool_name = self._select_rag_tool_name(
                tool_executor=tool_executor,
                request_id=request_id,
            )
            tool_input = self._build_rag_tool_input(
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
            )
            plan_run = self._run_plan_agent(
                tool_executor=tool_executor,
                session_id=session_id,
                request_id=request_id,
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
                tool_name=tool_name,
                tool_input=tool_input,
            )
            return self._agent_execution_result_from_run(
                message=message,
                complexity=complexity,
                mounted_knowledge_sources=mounted_knowledge_sources,
                plan_run=plan_run,
            )

        react_run = self._run_react_agent(
            tool_executor=tool_executor,
            session_id=session_id,
            request_id=request_id,
            message=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            complexity=complexity,
        )
        return self._agent_execution_result_from_run(
            message=message,
            complexity=complexity,
            mounted_knowledge_sources=mounted_knowledge_sources,
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
            mounted_knowledge_sources: tuple[str, ...],
            complexity: TaskComplexity | None,
    ) -> ReActRun:
        if self.scene_definition.scene == "generic_assistant":
            return self._run_direct_react_retrieval(
                tool_executor=tool_executor,
                session_id=session_id,
                request_id=request_id,
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
                complexity=complexity,
            )
        scene_policy = self._build_react_scene_policy(
            tool_executor=tool_executor,
            message=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
        runtime = ReActRuntime(
            tool_executor=tool_executor,
            action_selector=LLMReActActionSelector(
                model_client=self.model,
                model_complexity=complexity or "simple",
            ),
            scene_policy=scene_policy,
            turn_id_factory=lambda index: f"turn-{index}",
            max_turns=scene_policy.max_turns,
        )
        try:
            react_run = runtime.run(
                session_id=session_id,
                request_id=request_id,
                user_goal=message,
                react_run_id=f"react-{request_id}",
            )
        except ReActSelectorError:
            return self._run_direct_react_retrieval(
                tool_executor=tool_executor,
                session_id=session_id,
                request_id=request_id,
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
                complexity=complexity,
            )

        if react_run.workflow_status == "waiting_user" and react_run.metadata.get("latest_selector_failure"):
            return self._run_direct_react_retrieval(
                tool_executor=tool_executor,
                session_id=session_id,
                request_id=request_id,
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
                complexity=complexity,
            )
        return react_run

    def _build_react_scene_policy(
            self,
            *,
            tool_executor: ToolExecutor,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
    ) -> ReActScenePolicy:
        """把 scene metadata 和 /chat 当前上下文整理成 LLM ReAct 调度策略。"""
        allowed_tools = sorted(tool_executor.allowed_tools)
        preferred_tools = self._react_preferred_tools(allowed_tools)
        tool_input_hints = self._react_tool_input_hints(
            message=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            allowed_tools=allowed_tools,
        )
        return ReActScenePolicy.from_metadata(
            getattr(self.scene_definition, "metadata", {}) or {},
            default_preferred_tools=preferred_tools,
            default_max_turns=2,
            default_no_evidence_action=self._react_no_evidence_action(),
            tool_input_hints=tool_input_hints,
        )

    def _react_preferred_tools(self, allowed_tools: list[str]) -> list[str]:
        preferred: list[str] = []
        for tool_name in (AGENTIC_RAG_TOOL_NAME, NATIVE_RAG_TOOL_NAME):
            if tool_name in allowed_tools:
                preferred.append(tool_name)
        preferred.extend(tool_name for tool_name in allowed_tools if tool_name not in preferred)
        return preferred

    def _react_tool_input_hints(
            self,
            *,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            allowed_tools: list[str],
    ) -> dict[str, dict[str, Any]]:
        rag_input = self._build_rag_tool_input(
            message=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
        hints: dict[str, dict[str, Any]] = {}
        for tool_name in allowed_tools:
            if "rag" in tool_name or "search" in tool_name:
                hints[tool_name] = dict(rag_input)
        return hints

    def _react_no_evidence_action(self) -> str:
        if self.scene_definition.retrieval_policy.no_hit_strategy == "ask_user":
            return "ask_user"
        return "final_answer"

    def _run_direct_react_retrieval(
            self,
            *,
            tool_executor: ToolExecutor,
            session_id: str,
            request_id: str,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            complexity: TaskComplexity | None,
    ) -> ReActRun:
        """在 selector 不可用时直接执行一次 RAG 检索，保留 generic assistant 的询问语义。"""
        tool_name = self._select_rag_tool_name(
            tool_executor=tool_executor,
            request_id=request_id,
        )
        observation = tool_executor.execute(
            tool_name=tool_name,
            input_payload=self._build_rag_tool_input(
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
            ),
        )
        scene_policy = self._build_react_scene_policy(
            tool_executor=tool_executor,
            message=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
        turn = ReActTurn(
            turn_id="turn-1",
            round_index=1,
            goal=message,
            action=ReActAction(
                action_type="tool_call",
                tool_name=tool_name,
                input={"query": message},
                rationale_summary="首轮先调用允许的 RAG 工具。",
                metadata={},
            ),
            status="waiting_user" if observation.requires_user else "succeeded",
            input={"query": message},
            tool_name=tool_name,
            observation=observation,
            observation_summary=observation.result_summary,
            result_summary=observation.result_summary,
            error=observation.error,
        )
        final_turn = ReActTurn(
            turn_id="turn-2",
            round_index=2,
            goal=message,
            action=ReActAction(
                action_type="final_answer",
                rationale_summary="已有工具观察，进入最终汇总。",
                metadata={},
            ),
            status="succeeded",
        )
        # 直接回退时补齐第二轮汇总，保证 SSE 事件仍然投影成标准 ReAct 结构。
        return ReActRun(
            react_run_id=f"react-{request_id}",
            session_id=session_id,
            request_id=request_id,
            user_goal=message,
            workflow_status="waiting_user" if observation.requires_user else "succeeded",
            max_turns=scene_policy.max_turns,
            turns=[turn, final_turn] if not observation.requires_user else [turn],
            observations=[observation],
            current_turn_id=turn.turn_id,
            current_tool_call=observation.execution if observation.requires_user else None,
            final_answer=observation.result_summary if not observation.requires_user else None,
            result_summary=observation.result_summary,
            error=observation.error,
            metadata={
                "direct_retrieval_fallback": True,
                "selected_tool": tool_name,
                "query_complexity": complexity,
                "attempted_tools": [tool_name],
                "latest_action_selection": {
                    "round_index": 2 if not observation.requires_user else 1,
                    "selector_attempt": 1,
                    "status": "validated",
                    "action_type": "final_answer" if not observation.requires_user else "tool_call",
                    "tool_name": None if not observation.requires_user else tool_name,
                    "rationale_summary": "已有工具观察，进入最终汇总。"
                    if not observation.requires_user
                    else "首轮先调用允许的 RAG 工具。",
                    "validation_result": "passed",
                    "error": None,
                },
                "action_selection_audits": [
                    {
                        "round_index": 1,
                        "selector_attempt": 1,
                        "status": "validated",
                        "action_type": "tool_call",
                        "tool_name": tool_name,
                        "rationale_summary": "首轮先调用允许的 RAG 工具。",
                        "error": None,
                    },
                    {
                        "round_index": 2,
                        "selector_attempt": 1,
                        "status": "validated",
                        "action_type": "final_answer",
                        "tool_name": None,
                        "rationale_summary": "已有工具观察，进入最终汇总。",
                        "error": None,
                    },
                ] if not observation.requires_user else [
                    {
                        "round_index": 1,
                        "selector_attempt": 1,
                        "status": "validated",
                        "action_type": "tool_call",
                        "tool_name": tool_name,
                        "rationale_summary": "首轮先调用允许的 RAG 工具。",
                        "error": None,
                    }
                ],
            },
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
        scene_policy = self._build_plan_scene_policy(
            tool_name=tool_name,
            tool_input=tool_input,
        )
        plan_run = planner.create_plan(
            session_id=session_id,
            request_id=request_id,
            user_goal=message,
            mounted_knowledge_sources=mounted_knowledge_sources,
            candidate_tools=self._plan_candidate_tools(
                tool_executor=tool_executor,
                default_tool_name=tool_name,
                scene_policy=scene_policy,
            ),
            scene_policy=scene_policy,
        )
        return PlanExecutor(tool_executor=tool_executor).execute(plan_run)

    def _build_plan_scene_policy(
            self,
            *,
            tool_name: str,
            tool_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        """读取 scene 暴露的 plan 策略；没有策略时保持保守单工具计划。"""
        policy = self._agent_runtime_plan_policy()
        if not self._has_explicit_plan_policy(policy):
            policy["preferred_plan_tools"] = [tool_name]

        # RAG adapter 的输入由 application 统一注入检索策略，scene 可按工具覆盖。
        plan_tool_inputs = dict(policy.get("plan_tool_inputs") or {})
        plan_tool_inputs.setdefault(tool_name, dict(tool_input))
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
            tool_executor: ToolExecutor,
            default_tool_name: str,
            scene_policy: Mapping[str, Any],
    ) -> tuple[str, ...]:
        selected: list[str] = []
        selected.extend(self._policy_step_tool_names(scene_policy.get("plan_steps")))
        for key in ("preferred_plan_tools", "default_plan_tools", "plan_tools", "candidate_tools"):
            selected.extend(self._policy_tool_names(scene_policy.get(key)))
        if not selected:
            selected.append(default_tool_name)
        allowed = tool_executor.allowed_tools
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
        run = self._require_single_agent_run(react_run=react_run, plan_run=plan_run)
        if not run.observations:
            return self._agent_execution_result_from_observationless_run(
                message=message,
                mounted_knowledge_sources=mounted_knowledge_sources,
                react_run=react_run,
                plan_run=plan_run,
            )
        observations = self._run_level_observations(run)
        successful_observations = [
            observation for observation in observations if observation.success
        ]
        current_observation = observations[-1]
        documents = self._deduplicate_documents(
            self._documents_from_observations(successful_observations)
        )
        candidate_citations = self._citation_mapper.citations_from_documents(documents)
        final_decision = self._final_decision_from_observations(observations)
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
            citations=citations,
            knowledge_used=knowledge_used,
            final_decision=final_decision,
        )
        answer_mode = self._resolve_answer_mode(
            final_decision=final_decision,
            knowledge_used=knowledge_used,
        )
        adopted_documents = documents if knowledge_used else []
        tool_event = self._tool_event_from_observations(
            observations=observations,
            successful_observation_count=len(successful_observations),
            document_count=len(documents),
            retrieval_trace=retrieval_trace,
            mounted_knowledge_sources=mounted_knowledge_sources,
        )
        current_tool_call = self._current_tool_call_from_run(
            run=run,
            observation=current_observation,
        )
        return AgentRuntimeExecutionResult(
            documents=adopted_documents,
            tool_event=tool_event,
            retrieval_trace=retrieval_trace,
            citations=citations,
            knowledge_used=knowledge_used,
            final_decision=final_decision,
            follow_up_question=self._follow_up_question_from_observations(observations),
            answer_mode=answer_mode,
            react_run=react_run.model_dump() if react_run is not None else None,
            plan_run=plan_run.model_dump() if plan_run is not None else None,
            current_turn_id=self._event_turn_id(react_run) if react_run is not None else None,
            current_step_id=self._event_step_id(plan_run) if plan_run is not None else None,
            current_tool_call=(
                current_tool_call.model_dump()
                if current_tool_call is not None
                else None
            ),
            tool_observation=current_observation.model_dump(),
        )

    def _agent_execution_result_from_observationless_run(
            self,
            *,
            message: str,
            mounted_knowledge_sources: tuple[str, ...],
            react_run: ReActRun | None = None,
            plan_run: PlanRun | None = None,
    ) -> AgentRuntimeExecutionResult:
        run = self._require_single_agent_run(react_run=react_run, plan_run=plan_run)
        if run.workflow_status in {"failed", "cancelled"}:
            raise ChatServiceError(
                status_code=500,
                code="AGENT_RUNTIME_RUN_FAILED",
                message=run.error or run.result_summary or "Agent runtime run failed.",
                request_id=run.request_id,
            )
        final_decision = self._final_decision_from_observationless_run(run)
        follow_up_question = self._follow_up_question_from_observationless_run(run)
        retrieval_trace = RetrievalTrace(
            original_query=message,
            final_query=message,
            rewritten_query=None,
            tool_call_count=0,
            candidate_tools=list(
                self.scene_definition.resolve_candidate_retrieval_tools(
                    mounted_knowledge_sources
                )
            ),
            exit_reason=self._exit_reason_from_observationless_run(run),
            final_decision=final_decision,
            success=False,
            follow_up_question=follow_up_question,
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
            final_decision=final_decision,
            follow_up_question=follow_up_question,
            answer_mode=self._resolve_answer_mode(
                final_decision=final_decision,
                knowledge_used=False,
            ),
            react_run=react_run.model_dump() if react_run is not None else None,
            plan_run=plan_run.model_dump() if plan_run is not None else None,
            current_turn_id=self._event_turn_id(react_run) if react_run is not None else None,
            current_step_id=self._event_step_id(plan_run) if plan_run is not None else None,
            current_tool_call=run.current_tool_call.model_dump() if run.current_tool_call else None,
            tool_observation=None,
        )

    def _final_decision_from_observationless_run(
            self,
            run: ReActRun | PlanRun,
    ) -> RuntimeFinalDecision:
        if isinstance(run, ReActRun):
            turn = self._latest_react_turn(run)
            action_type = turn.action.action_type if turn is not None else None
            if run.workflow_status == "waiting_user" or action_type == "ask_user":
                return "ask_user"
        return "no_evidence"

    def _follow_up_question_from_observationless_run(
            self,
            run: ReActRun | PlanRun,
    ) -> str | None:
        if not isinstance(run, ReActRun):
            return None
        turn = self._latest_react_turn(run)
        if turn is None:
            return None
        if turn.action.instruction:
            return turn.action.instruction
        if turn.result_summary:
            return turn.result_summary
        hitl = turn.metadata.get("hitl") if isinstance(turn.metadata, Mapping) else None
        if isinstance(hitl, Mapping):
            user_prompt = hitl.get("user_prompt")
            if isinstance(user_prompt, str) and user_prompt.strip():
                return user_prompt.strip()
        return None

    def _exit_reason_from_observationless_run(self, run: ReActRun | PlanRun) -> str:
        if run.workflow_status == "waiting_user":
            return "ask_user"
        if run.workflow_status == "succeeded":
            return "no_tool_observation"
        return str(run.workflow_status)

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

    def _latest_react_turn(self, run: ReActRun) -> ReActTurn | None:
        if run.current_turn_id:
            for turn in reversed(run.turns):
                if turn.turn_id == run.current_turn_id:
                    return turn
        return run.turns[-1] if run.turns else None

    def _require_single_agent_run(
            self,
            *,
            react_run: ReActRun | None,
            plan_run: PlanRun | None,
    ) -> ReActRun | PlanRun:
        if (react_run is None) == (plan_run is None):
            raise ChatServiceError(
                status_code=500,
                code="AGENT_RUNTIME_RUN_INVALID",
                message="Exactly one agent runtime run must be provided.",
                request_id=(react_run or plan_run).request_id if (react_run or plan_run) else "N/A",
            )
        return react_run if react_run is not None else plan_run  # type: ignore[return-value]

    def _run_level_observations(self, run: ReActRun | PlanRun) -> list[ToolObservation]:
        if run.observations:
            return list(run.observations)
        raise ChatServiceError(
            status_code=500,
            code="AGENT_RUNTIME_DATA_INCOMPLETE",
            message="Agent runtime run-level observations are missing.",
            request_id=run.request_id,
        )

    def _documents_from_observations(
            self,
            observations: list[ToolObservation],
    ) -> list[Document]:
        documents: list[Document] = []
        for observation in observations:
            documents.extend(self._documents_from_observation(observation))
        return documents

    def _deduplicate_documents(self, documents: list[Document]) -> list[Document]:
        """按稳定证据标识去重，避免多步骤聚合后重复塞入回答上下文。"""
        deduplicated: list[Document] = []
        seen: set[tuple[str, str]] = set()
        for document in documents:
            key = self._document_identity(document)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(document)
        return deduplicated

    def _document_identity(self, document: Document) -> tuple[str, str]:
        metadata = document.metadata
        namespace = str(metadata.get("namespace", "knowledge"))
        value = (
            metadata.get("chunk_id")
            or metadata.get("citation_id")
            or metadata.get("document_id")
            or metadata.get("source_path")
            or document.page_content[:256]
        )
        return namespace, str(value)

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

    def _follow_up_question_from_observations(
            self,
            observations: list[ToolObservation],
    ) -> str | None:
        for observation in reversed(observations):
            value = self._follow_up_question_from_observation(observation)
            if value:
                return value
        return None

    def _final_decision_from_observations(
            self,
            observations: list[ToolObservation],
    ) -> RuntimeFinalDecision | None:
        decisions = [
            decision
            for observation in observations
            if (decision := self._final_decision_from_observation(observation)) is not None
        ]
        if "answer_with_evidence" in decisions:
            return "answer_with_evidence"
        if decisions:
            return decisions[-1]
        if any(observation.success for observation in observations):
            return "no_evidence"
        return "retrieval_failed"

    def _raw_retrieval_trace(self, observation: ToolObservation) -> dict[str, Any]:
        trace = observation.trace.get("retrieval_trace")
        return dict(trace) if isinstance(trace, Mapping) else {}

    def _retrieval_trace_from_observations(
            self,
            *,
            message: str,
            observations: list[ToolObservation],
            mounted_knowledge_sources: tuple[str, ...],
            citations: list[Citation],
            knowledge_used: bool,
            final_decision: RuntimeFinalDecision | None,
    ) -> RetrievalTrace:
        raw_traces = [self._raw_retrieval_trace(observation) for observation in observations]
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
            ),
            exit_reason=self._last_trace_text(raw_traces, "exit_reason"),
            final_decision=final_decision,
            success=any(observation.success for observation in observations),
            follow_up_question=self._follow_up_question_from_observations(observations),
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
            raw_trace = self._raw_retrieval_trace(observation)
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
        return list(self.scene_definition.resolve_candidate_retrieval_tools(
            mounted_knowledge_sources
        ))

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
    ) -> dict[str, Any]:
        current_observation = observations[-1]
        raw_trace = self._raw_retrieval_trace(current_observation)
        return {
            "stage": "retrieval",
            "mode": "agent_runtime_tool",
            "tool_name": current_observation.tool_name,
            "tool_names": self._tool_names_from_observations(observations),
            "retrieval_policy": self._build_policy_summary(self.scene_definition.retrieval_policy),
            "candidate_tools": list(
                self.scene_definition.resolve_candidate_retrieval_tools(
                    mounted_knowledge_sources
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
                self._raw_retrieval_trace(observation) for observation in observations
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

    def _current_tool_call_from_run(
            self,
            *,
            run: ReActRun | PlanRun,
            observation: ToolObservation,
    ) -> ToolExecutionMetadata | None:
        return run.current_tool_call or observation.execution

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


