from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from backend.application.runtime.api.chat.schemas import (
    ChatRequest,
    ChatResumeRequest,
    ChatResumeResponse,
    ChatResponse,
    HitlResumePayload,
    HitlState,
)
from backend.application.runtime.chat_service_parts.contracts import ChatServiceError, PreparedChatTurn
from backend.application.runtime.graph_runtime import HitlResumeError, HitlResumeInput, HitlWaitInput
from backend.platform.agent_runtime.tool_executor import ToolExecutor
from backend.scenes.generic_assistant.hitl import (
    GenericAssistantHitlOptions,
    GenericAssistantHitlPlanner,
    GenericAssistantHitlWaitPlan,
)


class ChatHitlMixin:
    def _try_create_hitl_wait_response(self, prepared: PreparedChatTurn) -> ChatResponse | None:
        """HITL 开启时，把 ask_user 追问改成等待用户补充。"""
        planner = self._resolve_hitl_planner(prepared)
        if not planner.should_wait_for_clarification(prepared):
            return None
        wait_plan = planner.build_clarification_wait(prepared)
        result = self.graph_runtime.create_hitl_wait(
            wait=self._build_hitl_wait_input(prepared=prepared, wait_plan=wait_plan),
            interrupt_id=wait_plan.interrupt_id,
        )
        hitl_payload = result.state.get("hitl")
        if hitl_payload is None:
            raise ChatServiceError(
                status_code=500,
                code="HITL_WAIT_STATE_MISSING",
                message="HITL waiting state was not created.",
                request_id=prepared.request_id,
            )
        return self._build_hitl_wait_response(
            prepared=prepared,
            hitl=HitlState(**hitl_payload),
            result_state=result.state,
        )

    def _build_hitl_wait_input(
            self,
            *,
            prepared: PreparedChatTurn,
            wait_plan: GenericAssistantHitlWaitPlan,
    ) -> HitlWaitInput:
        """把 generic HITL 等待计划转换成 graph runtime 能保存的输入。"""
        return HitlWaitInput(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            reason=wait_plan.reason,
            pending_action=wait_plan.pending_action,
            allowed_actions=wait_plan.allowed_actions,
            proposed_tool_call=wait_plan.proposed_tool_call,
            suggested_responses=wait_plan.suggested_responses,
            allow_freeform_response=wait_plan.allow_freeform_response,
            metadata={
                "scene": prepared.scene_metadata.scene,
                "agent": prepared.scene_metadata.agent,
                "agent_mode": prepared.agent_mode,
                "answer_mode": prepared.answer_mode,
                "final_decision": prepared.final_decision,
                **self._build_agent_hitl_metadata(prepared),
                **dict(wait_plan.metadata or {}),
            },
        )

    def _build_hitl_wait_response(
            self,
            *,
            prepared: PreparedChatTurn,
            hitl: HitlState,
            result_state: Mapping[str, Any],
    ) -> ChatResponse:
        """返回等待态响应；这里不生成模型答案，也不写最终对话轮次。"""
        return ChatResponse(
            session_id=prepared.session_id,
            request_id=prepared.request_id,
            answer=hitl.reason,
            knowledge_used=False,
            scene=prepared.scene_metadata.scene,
            agent=prepared.scene_metadata.agent,
            status="waiting_user",
            state="waiting_user",
            run_id=str(result_state.get("run_id") or "") or None,
            state_event="interrupt",
            hitl=hitl,
            citations=[],
            retrieval_trace=prepared.retrieval_trace.model_copy(
                update={
                    "citations": [],
                    "knowledge_used": False,
                    "top_k_chunks": [],
                }
            ),
        )

    def _resolve_hitl_planner(self, prepared: PreparedChatTurn) -> GenericAssistantHitlPlanner:
        """优先使用本次请求显式打开的 HITL 澄清开关。"""
        if not prepared.hitl_clarification_enabled:
            return self._generic_hitl_planner
        return GenericAssistantHitlPlanner(
            GenericAssistantHitlOptions(clarification_enabled=True),
            suggestion_model=self.model,
        )

    def _build_hitl_wait_event(self, response: ChatResponse) -> dict[str, Any]:
        """构造 SSE waiting_user 事件，只暴露前端恢复所需的字段。"""
        hitl = response.hitl.model_dump() if response.hitl is not None else None
        return {
            "session_id": response.session_id,
            "request_id": response.request_id,
            "status": response.status,
            "state": response.state,
            "run_id": response.run_id,
            "state_event": response.state_event,
            "hitl": hitl,
        }

    def _run_hitl_resume(
            self,
            *,
            payload: ChatResumeRequest,
            request_id: str,
    ) -> Any:
        """调用 graph runtime 恢复等待点，并把 runtime 错误转换成 API 错误。"""
        timestamp = datetime.now(UTC).isoformat()
        self._ensure_resume_session_ready(
            session_id=payload.session_id,
            timestamp=timestamp,
            request_id=request_id,
            scene=self.scene_definition.scene,
        )
        try:
            return self.graph_runtime.resume_hitl(
                resume=HitlResumeInput(
                    session_id=payload.session_id,
                    request_id=request_id,
                    interrupt_id=payload.interrupt_id,
                    action=payload.action,
                    payload=payload.payload.model_dump(),
                    metadata={
                        "scene": self.scene_definition.scene,
                        "agent": self._scene_metadata().agent,
                    },
                ),
                approve_executor=self._execute_approved_scene_tool,
                respond_handler=self._handle_clarification_response,
            )
        except HitlResumeError as exc:
            raise ChatServiceError(
                status_code=409,
                code="HITL_RESUME_REJECTED",
                message=str(exc),
                request_id=request_id,
            ) from exc

    def _execute_approved_scene_tool(self, proposed_tool_call: Mapping[str, Any]) -> dict[str, Any]:
        """执行用户批准的 scene 工具；仍通过 ToolExecutor 做工具边界校验。"""
        tool_name = str(proposed_tool_call.get("tool_name") or "").strip()
        args = proposed_tool_call.get("args")
        if not tool_name:
            raise HitlResumeError("proposed_tool_call.tool_name is required.")
        if not isinstance(args, Mapping):
            raise HitlResumeError("proposed_tool_call.args must be an object.")

        tools = {tool.name: tool for tool in self.scene_definition.build_tools()}
        if tool_name not in tools:
            raise HitlResumeError("proposed tool is not available in current scene.")

        executor = ToolExecutor(tools=tools, allowed_tools={tool_name})
        observation = executor.execute(tool_name=tool_name, input_payload=dict(args))
        return observation.model_dump()

    def _handle_clarification_response(self, resume_payload: Mapping[str, Any]) -> dict[str, Any]:
        """用用户补充内容继续跑 generic 检索；没有证据时仍走原有 fallback。"""
        session_id = str(resume_payload.get("session_id") or "").strip()
        request_id = str(resume_payload.get("request_id") or "").strip()
        response = str(resume_payload.get("response") or "").strip()
        if not session_id or not request_id:
            raise HitlResumeError("respond resume payload must include session_id and request_id.")

        prepared = self._prepare_existing_session_turn(
            session_id=session_id,
            request_id=request_id,
            timestamp=datetime.now(UTC).isoformat(),
            message=response,
        )
        answer, citations = self._generate_answer_direct(prepared)
        self._persist_turn(prepared=prepared, answer=answer, citations=citations)
        return {
            "status": "succeeded",
            "answer": answer,
            "knowledge_used": prepared.knowledge_used,
            "citations": [citation.model_dump() for citation in citations],
            "retrieval_trace": prepared.retrieval_trace.model_dump(),
        }

    def _build_resume_response(
            self,
            *,
            payload: ChatResumeRequest,
            request_id: str,
            result_state: Mapping[str, Any],
    ) -> ChatResumeResponse:
        """把 graph runtime 的最新状态转换成 `/chat/resume` 响应体。"""
        raw_hitl = result_state.get("hitl")
        raw_resume_payload = result_state.get("hitl_resume")
        return ChatResumeResponse(
            session_id=payload.session_id,
            request_id=request_id,
            status=str(result_state.get("status") or "succeeded"),
            state=str(result_state.get("status") or "succeeded"),
            final_state=(
                str(result_state.get("final_state"))
                if result_state.get("final_state") is not None
                else None
            ),
            run_id=(
                str(result_state.get("run_id"))
                if result_state.get("run_id") is not None
                else None
            ),
            state_event=(
                str(result_state.get("state_event"))
                if result_state.get("state_event") is not None
                else None
            ),
            answer=(
                str(result_state.get("answer"))
                if result_state.get("answer") is not None
                else None
            ),
            knowledge_used=bool(result_state.get("knowledge_used", False)),
            citations=list(result_state.get("citations") or []),
            retrieval_trace=(
                dict(result_state.get("retrieval_trace") or {})
                if result_state.get("retrieval_trace")
                else None
            ),
            hitl=HitlState(**raw_hitl) if isinstance(raw_hitl, Mapping) else None,
            resume_payload=(
                HitlResumePayload(**raw_resume_payload)
                if isinstance(raw_resume_payload, Mapping)
                else None
            ),
        )



