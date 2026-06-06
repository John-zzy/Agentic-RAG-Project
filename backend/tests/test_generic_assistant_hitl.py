from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from backend.application.runtime.api.app import create_app
from backend.application.runtime.api.chat.schemas import (
    ChatRequest,
    ChatResumeRequest,
    HitlResumePayload,
)
from backend.application.runtime.assembly.runtime_factory import ChatGraphRuntime
from backend.application.runtime.service import ChatService
from backend.platform.agent_runtime.chat_graph.contracts import (
    HitlResumeInput,
    HitlWaitInput,
)
from backend.platform.config.settings import AppSettings
from backend.platform.memory.base.session_store import SQLiteSessionStore
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.rag.retrieval.documents import DocumentChunkRetrievalResult
from backend.platform.search_foundation import VectorStoreDocument
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.config import DEFAULT_RUNTIME_CHECKPOINT_NS
from backend.scenes.generic_assistant.definition import build_generic_assistant_scene_definition
from backend.scenes.generic_assistant.hitl import GenericAssistantHitlPlanner, GenericAssistantHitlWaitPlan
from backend.scenes.generic_assistant.tools import (
    GenericHitlFakeExternalApiTool,
    GenericHitlFakeWriteTool,
)
from backend.tests.test_support import make_test_runtime_dir


class EmptyDocumentRetrievalService:
    """测试用空文档检索服务，固定返回无命中。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> list[Any]:
        """记录检索参数并返回空结果。"""
        self.calls.append(dict(kwargs))
        return []


class KeywordDocumentRetrievalService:
    """测试用文档检索服务，只在用户补充指定关键词后返回命中文档。"""

    def __init__(self, *, allowed_terms: tuple[str, ...] = ("安全合规",)) -> None:
        self.allowed_terms = allowed_terms
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> list[DocumentChunkRetrievalResult]:
        """记录检索参数；命中关键词时返回一条稳定文档。"""
        self.calls.append(dict(kwargs))
        query = str(kwargs.get("query") or "")
        if not any(term in query for term in self.allowed_terms):
            return []
        return [
            DocumentChunkRetrievalResult(
                document=VectorStoreDocument(
                    id="doc-hitl-policy",
                    content="安全合规政策要求审批操作必须记录操作者、原因和时间。",
                    metadata={
                        "namespace": "documents",
                        "source_path": "security-policy.md",
                        "source_name": "security-policy.md",
                        "chunk_id": "chunk-hitl-policy",
                        "chunk_index": 0,
                    },
                ),
                score=0.93,
                vector_score=0.93,
                vector_rank=1,
                matched_by=["vector"],
            )
        ]


class NoModelCalls:
    """测试用模型客户端，任何模型调用都表示 HITL 等待分支走错了。"""

    def get_runnable(
            self,
            complexity: str = "simple",
            prompt_template: Any | None = None,
            *,
            output_parser: Any | None = None,
    ) -> Any:
        """返回一个会失败的 runnable，避免测试误调用模型。"""
        del complexity, prompt_template, output_parser
        raise AssertionError("HITL waiting branch should not call model.")

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        """HITL 等待分支不应同步调用模型。"""
        del runnable, input, config
        raise AssertionError("HITL waiting branch should not invoke model.")

    def stream_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> Any:
        """HITL 等待分支不应流式调用模型。"""
        del runnable, input, config
        raise AssertionError("HITL waiting branch should not stream model.")


class FollowUpSafeModel(NoModelCalls):
    """测试默认 ask_user fallback 时使用的模型；非证据分支不会调用它。"""

    def get_runnable(
            self,
            complexity: str = "simple",
            prompt_template: Any | None = None,
            *,
            output_parser: Any | None = None,
    ) -> Any:
        """提供兜底 runnable；如果真的被调用也能返回稳定文本。"""
        del complexity, prompt_template, output_parser
        return RunnableLambda(lambda _: "unused")


class FallbackSuggestionModel(FollowUpSafeModel):
    """测试用 HITL 建议模型，返回非 JSON 以触发 planner 的兜底建议。"""

    def get_runnable(
            self,
            complexity: str = "simple",
            prompt_template: Any | None = None,
            *,
            output_parser: Any | None = None,
    ) -> Any:
        """返回一个稳定的非 JSON 输出，避免测试依赖真实模型。"""
        del complexity, prompt_template, output_parser
        return RunnableLambda(lambda _: "not-json")

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        """执行非 JSON runnable，让 HITL planner 走兜底解析。"""
        del config
        return str(runnable.invoke(input))


class GeneratedSuggestionModel(FollowUpSafeModel):
    """测试用 HITL 建议模型，返回模型生成的动态选项。"""

    def __init__(self) -> None:
        self.invoke_runnable_calls: list[dict[str, Any]] = []

    def get_runnable(
            self,
            complexity: str = "simple",
            prompt_template: Any | None = None,
            *,
            output_parser: Any | None = None,
    ) -> Any:
        """构造稳定 JSON 输出，模拟真实模型生成 HITL 选项。"""
        del complexity, prompt_template, output_parser
        payload = {
            "question": "你希望我按哪类信息继续检索？",
            "suggestions": [
                {
                    "label": "缺少文档主题",
                    "description": "需要明确想查哪个知识主题，才能继续召回。",
                    "value": "我想查询当前知识库中与这个问题相关的具体文档主题。",
                },
                {
                    "label": "缺少关键词",
                    "description": "需要补充可用于检索的关键词或术语。",
                    "value": "我想查询当前问题涉及的关键词、术语或流程说明。",
                },
                {
                    "label": "缺少业务范围",
                    "description": "需要限定查询的业务规则、流程或文档范围。",
                    "value": "我想查询当前知识库中的业务规则、操作流程或限制条件。",
                },
            ],
        }
        return RunnableLambda(lambda _: json.dumps(payload, ensure_ascii=False))

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        """记录模型调用，便于测试确认 HITL 建议来自模型。"""
        self.invoke_runnable_calls.append({"input": input, "config": config})
        return str(runnable.invoke(input))


class EvidenceAnswerModel(FollowUpSafeModel):
    """测试证据回答分支的模型，返回固定答案并记录调用次数。"""

    def __init__(self, answer: str = "根据安全合规政策，需要记录审批动作。[1]") -> None:
        self.answer = answer
        self.invoke_runnable_calls: list[dict[str, Any]] = []

    def get_runnable(
            self,
            complexity: str = "simple",
            prompt_template: Any | None = None,
            *,
            output_parser: Any | None = None,
    ) -> Any:
        """构造一个稳定 runnable，避免测试依赖真实模型。"""
        del complexity, output_parser
        runnable = RunnableLambda(lambda _: self.answer)
        if prompt_template is None:
            return runnable
        return prompt_template | runnable

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        """记录证据回答调用，并返回固定答案。"""
        self.invoke_runnable_calls.append({"input": input, "config": config})
        return str(runnable.invoke(input, config=config))


def _build_settings(test_name: str) -> tuple[AppSettings, SQLiteSessionStore]:
    runtime_dir = make_test_runtime_dir(test_name)
    sqlite_path = runtime_dir / "chat-sessions.db"
    app_settings = AppSettings(
        data_dir=runtime_dir,
        app={"active_scene": "generic_assistant"},
        session={"sqlite_path": sqlite_path, "window_size": 3},
    )
    return app_settings, SQLiteSessionStore(sqlite_path=sqlite_path)


def _build_chat_service(
        test_name: str,
        *,
        hitl_clarification_enabled: bool,
        include_hitl_test_tools: bool = False,
        model: Any | None = None,
        document_retrieval_service: Any | None = None,
) -> ChatService:
    app_settings, session_store = _build_settings(test_name)
    scene_definition = build_generic_assistant_scene_definition(
        app_settings=app_settings,
        document_retrieval_service=(
            document_retrieval_service or EmptyDocumentRetrievalService()
        ),  # type: ignore[arg-type]
        hitl_clarification_enabled=hitl_clarification_enabled,
        include_hitl_test_tools=include_hitl_test_tools,
    )
    return ChatService(
        scene_definition=scene_definition,
        app_settings=app_settings,
        session_store=session_store,
        context_builder=PromptContextBuilder(window_size=3),
        model=model or FallbackSuggestionModel(),
    )


def _build_runtime(test_name: str) -> ChatGraphRuntime:
    runtime_dir = make_test_runtime_dir(test_name)
    return ChatGraphRuntime(
        checkpointer=SQLiteLangGraphCheckpointer(runtime_dir / "langgraph.db")
    )


def _parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    """把测试客户端收到的 SSE 文本拆成事件名和 JSON 数据。"""
    events: list[dict[str, Any]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_payload = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            if line.startswith("data: "):
                data_payload = line[len("data: "):]
        events.append({"event": event_name, "data": data_payload})
    return events


def _to_wait_input(
        *,
        session_id: str,
        request_id: str,
        plan: GenericAssistantHitlWaitPlan,
) -> HitlWaitInput:
    """把 generic HITL 等待计划转换成 runtime 测试输入。"""
    return HitlWaitInput(
        session_id=session_id,
        request_id=request_id,
        reason=plan.reason,
        pending_action=plan.pending_action,
        allowed_actions=plan.allowed_actions,
        proposed_tool_call=plan.proposed_tool_call,
        suggested_responses=plan.suggested_responses,
        allow_freeform_response=plan.allow_freeform_response,
        metadata=plan.metadata,
    )


def test_generic_assistant_ask_user_stays_follow_up_when_hitl_disabled() -> None:
    service = _build_chat_service(
        "generic-hitl-disabled-keeps-follow-up",
        hitl_clarification_enabled=False,
        model=FollowUpSafeModel(),
    )

    response = service.chat(ChatRequest(message="你好"))

    assert response.status is None
    assert response.hitl is None
    assert response.knowledge_used is False
    assert response.citations == []
    assert "请补充" in response.answer
    assert response.retrieval_trace is not None
    assert response.retrieval_trace.final_decision == "ask_user"


def test_generic_assistant_ask_user_creates_clarification_wait_when_hitl_enabled() -> None:
    service = _build_chat_service(
        "generic-hitl-enabled-ask-user-wait",
        hitl_clarification_enabled=True,
    )

    response = service.chat(ChatRequest(message="你好"))

    assert response.status == "waiting_user"
    assert response.state == "waiting_user"
    assert response.state_event == "interrupt"
    assert response.answer == response.hitl.reason
    assert response.knowledge_used is False
    assert response.citations == []
    assert response.hitl is not None
    assert response.hitl.thread_id == response.session_id
    assert response.hitl.pending_action == "clarification"
    assert response.hitl.allowed_actions == ["respond", "reject"]
    assert response.hitl.allow_freeform_response is True
    assert response.hitl.metadata["mode"] == "react"
    assert response.hitl.metadata["react_run_id"]
    assert response.hitl.metadata["current_turn_id"] == "turn-1"
    assert {item.suggestion_id for item in response.hitl.suggested_responses} == {
        "clarify_topic",
        "clarify_term",
        "clarify_scope",
    }
    assert response.retrieval_trace is not None
    assert response.retrieval_trace.final_decision == "ask_user"


def test_generic_assistant_hitl_uses_model_generated_clarification_options() -> None:
    model = GeneratedSuggestionModel()
    service = _build_chat_service(
        "generic-hitl-model-generated-options",
        hitl_clarification_enabled=True,
        model=model,
    )

    response = service.chat(ChatRequest(message="你好"))

    assert response.status == "waiting_user"
    assert response.hitl is not None
    assert response.hitl.reason == "你希望我按哪类信息继续检索？"
    assert [item.suggestion_id for item in response.hitl.suggested_responses] == [
        "model_clarify_1",
        "model_clarify_2",
        "model_clarify_3",
    ]
    assert response.hitl.suggested_responses[0].label == "缺少文档主题"
    assert response.hitl.suggested_responses[0].metadata["source"] == "model"
    assert "当前知识库" in response.hitl.suggested_responses[0].value
    assert model.invoke_runnable_calls
    assert model.invoke_runnable_calls[0]["input"]["user_message"] == "你好"


def test_generic_assistant_request_can_enable_clarification_wait() -> None:
    service = _build_chat_service(
        "generic-hitl-request-enabled-ask-user-wait",
        hitl_clarification_enabled=False,
    )

    response = service.chat(
        ChatRequest(message="你好", hitl_clarification_enabled=True)
    )

    assert response.status == "waiting_user"
    assert response.hitl is not None
    assert response.hitl.pending_action == "clarification"
    assert response.hitl.allowed_actions == ["respond", "reject"]


def test_generic_assistant_chat_resume_api_accepts_suggested_response() -> None:
    service = _build_chat_service(
        "generic-hitl-resume-api-suggested-response",
        hitl_clarification_enabled=True,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        waiting_response = client.post("/chat", json={"message": "你好"})
        assert waiting_response.status_code == 200
        waiting_payload = waiting_response.json()

        resume_response = client.post(
            "/chat/resume",
            json={
                "session_id": waiting_payload["session_id"],
                "interrupt_id": waiting_payload["hitl"]["interrupt_id"],
                "action": "respond",
                "payload": {
                    "source": "suggested_response",
                    "suggestion_id": "clarify_topic",
                },
            },
        )

    assert resume_response.status_code == 200
    resume_payload = resume_response.json()
    assert resume_payload["session_id"] == waiting_payload["session_id"]
    assert resume_payload["status"] == "succeeded"
    assert resume_payload["final_state"] == "succeeded"
    assert resume_payload["hitl"] is None
    assert resume_payload["resume_payload"]["action"] == "respond"
    assert resume_payload["resume_payload"]["source"] == "suggested_response"
    assert resume_payload["resume_payload"]["suggestion_id"] == "clarify_topic"
    assert "你好" in resume_payload["resume_payload"]["response"]
    assert resume_payload["knowledge_used"] is False
    assert resume_payload["citations"] == []
    assert resume_payload["retrieval_trace"]["final_decision"] == "ask_user"


def test_generic_assistant_chat_resume_respond_continues_retrieval_with_clarification() -> None:
    document_service = KeywordDocumentRetrievalService()
    model = EvidenceAnswerModel()
    service = _build_chat_service(
        "generic-hitl-respond-continues-retrieval",
        hitl_clarification_enabled=True,
        model=model,
        document_retrieval_service=document_service,
    )

    waiting_response = service.chat(ChatRequest(message="你好"))
    response = service.resume(
        ChatResumeRequest(
            session_id=waiting_response.session_id,
            interrupt_id=waiting_response.hitl.interrupt_id,
            action="respond",
            payload=HitlResumePayload(
                response="安全合规政策",
                source="freeform",
            ),
        )
    )

    assert response.status == "succeeded"
    assert response.final_state == "succeeded"
    assert response.answer == "根据安全合规政策，需要记录审批动作。[1]"
    assert response.knowledge_used is True
    assert len(response.citations) == 1
    assert response.retrieval_trace is not None
    assert response.retrieval_trace["final_decision"] == "answer_with_evidence"
    assert response.resume_payload is not None
    assert response.resume_payload.response == "安全合规政策"
    assert response.resume_payload.source == "freeform"
    assert [call["query"] for call in document_service.calls] == ["安全合规政策"]
    business_calls = [
        call
        for call in model.invoke_runnable_calls
        if "react_allowed_tools_json" not in call["input"]
    ]
    assert len(business_calls) == 2
    assert business_calls[0]["input"]["user_message"] == "你好"
    assert "安全合规政策" in str(business_calls[1]["input"])
    assert service.session_store.count_messages(response.session_id) == 2


def test_generic_assistant_chat_resume_api_approves_test_write_tool() -> None:
    service = _build_chat_service(
        "generic-hitl-resume-api-approve-write",
        hitl_clarification_enabled=True,
        include_hitl_test_tools=True,
    )
    planner = GenericAssistantHitlPlanner()
    plan = planner.build_write_tool_wait(
        tool_name="generic_hitl_fake_write",
        operation="写入测试记录",
        args={"item_id": "item-api", "content": "approve 后写入"},
    )
    service.session_store.create_session(
        session_id="session-api-approve",
        scene="generic_assistant",
    )
    service.graph_runtime.create_hitl_wait(
        wait=_to_wait_input(
            session_id="session-api-approve",
            request_id="req-api-approve-wait",
            plan=plan,
        ),
        interrupt_id=plan.interrupt_id,
    )

    response = service.resume(
        ChatResumeRequest(
            session_id="session-api-approve",
            interrupt_id=plan.interrupt_id,
            action="approve",
        )
    )

    assert response.status == "succeeded"
    assert response.final_state == "succeeded"
    assert response.hitl is None
    assert response.answer == "已批准并执行待审批操作。"
    assert response.resume_payload is not None
    assert response.resume_payload.action == "approve"
    restored = service.graph_runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": "session-api-approve",
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )
    assert restored is not None
    checkpoint_values = restored.checkpoint["channel_values"]
    assert checkpoint_values["status"] == "succeeded"
    assert checkpoint_values["final_state"] == "succeeded"
    assert checkpoint_values["metadata"]["hitl_tool_result"]["metadata"]["side_effect"] == (
        "local_write"
    )


def test_generic_assistant_chat_resume_api_rejects_invalid_or_stale_interrupt() -> None:
    service = _build_chat_service(
        "generic-hitl-resume-api-invalid",
        hitl_clarification_enabled=True,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        waiting_response = client.post("/chat", json={"message": "你好"})
        waiting_payload = waiting_response.json()

        invalid_response = client.post(
            "/chat/resume",
            json={
                "session_id": waiting_payload["session_id"],
                "interrupt_id": "interrupt-stale",
                "action": "respond",
                "payload": {"response": "安全合规政策", "source": "freeform"},
            },
        )
        illegal_action_response = client.post(
            "/chat/resume",
            json={
                "session_id": waiting_payload["session_id"],
                "interrupt_id": waiting_payload["hitl"]["interrupt_id"],
                "action": "approve",
            },
        )
        accepted_response = client.post(
            "/chat/resume",
            json={
                "session_id": waiting_payload["session_id"],
                "interrupt_id": waiting_payload["hitl"]["interrupt_id"],
                "action": "reject",
            },
        )
        duplicate_response = client.post(
            "/chat/resume",
            json={
                "session_id": waiting_payload["session_id"],
                "interrupt_id": waiting_payload["hitl"]["interrupt_id"],
                "action": "reject",
            },
        )

    assert invalid_response.status_code == 409
    assert "interrupt_id" in invalid_response.json()["detail"]["message"]
    assert illegal_action_response.status_code == 409
    assert "action is not allowed" in illegal_action_response.json()["detail"]["message"]
    assert accepted_response.status_code == 200
    assert accepted_response.json()["status"] == "cancelled"
    assert accepted_response.json()["final_state"] == "cancelled"
    assert accepted_response.json()["answer"] == "已拒绝该人工等待项，未执行待审批调用。"
    assert accepted_response.json()["knowledge_used"] is False
    assert accepted_response.json()["citations"] == []
    assert duplicate_response.status_code == 409
    assert "terminal" in duplicate_response.json()["detail"]["message"]


def test_generic_assistant_sse_emits_waiting_user_for_clarification() -> None:
    service = _build_chat_service(
        "generic-hitl-sse-waiting-user",
        hitl_clarification_enabled=True,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "你好", "stream": True})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == [
        "start",
        "history",
        "tool",
        "waiting_user",
    ]
    tool_payload = events[2]["data"]
    assert '"turn_status": "waiting_user"' in tool_payload
    waiting_payload = events[-1]["data"]
    assert '"status": "waiting_user"' in waiting_payload
    assert '"pending_action": "clarification"' in waiting_payload
    assert '"suggested_responses": [' in waiting_payload


def test_generic_assistant_sse_resume_emits_resume_then_done() -> None:
    service = _build_chat_service(
        "generic-hitl-sse-resume",
        hitl_clarification_enabled=True,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        waiting_response = client.post("/chat", json={"message": "你好"})
        waiting_payload = waiting_response.json()
        response = client.post(
            "/chat/resume",
            json={
                "session_id": waiting_payload["session_id"],
                "interrupt_id": waiting_payload["hitl"]["interrupt_id"],
                "action": "reject",
                "stream": True,
            },
        )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["resume", "done"]
    assert waiting_payload["hitl"]["interrupt_id"] in events[0]["data"]
    assert '"action": "reject"' in events[0]["data"]
    assert '"status": "cancelled"' in events[1]["data"]
    assert '"final_state": "cancelled"' in events[1]["data"]


def test_generic_assistant_sse_resume_invalid_interrupt_emits_error_only() -> None:
    service = _build_chat_service(
        "generic-hitl-sse-resume-invalid",
        hitl_clarification_enabled=True,
    )
    app = create_app(chat_service=service)

    with TestClient(app) as client:
        waiting_response = client.post("/chat", json={"message": "你好"})
        waiting_payload = waiting_response.json()
        response = client.post(
            "/chat/resume",
            json={
                "session_id": waiting_payload["session_id"],
                "interrupt_id": "stale-interrupt",
                "action": "reject",
                "stream": True,
            },
        )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["error"]
    assert "interrupt_id does not match" in events[0]["data"]


def test_generic_write_tool_waits_until_approve_before_execution() -> None:
    runtime = _build_runtime("generic-hitl-write-tool-approve")
    planner = GenericAssistantHitlPlanner()
    write_tool = GenericHitlFakeWriteTool()
    plan = planner.build_write_tool_wait(
        tool_name=write_tool.name,
        operation="写入测试记录",
        args={"item_id": "item-1", "content": "需要审批后写入"},
    )

    wait_result = runtime.create_hitl_wait(
        wait=_to_wait_input(
            session_id="session-write",
            request_id="req-write-wait",
            plan=plan,
        ),
        interrupt_id=plan.interrupt_id,
    )
    assert write_tool.calls == []
    assert wait_result.state["hitl"]["suggested_responses"] == []
    assert wait_result.state["hitl"]["allow_freeform_response"] is False

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-write",
            request_id="req-write-approve",
            interrupt_id=plan.interrupt_id,
            action="approve",
        ),
        approve_executor=lambda proposed: write_tool.invoke(
            **dict(proposed["args"])
        ).model_dump(),
    )

    assert write_tool.calls == [{"item_id": "item-1", "content": "需要审批后写入"}]
    assert result.state["status"] == "succeeded"
    assert result.state["final_state"] == "succeeded"
    assert result.tool_result is not None
    assert result.tool_result["metadata"]["side_effect"] == "local_write"


def test_generic_external_api_reject_skips_side_effect() -> None:
    runtime = _build_runtime("generic-hitl-external-api-reject")
    planner = GenericAssistantHitlPlanner()
    external_tool = GenericHitlFakeExternalApiTool()
    plan = planner.build_external_api_wait(
        tool_name=external_tool.name,
        operation="调用外部测试 API",
        args={"endpoint": "https://example.test/webhook", "payload": {"approved": True}},
    )

    runtime.create_hitl_wait(
        wait=_to_wait_input(
            session_id="session-external",
            request_id="req-external-wait",
            plan=plan,
        ),
        interrupt_id=plan.interrupt_id,
    )
    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-external",
            request_id="req-external-reject",
            interrupt_id=plan.interrupt_id,
            action="reject",
        ),
        approve_executor=lambda proposed: external_tool.invoke(
            **dict(proposed["args"])
        ).model_dump(),
    )

    assert external_tool.calls == []
    assert result.state["status"] == "cancelled"
    assert result.state["final_state"] == "cancelled"
    assert "未执行" in result.state["answer"]


def test_generic_hitl_test_tools_are_opt_in_and_do_not_include_ecommerce() -> None:
    default_service = _build_chat_service(
        "generic-hitl-default-tools",
        hitl_clarification_enabled=False,
    )
    enabled_service = _build_chat_service(
        "generic-hitl-test-tools-enabled",
        hitl_clarification_enabled=True,
        include_hitl_test_tools=True,
    )

    default_tools = {tool.name for tool in default_service.scene_definition.build_tools()}
    enabled_tools = {tool.name for tool in enabled_service.scene_definition.build_tools()}

    assert "generic_hitl_fake_write" not in default_tools
    assert "generic_hitl_fake_external_api" not in default_tools
    assert "generic_hitl_fake_write" in enabled_tools
    assert "generic_hitl_fake_external_api" in enabled_tools
    assert not any(tool_name.startswith("ecommerce") for tool_name in enabled_tools)
    assert enabled_service.scene_definition.metadata["hitl"]["business_extensions_in_scope"] is False
