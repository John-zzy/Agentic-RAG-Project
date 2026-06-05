from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from backend.application.runtime.api.chat.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    HitlState,
    RetrievalTrace,
)
from backend.application.runtime.assembly.runtime_factory import (
    ChatGraphRuntime,
    HitlResumeError,
    HitlResumeInput,
    HitlWaitInput,
)
from backend.application.runtime.service import ChatService
from backend.application.runtime.stream_events import (
    GraphRuntimeStreamEvent,
    GraphStreamEventMapper,
)
from backend.platform.config.settings import AppSettings
from backend.platform.memory.base.session_store import SQLiteSessionStore
from backend.platform.memory.chat.prompt_context import PromptContextBuilder
from backend.platform.workflow.langgraph.checkpointer import SQLiteLangGraphCheckpointer
from backend.platform.workflow.langgraph.config import (
    DEFAULT_RUNTIME_CHECKPOINT_NS,
    build_runtime_graph_config,
)
from backend.platform.workflow.langgraph.lifecycle import (
    GraphRunLifecycleRecorder,
)
from backend.platform.workflow.langgraph.state import (
    RuntimeGraphState,
    build_runtime_hitl_state,
    build_runtime_graph_state,
)
from backend.platform.workflow.state_machine import (
    InvalidWorkflowTransitionError,
    UnknownWorkflowStateError,
    is_terminal,
    validate_transition,
)
from backend.scenes.base import SceneDefinition, SceneFallbackPolicy, SceneRetrievalPolicy
from backend.tests.test_support import make_test_runtime_dir


@dataclass(frozen=True)
class _SceneMetadata:
    scene: str = "generic_assistant"
    agent: str | None = None


@dataclass(frozen=True)
class _PreparedGraphTurn:
    session_id: str
    request_id: str
    user_message: str
    answer_mode: str
    final_decision: str | None
    knowledge_used: bool
    citations: list[Citation]
    retrieval_trace: RetrievalTrace
    scene_metadata: _SceneMetadata = _SceneMetadata()
    agent_mode: str = "react"
    agent_mode_reason: str | None = None
    agent_mode_signals: dict[str, Any] | None = None


class _SearchRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def search(self, query: str, **kwargs: Any) -> list[Document]:
        del query, kwargs
        return self._documents


class _FakeModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.invoke_runnable_calls: list[dict[str, Any]] = []

    def get_runnable(
        self,
        complexity: str = "simple",
        prompt_template: Any | None = None,
        *,
        output_parser: Any | None = None,
    ) -> Any:
        del complexity, output_parser
        if _is_react_selector_prompt(prompt_template):
            return RunnableLambda(_fake_react_selector_output)
        answer = RunnableLambda(lambda _: self.answer)
        if prompt_template is None:
            return answer
        return prompt_template | answer

    def invoke_runnable(self, runnable: Any, input: Any, *, config: Any | None = None) -> str:
        self.invoke_runnable_calls.append({"input": input, "config": config})
        return str(runnable.invoke(input, config=config))


def _is_react_selector_prompt(prompt_template: Any | None) -> bool:
    template = str(getattr(prompt_template, "template", "") or "")
    return "REACT_SELECTOR" in template


def _fake_react_selector_output(input: Any) -> str:
    if isinstance(input, dict):
        previous_turns = _loads_json_value(input.get("react_previous_turns_json"))
        if isinstance(previous_turns, list) and previous_turns:
            return json.dumps(
                {
                    "action_type": "final_answer",
                    "rationale_summary": "已有工具观察，进入最终汇总。",
                },
                ensure_ascii=False,
            )
        policy = _loads_json_value(input.get("react_scene_policy_json"))
        tool_name = "agentic_rag_search"
        if isinstance(policy, dict):
            preferred_tools = policy.get("preferred_tools")
            if isinstance(preferred_tools, list) and preferred_tools:
                tool_name = str(preferred_tools[0])
        return json.dumps(
            {
                "action_type": "tool_call",
                "tool_name": tool_name,
                "input": {},
                "rationale_summary": "首轮先调用允许的 RAG 工具。",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "action_type": "tool_call",
            "tool_name": "agentic_rag_search",
            "input": {},
            "rationale_summary": "首轮先调用允许的 RAG 工具。",
        },
        ensure_ascii=False,
    )


def _loads_json_value(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def test_runtime_graph_state_exposes_minimal_chat_execution_fields() -> None:
    assert set(RuntimeGraphState.__annotations__) == {
        "scene",
        "session_id",
        "request_id",
        "messages",
        "answer",
        "knowledge_used",
        "citations",
        "retrieval_trace",
        "metadata",
        "answer_mode",
        "status",
        "run_id",
        "state_event",
        "final_state",
        "retry_attempt",
        "retry_metadata",
        "hitl",
        "hitl_resume",
        "agent_mode",
        "agent_mode_reason",
        "agent_mode_signals",
        "react_run",
        "plan_run",
        "current_turn_id",
        "current_step_id",
        "current_tool_call",
        "documents",
        "tool_event",
        "final_decision",
        "follow_up_question",
        "tool_observation",
    }

    state = build_runtime_graph_state(
        session_id="session-state",
        request_id="req-state",
        messages=[HumanMessage(content="hello")],
        answer="world",
        knowledge_used=True,
        citations=[{"document_id": "doc-1", "chunk_id": "chunk-1"}],
        retrieval_trace={"final_decision": "evidence_answer"},
        metadata={"scene": "generic_assistant"},
        scene="generic_assistant",
        answer_mode="evidence_answer",
        status="succeeded",
    )

    assert state["session_id"] == "session-state"
    assert state["request_id"] == "req-state"
    assert state["messages"][0].content == "hello"
    assert state["answer"] == "world"
    assert state["knowledge_used"] is True
    assert state["citations"] == [{"document_id": "doc-1", "chunk_id": "chunk-1"}]
    assert state["retrieval_trace"] == {"final_decision": "evidence_answer"}
    assert state["scene"] == "generic_assistant"
    assert state["answer_mode"] == "evidence_answer"
    assert state["metadata"] == {"scene": "generic_assistant"}
    assert state["status"] == "succeeded"
    assert state["final_state"] is None
    assert state["retry_attempt"] == 0
    assert state["hitl"] is None
    assert state["hitl_resume"] is None
    assert state["agent_mode"] is None
    assert state["react_run"] is None
    assert state["plan_run"] is None
    assert state["current_turn_id"] is None
    assert state["current_step_id"] is None
    assert state["current_tool_call"] is None
    assert state["documents"] == []
    assert state["tool_event"] is None
    assert state["final_decision"] is None
    assert state["follow_up_question"] is None
    assert state["tool_observation"] is None


def test_runtime_graph_state_defaults_keep_non_evidence_branches_representable() -> None:
    state = build_runtime_graph_state(
        session_id="session-no-evidence",
        request_id="req-no-evidence",
        answer="I need more information.",
        retrieval_trace={"final_decision": "ask_user"},
    )

    assert state == {
        "session_id": "session-no-evidence",
        "request_id": "req-no-evidence",
        "scene": None,
        "messages": [],
        "answer": "I need more information.",
        "knowledge_used": False,
        "citations": [],
        "retrieval_trace": {"final_decision": "ask_user"},
        "metadata": {},
        "answer_mode": None,
        "status": "running",
        "run_id": None,
        "state_event": None,
        "final_state": None,
        "retry_attempt": 0,
        "retry_metadata": {},
        "hitl": None,
        "hitl_resume": None,
        "agent_mode": None,
        "agent_mode_reason": None,
        "agent_mode_signals": None,
        "react_run": None,
        "plan_run": None,
        "current_turn_id": None,
        "current_step_id": None,
        "current_tool_call": None,
        "documents": [],
        "tool_event": None,
        "final_decision": None,
        "follow_up_question": None,
        "tool_observation": None,
    }


def test_runtime_hitl_state_preserves_serializable_protocol_fields() -> None:
    hitl = build_runtime_hitl_state(
        interrupt_id="interrupt-1",
        thread_id="session-hitl",
        reason="需要用户补充文档主题。",
        pending_action="clarification",
        allowed_actions=["respond", "reject"],
        suggested_responses=[
            {
                "suggestion_id": "topic_scope",
                "label": "限定文档主题",
                "value": "我想查询安全合规政策相关内容。",
            }
        ],
        allow_freeform_response=True,
    )
    state = build_runtime_graph_state(
        session_id="session-hitl",
        request_id="req-hitl",
        status="waiting_user",
        hitl=hitl,
    )

    assert state["status"] == "waiting_user"
    assert set(state["hitl"]) == {
        "interrupt_id",
        "thread_id",
        "reason",
        "pending_action",
        "proposed_tool_call",
        "allowed_actions",
        "suggested_responses",
        "allow_freeform_response",
        "resume_payload",
        "metadata",
    }
    assert state["hitl"] == {
        "interrupt_id": "interrupt-1",
        "thread_id": "session-hitl",
        "reason": "需要用户补充文档主题。",
        "pending_action": "clarification",
        "proposed_tool_call": None,
        "allowed_actions": ["respond", "reject"],
        "suggested_responses": [
            {
                "suggestion_id": "topic_scope",
                "label": "限定文档主题",
                "value": "我想查询安全合规政策相关内容。",
            }
        ],
        "allow_freeform_response": True,
        "resume_payload": None,
        "metadata": {},
    }


def test_chat_response_accepts_optional_hitl_waiting_payload() -> None:
    hitl = HitlState(
        interrupt_id="interrupt-api",
        thread_id="session-api",
        reason="需要用户补充查询范围。",
        pending_action="clarification",
        allowed_actions=["respond", "reject"],
        suggested_responses=[
            {
                "suggestion_id": "term_scope",
                "label": "限定术语",
                "value": "请围绕权限审批流程继续检索。",
            }
        ],
        allow_freeform_response=True,
    )

    response = ChatResponse(
        session_id="session-api",
        request_id="req-api",
        answer="",
        knowledge_used=False,
        scene="generic_assistant",
        status="waiting_user",
        hitl=hitl,
    )

    payload = response.model_dump()
    assert payload["status"] == "waiting_user"
    assert payload["hitl"]["pending_action"] == "clarification"
    assert payload["hitl"]["suggested_responses"][0]["suggestion_id"] == "term_scope"
    assert payload["hitl"]["allow_freeform_response"] is True


def test_workflow_state_machine_accepts_core_success_hitl_and_retry_paths() -> None:
    assert validate_transition("created", "plan_start") == "planning"
    assert validate_transition("planning", "run_start") == "running"
    assert validate_transition("running", "success") == "succeeded"
    assert validate_transition("created", "run_start") == "running"
    assert validate_transition("running", "interrupt") == "waiting_user"
    assert validate_transition("waiting_user", "resume_approve") == "running"
    assert validate_transition("waiting_user", "resume_respond") == "running"
    assert validate_transition("waiting_user", "resume_reject") == "cancelled"
    assert validate_transition("running", "tool_error_retryable") == "retrying"
    assert validate_transition("retrying", "retry") == "running"
    assert validate_transition("retrying", "tool_error_final") == "failed"
    assert is_terminal("succeeded") is True
    assert is_terminal("failed") is True
    assert is_terminal("cancelled") is True


def test_workflow_state_machine_rejects_unknown_state_and_terminal_resume() -> None:
    try:
        validate_transition("unknown", "run_start")
    except UnknownWorkflowStateError as exc:
        assert "Unknown workflow state" in str(exc)
    else:
        raise AssertionError("unknown workflow state should be rejected")

    for state in ("succeeded", "failed", "cancelled"):
        try:
            validate_transition(state, "resume_approve")
        except InvalidWorkflowTransitionError as exc:
            assert exc.current_state == state
            assert exc.event == "resume_approve"
        else:
            raise AssertionError(f"{state} should reject resume")


def test_runtime_graph_config_binds_thread_id_to_session_id_and_metadata_to_request() -> None:
    config = build_runtime_graph_config(
        session_id="session-config",
        request_id="req-config",
        metadata={
            "request_id": "stale-request",
            "session_id": "stale-session",
            "scene": "generic_assistant",
        },
    )

    assert config["configurable"]["thread_id"] == "session-config"
    assert config["configurable"]["checkpoint_ns"] == DEFAULT_RUNTIME_CHECKPOINT_NS
    assert config["metadata"] == {
        "request_id": "req-config",
        "session_id": "session-config",
        "scene": "generic_assistant",
    }


def test_graph_run_lifecycle_records_success_path_with_thread_and_request() -> None:
    recorder = GraphRunLifecycleRecorder()

    run = recorder.create_run(
        thread_id="session-success",
        request_id="req-success",
        metadata={"scene": "generic_assistant"},
        run_id="run-success",
    )
    recorder.mark_running(run)
    recorder.mark_succeeded(run)

    events = recorder.events(run.run_id)
    assert tuple(event.status for event in events) == ("created", "running", "succeeded")
    assert all(event.thread_id == "session-success" for event in events)
    assert all(event.request_id == "req-success" for event in events)
    assert all(event.metadata == {"scene": "generic_assistant"} for event in events)
    assert recorder.events_for_thread(
        thread_id="session-success",
        request_id="req-success",
    ) == events
    assert recorder.latest(run) is events[-1]


def test_graph_run_lifecycle_records_failed_path_with_thread_request_and_error() -> None:
    recorder = GraphRunLifecycleRecorder()

    run = recorder.create_run(
        thread_id="session-failed",
        request_id="req-failed",
        metadata={"scene": "generic_assistant"},
        run_id="run-failed",
    )
    recorder.mark_running(run)
    failed = recorder.mark_failed(run, RuntimeError("model invocation failed"))

    events = recorder.events(run.run_id)
    assert tuple(event.status for event in events) == ("created", "running", "failed")
    assert all(event.thread_id == "session-failed" for event in events)
    assert all(event.request_id == "req-failed" for event in events)
    assert failed.error == "RuntimeError: model invocation failed"
    assert recorder.latest(run) == failed


def test_graph_run_lifecycle_records_waiting_cancelled_and_retrying_paths() -> None:
    recorder = GraphRunLifecycleRecorder()
    waiting = recorder.create_run(
        thread_id="session-waiting",
        request_id="req-waiting",
        run_id="run-waiting",
    )
    recorder.mark_running(waiting)
    recorder.mark_waiting_user(waiting)
    assert recorder.statuses(waiting) == ("created", "running", "waiting_user")

    cancelled = recorder.create_run(
        thread_id="session-cancelled",
        request_id="req-cancelled",
        run_id="run-cancelled",
    )
    recorder.mark_running(cancelled)
    recorder.mark_cancelled(cancelled)
    assert recorder.statuses(cancelled) == ("created", "running", "cancelled")

    retrying = recorder.create_run(
        thread_id="session-retry",
        request_id="req-retry",
        run_id="run-retry",
    )
    recorder.mark_running(retrying)
    recorder.mark_retrying(retrying)
    recorder.mark_running(retrying)
    recorder.mark_failed(retrying, "retry exhausted")
    assert recorder.statuses(retrying) == (
        "created",
        "running",
        "retrying",
        "running",
        "failed",
    )


def test_chat_graph_runtime_invokes_answer_graph_and_persists_checkpoint_metadata() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-checkpoint")
    prepared = _prepared_graph_turn(
        session_id="session-graph",
        request_id="req-graph",
        user_message="新问题",
    )

    result = runtime.invoke(
        prepared=prepared,
        answer_builder=lambda turn: ("graph answer", turn.citations),
        history_loader=lambda _: [
            HumanMessage(content="旧问题"),
            AIMessage(content="旧答案"),
        ],
    )
    restored = runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": "session-graph",
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )

    assert result.config["configurable"]["thread_id"] == "session-graph"
    assert result.config["metadata"]["request_id"] == "req-graph"
    assert result.state["answer"] == "graph answer"
    assert [message.content for message in result.state["messages"]] == [
        "旧问题",
        "旧答案",
        "新问题",
        "graph answer",
    ]
    assert restored is not None
    assert restored.metadata["request_id"] == "req-graph"
    assert restored.metadata["session_id"] == "session-graph"
    assert restored.checkpoint["channel_values"]["answer"] == "graph answer"
    assert restored.checkpoint["channel_values"]["status"] == "succeeded"
    assert restored.checkpoint["channel_values"]["final_state"] == "succeeded"
    checkpoint_values = restored.checkpoint["channel_values"]
    assert checkpoint_values["agent_mode"] == "react"
    assert checkpoint_values["react_run"]["workflow_status"] == "succeeded"
    assert checkpoint_values["react_run"]["turns"][0]["status"] == "succeeded"
    assert checkpoint_values["react_run"]["turns"][0]["observation"]["trace"]["retrieval_trace"][
        "final_decision"
    ] == "answer_with_evidence"
    assert checkpoint_values["current_turn_id"] is None
    assert checkpoint_values["current_tool_call"] is None


def test_chat_graph_runtime_persists_plan_run_for_plan_mode() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-plan-checkpoint")
    prepared = _PreparedGraphTurn(
        session_id="session-plan",
        request_id="req-plan",
        user_message="请分步骤规划并汇总 graph runtime",
        answer_mode="evidence_answer",
        final_decision="answer_with_evidence",
        knowledge_used=True,
        citations=[_citation()],
        retrieval_trace=_retrieval_trace(citations=[_citation()]),
        agent_mode="plan",
    )

    result = runtime.invoke(
        prepared=prepared,
        answer_builder=lambda turn: ("plan graph answer", turn.citations),
        history_loader=lambda _: [],
    )

    assert result.state["agent_mode"] == "plan"
    assert result.state["plan_run"]["workflow_status"] == "succeeded"
    assert result.state["plan_run"]["steps"][0]["status"] == "succeeded"
    assert result.state["plan_run"]["steps"][0]["tool_name"] == "native_rag_search"
    assert result.state["react_run"] is None


def test_chat_graph_runtime_tolerates_legacy_checkpoint_without_agent_fields() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-legacy-agent-fields")
    config = build_runtime_graph_config(
        session_id="session-legacy-agent",
        request_id="req-legacy-agent",
    )
    state = build_runtime_graph_state(
        session_id="session-legacy-agent",
        request_id="req-legacy-agent",
        answer="legacy answer",
    )
    # 模拟旧 checkpoint：channel values 不包含 orchestration 字段。
    legacy_values = {
        key: value
        for key, value in state.items()
        if key
        not in {
            "agent_mode",
            "react_run",
            "plan_run",
            "current_turn_id",
            "current_step_id",
            "current_tool_call",
        }
    }
    runtime._persist_state_update(  # noqa: SLF001 - 测试旧 checkpoint 读取边界。
        state=state,
        config=config,
        update=legacy_values,
    )

    loaded = runtime._load_or_build_thread_state(  # noqa: SLF001 - 验证容忍旧字段缺失。
        session_id="session-legacy-agent",
        request_id="req-legacy-agent",
        config=config,
        require_checkpoint=True,
    )

    assert loaded["answer"] == "legacy answer"
    assert loaded["agent_mode"] is None
    assert loaded["react_run"] is None
    assert loaded["plan_run"] is None


def test_chat_graph_runtime_reload_preserves_extended_agent_state_fields() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-extended-agent-state")
    config = build_runtime_graph_config(
        session_id="session-extended-agent-state",
        request_id="req-extended-agent-state",
    )
    document = Document(
        page_content="Graph state 文档内容。",
        metadata={"citation_id": "chunk-extended"},
    )
    state = build_runtime_graph_state(
        session_id="session-extended-agent-state",
        request_id="req-extended-agent-state",
        documents=[document],
        tool_event={"stage": "tool_done", "documents": 1},
        final_decision="ask_user",
        follow_up_question="请补充查询范围。",
        tool_observation={
            "tool_name": "native_rag_search",
            "success": False,
        },
    )
    runtime._persist_state_update(  # noqa: SLF001 - 验证 checkpoint 字段 round-trip。
        state=state,
        config=config,
        update=state,
    )

    loaded = runtime._load_or_build_thread_state(  # noqa: SLF001 - 验证 checkpoint 回读字段。
        session_id="session-extended-agent-state",
        request_id="req-extended-agent-state",
        config=config,
        require_checkpoint=True,
    )

    assert loaded["documents"][0].page_content == "Graph state 文档内容。"
    assert loaded["tool_event"] == {"stage": "tool_done", "documents": 1}
    assert loaded["final_decision"] == "ask_user"
    assert loaded["follow_up_question"] == "请补充查询范围。"
    assert loaded["tool_observation"] == {
        "tool_name": "native_rag_search",
        "success": False,
    }


def test_chat_graph_runtime_seeds_legacy_history_only_without_checkpoint() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-history-seed")
    loader_calls = 0

    def history_loader(_: _PreparedGraphTurn) -> list[BaseMessage]:
        nonlocal loader_calls
        loader_calls += 1
        return [HumanMessage(content="legacy question")]

    first = _prepared_graph_turn(
        session_id="session-seed",
        request_id="req-seed-1",
        user_message="first graph question",
    )
    second = _prepared_graph_turn(
        session_id="session-seed",
        request_id="req-seed-2",
        user_message="second graph question",
    )

    runtime.invoke(
        prepared=first,
        answer_builder=lambda turn: ("first answer", turn.citations),
        history_loader=history_loader,
    )
    result = runtime.invoke(
        prepared=second,
        answer_builder=lambda turn: ("second answer", turn.citations),
        history_loader=history_loader,
    )

    assert loader_calls == 1
    assert [message.content for message in result.state["messages"]] == [
        "legacy question",
        "first graph question",
        "first answer",
        "second graph question",
        "second answer",
    ]


def test_chat_graph_runtime_creates_hitl_wait_checkpoint() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-wait")

    result = runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-hitl-wait",
            request_id="req-hitl-wait",
            reason="需要人工补充查询范围。",
            pending_action="clarification",
            allowed_actions=["respond", "reject"],
            suggested_responses=[
                {
                    "suggestion_id": "topic_scope",
                    "label": "限定主题",
                    "value": "请查询安全合规政策。",
                }
            ],
            allow_freeform_response=True,
            metadata={"scene": "generic_assistant"},
        ),
        interrupt_id="interrupt-wait",
    )
    restored = runtime.checkpointer.get_tuple(result.config)

    assert result.state["status"] == "waiting_user"
    assert result.state["state_event"] == "interrupt"
    assert result.state["hitl"]["interrupt_id"] == "interrupt-wait"
    assert restored is not None
    assert restored.checkpoint["channel_values"]["status"] == "waiting_user"
    assert restored.checkpoint["channel_values"]["hitl"]["allowed_actions"] == [
        "respond",
        "reject",
    ]
    assert runtime.lifecycle.events(result.run_id)[-1].status == "waiting_user"


def test_chat_graph_runtime_creates_plan_hitl_wait_checkpoint() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-plan-hitl-wait")

    result = runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-plan-hitl",
            request_id="req-plan-hitl-wait",
            reason="计划步骤需要人工审批。",
            pending_action="tool_approval",
            proposed_tool_call={
                "tool_name": "native_rag_search",
                "args": {"query": "审批后继续"},
            },
            allowed_actions=["approve", "reject"],
            metadata={
                "mode": "plan",
                "plan_run_id": "plan-wait",
                "current_step_id": "step-approval",
                "user_goal": "审批后继续",
            },
        ),
        interrupt_id="interrupt-plan-hitl",
    )

    assert result.state["agent_mode"] == "plan"
    assert result.state["current_step_id"] == "step-approval"
    assert result.state["current_tool_call"]["tool_name"] == "native_rag_search"
    assert result.state["hitl"]["metadata"]["mode"] == "plan"
    assert result.state["plan_run"]["workflow_status"] == "waiting_user"
    assert result.state["plan_run"]["steps"][0]["status"] == "waiting_user"


def test_chat_graph_runtime_plan_reject_cancels_waiting_step() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-plan-hitl-reject")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-plan-reject",
            request_id="req-plan-reject-wait",
            reason="计划步骤需要人工审批。",
            pending_action="tool_approval",
            proposed_tool_call={
                "tool_name": "native_rag_search",
                "args": {"query": "拒绝后不执行"},
            },
            allowed_actions=["approve", "reject"],
            metadata={
                "mode": "plan",
                "plan_run_id": "plan-reject",
                "current_step_id": "step-reject",
                "user_goal": "拒绝后不执行",
            },
        ),
        interrupt_id="interrupt-plan-reject",
    )
    executed = False

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-plan-reject",
            request_id="req-plan-reject-resume",
            interrupt_id="interrupt-plan-reject",
            action="reject",
        ),
        approve_executor=lambda _: {"executed": executed},
    )

    assert executed is False
    assert result.state["status"] == "cancelled"
    assert result.state["plan_run"]["workflow_status"] == "cancelled"
    assert result.state["plan_run"]["steps"][0]["status"] == "cancelled"
    assert result.state["current_step_id"] is None
    assert result.state["current_tool_call"] is None


def test_chat_graph_runtime_creates_react_clarification_wait_metadata() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-react-hitl-wait")

    result = runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-react-hitl",
            request_id="req-react-hitl",
            reason="需要补充文档主题。",
            pending_action="clarification",
            allowed_actions=["respond", "reject"],
            suggested_responses=[{"suggestion_id": "topic", "label": "主题", "value": "安全"}],
            allow_freeform_response=True,
            metadata={
                "mode": "react",
                "react_run_id": "react-req-react-hitl",
                "current_turn_id": "turn-1",
                "user_goal": "查询制度",
            },
        ),
        interrupt_id="interrupt-react-hitl",
    )

    assert result.state["agent_mode"] == "react"
    assert result.state["current_turn_id"] == "turn-1"
    assert result.state["hitl"]["metadata"]["mode"] == "react"
    assert result.state["hitl"]["metadata"]["react_run_id"] == "react-req-react-hitl"
    assert result.state["react_run"]["react_run_id"] == "react-req-react-hitl"
    assert result.state["react_run"]["turns"][0]["status"] == "waiting_user"


def test_chat_graph_runtime_creates_plan_approval_wait_metadata_and_approve_resumes_step() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-plan-hitl-approve")

    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-plan-hitl",
            request_id="req-plan-hitl",
            reason="计划步骤需要审批。",
            pending_action="tool_approval",
            proposed_tool_call={"tool_name": "generic_write", "args": {"query": "publish"}},
            allowed_actions=["approve", "reject"],
            metadata={
                "mode": "plan",
                "plan_run_id": "plan-req-plan-hitl",
                "current_step_id": "step-1",
                "user_goal": "发布文档",
            },
        ),
        interrupt_id="interrupt-plan-hitl",
    )

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-plan-hitl",
            request_id="req-plan-hitl-resume",
            interrupt_id="interrupt-plan-hitl",
            action="approve",
        ),
        approve_executor=lambda tool_call: {"executed_tool": tool_call["tool_name"]},
    )

    assert result.state["status"] == "succeeded"
    assert result.state["plan_run"]["workflow_status"] == "succeeded"
    assert result.state["plan_run"]["steps"][0]["status"] == "succeeded"
    assert result.state["plan_run"]["final_answer"] == "已批准并执行待审批操作。"
    assert result.state["plan_run"]["result_summary"] == "已批准并执行待审批操作。"
    assert result.state["hitl_resume"]["metadata"]["mode"] == "plan"
    assert result.tool_result == {"executed_tool": "generic_write"}


def test_chat_graph_runtime_plan_respond_settles_nested_run_result() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-plan-hitl-respond")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-plan-respond",
            request_id="req-plan-respond-wait",
            reason="需要补充执行范围。",
            pending_action="clarification",
            allowed_actions=["respond", "reject"],
            suggested_responses=[{"suggestion_id": "scope", "label": "范围", "value": "补充范围"}],
            allow_freeform_response=True,
            metadata={
                "mode": "plan",
                "plan_run_id": "plan-respond",
                "current_step_id": "step-1",
                "user_goal": "执行计划",
            },
        ),
        interrupt_id="interrupt-plan-respond",
    )

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-plan-respond",
            request_id="req-plan-respond-resume",
            interrupt_id="interrupt-plan-respond",
            action="respond",
            payload={"response": "补充范围", "source": "freeform"},
        ),
        respond_handler=lambda payload, state: {
            "status": "succeeded",
            "answer": f"继续执行：{payload['response']}",
        },
    )

    assert result.state["status"] == "succeeded"
    assert result.state["plan_run"]["workflow_status"] == "succeeded"
    assert result.state["plan_run"]["steps"][0]["status"] == "succeeded"
    assert result.state["plan_run"]["final_answer"] == "继续执行：补充范围"
    assert result.state["plan_run"]["result_summary"] == "继续执行：补充范围"


def test_chat_graph_runtime_react_respond_settles_nested_run_result() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-react-hitl-respond")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-react-respond",
            request_id="req-react-respond-wait",
            reason="需要补充文档主题。",
            pending_action="clarification",
            allowed_actions=["respond", "reject"],
            suggested_responses=[{"suggestion_id": "topic", "label": "主题", "value": "安全"}],
            allow_freeform_response=True,
            metadata={
                "mode": "react",
                "react_run_id": "react-respond",
                "current_turn_id": "turn-1",
                "user_goal": "查询制度",
            },
        ),
        interrupt_id="interrupt-react-respond",
    )

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-react-respond",
            request_id="req-react-respond-resume",
            interrupt_id="interrupt-react-respond",
            action="respond",
            payload={"response": "安全制度", "source": "freeform"},
        ),
        respond_handler=lambda payload, state: {
            "status": "succeeded",
            "answer": f"继续检索：{payload['response']}",
        },
    )

    assert result.state["status"] == "succeeded"
    assert result.state["react_run"]["workflow_status"] == "succeeded"
    assert result.state["react_run"]["turns"][0]["status"] == "succeeded"
    assert result.state["react_run"]["final_answer"] == "继续检索：安全制度"
    assert result.state["react_run"]["result_summary"] == "继续检索：安全制度"


def test_chat_graph_runtime_rejects_stale_agent_runtime_hitl_identity() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-stale-agent-hitl")

    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-stale-agent-hitl",
            request_id="req-stale-agent-hitl",
            reason="需要补充文档主题。",
            pending_action="clarification",
            allowed_actions=["respond", "reject"],
            allow_freeform_response=True,
            metadata={
                "mode": "react",
                "react_run_id": "react-expected",
                "current_turn_id": "turn-1",
            },
        ),
        interrupt_id="interrupt-stale-agent-hitl",
    )
    config = build_runtime_graph_config(
        session_id="session-stale-agent-hitl",
        request_id="req-stale-agent-hitl",
    )
    state = runtime._load_or_build_thread_state(  # noqa: SLF001 - stale checkpoint setup.
        session_id="session-stale-agent-hitl",
        request_id="req-stale-agent-hitl",
        config=config,
        require_checkpoint=True,
    )
    runtime._persist_state_update(  # noqa: SLF001 - stale checkpoint setup.
        state=state,
        config=config,
        update={"current_turn_id": "turn-stale"},
    )

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-stale-agent-hitl",
                request_id="req-stale-agent-hitl-resume",
                interrupt_id="interrupt-stale-agent-hitl",
                action="respond",
                payload={"response": "安全", "source": "freeform"},
            ),
            respond_handler=lambda payload, state: {"status": "succeeded", "answer": "should not run"},
        )
    except HitlResumeError as exc:
        assert "current_turn_id" in str(exc)
    else:
        raise AssertionError("stale ReAct turn id should reject resume")


def test_chat_graph_runtime_rejects_invalid_hitl_wait_protocol() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-invalid-wait")

    try:
        runtime.create_hitl_wait(
            wait=HitlWaitInput(
                session_id="session-hitl-invalid-wait",
                request_id="req-hitl-invalid-wait",
                reason="审批等待态不能返回澄清选项。",
                pending_action="tool_approval",
                proposed_tool_call={"tool_name": "generic_write_tool"},
                allowed_actions=["approve", "respond"],
            ),
            interrupt_id="interrupt-invalid-wait",
        )
    except HitlResumeError as exc:
        assert "approval HITL wait cannot include respond" in str(exc)
    else:
        raise AssertionError("approval HITL wait should reject respond action")

    try:
        runtime.create_hitl_wait(
            wait=HitlWaitInput(
                session_id="session-hitl-invalid-wait",
                request_id="req-hitl-invalid-wait-2",
                reason="澄清等待态必须允许 respond。",
                pending_action="clarification",
                allowed_actions=["reject"],
            ),
            interrupt_id="interrupt-invalid-wait-2",
        )
    except HitlResumeError as exc:
        assert "requires respond action" in str(exc)
    else:
        raise AssertionError("clarification HITL wait should require respond action")

    statuses = tuple(
        event.status
        for event in runtime.lifecycle.events_for_thread(
            thread_id="session-hitl-invalid-wait"
        )
    )
    assert statuses == ("created", "running", "failed", "created", "running", "failed")


def test_chat_graph_runtime_resume_approve_executes_proposed_tool_once() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-approve")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-hitl-approve",
            request_id="req-hitl-approve-wait",
            reason="外部 API 调用需要审批。",
            pending_action="external_api_approval",
            proposed_tool_call={"tool_name": "generic_external_webhook_call"},
            allowed_actions=["approve", "reject"],
        ),
        interrupt_id="interrupt-approve",
    )
    executed_calls: list[dict[str, Any]] = []

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-hitl-approve",
            request_id="req-hitl-approve-resume",
            interrupt_id="interrupt-approve",
            action="approve",
            payload={"reason": "approved"},
        ),
        approve_executor=lambda tool_call: executed_calls.append(dict(tool_call))
        or {"executed": True},
    )

    assert executed_calls == [{"tool_name": "generic_external_webhook_call"}]
    assert result.tool_result == {"executed": True}
    assert result.config["configurable"]["thread_id"] == "session-hitl-approve"
    assert result.state["status"] == "succeeded"
    assert result.state["final_state"] == "succeeded"
    assert result.state["hitl"] is None
    assert result.state["hitl_resume"]["action"] == "approve"
    assert result.state["hitl_resume"]["interrupt_id"] == "interrupt-approve"
    assert result.state["metadata"]["hitl_resume"]["action"] == "approve"

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-hitl-approve",
                request_id="req-hitl-approve-duplicate",
                interrupt_id="interrupt-approve",
                action="approve",
                payload={"reason": "duplicate approve"},
            ),
            approve_executor=lambda tool_call: executed_calls.append(dict(tool_call))
            or {"executed": True},
        )
    except HitlResumeError as exc:
        assert "terminal" in str(exc)
    else:
        raise AssertionError("duplicate resume should be rejected after HITL is cleared")
    assert executed_calls == [{"tool_name": "generic_external_webhook_call"}]


def test_chat_graph_runtime_resume_consumes_wait_before_tool_side_effect() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-approve-checkpoint-failure")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-hitl-side-effect",
            request_id="req-hitl-side-effect-wait",
            reason="外部 API 调用需要审批。",
            pending_action="external_api_approval",
            proposed_tool_call={"tool_name": "generic_external_webhook_call"},
            allowed_actions=["approve", "reject"],
        ),
        interrupt_id="interrupt-side-effect",
    )
    original_persist_state_update = runtime._persist_state_update
    persist_calls = 0

    def flaky_persist_state_update(**kwargs: Any) -> RuntimeGraphState:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise RuntimeError("final checkpoint failed")
        return original_persist_state_update(**kwargs)

    runtime._persist_state_update = flaky_persist_state_update  # type: ignore[method-assign]
    executed_calls: list[dict[str, Any]] = []

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-hitl-side-effect",
                request_id="req-hitl-side-effect-resume",
                interrupt_id="interrupt-side-effect",
                action="approve",
            ),
            approve_executor=lambda tool_call: executed_calls.append(dict(tool_call))
            or {"executed": True},
        )
    except RuntimeError as exc:
        assert "final checkpoint failed" in str(exc)
    else:
        raise AssertionError("final checkpoint failure should be surfaced")

    restored = runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": "session-hitl-side-effect",
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )
    assert restored is not None
    assert restored.checkpoint["channel_values"]["status"] == "failed"
    assert restored.checkpoint["channel_values"]["hitl"] is None
    assert executed_calls == [{"tool_name": "generic_external_webhook_call"}]

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-hitl-side-effect",
                request_id="req-hitl-side-effect-duplicate",
                interrupt_id="interrupt-side-effect",
                action="approve",
            ),
            approve_executor=lambda tool_call: executed_calls.append(dict(tool_call)),
        )
    except HitlResumeError as exc:
        assert "terminal" in str(exc)
    else:
        raise AssertionError("failed terminal checkpoint should reject duplicate resume")
    assert executed_calls == [{"tool_name": "generic_external_webhook_call"}]


def test_chat_graph_runtime_resume_reject_skips_proposed_tool() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-reject")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-hitl-reject",
            request_id="req-hitl-reject-wait",
            reason="写操作需要审批。",
            pending_action="tool_approval",
            proposed_tool_call={"tool_name": "generic_knowledge_document_publish"},
            allowed_actions=["approve", "reject"],
        ),
        interrupt_id="interrupt-reject",
    )

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-hitl-reject",
            request_id="req-hitl-reject-resume",
            interrupt_id="interrupt-reject",
            action="reject",
            payload={"reason": "not allowed"},
        ),
        approve_executor=lambda _: {"should_not_execute": True},
    )

    assert result.tool_result is None
    assert result.state["status"] == "cancelled"
    assert result.state["final_state"] == "cancelled"
    assert result.state["hitl"] is None
    assert "未执行" in result.state["answer"]
    assert result.state["hitl_resume"]["action"] == "reject"
    assert result.state["metadata"]["hitl_resume"]["action"] == "reject"


def test_chat_graph_runtime_resume_respond_records_source_and_suggestion() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-respond")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-hitl-respond",
            request_id="req-hitl-respond-wait",
            reason="需要用户补充文档主题。",
            pending_action="clarification",
            allowed_actions=["respond", "reject"],
            suggested_responses=[
                {
                    "suggestion_id": "topic_scope",
                    "label": "限定主题",
                    "value": "安全合规政策",
                }
            ],
            allow_freeform_response=True,
        ),
        interrupt_id="interrupt-respond",
    )

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-hitl-respond",
            request_id="req-hitl-respond-resume",
            interrupt_id="interrupt-respond",
            action="respond",
            payload={
                "response": "安全合规政策",
                "source": "suggested_response",
                "suggestion_id": "topic_scope",
            },
        ),
        respond_handler=lambda payload, state: {
            "status": "succeeded",
            "answer": f"继续检索：{payload['response']}",
        },
    )

    assert result.state["status"] == "succeeded"
    assert result.state["final_state"] == "succeeded"
    assert result.state["hitl"] is None
    assert result.state["answer"] == "继续检索：安全合规政策"
    assert result.state["hitl_resume"]["source"] == "suggested_response"
    assert result.state["hitl_resume"]["suggestion_id"] == "topic_scope"
    assert result.state["metadata"]["hitl_resume"]["source"] == "suggested_response"
    assert result.state["metadata"]["hitl_resume"]["suggestion_id"] == "topic_scope"


def test_chat_graph_runtime_resume_respond_requires_handler_and_valid_payload() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-respond-validation")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-hitl-respond-validation",
            request_id="req-hitl-respond-validation-wait",
            reason="需要用户补充文档主题。",
            pending_action="clarification",
            allowed_actions=["respond", "reject"],
            suggested_responses=[
                {
                    "suggestion_id": "topic_scope",
                    "label": "限定主题",
                    "value": "安全合规政策",
                }
            ],
            allow_freeform_response=False,
        ),
        interrupt_id="interrupt-respond-validation",
    )

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-hitl-respond-validation",
                request_id="req-hitl-respond-validation-no-handler",
                interrupt_id="interrupt-respond-validation",
                action="respond",
                payload={"source": "suggested_response", "suggestion_id": "topic_scope"},
            )
        )
    except HitlResumeError as exc:
        assert "respond_handler" in str(exc)
    else:
        raise AssertionError("respond should require a handler")

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-hitl-respond-validation",
                request_id="req-hitl-respond-validation-freeform",
                interrupt_id="interrupt-respond-validation",
                action="respond",
                payload={"response": "自由输入", "source": "freeform"},
            ),
            respond_handler=lambda payload, state: {"status": "succeeded", "answer": payload["response"]},
        )
    except HitlResumeError as exc:
        assert "freeform response is not allowed" in str(exc)
    else:
        raise AssertionError("freeform response should be rejected when disabled")

    result = runtime.resume_hitl(
        resume=HitlResumeInput(
            session_id="session-hitl-respond-validation",
            request_id="req-hitl-respond-validation-suggested",
            interrupt_id="interrupt-respond-validation",
            action="respond",
            payload={"source": "suggested_response", "suggestion_id": "topic_scope"},
        ),
        respond_handler=lambda payload, state: {
            "status": "succeeded",
            "answer": f"继续检索：{payload['response']}",
        },
    )

    assert result.state["answer"] == "继续检索：安全合规政策"
    assert result.state["hitl_resume"]["response"] == "安全合规政策"


def test_chat_graph_runtime_resume_rejects_invalid_interrupt_or_edit() -> None:
    runtime = _build_chat_graph_runtime("runtime-graph-hitl-invalid")
    runtime.create_hitl_wait(
        wait=HitlWaitInput(
            session_id="session-hitl-invalid",
            request_id="req-hitl-invalid-wait",
            reason="需要人工处理。",
            pending_action="clarification",
            allowed_actions=["respond", "edit"],
        ),
        interrupt_id="interrupt-valid",
    )

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-hitl-invalid",
                request_id="req-hitl-invalid-resume-1",
                interrupt_id="interrupt-stale",
                action="respond",
            )
        )
    except HitlResumeError as exc:
        assert "interrupt_id" in str(exc)
    else:
        raise AssertionError("stale interrupt_id should be rejected")

    try:
        runtime.resume_hitl(
            resume=HitlResumeInput(
                session_id="session-hitl-invalid",
                request_id="req-hitl-invalid-resume-2",
                interrupt_id="interrupt-valid",
                action="edit",
            )
        )
    except HitlResumeError as exc:
        assert "edit action is not supported" in str(exc)
    else:
        raise AssertionError("edit should be rejected before implementation")


def test_chat_service_non_streaming_answer_runs_through_graph_runtime() -> None:
    runtime_dir = make_test_runtime_dir("runtime-graph-chat-service")
    sqlite_path = runtime_dir / "sessions.db"
    graph_runtime = _build_chat_graph_runtime("runtime-graph-chat-service")
    service = ChatService(
        scene_definition=_build_scene_definition(
            [
                Document(
                    page_content="Graph runtime 文档证据。",
                    metadata={
                        "document_id": "doc-graph",
                        "source_path": "graph.md",
                        "namespace": "documents",
                        "chunk_id": "chunk-graph",
                        "chunk_index": 0,
                        "score": 0.91,
                    },
                )
            ]
        ),
        app_settings=AppSettings(
            data_dir=runtime_dir,
            session={"sqlite_path": sqlite_path, "window_size": 3},
        ),
        session_store=SQLiteSessionStore(sqlite_path=sqlite_path),
        context_builder=PromptContextBuilder(window_size=3),
        model=_FakeModel("这是 graph runtime 生成的回答。"),
        graph_runtime=graph_runtime,
    )

    response = service.chat(ChatRequest(message="请回答 graph runtime"))
    restored = graph_runtime.checkpointer.get_tuple(
        {
            "configurable": {
                "thread_id": response.session_id,
                "checkpoint_ns": DEFAULT_RUNTIME_CHECKPOINT_NS,
            }
        }
    )

    assert response.answer == "这是 graph runtime 生成的回答。\n\n参考来源：[1]"
    assert response.knowledge_used is True
    assert len(response.citations) == 1
    assert restored is not None
    assert restored.metadata["request_id"] == response.request_id
    assert restored.metadata["session_id"] == response.session_id
    assert restored.checkpoint["channel_values"]["answer"] == response.answer
    assert restored.checkpoint["channel_values"]["retrieval_trace"]["final_decision"] == (
        "answer_with_evidence"
    )
    assert service.session_store.count_turns(response.session_id) == 1
    assert service.session_store.count_messages(response.session_id) == 2
    messages, total_messages = service.session_store.get_session_messages(
        session_id=response.session_id,
        limit=10,
    )
    assert total_messages == 2
    assert [message.message_type for message in messages] == ["human", "ai"]
    assert messages[1].content == response.answer
    assert messages[1].citations[0]["citation_id"] == "chunk-graph"


def test_graph_stream_event_mapper_keeps_business_event_protocol() -> None:
    mapper = GraphStreamEventMapper()

    events = list(
        mapper.map_events(
            [
                GraphRuntimeStreamEvent("graph_run_created", {"request_id": "req-stream"}),
                GraphRuntimeStreamEvent("history_snapshot", {"messages": []}),
                GraphRuntimeStreamEvent(
                    "retrieval_tool_result",
                    {"retrieval_trace": {"final_decision": "answer_with_evidence"}},
                ),
                GraphRuntimeStreamEvent("answer_chunk", {"delta": "hello"}),
                GraphRuntimeStreamEvent(
                    "human_waiting",
                    {"hitl": {"interrupt_id": "interrupt-stream"}},
                ),
                GraphRuntimeStreamEvent(
                    "human_resume",
                    {"interrupt_id": "interrupt-stream", "action": "respond"},
                ),
                GraphRuntimeStreamEvent(
                    "graph_run_succeeded",
                    {"retrieval_trace": {"final_decision": "answer_with_evidence"}},
                ),
                GraphRuntimeStreamEvent(
                    "graph_run_failed",
                    {"code": "MODEL_INVOCATION_FAILED", "request_id": "req-stream"},
                ),
            ]
        )
    )

    assert [event.event for event in events] == [
        "start",
        "history",
        "tool",
        "chunk",
        "waiting_user",
        "resume",
        "done",
        "error",
    ]
    assert all(not event.event.startswith("graph_") for event in events)
    assert all("graph_run_created" not in event.data for event in events)


def _build_chat_graph_runtime(test_name: str) -> ChatGraphRuntime:
    runtime_dir = make_test_runtime_dir(test_name)
    return ChatGraphRuntime(
        checkpointer=SQLiteLangGraphCheckpointer(runtime_dir / "langgraph.db")
    )


def _prepared_graph_turn(
    *,
    session_id: str,
    request_id: str,
    user_message: str,
) -> _PreparedGraphTurn:
    citation = _citation()
    return _PreparedGraphTurn(
        session_id=session_id,
        request_id=request_id,
        user_message=user_message,
        answer_mode="evidence_answer",
        final_decision="answer_with_evidence",
        knowledge_used=True,
        citations=[citation],
        retrieval_trace=_retrieval_trace(citations=[citation]),
    )


def _citation() -> Citation:
    return Citation(
        index=1,
        citation_id="chunk-graph",
        namespace="documents",
        source_kind="document_chunk",
        source_name="graph.md",
        source_path="graph.md",
        document_id="doc-graph",
        chunk_id="chunk-graph",
        chunk_index=0,
        snippet="Graph runtime 文档证据。",
        score=0.91,
        vector_score=0.91,
        keyword_score=None,
        vector_rank=1,
        keyword_rank=None,
        rerank_score=None,
        matched_by=["vector"],
        rank=1,
    )


def _retrieval_trace(*, citations: list[Citation]) -> RetrievalTrace:
    return RetrievalTrace(
        original_query="原始问题",
        final_query="原始问题",
        tool_call_count=1,
        candidate_tools=["knowledge_document_search"],
        exit_reason="sufficient",
        final_decision="answer_with_evidence",
        success=True,
        raw_candidates_count=1,
        filtered_candidates_count=1,
        citations=citations,
        knowledge_used=True,
    )


def _build_scene_definition(documents: list[Document]) -> SceneDefinition:
    return SceneDefinition(
        scene="generic_assistant",
        name="Generic Assistant",
        description="Runtime graph test scene.",
        build_retriever=lambda: _SearchRetriever(documents),
        build_tools=lambda: (),
        candidate_retrieval_tools_resolver=lambda mounted: ("knowledge_document_search",),
        system_prompt="你是测试助手。",
        fallback_policy=SceneFallbackPolicy(no_hit_message="没有可用证据。"),
        infer_complexity=lambda message: "simple",
        retrieval_policy=SceneRetrievalPolicy(min_relevance_score=0.0),
    )
