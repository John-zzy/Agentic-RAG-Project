from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from backend.platform.agent_runtime.core.contracts import ToolObservation
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.factory import build_agent_middleware
from backend.platform.agent_runtime.react.factory import (
    ReActProviderFactory,
)
from backend.platform.agent_runtime.react.projection import (
    project_react_agent_output,
)
from backend.platform.agent_runtime.react.runtime import ReActRuntime
from backend.platform.agent_runtime.tooling.executor import ToolExecutor
from backend.platform.models.base.router import TaskComplexity
from backend.platform.tools.base import SceneTool, ToolResult


class _LookupArgs(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=1, ge=1)


class _SequencedToolChatModel(BaseChatModel):
    responses: list[AIMessage]
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "sequenced-tool-chat-model"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        del tool_choice, kwargs
        self.bound_tools = [str(getattr(tool, "name", tool)) for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        if not self.responses:
            raise AssertionError("No model response configured.")
        return ChatResult(
            generations=[ChatGeneration(message=self.responses.pop(0))]
        )


class _LookupTool(SceneTool):
    name = "lookup_policy"
    description = "Lookup policy records."
    capability_type = "retrieval"
    args_schema = _LookupArgs

    def __init__(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._records = records or [{"policy": "travel"}]
        self._citations = citations or [{"citation_id": "policy-1"}]
        self._metadata = metadata or {
            "trace": {
                "retrieval_trace": {
                    "final_decision": "answer_with_evidence",
                    "rounds": [{"tool_name": self.name, "result_count": 1}],
                }
            }
        }

    def invoke(self, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult.ok(
            tool_name=self.name,
            records=self._records,
            citations=self._citations,
            metadata=self._metadata,
        )


class _InventoryTool(SceneTool):
    name = "lookup_inventory"
    description = "Lookup inventory records."
    capability_type = "retrieval"
    args_schema = _LookupArgs

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult.ok(
            tool_name=self.name,
            records=[{"sku": "sku-1", "available": True}],
            citations=[{"citation_id": "inventory-1"}],
        )


def test_langchain_react_provider_direct_final_answer() -> None:
    model = _SequencedToolChatModel(responses=[AIMessage(content="直接回答。")])
    runtime = _build_runtime(model=model, tools={})

    run = runtime.run(
        session_id="session-direct",
        request_id="request-direct",
        user_goal="打个招呼",
        react_run_id="react-direct",
    )

    assert run.workflow_status == "succeeded"
    assert run.final_answer == "直接回答。"
    assert run.observations == []
    assert run.turns[-1].action.action_type == "final_answer"
    assert model.bound_tools == []


def test_langchain_react_provider_rag_tool_answer_preserves_observation() -> None:
    model = _SequencedToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_policy",
                        "args": {"query": "差旅制度"},
                        "id": "call-policy",
                    }
                ],
            ),
            AIMessage(content="根据制度，需要提交发票。"),
        ]
    )
    lookup = _LookupTool()
    runtime = _build_runtime(
        model=model,
        tools={"lookup_policy": lookup},
        allowed_tools={"lookup_policy"},
    )

    run = runtime.run(
        session_id="session-rag",
        request_id="request-rag",
        user_goal="差旅报销要求？",
        react_run_id="react-rag",
    )

    assert run.workflow_status == "succeeded"
    assert lookup.calls == [{"query": "差旅制度", "limit": 1}]
    assert run.observations[0].tool_name == "lookup_policy"
    assert run.observations[0].citations == [{"citation_id": "policy-1"}]
    assert run.observations[0].trace["retrieval_trace"]["final_decision"] == (
        "answer_with_evidence"
    )
    assert run.observations[0].tool_call_id == "call-policy"
    assert run.final_answer == "根据制度，需要提交发票。"


def test_langchain_react_provider_records_multi_turn_tool_progression() -> None:
    model = _SequencedToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_policy",
                        "args": {"query": "退货制度"},
                        "id": "call-policy",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_inventory",
                        "args": {"query": "sku-1"},
                        "id": "call-inventory",
                    }
                ],
            ),
            AIMessage(content="制度和库存均已确认。"),
        ]
    )
    policy = _LookupTool()
    inventory = _InventoryTool()
    runtime = _build_runtime(
        model=model,
        tools={"lookup_policy": policy, "lookup_inventory": inventory},
        allowed_tools={"lookup_policy", "lookup_inventory"},
        max_turns=4,
    )

    run = runtime.run(
        session_id="session-multi",
        request_id="request-multi",
        user_goal="检查退货制度和库存。",
        react_run_id="react-multi",
    )

    assert run.workflow_status == "succeeded"
    assert [turn.tool_name for turn in run.turns] == [
        "lookup_policy",
        "lookup_inventory",
        None,
    ]
    assert [observation.tool_name for observation in run.observations] == [
        "lookup_policy",
        "lookup_inventory",
    ]
    assert policy.calls == [{"query": "退货制度", "limit": 1}]
    assert inventory.calls == [{"query": "sku-1", "limit": 1}]


def test_langchain_react_provider_invalid_tool_is_recorded_without_invocation() -> None:
    model = _SequencedToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "blocked_tool",
                        "args": {"query": "外部系统"},
                        "id": "call-blocked",
                    }
                ],
            ),
            AIMessage(content="无法调用未授权工具。"),
        ]
    )
    lookup = _LookupTool()
    runtime = _build_runtime(
        model=model,
        tools={"lookup_policy": lookup},
        allowed_tools={"lookup_policy"},
    )

    run = runtime.run(
        session_id="session-invalid",
        request_id="request-invalid",
        user_goal="调用未授权工具。",
        react_run_id="react-invalid",
    )

    assert lookup.calls == []
    assert run.observations[0].success is False
    assert run.observations[0].tool_name == "blocked_tool"
    assert "not allowed" in (run.observations[0].error or "")
    assert run.workflow_status == "failed"
    assert run.final_answer is None


def test_langchain_react_tool_middleware_rejects_before_toolnode_execution() -> None:
    model = _SequencedToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_policy",
                        "args": {"query": "外部系统"},
                        "id": "call-blocked",
                    }
                ],
            ),
            AIMessage(content="无法调用未授权工具。"),
        ]
    )
    lookup = _LookupTool()
    runtime = _build_runtime(
        model=model,
        tools={"lookup_policy": lookup},
        allowed_tools=set(),
    )

    run = runtime.run(
        session_id="session-policy",
        request_id="request-policy",
        user_goal="调用未授权工具。",
        react_run_id="react-policy",
    )

    assert run.workflow_status == "failed"
    assert lookup.calls == []
    assert run.observations[0].tool_call_id == "call-blocked"
    assert "not allowed" in (run.observations[0].error or "")


def test_langchain_react_provider_no_hit_fallback_projection() -> None:
    no_hit_tool = _LookupTool(
        records=[],
        citations=[],
        metadata={
            "trace": {
                "retrieval_trace": {
                    "final_decision": "no_evidence",
                    "no_hit_fallback": True,
                }
            }
        },
    )
    model = _SequencedToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_policy",
                        "args": {"query": "不存在的制度"},
                        "id": "call-no-hit",
                    }
                ],
            ),
            AIMessage(content="未找到相关证据。"),
        ]
    )
    runtime = _build_runtime(
        model=model,
        tools={"lookup_policy": no_hit_tool},
        allowed_tools={"lookup_policy"},
    )

    run = runtime.run(
        session_id="session-no-hit",
        request_id="request-no-hit",
        user_goal="查询不存在的制度。",
        react_run_id="react-no-hit",
    )

    assert run.workflow_status == "succeeded"
    retrieval_trace = run.observations[0].trace["retrieval_trace"]
    assert retrieval_trace["final_decision"] == "no_evidence"
    assert retrieval_trace["no_hit_fallback"] is True
    assert run.final_answer == "未找到相关证据。"


def test_langchain_react_projection_does_not_expose_hidden_chain_of_thought() -> None:
    observation = ToolObservation(
        tool_name="lookup_policy",
        success=True,
        result_summary="命中制度。",
    )
    projection = project_react_agent_output(
        output={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup_policy",
                            "args": {"query": "制度"},
                            "id": "call-1",
                        }
                    ],
                    additional_kwargs={"chain_of_thought": "hidden reasoning"},
                )
            ]
        },
        session_id="session-projection",
        request_id="request-projection",
        user_goal="查制度",
        react_run_id="react-projection",
        max_turns=2,
        trace_events=[{"safe": True, "chain_of_thought": "hidden"}],
    )

    assert "hidden reasoning" not in str(projection.model_dump())
    assert "chain_of_thought" not in str(projection.model_dump())
    assert observation.tool_name == "lookup_policy"


def _build_runtime(
    *,
    model: _SequencedToolChatModel,
    tools: dict[str, SceneTool],
    allowed_tools: set[str] | None = None,
    max_turns: int = 3,
) -> ReActRuntime:
    resolved_allowed_tools = set(tools.keys()) if allowed_tools is None else allowed_tools
    context = AgentRuntimeContext.build(
        session_id="session-test",
        request_id="request-test",
        scene="generic_assistant",
        complexity="simple",
        workflow={"thread_id": "session-test", "checkpoint_ns": "react:test"},
    )
    bundle = build_agent_middleware(
        context=context,
        allowed_tools=sorted(resolved_allowed_tools),
        max_calls_per_tool=max_turns,
    )
    provider_factory = ReActProviderFactory(
        model_provider=lambda complexity: _model_for_complexity(model, complexity),
        middleware_bundle=bundle,
    )
    return ReActRuntime(
        tool_executor=ToolExecutor(
            tools=tools,
            allowed_tools=resolved_allowed_tools,
        ),
        provider_factory=provider_factory,
        middleware_bundle=bundle,
        context=context,
        system_prompt="你是一个测试助手。",
        complexity="simple",
        max_turns=max_turns,
    )


def _model_for_complexity(
    model: _SequencedToolChatModel,
    complexity: TaskComplexity,
) -> BaseChatModel:
    assert complexity == "simple"
    return model
