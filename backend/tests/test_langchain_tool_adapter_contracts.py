from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel, Field

from backend.platform.agent_runtime import (
    ToolExecutor,
    build_langchain_tools_from_executor,
    observation_from_langchain_artifact,
)
from backend.platform.agent_runtime.core.contracts import ToolObservation
from backend.platform.agent_runtime.tooling.rag import NativeRagToolAdapter
from backend.platform.models.llm.client import ModelClient
from backend.platform.rag.contracts import RetrievalResult
from backend.platform.tools.base import SceneTool, ToolResult


class _LookupArgs(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=1, ge=1)


class _FakeChatModel(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        del messages, stop, run_manager, kwargs
        return "ok"


class _LookupTool(SceneTool):
    name = "lookup_policy"
    description = "Lookup policy records."
    capability_type = "retrieval"
    args_schema = _LookupArgs

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(kwargs))
        return ToolResult.ok(
            tool_name=self.name,
            records=[dict(kwargs)],
            citations=[{"citation_id": "policy-1"}],
        )


class _ApprovalTool(SceneTool):
    name = "approval_tool"
    description = "Ask for approval."
    capability_type = "action"
    args_schema = _LookupArgs

    def invoke(self, **kwargs: Any) -> ToolObservation:
        del kwargs
        return ToolObservation(
            tool_name=self.name,
            success=False,
            requires_user=True,
            user_prompt="是否批准查询外部系统？",
            result_summary="等待用户批准。",
            trace={"hitl": {"pending_action": "tool_approval"}},
        )


class _FailingTool(SceneTool):
    name = "failing_tool"
    description = "Always fails."
    capability_type = "retrieval"
    args_schema = _LookupArgs

    def invoke(self, **kwargs: Any) -> ToolResult:
        del kwargs
        raise TimeoutError("lookup timed out")


class _FakeNativeRetriever:
    def retrieve(self, **kwargs: Any) -> RetrievalResult:
        query = str(kwargs["query"])
        return RetrievalResult.ok(
            tool_name="knowledge_document_search",
            query=query,
            records=[{"chunk_id": "chunk-1", "text": "差旅报销需提交发票。"}],
            documents=[
                Document(
                    page_content="差旅报销需提交发票。",
                    metadata={"citation_id": "chunk-1", "source": "policy.md"},
                )
            ],
            citations=[
                {
                    "citation_id": "chunk-1",
                    "snippet": "差旅报销需提交发票。",
                    "source_type": "document",
                }
            ],
            metadata={
                "retrieval_trace": {
                    "raw_candidates_count": 1,
                    "adopted_citations": [{"citation_id": "chunk-1"}],
                    "answer_mode": "evidence_answer",
                    "no_hit_fallback": False,
                }
            },
        )


def test_langchain_tool_generates_schema_from_neutral_tool_definition() -> None:
    executor = ToolExecutor(
        tools={"lookup_policy": _LookupTool()},
        allowed_tools={"lookup_policy"},
    )

    tool = build_langchain_tools_from_executor(executor)[0]

    schema = tool.args_schema.model_json_schema()
    assert tool.name == "lookup_policy"
    assert tool.description == "Lookup policy records."
    assert schema["properties"]["query"]["minLength"] == 1
    assert schema["properties"]["limit"]["minimum"] == 1
    assert "tool_call_id" not in schema["properties"]


def test_langchain_tool_exposes_only_allowed_tools_for_scene_scope() -> None:
    executor = ToolExecutor(
        tools={
            "lookup_policy": _LookupTool(),
            "failing_tool": _FailingTool(),
        },
        allowed_tools={"lookup_policy"},
    )

    tools = build_langchain_tools_from_executor(executor)

    assert [tool.name for tool in tools] == ["lookup_policy"]


def test_langchain_tool_invocation_returns_tool_message_with_observation_artifact() -> None:
    neutral_tool = _LookupTool()
    executor = ToolExecutor(
        tools={"lookup_policy": neutral_tool},
        allowed_tools={"lookup_policy"},
    )
    tool = build_langchain_tools_from_executor(executor)[0]

    message = _invoke_tool_call(
        tool,
        tool_call_id="call-1",
        args={"query": "报销", "limit": 2},
    )

    assert message.content == "lookup_policy succeeded with 1 record(s)."
    observation = observation_from_langchain_artifact(message.artifact)
    assert neutral_tool.calls == [{"query": "报销", "limit": 2}]
    assert observation.success is True
    assert message.tool_call_id == "call-1"
    assert observation.tool_call_id is not None
    assert observation.execution is not None
    assert observation.citations == [{"citation_id": "policy-1"}]


def test_langchain_tool_normalizes_invalid_input_without_tool_invocation() -> None:
    neutral_tool = _LookupTool()
    executor = ToolExecutor(
        tools={"lookup_policy": neutral_tool},
        allowed_tools={"lookup_policy"},
    )
    tool = build_langchain_tools_from_executor(executor)[0]

    with pytest.raises(Exception):
        _invoke_tool_call(
            tool,
            tool_call_id="call-invalid",
            args={"query": "报销", "limit": 0},
        )

    assert neutral_tool.calls == []


def test_langchain_tool_normalizes_retryable_failure() -> None:
    executor = ToolExecutor(
        tools={"failing_tool": _FailingTool()},
        allowed_tools={"failing_tool"},
    )
    tool = build_langchain_tools_from_executor(executor)[0]

    message = _invoke_tool_call(
        tool,
        tool_call_id="call-fail",
        args={"query": "制度"},
    )

    observation = observation_from_langchain_artifact(message.artifact)
    assert message.content == "failing_tool failed: lookup timed out"
    assert observation.success is False
    assert observation.retryable is True
    assert observation.error == "lookup timed out"
    assert message.artifact["status"] == "retryable"


def test_langchain_tool_preserves_hitl_observation() -> None:
    executor = ToolExecutor(
        tools={"approval_tool": _ApprovalTool()},
        allowed_tools={"approval_tool"},
    )
    tool = build_langchain_tools_from_executor(executor)[0]

    message = _invoke_tool_call(
        tool,
        tool_call_id="call-hitl",
        args={"query": "订单"},
    )

    observation = observation_from_langchain_artifact(message.artifact)
    assert message.content == "是否批准查询外部系统？"
    assert message.artifact["status"] == "waiting_user"
    assert observation.requires_user is True
    assert observation.user_prompt == "是否批准查询外部系统？"
    assert observation.trace["hitl"]["pending_action"] == "tool_approval"


def test_langchain_rag_tool_artifact_preserves_evidence_metadata() -> None:
    rag_tool = NativeRagToolAdapter(
        name="native_rag_search",
        retriever=_FakeNativeRetriever(),
    )
    executor = ToolExecutor(
        tools={"native_rag_search": rag_tool},
        allowed_tools={"native_rag_search"},
    )
    tool = build_langchain_tools_from_executor(executor)[0]

    message = _invoke_tool_call(
        tool,
        tool_call_id="call-rag",
        args={"query": "报销制度"},
    )

    assert "native_rag_search returned" in message.content
    observation = observation_from_langchain_artifact(message.artifact)
    retrieval_trace = observation.trace["retrieval_trace"]
    assert observation.citations[0]["citation_id"] == "chunk-1"
    assert retrieval_trace["adopted_citations"] == [{"citation_id": "chunk-1"}]
    assert retrieval_trace["final_decision"] == "answer_with_evidence"
    assert retrieval_trace["answer_mode"] == "evidence_answer"
    assert retrieval_trace["no_hit_fallback"] is False
    assert observation.metadata["adopted_citations"] == [{"citation_id": "chunk-1"}]


def test_chat_model_provider_resolves_model_by_complexity(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def factory(**kwargs: Any) -> BaseChatModel:
        observed.append(str(kwargs["model"]))
        return _FakeChatModel()

    client = ModelClient(chat_model_factory=factory)
    monkeypatch.setattr(
        "backend.platform.models.llm.client.get_model_for_task",
        lambda complexity: type(
            "RoutedModelStub",
            (),
            {
                "complexity": complexity,
                "provider": "fake",
                "api_key": "test-key",
                "model_name": f"fake-{complexity}",
                "api_base": "https://example.test",
                "timeout_seconds": 30,
                "temperature": 0.1,
                "max_tokens": 128,
                "supports_streaming": True,
            },
        )(),
    )

    provider = client.get_chat_model_provider()

    assert isinstance(provider("complex"), BaseChatModel)
    assert observed == ["fake-complex"]


def _invoke_tool_call(
    tool: Any,
    *,
    tool_call_id: str,
    args: dict[str, Any],
) -> ToolMessage:
    message = tool.invoke(
        {
            "type": "tool_call",
            "id": tool_call_id,
            "name": tool.name,
            "args": args,
        }
    )
    assert isinstance(message, ToolMessage)
    return message
