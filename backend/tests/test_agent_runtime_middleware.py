from __future__ import annotations

from pydantic import BaseModel, Field

from backend.platform.agent_runtime.core.contracts import ToolExecutionMetadata, ToolObservation
from backend.platform.agent_runtime.middleware import (
    AgentRuntimeContext,
    SharedModelCallGuard,
    ModelGuardMiddleware,
    ModelGuardPolicy,
    RuntimeTraceMiddleware,
    ToolObservationMiddleware,
    ToolPolicyMiddleware,
    build_agent_middleware,
    build_tool_policy_config,
    observation_status,
)
from backend.platform.tools.base import ToolResult


class _LookupArgs(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=1, ge=1)


def _context() -> AgentRuntimeContext:
    return AgentRuntimeContext.build(
        session_id="session-1",
        request_id="request-1",
        scene="generic_assistant",
        mounted_knowledge_sources=("documents",),
        complexity="moderate",
        provider_name="dashscope",
        workflow={
            "run_id": "run-1",
            "thread_id": "thread-1",
            "checkpoint_ns": "chat_runtime",
            "interrupt_id": "interrupt-1",
            "metadata": {"checkpoint_payload": {"raw": "hidden"}},
        },
        audit={
            "trace_id": "trace-1",
            "metadata": {"safe": "yes", "api_key": "secret"},
        },
        request_metadata={
            "safe_request": "visible",
            "prompt": "do not serialize",
            "history": ["hidden"],
        },
    )


def test_runtime_context_propagates_safe_metadata_without_sensitive_fields() -> None:
    context = _context()

    safe = context.to_safe_metadata()

    assert safe["session_id"] == "session-1"
    assert safe["request_id"] == "request-1"
    assert safe["scene"] == "generic_assistant"
    assert safe["mounted_knowledge_sources"] == ["documents"]
    assert safe["complexity"] == "moderate"
    assert safe["provider_name"] == "dashscope"
    assert safe["workflow"]["run_id"] == "run-1"
    assert safe["workflow"]["thread_id"] == "thread-1"
    assert "interrupt_id" in safe["workflow"]
    assert safe["audit"]["metadata"] == {"safe": "yes"}
    assert safe["request_metadata"] == {"safe_request": "visible"}


def test_factory_returns_ordered_middleware_with_shared_context() -> None:
    context = _context()

    bundle = build_agent_middleware(
        context=context,
        allowed_tools=("knowledge_document_search",),
        tool_source_scope={"knowledge_document_search": ("documents",)},
    )

    assert bundle.context is context
    assert [component.__class__.__name__ for component in bundle.ordered] == [
        "DynamicPromptMiddleware",
        "ModelGuardMiddleware",
        "ToolPolicyMiddleware",
        "ToolObservationMiddleware",
        "RuntimeTraceMiddleware",
    ]
    assert bundle.hitl_interrupts == {}


def test_factory_builds_langchain_hitl_interrupts_for_high_risk_tools() -> None:
    bundle = build_agent_middleware(
        context=_context(),
        allowed_tools=("knowledge_document_search", "write_record"),
        high_risk_tools=("write_record",),
    )

    assert bundle.hitl_interrupts == {
        "write_record": {"allowed_decisions": ["approve", "reject", "respond"]}
    }


def test_dynamic_prompt_composes_scene_owned_prompt_and_filters_resume_payload() -> None:
    bundle = build_agent_middleware(
        context=_context(),
        allowed_tools=("knowledge_document_search",),
    )

    result = bundle.dynamic_prompt.compose_from_parts(
        context=bundle.context,
        scene_prompt="Scene-owned prompt.",
        history_view=("User asked about policy.",),
        mounted_knowledge_policy="Use mounted documents only.",
        resume_metadata={
            "interrupt_id": "interrupt-1",
            "raw_checkpoint_payload": {"secret": "hidden"},
            "safe_resume_reason": "approved",
        },
    )

    assert "Scene-owned prompt." in result.system_prompt
    assert "User asked about policy." in result.system_prompt
    assert "Use mounted documents only." in result.system_prompt
    assert "safe_resume_reason" in result.system_prompt
    assert "raw_checkpoint_payload" not in result.system_prompt
    assert "hidden" not in result.system_prompt


def test_model_guard_classifies_empty_output_and_tracks_latency_retry_fallback() -> None:
    calls = 0

    def empty_model() -> str:
        nonlocal calls
        calls += 1
        return "  "

    guard = ModelGuardMiddleware(
        policy=ModelGuardPolicy(
            max_attempts=2,
            retry_on=ValueError,
            fallback="fallback answer",
        )
    )

    result = guard.invoke(
        empty_model,
        context=_context(),
        token_metadata={"input_tokens": 4, "output_tokens": 0},
    )

    assert calls == 2
    assert result.success is True
    assert result.output == "fallback answer"
    assert result.error == "Model returned empty content"
    assert result.metadata.provider == "dashscope"
    assert result.metadata.complexity == "moderate"
    assert result.metadata.retry_count == 1
    assert result.metadata.fallback_used is True
    assert result.metadata.latency_ms >= 0
    assert result.metadata.token_usage == {"input_tokens": 4, "output_tokens": 0}
    assert result.metadata.error_classification is not None
    assert result.metadata.error_classification["category"] == "model_empty_output"


def test_shared_model_call_guard_records_trace_for_typed_output() -> None:
    trace = RuntimeTraceMiddleware()
    guard = SharedModelCallGuard(
        guard=ModelGuardMiddleware(policy=ModelGuardPolicy(max_attempts=1)),
        trace=trace,
    )

    output = guard.invoke(
        lambda: {"answer": "ok"},
        context=_context(),
        metadata={"operation": "unit.model_call", "prompt": "hidden"},
        output_type=dict,
    )

    assert output == {"answer": "ok"}
    assert trace.events[-1].event_type == "model_call"
    assert trace.events[-1].metadata["operation"] == "unit.model_call"
    assert "prompt" not in trace.events[-1].metadata


def test_tool_policy_rejects_unmounted_or_invalid_tool_before_execution() -> None:
    policy = ToolPolicyMiddleware(
        build_tool_policy_config(
            allowed_tools=("knowledge_document_search",),
            tool_source_scope={"knowledge_document_search": ("documents", "ecommerce")},
        )
    )

    missing_source = policy.validate(
        tool_name="knowledge_document_search",
        input_payload={"query": "policy"},
        context=_context(),
        args_schema=_LookupArgs,
    )
    unavailable = policy.validate(
        tool_name="unsafe_tool",
        input_payload={"query": "policy"},
        context=_context(),
        args_schema=_LookupArgs,
    )
    invalid_input = ToolPolicyMiddleware(
        build_tool_policy_config(allowed_tools=("knowledge_document_search",))
    ).validate(
        tool_name="knowledge_document_search",
        input_payload={"query": "policy", "limit": 0},
        context=_context(),
        args_schema=_LookupArgs,
    )

    assert missing_source.allowed is False
    assert "missing ecommerce" in str(missing_source.reason)
    assert unavailable.allowed is False
    assert "not allowed" in str(unavailable.reason)
    assert invalid_input.allowed is False
    assert "Invalid input" in str(invalid_input.reason)


def test_tool_policy_accepts_allowed_tool_and_applies_call_limit() -> None:
    policy = ToolPolicyMiddleware(
        build_tool_policy_config(
            allowed_tools=("write_record",),
            high_risk_tools=("write_record",),
            max_calls_per_tool=1,
        )
    )

    accepted = policy.validate(
        tool_name="write_record",
        input_payload={"query": "order"},
        context=_context(),
        args_schema=_LookupArgs,
    )
    limited = policy.validate(
        tool_name="write_record",
        input_payload={"query": "order"},
        context=_context(),
        args_schema=_LookupArgs,
    )

    assert accepted.allowed is True
    assert accepted.risk_level == "high"
    assert accepted.input_payload == {"query": "order", "limit": 1}
    assert limited.allowed is False
    assert "limit exceeded" in str(limited.reason)


def test_tool_observation_normalizes_tool_result_and_status() -> None:
    middleware = ToolObservationMiddleware()

    observation = middleware.normalize(
        tool_name="lookup_policy",
        result=ToolResult.ok(
            tool_name="lookup_policy",
            records=[{"policy": "ok"}],
            citations=[{"citation_id": "policy-1"}],
            metadata={
                "trace": {
                    "retrieval_trace": {
                        "final_decision": "answer_with_evidence",
                        "answer_mode": "rag",
                        "no_hit_fallback": False,
                    }
                }
            },
        ),
        execution=ToolExecutionMetadata(tool_name="lookup_policy", tool_call_id="call-1"),
    )
    failed = middleware.normalize(
        tool_name="lookup_policy",
        error=TimeoutError("lookup timed out"),
    )

    assert observation.success is True
    assert observation.output["records"] == [{"policy": "ok"}]
    assert observation.citations == [{"citation_id": "policy-1"}]
    assert observation.trace["retrieval_trace"]["final_decision"] == "answer_with_evidence"
    assert observation.trace["retrieval_trace"]["answer_mode"] == "rag"
    assert observation_status(observation) == "succeeded"
    assert failed.success is False
    assert failed.retryable is True
    assert observation_status(failed) == "retryable"


def test_trace_filters_prompt_history_tool_args_and_secret_fields() -> None:
    trace = RuntimeTraceMiddleware()
    context = _context()
    observation = ToolObservation(
        tool_name="lookup_policy",
        success=False,
        retryable=False,
        error="Invalid input for tool: lookup_policy.",
    )

    event = trace.record_tool_call(
        context=context,
        observation=observation,
        metadata={
            "latency_ms": 10,
            "tool_args": {"query": "hidden"},
            "raw_prompt": "hidden",
            "secret": "hidden",
            "safe": {"provider": "dashscope"},
        },
    )

    assert event.metadata["tool_name"] == "lookup_policy"
    assert event.metadata["tool_status"] == "failed"
    assert event.metadata["error_classification"] == "validation"
    assert event.metadata["safe"] == {"provider": "dashscope"}
    encoded = str(event.metadata)
    assert "hidden" not in encoded
    assert "tool_args" not in encoded
    assert "raw_prompt" not in encoded
    assert "secret" not in encoded
    assert trace.events == (event,)
