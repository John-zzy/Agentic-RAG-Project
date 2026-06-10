from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from langchain.tools import BaseTool
from pydantic import BaseModel

from backend.platform.agent_runtime.core.contracts import (
    ToolExecutionMetadata,
    ToolObservation,
)
from backend.platform.agent_runtime.tooling.idempotency import (
    ToolExecutionContext,
    ToolIdempotencyStore,
    attach_idempotency_trace,
    build_input_hash,
    build_tool_idempotency_key,
    compensation_status_for_tool,
    is_side_effect_tool,
)
from backend.platform.agent_runtime.middleware.context import AgentRuntimeContext
from backend.platform.agent_runtime.middleware.tool_observation import ToolObservationMiddleware
from backend.platform.agent_runtime.middleware.tool_policy import (
    ToolPolicyMiddleware,
    build_tool_policy_config,
)
from backend.platform.agent_runtime.middleware.trace import RuntimeTraceMiddleware
from backend.platform.agent_runtime.core.validation import (
    ToolAccessValidationError,
    ToolInputValidationError,
    ensure_tool_allowed,
    validate_tool_input,
)
from backend.platform.rag.contracts import RetrievalResult
from backend.platform.tools.base import SceneTool, ToolResult


class ToolExecutor:
    """统一执行 scene tools、RAG tools 和 internal tools 的平台边界。"""

    def __init__(
        self,
        *,
        tools: Mapping[str, Any],
        allowed_tools: Sequence[str] | set[str] | None = None,
        idempotency_store: ToolIdempotencyStore | None = None,
        trace_middleware: RuntimeTraceMiddleware | None = None,
    ) -> None:
        self._tools = dict(tools)
        self._allowed_tools = (
            set(self._tools.keys()) if allowed_tools is None else set(allowed_tools)
        )
        self._idempotency_store = idempotency_store
        self._policy = ToolPolicyMiddleware(
            build_tool_policy_config(allowed_tools=self._allowed_tools)
        )
        self._observation = ToolObservationMiddleware()
        self._trace = trace_middleware

    @classmethod
    def from_scene(
        cls,
        *,
        scene_definition: Any,
        mounted_knowledge_sources: Sequence[str] = (),
        candidate_retrieval_tools: Sequence[str] | None = None,
        rag_tools: Mapping[str, Any] | None = None,
        internal_tools: Mapping[str, Any] | None = None,
        idempotency_store: ToolIdempotencyStore | None = None,
        trace_middleware: RuntimeTraceMiddleware | None = None,
    ) -> "ToolExecutor":
        """按 active scene 和挂载知识源解析可调用工具集合。"""
        tools: dict[str, Any] = {}
        allowed_tools: set[str] = set()
        for tool in scene_definition.build_tools():
            _register_tool(tools, tool)

        resolved_candidate_tools = set(
            candidate_retrieval_tools
            if candidate_retrieval_tools is not None
            else scene_definition.resolve_candidate_retrieval_tools(
                tuple(mounted_knowledge_sources)
            )
        )
        # scene resolver 是 mounted knowledge source 的边界；只放行当前会话解析出的工具名。
        allowed_tools.update(tool_name for tool_name in resolved_candidate_tools if tool_name in tools)
        if _should_allow_scene_structured_tools(
            scene_definition=scene_definition,
            mounted_knowledge_sources=mounted_knowledge_sources,
        ):
            allowed_tools.update(tools.keys())

        for tool in (rag_tools or {}).values():
            _register_tool(tools, tool)
            allowed_tools.add(tool.name)
        for tool in (internal_tools or {}).values():
            _register_tool(tools, tool)
            allowed_tools.add(tool.name)

        return cls(
            tools=tools,
            allowed_tools=allowed_tools,
            idempotency_store=idempotency_store,
            trace_middleware=trace_middleware,
        )

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(self._allowed_tools)

    @property
    def registered_tools(self) -> frozenset[str]:
        return frozenset(self._tools.keys())

    def get_registered_tool(self, tool_name: str) -> Any:
        """读取已注册工具定义，供只负责形状转换的 adapter 复用。"""
        if tool_name not in self._tools:
            raise ToolAccessValidationError(f"Tool is not registered: {tool_name}.")
        return self._tools[tool_name]

    def validate_call(
        self,
        *,
        tool_name: str,
        input_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """只校验工具访问和输入结构，不触发实际工具调用。"""
        ensure_tool_allowed(tool_name, self._allowed_tools)
        if tool_name not in self._tools:
            raise ToolAccessValidationError(f"Tool is not registered: {tool_name}.")
        return validate_tool_input(
            tool_name=tool_name,
            input_payload=input_payload,
            args_schema=_resolve_args_schema(self._tools[tool_name]),
        )

    def execute(
        self,
        *,
        tool_name: str,
        input_payload: Mapping[str, Any] | None = None,
        tool_call_id: str | None = None,
        attempt: int = 0,
        max_attempts: int = 2,
        execution_context: ToolExecutionContext | None = None,
    ) -> ToolObservation:
        idempotency_key: str | None = None
        tool: Any | None = None
        execution = ToolExecutionMetadata(
            tool_name=tool_name,
            tool_call_id=tool_call_id or str(uuid4()),
            attempt=attempt,
            max_attempts=max_attempts,
        )
        try:
            context = _build_tool_runtime_context(
                tool_name=tool_name,
                execution_context=execution_context,
            )
            decision = self._policy.validate(
                tool_name=tool_name,
                input_payload=input_payload,
                context=context,
                args_schema=_resolve_args_schema(self._tools.get(tool_name)),
            )
            if not decision.allowed:
                return self._record_trace(
                    context=context,
                    observation=self._observation.normalize(
                        tool_name=tool_name,
                        error=ToolAccessValidationError(decision.reason or "Tool policy rejected call."),
                        execution=execution.model_copy(update={"retryable": False}),
                        retryable=False,
                    ),
                )
            if tool_name not in self._tools:
                raise ToolAccessValidationError(f"Tool is not registered: {tool_name}.")
            validated_input = decision.input_payload
            tool = self._tools[tool_name]
            idempotency_key = self._resolve_idempotency_key(
                tool=tool,
                tool_name=tool_name,
                input_payload=validated_input,
                execution_context=execution_context,
            )
            if idempotency_key is not None:
                execution = execution.model_copy(update={"idempotency_key": idempotency_key})
                reused = self._reuse_or_begin_invocation(
                    key=idempotency_key,
                    tool=tool,
                    tool_name=tool_name,
                    input_payload=validated_input,
                    execution_context=execution_context,
                )
                if reused is not None:
                    return reused
            result = _invoke_tool(tool, validated_input)
            observation = self._observation.normalize(
                tool_name=tool_name,
                result=result,
                execution=execution,
            )
            return self._record_idempotent_observation(
                observation=self._record_trace(context=context, observation=observation),
                tool=tool,
                idempotency_key=idempotency_key,
            )
        except (ToolAccessValidationError, ToolInputValidationError) as exc:
            return self._record_trace(
                context=_build_tool_runtime_context(
                    tool_name=tool_name,
                    execution_context=execution_context,
                ),
                observation=self._observation.normalize(
                    tool_name=tool_name,
                    error=exc,
                    execution=execution.model_copy(update={"retryable": False}),
                    retryable=False,
                ),
            )
        except Exception as exc:
            retryable = _is_retryable_exception(exc)
            context = _build_tool_runtime_context(
                tool_name=tool_name,
                execution_context=execution_context,
            )
            observation = self._record_trace(
                context=context,
                observation=self._observation.normalize(
                    tool_name=tool_name,
                    error=exc,
                    execution=execution.model_copy(update={"retryable": retryable}),
                    retryable=retryable,
                ),
            )
            if tool is None:
                return observation
            return self._record_idempotent_observation(
                observation=observation,
                tool=tool,
                idempotency_key=idempotency_key,
            )

    def _resolve_idempotency_key(
        self,
        *,
        tool: Any,
        tool_name: str,
        input_payload: Mapping[str, Any],
        execution_context: ToolExecutionContext | None,
    ) -> str | None:
        if self._idempotency_store is None or execution_context is None:
            return None
        if not is_side_effect_tool(tool):
            return None
        return build_tool_idempotency_key(
            context=execution_context,
            tool_name=tool_name,
            input_payload=input_payload,
        )

    def _reuse_or_begin_invocation(
        self,
        *,
        key: str,
        tool: Any,
        tool_name: str,
        input_payload: Mapping[str, Any],
        execution_context: ToolExecutionContext | None,
    ) -> ToolObservation | None:
        if self._idempotency_store is None or execution_context is None:
            return None
        compensation_status = compensation_status_for_tool(tool)
        existing = self._idempotency_store.begin_invocation(
            key=key,
            tool_name=tool_name,
            input_hash=build_input_hash(input_payload),
            context=execution_context,
            compensation_status=compensation_status,
        )
        if existing is None:
            return None
        if existing.status == "succeeded" and existing.observation is not None:
            return attach_idempotency_trace(
                observation=existing.observation,
                key=key,
                status="succeeded",
                reused=True,
                compensation_status=existing.compensation_status,
            )
        if existing.status == "failed" and existing.observation is not None:
            return attach_idempotency_trace(
                observation=existing.observation,
                key=key,
                status="failed",
                reused=True,
                compensation_status=existing.compensation_status,
            )
        return ToolObservation(
            tool_name=tool_name,
            success=False,
            result_summary=f"{tool_name} is already pending for this idempotency key.",
            retryable=True,
            error="Tool invocation is already pending.",
            execution=ToolExecutionMetadata(
                tool_name=tool_name,
                idempotency_key=key,
                retryable=True,
            ),
            metadata={
                "idempotency": {
                    "idempotency_key": key,
                    "status": "pending",
                    "reused": True,
                    "compensation_status": existing.compensation_status,
                },
                "compensation_status": existing.compensation_status,
            },
            trace={
                "idempotency": {
                    "idempotency_key": key,
                    "status": "pending",
                    "reused": True,
                    "compensation_status": existing.compensation_status,
                },
                "compensation": {"status": existing.compensation_status},
            },
        )

    def _record_idempotent_observation(
        self,
        *,
        observation: ToolObservation,
        tool: Any,
        idempotency_key: str | None,
    ) -> ToolObservation:
        if self._idempotency_store is None or idempotency_key is None:
            return observation
        compensation_status = compensation_status_for_tool(tool)
        traced = attach_idempotency_trace(
            observation=observation,
            key=idempotency_key,
            status="succeeded" if observation.success else "failed",
            reused=False,
            compensation_status=compensation_status,
        )
        if traced.success:
            self._idempotency_store.mark_succeeded(
                key=idempotency_key,
                observation=traced,
                compensation_status=compensation_status,
            )
        else:
            self._idempotency_store.mark_failed(
                key=idempotency_key,
                observation=traced,
                compensation_status=compensation_status,
            )
        return traced

    def _record_trace(
        self,
        *,
        context: AgentRuntimeContext,
        observation: ToolObservation,
    ) -> ToolObservation:
        if self._trace is not None:
            self._trace.record_tool_call(context=context, observation=observation)
        return observation


def _register_tool(tools: dict[str, Any], tool: Any) -> None:
    tool_name = getattr(tool, "name", None)
    if not tool_name:
        raise ValueError("Tool name is required.")
    if tool_name in tools:
        raise ValueError(f"Duplicate tool registration: {tool_name}.")
    tools[tool_name] = tool


def _should_allow_scene_structured_tools(
    *,
    scene_definition: Any,
    mounted_knowledge_sources: Sequence[str],
) -> bool:
    """业务知识源挂载后，允许该 scene 的结构化业务工具参与 Agent 编排。"""
    metadata = getattr(scene_definition, "metadata", {}) or {}
    knowledge_sources = set(metadata.get("knowledge_sources") or ())
    mounted_sources = set(mounted_knowledge_sources)
    business_sources = knowledge_sources - {"documents"}
    return bool(business_sources & mounted_sources)


def _resolve_args_schema(tool: Any) -> type[BaseModel] | None:
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
        return args_schema
    return None


def _invoke_tool(tool: Any, input_payload: dict[str, Any]) -> Any:
    if isinstance(tool, BaseTool):
        return tool.invoke(input_payload)
    if isinstance(tool, SceneTool):
        return tool.invoke(**input_payload)
    # Agent Runtime 自有工具约定接收一个 mapping，避免和 SceneTool kwargs 协议混用。
    return tool.invoke(input_payload)


def _build_tool_runtime_context(
    *,
    tool_name: str,
    execution_context: ToolExecutionContext | None,
) -> AgentRuntimeContext:
    metadata = dict(execution_context.metadata or {}) if execution_context else {}
    return AgentRuntimeContext.build(
        session_id=execution_context.session_id if execution_context else "tool-executor",
        request_id=execution_context.request_id if execution_context else "tool-executor",
        scene=str(metadata.get("scene") or "platform.tool_executor"),
        mounted_knowledge_sources=tuple(metadata.get("mounted_knowledge_sources") or ("documents",)),
        complexity=str(metadata.get("complexity") or "simple"),
        workflow={
            "run_id": execution_context.run_id if execution_context else None,
            "node_name": execution_context.node_name if execution_context else None,
            "metadata": {
                "step_id": execution_context.step_id if execution_context else None,
                "tool_name": tool_name,
                "mode": metadata.get("mode"),
            },
        },
        request_metadata={"operation": "tool.execute"},
    )


def _tool_result_to_observation(
    *,
    tool_name: str,
    result: ToolResult,
    execution: ToolExecutionMetadata,
) -> ToolObservation:
    requires_user = bool(result.metadata.get("requires_user"))
    retryable = bool(result.metadata.get("retryable", not result.success))
    return ToolObservation(
        tool_name=tool_name,
        success=result.success,
        output={
            "records": list(result.records),
            "confidence": result.confidence,
            "metadata": dict(result.metadata),
        },
        result_summary=_tool_result_summary(tool_name=tool_name, result=result),
        citations=list(result.citations) if result.success else [],
        trace=dict(result.metadata.get("trace") or {}),
        retryable=retryable,
        requires_user=requires_user,
        user_prompt=result.metadata.get("user_prompt"),
        error=result.error,
        execution=execution.model_copy(update={"retryable": retryable}),
        tool_call_id=execution.tool_call_id,
        metadata=dict(result.metadata),
    )


def _tool_result_summary(*, tool_name: str, result: ToolResult) -> str:
    if result.success:
        return f"{tool_name} succeeded with {len(result.records)} record(s)."
    return f"{tool_name} failed: {result.error or 'unknown error'}."


def _is_retryable_exception(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError))
