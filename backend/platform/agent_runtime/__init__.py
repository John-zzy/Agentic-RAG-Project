from __future__ import annotations

from typing import Any

from backend.platform.agent_runtime.core.contracts import (
    AgentMode,
    AgentRun,
    PlanRun,
    PlanStep,
    PlanStepStatus,
    ReActAction,
    ReActActionType,
    ReActRun,
    ReActTurn,
    ReActTurnStatus,
    RetryMetadata,
    ToolExecutionMetadata,
    ToolObservation,
    collect_successful_tool_observations,
)
from backend.platform.agent_runtime.core.validation import (
    AgentRuntimeValidationError,
    PlanDependencyValidationError,
    ToolAccessValidationError,
    ToolInputValidationError,
    build_retry_metadata,
    ensure_tool_allowed,
    validate_plan_dependencies,
    validate_plan_tool_allowlist,
    validate_tool_input,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LangChainToolFactory": (
        "backend.platform.agent_runtime.tooling.langchain",
        "LangChainToolFactory",
    ),
    "build_langchain_tools_from_executor": (
        "backend.platform.agent_runtime.tooling.langchain",
        "build_langchain_tools_from_executor",
    ),
    "observation_from_langchain_artifact": (
        "backend.platform.agent_runtime.tooling.langchain",
        "observation_from_langchain_artifact",
    ),
    "MinimalModeSelector": ("backend.platform.agent_runtime.core.mode_selector", "MinimalModeSelector"),
    "ModeSelection": ("backend.platform.agent_runtime.core.mode_selector", "ModeSelection"),
    "ModeSelectionContext": ("backend.platform.agent_runtime.core.mode_selector", "ModeSelectionContext"),
    "ModeSelector": ("backend.platform.agent_runtime.core.mode_selector", "ModeSelector"),
    "LangChainPlanPlanner": (
        "backend.platform.agent_runtime.plan.planner",
        "LangChainPlanPlanner",
    ),
    "PlanDraft": ("backend.platform.agent_runtime.plan.planner", "PlanDraft"),
    "PlanFinalSynthesizer": (
        "backend.platform.agent_runtime.plan.synthesis",
        "PlanFinalSynthesizer",
    ),
    "PlanPlannerContext": (
        "backend.platform.agent_runtime.plan.planner",
        "PlanPlannerContext",
    ),
    "PlanStepDraft": ("backend.platform.agent_runtime.plan.planner", "PlanStepDraft"),
    "PlanSynthesisContext": (
        "backend.platform.agent_runtime.plan.synthesis",
        "PlanSynthesisContext",
    ),
    "PlanSynthesisResult": (
        "backend.platform.agent_runtime.plan.synthesis",
        "PlanSynthesisResult",
    ),
    "StepSummarySynthesizer": (
        "backend.platform.agent_runtime.plan.synthesis",
        "StepSummarySynthesizer",
    ),
    "AGENTIC_RAG_TOOL_NAME": (
        "backend.platform.agent_runtime.tooling.rag",
        "AGENTIC_RAG_TOOL_NAME",
    ),
    "NATIVE_RAG_TOOL_NAME": ("backend.platform.agent_runtime.tooling.rag", "NATIVE_RAG_TOOL_NAME"),
    "AgenticRAGToolAdapter": (
        "backend.platform.agent_runtime.tooling.rag",
        "AgenticRAGToolAdapter",
    ),
    "AgenticRagToolAdapter": (
        "backend.platform.agent_runtime.tooling.rag",
        "AgenticRagToolAdapter",
    ),
    "NativeRAGToolAdapter": (
        "backend.platform.agent_runtime.tooling.rag",
        "NativeRAGToolAdapter",
    ),
    "NativeRagToolAdapter": (
        "backend.platform.agent_runtime.tooling.rag",
        "NativeRagToolAdapter",
    ),
    "RAGToolAdapter": ("backend.platform.agent_runtime.tooling.rag", "RAGToolAdapter"),
    "RAGToolInput": ("backend.platform.agent_runtime.tooling.rag", "RAGToolInput"),
    "agentic_outcome_to_observation": (
        "backend.platform.agent_runtime.tooling.rag",
        "agentic_outcome_to_observation",
    ),
    "build_rag_tool_adapters": (
        "backend.platform.agent_runtime.tooling.rag",
        "build_rag_tool_adapters",
    ),
    "retrieval_result_to_observation": (
        "backend.platform.agent_runtime.tooling.rag",
        "retrieval_result_to_observation",
    ),
    "ReActRuntime": ("backend.platform.agent_runtime.react", "ReActRuntime"),
    "ReActScenePolicy": ("backend.platform.agent_runtime.react", "ReActScenePolicy"),
    "ToolExecutor": ("backend.platform.agent_runtime.tooling.executor", "ToolExecutor"),
}

_ALIASES: dict[str, str] = {
    "PlanSummarySynthesizer": "StepSummarySynthesizer",
}


def __getattr__(name: str) -> Any:
    target_name = _ALIASES.get(name, name)
    if target_name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[target_name]

    # 包初始化保持轻量，只有调用方显式需要重对象时才加载对应子模块。
    from importlib import import_module

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AGENTIC_RAG_TOOL_NAME",
    "AgentMode",
    "AgentRun",
    "AgentRuntimeValidationError",
    "AgenticRAGToolAdapter",
    "AgenticRagToolAdapter",
    "LangChainToolFactory",
    "NATIVE_RAG_TOOL_NAME",
    "NativeRAGToolAdapter",
    "NativeRagToolAdapter",
    "LangChainPlanPlanner",
    "MinimalModeSelector",
    "ModeSelection",
    "ModeSelectionContext",
    "ModeSelector",
    "PlanDependencyValidationError",
    "PlanDraft",
    "PlanFinalSynthesizer",
    "PlanPlannerContext",
    "PlanRun",
    "PlanStep",
    "PlanStepDraft",
    "PlanStepStatus",
    "PlanSummarySynthesizer",
    "PlanSynthesisContext",
    "PlanSynthesisResult",
    "RAGToolAdapter",
    "RAGToolInput",
    "ReActAction",
    "ReActActionType",
    "ReActRuntime",
    "ReActRun",
    "ReActScenePolicy",
    "ReActTurn",
    "ReActTurnStatus",
    "RetryMetadata",
    "StepSummarySynthesizer",
    "ToolAccessValidationError",
    "ToolExecutor",
    "ToolExecutionMetadata",
    "ToolInputValidationError",
    "ToolObservation",
    "agentic_outcome_to_observation",
    "build_langchain_tools_from_executor",
    "build_rag_tool_adapters",
    "build_retry_metadata",
    "collect_successful_tool_observations",
    "ensure_tool_allowed",
    "observation_from_langchain_artifact",
    "retrieval_result_to_observation",
    "validate_plan_dependencies",
    "validate_plan_tool_allowlist",
    "validate_tool_input",
]
