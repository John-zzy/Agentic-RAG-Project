from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from backend.platform.agent_runtime.middleware.factory import AgentMiddlewareBundle
from backend.platform.agent_runtime.react.middleware import (
    LangChainModelGuardAdapter,
    LangChainToolBoundaryAdapter,
)
from backend.platform.agent_runtime.react.state import (
    ReActContext,
    ReActState,
)
from backend.platform.models.base.router import TaskComplexity


class ReActProviderFactory:
    """Build LangChain `create_agent` providers for ReAct runs."""

    def __init__(
        self,
        *,
        model_provider: Callable[[TaskComplexity], BaseChatModel],
        middleware_bundle: AgentMiddlewareBundle,
        checkpointer: Any | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._middleware_bundle = middleware_bundle
        self._checkpointer = checkpointer

    def build(
        self,
        *,
        complexity: TaskComplexity,
        tools: Sequence[BaseTool],
        system_prompt: str,
    ) -> Any:
        model = self._model_provider(complexity)
        middleware: list[Any] = [
            LangChainModelGuardAdapter(bundle=self._middleware_bundle),
        ]
        if self._middleware_bundle.hitl_interrupts:
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=self._middleware_bundle.hitl_interrupts,
                )
            )
        middleware.append(LangChainToolBoundaryAdapter(bundle=self._middleware_bundle))
        return create_agent(
            model=model,
            tools=list(tools),
            system_prompt=system_prompt,
            middleware=middleware,
            state_schema=ReActState,
            context_schema=ReActContext,
            checkpointer=self._checkpointer,
            name="langchain_react_provider",
        )
