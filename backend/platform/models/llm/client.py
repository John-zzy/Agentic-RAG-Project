from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable
from pydantic import BaseModel

from backend.platform.models.base.router import TaskComplexity, RoutedModel, get_model_for_task
from backend.platform.models.llm.guards import (
    JsonSchemaGuard,
    ModelGuardAdapter,
    ModelGuardConfig,
)


class ModelClient:
    """封装基于复杂度路由的 LangChain 聊天模型与 runnable 入口。"""

    def __init__(
        self,
        chat_model_factory: Callable[..., Any] | None = None,
        *,
        guard_adapter: ModelGuardAdapter | None = None,
        schema_guard: JsonSchemaGuard | None = None,
    ) -> None:
        """初始化模型客户端与默认提示词模板。"""
        self._chat_model_factory = chat_model_factory
        self._prompt_template = PromptTemplate.from_template("{prompt}")
        self._output_parser = StrOutputParser()
        self._guard_adapter = guard_adapter or ModelGuardAdapter()
        self._schema_guard = schema_guard or JsonSchemaGuard()

    def build_chat_model(self, routed_model: RoutedModel) -> BaseChatModel:
        """根据路由结果构造 LangChain `BaseChatModel` 实例。"""
        if not routed_model.api_key:
            raise ValueError(f"Missing API key for model complexity: {routed_model.complexity}")

        chat_model_cls = self._resolve_chat_model_factory()
        model_kwargs: dict[str, Any] = {
            "model": routed_model.model_name,
            "api_key": routed_model.api_key,
            "base_url": routed_model.api_base,
            "timeout": routed_model.timeout_seconds,
            "temperature": routed_model.temperature,
            "max_tokens": routed_model.max_tokens,
        }
        extra_body = self._build_provider_extra_body(routed_model)
        if extra_body:
            model_kwargs["extra_body"] = extra_body

        chat_model = chat_model_cls(**model_kwargs)
        if not isinstance(chat_model, BaseChatModel):
            raise TypeError("Configured chat model factory must return a LangChain BaseChatModel instance")
        return chat_model

    def _build_provider_extra_body(self, routed_model: RoutedModel) -> dict[str, Any]:
        """为 OpenAI-compatible provider 补充供应商特定参数。"""
        provider = str(getattr(routed_model, "provider", "")).strip().lower()
        model_name = routed_model.model_name.strip().lower()
        if provider == "dashscope" and model_name.startswith("qwen3"):
            return {"enable_thinking": False}
        return {}

    def build_chat_model_for_complexity(self, complexity: TaskComplexity) -> BaseChatModel:
        """按任务复杂度路由并返回 LangChain 聊天模型。"""
        routed_model = get_model_for_task(complexity)
        return self.build_chat_model(routed_model)

    def get_chat_model(self, complexity: TaskComplexity = "simple") -> BaseChatModel:
        """返回可直接给 LangChain Agent 或 LangGraph 使用的聊天模型。"""
        return self.build_chat_model_for_complexity(complexity)

    def get_runnable(
        self,
        complexity: TaskComplexity = "simple",
        prompt_template: BasePromptTemplate | None = None,
        *,
        output_parser: Runnable[Any, Any] | None = None,
    ) -> RunnableSerializable[Any, Any]:
        """返回可组合的 LangChain runnable。

        未传入 `prompt_template` 时直接返回 `BaseChatModel`，作为 LCEL 可继续组合的 runnable。
        传入模板后返回 `prompt -> model -> parser` 链。
        """
        chat_model = self.get_chat_model(complexity)
        if prompt_template is None:
            return chat_model

        parser = output_parser or self._output_parser
        return prompt_template | chat_model | parser

    def invoke_runnable(
        self,
        runnable: Runnable[Any, Any],
        input: Any,
        *,
        config: Any | None = None,
        complexity: TaskComplexity | str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """执行 runnable，并通过共享模型 guard 处理重试、空输出和失败分类。"""
        return self._guard_adapter.invoke(
            lambda: runnable.invoke(input, config=config),
            config=self._build_guard_config(complexity=complexity),
            metadata=metadata,
        )

    def stream_runnable(
        self,
        runnable: Runnable[Any, Any],
        input: Any,
        *,
        config: Any | None = None,
        complexity: TaskComplexity | str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """流式执行 runnable，并通过共享模型 guard 过滤空 chunk。"""
        yield from self._guard_adapter.stream(
            lambda: runnable.stream(input, config=config),
            config=self._build_guard_config(complexity=complexity),
            metadata=metadata,
        )

    def invoke_json_schema(
        self,
        runnable: Runnable[Any, Any],
        input: Any,
        *,
        schema_model: type[BaseModel],
        schema_source: str,
        config: Any | None = None,
        complexity: TaskComplexity | str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        """执行模型并校验结构化 JSON 输出。"""
        raw_output = self.invoke_runnable(
            runnable,
            input,
            config=config,
            complexity=complexity,
            metadata={**dict(metadata or {}), "schema_source": schema_source},
        )
        return self._schema_guard.validate(
            raw_output,
            schema_model=schema_model,
            source=schema_source,
            metadata=metadata,
        )

    def _resolve_chat_model_factory(self) -> Callable[..., Any]:
        """延迟解析 ChatOpenAI 工厂，支持依赖注入。"""
        if self._chat_model_factory is not None:
            return self._chat_model_factory

        try:
            from langchain_openai import ChatOpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "langchain-openai is required for model execution. "
                "Install backend/requirements.txt and retry."
            ) from exc

        self._chat_model_factory = ChatOpenAI
        return ChatOpenAI

    def invoke_template(
        self,
        prompt_template: BasePromptTemplate,
        variables: dict[str, Any],
        complexity: TaskComplexity = "simple",
    ) -> str:
        """兼容 helper：基于模板执行一次同步文本调用。"""
        runnable = self.get_runnable(
            complexity=complexity,
            prompt_template=prompt_template,
            output_parser=self._output_parser,
        )
        return str(self.invoke_runnable(runnable, variables)).strip()

    def invoke(self, prompt: str, complexity: TaskComplexity = "simple") -> str:
        """使用默认模板执行一次非流式文本调用。"""
        return self.invoke_template(
            prompt_template=self._prompt_template,
            variables={"prompt": prompt},
            complexity=complexity,
        )

    def stream_template(
        self,
        prompt_template: BasePromptTemplate,
        variables: dict[str, Any],
        complexity: TaskComplexity = "simple",
    ) -> Iterator[str]:
        """兼容 helper：基于模板执行一次流式文本调用。"""
        routed_model = get_model_for_task(complexity)
        if not routed_model.supports_streaming:
            raise ValueError(f"Streaming is not supported for model complexity: {routed_model.complexity}")

        runnable = self.get_runnable(
            complexity=routed_model.complexity,
            prompt_template=prompt_template,
            output_parser=self._output_parser,
        )
        for chunk in self.stream_runnable(runnable, variables):
            yield str(chunk)

    def stream(self, prompt: str, complexity: TaskComplexity = "simple") -> Iterator[str]:
        """以流式方式输出模型生成的文本片段。"""
        yield from self.stream_template(
            prompt_template=self._prompt_template,
            variables={"prompt": prompt},
            complexity=complexity,
        )

    def _build_guard_config(self, *, complexity: TaskComplexity | str) -> ModelGuardConfig:
        return ModelGuardConfig(complexity=str(complexity))


model_client = ModelClient()


def get_chat_model(complexity: TaskComplexity = "simple") -> BaseChatModel:
    """模块级快捷入口，返回 LangChain 聊天模型。"""
    return model_client.get_chat_model(complexity)


def get_runnable(
    complexity: TaskComplexity = "simple",
    prompt_template: BasePromptTemplate | None = None,
    *,
    output_parser: Runnable[Any, Any] | None = None,
) -> RunnableSerializable[Any, Any]:
    """模块级快捷入口，返回 LangChain runnable。"""
    return model_client.get_runnable(
        complexity=complexity,
        prompt_template=prompt_template,
        output_parser=output_parser,
    )


def invoke_runnable(
    runnable: Runnable[Any, Any],
    input: Any,
    *,
    config: Any | None = None,
    complexity: TaskComplexity | str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """模块级快捷入口，执行 runnable 并应用统一空结果保护。"""
    return model_client.invoke_runnable(
        runnable,
        input,
        config=config,
        complexity=complexity,
        metadata=metadata,
    )


def stream_runnable(
    runnable: Runnable[Any, Any],
    input: Any,
    *,
    config: Any | None = None,
    complexity: TaskComplexity | str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """模块级快捷入口，流式执行 runnable 并过滤空 chunk。"""
    yield from model_client.stream_runnable(
        runnable,
        input,
        config=config,
        complexity=complexity,
        metadata=metadata,
    )
