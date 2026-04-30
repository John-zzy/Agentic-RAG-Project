from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable

from backend.models.base.router import RoutedModel, TaskComplexity, get_model_for_task


class ModelClient:
    """封装基于复杂度路由的 LangChain 聊天模型入口。"""

    def __init__(self, chat_model_factory: Callable[..., Any] | None = None) -> None:
        """初始化模型客户端与默认提示词模板。"""
        self._chat_model_factory = chat_model_factory
        self._prompt_template = PromptTemplate.from_template("{prompt}")
        self._output_parser = StrOutputParser()

    def build_chat_model(self, routed_model: RoutedModel) -> BaseChatModel:
        """根据路由结果构造 LangChain `BaseChatModel` 实例。"""
        if not routed_model.api_key:
            raise ValueError(f"Missing API key for model complexity: {routed_model.complexity}")

        chat_model_cls = self._resolve_chat_model_factory()
        chat_model = chat_model_cls(
            model=routed_model.model_name,
            api_key=routed_model.api_key,
            base_url=routed_model.api_base,
            timeout=routed_model.timeout_seconds,
            temperature=routed_model.temperature,
            max_tokens=routed_model.max_tokens,
        )
        if not isinstance(chat_model, BaseChatModel):
            raise TypeError("Configured chat model factory must return a LangChain BaseChatModel instance")
        return chat_model

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

        未传入 `prompt_template` 时直接返回 `BaseChatModel`，方便后续绑定 tools。
        传入模板后返回 `prompt -> model -> parser` 链。
        """
        chat_model = self.get_chat_model(complexity)
        if prompt_template is None:
            return chat_model

        parser = output_parser or self._output_parser
        return prompt_template | chat_model | parser

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

    def _build_text_chain(self, routed_model: RoutedModel) -> RunnableSerializable[dict[str, Any], str]:
        """构造兼容旧字符串 API 的文本输出链。"""
        return self.get_runnable(
            complexity=routed_model.complexity,
            prompt_template=self._prompt_template,
            output_parser=self._output_parser,
        )

    def invoke_template(
        self,
        prompt_template: BasePromptTemplate,
        variables: dict[str, Any],
        complexity: TaskComplexity = "simple",
    ) -> str:
        """使用指定模板同步调用模型，并兼容返回纯文本。"""
        chain = self.get_runnable(
            complexity=complexity,
            prompt_template=prompt_template,
            output_parser=self._output_parser,
        )
        content = chain.invoke(variables)
        if not content:
            raise ValueError("Model returned empty content")
        return str(content).strip()

    def invoke(self, prompt: str, complexity: TaskComplexity = "simple") -> str:
        """使用默认模板执行一次非流式文本调用。"""
        return self.invoke_template(
            prompt_template=self._prompt_template,
            variables={"prompt": prompt},
            complexity=complexity,
        )

    def stream(self, prompt: str, complexity: TaskComplexity = "simple") -> Iterator[str]:
        """以流式方式输出模型生成的文本片段。"""
        routed_model = get_model_for_task(complexity)
        if not routed_model.supports_streaming:
            raise ValueError(f"Streaming is not supported for model complexity: {routed_model.complexity}")

        chain = self._build_text_chain(routed_model)
        yielded = False
        for chunk in chain.stream({"prompt": prompt}):
            text = str(chunk).strip()
            if not text:
                continue
            yielded = True
            yield text

        if not yielded:
            raise ValueError("Model returned empty streaming content")


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
