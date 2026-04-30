from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate

from backend.models.base.router import RoutedModel, TaskComplexity, get_model_for_task


class ModelClient:
    """封装基于路由结果的聊天模型调用与流式输出能力。"""

    def __init__(self, chat_model_factory: Callable[..., Any] | None = None) -> None:
        """初始化模型客户端与默认提示模板。"""
        self._chat_model_factory = chat_model_factory
        self._prompt_template = PromptTemplate.from_template("{prompt}")
        self._output_parser = StrOutputParser()

    def build_chat_model(self, routed_model: RoutedModel) -> Any:
        """根据路由信息实例化聊天模型。"""
        if not routed_model.api_key:
            raise ValueError(f"Missing API key for model complexity: {routed_model.complexity}")

        chat_model_cls = self._resolve_chat_model_factory()
        return chat_model_cls(
            model=routed_model.model_name,
            api_key=routed_model.api_key,
            base_url=routed_model.api_base,
            timeout=routed_model.timeout_seconds,
            temperature=routed_model.temperature,
            max_tokens=routed_model.max_tokens,
        )

    def build_chat_model_for_complexity(self, complexity: TaskComplexity) -> Any:
        """按任务复杂度选择并构建模型。"""
        routed_model = get_model_for_task(complexity)
        return self.build_chat_model(routed_model)

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

    def _build_chain(self, routed_model: RoutedModel):
        """构造 prompt -> model -> parser 的调用链。"""
        chat_model = self.build_chat_model(routed_model)
        return self._prompt_template | chat_model | self._output_parser

    def invoke_template(
        self,
        prompt_template: BasePromptTemplate,
        variables: dict[str, Any],
        complexity: TaskComplexity = "simple",
    ) -> str:
        """使用指定模板同步调用模型并返回文本结果。"""
        routed_model = get_model_for_task(complexity)
        chat_model = self.build_chat_model(routed_model)
        chain = prompt_template | chat_model | self._output_parser
        content = chain.invoke(variables)
        if not content:
            raise ValueError("Model returned empty content")
        return str(content).strip()

    def invoke(self, prompt: str, complexity: TaskComplexity = "simple") -> str:
        """使用默认模板执行一次非流式调用。"""
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

        chain = self._build_chain(routed_model)
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
