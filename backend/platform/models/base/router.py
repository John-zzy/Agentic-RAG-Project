from dataclasses import dataclass
from typing import Literal

from backend.platform.config.settings import ModelEndpointConfig, settings

TaskComplexity = Literal["simple", "moderate", "complex"]


@dataclass(frozen=True)
class RoutedModel:
    """描述一次路由后的模型配置，供调用层直接消费。"""

    complexity: TaskComplexity
    provider: str
    model_name: str
    api_base: str | None
    api_key: str | None
    supports_streaming: bool
    timeout_seconds: int
    max_tokens: int
    temperature: float

    @classmethod
    def from_config(
        cls,
        complexity: TaskComplexity,
        config: ModelEndpointConfig,
    ) -> "RoutedModel":
        """将配置对象转换为可直接路由的模型描述。"""
        return cls(
            complexity=complexity,
            provider=config.provider,
            model_name=config.model_name,
            api_base=config.api_base,
            api_key=config.api_key,
            supports_streaming=config.supports_streaming,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )


class ModelRouter:
    """维护复杂度到具体模型配置的选择逻辑。"""

    def __init__(self) -> None:
        """加载全局模型路由配置。"""
        self._config = settings.models

    def select(self, complexity: TaskComplexity) -> RoutedModel:
        """按复杂度选择模型配置。"""
        config = getattr(self._config, complexity)
        return RoutedModel.from_config(complexity, config)

    def route_by_complexity(self, complexity: TaskComplexity) -> RoutedModel:
        """复杂度路由的别名方法。"""
        return self.select(complexity)


router = ModelRouter()


def get_model_for_task(complexity: TaskComplexity) -> RoutedModel:
    """对外暴露的模型路由入口。"""
    return router.select(complexity)
