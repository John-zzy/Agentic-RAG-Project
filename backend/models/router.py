from dataclasses import dataclass
from typing import Literal

from backend.config.settings import ModelEndpointConfig, settings

TaskComplexity = Literal["simple", "moderate", "complex"]


@dataclass(frozen=True)
class RoutedModel:
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
    def __init__(self) -> None:
        self._config = settings.models

    def select(self, complexity: TaskComplexity) -> RoutedModel:
        config = getattr(self._config, complexity)
        return RoutedModel.from_config(complexity, config)

    def route_by_complexity(self, complexity: TaskComplexity) -> RoutedModel:
        return self.select(complexity)


router = ModelRouter()


def get_model_for_task(complexity: TaskComplexity) -> RoutedModel:
    return router.select(complexity)
