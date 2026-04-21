from collections.abc import Iterator

from openai import OpenAI

from backend.models.router import RoutedModel, get_model_for_task


class ModelClient:
    def build_client(self, routed_model: RoutedModel) -> OpenAI:
        if not routed_model.api_key:
            raise ValueError(f"Missing API key for model complexity: {routed_model.complexity}")

        return OpenAI(
            api_key=routed_model.api_key,
            base_url=routed_model.api_base,
            timeout=routed_model.timeout_seconds,
        )

    def _create_completion(
        self,
        prompt: str,
        routed_model: RoutedModel,
        *,
        stream: bool,
    ):
        client = self.build_client(routed_model)
        return client.chat.completions.create(
            model=routed_model.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=routed_model.temperature,
            max_tokens=routed_model.max_tokens,
            stream=stream,
        )

    def invoke(self, prompt: str, complexity: str = "simple") -> str:
        routed_model = get_model_for_task(complexity)  # type: ignore[arg-type]
        response = self._create_completion(prompt, routed_model, stream=False)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned empty content")
        return content.strip()

    def stream(self, prompt: str, complexity: str = "simple") -> Iterator[str]:
        routed_model = get_model_for_task(complexity)  # type: ignore[arg-type]
        if not routed_model.supports_streaming:
            raise ValueError(f"Streaming is not supported for model complexity: {routed_model.complexity}")

        response_stream = self._create_completion(prompt, routed_model, stream=True)
        yielded = False
        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content
            if not content:
                continue
            yielded = True
            yield content

        if not yielded:
            raise ValueError("Model returned empty streaming content")


model_client = ModelClient()
