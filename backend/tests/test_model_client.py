from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from backend.platform.models.llm.client import ModelClient


class _FakeChatModel(SimpleChatModel):
    response_text: str = "mock-response"
    stream_chunks: tuple[str, ...] = ("mock-response",)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        del messages, stop, run_manager, kwargs
        return self.response_text

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        del messages, stop, run_manager, kwargs
        for chunk in self.stream_chunks:
            yield AIMessage(content=chunk)


def _build_client(
    *,
    response_text: str = "mock-response",
    stream_chunks: tuple[str, ...] = ("mock-response",),
) -> ModelClient:
    def _factory(**kwargs: Any) -> BaseChatModel:
        del kwargs
        return _FakeChatModel(response_text=response_text, stream_chunks=stream_chunks)

    return ModelClient(chat_model_factory=_factory)


def test_get_chat_model_returns_base_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()
    monkeypatch.setattr(
        "backend.platform.models.llm.client.get_model_for_task",
        lambda complexity: type(
            "RoutedModelStub",
            (),
            {
                "complexity": complexity,
                "api_key": "test-key",
                "model_name": "fake-model",
                "api_base": "https://example.test",
                "timeout_seconds": 30,
                "temperature": 0.1,
                "max_tokens": 128,
                "supports_streaming": True,
            },
        )(),
    )

    model = client.get_chat_model("simple")

    assert isinstance(model, BaseChatModel)


def test_dashscope_qwen3_chat_model_disables_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_kwargs: dict[str, Any] = {}

    def _factory(**kwargs: Any) -> BaseChatModel:
        observed_kwargs.update(kwargs)
        return _FakeChatModel()

    client = ModelClient(chat_model_factory=_factory)
    monkeypatch.setattr(
        "backend.platform.models.llm.client.get_model_for_task",
        lambda complexity: type(
            "RoutedModelStub",
            (),
            {
                "complexity": complexity,
                "provider": "dashscope",
                "api_key": "test-key",
                "model_name": "qwen3.6-plus",
                "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "timeout_seconds": 30,
                "temperature": 0.1,
                "max_tokens": 128,
                "supports_streaming": True,
            },
        )(),
    )

    client.get_chat_model("simple")

    assert observed_kwargs["extra_body"] == {"enable_thinking": False}


def test_get_runnable_without_prompt_returns_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()
    monkeypatch.setattr(
        "backend.platform.models.llm.client.get_model_for_task",
        lambda complexity: type(
            "RoutedModelStub",
            (),
            {
                "complexity": complexity,
                "api_key": "test-key",
                "model_name": "fake-model",
                "api_base": "https://example.test",
                "timeout_seconds": 30,
                "temperature": 0.1,
                "max_tokens": 128,
                "supports_streaming": True,
            },
        )(),
    )

    runnable = client.get_runnable("simple")

    assert isinstance(runnable, BaseChatModel)


def test_get_runnable_with_prompt_builds_prompt_model_parser_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client(response_text="hello from chain")
    monkeypatch.setattr(
        "backend.platform.models.llm.client.get_model_for_task",
        lambda complexity: type(
            "RoutedModelStub",
            (),
            {
                "complexity": complexity,
                "api_key": "test-key",
                "model_name": "fake-model",
                "api_base": "https://example.test",
                "timeout_seconds": 30,
                "temperature": 0.1,
                "max_tokens": 128,
                "supports_streaming": True,
            },
        )(),
    )
    prompt = PromptTemplate.from_template("say {topic}")

    runnable = client.get_runnable("simple", prompt_template=prompt)

    assert runnable.invoke({"topic": "hi"}) == "hello from chain"


def test_invoke_runnable_returns_non_empty_content() -> None:
    client = _build_client()
    runnable = RunnableLambda(lambda payload: f" answer:{payload['name']} ")

    result = client.invoke_runnable(runnable, {"name": "alice"})

    assert result == "answer:alice"


def test_invoke_runnable_raises_on_empty_content() -> None:
    client = _build_client()
    runnable = RunnableLambda(lambda payload: "")

    with pytest.raises(ValueError, match="Model returned empty content"):
        client.invoke_runnable(runnable, {"name": "alice"})


def test_stream_runnable_yields_non_empty_chunks() -> None:
    client = _build_client()

    def _stream(_: Any) -> Iterator[str]:
        yield ""
        yield "a"
        yield "b"

    runnable = RunnableLambda(_stream)

    chunks = list(client.stream_runnable(runnable, {"name": "alice"}))

    assert chunks == ["a", "b"]


def test_stream_runnable_raises_when_no_effective_chunks() -> None:
    client = _build_client()

    def _stream(_: Any) -> Iterator[str]:
        yield ""
        yield ""

    runnable = RunnableLambda(_stream)

    with pytest.raises(ValueError, match="Model returned empty streaming content"):
        list(client.stream_runnable(runnable, {"name": "alice"}))


def test_invoke_template_delegates_to_runnable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()
    prompt = PromptTemplate.from_template("say {topic}")
    observed: dict[str, Any] = {}

    def _fake_get_runnable(*args: Any, **kwargs: Any) -> RunnableLambda:
        observed["get_runnable"] = kwargs
        return RunnableLambda(lambda payload: f" delegated:{payload['topic']} ")

    def _fake_invoke_runnable(runnable: Any, payload: dict[str, Any]) -> str:
        observed["invoke_runnable"] = {"runnable": runnable, "payload": payload}
        return runnable.invoke(payload)

    monkeypatch.setattr(client, "get_runnable", _fake_get_runnable)
    monkeypatch.setattr(client, "invoke_runnable", _fake_invoke_runnable)

    result = client.invoke_template(prompt, {"topic": "hi"}, complexity="simple")

    assert result == "delegated:hi"
    assert observed["get_runnable"]["prompt_template"] == prompt
    assert observed["invoke_runnable"]["payload"] == {"topic": "hi"}


def test_stream_template_delegates_to_runnable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()
    prompt = PromptTemplate.from_template("say {topic}")
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        "backend.platform.models.llm.client.get_model_for_task",
        lambda complexity: type(
            "RoutedModelStub",
            (),
            {
                "complexity": complexity,
                "supports_streaming": True,
            },
        )(),
    )

    def _fake_get_runnable(*args: Any, **kwargs: Any) -> RunnableLambda:
        observed["get_runnable"] = kwargs
        return RunnableLambda(lambda payload: payload["topic"])

    def _fake_stream_runnable(runnable: Any, payload: dict[str, Any]) -> Iterator[str]:
        observed["stream_runnable"] = {"runnable": runnable, "payload": payload}
        yield ""
        yield "x"
        yield "y"

    monkeypatch.setattr(client, "get_runnable", _fake_get_runnable)
    monkeypatch.setattr(client, "stream_runnable", _fake_stream_runnable)

    chunks = list(client.stream_template(prompt, {"topic": "hi"}, complexity="simple"))

    assert chunks == ["", "x", "y"]
    assert observed["get_runnable"]["prompt_template"] == prompt
    assert observed["stream_runnable"]["payload"] == {"topic": "hi"}


def test_stream_template_rejects_unsupported_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_client()
    prompt = PromptTemplate.from_template("say {topic}")
    monkeypatch.setattr(
        "backend.platform.models.llm.client.get_model_for_task",
        lambda complexity: type(
            "RoutedModelStub",
            (),
            {
                "complexity": complexity,
                "supports_streaming": False,
            },
        )(),
    )

    with pytest.raises(ValueError, match="Streaming is not supported for model complexity: simple"):
        list(client.stream_template(prompt, {"topic": "hi"}, complexity="simple"))
