import pytest

from backend.models.llm.client import model_client


@pytest.mark.integration
def test_model_invocation_returns_text_for_hello() -> None:
    response = model_client.invoke("hello", complexity="simple")

    assert isinstance(response, str)
    assert response.strip() != ""


@pytest.mark.integration
def test_model_stream_returns_text_chunks_for_hello() -> None:
    chunks = list(model_client.stream("hello", complexity="simple"))
    response = "".join(chunks).strip()

    assert chunks
    assert isinstance(response, str)
    assert response != ""
