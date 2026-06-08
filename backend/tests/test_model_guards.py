from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from backend.platform.agent_runtime.failures import FailureCategory
from backend.platform.models.llm.client import ModelClient
from backend.platform.models.llm.guards import ModelGuardFailureError, ModelSchemaValidationError


class _StructuredOutput(BaseModel):
    query: str = Field(min_length=1)
    reason: str = ""


def test_model_guard_classifies_empty_output() -> None:
    client = ModelClient()
    runnable = RunnableLambda(lambda _: "")

    with pytest.raises(ModelGuardFailureError) as exc_info:
        client.invoke_runnable(runnable, {"query": "x"}, complexity="simple")

    payload = exc_info.value.failure_payload
    assert payload["category"] == FailureCategory.MODEL_EMPTY_OUTPUT
    assert payload["retryable"] is False
    assert payload["metadata"]["call_method"] == "invoke"
    assert payload["metadata"]["complexity"] == "simple"


def test_model_schema_guard_classifies_invalid_json() -> None:
    client = ModelClient()
    runnable = RunnableLambda(lambda _: "{invalid json")

    with pytest.raises(ModelSchemaValidationError) as exc_info:
        client.invoke_json_schema(
            runnable,
            {"query": "x"},
            schema_model=_StructuredOutput,
            schema_source="query_rewrite",
            complexity="simple",
        )

    payload = exc_info.value.failure_payload
    assert payload["category"] == FailureCategory.MODEL_SCHEMA_ERROR
    assert payload["retryable"] is False
    assert payload["metadata"]["schema_source"] == "query_rewrite"


def test_model_guard_retries_retryable_model_exception() -> None:
    attempts = 0

    def _flaky(_: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary model connection")
        return "ok"

    client = ModelClient()
    result = client.invoke_runnable(RunnableLambda(_flaky), {"query": "x"}, complexity="simple")

    assert result == "ok"
    assert attempts == 2


def test_model_guard_classifies_non_retryable_model_exception() -> None:
    client = ModelClient()
    runnable = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("model exploded")))

    with pytest.raises(ModelGuardFailureError) as exc_info:
        client.invoke_runnable(runnable, {"query": "x"}, complexity="complex")

    payload = exc_info.value.failure_payload
    assert payload["category"] == FailureCategory.MODEL_ERROR
    assert payload["retryable"] is False
    assert payload["exception_type"] == "RuntimeError"
    assert payload["metadata"]["complexity"] == "complex"
