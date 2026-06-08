from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from backend.platform.models.llm.guards.errors import ModelSchemaValidationError


SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


class JsonSchemaGuard:
    """校验模型 JSON 输出，阻止无效结构进入工具调用等副作用路径。"""

    def validate(
        self,
        raw_output: Any,
        *,
        schema_model: type[SchemaModelT],
        source: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> SchemaModelT:
        try:
            payload = self._load_json_object(raw_output)
            return schema_model.model_validate(payload)
        except ModelSchemaValidationError:
            raise
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise self._build_error(
                exc,
                source=source,
                metadata=metadata,
            ) from exc

    def _load_json_object(self, raw_output: Any) -> dict[str, Any]:
        if isinstance(raw_output, Mapping):
            return dict(raw_output)
        if hasattr(raw_output, "content"):
            return self._load_json_object(getattr(raw_output, "content"))
        text = str(raw_output or "").strip()
        if not text:
            raise ValueError("structured model output is empty.")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("structured model output must be a JSON object.")
        return payload

    def _build_error(
        self,
        exc: BaseException,
        *,
        source: str,
        metadata: Mapping[str, Any] | None,
    ) -> ModelSchemaValidationError:
        from backend.platform.agent_runtime.quality.failures import FailureCategory, FailureRecord

        record = FailureRecord(
            category=FailureCategory.MODEL_SCHEMA_ERROR,
            retryable=False,
            message=_schema_error_message(exc, source=source),
            source="model",
            exception_type=exc.__class__.__name__,
            metadata={
                "schema_source": source,
                **dict(metadata or {}),
            },
        )
        return ModelSchemaValidationError(record.message, failure_payload=record.to_payload())


def _schema_error_message(exc: BaseException, *, source: str) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"{source} output must be a valid JSON object."
    if isinstance(exc, ValidationError):
        messages: list[str] = []
        for error in exc.errors():
            field = ".".join(str(item) for item in error.get("loc", ())) or source
            messages.append(f"{field}: {error.get('msg', 'invalid value')}")
        detail = "; ".join(messages) if messages else "invalid structured output"
        return f"{source} output is invalid: {detail}."
    return f"{source} output is invalid: {exc}"
