from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ModelGuardError(Exception):
    """模型 guard 的统一异常基类，携带稳定 failure payload。"""

    def __init__(self, message: str, *, failure_payload: Mapping[str, Any]) -> None:
        self.failure_payload = dict(failure_payload)
        super().__init__(message)


class ModelGuardFailureError(ModelGuardError, ValueError):
    """模型调用失败或输出为空。"""


class ModelSchemaValidationError(ModelGuardError, ValueError):
    """模型结构化输出不满足 schema。"""
