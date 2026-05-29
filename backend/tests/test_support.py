from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from backend.platform.search_foundation import LocalHashingEmbedder


TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
ARTIFACTS_DIR = TESTS_DIR / "artifacts"


class FakeEmbeddingStrategy:
    """测试用 embedding，避免单元测试访问真实模型服务。"""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.calls: list[str] = []
        self._embedder = LocalHashingEmbedder(dimensions=dimensions)

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._embedder.embed(text)


def make_test_runtime_dir(name: str) -> Path:
    path = ARTIFACTS_DIR / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def tmp_path() -> Path:
    """提供项目内临时目录，替代当前 pytest 配置禁用的内置 tmp_path。"""
    path = make_test_runtime_dir("tmp-path")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
