from pathlib import Path
import shutil
from uuid import uuid4

import pytest


TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
ARTIFACTS_DIR = TESTS_DIR / "artifacts"


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
