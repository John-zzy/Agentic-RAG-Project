from pathlib import Path
from uuid import uuid4


TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
ARTIFACTS_DIR = TESTS_DIR / "artifacts"


def make_test_runtime_dir(name: str) -> Path:
    path = ARTIFACTS_DIR / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
