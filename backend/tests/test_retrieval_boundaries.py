from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iter_python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if path.is_file()]


def test_platform_rag_does_not_import_knowledge_base_modules() -> None:
    rag_root = PROJECT_ROOT / "platform" / "rag"
    forbidden = (
        "backend.platform.knowledge.base.store",
        "backend.platform.knowledge.base.relevance",
        "backend.platform.knowledge.base.",
    )

    offenders: list[str] = []
    for path in _iter_python_files(rag_root):
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_platform_retrieval_does_not_import_knowledge_or_rag_modules() -> None:
    retrieval_root = PROJECT_ROOT / "platform" / "retrieval"
    forbidden = (
        "backend.platform.knowledge.",
        "backend.platform.rag.",
    )

    offenders: list[str] = []
    for path in _iter_python_files(retrieval_root):
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
