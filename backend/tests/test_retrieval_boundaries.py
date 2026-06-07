from __future__ import annotations

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def _iter_python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if path.is_file()]


def _iter_project_python_files() -> list[Path]:
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        if "openspec" in path.parts and "changes" in path.parts:
            continue
        files.append(path)
    return files


def test_platform_rag_does_not_import_knowledge_base_modules() -> None:
    rag_root = PROJECT_ROOT / "platform" / "rag"
    forbidden = (
        "backend.platform.knowledge.base.store",
        "backend.platform.knowledge.base.relevance",
        "backend.platform.knowledge.repositories",
        "backend.platform.knowledge.documents.store_support",
        "backend.platform.knowledge.base.",
    )

    offenders: list[str] = []
    for path in _iter_python_files(rag_root):
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_platform_search_foundation_does_not_import_knowledge_rag_or_scene_modules() -> None:
    search_foundation_root = PROJECT_ROOT / "platform" / "search_foundation"
    forbidden = (
        "backend.platform.knowledge.",
        "backend.platform.rag.",
        "backend.scenes.",
    )

    offenders: list[str] = []
    for path in _iter_python_files(search_foundation_root):
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_new_modular_rag_stage_imports_are_canonical() -> None:
    modules = (
        "backend.platform.rag.contracts",
        "backend.platform.rag.pre_retrieval.query_rewrite",
        "backend.platform.rag.retrieval.documents.service",
        "backend.platform.rag.retrieval.documents.types",
        "backend.platform.rag.retrieval.documents.semantic",
        "backend.platform.rag.retrieval.documents.keyword",
        "backend.platform.rag.retrieval.documents.keyword_scoring",
        "backend.platform.rag.retrieval.documents.fusion",
        "backend.platform.rag.retrieval.documents.embedding",
        "backend.platform.rag.retrieval.documents.filters",
        "backend.platform.rag.post_retrieval.rerank",
        "backend.platform.rag.orchestration.decisions",
        "backend.platform.rag.orchestration.agentic",
        "backend.platform.rag.orchestration.retrieval_graph",
    )

    imported = {module: importlib.import_module(module) for module in modules}

    assert hasattr(imported["backend.platform.rag.contracts"], "RetrievalTool")
    assert hasattr(imported["backend.platform.rag.pre_retrieval.query_rewrite"], "QueryRewrite")
    assert hasattr(imported["backend.platform.rag.retrieval.documents.service"], "DocumentRetrievalService")
    assert hasattr(imported["backend.platform.rag.post_retrieval.rerank"], "IdentityRetrievalReranker")
    assert hasattr(imported["backend.platform.rag.orchestration.decisions"], "SufficiencyDecision")
    assert hasattr(imported["backend.platform.rag.orchestration.agentic"], "AgenticRetriever")
    assert hasattr(imported["backend.platform.rag.orchestration.retrieval_graph"], "build_agentic_rag_graph")


def test_agentic_rag_retrieval_graph_does_not_keep_legacy_langgraph_package() -> None:
    legacy_package = PROJECT_ROOT / "platform" / "rag" / "orchestration" / "langgraph"

    assert not legacy_package.exists()


def test_project_code_does_not_import_legacy_root_rag_modules() -> None:
    old_module_names = (
        "core",
        "agentic",
        "rerank",
        "document_retrieval",
        "document_retrieval_embedding",
        "document_retrieval_fusion",
        "document_retrieval_keyword",
        "document_retrieval_keyword_scoring",
        "document_retrieval_rules",
        "document_retrieval_semantic",
        "document_retrieval_service",
        "document_retrieval_types",
    )
    legacy_import_tokens = tuple(
        f"{prefix}{module}"
        for module in old_module_names
        for prefix in (
            "from backend.platform.rag.",
            "import backend.platform.rag.",
        )
    )

    offenders: list[str] = []
    for path in _iter_project_python_files():
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in legacy_import_tokens):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
