from __future__ import annotations

import json
from pathlib import Path
import re


ROOT_DIR = Path(__file__).resolve().parents[2]
EVALS_DIR = ROOT_DIR / "backend" / "evals"
SAMPLES_DIR = EVALS_DIR / "samples"
FIXTURES_DIR = EVALS_DIR / "fixtures"
FORBIDDEN_TERMS = ("ecommerce", "product", "inventory", "sku", "商品", "订单", "库存")
FORBIDDEN_ID_PATTERNS = (
    re.compile(r"\bP\d{3,}\b", re.IGNORECASE),
    re.compile(r"\bSKU\b", re.IGNORECASE),
    re.compile(r"\bO\d{6,}\b", re.IGNORECASE),
)


def test_minimal_eval_sample_manifest_is_well_formed() -> None:
    manifest = json.loads((SAMPLES_DIR / "minimal.json").read_text(encoding="utf-8"))

    assert manifest["sample_set"] == "minimal"
    assert manifest["namespace"] == "documents"
    assert isinstance(manifest["fixtures"], list) and manifest["fixtures"]
    assert isinstance(manifest["samples"], list) and manifest["samples"]

    fixture_ids = set()
    fixture_filenames = set()
    for fixture in manifest["fixtures"]:
        assert set(fixture) == {"id", "filename"}
        assert fixture["id"] not in fixture_ids
        fixture_ids.add(fixture["id"])
        fixture_filenames.add(fixture["filename"])
        assert (FIXTURES_DIR / fixture["filename"]).exists()
        normalized = f"{fixture['id']} {fixture['filename']}".lower()
        assert not any(term in normalized for term in FORBIDDEN_TERMS)

    sample_ids = set()
    for sample in manifest["samples"]:
        assert {"sample_id", "query", "source_doc", "expected"} <= set(sample)
        assert sample["sample_id"] not in sample_ids
        sample_ids.add(sample["sample_id"])
        normalized = sample["sample_id"].lower()
        assert not any(term in normalized for term in FORBIDDEN_TERMS)
        if sample["source_doc"] is not None:
            assert sample["source_doc"] in fixture_filenames
        assert isinstance(sample["expected"], dict)
        assert "knowledge_used" in sample["expected"]
        assert "min_citations" in sample["expected"]
        for pattern in FORBIDDEN_ID_PATTERNS:
            assert not pattern.search(sample["query"])


def test_eval_fixtures_remain_generic_document_assets() -> None:
    fixture_names = sorted(path.name for path in FIXTURES_DIR.glob("*.md"))

    assert fixture_names == [
        "eval-harness-it-policy.md",
        "eval-harness-quickstart.md",
        "eval-harness-support-faq.md",
    ]
    for name in fixture_names:
        assert not any(term in name.lower() for term in FORBIDDEN_TERMS)
