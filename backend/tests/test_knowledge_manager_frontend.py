from __future__ import annotations

from pathlib import Path


PAGE = Path(__file__).resolve().parents[2] / "frontend" / "knowledge-manager.html"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_knowledge_manager_has_processing_dialog_contract() -> None:
    html = _page()

    assert 'id="processingCard"' in html
    assert 'id="processingRules"' in html
    assert 'id="originalSamples"' in html
    assert 'id="processedSamples"' in html
    assert 'id="processingConfig"' in html
    assert 'id="submitProcessingButton"' in html
    assert 'id="rechunkButton"' in html
    assert 'data-action="reprocess"' in html
    assert 'data-action="rechunk"' in html
    assert 'data-action="index"' in html


def test_knowledge_manager_upload_opens_preprocess_preview() -> None:
    html = _page()

    assert "await openProcessingDialog({" in html
    assert 'mode: "create"' in html
    assert 'apiRequest("/knowledge/documents/preprocess-preview"' in html
    assert "isSupportedUpload(data)" in html
    assert "preview.chunk_size" in html
    assert "preview.chunk_overlap" in html
    assert 'id="chunkSizeInput"' in html
    assert 'id="chunkOverlapInput"' in html
    assert "function readProcessingConfig" in html
    assert "chunk_size: payload.chunk_size" in html
    assert "chunk_overlap: payload.chunk_overlap" in html
    assert html.index('id="chunkSizeInput"') > html.index('id="processingCard"')


def test_knowledge_manager_rule_toggle_refreshes_preview_and_submit_updates_state() -> None:
    html = _page()

    assert 'refs.processingRules.addEventListener("change"' in html
    assert "state.processing.selectedRules = Array.from" in html
    assert "refreshProcessingPreview();" in html
    assert "function splitPreviewChunks" in html
    assert 'apiRequest("/knowledge/documents", {' in html
    assert '"/reprocess"' in html
    assert '"/rechunk"' in html
    assert "await refreshFiles(false);" in html


def test_knowledge_manager_places_processing_actions_together_and_removes_full_document_button() -> None:
    html = _page()

    actions_start = html.index('<div class="processing-actions">')
    actions_end = html.index("</div>", actions_start)
    actions_html = html[actions_start:actions_end]

    assert 'id="refreshPreviewButton"' in actions_html
    assert 'id="submitProcessingButton"' in actions_html
    assert actions_html.index('id="refreshPreviewButton"') < actions_html.index('id="submitProcessingButton"')
    assert "查看完整文档" not in html


def test_knowledge_manager_displays_processing_metadata_and_awaiting_state() -> None:
    html = _page()

    assert '["awaiting_processing"].includes(value)' in html
    assert "formatRules(document.processing_rules)" in html
    assert "formatStats(document.processing_stats)" in html
    assert "version.processing_rules" in html
    assert "version.processing_stats" in html
