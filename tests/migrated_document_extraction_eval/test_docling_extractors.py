from __future__ import annotations

from pathlib import Path

import pytest
from jb_hackathon_data.docling_extractors import pdf_pipeline_options
from jb_hackathon_data.models import DocumentBlock


def test_primary_pdf_pipeline_keeps_ocr_off_until_needed() -> None:
    options = pdf_pipeline_options(do_ocr=False)

    assert options.do_ocr is False
    assert options.do_table_structure is True


def test_extract_docling_document_writes_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = tmp_path / "sample.pdf"
    document_path.write_bytes(b"fake")
    sidecar_dir = tmp_path / "docling_documents"

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "fallback markdown"

        def save_as_json(self, filename: str | Path, **_: object) -> None:
            Path(filename).write_text('{"doc": "saved"}', encoding="utf-8")

    def fake_blocks(_: FakeDocument, *, source_path: str) -> tuple[DocumentBlock, ...]:
        return (
            DocumentBlock(
                block_id=f"{source_path}:0",
                type="paragraph",
                text="문서 본문",
                page_no=1,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(("docling_ref", "#/texts/0"),),
            ),
        )

    import jb_hackathon_data.docling_extractors as module

    monkeypatch.setattr(module, "_convert_with_docling", lambda path, *, do_ocr: FakeDocument())
    monkeypatch.setattr(module, "canonical_blocks_from_docling", fake_blocks)

    row = module.extract_docling_document(document_path, sidecar_dir=sidecar_dir)

    assert row.status == "success"
    assert row.docling_document_path is not None
    assert (sidecar_dir / f"{row.source_id}.json").exists()

