from __future__ import annotations

from pathlib import Path

import pytest

from jb_hackathon_data.io_utils import ExtractionInputError, iter_raw_documents
from jb_hackathon_data.merge_extraction_results import merge_rows
from jb_hackathon_data.models import ExtractedDocument, Route, SourceId, Status
from jb_hackathon_data.validate_extraction_results import validate_extraction_rows


def _make_document(
    source_path: str,
    *,
    status: Status = "success",
    route: Route = "pdf_text_layer",
    extension: str = ".pdf",
    text: str = "content",
    error: str | None = None,
) -> ExtractedDocument:
    return ExtractedDocument(
        source_id=SourceId(source_path),
        source_path=source_path,
        source_sha256="sha",
        extension=extension,
        route=route,
        parser_name="parser",
        parser_version="1.0",
        status=status,
        text=text,
        pages_or_sections=(),
        error=error,
    )


def _make_raw_docs(count: int) -> tuple[Path, ...]:
    return tuple(Path(f"raw-{idx}.pdf") for idx in range(count))


def _make_rows(raw_paths: tuple[Path, ...]) -> tuple[ExtractedDocument, ...]:
    return tuple(
        _make_document(
            source_path=f"doc-{idx}.pdf",
            route="pdf_text_layer",
            text="ok",
            error=None,
        )
        for idx, _ in enumerate(raw_paths)
    )


def test_merge_rows_prefers_structured_ocr_over_pdf_text() -> None:
    source_path = "docs/sample.pdf"
    rows = (
        _make_document(source_path=source_path, route="pdf_text_layer", text="text"),
        _make_document(source_path=source_path, route="pdf_structured_ocr", text="structured"),
    )

    merged = merge_rows(rows)
    assert len(merged) == 1
    assert merged[0].route == "pdf_structured_ocr"


def test_validate_extraction_rows_rejects_duplicate_source_path_with_expected_error() -> None:
    raw_docs = _make_raw_docs(23)
    duplicate_rows: list[ExtractedDocument] = []
    duplicate_rows.extend(
        [
            _make_document(source_path="doc-0.pdf", route="pdf_text_layer"),
            _make_document(source_path="doc-0.pdf", route="pdf_structured_ocr"),
        ]
    )
    duplicate_rows.extend(_make_rows(tuple(raw_docs[2:23])))

    with pytest.raises(
        ExtractionInputError,
        match="duplicate source_path in extraction results",
    ):
        validate_extraction_rows(raw_docs, tuple(duplicate_rows))


def test_iter_raw_documents_includes_docling_supported_formats(tmp_path: Path) -> None:
    for filename in ("a.pdf", "b.docx", "c.pptx", "d.xlsx", "e.html", "f.png", "g.hwp", "ignored.txt"):
        _ = (tmp_path / filename).write_bytes(b"")

    documents = iter_raw_documents(tmp_path)

    assert tuple(path.name for path in documents) == ("a.pdf", "b.docx", "c.pptx", "d.xlsx", "e.html", "f.png", "g.hwp")

