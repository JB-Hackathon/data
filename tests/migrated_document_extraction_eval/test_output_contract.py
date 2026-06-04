from __future__ import annotations

from pathlib import Path

import json

import pytest

from jb_hackathon_data.io_utils import ExtractionInputError
from jb_hackathon_data.models import BoundingBox, DocumentBlock, ExtractedDocument, SourceId, Status
from jb_hackathon_data.parse_models import parse_extracted_document
from jb_hackathon_data.validate_extraction_results import (
    validate_extraction_rows,
    validate_output_directory,
)


def _touch(path: Path, content: str = "") -> None:
    _ = path.write_text(content, encoding="utf-8")


def _make_raw_docs(count: int) -> tuple[Path, ...]:
    return tuple(Path(_source_path_for_idx(idx)) for idx in range(count))


def _make_row(
    source_path: str,
    *,
    extension: str = ".pdf",
    status: Status = "success",
    text: str = "ok",
) -> ExtractedDocument:
    return ExtractedDocument(
        source_id=SourceId(source_path),
        source_path=source_path,
        source_sha256="sha",
        extension=extension,
        route="pdf_text_layer",
        parser_name="parser",
        parser_version="1.0",
        status=status,
        text=text,
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id=f"{source_path}:0",
                type="paragraph",
                text=text,
                page_no=1,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
        )
        if text
        else (),
    )


def _make_rows(count: int, *, success_text: str = "ok", has_blank_success: bool = False) -> tuple[ExtractedDocument, ...]:
    rows: list[ExtractedDocument] = []
    for idx in range(count):
        source_path = _source_path_for_idx(idx)
        extension = Path(source_path).suffix
        text = "" if has_blank_success and idx == 0 else success_text
        row = _make_row(
            source_path=source_path,
            extension=extension,
            status="success",
            text=text,
        )
        rows.append(row)
    return tuple(rows)


def _source_path_for_idx(idx: int) -> str:
    extension = ".pdf" if idx < 18 else ".hwp"
    return f"raw-{idx}{extension}"


def test_validate_output_directory_with_only_raw_documents_jsonl_succeeds(tmp_path: Path) -> None:
    raw_documents = tmp_path / "raw_documents.jsonl"
    _touch(raw_documents)
    assert validate_output_directory(tmp_path) == raw_documents


def test_validate_output_directory_with_missing_result_file_fails(tmp_path: Path) -> None:
    _touch(tmp_path / "other.txt")

    with pytest.raises(
        ExtractionInputError,
        match="result file not found",
    ):
        _ = validate_output_directory(tmp_path)


def test_validate_output_directory_with_extra_output_file_fails(tmp_path: Path) -> None:
    _touch(tmp_path / "raw_documents.jsonl")
    _touch(tmp_path / "other.txt")

    with pytest.raises(
        ExtractionInputError,
        match="unexpected output file: other.txt",
    ):
        _ = validate_output_directory(tmp_path)


def test_validate_extraction_rows_rejects_wrong_row_count() -> None:
    raw_docs = _make_raw_docs(23)
    rows = _make_rows(22)

    with pytest.raises(
        ExtractionInputError,
        match="expected 23 extraction rows, found 22",
    ):
        validate_extraction_rows(raw_docs, rows)


def test_validate_extraction_rows_accepts_current_raw_document_count() -> None:
    raw_docs = _make_raw_docs(31)
    rows = _make_rows(31)

    validate_extraction_rows(raw_docs, rows)


def test_validate_extraction_rows_rejects_success_row_with_no_text() -> None:
    raw_docs = _make_raw_docs(23)
    rows = _make_rows(23, has_blank_success=True)

    with pytest.raises(
        ExtractionInputError,
        match="success row has no text",
    ):
        validate_extraction_rows(raw_docs, rows)


def test_validate_extraction_rows_rejects_success_row_with_no_blocks() -> None:
    raw_docs = _make_raw_docs(23)
    rows = list(_make_rows(23))
    rows[0] = ExtractedDocument(
        source_id=SourceId("raw-0.pdf"),
        source_path="raw-0.pdf",
        source_sha256="sha",
        extension=".pdf",
        route="docling_structured",
        parser_name="docling",
        parser_version="1.0",
        status="success",
        text="text without blocks",
        pages_or_sections=(),
        error=None,
        blocks=(),
    )

    with pytest.raises(
        ExtractionInputError,
        match="success row has no blocks",
    ):
        validate_extraction_rows(raw_docs, tuple(rows))


def test_validate_extraction_rows_rejects_source_path_mismatch() -> None:
    raw_docs = _make_raw_docs(23)
    rows = list(_make_rows(23))
    rows[0] = _make_row(source_path="not-in-raw.pdf", extension=".pdf")

    with pytest.raises(
        ExtractionInputError,
        match="source_path mismatch",
    ):
        validate_extraction_rows(raw_docs, tuple(rows))


def test_extracted_document_json_round_trips_canonical_blocks() -> None:
    row = ExtractedDocument(
        source_id=SourceId("doc-source"),
        source_path="raw/sample.pdf",
        source_sha256="sha",
        extension=".pdf",
        route="docling_structured",
        parser_name="docling",
        parser_version="2.0",
        status="success",
        text="heading\nparagraph",
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="raw/sample.pdf:0",
                type="heading",
                text="heading",
                page_no=2,
                bbox=BoundingBox(left=1.0, top=2.0, right=3.0, bottom=4.0),
                level=1,
                section_path=("heading",),
                table_markdown=None,
                metadata=(("docling_label", "section_header"),),
            ),
        ),
    )

    payload = json.loads(row.to_json_text())
    parsed = parse_extracted_document(payload)

    assert parsed.blocks == row.blocks
    assert parsed.blocks[0].bbox == BoundingBox(left=1.0, top=2.0, right=3.0, bottom=4.0)

