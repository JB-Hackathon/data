from __future__ import annotations

from jb_hackathon_data.models import DocumentBlock, ExtractedDocument, PageSection, SourceId, Structure
from jb_hackathon_data.text_cleaning import clean_document, clean_text


def _make_row() -> ExtractedDocument:
    return ExtractedDocument(
        source_id=SourceId("row-source"),
        source_path="path/test.pdf",
        source_sha256="sha",
        extension=".pdf",
        route="pdf_text_layer",
        parser_name="parser",
        parser_version="1.0",
        status="success",
        text="제 1 조\r\n본문\n1\n- 10 -\n끝",
        pages_or_sections=(
            PageSection(
                page_no=1,
                section_path="doc",
                text="제2조\r\n항1\n\n\n---\n2",
                tables=("A  |  B \r\n",),
                structure=Structure((), ()),
            ),
        ),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="path/test.pdf:0",
                type="heading",
                text="제2조\r\n항1",
                page_no=1,
                bbox=None,
                level=1,
                section_path=("제2조",),
                table_markdown=None,
                metadata=(("docling_label", "section_header"),),
            ),
        ),
    )


def _make_failed_row() -> ExtractedDocument:
    return ExtractedDocument(
        source_id=SourceId("failed-source"),
        source_path="path/failed.hwp",
        source_sha256="sha2",
        extension=".hwp",
        route="hwp_parser",
        parser_name="hwp-parser",
        parser_version="2.0",
        status="failed",
        text="제3조\r\n본문\n\n1\n",
        pages_or_sections=(
            PageSection(
                page_no=None,
                section_path="doc",
                text="표1\r\n1\r\n",
                tables=("   ",),
                structure=Structure((), ()),
            ),
        ),
        error="empty text",
        blocks=(),
    )


def test_clean_document_cleans_text_sections_and_sections_structure() -> None:
    row = _make_row()
    cleaned = clean_document(row)

    assert cleaned is not row
    assert row.text == "제 1 조\r\n본문\n1\n- 10 -\n끝"
    assert cleaned.text == clean_text(row.text)
    assert cleaned.pages_or_sections[0] is not row.pages_or_sections[0]
    assert row.pages_or_sections[0].text == "제2조\r\n항1\n\n\n---\n2"
    assert row.pages_or_sections[0].tables == ("A  |  B \r\n",)

    section = cleaned.pages_or_sections[0]
    assert section.text == clean_text(row.pages_or_sections[0].text)
    assert section.tables == (clean_text(row.pages_or_sections[0].tables[0]),)
    assert section.structure == row.pages_or_sections[0].structure

    block = cleaned.blocks[0]
    assert block.text == clean_text(row.blocks[0].text)
    assert block.section_path == row.blocks[0].section_path
    assert block.metadata == row.blocks[0].metadata

    assert cleaned.source_id == row.source_id
    assert cleaned.source_path == row.source_path
    assert cleaned.source_sha256 == row.source_sha256
    assert cleaned.extension == row.extension
    assert cleaned.route == row.route
    assert cleaned.parser_name == row.parser_name
    assert cleaned.parser_version == row.parser_version
    assert cleaned.status == row.status
    assert cleaned.error == row.error


def test_failed_row_metadata_is_preserved_while_cleaning_text_and_sections() -> None:
    row = _make_failed_row()
    cleaned = clean_document(row)

    assert cleaned.status == "failed"
    assert cleaned.error == "empty text"
    assert cleaned.text == clean_text(row.text)
    assert cleaned.pages_or_sections[0].text == clean_text(row.pages_or_sections[0].text)

