from __future__ import annotations

from jb_hackathon_data.block_fallback_chunker import chunk_blocks_with_fallback
from jb_hackathon_data.models import DocumentBlock, ExtractedDocument, SourceId


def test_hwp_blocks_chunk_without_docling_sidecar() -> None:
    row = ExtractedDocument(
        source_id=SourceId("hwp-source-0001"),
        source_path="raw/sample.hwp",
        source_sha256="b" * 64,
        extension=".hwp",
        route="hwp_parser",
        parser_name="unhwp",
        parser_version="0.5.0",
        status="success",
        text="제목\n본문",
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="raw/sample.hwp:0",
                type="heading",
                text="제목",
                page_no=None,
                bbox=None,
                level=1,
                section_path=("제목",),
                table_markdown=None,
                metadata=(),
            ),
            DocumentBlock(
                block_id="raw/sample.hwp:1",
                type="paragraph",
                text="본문",
                page_no=None,
                bbox=None,
                level=None,
                section_path=("제목",),
                table_markdown=None,
                metadata=(),
            ),
        ),
    )

    chunks = chunk_blocks_with_fallback(row, tokenizer_model_id="BAAI/bge-m3")

    assert len(chunks) == 1
    assert chunks[0].block_ids == ("raw/sample.hwp:0", "raw/sample.hwp:1")
    assert chunks[0].docling_refs == ()


def test_fallback_filters_empty_images_and_footers() -> None:
    row = ExtractedDocument(
        source_id=SourceId("hwp-source-0001"),
        source_path="raw/sample.hwp",
        source_sha256="b" * 64,
        extension=".hwp",
        route="hwp_parser",
        parser_name="unhwp",
        parser_version="0.5.0",
        status="success",
        text="",
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="raw/sample.hwp:0",
                type="image",
                text="",
                page_no=None,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
            DocumentBlock(
                block_id="raw/sample.hwp:1",
                type="footer",
                text="1",
                page_no=1,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
        ),
    )

    assert chunk_blocks_with_fallback(row, tokenizer_model_id="BAAI/bge-m3") == ()


def test_fallback_packs_blocks_under_token_limit() -> None:
    row = ExtractedDocument(
        source_id=SourceId("hwp-source-0002"),
        source_path="raw/sample.hwp",
        source_sha256="e" * 64,
        extension=".hwp",
        route="hwp_parser",
        parser_name="unhwp",
        parser_version="0.5.0",
        status="success",
        text="one two\nthree four\nfive six",
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="raw/sample.hwp:0",
                type="paragraph",
                text="one two",
                page_no=None,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
            DocumentBlock(
                block_id="raw/sample.hwp:1",
                type="paragraph",
                text="three four",
                page_no=None,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
            DocumentBlock(
                block_id="raw/sample.hwp:2",
                type="paragraph",
                text="five six",
                page_no=None,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
        ),
    )

    chunks = chunk_blocks_with_fallback(
        row,
        tokenizer_model_id="BAAI/bge-m3",
        max_tokens=3,
        count_tokens=lambda text: len(text.split()),
    )

    assert [chunk.block_ids for chunk in chunks] == [
        ("raw/sample.hwp:0",),
        ("raw/sample.hwp:1",),
        ("raw/sample.hwp:2",),
    ]


def test_fallback_search_text_uses_document_name_section_and_original_text() -> None:
    row = ExtractedDocument(
        source_id=SourceId("hwp-source-0002"),
        source_path="raw/규제정책안.hwp",
        source_sha256="f" * 64,
        extension=".hwp",
        route="hwp_parser",
        parser_name="unhwp",
        parser_version="0.5.0",
        status="success",
        text="문서 제목\n본문",
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="raw/규제정책안.hwp:0",
                type="heading",
                text="제1부 규제",
                page_no=None,
                bbox=None,
                level=1,
                section_path=("제1부", "제1장"),
                table_markdown=None,
                metadata=(),
            ),
            DocumentBlock(
                block_id="raw/규제정책안.hwp:1",
                type="paragraph",
                text="본문",
                page_no=None,
                bbox=None,
                level=None,
                section_path=("제1부", "제1장"),
                table_markdown=None,
                metadata=(),
            ),
        ),
    )

    chunk = chunk_blocks_with_fallback(row, tokenizer_model_id="BAAI/bge-m3")[0]

    assert chunk.text == "제1부 규제\n\n본문"
    assert chunk.search_text == (
        "문서명: 규제정책안\n"
        "섹션: 제1부 > 제1장\n"
        "조항/섹션명: 제1장\n"
        "내용:\n제1부 규제\n\n본문"
    )


def test_fallback_search_text_omits_empty_heading_lines() -> None:
    row = ExtractedDocument(
        source_id=SourceId("hwp-source-0003"),
        source_path="raw/empty_section.hwp",
        source_sha256="g" * 64,
        extension=".hwp",
        route="hwp_parser",
        parser_name="unhwp",
        parser_version="0.5.0",
        status="success",
        text="본문",
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="raw/empty_section.hwp:0",
                type="paragraph",
                text="본문",
                page_no=None,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
        ),
    )

    chunk = chunk_blocks_with_fallback(row, tokenizer_model_id="BAAI/bge-m3")[0]

    assert chunk.search_text == "문서명: empty_section\n내용:\n본문"
