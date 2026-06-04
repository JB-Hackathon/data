from __future__ import annotations

from pathlib import Path

import pytest

from jb_hackathon_data.hwp_blocks import (
    MAX_HWP_BLOCK_CHARS,
    canonical_blocks_from_markdown,
)
from jb_hackathon_data.hwp_extractors import extract_hwp


def test_hwp_markdown_structure_becomes_chunkable_blocks() -> None:
    markdown = """
# 금융투자회사의 영업 및 업무에 관한 규정

## 제1장 총칙

이 규정은 금융투자회사의 영업 및 업무에 관한 사항을 정한다.

1. 광고는 사실과 다르게 표시해서는 안 된다.
2. 소비자가 오인할 수 있는 표현을 피해야 한다.

| 구분 | 내용 |
| --- | --- |
| 광고 | 심의 대상 |
""".strip()

    blocks = canonical_blocks_from_markdown(
        markdown,
        source_path="raw/sample.hwp",
        parser_name="unhwp",
    )

    assert [block.type for block in blocks] == [
        "heading",
        "heading",
        "paragraph",
        "list",
        "table",
    ]
    assert blocks[2].section_path == (
        "금융투자회사의 영업 및 업무에 관한 규정",
        "제1장 총칙",
    )
    assert blocks[-1].table_markdown is not None


def test_hwp_markdown_long_paragraph_is_split_for_chunking() -> None:
    markdown = "# 제목\n\n" + ("광고 심의 기준 " * 900)

    blocks = canonical_blocks_from_markdown(
        markdown,
        source_path="raw/long.hwp",
        parser_name="unhwp",
    )
    text_blocks = [block for block in blocks if block.type != "heading"]

    assert len(text_blocks) > 1
    assert max(len(block.text) for block in text_blocks) <= MAX_HWP_BLOCK_CHARS


def test_hwp_markdown_long_word_inside_line_is_split_for_chunking() -> None:
    markdown = "# 제목\n\n" + ("x" * (MAX_HWP_BLOCK_CHARS + 3)) + " 다음 문장"

    blocks = canonical_blocks_from_markdown(
        markdown,
        source_path="raw/long-word.hwp",
        parser_name="unhwp",
    )
    text_blocks = [block for block in blocks if block.type != "heading"]

    assert len(text_blocks) > 1
    assert max(len(block.text) for block in text_blocks) <= MAX_HWP_BLOCK_CHARS


def test_extract_hwp_prefers_unhwp_markdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import unhwp

    document_path = tmp_path / "sample.hwp"
    document_path.write_bytes(b"fake hwp bytes")
    markdown = "# 제목\n\n본문입니다.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"

    monkeypatch.setattr(unhwp, "to_markdown_with_cleanup", lambda path: markdown)

    document = extract_hwp(document_path)

    assert document.status == "success"
    assert document.parser_name == "unhwp"
    assert [block.type for block in document.blocks] == ["heading", "paragraph", "table"]
    assert document.pages_or_sections[0].structure.headings == ("제목",)
