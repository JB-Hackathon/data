from __future__ import annotations

from jb_hackathon_data.chunk_filters import apply_default_chunk_filter
from jb_hackathon_data.chunk_models import RagChunk


def test_default_filter_drops_image_placeholder_and_low_information_chunks() -> None:
    image = _chunk(
        index=0,
        text="<!-- image -->",
        search_text="<!-- image -->",
        token_count=6,
        table_markdown=None,
    )
    low_information = _chunk(
        index=1,
        text="➜",
        search_text="➜",
        token_count=1,
        table_markdown=None,
    )
    table = _chunk(
        index=2,
        text="| 항목 | 내용 |\n| --- | --- |\n| 금리 | 표시 |",
        search_text="광고 기준\n| 항목 | 내용 |\n| --- | --- |\n| 금리 | 표시 |",
        token_count=12,
        table_markdown="| 항목 | 내용 |\n| --- | --- |\n| 금리 | 표시 |",
    )
    useful_short = _chunk(
        index=3,
        text="금융위원회 금융감독원",
        search_text="금융광고규제 가이드라인\n금융위원회 금융감독원",
        token_count=15,
        table_markdown=None,
    )

    kept = apply_default_chunk_filter((image, low_information, table, useful_short))

    assert tuple(chunk.chunk_id for chunk in kept) == (table.chunk_id, useful_short.chunk_id)


def _chunk(
    *,
    index: int,
    text: str,
    search_text: str,
    token_count: int,
    table_markdown: str | None,
) -> RagChunk:
    return RagChunk.from_parts(
        source_id="source-1",
        source_path="raw/sample.pdf",
        source_sha256="a" * 64,
        route="docling_structured",
        parser_name="docling",
        parser_version="2.96.1",
        chunk_index=index,
        chunk_type="table" if table_markdown else "section",
        heading_path=("금융광고규제 가이드라인",),
        page_refs=(1,),
        bbox_refs=(),
        docling_refs=(f"#/texts/{index}",),
        block_ids=(),
        tokenizer_model_id="BAAI/bge-m3",
        token_count=token_count,
        text=text,
        search_text=search_text,
        table_markdown=table_markdown,
        metadata=(("chunker", "docling_hybrid"),),
    )
