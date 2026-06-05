from __future__ import annotations

import json

import pytest

from jb_hackathon_data.chunk_models import ChunkPayloadError, RagChunk, parse_rag_chunk


def test_chunk_id_is_stable_for_same_content_and_provenance() -> None:
    first = RagChunk.from_parts(
        source_id="source-1",
        source_path="raw/sample.pdf",
        source_sha256="a" * 64,
        route="docling_structured",
        parser_name="docling",
        parser_version="2.96.1",
        chunk_index=0,
        chunk_type="section",
        heading_path=("제목",),
        page_refs=(1,),
        bbox_refs=(),
        docling_refs=("#/texts/0",),
        block_ids=(),
        tokenizer_model_id="BAAI/bge-m3",
        token_count=12,
        text="본문",
        search_text="제목\n본문",
        table_markdown=None,
        metadata=(("chunker", "docling_hybrid"),),
    )
    second = RagChunk.from_parts(
        source_id="source-1",
        source_path="raw/sample.pdf",
        source_sha256="a" * 64,
        route="docling_structured",
        parser_name="docling",
        parser_version="2.96.1",
        chunk_index=0,
        chunk_type="section",
        heading_path=("제목",),
        page_refs=(1,),
        bbox_refs=(),
        docling_refs=("#/texts/0",),
        block_ids=(),
        tokenizer_model_id="BAAI/bge-m3",
        token_count=12,
        text="본문",
        search_text="제목\n본문",
        table_markdown=None,
        metadata=(("chunker", "docling_hybrid"),),
    )

    assert first.chunk_id == second.chunk_id


def test_parse_rag_chunk_rejects_missing_provenance() -> None:
    payload = json.loads(
        RagChunk.from_parts(
            source_id="source-1",
            source_path="raw/sample.pdf",
            source_sha256="a" * 64,
            route="docling_structured",
            parser_name="docling",
            parser_version="2.96.1",
            chunk_index=0,
            chunk_type="section",
            heading_path=(),
            page_refs=(),
            bbox_refs=(),
            docling_refs=("#/texts/0",),
            block_ids=(),
            tokenizer_model_id="BAAI/bge-m3",
            token_count=1,
            text="본문",
            search_text="본문",
            table_markdown=None,
            metadata=(),
        ).to_json_text()
    )
    del payload["docling_refs"]

    with pytest.raises(ChunkPayloadError, match="provenance"):
        parse_rag_chunk(payload)
