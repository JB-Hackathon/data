from __future__ import annotations

import pytest

from jb_hackathon_data.chunk_models import RagChunk
from jb_hackathon_data.validate_chunks import ChunkValidationError, validate_chunk_rows


def test_validate_chunks_rejects_oversized_chunk() -> None:
    chunk = RagChunk.from_parts(
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
        token_count=769,
        text="본문",
        search_text="본문",
        table_markdown=None,
        metadata=(),
    )

    with pytest.raises(ChunkValidationError, match="token"):
        validate_chunk_rows((chunk,), max_tokens=768)
