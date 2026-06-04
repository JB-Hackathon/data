"""Validation for RAG chunk rows."""

from __future__ import annotations

from collections.abc import Iterable

from .chunk_models import RagChunk


class ChunkValidationError(RuntimeError):
    """Raised when chunk rows are not usable for RAG ingestion."""


def validate_chunk_rows(rows: Iterable[RagChunk], *, max_tokens: int) -> None:
    seen_ids: set[str] = set()
    for row in rows:
        if not row.text.strip() or not row.search_text.strip():
            raise ChunkValidationError(f"empty chunk text: {row.chunk_id}")
        if row.token_count > max_tokens:
            raise ChunkValidationError(f"chunk token count exceeds limit: {row.chunk_id}")
        if not row.docling_refs and not row.block_ids:
            raise ChunkValidationError(f"chunk has no source provenance: {row.chunk_id}")
        if row.chunk_id in seen_ids:
            raise ChunkValidationError(f"duplicate chunk_id: {row.chunk_id}")
        seen_ids.add(row.chunk_id)
