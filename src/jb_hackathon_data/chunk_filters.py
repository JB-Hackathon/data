"""Default lightweight filters for RAG chunks before indexing."""

from __future__ import annotations

from typing import Final

from .chunk_models import RagChunk

IMAGE_PLACEHOLDER_TEXT: Final = "<!-- image -->"
LOW_INFORMATION_TOKEN_LIMIT: Final = 20
LOW_INFORMATION_CHAR_LIMIT: Final = 1


def apply_default_chunk_filter(chunks: tuple[RagChunk, ...]) -> tuple[RagChunk, ...]:
    return tuple(chunk for chunk in chunks if _keep_chunk(chunk))


def _keep_chunk(chunk: RagChunk) -> bool:
    text = chunk.text.strip()
    if text == IMAGE_PLACEHOLDER_TEXT:
        return False
    if chunk.table_markdown is not None:
        return True
    return chunk.token_count >= LOW_INFORMATION_TOKEN_LIMIT or _meaningful_char_count(text) >= LOW_INFORMATION_CHAR_LIMIT


def _meaningful_char_count(text: str) -> int:
    return sum(1 for char in text if char.isalnum())
