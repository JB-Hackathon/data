"""Fallback chunking for extraction rows without a Docling sidecar."""

from __future__ import annotations

from collections.abc import Callable

from .chunk_models import ChunkType, RagChunk
from .search_text import build_search_text
from .models import DocumentBlock, ExtractedDocument

_SKIP_BLOCK_TYPES = frozenset({"image", "footer", "page_break"})
TokenCounter = Callable[[str], int]


def chunk_blocks_with_fallback(
    row: ExtractedDocument,
    *,
    tokenizer_model_id: str,
    max_tokens: int = 768,
    count_tokens: TokenCounter | None = None,
) -> tuple[RagChunk, ...]:
    if row.status != "success":
        return ()

    blocks = tuple(block for block in row.blocks if _has_chunkable_text(block))
    if not blocks:
        return ()

    counter = count_tokens or _rough_token_count
    return tuple(
        _chunk_for_group(
            row,
            group,
            chunk_index=index,
            tokenizer_model_id=tokenizer_model_id,
            count_tokens=counter,
            max_tokens=max_tokens,
        )
        for index, group in enumerate(_pack_blocks(blocks, max_tokens=max_tokens, count_tokens=counter))
    )


def _chunk_for_group(
    row: ExtractedDocument,
    blocks: tuple[DocumentBlock, ...],
    *,
    chunk_index: int,
    tokenizer_model_id: str,
    count_tokens: TokenCounter,
    max_tokens: int,
) -> RagChunk:
    text = "\n\n".join(_block_text(block) for block in blocks).strip()
    heading_path = _heading_path(blocks)
    search_text = build_search_text(source_path=row.source_path, heading_path=heading_path, text=text)
    table_markdown = _table_markdown(blocks)
    token_count = count_tokens(_token_count_text(heading_path=heading_path, text=text))
    return RagChunk.from_parts(
        source_id=str(row.source_id),
        source_path=row.source_path,
        source_sha256=row.source_sha256,
        route=row.route,
        parser_name=row.parser_name,
        parser_version=row.parser_version,
        chunk_index=chunk_index,
        chunk_type="overflow" if token_count > max_tokens else _chunk_type(blocks),
        heading_path=heading_path,
        page_refs=_page_refs(blocks),
        bbox_refs=tuple(block.bbox for block in blocks if block.bbox is not None),
        docling_refs=(),
        block_ids=tuple(block.block_id for block in blocks),
        tokenizer_model_id=tokenizer_model_id,
        token_count=token_count,
        text=text,
        search_text=search_text,
        table_markdown=table_markdown,
        metadata=(("chunker", "block_fallback"), ("overlap_tokens", "0")),
    )


def _pack_blocks(
    blocks: tuple[DocumentBlock, ...],
    *,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> tuple[tuple[DocumentBlock, ...], ...]:
    groups: list[tuple[DocumentBlock, ...]] = []
    current: tuple[DocumentBlock, ...] = ()
    for block in blocks:
        candidate = (*current, block)
        if current and _group_token_count(candidate, count_tokens) > max_tokens:
            groups.append(current)
            current = (block,)
            continue
        current = candidate
    if current:
        groups.append(current)
    return tuple(groups)


def _group_token_count(blocks: tuple[DocumentBlock, ...], count_tokens: TokenCounter) -> int:
    text = "\n\n".join(_block_text(block) for block in blocks).strip()
    return count_tokens(_token_count_text(heading_path=_heading_path(blocks), text=text))


def _token_count_text(*, heading_path: tuple[str, ...], text: str) -> str:
    return "\n".join((*heading_path, text)).strip()


def _has_chunkable_text(block: DocumentBlock) -> bool:
    return block.type not in _SKIP_BLOCK_TYPES and bool(_block_text(block).strip())


def _block_text(block: DocumentBlock) -> str:
    return block.table_markdown or block.text


def _heading_path(blocks: tuple[DocumentBlock, ...]) -> tuple[str, ...]:
    for block in blocks:
        if block.section_path:
            return block.section_path
    return ()


def _page_refs(blocks: tuple[DocumentBlock, ...]) -> tuple[int, ...]:
    pages = {block.page_no for block in blocks if block.page_no is not None}
    return tuple(sorted(pages))


def _table_markdown(blocks: tuple[DocumentBlock, ...]) -> str | None:
    tables = tuple(block.table_markdown for block in blocks if block.table_markdown is not None)
    if not tables:
        return None
    return "\n\n".join(tables)


def _chunk_type(blocks: tuple[DocumentBlock, ...]) -> ChunkType:
    block_types = {block.type for block in blocks}
    if block_types == {"table"}:
        return "table"
    if "table" in block_types:
        return "mixed"
    return "section"


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split()))
