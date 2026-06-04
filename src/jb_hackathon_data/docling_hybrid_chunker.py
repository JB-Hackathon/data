"""Docling HybridChunker adapter for saved DoclingDocument sidecars."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeGuard, TypeVar

from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import DoclingDocument

from .chunk_models import ChunkType, RagChunk
from .search_text import build_search_text
from .models import BoundingBox, ExtractedDocument


class ChunkLike(Protocol):
    @property
    def text(self) -> str:
        raise NotImplementedError


ChunkT = TypeVar("ChunkT", bound=ChunkLike)


class ChunkerLike(Protocol[ChunkT]):
    def chunk(self, sidecar_path: Path, /) -> Iterable[ChunkT]:
        raise NotImplementedError

    def contextualize(self, chunk: ChunkT, /) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str, /) -> int:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DoclingHybridChunkerAdapter:
    chunker: HybridChunker

    def chunk(self, sidecar_path: Path) -> Iterable[BaseChunk]:
        document = DoclingDocument.load_from_json(sidecar_path)
        return self.chunker.chunk(document)

    def contextualize(self, chunk: BaseChunk) -> str:
        return self.chunker.contextualize(chunk)

    def count_tokens(self, text: str) -> int:
        return self.chunker.tokenizer.count_tokens(text)


def chunk_docling_sidecar(
    row: ExtractedDocument,
    sidecar_path: Path,
    chunker: ChunkerLike[ChunkT],
    *,
    tokenizer_model_id: str,
) -> tuple[RagChunk, ...]:
    chunks: list[RagChunk] = []
    for index, chunk in enumerate(chunker.chunk(sidecar_path)):
        text = chunk.text.strip()
        if not text:
            continue
        token_count_text = chunker.contextualize(chunk).strip() or text
        docling_refs = _docling_refs(chunk)
        heading_path = _headings(chunk)
        search_text = build_search_text(source_path=row.source_path, heading_path=heading_path, text=text)
        table_markdown = text if _is_table_chunk(text, docling_refs) else None
        chunks.append(
            RagChunk.from_parts(
                source_id=str(row.source_id),
                source_path=row.source_path,
                source_sha256=row.source_sha256,
                route=row.route,
                parser_name=row.parser_name,
                parser_version=row.parser_version,
                chunk_index=index,
                chunk_type=_chunk_type(text, docling_refs),
                heading_path=heading_path,
                page_refs=_page_refs(chunk),
                bbox_refs=_bbox_refs(chunk),
                docling_refs=docling_refs,
                block_ids=(),
                tokenizer_model_id=tokenizer_model_id,
                token_count=chunker.count_tokens(token_count_text),
                text=text,
                search_text=search_text,
                table_markdown=table_markdown,
                metadata=(("chunker", "docling_hybrid"), ("overlap_tokens", "0")),
            )
        )
    return tuple(chunks)


def _headings(chunk: ChunkLike) -> tuple[str, ...]:
    headings = getattr(chunk, "headings", None)
    if _is_str_sequence(headings):
        return tuple(headings)

    meta = getattr(chunk, "meta", None)
    meta_headings = getattr(meta, "headings", None)
    if _is_str_sequence(meta_headings):
        return tuple(meta_headings)
    return ()


def _docling_refs(chunk: ChunkLike) -> tuple[str, ...]:
    refs = getattr(chunk, "docling_refs", None)
    if _is_str_sequence(refs):
        return tuple(refs)

    meta = getattr(chunk, "meta", None)
    doc_items = getattr(meta, "doc_items", None)
    if not _is_sequence(doc_items):
        return ()

    collected: list[str] = []
    for item in doc_items:
        ref = getattr(item, "self_ref", None)
        if isinstance(ref, str) and ref:
            collected.append(ref)
    return tuple(collected)


def _page_refs(chunk: ChunkLike) -> tuple[int, ...]:
    refs = getattr(chunk, "page_refs", None)
    if _is_int_sequence(refs):
        return tuple(sorted(set(refs)))

    pages: set[int] = set()
    for prov in _provenance_items(chunk):
        page_no = getattr(prov, "page_no", None)
        if isinstance(page_no, int):
            pages.add(page_no)
    return tuple(sorted(pages))


def _bbox_refs(chunk: ChunkLike) -> tuple[BoundingBox, ...]:
    refs: list[BoundingBox] = []
    for prov in _provenance_items(chunk):
        bbox = getattr(prov, "bbox", None)
        left = getattr(bbox, "l", None)
        top = getattr(bbox, "t", None)
        right = getattr(bbox, "r", None)
        bottom = getattr(bbox, "b", None)
        if _is_number(left) and _is_number(top) and _is_number(right) and _is_number(bottom):
            refs.append(BoundingBox(left=float(left), top=float(top), right=float(right), bottom=float(bottom)))
    return tuple(refs)


def _provenance_items(chunk: ChunkLike) -> tuple[object, ...]:
    meta = getattr(chunk, "meta", None)
    doc_items = getattr(meta, "doc_items", None)
    if not _is_sequence(doc_items):
        return ()
    provenance: list[object] = []
    for item in doc_items:
        prov = getattr(item, "prov", None)
        if _is_sequence(prov):
            provenance.extend(prov)
    return tuple(provenance)


def _chunk_type(text: str, docling_refs: tuple[str, ...]) -> ChunkType:
    if _is_table_chunk(text, docling_refs):
        return "table"
    return "section"


def _is_table_chunk(text: str, docling_refs: tuple[str, ...]) -> bool:
    return any("/tables/" in ref for ref in docling_refs) or ("|" in text and "---" in text)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str)


def _is_str_sequence(value: object) -> TypeGuard[Sequence[str]]:
    return _is_sequence(value) and all(isinstance(item, str) for item in value)


def _is_int_sequence(value: object) -> TypeGuard[Sequence[int]]:
    return _is_sequence(value) and all(isinstance(item, int) for item in value)


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)
