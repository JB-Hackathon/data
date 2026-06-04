"""Typed models for RAG chunk rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .models import BoundingBox, Route

ChunkType = Literal["section", "table", "mixed", "overflow"]


class ChunkPayloadError(RuntimeError):
    """Raised when a JSON row does not match the RAG chunk contract."""


@dataclass(frozen=True, slots=True)
class RagChunk:
    chunk_id: str
    source_id: str
    source_path: str
    source_sha256: str
    route: Route
    parser_name: str
    parser_version: str
    chunk_index: int
    chunk_type: ChunkType
    heading_path: tuple[str, ...]
    page_refs: tuple[int, ...]
    bbox_refs: tuple[BoundingBox, ...]
    docling_refs: tuple[str, ...]
    block_ids: tuple[str, ...]
    tokenizer_model_id: str
    token_count: int
    text: str
    search_text: str
    table_markdown: str | None
    metadata: tuple[tuple[str, str], ...]

    @classmethod
    def from_parts(
        cls,
        *,
        source_id: str,
        source_path: str,
        source_sha256: str,
        route: Route,
        parser_name: str,
        parser_version: str,
        chunk_index: int,
        chunk_type: ChunkType,
        heading_path: tuple[str, ...],
        page_refs: tuple[int, ...],
        bbox_refs: tuple[BoundingBox, ...],
        docling_refs: tuple[str, ...],
        block_ids: tuple[str, ...],
        tokenizer_model_id: str,
        token_count: int,
        text: str,
        search_text: str,
        table_markdown: str | None,
        metadata: tuple[tuple[str, str], ...],
    ) -> RagChunk:
        chunk_id = _chunk_id(
            source_id=source_id,
            source_sha256=source_sha256,
            chunk_index=chunk_index,
            heading_path=heading_path,
            docling_refs=docling_refs,
            block_ids=block_ids,
            text=text,
        )
        return cls(
            chunk_id=chunk_id,
            source_id=source_id,
            source_path=source_path,
            source_sha256=source_sha256,
            route=route,
            parser_name=parser_name,
            parser_version=parser_version,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            heading_path=heading_path,
            page_refs=page_refs,
            bbox_refs=bbox_refs,
            docling_refs=docling_refs,
            block_ids=block_ids,
            tokenizer_model_id=tokenizer_model_id,
            token_count=token_count,
            text=text,
            search_text=search_text,
            table_markdown=table_markdown,
            metadata=metadata,
        )

    def to_json_text(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


def write_chunk_jsonl(path: str, rows: Iterable[RagChunk]) -> None:
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            _ = file.write(row.to_json_text())
            _ = file.write("\n")


def parse_rag_chunk(payload: Mapping[str, object]) -> RagChunk:
    for key in ("page_refs", "bbox_refs", "docling_refs", "block_ids"):
        if key not in payload:
            raise ChunkPayloadError("provenance fields are required")

    return RagChunk(
        chunk_id=_required_str(payload, "chunk_id"),
        source_id=_required_str(payload, "source_id"),
        source_path=_required_str(payload, "source_path"),
        source_sha256=_required_str(payload, "source_sha256"),
        route=_route(_required_str(payload, "route")),
        parser_name=_required_str(payload, "parser_name"),
        parser_version=_required_str(payload, "parser_version"),
        chunk_index=_required_int(payload, "chunk_index"),
        chunk_type=_chunk_type(_required_str(payload, "chunk_type")),
        heading_path=_str_tuple(payload.get("heading_path"), "heading_path"),
        page_refs=_int_tuple(payload.get("page_refs"), "page_refs"),
        bbox_refs=_parse_bboxes(payload.get("bbox_refs")),
        docling_refs=_str_tuple(payload.get("docling_refs"), "docling_refs"),
        block_ids=_str_tuple(payload.get("block_ids"), "block_ids"),
        tokenizer_model_id=_required_str(payload, "tokenizer_model_id"),
        token_count=_required_int(payload, "token_count"),
        text=_required_str(payload, "text"),
        search_text=_required_str(payload, "search_text"),
        table_markdown=_optional_str(payload, "table_markdown"),
        metadata=_metadata_tuple(payload.get("metadata")),
    )


def _chunk_id(
    *,
    source_id: str,
    source_sha256: str,
    chunk_index: int,
    heading_path: tuple[str, ...],
    docling_refs: tuple[str, ...],
    block_ids: tuple[str, ...],
    text: str,
) -> str:
    payload = {
        "source_id": source_id,
        "source_sha256": source_sha256,
        "chunk_index": chunk_index,
        "heading_path": heading_path,
        "docling_refs": docling_refs,
        "block_ids": block_ids,
        "text": text,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"chk_{digest[:24]}"


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ChunkPayloadError(f"{key} must be a string")
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ChunkPayloadError(f"{key} must be null or a string")


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ChunkPayloadError(f"{key} must be an integer")


def _str_tuple(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ChunkPayloadError(f"{key} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ChunkPayloadError(f"{key} items must be strings")
        items.append(item)
    return tuple(items)


def _int_tuple(value: object, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ChunkPayloadError(f"{key} must be a list")
    items: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ChunkPayloadError(f"{key} items must be integers")
        items.append(item)
    return tuple(items)


def _parse_bboxes(value: object) -> tuple[BoundingBox, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ChunkPayloadError("bbox_refs must be a list")
    return tuple(_parse_bbox(item) for item in value)


def _parse_bbox(value: object) -> BoundingBox:
    if not isinstance(value, Mapping):
        raise ChunkPayloadError("bbox ref must be an object")
    return BoundingBox(
        left=_required_float(value, "left"),
        top=_required_float(value, "top"),
        right=_required_float(value, "right"),
        bottom=_required_float(value, "bottom"),
    )


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ChunkPayloadError(f"{key} must be a number")


def _metadata_tuple(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ChunkPayloadError("metadata must be a list")
    items: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            raise ChunkPayloadError("metadata items must be key-value pairs")
        key, entry_value = item
        if not isinstance(key, str) or not isinstance(entry_value, str):
            raise ChunkPayloadError("metadata item values must be strings")
        items.append((key, entry_value))
    return tuple(items)


def _route(value: str) -> Route:
    match value:
        case "docling_structured" | "pdf_text_layer" | "pdf_structured_ocr" | "hwp_parser":
            return value
        case _:
            raise ChunkPayloadError(f"unknown route: {value}")


def _chunk_type(value: str) -> ChunkType:
    match value:
        case "section" | "table" | "mixed" | "overflow":
            return value
        case _:
            raise ChunkPayloadError(f"unknown chunk type: {value}")
