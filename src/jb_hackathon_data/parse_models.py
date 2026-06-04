"""Parse JSON-compatible payloads back into typed extraction models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from .models import BoundingBox, DocumentBlock, ExtractedDocument, PageSection, SourceId, Structure


class ExtractionPayloadError(RuntimeError):
    """Raised when a JSON row does not match the extraction contract."""


def parse_extracted_document(payload: Mapping[str, object]) -> ExtractedDocument:
    return ExtractedDocument(
        source_id=SourceId(_required_str(payload, "source_id")),
        source_path=_required_str(payload, "source_path"),
        source_sha256=_required_str(payload, "source_sha256"),
        extension=_required_str(payload, "extension"),
        route=_route(_required_str(payload, "route")),
        parser_name=_required_str(payload, "parser_name"),
        parser_version=_required_str(payload, "parser_version"),
        status=_status(_required_str(payload, "status")),
        text=_required_str(payload, "text"),
        pages_or_sections=_parse_sections(payload.get("pages_or_sections")),
        error=_optional_str(payload, "error"),
        blocks=_parse_blocks(payload.get("blocks")),
        docling_document_path=_optional_str(payload, "docling_document_path"),
    )


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ExtractionPayloadError(f"{key} must be a string")
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ExtractionPayloadError(f"{key} must be null or a string")


def _parse_sections(value: object) -> tuple[PageSection, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ExtractionPayloadError("pages_or_sections must be a list")
    return tuple(_parse_section(item) for item in value)


def _parse_section(value: object) -> PageSection:
    if not isinstance(value, Mapping):
        raise ExtractionPayloadError("page section must be an object")
    page_no = value.get("page_no")
    section_path = value.get("section_path")
    return PageSection(
        page_no=page_no if isinstance(page_no, int) else None,
        section_path=section_path if isinstance(section_path, str) else None,
        text=_required_str(value, "text"),
        tables=_str_tuple(value.get("tables"), "tables"),
        structure=_parse_structure(value.get("structure")),
    )


def _parse_structure(value: object) -> Structure:
    if not isinstance(value, Mapping):
        raise ExtractionPayloadError("structure must be an object")
    return Structure(
        headings=_str_tuple(value.get("headings"), "headings"),
        article_markers=_str_tuple(value.get("article_markers"), "article_markers"),
    )


def _parse_blocks(value: object) -> tuple[DocumentBlock, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ExtractionPayloadError("blocks must be a list")
    return tuple(_parse_block(item) for item in value)


def _parse_block(value: object) -> DocumentBlock:
    if not isinstance(value, Mapping):
        raise ExtractionPayloadError("block must be an object")
    page_no = value.get("page_no")
    level = value.get("level")
    return DocumentBlock(
        block_id=_required_str(value, "block_id"),
        type=_block_type(_required_str(value, "type")),
        text=_required_str(value, "text"),
        page_no=page_no if isinstance(page_no, int) else None,
        bbox=_parse_bbox(value.get("bbox")),
        level=level if isinstance(level, int) else None,
        section_path=_str_tuple(value.get("section_path"), "section_path"),
        table_markdown=_optional_str(value, "table_markdown"),
        metadata=_metadata_tuple(value.get("metadata")),
    )


def _parse_bbox(value: object) -> BoundingBox | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExtractionPayloadError("bbox must be an object or null")
    return BoundingBox(
        left=_required_float(value, "left"),
        top=_required_float(value, "top"),
        right=_required_float(value, "right"),
        bottom=_required_float(value, "bottom"),
    )


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    raise ExtractionPayloadError(f"{key} must be a number")


def _metadata_tuple(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ExtractionPayloadError("metadata must be a list")
    items: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            raise ExtractionPayloadError("metadata items must be key-value pairs")
        key, entry_value = item
        if not isinstance(key, str) or not isinstance(entry_value, str):
            raise ExtractionPayloadError("metadata item values must be strings")
        items.append((key, entry_value))
    return tuple(items)


def _str_tuple(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ExtractionPayloadError(f"{key} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ExtractionPayloadError(f"{key} items must be strings")
        items.append(item)
    return tuple(items)


def _block_type(
    value: str,
) -> Literal["title", "heading", "paragraph", "list", "table", "image", "footer", "page_break", "unknown"]:
    match value:
        case "title" | "heading" | "paragraph" | "list" | "table" | "image" | "footer" | "page_break" | "unknown":
            return value
        case _:
            raise ExtractionPayloadError(f"unknown block type: {value}")


def _route(value: str) -> Literal["docling_structured", "pdf_text_layer", "pdf_structured_ocr", "hwp_parser"]:
    match value:
        case "docling_structured" | "pdf_text_layer" | "pdf_structured_ocr" | "hwp_parser":
            return value
        case _:
            raise ExtractionPayloadError(f"unknown route: {value}")


def _status(value: str) -> Literal["success", "failed", "skipped"]:
    match value:
        case "success" | "failed" | "skipped":
            return value
        case _:
            raise ExtractionPayloadError(f"unknown status: {value}")
