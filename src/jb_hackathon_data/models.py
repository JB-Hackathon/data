"""Typed models for document extraction results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, NewType

SourceId = NewType("SourceId", str)
Route = Literal["docling_structured", "pdf_text_layer", "pdf_structured_ocr", "hwp_parser"]
Status = Literal["success", "failed", "skipped"]
BlockType = Literal["title", "heading", "paragraph", "list", "table", "image", "footer", "page_break", "unknown"]


@dataclass(frozen=True, slots=True)
class Structure:
    headings: tuple[str, ...]
    article_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PageSection:
    page_no: int | None
    section_path: str | None
    text: str
    tables: tuple[str, ...]
    structure: Structure


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    block_id: str
    type: BlockType
    text: str
    page_no: int | None
    bbox: BoundingBox | None
    level: int | None
    section_path: tuple[str, ...]
    table_markdown: str | None
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    source_id: SourceId
    source_path: str
    source_sha256: str
    extension: str
    route: Route
    parser_name: str
    parser_version: str
    status: Status
    text: str
    pages_or_sections: tuple[PageSection, ...]
    error: str | None
    blocks: tuple[DocumentBlock, ...] = ()
    docling_document_path: str | None = None

    def to_json_text(self) -> str:
        import json

        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))
