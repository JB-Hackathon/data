"""Utilities for cleaning raw OCR/text extraction output."""

from __future__ import annotations

import re

from .models import DocumentBlock, ExtractedDocument, PageSection

_CRLF_RE = re.compile(r"\r\n?")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_FOOTER_RE = re.compile(r"^-+\s*\d+\s*-+$")


def clean_text(text: str) -> str:
    """Normalize whitespace and remove extraction artifacts such as lone page numbers."""

    text = _CRLF_RE.sub("\n", text)
    text = _CONTROL_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)

    lines = text.splitlines()
    cleaned_lines: list[str] = []
    blank_streak = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if blank_streak < 2:
                cleaned_lines.append("")
            blank_streak += 1
            continue

        if _is_footer_marker(stripped):
            continue

        cleaned_lines.append(stripped if not blank_streak else stripped)
        blank_streak = 0

    return "\n".join(cleaned_lines)


def _is_footer_marker(line: str) -> bool:
    return bool(re.fullmatch(r"\d+", line) or _FOOTER_RE.fullmatch(line))


def clean_document(row: ExtractedDocument) -> ExtractedDocument:
    cleaned_sections = tuple(_clean_section(section) for section in row.pages_or_sections)
    cleaned_blocks = tuple(_clean_block(block) for block in row.blocks)
    return ExtractedDocument(
        source_id=row.source_id,
        source_path=row.source_path,
        source_sha256=row.source_sha256,
        extension=row.extension,
        route=row.route,
        parser_name=row.parser_name,
        parser_version=row.parser_version,
        status=row.status,
        text=clean_text(row.text),
        pages_or_sections=cleaned_sections,
        error=row.error,
        blocks=cleaned_blocks,
        docling_document_path=row.docling_document_path,
    )


def _clean_section(section: PageSection) -> PageSection:
    cleaned_text = clean_text(section.text)
    return PageSection(
        page_no=section.page_no,
        section_path=section.section_path,
        text=cleaned_text,
        tables=tuple(clean_text(value) for value in section.tables),
        structure=section.structure,
    )


def _clean_block(block: DocumentBlock) -> DocumentBlock:
    return DocumentBlock(
        block_id=block.block_id,
        type=block.type,
        text=clean_text(block.text),
        page_no=block.page_no,
        bbox=block.bbox,
        level=block.level,
        section_path=block.section_path,
        table_markdown=clean_text(block.table_markdown) if block.table_markdown is not None else None,
        metadata=block.metadata,
    )
