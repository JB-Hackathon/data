"""Build chunk-ready HWP blocks from Markdown and plain text."""

from __future__ import annotations

import re
from pathlib import Path

from .io_utils import relative_source_path
from .models import BlockType, DocumentBlock

MAX_HWP_BLOCK_CHARS = 2500

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]+")


def canonical_blocks_from_markdown(
    markdown: str,
    *,
    source_path: str,
    parser_name: str,
) -> tuple[DocumentBlock, ...]:
    blocks: list[DocumentBlock] = []
    heading_stack: list[str] = []
    pending_lines: list[str] = []
    pending_type: BlockType | None = None

    def flush() -> None:
        nonlocal pending_lines, pending_type
        if pending_type is None:
            return
        text = "\n".join(pending_lines).strip()
        if text:
            _append_split_blocks(
                blocks,
                source_path=source_path,
                parser_name=parser_name,
                block_type=pending_type,
                text=text,
                section_path=tuple(heading_stack),
                level=None,
            )
        pending_lines = []
        pending_type = None

    for raw_line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            flush()
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            heading_text = _plain_markdown_text(heading_match.group(2))
            del heading_stack[level - 1 :]
            heading_stack.append(heading_text)
            blocks.append(
                _block(
                    source_path=source_path,
                    index=len(blocks),
                    parser_name=parser_name,
                    block_type="heading",
                    text=heading_text,
                    section_path=tuple(heading_stack),
                    level=level,
                    table_markdown=None,
                )
            )
            continue

        next_type = _line_type(stripped)
        if pending_type is not None and pending_type != next_type:
            flush()
        pending_type = next_type
        pending_lines.append(stripped)

    flush()
    return tuple(blocks)


def blocks_from_plain_text(
    *,
    path: Path,
    parser_name: str,
    text: str,
    tables: tuple[str, ...],
) -> tuple[DocumentBlock, ...]:
    blocks: list[DocumentBlock] = []
    source_path = relative_source_path(path)
    for segment in _split_text(text):
        blocks.append(
            _block(
                source_path=source_path,
                index=len(blocks),
                parser_name=parser_name,
                block_type="paragraph",
                text=segment,
                section_path=(),
                level=None,
                table_markdown=None,
                metadata=(("parser", parser_name), ("source_format", "plain_text")),
            )
        )
    for table in tables:
        for segment in _split_text(table):
            blocks.append(
                _block(
                    source_path=source_path,
                    index=len(blocks),
                    parser_name=parser_name,
                    block_type="table",
                    text=segment,
                    section_path=(),
                    level=None,
                    table_markdown=segment,
                    metadata=(("parser", parser_name), ("source_format", "table_markdown")),
                )
            )
    return tuple(blocks)


def block_text(block: DocumentBlock) -> str:
    return block.table_markdown or block.text


def _line_type(stripped: str) -> BlockType:
    if _is_table_line(stripped):
        return "table"
    if _LIST_RE.match(stripped):
        return "list"
    return "paragraph"


def _is_table_line(stripped: str) -> bool:
    return (stripped.startswith("|") and stripped.count("|") >= 2) or bool(_TABLE_SEPARATOR_RE.match(stripped))


def _append_split_blocks(
    blocks: list[DocumentBlock],
    *,
    source_path: str,
    parser_name: str,
    block_type: BlockType,
    text: str,
    section_path: tuple[str, ...],
    level: int | None,
) -> None:
    for segment_index, segment in enumerate(_split_text(text), start=1):
        metadata = (("parser", parser_name), ("source_format", "markdown"))
        if len(segment) < len(text) or segment_index > 1:
            metadata = (*metadata, ("split_index", str(segment_index)))
        blocks.append(
            _block(
                source_path=source_path,
                index=len(blocks),
                parser_name=parser_name,
                block_type=block_type,
                text=segment,
                section_path=section_path,
                level=level,
                table_markdown=segment if block_type == "table" else None,
                metadata=metadata,
            )
        )


def _split_text(text: str) -> tuple[str, ...]:
    segments: list[str] = []
    current = ""
    for line in text.splitlines() or [text]:
        for part in _split_long_line(line):
            candidate = part if not current else f"{current}\n{part}"
            if len(candidate) <= MAX_HWP_BLOCK_CHARS:
                current = candidate
                continue
            if current:
                segments.append(current)
            current = part
    if current:
        segments.append(current)
    return tuple(segment for segment in segments if segment.strip())


def _split_long_line(line: str) -> tuple[str, ...]:
    if len(line) <= MAX_HWP_BLOCK_CHARS:
        return (line,)
    words = line.split(" ")
    if len(words) == 1:
        return tuple(line[index : index + MAX_HWP_BLOCK_CHARS] for index in range(0, len(line), MAX_HWP_BLOCK_CHARS))

    parts: list[str] = []
    current = ""
    for word in words:
        if len(word) > MAX_HWP_BLOCK_CHARS:
            if current:
                parts.append(current)
                current = ""
            chunks = tuple(
                word[index : index + MAX_HWP_BLOCK_CHARS]
                for index in range(0, len(word), MAX_HWP_BLOCK_CHARS)
            )
            parts.extend(chunks[:-1])
            current = chunks[-1]
            continue
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= MAX_HWP_BLOCK_CHARS:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = word
    if current:
        parts.append(current)
    return tuple(parts)


def _plain_markdown_text(text: str) -> str:
    return _MARKDOWN_EMPHASIS_RE.sub("", text).strip()


def _block(
    *,
    source_path: str,
    index: int,
    parser_name: str,
    block_type: BlockType,
    text: str,
    section_path: tuple[str, ...],
    level: int | None,
    table_markdown: str | None,
    metadata: tuple[tuple[str, str], ...] | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        block_id=f"{source_path}:hwp-{index}",
        type=block_type,
        text=text,
        page_no=None,
        bbox=None,
        level=level,
        section_path=section_path,
        table_markdown=table_markdown,
        metadata=metadata or (("parser", parser_name), ("source_format", "markdown")),
    )
