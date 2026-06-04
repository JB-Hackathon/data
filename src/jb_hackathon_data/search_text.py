"""Build search-oriented text for RAG chunks."""

from __future__ import annotations

from pathlib import Path


def build_search_text(*, source_path: str, heading_path: tuple[str, ...], text: str) -> str:
    parts = [f"문서명: {_document_name(source_path)}"]
    headings = _clean_headings(heading_path)
    section = _section_text(headings)
    if section:
        parts.append(f"섹션: {section}")
        parts.append(f"조항/섹션명: {headings[-1]}")
    parts.append(f"내용:\n{text}")
    return "\n".join(parts)


def _document_name(source_path: str) -> str:
    name = Path(source_path).stem.strip()
    return name or source_path


def _clean_headings(heading_path: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(part.strip() for part in heading_path if part.strip())


def _section_text(heading_path: tuple[str, ...]) -> str:
    return " > ".join(part for part in heading_path if part)
