"""HWP and HWPX extraction routes."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from .hwp_blocks import block_text, blocks_from_plain_text, canonical_blocks_from_markdown
from .io_utils import relative_source_path, sha256_file, source_id_from_sha
from .models import DocumentBlock, ExtractedDocument, PageSection, Structure


def extract_hwp(path: Path) -> ExtractedDocument:
    sha256 = sha256_file(path)
    try:
        return _extract_with_unhwp(path, sha256)
    except Exception as first_error:  # noqa: BLE001, BROAD_EXCEPT_OK
        try:
            return _extract_with_hwp_hwpx_parser(path, sha256)
        except Exception as second_error:  # noqa: BLE001, BROAD_EXCEPT_OK
            return _failed(path, sha256, first_error, second_error)


def _extract_with_unhwp(path: Path, sha256: str, _first_error: Exception | None = None) -> ExtractedDocument:
    import unhwp

    markdown = unhwp.to_markdown_with_cleanup(path).strip()
    if not markdown:
        markdown = unhwp.to_markdown(path).strip()
    if not markdown:
        markdown = unhwp.extract_text(path).strip()

    blocks = canonical_blocks_from_markdown(
        markdown,
        source_path=relative_source_path(path),
        parser_name="unhwp",
    )
    return _success_from_blocks(
        path=path,
        sha256=sha256,
        parser_name="unhwp",
        parser_version=version("unhwp"),
        blocks=blocks,
    )


def _extract_with_hwp_hwpx_parser(path: Path, sha256: str) -> ExtractedDocument:
    import hwp_hwpx_parser

    reader = hwp_hwpx_parser.read(path)
    try:
        text = reader.extract_text().strip()
        tables = tuple(str(table) for table in reader.get_tables_as_markdown())
        blocks = blocks_from_plain_text(
            path=path,
            parser_name="hwp-hwpx-parser",
            text=text,
            tables=tables,
        )
        return _success_from_blocks(
            path=path,
            sha256=sha256,
            parser_name="hwp-hwpx-parser",
            parser_version=version("hwp-hwpx-parser"),
            blocks=blocks,
        )
    finally:
        reader.close()


def _success_from_blocks(
    *,
    path: Path,
    sha256: str,
    parser_name: str,
    parser_version: str,
    blocks: tuple[DocumentBlock, ...],
) -> ExtractedDocument:
    text = "\n\n".join(block_text(block) for block in blocks if block_text(block)).strip()
    tables = tuple(block.table_markdown for block in blocks if block.table_markdown is not None)
    headings = tuple(block.text for block in blocks if block.type in {"title", "heading"} and block.text)
    section = PageSection(
        page_no=None,
        section_path="document",
        text=text,
        tables=tables,
        structure=Structure(headings=headings, article_markers=()),
    )
    return ExtractedDocument(
        source_id=source_id_from_sha(sha256),
        source_path=relative_source_path(path),
        source_sha256=sha256,
        extension=path.suffix.lower(),
        route="hwp_parser",
        parser_name=parser_name,
        parser_version=parser_version,
        status="success" if text or tables else "failed",
        text=text,
        pages_or_sections=(section,) if text or tables else (),
        error=None if text or tables else f"{parser_name} returned empty text",
        blocks=blocks,
    )


def _failed(
    path: Path,
    sha256: str,
    first_error: Exception,
    second_error: Exception,
) -> ExtractedDocument:
    return ExtractedDocument(
        source_id=source_id_from_sha(sha256),
        source_path=relative_source_path(path),
        source_sha256=sha256,
        extension=path.suffix.lower(),
        route="hwp_parser",
        parser_name="unhwp;hwp-hwpx-parser",
        parser_version=f"{_package_version('unhwp')};{_package_version('hwp-hwpx-parser')}",
        status="failed",
        text="",
        pages_or_sections=(),
        error=(
            f"unhwp={type(first_error).__name__}: {first_error}; "
            f"hwp-hwpx-parser={type(second_error).__name__}: {second_error}"
        ),
    )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except Exception:  # noqa: BLE001, BROAD_EXCEPT_OK
        return "not-installed"
