"""Validate final extraction JSONL output."""

from __future__ import annotations

import typer
from pathlib import Path
from typing import Annotated

from .io_utils import ExtractionInputError, iter_raw_documents, read_jsonl, relative_source_path, resolve_root_path
from .models import ExtractedDocument

app = typer.Typer(add_completion=False)
RESULT_FILENAME = "raw_documents.jsonl"


@app.command()
def main(
    raw_dir: Annotated[str, typer.Option("--raw-dir")],
    results: Annotated[str, typer.Option("--results")],
) -> None:
    raw_path = resolve_root_path(raw_dir)
    result_candidate = resolve_root_path(results)
    if result_candidate.is_dir():
        result_path = validate_output_directory(result_candidate)
    elif result_candidate.is_file() and result_candidate.name == RESULT_FILENAME:
        result_path = validate_output_directory(result_candidate.parent)
    else:
        result_path = result_candidate
    try:
        raw_docs = iter_raw_documents(raw_path)
        rows = read_jsonl(result_path)
        validate_extraction_rows(raw_docs, rows)
    except ExtractionInputError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"validated_sources={len(rows)}")


def validate_output_directory(output_dir: Path) -> Path:
    """Validate that output directory contains only `raw_documents.jsonl` as file."""

    if not output_dir.exists():
        raise ExtractionInputError(f"output directory not found: {output_dir}")
    if not output_dir.is_dir():
        raise ExtractionInputError(f"output path is not a directory: {output_dir}")

    output_files = tuple(path for path in output_dir.iterdir() if path.is_file())
    result_path = output_dir / RESULT_FILENAME
    if not result_path.exists():
        raise ExtractionInputError(f"result file not found: {result_path}")

    for path in output_files:
        if path.name != RESULT_FILENAME:
            raise ExtractionInputError(f"unexpected output file: {path.name}")

    return result_path


def validate_extraction_rows(
    raw_docs: tuple[Path, ...],
    rows: tuple[ExtractedDocument, ...],
) -> None:
    _validate_counts(raw_docs, rows)
    _validate_rows(raw_docs, rows)


def _validate_counts(raw_docs: tuple[Path, ...], rows: tuple[ExtractedDocument, ...]) -> None:
    if len(rows) != len(raw_docs):
        raise ExtractionInputError(f"expected {len(raw_docs)} extraction rows, found {len(rows)}")


def _validate_rows(raw_docs: tuple[Path, ...], rows: tuple[ExtractedDocument, ...]) -> None:
    source_paths = tuple(row.source_path for row in rows)
    if len(set(source_paths)) != len(source_paths):
        raise ExtractionInputError("duplicate source_path in extraction results")
    _validate_source_paths(raw_docs, source_paths)
    raw_pdf_count = sum(1 for path in raw_docs if path.suffix.lower() == ".pdf")
    raw_hwp_count = sum(1 for path in raw_docs if path.suffix.lower() in {".hwp", ".hwpx"})
    pdf_count = sum(1 for row in rows if row.extension == ".pdf")
    hwp_count = sum(1 for row in rows if row.extension in {".hwp", ".hwpx"})
    if pdf_count != raw_pdf_count or hwp_count != raw_hwp_count:
        expected = f"expected pdf={raw_pdf_count}, hwp={raw_hwp_count}"
        found = f"found pdf={pdf_count}, hwp={hwp_count}"
        raise ExtractionInputError(
            f"unexpected source mix: {expected}; {found}"
        )
    for row in rows:
        has_text = bool(row.text.strip()) or any(section.text.strip() for section in row.pages_or_sections)
        if row.status == "success" and not has_text:
            raise ExtractionInputError(f"success row has no text: {row.source_path}")
        if row.status == "success" and not row.blocks:
            raise ExtractionInputError(f"success row has no blocks: {row.source_path}")
        if row.status == "failed" and not row.error:
            raise ExtractionInputError(f"failed row has no error: {row.source_path}")


def _validate_source_paths(raw_docs: tuple[Path, ...], source_paths: tuple[str, ...]) -> None:
    expected = {_raw_doc_source_path(path) for path in raw_docs}
    actual = set(source_paths)
    if expected == actual:
        return

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing={missing[:3]}")
    if unexpected:
        details.append(f"unexpected={unexpected[:3]}")
    raise ExtractionInputError(f"source_path mismatch: {'; '.join(details)}")


def _raw_doc_source_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return relative_source_path(path)
    except ValueError:
        return path.name


if __name__ == "__main__":
    app()
