"""Merge intermediate extraction rows into the final result JSONL."""

from __future__ import annotations

import typer
from typing import Annotated

from .io_utils import read_jsonl, resolve_root_path, write_jsonl
from .models import ExtractedDocument

app = typer.Typer(add_completion=False)


@app.command()
def main(
    inputs: Annotated[list[str], typer.Option("--inputs")],
    out: Annotated[str, typer.Option("--out")],
) -> None:
    rows = tuple(row for input_path in inputs for row in read_jsonl(resolve_root_path(input_path)))
    merged = _merge_rows(rows)
    out_path = resolve_root_path(out)
    write_jsonl(out_path, merged)
    typer.echo(f"final_results={len(merged)}")


def _merge_rows(rows: tuple[ExtractedDocument, ...]) -> tuple[ExtractedDocument, ...]:
    by_source: dict[str, ExtractedDocument] = {}
    for row in rows:
        current = by_source.get(row.source_path)
        if current is None or _rank(row) > _rank(current):
            by_source[row.source_path] = row
    return tuple(by_source[key] for key in sorted(by_source))


merge_rows = _merge_rows


def _rank(row: ExtractedDocument) -> int:
    if row.status == "success":
        if row.route == "docling_structured":
            return 40
        return 30 if row.route == "pdf_structured_ocr" else 20
    if row.status == "failed":
        return 10
    return 0


if __name__ == "__main__":
    app()
