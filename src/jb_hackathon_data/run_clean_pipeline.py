from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .docling_extractors import extract_docling_document, is_docling_supported_extension
from .hwp_extractors import extract_hwp
from .io_utils import ExtractionInputError, iter_raw_documents, resolve_root_path, write_jsonl
from .merge_extraction_results import merge_rows
from .models import ExtractedDocument
from .text_cleaning import clean_document
from .validate_extraction_results import validate_extraction_rows, validate_output_directory


app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_dir: Annotated[str, typer.Option("--raw-dir")],
    out: Annotated[str, typer.Option("--out")],
) -> None:
    try:
        raw_path = resolve_root_path(raw_dir)
        out_path = resolve_root_path(out)
        sidecar_dir = out_path.parent / "docling_documents"
        raw_documents = iter_raw_documents(raw_path)
        extracted_rows = tuple(_extract_document(document, docling_sidecar_dir=sidecar_dir) for document in raw_documents)
        merged_rows = merge_rows(extracted_rows)
        cleaned_rows = tuple(clean_document(row) for row in merged_rows)
        write_jsonl(out_path, cleaned_rows)
        validate_extraction_rows(raw_documents, cleaned_rows)
        _ = validate_output_directory(out_path.parent)
    except ExtractionInputError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"documents={len(raw_documents)}")
    typer.echo(f"final_results={len(cleaned_rows)}")


def _extract_document(path: Path, *, docling_sidecar_dir: Path) -> ExtractedDocument:
    match path.suffix.lower():
        case extension if is_docling_supported_extension(extension):
            return extract_docling_document(path, sidecar_dir=docling_sidecar_dir)
        case ".hwp" | ".hwpx":
            return extract_hwp(path)
        case _:
            raise ExtractionInputError(f"unsupported file extension: {path.suffix.lower()}")


if __name__ == "__main__":
    app()
