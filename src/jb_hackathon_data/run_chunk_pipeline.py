"""Run extraction-result to RAG-chunk conversion."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from .block_fallback_chunker import chunk_blocks_with_fallback
from .chunk_filters import apply_default_chunk_filter
from .chunk_models import RagChunk, write_chunk_jsonl
from .docling_chunk_config import DEFAULT_TOKENIZER_MODEL_ID, MAX_CHUNK_TOKENS, build_docling_chunker, build_token_counter
from .docling_hybrid_chunker import DoclingHybridChunkerAdapter, chunk_docling_sidecar
from .io_utils import ExtractionInputError, read_jsonl, resolve_root_path
from .models import ExtractedDocument
from .validate_chunks import validate_chunk_rows

app = typer.Typer(add_completion=False)


@app.command()
def main(
    results: Annotated[str, typer.Option("--results")],
    out: Annotated[str, typer.Option("--out")],
    summary_out: Annotated[str, typer.Option("--summary-out")],
    tokenizer_model_id: Annotated[str, typer.Option("--tokenizer-model-id")] = DEFAULT_TOKENIZER_MODEL_ID,
) -> None:
    try:
        results_path = resolve_root_path(results)
        out_path = resolve_root_path(out)
        summary_path = resolve_root_path(summary_out)
        rows = read_jsonl(results_path)
        raw_chunks = _chunk_rows(rows, tokenizer_model_id=tokenizer_model_id)
        chunks = apply_default_chunk_filter(raw_chunks)
        validate_chunk_rows(chunks, max_tokens=MAX_CHUNK_TOKENS)
        write_chunk_jsonl(str(out_path), chunks)
        _write_summary(summary_path, rows=rows, chunks=chunks, chunks_before_filter=len(raw_chunks))
    except ExtractionInputError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"documents={len(rows)}")
    typer.echo(f"chunks={len(chunks)}")


def _chunk_rows(rows: tuple[ExtractedDocument, ...], *, tokenizer_model_id: str) -> tuple[RagChunk, ...]:
    docling_chunker: DoclingHybridChunkerAdapter | None = None
    fallback_count_tokens: Callable[[str], int] | None = None
    chunks: list[RagChunk] = []
    for row in rows:
        if row.status != "success":
            continue
        if row.docling_document_path:
            if docling_chunker is None:
                docling_chunker = build_docling_chunker(tokenizer_model_id=tokenizer_model_id)
                fallback_count_tokens = docling_chunker.count_tokens
            chunks.extend(
                chunk_docling_sidecar(
                    row,
                    _resolve_sidecar_path(row.docling_document_path),
                    docling_chunker,
                    tokenizer_model_id=tokenizer_model_id,
                )
            )
        else:
            if fallback_count_tokens is None:
                fallback_count_tokens = build_token_counter(tokenizer_model_id=tokenizer_model_id).count_tokens
            chunks.extend(
                chunk_blocks_with_fallback(
                    row,
                    tokenizer_model_id=tokenizer_model_id,
                    max_tokens=MAX_CHUNK_TOKENS,
                    count_tokens=fallback_count_tokens,
                )
            )
    return tuple(chunks)


def _resolve_sidecar_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else resolve_root_path(path_text)


def _write_summary(
    path: Path,
    *,
    rows: tuple[ExtractedDocument, ...],
    chunks: tuple[RagChunk, ...],
    chunks_before_filter: int,
) -> None:
    payload = {
        "documents": len(rows),
        "success_documents": sum(1 for row in rows if row.status == "success"),
        "chunks_before_filter": chunks_before_filter,
        "filtered_chunks": chunks_before_filter - len(chunks),
        "chunks": len(chunks),
        "docling_chunks": sum(1 for chunk in chunks if chunk.docling_refs),
        "fallback_chunks": sum(1 for chunk in chunks if chunk.block_ids),
        "max_tokens": MAX_CHUNK_TOKENS,
        "overlap_tokens": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
