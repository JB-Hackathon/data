"""Add embedding vectors to RAG chunk JSONL rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from .io_utils import ExtractionInputError, resolve_root_path


DEFAULT_EMBEDDING_MODEL_ID = "nlpai-lab/KURE-v1"
DEFAULT_BATCH_SIZE = 16

app = typer.Typer(add_completion=False)


@app.command()
def main(
    chunks: Annotated[str, typer.Option("--chunks")],
    out: Annotated[str, typer.Option("--out")],
    embedding_model_id: Annotated[str, typer.Option("--embedding-model-id")] = DEFAULT_EMBEDDING_MODEL_ID,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = DEFAULT_BATCH_SIZE,
) -> None:
    try:
        chunks_path = resolve_root_path(chunks)
        out_path = resolve_root_path(out)
        rows = _read_jsonl(chunks_path)
        embedded_rows = _embed_rows(rows, model_id=embedding_model_id, batch_size=batch_size)
        _write_jsonl(out_path, embedded_rows)
    except ExtractionInputError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    typer.echo(f"chunks={len(rows)}")
    typer.echo(f"embedding_model_id={embedding_model_id}")
    typer.echo(f"out={out_path}")


def _embed_rows(rows: list[dict[str, Any]], *, model_id: str, batch_size: int) -> list[dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ExtractionInputError(
            "sentence-transformers is required to generate embeddings. "
            "Install dependencies with `uv sync`, or add it with `uv add sentence-transformers`."
        ) from exc

    model = SentenceTransformer(model_id)
    texts = [_embedding_text(row, index=index) for index, row in enumerate(rows, start=1)]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embedded_rows: list[dict[str, Any]] = []
    for row, vector in zip(rows, vectors, strict=True):
        embedded = dict(row)
        embedded["embedding"] = [float(value) for value in vector.tolist()]
        embedded["embedding_model_id"] = model_id
        embedded_rows.append(embedded)

    return embedded_rows


def _embedding_text(row: dict[str, Any], *, index: int) -> str:
    search_text = row.get("search_text")
    text = row.get("text")
    if isinstance(search_text, str) and search_text.strip():
        return search_text
    if isinstance(text, str) and text.strip():
        return text
    raise ExtractionInputError(f"line {index}: search_text or text is required for embedding")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ExtractionInputError(f"line {line_no}: JSON row must be an object")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


if __name__ == "__main__":
    app()
