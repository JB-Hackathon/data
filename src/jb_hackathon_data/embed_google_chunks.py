"""Rewrite RAG chunk embeddings with Google Gen AI embeddings."""

from __future__ import annotations

from typing import Annotated

import typer

from .embedding_pipeline import (
    EmbeddingPipelineError,
    PartialEmbeddingProgressError,
    RewriteStats,
    ensure_google_api_key,
    load_env_file,
    rewrite_embedding_file,
)
from .google_embedding_provider import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_POLL_INTERVAL_SECONDS,
    build_google_provider,
)
from .io_utils import resolve_root_path

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: Annotated[str, typer.Option("--input")] = "outputs/chunking/rag_chunks.jsonl",
    output_path: Annotated[str | None, typer.Option("--out")] = None,
    work_path: Annotated[str | None, typer.Option("--work-path")] = None,
    env_file: Annotated[str, typer.Option("--env-file")] = ".env",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    output_dimensionality: Annotated[
        int,
        typer.Option("--output-dimensionality", min=1),
    ] = DEFAULT_OUTPUT_DIMENSIONALITY,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = DEFAULT_BATCH_SIZE,
    max_new_rows: Annotated[int | None, typer.Option("--max-new-rows", min=1)] = None,
    api_mode: Annotated[str, typer.Option("--api-mode")] = "batch",
    poll_interval_seconds: Annotated[
        int,
        typer.Option("--poll-interval-seconds", min=1),
    ] = DEFAULT_POLL_INTERVAL_SECONDS,
) -> None:
    try:
        stats = _rewrite_google_embeddings(
            input_path=input_path,
            output_path=output_path,
            work_path=work_path,
            env_file=env_file,
            model=model,
            output_dimensionality=output_dimensionality,
            batch_size=batch_size,
            max_new_rows=max_new_rows,
            api_mode=api_mode,
            poll_interval_seconds=poll_interval_seconds,
        )
    except PartialEmbeddingProgressError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc
    except EmbeddingPipelineError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"rows={stats.rows}")
    typer.echo(f"embedding_model_id={stats.model_id}")
    typer.echo(f"embedding_dimension={stats.dimension}")


def _rewrite_google_embeddings(
    *,
    input_path: str,
    output_path: str | None,
    work_path: str | None,
    env_file: str,
    model: str,
    output_dimensionality: int,
    batch_size: int,
    max_new_rows: int | None,
    api_mode: str,
    poll_interval_seconds: int,
) -> RewriteStats:
    resolved_input = resolve_root_path(input_path)
    resolved_output = resolved_input if output_path is None else resolve_root_path(output_path)
    resolved_work = None if work_path is None else resolve_root_path(work_path)
    load_env_file(resolve_root_path(env_file))
    ensure_google_api_key()
    provider = build_google_provider(
        api_mode=api_mode,
        model=model,
        output_dimensionality=output_dimensionality,
        poll_interval_seconds=poll_interval_seconds,
    )
    return rewrite_embedding_file(
        input_path=resolved_input,
        output_path=resolved_output,
        provider=provider,
        batch_size=batch_size,
        expected_dimension=output_dimensionality,
        work_path=resolved_work,
        max_new_rows=max_new_rows,
    )


if __name__ == "__main__":
    app()
