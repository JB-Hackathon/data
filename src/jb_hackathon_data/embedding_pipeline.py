"""Shared JSONL embedding rewrite pipeline."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonRow = dict[str, JsonValue]


class EmbeddingPipelineError(RuntimeError):
    """Raised when chunk embeddings cannot be rewritten safely."""


class PartialEmbeddingProgressError(EmbeddingPipelineError):
    """Raised when partial work was saved but final output is intentionally untouched."""


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        """Stable model identifier written into each chunk row."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Return one embedding vector for each input text."""
        ...


@dataclass(frozen=True, slots=True)
class RewriteStats:
    rows: int
    dimension: int
    model_id: str


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = _parse_env_line(stripped, line_no, path)
        os.environ.setdefault(key, value)


def ensure_google_api_key() -> None:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    raise EmbeddingPipelineError("GEMINI_API_KEY or GOOGLE_API_KEY is required")


def rewrite_embedding_file(
    *,
    input_path: Path,
    output_path: Path,
    provider: EmbeddingProvider,
    batch_size: int,
    expected_dimension: int,
    work_path: Path | None = None,
    max_new_rows: int | None = None,
) -> RewriteStats:
    _validate_rewrite_options(
        batch_size=batch_size,
        expected_dimension=expected_dimension,
        max_new_rows=max_new_rows,
    )
    rows = _read_json_rows(input_path)
    search_texts = tuple(_required_search_text(row, row_index=index) for index, row in enumerate(rows, start=1))
    rewritten_rows = list(
        _load_work_rows(
            work_path=work_path,
            source_rows=rows,
            provider=provider,
            expected_dimension=expected_dimension,
        )
    )
    pending_indexes = tuple(
        index
        for index, row in enumerate(rewritten_rows)
        if not _has_expected_embedding(row=row, provider=provider, expected_dimension=expected_dimension)
    )
    selected_indexes = pending_indexes if max_new_rows is None else pending_indexes[:max_new_rows]
    if len(selected_indexes) < len(pending_indexes) and work_path is None:
        raise EmbeddingPipelineError("work_path is required when max_new_rows leaves unfinished rows")
    for batch_indexes in _index_batches(selected_indexes, size=batch_size):
        _rewrite_batch(
            rows=rows,
            rewritten_rows=rewritten_rows,
            search_texts=search_texts,
            batch_indexes=batch_indexes,
            provider=provider,
            expected_dimension=expected_dimension,
        )
        if work_path is not None:
            _write_json_rows_atomic(work_path, rewritten_rows)
    if len(selected_indexes) < len(pending_indexes):
        remaining = len(pending_indexes) - len(selected_indexes)
        raise PartialEmbeddingProgressError(
            f"partial progress saved to {work_path}: {remaining} rows remain before final output is written"
        )
    _write_json_rows_atomic(output_path, rewritten_rows)
    if work_path is not None and work_path != output_path and work_path.exists():
        work_path.unlink()
    return RewriteStats(rows=len(rewritten_rows), dimension=expected_dimension, model_id=provider.model_id)


def _validate_rewrite_options(
    *,
    batch_size: int,
    expected_dimension: int,
    max_new_rows: int | None,
) -> None:
    if batch_size < 1:
        raise EmbeddingPipelineError("batch_size must be positive")
    if expected_dimension < 1:
        raise EmbeddingPipelineError("expected_dimension must be positive")
    if max_new_rows is not None and max_new_rows < 1:
        raise EmbeddingPipelineError("max_new_rows must be positive")


def _rewrite_batch(
    *,
    rows: tuple[JsonRow, ...],
    rewritten_rows: list[JsonRow],
    search_texts: tuple[str, ...],
    batch_indexes: tuple[int, ...],
    provider: EmbeddingProvider,
    expected_dimension: int,
) -> None:
    batch_texts = tuple(search_texts[index] for index in batch_indexes)
    vectors = provider.embed_documents(batch_texts)
    if len(vectors) != len(batch_indexes):
        raise EmbeddingPipelineError("embedding response count does not match batch size")
    for row_index, vector in zip(batch_indexes, vectors, strict=True):
        rewritten_rows[row_index] = _with_embedding(
            row=rows[row_index],
            vector=vector,
            model_id=provider.model_id,
            expected_dimension=expected_dimension,
            row_index=row_index + 1,
        )


def _parse_env_line(line: str, line_no: int, path: Path) -> tuple[str, str]:
    key, separator, value = line.partition("=")
    clean_key = key.lstrip("\ufeff").strip()
    if not separator or not clean_key:
        raise EmbeddingPipelineError(f"invalid env line at {path}:{line_no}")
    return clean_key, _unquote_env_value(value.strip())


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_json_rows(path: Path) -> tuple[JsonRow, ...]:
    if not path.exists():
        raise EmbeddingPipelineError(f"input file not found: {path}")
    rows: list[JsonRow] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbeddingPipelineError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(payload, dict):
                raise EmbeddingPipelineError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(dict(payload))
    if not rows:
        raise EmbeddingPipelineError(f"input file has no rows: {path}")
    return tuple(rows)


def _required_search_text(row: Mapping[str, JsonValue], *, row_index: int) -> str:
    value = row.get("search_text")
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingPipelineError(f"row {row_index} search_text must be a non-empty string")
    return value


def _load_work_rows(
    *,
    work_path: Path | None,
    source_rows: tuple[JsonRow, ...],
    provider: EmbeddingProvider,
    expected_dimension: int,
) -> tuple[JsonRow, ...]:
    if work_path is None or not work_path.exists():
        return source_rows
    work_rows = _read_json_rows(work_path)
    if len(work_rows) != len(source_rows):
        raise EmbeddingPipelineError("work file row count does not match input rows")
    for index, (source_row, work_row) in enumerate(zip(source_rows, work_rows, strict=True), start=1):
        if _without_embedding_fields(source_row) != _without_embedding_fields(work_row):
            raise EmbeddingPipelineError(f"work file row {index} does not match input row")
        if not _has_expected_embedding(row=work_row, provider=provider, expected_dimension=expected_dimension):
            return work_rows[: index - 1] + source_rows[index - 1 :]
    return work_rows


def _has_expected_embedding(
    *,
    row: Mapping[str, JsonValue],
    provider: EmbeddingProvider,
    expected_dimension: int,
) -> bool:
    if row.get("embedding_model_id") != provider.model_id:
        return False
    embedding = row.get("embedding")
    if not isinstance(embedding, list) or len(embedding) != expected_dimension:
        return False
    return all(isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) for value in embedding)


def _index_batches(indexes: Sequence[int], *, size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(indexes), size):
        yield tuple(indexes[start : start + size])


def _without_embedding_fields(row: Mapping[str, JsonValue]) -> JsonRow:
    return {key: value for key, value in row.items() if key not in {"embedding_model_id", "embedding"}}


def _with_embedding(
    *,
    row: Mapping[str, JsonValue],
    vector: Sequence[float],
    model_id: str,
    expected_dimension: int,
    row_index: int,
) -> JsonRow:
    if len(vector) != expected_dimension:
        raise EmbeddingPipelineError(
            f"row {row_index} embedding dimension {len(vector)} does not match {expected_dimension}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingPipelineError(f"row {row_index} embedding contains non-finite values")
    rewritten = dict(row)
    rewritten["embedding_model_id"] = model_id
    rewritten["embedding"] = list(vector)
    return rewritten


def _write_json_rows_atomic(path: Path, rows: Sequence[Mapping[str, JsonValue]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        _write_json_rows(temp_path, rows)
        temp_path.replace(path)
    except OSError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise EmbeddingPipelineError(f"failed to write output file: {path}") from exc


def _write_json_rows(path: Path, rows: Sequence[Mapping[str, JsonValue]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            _ = file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            _ = file.write("\n")
