from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from jb_hackathon_data.embedding_pipeline import (
    EmbeddingPipelineError,
    PartialEmbeddingProgressError,
    load_env_file,
    rewrite_embedding_file,
)


class FakeEmbeddingProvider:
    model_id: str = "google/fake-embedding:3"

    def __init__(self, vectors: tuple[tuple[float, ...], ...]) -> None:
        self._vectors: tuple[tuple[float, ...], ...] = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return self._vectors[: len(texts)]


def test_rewrite_embedding_file_replaces_only_embedding_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "rag_chunks.jsonl"
    output_path = tmp_path / "rag_chunks.google.jsonl"
    rows = (
        _row("chk_1", "검색 본문 1", (0.1, 0.2)),
        _row("chk_2", "검색 본문 2", (0.3, 0.4)),
    )
    _write_rows(input_path, rows)
    provider = FakeEmbeddingProvider(((1.0, 1.1, 1.2), (2.0, 2.1, 2.2)))

    stats = rewrite_embedding_file(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        batch_size=2,
        expected_dimension=3,
    )

    rewritten_rows = _read_rows(output_path)
    assert stats.rows == 2
    assert provider.calls == [("검색 본문 1", "검색 본문 2")]
    assert [_without_embedding_fields(row) for row in rewritten_rows] == [
        _without_embedding_fields(row) for row in rows
    ]
    assert [row["embedding_model_id"] for row in rewritten_rows] == [
        "google/fake-embedding:3",
        "google/fake-embedding:3",
    ]
    assert [row["embedding"] for row in rewritten_rows] == [
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
    ]


def test_rewrite_embedding_file_rejects_missing_search_text_without_output(tmp_path: Path) -> None:
    input_path = tmp_path / "rag_chunks.jsonl"
    output_path = tmp_path / "rag_chunks.google.jsonl"
    row = _row("chk_1", "검색 본문", (0.1, 0.2))
    del row["search_text"]
    _write_rows(input_path, (row,))
    provider = FakeEmbeddingProvider(((1.0, 1.1, 1.2),))

    with pytest.raises(EmbeddingPipelineError, match="search_text"):
        rewrite_embedding_file(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            batch_size=1,
            expected_dimension=3,
        )

    assert not output_path.exists()


def test_rewrite_embedding_file_rejects_wrong_embedding_dimension_without_output(tmp_path: Path) -> None:
    input_path = tmp_path / "rag_chunks.jsonl"
    output_path = tmp_path / "rag_chunks.google.jsonl"
    _write_rows(input_path, (_row("chk_1", "검색 본문", (0.1, 0.2)),))
    provider = FakeEmbeddingProvider(((1.0, 1.1),))

    with pytest.raises(EmbeddingPipelineError, match="dimension"):
        rewrite_embedding_file(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            batch_size=1,
            expected_dimension=3,
        )

    assert not output_path.exists()


def test_rewrite_embedding_file_resumes_from_work_file(tmp_path: Path) -> None:
    input_path = tmp_path / "rag_chunks.jsonl"
    output_path = tmp_path / "rag_chunks.google.jsonl"
    work_path = tmp_path / "rag_chunks.work.jsonl"
    source_rows = (
        _row("chk_1", "검색 본문 1", (0.1, 0.2)),
        _row("chk_2", "검색 본문 2", (0.3, 0.4)),
    )
    work_rows = (
        _google_row(source_rows[0], (9.0, 9.1, 9.2)),
        source_rows[1],
    )
    _write_rows(input_path, source_rows)
    _write_rows(work_path, work_rows)
    provider = FakeEmbeddingProvider(((2.0, 2.1, 2.2),))

    stats = rewrite_embedding_file(
        input_path=input_path,
        output_path=output_path,
        provider=provider,
        batch_size=1,
        expected_dimension=3,
        work_path=work_path,
    )

    rewritten_rows = _read_rows(output_path)
    assert stats.rows == 2
    assert provider.calls == [("검색 본문 2",)]
    assert [row["embedding"] for row in rewritten_rows] == [
        [9.0, 9.1, 9.2],
        [2.0, 2.1, 2.2],
    ]
    assert not work_path.exists()


def test_rewrite_embedding_file_saves_partial_work_without_output(tmp_path: Path) -> None:
    input_path = tmp_path / "rag_chunks.jsonl"
    output_path = tmp_path / "rag_chunks.google.jsonl"
    work_path = tmp_path / "rag_chunks.work.jsonl"
    rows = (
        _row("chk_1", "검색 본문 1", (0.1, 0.2)),
        _row("chk_2", "검색 본문 2", (0.3, 0.4)),
    )
    _write_rows(input_path, rows)
    provider = FakeEmbeddingProvider(((1.0, 1.1, 1.2),))

    with pytest.raises(PartialEmbeddingProgressError, match="partial progress"):
        rewrite_embedding_file(
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            batch_size=1,
            expected_dimension=3,
            work_path=work_path,
            max_new_rows=1,
        )

    work_rows = _read_rows(work_path)
    assert not output_path.exists()
    assert provider.calls == [("검색 본문 1",)]
    assert [row["embedding_model_id"] for row in work_rows] == [
        "google/fake-embedding:3",
        "nlpai-lab/KURE-v1",
    ]


def test_load_env_file_accepts_utf8_bom_from_powershell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("\ufeffGEMINI_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    load_env_file(env_path)

    assert os.environ["GEMINI_API_KEY"] == "test-key"


def _row(chunk_id: str, search_text: str, embedding: tuple[float, ...]) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": "source-1",
        "source_path": "raw/sample.pdf",
        "chunk_index": 1,
        "heading_path": ["기준"],
        "page_refs": [1],
        "bbox_refs": [{"left": 1.0, "top": 2.0, "right": 3.0, "bottom": 4.0}],
        "token_count": 12,
        "text": "원문 본문",
        "search_text": search_text,
        "embedding_model_id": "nlpai-lab/KURE-v1",
        "embedding": list(embedding),
    }


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            _ = file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            _ = file.write("\n")


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _without_embedding_fields(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"embedding_model_id", "embedding"}}


def _google_row(row: dict[str, object], embedding: tuple[float, ...]) -> dict[str, object]:
    rewritten = dict(row)
    rewritten["embedding_model_id"] = "google/fake-embedding:3"
    rewritten["embedding"] = list(embedding)
    return rewritten
