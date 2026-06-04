from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from jb_hackathon_data.chunk_models import RagChunk
from jb_hackathon_data.models import DocumentBlock, ExtractedDocument, SourceId
from jb_hackathon_data import run_chunk_pipeline
from jb_hackathon_data.run_chunk_pipeline import app


def test_run_chunk_pipeline_writes_chunks_and_summary(tmp_path: Path) -> None:
    sidecar = tmp_path / "docling.json"
    sidecar.write_text("{}", encoding="utf-8")
    extraction = tmp_path / "raw_documents.jsonl"
    row = ExtractedDocument(
        source_id=SourceId("hwp-source-0001"),
        source_path="raw/sample.hwp",
        source_sha256="d" * 64,
        extension=".hwp",
        route="hwp_parser",
        parser_name="unhwp",
        parser_version="0.5.0",
        status="success",
        text="본문",
        pages_or_sections=(),
        error=None,
        blocks=(
            DocumentBlock(
                block_id="raw/sample.hwp:0",
                type="paragraph",
                text="본문",
                page_no=None,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
        ),
    )
    extraction.write_text(row.to_json_text() + "\n", encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    summary = tmp_path / "summary.json"

    result = CliRunner().invoke(
        app,
        [
            "--results",
            str(extraction),
            "--out",
            str(chunks),
            "--summary-out",
            str(summary),
        ],
    )

    assert result.exit_code == 0
    assert chunks.exists()
    assert '"chunks": 1' in summary.read_text(encoding="utf-8")


def test_run_chunk_pipeline_applies_default_filter_and_reports_summary(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    extraction = tmp_path / "raw_documents.jsonl"
    row = ExtractedDocument(
        source_id=SourceId("pdf-source-0001"),
        source_path="raw/sample.pdf",
        source_sha256="e" * 64,
        extension=".pdf",
        route="docling_structured",
        parser_name="docling",
        parser_version="2.96.1",
        status="success",
        text="본문",
        pages_or_sections=(),
        error=None,
        blocks=(),
    )
    extraction.write_text(row.to_json_text() + "\n", encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    summary = tmp_path / "summary.json"

    def chunk_rows(_: tuple[ExtractedDocument, ...], *, tokenizer_model_id: str) -> tuple[RagChunk, ...]:
        return (
            _chunk(index=0, text="본문", token_count=30, tokenizer_model_id=tokenizer_model_id),
            _chunk(index=1, text="<!-- image -->", token_count=6, tokenizer_model_id=tokenizer_model_id),
        )

    monkeypatch.setattr(run_chunk_pipeline, "_chunk_rows", chunk_rows)

    result = CliRunner().invoke(
        app,
        [
            "--results",
            str(extraction),
            "--out",
            str(chunks),
            "--summary-out",
            str(summary),
        ],
    )

    assert result.exit_code == 0
    summary_text = summary.read_text(encoding="utf-8")
    assert '"chunks_before_filter": 2' in summary_text
    assert '"filtered_chunks": 1' in summary_text
    assert '"chunks": 1' in summary_text
    assert "<!-- image -->" not in chunks.read_text(encoding="utf-8")


def _chunk(
    *,
    index: int,
    text: str,
    token_count: int,
    tokenizer_model_id: str,
) -> RagChunk:
    return RagChunk.from_parts(
        source_id="source-1",
        source_path="raw/sample.pdf",
        source_sha256="a" * 64,
        route="docling_structured",
        parser_name="docling",
        parser_version="2.96.1",
        chunk_index=index,
        chunk_type="section",
        heading_path=("기준",),
        page_refs=(1,),
        bbox_refs=(),
        docling_refs=(f"#/texts/{index}",),
        block_ids=(),
        tokenizer_model_id=tokenizer_model_id,
        token_count=token_count,
        text=text,
        search_text=text,
        table_markdown=None,
        metadata=(("chunker", "docling_hybrid"),),
    )
