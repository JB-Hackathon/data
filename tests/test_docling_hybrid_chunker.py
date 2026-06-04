from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from jb_hackathon_data.docling_hybrid_chunker import ChunkerLike, chunk_docling_sidecar
from jb_hackathon_data.models import ExtractedDocument, SourceId
from typing_extensions import override


@dataclass(frozen=True, slots=True)
class FakeChunk:
    text: str
    headings: tuple[str, ...]
    docling_refs: tuple[str, ...]
    page_refs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FakeChunker(ChunkerLike[FakeChunk]):
    @override
    def chunk(self, _: Path, /) -> Iterable[FakeChunk]:
        return (
            FakeChunk(
                text="| 항목 | 내용 |\n| --- | --- |\n| 금리 | 최고 금리 표시 |",
                headings=("광고 기준",),
                docling_refs=("#/tables/0",),
                page_refs=(3,),
            ),
        )

    @override
    def contextualize(self, chunk: FakeChunk, /) -> str:
        return "\n".join((*chunk.headings, chunk.text))

    @override
    def count_tokens(self, text: str, /) -> int:
        return len(text.split())


def test_docling_sidecar_chunks_to_rag_chunks(tmp_path: Path) -> None:
    sidecar = tmp_path / "docling.json"
    sidecar.write_text("{}", encoding="utf-8")
    row = ExtractedDocument(
        source_id=SourceId("pdf-source-0001"),
        source_path="raw/sample.pdf",
        source_sha256="c" * 64,
        extension=".pdf",
        route="docling_structured",
        parser_name="docling",
        parser_version="2.96.1",
        status="success",
        text="table text",
        pages_or_sections=(),
        error=None,
        blocks=(),
        docling_document_path=sidecar.as_posix(),
    )

    chunks = chunk_docling_sidecar(row, sidecar, FakeChunker(), tokenizer_model_id="BAAI/bge-m3")

    assert len(chunks) == 1
    assert chunks[0].docling_refs == ("#/tables/0",)
    assert chunks[0].heading_path == ("광고 기준",)
    assert chunks[0].search_text.startswith("문서명: sample")
    assert chunks[0].table_markdown is not None


def test_docling_sidecar_search_text_uses_search_context_without_mutating_text(tmp_path: Path) -> None:
    sidecar = tmp_path / "docling.json"
    sidecar.write_text("{}", encoding="utf-8")
    row = ExtractedDocument(
        source_id=SourceId("pdf-source-0002"),
        source_path="raw/docling_report_2026.pdf",
        source_sha256="d" * 64,
        extension=".pdf",
        route="docling_structured",
        parser_name="docling",
        parser_version="2.96.1",
        status="success",
        text="table text",
        pages_or_sections=(),
        error=None,
        blocks=(),
        docling_document_path=sidecar.as_posix(),
    )

    chunk = chunk_docling_sidecar(row, sidecar, FakeChunker(), tokenizer_model_id="BAAI/bge-m3")[0]

    assert chunk.text == "| 항목 | 내용 |\n| --- | --- |\n| 금리 | 최고 금리 표시 |"
    assert chunk.search_text == (
        "문서명: docling_report_2026\n"
        "섹션: 광고 기준\n"
        "조항/섹션명: 광고 기준\n"
        "내용:\n| 항목 | 내용 |\n| --- | --- |\n| 금리 | 최고 금리 표시 |"
    )
