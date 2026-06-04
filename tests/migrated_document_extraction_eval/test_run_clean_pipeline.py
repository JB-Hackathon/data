from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from jb_hackathon_data import run_clean_pipeline
from jb_hackathon_data.models import DocumentBlock, ExtractedDocument, PageSection, Route, SourceId, Status, Structure


def _make_row(
    path: Path,
    *,
    route: Route,
    status: Status,
    text: str,
    parser_name: str,
    error: str | None = None,
) -> ExtractedDocument:
    extension = path.suffix.lower()
    return ExtractedDocument(
        source_id=SourceId(path.name.ljust(16, "0")[:16]),
        source_path=path.name,
        source_sha256="0" * 64,
        extension=extension,
        route=route,
        parser_name=parser_name,
        parser_version="0.0.0",
        status=status,
        text=text,
        pages_or_sections=(
            PageSection(
                page_no=None,
                section_path="document",
                text=text,
                tables=(),
                structure=Structure((), ()),
            ),
        ),
        error=error,
        blocks=(
            DocumentBlock(
                block_id=f"{path.name}:0",
                type="paragraph",
                text=text,
                page_no=1,
                bbox=None,
                level=None,
                section_path=(),
                table_markdown=None,
                metadata=(),
            ),
        )
        if text
        else (),
    )


def _make_raw_documents(base_dir: Path, *, pdf_count: int = 18, hwp_count: int = 5) -> tuple[Path, ...]:
    paths: list[Path] = []
    for idx in range(pdf_count):
        path = base_dir / f"raw-{idx:02d}.pdf"
        _ = path.write_bytes(b"")
        paths.append(path)
    for idx in range(hwp_count):
        path = base_dir / f"raw-hwp-{idx:02d}.hwp"
        _ = path.write_bytes(b"")
        paths.append(path)
    return tuple(paths)


def _docling(path: Path) -> ExtractedDocument:
    return _make_row(
        path,
        route="docling_structured",
        status="success",
        parser_name="docling",
        text=f"docling text for {path.name}",
    )


def _hwp(path: Path) -> ExtractedDocument:
    return _make_row(
        path,
        route="hwp_parser",
        status="success",
        parser_name="hwp-hwpx-parser",
        text=f"hwp text for {path.name}",
    )


def test_missing_raw_directory_exits_with_exit_code_2() -> None:
    result = CliRunner().invoke(
        run_clean_pipeline.app,
        [
            "--raw-dir",
            "does-not-exist",
            "--out",
            "outputs/raw_documents.jsonl",
        ],
    )

    assert result.exit_code == 2
    assert "raw directory not found" in result.output


def test_docling_is_primary_for_pdf_documents(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _ = _make_raw_documents(raw_dir)

    calls: dict[str, int] = {"docling": 0, "hwp": 0}

    def extract_docling_document(path: Path, *, sidecar_dir: Path | None = None) -> ExtractedDocument:
        _ = sidecar_dir
        calls["docling"] += 1
        return _docling(path)

    def extract_hwp(path: Path) -> ExtractedDocument:
        calls["hwp"] += 1
        return _hwp(path)

    monkeypatch.setattr(run_clean_pipeline, "extract_docling_document", extract_docling_document)
    monkeypatch.setattr(run_clean_pipeline, "extract_hwp", extract_hwp)

    output_file = tmp_path / "outputs" / "raw_documents.jsonl"
    result = CliRunner().invoke(
        run_clean_pipeline.app,
        [
            "--raw-dir",
            str(raw_dir),
            "--out",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert calls["docling"] == 18
    assert calls["hwp"] == 5
    assert "documents=23" in result.output
    assert "final_results=23" in result.output
    files = sorted(path.name for path in output_file.parent.iterdir() if path.is_file())
    assert files == ["raw_documents.jsonl"]
    assert len(output_file.read_text(encoding="utf-8").splitlines()) == 23

