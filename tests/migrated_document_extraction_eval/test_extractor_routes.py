from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from jb_hackathon_data import hwp_extractors
from jb_hackathon_data.models import PageSection, SourceId, Structure, ExtractedDocument
from pytest import MonkeyPatch


def _touch(path: Path, content: bytes = b"") -> Path:
    _ = path.write_bytes(content)
    return path


def _hwp_result(path: Path, parser_name: str, text: str) -> ExtractedDocument:
    section = PageSection(page_no=None, section_path="document", text=text, tables=(), structure=Structure((), ()))
    return ExtractedDocument(
        source_id=SourceId("a" * 16),
        source_path=path.name,
        source_sha256="a" * 64,
        extension=path.suffix.lower(),
        route="hwp_parser",
        parser_name=parser_name,
        parser_version="0.0.0",
        status="success" if text else "failed",
        text=text,
        pages_or_sections=(section,) if text else (),
        error=None if text else f"{parser_name} returned empty text",
    )


def _stub_source_path(path: Path) -> str:
    return path.name


def _raise_unhwp_runtime_error(_path: Path, _sha: str, _first_error: Exception | None = None) -> NoReturn:
    raise RuntimeError("unhwp crashed")


def _raise_primary_runtime_error_for_count(_path: Path, _sha: str) -> NoReturn:
    raise RuntimeError("primary failed")


def _raise_value_error(*_args: object, **_kwargs: object) -> NoReturn:
    raise ValueError("fallback failed")


def _hwp_primary_success(path: Path, _sha: str) -> ExtractedDocument:
    return _hwp_result(path, "hwp-hwpx-parser", "primary text")


def _hwp_primary_retries_then_unhwp(path: Path, _sha: str) -> ExtractedDocument:
    return _hwp_result(path, "unhwp", "fallback text")


def _hwp_unhwp_success(path: Path, _sha: str, _first_error: Exception | None = None) -> ExtractedDocument:
    return _hwp_primary_retries_then_unhwp(path, _sha)


def test_extract_hwp_unhwp_failure_then_hwp_hwpx_parser_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("jb_hackathon_data.hwp_extractors.relative_source_path", _stub_source_path)
    monkeypatch.setattr("jb_hackathon_data.hwp_extractors._extract_with_unhwp", _raise_unhwp_runtime_error)
    monkeypatch.setattr("jb_hackathon_data.hwp_extractors._extract_with_hwp_hwpx_parser", _hwp_primary_success)

    path = _touch(tmp_path / "primary-success.hwp")
    result = hwp_extractors.extract_hwp(path)

    assert result.status == "success"
    assert result.parser_name == "hwp-hwpx-parser"
    assert result.text == "primary text"


def test_extract_hwp_unhwp_primary_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("jb_hackathon_data.hwp_extractors.relative_source_path", _stub_source_path)
    monkeypatch.setattr(
        "jb_hackathon_data.hwp_extractors._extract_with_unhwp",
        _hwp_unhwp_success,
    )

    path = _touch(tmp_path / "fallback-success.hwp")
    result = hwp_extractors.extract_hwp(path)

    assert result.status == "success"
    assert result.parser_name == "unhwp"
    assert result.text == "fallback text"


def test_extract_hwp_primary_and_fallback_fail(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("jb_hackathon_data.hwp_extractors.relative_source_path", _stub_source_path)
    monkeypatch.setattr("jb_hackathon_data.hwp_extractors._extract_with_hwp_hwpx_parser", _raise_primary_runtime_error_for_count)
    monkeypatch.setattr("jb_hackathon_data.hwp_extractors._extract_with_unhwp", _raise_value_error)

    path = _touch(tmp_path / "fallback-failed.hwp")
    result = hwp_extractors.extract_hwp(path)

    assert result.status == "failed"
    assert result.parser_name == "unhwp;hwp-hwpx-parser"
    assert result.error is not None
    assert "unhwp=ValueError: fallback failed" in result.error
    assert "hwp-hwpx-parser=" in result.error
    assert "RuntimeError: primary failed" in result.error

