"""Filesystem helpers for extraction evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .models import ExtractedDocument, SourceId

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".hwp",
        ".hwpx",
    }
)


class ExtractionInputError(RuntimeError):
    """Raised when input files or result files are not usable."""


def project_root_from_package() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_root_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root_from_package() / path


def iter_raw_documents(raw_dir: Path) -> tuple[Path, ...]:
    if not raw_dir.exists():
        raise ExtractionInputError(f"raw directory not found: {raw_dir}")
    docs = tuple(
        sorted(
            path
            for path in raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )
    if not docs:
        raise ExtractionInputError(f"no supported raw documents found: {raw_dir}")
    return docs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id_from_sha(sha256: str) -> SourceId:
    return SourceId(sha256[:16])


def relative_source_path(path: Path) -> str:
    root = project_root_from_package()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def write_jsonl(path: Path, rows: Iterable[ExtractedDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            _ = file.write(row.to_json_text())
            _ = file.write("\n")


def read_jsonl(path: Path) -> tuple[ExtractedDocument, ...]:
    if not path.exists():
        raise ExtractionInputError(f"result file not found: {path}")
    rows: list[ExtractedDocument] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if line.strip():
                rows.append(_parse_document_json(line, line_no, path))
    return tuple(rows)


def _parse_document_json(line: str, line_no: int, path: Path) -> ExtractedDocument:
    from .parse_models import parse_extracted_document

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ExtractionInputError(f"invalid JSONL at {path}:{line_no}") from exc
    if not isinstance(payload, dict):
        raise ExtractionInputError(f"JSONL row must be an object at {path}:{line_no}")
    return parse_extracted_document(payload)
