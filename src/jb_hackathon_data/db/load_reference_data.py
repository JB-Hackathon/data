"""Load raw reference documents and RAG chunks into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import Connection
from psycopg.types.json import Jsonb


DEFAULT_DATABASE_URL = "postgresql://jbuser:jbpass@localhost:5432/jbdb"
DEFAULT_ISSUING_AUTHORITY = "unknown"
DEFAULT_DOCUMENT_TYPE = "other"
DOCUMENT_SUFFIXES = {".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".txt", ".md"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class LoadStats:
    documents: int
    chunks: int


def main() -> None:
    args = _parse_args()
    database_url = args.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL

    stats = load_reference_data(
        database_url=database_url,
        raw_dir=args.raw_dir,
        metadata_path=args.metadata_path,
        chunks_path=args.chunks_path,
        reset=args.reset,
    )

    print(f"Loaded reference_documents: {stats.documents}")
    print(f"Loaded reference_document_chunks: {stats.chunks}")


def load_reference_data(
    *,
    database_url: str,
    raw_dir: Path,
    metadata_path: Path,
    chunks_path: Path,
    reset: bool,
) -> LoadStats:
    raw_files = sorted(
        path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    )
    metadata_by_path = _load_reference_document_metadata(metadata_path)

    with psycopg.connect(database_url) as conn:
        if reset:
            _reset_reference_tables(conn)

        document_ids = _upsert_reference_documents(
            conn,
            raw_files=raw_files,
            raw_dir=raw_dir,
            metadata_by_path=metadata_by_path,
        )
        conn.commit()
        print(f"Upserted reference_documents: {len(document_ids)}")

        chunks = _upsert_reference_chunks(conn, chunks_path=chunks_path, document_ids=document_ids)
        conn.commit()

    return LoadStats(documents=len(document_ids), chunks=chunks)


DbConnection = Connection[tuple[object, ...]]


def _reset_reference_tables(conn: DbConnection) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE reference_document_chunks, reference_documents RESTART IDENTITY CASCADE")


def _load_reference_document_metadata(metadata_path: Path) -> dict[str, JsonObject]:
    if not metadata_path.exists():
        return {}

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"{metadata_path} must contain a JSON array")

    metadata_by_path: dict[str, JsonObject] = {}
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"{metadata_path} item {index} must be an object")
        source_file_path = item.get("source_file_path")
        if not isinstance(source_file_path, str) or not source_file_path:
            raise RuntimeError(f"{metadata_path} item {index} must contain source_file_path")
        metadata_by_path[source_file_path] = item

    return metadata_by_path


def _upsert_reference_documents(
    conn: DbConnection,
    *,
    raw_files: Iterable[Path],
    raw_dir: Path,
    metadata_by_path: dict[str, JsonObject],
) -> dict[str, int]:
    document_ids: dict[str, int] = {}

    with conn.cursor() as cur:
        for raw_file in raw_files:
            source_file_path = _source_file_path(raw_file, raw_dir=raw_dir)
            metadata = metadata_by_path.get(source_file_path, {})
            document_type = _metadata_str(metadata, "document_type") or DEFAULT_DOCUMENT_TYPE
            title = _metadata_str(metadata, "title") or raw_file.stem
            issuing_authority = _metadata_str(metadata, "issuing_authority") or DEFAULT_ISSUING_AUTHORITY
            issued_date = _metadata_date(metadata, "issued_date")

            cur.execute(
                """
                INSERT INTO reference_documents (
                  document_type,
                  title,
                  issuing_authority,
                  issued_date,
                  source_file_path
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_file_path)
                DO UPDATE SET
                  document_type = EXCLUDED.document_type,
                  title = EXCLUDED.title,
                  issuing_authority = EXCLUDED.issuing_authority,
                  issued_date = EXCLUDED.issued_date
                RETURNING document_id
                """,
                (
                    document_type,
                    title,
                    issuing_authority,
                    issued_date,
                    source_file_path,
                ),
            )
            document_id = cur.fetchone()
            if document_id is None:
                raise RuntimeError(f"failed to upsert reference document: {source_file_path}")
            raw_document_id = document_id[0]
            if not isinstance(raw_document_id, int) or isinstance(raw_document_id, bool):
                raise RuntimeError(f"document_id must be an integer: {source_file_path}")

            document_ids[source_file_path] = raw_document_id

    return document_ids


def _metadata_str(metadata: JsonObject, key: str) -> str | None:
    value = metadata.get(key)
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(f"reference document metadata field {key} must be null or a string")


def _metadata_date(metadata: JsonObject, key: str) -> str | None:
    value = _metadata_str(metadata, key)
    if value is None or DATE_PATTERN.match(value):
        return value
    return None


def _upsert_reference_chunks(
    conn: DbConnection,
    *,
    chunks_path: Path,
    document_ids: dict[str, int],
) -> int:
    inserted = 0

    with chunks_path.open(encoding="utf-8") as file, conn.cursor() as cur:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue

            payload = json.loads(line)
            source_path = _required_str(payload, "source_path", line_no=line_no)
            document_id = document_ids.get(source_path)
            if document_id is None:
                raise RuntimeError(f"line {line_no}: source_path not found in raw files: {source_path}")

            chunk_index = _required_int(payload, "chunk_index", line_no=line_no)
            chunk_text = _required_str(payload, "text", line_no=line_no)
            search_text = _optional_str(payload, "search_text")
            embedding = _required_embedding(payload, line_no=line_no)
            metadata = _chunk_metadata(payload)

            cur.execute(
                """
                INSERT INTO reference_document_chunks (
                  document_id,
                  chunk_index,
                  chunk_text,
                  search_text,
                  embedding,
                  metadata
                )
                VALUES (%s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT (document_id, chunk_index)
                DO UPDATE SET
                  chunk_text = EXCLUDED.chunk_text,
                  search_text = EXCLUDED.search_text,
                  embedding = EXCLUDED.embedding,
                  metadata = EXCLUDED.metadata
                """,
                (
                    document_id,
                    chunk_index,
                    chunk_text,
                    search_text,
                    _vector_literal(embedding),
                    Jsonb(metadata),
                ),
            )
            inserted += 1
            if inserted % 100 == 0:
                print(f"Upserted reference_document_chunks: {inserted}")

    return inserted


def _chunk_metadata(payload: JsonObject) -> JsonObject:
    metadata_keys = (
        "chunk_id",
        "source_id",
        "source_path",
        "heading_path",
        "page_refs",
        "bbox_refs",
        "token_count",
        "embedding_model_id",
    )
    return {key: payload[key] for key in metadata_keys if key in payload}


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _source_file_path(path: Path, *, raw_dir: Path) -> str:
    try:
        relative_path = path.relative_to(raw_dir)
    except ValueError:
        relative_path = Path(path.name)
    return Path(raw_dir.name, relative_path).as_posix()


def _required_str(payload: JsonObject, key: str, *, line_no: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"line {line_no}: {key} must be a string")
    return value


def _optional_str(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(f"{key} must be null or a string")


def _required_int(payload: JsonObject, key: str, *, line_no: int) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"line {line_no}: {key} must be an integer")
    return value


def _required_embedding(payload: JsonObject, *, line_no: int) -> list[float]:
    value = payload.get("embedding")
    if not isinstance(value, list) or not value:
        message = (
            f"line {line_no}: embedding must be a non-empty list. Run "
            "`python -m jb_hackathon_data.embed_google_chunks` before loading chunks into DB."
        )
        raise RuntimeError(message)

    values: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise RuntimeError(f"line {line_no}: embedding items must be numbers")
        values.append(float(item))

    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or local jbdb.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw"),
        help="Directory containing source reference documents.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("raw/reference_documents.json"),
        help="JSON file containing document metadata extracted from README.xlsx.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("outputs/chunking/rag_chunks.jsonl"),
        help="JSONL file containing RAG chunks with embeddings.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate reference document tables before loading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
