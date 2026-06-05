"""Sample reference-document hybrid retrieval for RAG.

This module searches reference_document_chunks with both pgvector and
PostgreSQL full-text search, then combines both rankings with RRF.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import LiteralString

import psycopg
from psycopg.rows import dict_row

from jb_hackathon_data.embedding_pipeline import ensure_google_api_key, load_env_file
from jb_hackathon_data.google_embedding_provider import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    GoogleEmbeddingProvider,
)
from jb_hackathon_data.io_utils import resolve_root_path


DEFAULT_DATABASE_URL = "postgresql://jbuser:jbpass@localhost:5432/jbdb"
DEFAULT_QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: int
    document_id: int
    title: str
    document_type: str
    issuing_authority: str
    source_file_path: str
    chunk_index: int
    chunk_text: str
    rrf_score: float
    vector_rank: int | None
    text_rank: int | None
    vector_score: float | None
    text_score: float | None


def main() -> None:
    args = _parse_args()
    database_url = args.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    load_env_file(resolve_root_path(args.env_file))
    ensure_google_api_key()
    query_embedding = embed_query(
        args.query,
        model=args.embedding_model,
        output_dimensionality=args.output_dimensionality,
    )

    results = search_reference_documents(
        database_url=database_url,
        query=args.query,
        query_embedding=query_embedding,
        top_k=args.top_k,
        vector_limit=args.vector_limit,
        text_limit=args.text_limit,
        rrf_k=args.rrf_k,
        document_type=args.document_type,
        issuing_authority=args.issuing_authority,
    )

    print_results(results)
    if args.show_prompt:
        print("\n--- RAG prompt sample ---")
        print(build_rag_prompt(query=args.query, results=results))


def embed_query(
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
) -> list[float]:
    """Embed the user query with the same model used for stored chunks."""
    provider = GoogleEmbeddingProvider(
        model=model,
        output_dimensionality=output_dimensionality,
        task_type=DEFAULT_QUERY_TASK_TYPE,
    )
    vectors = provider.embed_documents((query,))
    if len(vectors) != 1:
        raise RuntimeError("Google query embedding response count does not match input")
    return list(vectors[0])


def search_reference_documents(
    *,
    database_url: str,
    query: str,
    query_embedding: list[float],
    top_k: int = 5,
    vector_limit: int = 30,
    text_limit: int = 30,
    rrf_k: int = DEFAULT_RRF_K,
    document_type: str | None = None,
    issuing_authority: str | None = None,
) -> list[SearchResult]:
    params: dict[str, object] = {
        "query": query,
        "query_embedding": _vector_literal(query_embedding),
        "top_k": top_k,
        "vector_limit": vector_limit,
        "text_limit": text_limit,
        "rrf_k": rrf_k,
        "document_type": document_type,
        "issuing_authority": issuing_authority,
    }

    with psycopg.connect(database_url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_hybrid_search_sql(), params)
        rows = cur.fetchall()

    return [_search_result_from_row(row) for row in rows]


def _search_result_from_row(row: Mapping[str, object]) -> SearchResult:
    return SearchResult(
        chunk_id=_int_field(row, "chunk_id"),
        document_id=_int_field(row, "document_id"),
        title=_str_field(row, "title"),
        document_type=_str_field(row, "document_type"),
        issuing_authority=_str_field(row, "issuing_authority"),
        source_file_path=_str_field(row, "source_file_path"),
        chunk_index=_int_field(row, "chunk_index"),
        chunk_text=_str_field(row, "chunk_text"),
        rrf_score=_float_field(row, "rrf_score"),
        vector_rank=_optional_int_field(row, "vector_rank"),
        text_rank=_optional_int_field(row, "text_rank"),
        vector_score=_optional_float_field(row, "vector_score"),
        text_score=_optional_float_field(row, "text_score"),
    )


def _hybrid_search_sql() -> LiteralString:
    return """
    WITH query AS (
      SELECT
        %(query_embedding)s::vector AS embedding,
        websearch_to_tsquery('simple', %(query)s) AS ts_query
    ),
    filtered_chunks AS (
      SELECT
        c.chunk_id,
        c.document_id,
        c.chunk_index,
        c.chunk_text,
        c.embedding,
        c.search_vector,
        d.title,
        d.document_type::text AS document_type,
        d.issuing_authority,
        d.source_file_path
      FROM reference_document_chunks c
      JOIN reference_documents d ON d.document_id = c.document_id
      WHERE (%(document_type)s::text IS NULL OR d.document_type::text = %(document_type)s::text)
        AND (%(issuing_authority)s::text IS NULL OR d.issuing_authority = %(issuing_authority)s::text)
    ),
    vector_matches AS (
      SELECT
        chunk_id,
        row_number() OVER (ORDER BY embedding <=> (SELECT embedding FROM query)) AS vector_rank,
        1 - (embedding <=> (SELECT embedding FROM query)) AS vector_score
      FROM filtered_chunks
      ORDER BY embedding <=> (SELECT embedding FROM query)
      LIMIT %(vector_limit)s
    ),
    text_matches AS (
      SELECT
        chunk_id,
        row_number() OVER (ORDER BY ts_rank_cd(search_vector, (SELECT ts_query FROM query)) DESC) AS text_rank,
        ts_rank_cd(search_vector, (SELECT ts_query FROM query)) AS text_score
      FROM filtered_chunks
      WHERE search_vector @@ (SELECT ts_query FROM query)
      ORDER BY ts_rank_cd(search_vector, (SELECT ts_query FROM query)) DESC
      LIMIT %(text_limit)s
    ),
    combined AS (
      SELECT
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        v.vector_rank,
        t.text_rank,
        v.vector_score,
        t.text_score,
        COALESCE(1.0 / (%(rrf_k)s + v.vector_rank), 0.0)
          + COALESCE(1.0 / (%(rrf_k)s + t.text_rank), 0.0) AS rrf_score
      FROM vector_matches v
      FULL OUTER JOIN text_matches t ON t.chunk_id = v.chunk_id
    )
    SELECT
      f.chunk_id,
      f.document_id,
      f.title,
      f.document_type,
      f.issuing_authority,
      f.source_file_path,
      f.chunk_index,
      f.chunk_text,
      c.rrf_score,
      c.vector_rank,
      c.text_rank,
      c.vector_score,
      c.text_score
    FROM combined c
    JOIN filtered_chunks f ON f.chunk_id = c.chunk_id
    ORDER BY c.rrf_score DESC, c.vector_rank NULLS LAST, c.text_rank NULLS LAST
    LIMIT %(top_k)s
    """


def build_rag_prompt(*, query: str, results: list[SearchResult]) -> str:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] {result.title}",
                    f"- type: {result.document_type}",
                    f"- authority: {result.issuing_authority}",
                    f"- source: {result.source_file_path}",
                    f"- chunk_index: {result.chunk_index}",
                    result.chunk_text,
                ]
            )
        )

    context = "\n\n".join(context_blocks)
    return f"""You are a compliance review assistant.
Answer the user query using only the reference context below.
If the context is insufficient, say what is missing.

Reference context:
{context}

User query:
{query}
"""


def print_results(results: list[SearchResult]) -> None:
    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result.title}")
        print(f"document_type={result.document_type} authority={result.issuing_authority}")
        print(f"source={result.source_file_path} chunk_index={result.chunk_index}")
        print(
        (
            "score="
            f"{result.rrf_score:.6f} "
            f"vector_rank={result.vector_rank} "
            f"text_rank={result.text_rank} "
            f"vector_score={_format_optional_score(result.vector_score)} "
            f"text_score={_format_optional_score(result.text_score)}"
        )
    )
        print(_preview(result.chunk_text))


def _preview(text: str, *, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _format_optional_score(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _int_field(row: Mapping[str, object], key: str) -> int:
    value = row[key]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RuntimeError(f"{key} must be an integer")


def _optional_int_field(row: Mapping[str, object], key: str) -> int | None:
    value = row[key]
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RuntimeError(f"{key} must be null or an integer")


def _str_field(row: Mapping[str, object], key: str) -> str:
    value = row[key]
    if isinstance(value, str):
        return value
    raise RuntimeError(f"{key} must be a string")


def _float_field(row: Mapping[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise RuntimeError(f"{key} must be numeric")


def _optional_float_field(row: Mapping[str, object], key: str) -> float | None:
    value = row[key]
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise RuntimeError(f"{key} must be null or numeric")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="User query to search against reference documents.")
    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection URL. Defaults to DATABASE_URL or local jbdb.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file containing GEMINI_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        help="Google embedding model. Must match the stored chunk embedding model.",
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=DEFAULT_OUTPUT_DIMENSIONALITY,
        help="Google embedding output dimension. Must match the stored vectors.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of final RRF results.")
    parser.add_argument("--vector-limit", type=int, default=30, help="Candidate size from vector search.")
    parser.add_argument("--text-limit", type=int, default=30, help="Candidate size from tsvector search.")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF rank constant.")
    parser.add_argument("--document-type", help="Optional reference_document_type filter.")
    parser.add_argument("--issuing-authority", help="Optional issuing_authority exact-match filter.")
    parser.add_argument("--show-prompt", action="store_true", help="Print a sample RAG prompt with retrieved context.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
