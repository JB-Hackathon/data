"""Docling HybridChunker defaults for the MVP RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc.document import DoclingDocument
from typing_extensions import override

from .docling_hybrid_chunker import DoclingHybridChunkerAdapter

DEFAULT_TOKENIZER_MODEL_ID = "BAAI/bge-m3"
MAX_CHUNK_TOKENS = 2048
CHUNK_OVERLAP_TOKENS = 0


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    @override
    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


def build_docling_chunker(
    *,
    tokenizer_model_id: str = DEFAULT_TOKENIZER_MODEL_ID,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> DoclingHybridChunkerAdapter:
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_model_id),
        max_tokens=max_tokens,
    )
    return DoclingHybridChunkerAdapter(
        HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
            repeat_table_header=True,
            serializer_provider=MarkdownTableSerializerProvider(),
        )
    )


def build_token_counter(*, tokenizer_model_id: str = DEFAULT_TOKENIZER_MODEL_ID) -> HuggingFaceTokenCounter:
    from transformers import AutoTokenizer

    return HuggingFaceTokenCounter(
        tokenizer=HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_model_id),
            max_tokens=MAX_CHUNK_TOKENS,
        )
    )


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenCounter:
    tokenizer: HuggingFaceTokenizer

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)
