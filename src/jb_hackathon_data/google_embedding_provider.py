"""Google Gen AI embedding providers."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .embedding_pipeline import EmbeddingPipelineError, EmbeddingProvider

DEFAULT_MODEL: Final = "gemini-embedding-2"
DEFAULT_OUTPUT_DIMENSIONALITY: Final = 1024
DEFAULT_TASK_TYPE: Final = "RETRIEVAL_DOCUMENT"
DEFAULT_BATCH_SIZE: Final = 1024
DEFAULT_POLL_INTERVAL_SECONDS: Final = 5
EMBEDDING_MODEL_NAMESPACE: Final = "google"


@dataclass(frozen=True, slots=True)
class GoogleEmbeddingProvider:
    model: str
    output_dimensionality: int
    task_type: str = DEFAULT_TASK_TYPE

    @property
    def model_id(self) -> str:
        return f"{EMBEDDING_MODEL_NAMESPACE}/{self.model}:{self.output_dimensionality}"

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        from google import genai
        from google.genai import errors, types

        client = genai.Client()
        try:
            response = client.models.embed_content(
                model=self.model,
                contents=list(texts),
                config=types.EmbedContentConfig(
                    output_dimensionality=self.output_dimensionality,
                    task_type=self.task_type,
                ),
            )
        except errors.APIError as exc:
            raise EmbeddingPipelineError(f"Google embedding request failed: {exc}") from exc
        embeddings = response.embeddings
        if embeddings is None:
            raise EmbeddingPipelineError("Google embedding response did not contain embeddings")
        vectors: list[tuple[float, ...]] = []
        for index, embedding in enumerate(embeddings, start=1):
            values = embedding.values
            if values is None:
                raise EmbeddingPipelineError(f"Google embedding {index} did not contain values")
            vectors.append(tuple(float(value) for value in values))
        return tuple(vectors)


@dataclass(frozen=True, slots=True)
class GoogleBatchEmbeddingProvider:
    model: str
    output_dimensionality: int
    task_type: str = DEFAULT_TASK_TYPE
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS

    @property
    def model_id(self) -> str:
        return f"{EMBEDDING_MODEL_NAMESPACE}/{self.model}:{self.output_dimensionality}"

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        from google import genai
        from google.genai import errors

        client = genai.Client()
        try:
            job = client.batches.create_embeddings(
                model=self.model,
                src={
                    "inlined_requests": {
                        "contents": list(texts),
                        "config": {
                            "output_dimensionality": self.output_dimensionality,
                            "task_type": self.task_type,
                        },
                    },
                },
                config={"display_name": "jb-hackathon-rag-embeddings"},
            )
            job_name = job.name
            if job_name is None:
                raise EmbeddingPipelineError("Google embedding batch job did not return a name")
            while True:
                job = client.batches.get(name=job_name)
                state = str(job.state)
                if state.endswith("SUCCEEDED"):
                    break
                if state.endswith(("FAILED", "CANCELLED", "EXPIRED")):
                    raise EmbeddingPipelineError(f"Google embedding batch job ended with {state}")
                time.sleep(self.poll_interval_seconds)
        except errors.APIError as exc:
            raise EmbeddingPipelineError(f"Google embedding batch request failed: {exc}") from exc

        destination = job.dest
        if destination is None or destination.inlined_embed_content_responses is None:
            raise EmbeddingPipelineError("Google embedding batch job did not return inline responses")
        responses = destination.inlined_embed_content_responses
        if len(responses) != len(texts):
            raise EmbeddingPipelineError("embedding response count does not match input rows")
        vectors: list[tuple[float, ...]] = []
        for index, item in enumerate(responses, start=1):
            if item.error is not None:
                raise EmbeddingPipelineError(f"Google embedding batch item {index} failed: {item.error}")
            response = item.response
            if response is None or response.embedding is None or response.embedding.values is None:
                raise EmbeddingPipelineError(f"Google embedding batch item {index} did not contain values")
            vectors.append(tuple(float(value) for value in response.embedding.values))
        return tuple(vectors)


def build_google_provider(
    *,
    api_mode: str,
    model: str,
    output_dimensionality: int,
    poll_interval_seconds: int,
) -> EmbeddingProvider:
    match api_mode:
        case "batch":
            return GoogleBatchEmbeddingProvider(
                model=model,
                output_dimensionality=output_dimensionality,
                poll_interval_seconds=poll_interval_seconds,
            )
        case "sync":
            return GoogleEmbeddingProvider(model=model, output_dimensionality=output_dimensionality)
        case _:
            raise EmbeddingPipelineError("api_mode must be batch or sync")
