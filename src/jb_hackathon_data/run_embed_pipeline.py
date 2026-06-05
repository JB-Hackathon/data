"""Compatibility entrypoint for Google RAG chunk embedding generation."""

from __future__ import annotations

from .embed_google_chunks import app


if __name__ == "__main__":
    app()
