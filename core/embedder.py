"""
core/embedder.py

Wraps OpenAI's text-embedding-3-small for batch embedding of chunks.

Design choices:
  - text-embedding-3-small: best cost/quality ratio for retrieval tasks
  - Batching: OpenAI allows up to 2048 inputs per request — we batch at 100
    to stay safe and keep latency predictable
  - Returns raw float lists (ChromaDB expects this format)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI, RateLimitError, APIError

from core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
MAX_RETRIES = 3


class Embedder:
    """
    Converts text → dense vector embeddings using OpenAI.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(["text one", "text two"])
    """

    def __init__(self) -> None:
        settings.validate()
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts. Returns one vector per text.
        Automatically batches large lists.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            logger.debug(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} texts)")
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch with retry on rate limit."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )
                # Results come back in order
                return [item.embedding for item in response.data]

            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Embedding rate limited. Retrying in {wait}s...")
                time.sleep(wait)

            except APIError as e:
                raise RuntimeError(f"Embedding API error: {e}") from e

        raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries.")