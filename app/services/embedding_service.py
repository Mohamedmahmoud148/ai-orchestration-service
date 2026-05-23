"""
app/services/embedding_service.py

Embedding service for the RAG pipeline.

Uses the OpenAI text-embedding-3-small model via the official AsyncOpenAI
client.  Falls back to TF-IDF-style keyword overlap similarity when no
OPENAI_API_KEY is configured.

Environment variables:
  OPENAI_API_KEY  — OpenAI API key (direct, not via OpenRouter).
                    OpenRouter does not support the Embeddings endpoint,
                    so we always call api.openai.com directly.

Public API:
  embed_text(text)           → list[float]
  embed_batch(texts)         → list[list[float]]
  cosine_similarity(a, b)    → float
"""
from __future__ import annotations

import math
import os
import re
from typing import List, Optional

from app.core.logging import logger

_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension


def _zeros(dim: int = _EMBEDDING_DIM) -> List[float]:
    return [0.0] * dim


class EmbeddingService:
    """
    Async embedding service.

    Instantiation is lightweight — the OpenAI client is lazily created on
    first use so that the service can be imported without side-effects even
    when OPENAI_API_KEY is absent.
    """

    def __init__(self) -> None:
        self._client: Optional[object] = None   # AsyncOpenAI or None
        self._fallback_mode = False
        self._api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            logger.warning(
                "EmbeddingService: OPENAI_API_KEY not set — "
                "falling back to keyword-overlap similarity (no real embeddings)."
            )
            self._fallback_mode = True

    # ──────────────────────────────────────────────────────────────────────
    #  Client initialisation (lazy)
    # ──────────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None and not self._fallback_mode:
            try:
                from openai import AsyncOpenAI  # type: ignore
                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    # Explicitly target OpenAI — not OpenRouter
                    base_url="https://api.openai.com/v1",
                )
                logger.info(
                    "EmbeddingService: AsyncOpenAI client initialised "
                    "(model=%s).", _OPENAI_EMBEDDING_MODEL
                )
            except ImportError:
                logger.error(
                    "EmbeddingService: 'openai' package not installed — "
                    "switching to fallback mode."
                )
                self._fallback_mode = True
        return self._client

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    async def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single text string."""
        if not text or not text.strip():
            return _zeros()

        if self._fallback_mode:
            return self._keyword_vector(text)

        client = self._get_client()
        if client is None:
            return self._keyword_vector(text)

        try:
            response = await client.embeddings.create(
                model=_OPENAI_EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.error("EmbeddingService.embed_text error: %s", exc)
            return self._keyword_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts (one API call)."""
        if not texts:
            return []

        # Filter empty strings; preserve order via index mapping
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        results: List[List[float]] = [_zeros() for _ in texts]

        if not non_empty:
            return results

        if self._fallback_mode:
            for i, t in non_empty:
                results[i] = self._keyword_vector(t)
            return results

        client = self._get_client()
        if client is None:
            for i, t in non_empty:
                results[i] = self._keyword_vector(t)
            return results

        try:
            indices, batch_texts = zip(*non_empty)
            response = await client.embeddings.create(
                model=_OPENAI_EMBEDDING_MODEL,
                input=list(batch_texts),
            )
            for order, item in enumerate(response.data):
                results[indices[order]] = item.embedding
            return results
        except Exception as exc:
            logger.error("EmbeddingService.embed_batch error: %s", exc)
            for i, t in non_empty:
                results[i] = self._keyword_vector(t)
            return results

    # ──────────────────────────────────────────────────────────────────────
    #  Similarity
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Returns a value in [-1, 1] (typically [0, 1] for text embeddings).
        Returns 0.0 if either vector is zero.
        """
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ──────────────────────────────────────────────────────────────────────
    #  Fallback: keyword-overlap TF-IDF-style vector
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _keyword_vector(text: str) -> List[float]:
        """
        Very lightweight keyword-frequency vector used when the OpenAI API
        is unavailable.

        Produces a fixed-size (_EMBEDDING_DIM) float vector by hashing each
        token to a bucket and accumulating normalised term frequencies.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return _zeros()

        vec = [0.0] * _EMBEDDING_DIM
        freq: dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1

        for tok, count in freq.items():
            bucket = hash(tok) % _EMBEDDING_DIM
            vec[bucket] += count / len(tokens)  # normalised TF

        # L2-normalise so cosine_similarity still makes sense
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude > 0:
            vec = [v / magnitude for v in vec]

        return vec


# ── Singleton ─────────────────────────────────────────────────────────────────
embedding_service = EmbeddingService()
