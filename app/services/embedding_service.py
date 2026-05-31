"""
app/services/embedding_service.py

Embedding service for the RAG pipeline.

Provider priority (first available wins):
  1. OpenAI-compatible API — any provider that implements POST /embeddings.
     Configured via:
       OPENAI_API_KEY     — direct OpenAI key (most common)
       EMBEDDING_BASE_URL — base URL (default: https://api.openai.com/v1)
       EMBEDDING_MODEL    — model name (default: text-embedding-3-small)
     This covers: OpenAI directly, Azure OpenAI, Together AI, Voyage AI,
     any local server (LM Studio, Ollama, etc.).

  2. sentence-transformers — lightweight multilingual local embeddings.
     Used automatically if the `sentence-transformers` package is installed
     AND no API key is configured.
     Recommended model: paraphrase-multilingual-MiniLM-L12-v2 (420 MB).
     Covers Arabic text well.

  3. Keyword-overlap fallback — hash-bucket TF-IDF pseudo-embeddings.
     Deterministic, reproducible, zero deps, but semantically weak.
     Used when neither option above is available.

Environment variables:
  OPENAI_API_KEY     — required for provider 1
  EMBEDDING_BASE_URL — default: https://api.openai.com/v1
  EMBEDDING_MODEL    — default: text-embedding-3-small

Public API:
  embed_text(text)              → list[float]
  embed_batch(texts)            → list[list[float]]
  cosine_similarity(a, b)       → float
  is_using_real_embeddings()    → bool
  get_mode()                    → str   ("openai_api" | "sentence_transformers" | "keyword_fallback")
"""
from __future__ import annotations

import math
import re
from typing import List, Optional

from app.core.logging import logger

# Import settings lazily to avoid circular imports at module load time
def _get_settings():
    from app.core.config import settings
    return settings


# ── SentenceTransformer local model name ─────────────────────────────────────
# Multilingual, ~420 MB, works well for Arabic + English.
# Smaller alternative: paraphrase-multilingual-MiniLM-L12-v2 (same name).
_ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_ST_EMBEDDING_DIM = 384       # output dimension of the ST model above


class EmbeddingService:
    """
    Multi-provider embedding service.

    Mode is determined at construction time based on environment variables
    and available packages. The mode is logged at startup so you can always
    tell which provider is active.
    """

    def __init__(self) -> None:
        s = _get_settings()
        self._api_key: str = s.OPENAI_API_KEY or ""
        self._base_url: str = (s.EMBEDDING_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        self._model: str   = s.EMBEDDING_MODEL or "text-embedding-3-small"
        self._embedding_dim: int = 1536  # default for text-embedding-3-small

        self._openai_client = None     # AsyncOpenAI — lazy
        self._st_model = None          # SentenceTransformer — lazy

        # Determine mode
        if self._api_key:
            self._mode = "openai_api"
            logger.info(
                "EmbeddingService: mode=openai_api base=%s model=%s",
                self._base_url, self._model,
            )
        elif self._try_load_sentence_transformers():
            self._mode = "sentence_transformers"
            logger.info(
                "EmbeddingService: mode=sentence_transformers model=%s dim=%d",
                _ST_MODEL_NAME, _ST_EMBEDDING_DIM,
            )
            self._embedding_dim = _ST_EMBEDDING_DIM
        else:
            self._mode = "keyword_fallback"
            logger.warning(
                "EmbeddingService: mode=keyword_fallback — RAG semantic quality is DEGRADED. "
                "Set OPENAI_API_KEY (or install sentence-transformers) for real embeddings."
            )

    # ── Mode detection helpers ────────────────────────────────────────────────

    def _try_load_sentence_transformers(self) -> bool:
        """
        Try to load the SentenceTransformer model.
        Returns True if successful. Silent on failure.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(_ST_MODEL_NAME)
            return True
        except ImportError:
            return False
        except Exception as exc:
            logger.warning("EmbeddingService: sentence-transformers load failed — %s", exc)
            return False

    def _get_openai_client(self):
        if self._openai_client is None and self._api_key:
            try:
                from openai import AsyncOpenAI  # type: ignore
                self._openai_client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
                logger.info(
                    "EmbeddingService: AsyncOpenAI client ready (base=%s, model=%s)",
                    self._base_url, self._model,
                )
            except ImportError:
                logger.error("EmbeddingService: 'openai' package not installed.")
                self._mode = "keyword_fallback"
        return self._openai_client

    # ── Public status ─────────────────────────────────────────────────────────

    def is_using_real_embeddings(self) -> bool:
        """True if the service produces real semantic embeddings (not keyword-based)."""
        return self._mode in ("openai_api", "sentence_transformers")

    def get_mode(self) -> str:
        """Return the active embedding mode string."""
        return self._mode

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single text string."""
        if not text or not text.strip():
            return [0.0] * self._embedding_dim

        if self._mode == "openai_api":
            return await self._openai_embed_single(text)
        if self._mode == "sentence_transformers":
            return await self._st_embed_single(text)
        # Use instance dim so keyword vectors are consistent with the collection
        return self._keyword_vector(text, self._embedding_dim)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts."""
        if not texts:
            return []

        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        dim = self._embedding_dim
        results: List[List[float]] = [[0.0] * dim for _ in texts]

        if not non_empty:
            return results

        if self._mode == "openai_api":
            return await self._openai_embed_batch(texts, non_empty, results)
        if self._mode == "sentence_transformers":
            return await self._st_embed_batch(texts, non_empty, results)
        for i, t in non_empty:
            results[i] = self._keyword_vector(t, self._embedding_dim)
        return results

    # ── OpenAI-compatible provider ────────────────────────────────────────────

    async def _openai_embed_single(self, text: str) -> List[float]:
        client = self._get_openai_client()
        if client is None:
            return self._keyword_vector(text)
        try:
            response = await client.embeddings.create(
                model=self._model,
                input=text,
            )
            emb = response.data[0].embedding
            # Update dim from actual response if needed
            if len(emb) != self._embedding_dim:
                self._embedding_dim = len(emb)
            return emb
        except Exception as exc:
            logger.error("EmbeddingService._openai_embed_single error: %s", exc)
            return self._keyword_vector(text)

    async def _openai_embed_batch(
        self,
        texts: List[str],
        non_empty: list,
        results: List[List[float]],
    ) -> List[List[float]]:
        client = self._get_openai_client()
        if client is None:
            for i, t in non_empty:
                results[i] = self._keyword_vector(t)
            return results
        try:
            indices, batch_texts = zip(*non_empty)
            response = await client.embeddings.create(
                model=self._model,
                input=list(batch_texts),
            )
            for order, item in enumerate(response.data):
                emb = item.embedding
                if len(emb) != self._embedding_dim:
                    self._embedding_dim = len(emb)
                results[indices[order]] = emb
            return results
        except Exception as exc:
            logger.error("EmbeddingService._openai_embed_batch error: %s", exc)
            for i, t in non_empty:
                results[i] = self._keyword_vector(t)
            return results

    # ── SentenceTransformers local model ──────────────────────────────────────

    async def _st_embed_single(self, text: str) -> List[float]:
        if self._st_model is None:
            return self._keyword_vector(text)
        try:
            import asyncio
            emb = await asyncio.to_thread(self._st_model.encode, text)
            return emb.tolist()
        except Exception as exc:
            logger.error("EmbeddingService._st_embed_single error: %s", exc)
            return self._keyword_vector(text)

    async def _st_embed_batch(
        self,
        texts: List[str],
        non_empty: list,
        results: List[List[float]],
    ) -> List[List[float]]:
        if self._st_model is None:
            for i, t in non_empty:
                results[i] = self._keyword_vector(t)
            return results
        try:
            import asyncio
            batch_texts = [t for _, t in non_empty]
            embeddings = await asyncio.to_thread(self._st_model.encode, batch_texts)
            for idx, (i, _) in enumerate(non_empty):
                results[i] = embeddings[idx].tolist()
            return results
        except Exception as exc:
            logger.error("EmbeddingService._st_embed_batch error: %s", exc)
            for i, t in non_empty:
                results[i] = self._keyword_vector(t)
            return results

    # ── Cosine similarity ─────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two embedding vectors. Returns 0.0 if zero."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ── Keyword-overlap fallback ──────────────────────────────────────────────

    @staticmethod
    def _keyword_vector(text: str, dim: int = 1536) -> List[float]:
        """
        Hash-bucket TF-IDF pseudo-embeddings. Used only when no real embedding
        provider is available. Semantically weak but deterministic and fast.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return [0.0] * dim

        vec = [0.0] * dim
        freq: dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1

        for tok, count in freq.items():
            bucket = hash(tok) % dim
            vec[bucket] += count / len(tokens)

        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude > 0:
            vec = [v / magnitude for v in vec]
        return vec


# ── Singleton ─────────────────────────────────────────────────────────────────
embedding_service = EmbeddingService()
