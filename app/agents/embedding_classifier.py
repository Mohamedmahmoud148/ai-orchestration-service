"""
app/agents/embedding_classifier.py  —  Layer 1: Embedding Intent Classifier

Classifies user intent by comparing the message's embedding vector against
pre-computed per-intent centroids built from the INTENT_EXAMPLES bank.

Key properties:
  - Zero LLM calls — pure vector math after warm_up()
  - ~5–15 ms per classification (one API embed call or local ST model)
  - Language-agnostic: works on the preprocessed (normalized) message
  - Returns top-3 candidates + per-candidate score
  - Score >= HIGH_CONFIDENCE_THRESHOLD → skip LLM entirely (fast path)

Integration:
  - warm_up() is called ONCE at startup in main.py lifespan
  - EmbeddingIntentClassifier receives the existing EmbeddingService singleton
  - PlannerAgent receives the classifier as an optional parameter
    (None → Layer 1 is bypassed, backward compatible)

Shadow mode (EMBEDDING_CLASSIFIER_SHADOW_MODE=true):
  - Classifier runs and logs results
  - Existing planner behavior is NOT changed
  - Used for 1-week calibration before enabling the fast path
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from app.core.logging import logger

if TYPE_CHECKING:
    from app.services.embedding_service import EmbeddingService


@dataclass(frozen=True)
class EmbeddingClassificationResult:
    """
    Result from a single embedding-based classification call.

    intent          — top-ranked intent
    score           — cosine similarity against the intent centroid (0.0–1.0)
    candidates      — top-3 (intent, score) pairs, descending
    is_confident    — True when score >= HIGH_CONFIDENCE_THRESHOLD
    latency_ms      — wall-clock time for the embed + score pass
    """
    intent: str
    score: float
    candidates: list[tuple[str, float]]
    is_confident: bool
    latency_ms: float = 0.0


# Sentinel for "not yet warmed up" — never returned to callers.
_FALLBACK_RESULT = EmbeddingClassificationResult(
    intent="general_chat",
    score=0.0,
    candidates=[],
    is_confident=False,
    latency_ms=0.0,
)


class EmbeddingIntentClassifier:
    """
    Warm-start intent classifier backed by the project's EmbeddingService.

    Lifecycle:
      1. Instantiate with the shared EmbeddingService singleton.
      2. Await warm_up() once at startup — embeds all INTENT_EXAMPLES
         and computes normalized L2 centroids.
      3. Call classify() per request — returns EmbeddingClassificationResult.

    Thresholds (can be overridden via settings):
      HIGH_CONFIDENCE  0.82  → skip LLM, execute directly
      LOW_CONFIDENCE   0.55  → below this: do not trust embedding result at all

    Thread safety: warm_up() is guarded by an asyncio.Lock.  classify() is
    safe to call from concurrent requests once the classifier is ready.
    """

    # Default thresholds — overridden by settings in PlannerAgent
    HIGH_CONFIDENCE_THRESHOLD: float = 0.82
    LOW_CONFIDENCE_THRESHOLD:  float = 0.55

    def __init__(self, embedding_service: "EmbeddingService") -> None:
        self._svc = embedding_service
        self._centroids: dict[str, list[float]] = {}   # intent → unit vector
        self._ready = False
        self._warm_up_lock = asyncio.Lock()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def warm_up(self) -> None:
        """
        Pre-embed all intent examples and compute per-intent centroids.

        Called ONCE in main.py lifespan after EmbeddingService is confirmed ready.
        Re-entrant: if already warmed up, returns immediately.

        Cost: one batched embedding API call (~200–500 ms on first boot,
        cached by the model provider on subsequent boots via identical inputs).
        """
        async with self._warm_up_lock:
            if self._ready:
                return

            from app.agents.intent_examples import INTENT_EXAMPLES, validate_examples

            # Validate example bank — log warnings, don't crash
            for warning in validate_examples():
                logger.warning("EmbeddingIntentClassifier: %s", warning)

            if not self._svc.is_using_real_embeddings():
                logger.warning(
                    "EmbeddingIntentClassifier: EmbeddingService is in mode=%s. "
                    "Embedding quality is degraded — keyword_fallback vectors are "
                    "semantically weak.  Set OPENAI_API_KEY for real embeddings.",
                    self._svc.get_mode(),
                )

            t0 = time.perf_counter()
            total_examples = 0

            for intent, examples in INTENT_EXAMPLES.items():
                if not examples:
                    continue

                vectors = await self._svc.embed_batch(examples)
                total_examples += len(examples)

                # Centroid = mean of all example vectors, then L2-normalized
                dim = len(vectors[0]) if vectors else self._svc.embedding_dim
                centroid = [0.0] * dim
                for vec in vectors:
                    for i, v in enumerate(vec):
                        centroid[i] += v

                n = len(vectors)
                centroid = [x / n for x in centroid]

                # L2 normalize so cosine similarity == dot product
                magnitude = sum(x * x for x in centroid) ** 0.5
                if magnitude > 1e-9:
                    centroid = [x / magnitude for x in centroid]

                self._centroids[intent] = centroid

            elapsed = (time.perf_counter() - t0) * 1000
            self._ready = True
            logger.info(
                "EmbeddingIntentClassifier: ready — "
                "%d intents, %d examples, warm_up=%.0fms, mode=%s",
                len(self._centroids), total_examples, elapsed, self._svc.get_mode(),
            )

    # ── Classification ────────────────────────────────────────────────────────

    async def classify(
        self,
        message: str,
        clean_text: Optional[str] = None,
        high_threshold: Optional[float] = None,
    ) -> EmbeddingClassificationResult:
        """
        Classify `message` against all intent centroids.

        Parameters:
          message     — original user message (used as fallback)
          clean_text  — Layer-0 normalized text (preferred for embedding)
          high_threshold — override HIGH_CONFIDENCE_THRESHOLD for this call

        Returns EmbeddingClassificationResult.  On any error, returns the
        fallback result (intent=general_chat, score=0.0, is_confident=False)
        so the pipeline degrades gracefully to LLM classification.
        """
        if not self._ready:
            logger.debug("EmbeddingIntentClassifier: not ready — returning fallback")
            return _FALLBACK_RESULT

        t0 = time.perf_counter()
        text = (clean_text or message or "").strip()
        if not text:
            return _FALLBACK_RESULT

        threshold = high_threshold if high_threshold is not None else self.HIGH_CONFIDENCE_THRESHOLD

        try:
            query_vec = await self._svc.embed_text(text)

            # Score all intents via dot product (vectors are L2-normalized)
            scores: list[tuple[str, float]] = [
                (intent, self._dot(query_vec, centroid))
                for intent, centroid in self._centroids.items()
            ]
            scores.sort(key=lambda x: x[1], reverse=True)

            top_intent, top_score = scores[0]
            top_3 = scores[:3]
            elapsed = (time.perf_counter() - t0) * 1000

            result = EmbeddingClassificationResult(
                intent=top_intent,
                score=round(top_score, 4),
                candidates=top_3,
                is_confident=top_score >= threshold,
                latency_ms=round(elapsed, 1),
            )

            logger.debug(
                "EmbeddingClassifier: text=%.60r → intent=%r score=%.3f "
                "confident=%s top3=%s latency=%.0fms",
                text, top_intent, top_score, result.is_confident,
                [(i, f"{s:.3f}") for i, s in top_3],
                elapsed,
            )

            return result

        except Exception as exc:
            logger.error(
                "EmbeddingIntentClassifier.classify error — falling back: %s", exc
            )
            return _FALLBACK_RESULT

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def intent_count(self) -> int:
        return len(self._centroids)

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        """Fast dot product (== cosine similarity for L2-normalized vectors)."""
        if len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    def status_dict(self) -> dict:
        """Returns a status dict for the /health endpoint."""
        return {
            "ready": self._ready,
            "intent_count": len(self._centroids),
            "thresholds": {
                "high_confidence": self.HIGH_CONFIDENCE_THRESHOLD,
                "low_confidence": self.LOW_CONFIDENCE_THRESHOLD,
            },
        }
