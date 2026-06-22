"""
app/core/rag_health.py — RAG Health Diagnostics

Provides startup diagnostics and runtime health checking for the RAG pipeline.
Logs the exact state of each component so problems are immediately visible.

Usage (in main.py lifespan):
    from app.core.rag_health import log_rag_health
    await log_rag_health()
"""
from __future__ import annotations

import os
from app.core.logging import logger


async def log_rag_health() -> dict:
    """
    Diagnoses and logs the current state of the RAG pipeline.
    Returns a health summary dict.

    Components checked:
        1. OPENAI_API_KEY         → real embeddings vs keyword fallback
        2. EmbeddingService mode  → dimensions, provider
        3. VectorStore            → ChromaDB availability
        4. Example embedding      → live smoke test
    """
    health = {
        "openai_key_set":      False,
        "embedding_provider":  "unknown",
        "embedding_mode":      "unknown",
        "embedding_dimensions": 0,
        "vector_store":        "unavailable",
        "smoke_test_passed":   False,
        "status":              "degraded",
        "recommendations":     [],
    }

    # ── 1. Check embedding API key (dedicated or OpenAI) ──────────────────────
    openai_key = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    health["openai_key_set"] = bool(openai_key and len(openai_key) > 10)

    if not health["openai_key_set"]:
        health["recommendations"].append(
            "Set OPENAI_API_KEY in Railway environment for real semantic embeddings. "
            "Without it, RAG uses keyword matching only — quality is significantly degraded."
        )

    # ── 2. Check EmbeddingService ─────────────────────────────────────────────
    try:
        from app.services.embedding_service import embedding_service
        mode = getattr(embedding_service, "_mode", "unknown")
        health["embedding_mode"] = mode

        if mode == "keyword_fallback":
            health["embedding_provider"] = "keyword_overlap (no real embeddings)"
            health["recommendations"].append(
                "EmbeddingService is in keyword_fallback mode. "
                "Semantic similarity search is NOT active. Add OPENAI_API_KEY."
            )
        elif mode in ("openai", "openai_api", "sentence_transformers"):
            health["embedding_provider"] = mode
            # Try to get dimensions
            try:
                test_vec = await embedding_service.embed_text("test")
                if test_vec:
                    health["embedding_dimensions"] = len(test_vec)
                    health["smoke_test_passed"] = True
            except Exception as e:
                logger.warning("RAG health: embedding smoke test failed — %s", e)
    except Exception as e:
        logger.warning("RAG health: EmbeddingService check failed — %s", e)
        health["embedding_mode"] = "error"

    # ── 3. Check VectorStore ──────────────────────────────────────────────────
    try:
        from app.services.vector_store import vector_store
        if vector_store and getattr(vector_store, "_collection", None):
            count = vector_store._collection.count()
            health["vector_store"] = f"chroma ({count} vectors)"
        else:
            health["vector_store"] = "chroma (empty or unavailable)"
    except Exception as e:
        health["vector_store"] = f"error: {e}"

    # ── 4. Determine overall status ───────────────────────────────────────────
    if health["smoke_test_passed"] and "chroma" in health["vector_store"]:
        health["status"] = "healthy"
    elif health["embedding_mode"] == "keyword_fallback":
        health["status"] = "degraded_keyword_only"
    else:
        health["status"] = "degraded"

    # ── 5. Log summary ────────────────────────────────────────────────────────
    status_emoji = "✅" if health["status"] == "healthy" else "⚠️"
    logger.info(
        "%s RAG Health: status=%s embedding_mode=%s provider=%s dimensions=%d vector_store=%s",
        status_emoji,
        health["status"],
        health["embedding_mode"],
        health["embedding_provider"],
        health["embedding_dimensions"],
        health["vector_store"],
    )

    if health["recommendations"]:
        for rec in health["recommendations"]:
            logger.warning("RAG Recommendation: %s", rec)

    return health
