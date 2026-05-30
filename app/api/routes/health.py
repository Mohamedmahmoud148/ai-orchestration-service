"""
app/api/routes/health.py

Health + diagnostic endpoints.

  GET  /health           — liveness probe (Railway uses this). Always 200.
  GET  /health/detailed  — full operational snapshot (backend, RAG, memory).
  GET  /health/rag       — dedicated RAG health: ChromaDB + embeddings + indexing.

These endpoints never raise exceptions — a degraded status is more useful
than a 500 during debugging.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import logger

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  GET /health  — cheap liveness probe
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health_check() -> Dict[str, str]:
    """Simple liveness check — must stay cheap, runs frequently on Railway."""
    return {"status": "ok", "service": "fastapi-ai-service"}


# ─────────────────────────────────────────────────────────────────────────────
#  GET /health/rag  — dedicated RAG subsystem health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health/rag", tags=["System"])
async def health_rag() -> Dict[str, Any]:
    """
    Dedicated RAG health check.

    Returns:
      - embedding_mode: "openai_api" | "sentence_transformers" | "keyword_fallback"
      - real_embeddings: bool — True only if semantic embeddings are active
      - vector_store: collection stats (available, total_chunks, data_dir)
      - vector_store_probe: write+read test result
      - regulation_indexed: bool — whether regulation chunks are in the DB
      - status: "healthy" | "degraded" | "unavailable"

    This endpoint is safe to call frequently — it performs only light reads.
    The probe (write+read+delete cycle) IS slightly heavier but stays <50ms.
    """
    report: Dict[str, Any] = {"service": "fastapi-ai-service", "component": "rag"}

    # ── 1. Embedding service status ───────────────────────────────────────────
    try:
        from app.services.embedding_service import embedding_service
        mode = embedding_service.get_mode()
        report["embedding_mode"]     = mode
        report["embedding_model"]    = embedding_service._model
        report["embedding_base_url"] = embedding_service._base_url
        report["real_embeddings"]    = embedding_service.is_using_real_embeddings()
        report["embedding_dim"]      = embedding_service.embedding_dim
    except Exception as exc:
        report["embedding_mode"]  = "error"
        report["real_embeddings"] = False
        report["embedding_error"] = str(exc)

    # ── 2. Vector store (ChromaDB) ────────────────────────────────────────────
    try:
        from app.services.vector_store import vector_store
        stats = await vector_store.get_collection_stats()
        report["vector_store"] = stats
    except Exception as exc:
        report["vector_store"] = {"available": False, "error": str(exc)}

    # ── 3. Vector store probe (write/read/delete cycle) ───────────────────────
    try:
        from app.services.vector_store import vector_store
        if vector_store._available:
            probe_result = await vector_store.probe()
            report["vector_store_probe"] = probe_result
        else:
            report["vector_store_probe"] = {"ok": False, "reason": "ChromaDB not available"}
    except Exception as exc:
        report["vector_store_probe"] = {"ok": False, "error": str(exc)}

    # ── 4. Regulation index presence ─────────────────────────────────────────
    try:
        from app.services.regulation_indexer import is_any_regulation_indexed
        report["regulation_indexed"] = await is_any_regulation_indexed()
    except Exception as exc:
        report["regulation_indexed"] = False
        report["regulation_indexed_error"] = str(exc)

    # ── 5. Determine overall status ───────────────────────────────────────────
    vs = report.get("vector_store", {})
    probe = report.get("vector_store_probe", {})

    if not vs.get("available"):
        report["status"] = "unavailable"
        report["status_reason"] = "ChromaDB not available"
    elif not report.get("real_embeddings"):
        report["status"] = "degraded"
        report["status_reason"] = (
            f"Using keyword-overlap fallback instead of real embeddings "
            f"(mode={report.get('embedding_mode')}). "
            f"Set OPENAI_API_KEY to enable semantic search."
        )
    elif not probe.get("ok"):
        report["status"] = "degraded"
        report["status_reason"] = "ChromaDB probe failed: " + str(probe.get("reason") or probe.get("error"))
    elif not report.get("regulation_indexed"):
        report["status"] = "degraded"
        report["status_reason"] = "No regulation documents indexed yet. Call POST /admin/reindex-regulations."
    else:
        report["status"] = "healthy"

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  GET /health/detailed  — full operational snapshot
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health/detailed", tags=["System"])
async def health_detailed() -> Dict[str, Any]:
    """
    Full operational snapshot for dashboards + manual debugging.

    Covers: environment, backend circuit breaker, RAG, memory store.
    All checks are best-effort and never raise.
    """
    snapshot: Dict[str, Any] = {
        "status":      "ok",
        "service":     "fastapi-ai-service",
        "environment": settings.ENVIRONMENT,
    }

    # ── Backend circuit breaker ────────────────────────────────────────────
    try:
        from app.services.backend_client import tool_execution_client
        snapshot["backend"] = {
            "base_url":        tool_execution_client.base_url,
            "circuit_breaker": tool_execution_client.breaker_status(),
        }
    except Exception as exc:
        snapshot["backend"] = {"error": f"unavailable: {exc}"}

    # ── Backend connectivity ping ──────────────────────────────────────────
    try:
        from app.services.backend_client import tool_execution_client
        import asyncio
        async def _ping_backend():
            try:
                resp = await asyncio.wait_for(
                    tool_execution_client.fetch("/health", auth_header=None),
                    timeout=5.0,
                )
                return {"reachable": True, "response": str(resp)[:100]}
            except Exception as exc2:
                return {"reachable": False, "error": str(exc2)[:100]}
        snapshot["backend_ping"] = await _ping_backend()
    except Exception as exc:
        snapshot["backend_ping"] = {"reachable": False, "error": str(exc)[:100]}

    # ── Embedding service ──────────────────────────────────────────────────
    try:
        from app.services.embedding_service import embedding_service
        snapshot["embeddings"] = {
            "mode":           embedding_service.get_mode(),
            "real_embeddings": embedding_service.is_using_real_embeddings(),
            "model":          embedding_service._model,
            "dim":            embedding_service.embedding_dim,
        }
    except Exception as exc:
        snapshot["embeddings"] = {"error": str(exc)}

    # ── RAG / vector store ─────────────────────────────────────────────────
    try:
        from app.services.vector_store import vector_store
        rag_stats = await vector_store.get_collection_stats()
        snapshot["rag"] = rag_stats
    except Exception as exc:
        snapshot["rag"] = {"available": False, "error": f"unavailable: {exc}"}

    # ── Regulation index ────────────────────────────────────────────────────
    try:
        from app.services.regulation_indexer import is_any_regulation_indexed
        snapshot["regulation_indexed"] = await is_any_regulation_indexed()
    except Exception as exc:
        snapshot["regulation_indexed"] = False

    # ── Memory store ────────────────────────────────────────────────────────
    try:
        from app.services.memory_store import get_memory_store
        store = get_memory_store()
        mode = store.get_storage_mode()
        info = {
            "mode":          mode,                               # "redis" | "disk" | "none"
            "redis_enabled": mode == "redis",
            "disk_enabled":  mode == "disk",
        }
        if hasattr(store, "_disk_store") and store._disk_store:
            info["disk_stats"] = store._disk_store.stats()
        snapshot["memory_store"] = info
    except Exception as exc:
        snapshot["memory_store"] = {"error": f"unavailable: {exc}"}

    # ── Overall status ──────────────────────────────────────────────────────
    breaker = snapshot.get("backend", {}).get("circuit_breaker") or {}
    if breaker.get("state") == "open":
        snapshot["status"] = "degraded"
        snapshot["status_reason"] = "backend circuit breaker is OPEN"
    elif not snapshot.get("rag", {}).get("available"):
        snapshot["status"] = "degraded"
        snapshot["status_reason"] = "ChromaDB unavailable"

    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
#  Admin endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/admin/refresh-prompts", tags=["Admin"])
async def refresh_prompts() -> Dict[str, Any]:
    """Drop in-memory prompt cache so next request reloads from disk."""
    from app.prompts import clear_prompt_cache
    clear_prompt_cache()
    logger.info("Admin: prompt cache cleared")
    return {"status": "ok", "message": "Prompt cache cleared"}


@router.post("/admin/refresh-schema", tags=["Admin"])
async def refresh_schema() -> Dict[str, Any]:
    """Re-fetch .NET Swagger schema without restarting the service."""
    from app.core import api_discovery
    try:
        await api_discovery.fetch_and_filter_schema()
        return {"status": "ok", "message": "Backend Swagger schema reloaded"}
    except Exception as exc:
        logger.error("Admin: refresh_schema failed — %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


@router.post("/admin/reindex-regulations", tags=["Admin"])
async def admin_reindex_regulations() -> Dict[str, Any]:
    """Trigger a full regulation reindex. Safe to call while another run is in progress."""
    from app.services.regulation_indexer import index_all_active_regulations
    try:
        result = await index_all_active_regulations(auth_header=None)
        return {"status": "ok", "result": result}
    except Exception as exc:
        logger.error("Admin: reindex_regulations failed — %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}
