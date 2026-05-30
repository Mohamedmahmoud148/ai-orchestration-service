"""
app/api/routes/health.py

Health + diagnostic endpoints.

  GET  /health           — liveness probe (Railway uses this). Cheap, always 200.
  GET  /health/detailed  — operational status: breaker state, RAG availability,
                            Redis availability, indexed regulation count.
                            Safe to expose — no secrets, no PII, no internal keys.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import logger

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check() -> Dict[str, str]:
    """Simple liveness check — must stay cheap, runs frequently on Railway."""
    return {"status": "ok", "service": "fastapi-ai-service"}


@router.get("/health/detailed", tags=["System"])
async def health_detailed() -> Dict[str, Any]:
    """
    Operational snapshot for dashboards + manual debugging.

    Returns:
      - environment + version basics
      - backend circuit-breaker state (closed / open / half_open)
      - RAG status (chromadb available + indexed regulation count)
      - Redis memory status (connected / disabled)

    All checks are best-effort and never raise — a partial readout is
    more useful than a failure when something is degraded.
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
            "base_url":         tool_execution_client.base_url,
            "circuit_breaker":  tool_execution_client.breaker_status(),
        }
    except Exception as exc:
        snapshot["backend"] = {"error": f"unavailable: {exc}"}

    # ── RAG / vector store ─────────────────────────────────────────────────
    try:
        from app.services.vector_store import vector_store
        rag_stats = await vector_store.get_collection_stats()
        snapshot["rag"] = rag_stats
    except Exception as exc:
        snapshot["rag"] = {"available": False, "error": f"unavailable: {exc}"}

    # ── Regulation index presence (used by AcademicAdvisor v2) ────────────
    try:
        from app.services.regulation_indexer import is_any_regulation_indexed
        snapshot["regulation_indexed"] = await is_any_regulation_indexed()
    except Exception as exc:
        snapshot["regulation_indexed"] = False
        snapshot["regulation_indexed_error"] = str(exc)

    # ── Redis memory store ─────────────────────────────────────────────────
    try:
        from app.services.memory_store import get_memory_store
        store = get_memory_store()
        snapshot["memory_store"] = {
            "disabled":      bool(getattr(store, "_disabled", True)),
            "redis_url_set": bool(getattr(store, "redis_url", None)),
        }
    except Exception as exc:
        snapshot["memory_store"] = {"error": f"unavailable: {exc}"}

    # Overall status: degraded if breaker open OR RAG missing AND we depend on it
    breaker = snapshot.get("backend", {}).get("circuit_breaker") or {}
    if breaker.get("state") == "open":
        snapshot["status"] = "degraded"
        snapshot["status_reason"] = "backend circuit breaker is OPEN"

    return snapshot


# ─────────────────────────────────────────────────────────────────────────
#  Admin endpoints — operational refresh + cache management
#  Note: these are scoped under /admin. In production, expose them only on
#  the internal network OR add an Authorization header check. For now we
#  rely on the .NET reverse proxy / Railway private networking.
# ─────────────────────────────────────────────────────────────────────────


@router.post("/admin/refresh-prompts", tags=["Admin"])
async def refresh_prompts() -> Dict[str, Any]:
    """
    Drop the in-memory prompt cache so the next request reloads every
    prompt .md from disk. Use after editing files in app/prompts/.
    """
    from app.prompts import clear_prompt_cache
    clear_prompt_cache()
    logger.info("Admin: prompt cache cleared (will reload on next access)")
    return {"status": "ok", "message": "Prompt cache cleared"}


@router.post("/admin/refresh-schema", tags=["Admin"])
async def refresh_schema() -> Dict[str, Any]:
    """
    Re-fetch the .NET Swagger schema. Use after the backend adds new
    endpoints so the AI's dynamic-API router sees them without a service
    restart. Safe to call multiple times.
    """
    from app.core import api_discovery
    try:
        await api_discovery.fetch_and_filter_schema()
        return {
            "status": "ok",
            "message": "Backend Swagger schema reloaded",
        }
    except Exception as exc:
        logger.error("Admin: refresh_schema failed — %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


@router.post("/admin/reindex-regulations", tags=["Admin"])
async def admin_reindex_regulations() -> Dict[str, Any]:
    """
    Trigger a full regulation reindex. Uses the same lock as the startup
    auto-index, so calling this while a reindex is in progress is safe —
    it returns immediately with a 'skipped' status.
    """
    from app.services.regulation_indexer import index_all_active_regulations
    try:
        result = await index_all_active_regulations(auth_header=None)
        return {"status": "ok", "result": result}
    except Exception as exc:
        logger.error("Admin: reindex_regulations failed — %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}
