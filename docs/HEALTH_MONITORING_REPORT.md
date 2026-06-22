# Health & Monitoring Report

> **Files:** `api/routes/health.py`, `core/rag_health.py`, `core/observability.py`, `core/middleware.py` | **Date:** 2026-06-22

---

## 1. Health Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Cheap liveness probe (Railway). Always 200. |
| `GET /health/rag` | RAG subsystem: embedding mode, real_embeddings, vector store stats, write/read probe, regulation-index presence, overall status. |
| `GET /health/detailed` | Full snapshot: env, backend circuit breaker, backend ping, embeddings, RAG, regulation index, memory store. |

> `/health/embeddings` and `/health/vector-store` requested in Phase 16 are covered today by `/health/rag` (which includes both the embedding mode and a live vector-store probe). They can be added as thin aliases if the frontend needs separate URLs.

## 2. `/health/rag` Response

```json
{
  "service": "fastapi-ai-service",
  "component": "rag",
  "embedding_mode": "openai_api",
  "embedding_model": "text-embedding-3-small",
  "real_embeddings": true,
  "embedding_dim": 1536,
  "vector_store": { "available": true, "backend": "chroma", "total_chunks": 284 },
  "vector_store_probe": { "ok": true },
  "regulation_indexed": true,
  "status": "healthy"
}
```

**Status values:** `healthy` | `degraded` | `unavailable` (each with a `status_reason`).

## 3. Startup Diagnostics

`core/rag_health.log_rag_health()` logs a one-line RAG summary at boot (provider, mode, dimensions, vector store, smoke-test result) and emits actionable recommendations when degraded. The mode-string bug (`openai` vs `openai_api`) is fixed so OpenAI mode is recognised correctly.

## 4. Per-Request Observability

`CorrelationIDMiddleware` + `RequestTimingMiddleware` ([core/middleware.py](../app/core/middleware.py)) attach a correlation ID and latency to every request. `core/observability.py` carries structured logging.

**Phase 15 target fields** (recommended to log per request): `correlation_id`, `model`, `tokens`, `latency_ms`, `retrieval_count`, `chunk_count`, `embedding_provider`, `fallback_triggered`, `cache_hit`, `confidence_score`. Correlation ID, latency, model, and embedding provider are available today; tokens/retrieval/cache/confidence are the remaining additions.

## 5. Admin Endpoints

| Endpoint | Action |
|---|---|
| `POST /admin/refresh-prompts` | Clear prompt cache |
| `POST /admin/refresh-schema` | Re-fetch .NET Swagger schema |
| `POST /admin/reindex-regulations` | Full regulation reindex |

## 6. Frontend Health Indicator

The frontend can poll `GET /health/rag` and show a small badge:
- 🟢 `status: healthy`
- 🟡 `status: degraded` (show `status_reason` on hover)
- 🔴 `status: unavailable`
