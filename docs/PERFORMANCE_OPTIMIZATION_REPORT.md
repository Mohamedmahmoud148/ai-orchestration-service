# Performance Optimization Report

> **Date:** 2026-06-22

---

## 1. Done (this pass)

| Fix | Impact |
|---|---|
| `numpy<2` pin | Restores semantic embeddings (was keyword-only) |
| Dockerfile chroma_data perms | Persistent vector store (was lost on restart) |
| Real fallback chain | No downtime when OpenAI is overloaded |
| Model tiering helper | Right model for the task (cost + quality) |
| Batch embeddings (verified) | ~10x faster indexing vs per-chunk |
| Explicit degradation logging | Faster diagnosis, no silent failures |

## 2. High-Value Roadmap

### a) Redis response cache (Phase 10)
Cache key = `hash(role + intent + query + document_id)`, TTL 1h.
- **Cache:** regulations, general university Q&A, repeated explanations.
- **Never cache:** personal schedules, grades, sensitive data.
- Redis is already wired (`services/memory_store.py`) — add a thin cache wrapper at the chat route.

### b) Turn off classifier shadow mode (Phase 12)
`EMBEDDING_CLASSIFIER_SHADOW_MODE=True` means Layer 1 runs but does **not** drive routing — every request still pays for an LLM classification call.
- **After** confirming embeddings are healthy (`Fallback=False`), set `EMBEDDING_CLASSIFIER_SHADOW_MODE=False` on Railway.
- Saves one LLM round-trip per clearly-classified request.
- ⚠️ Do **not** flip this while embeddings are in `keyword_fallback` — classification quality would drop. Left at default `True` intentionally until the embedding key/redeploy is confirmed.

### c) Drop torch when on OpenAI embeddings
If `OPENAI_API_KEY`/`EMBEDDING_API_KEY` is set, the local ML stack (~1.2 GB) is unused. Removing `torch`/`sentence-transformers`/`transformers` shrinks the image and cuts cold start ~3-4x. `model_service.py` uses lazy imports, so the `hf/` path degrades gracefully when absent.

### d) Cold start
torch + ST load on every container boot. Dropping them (per c) removes the biggest cold-start cost.

## 3. Latency Budget (per request)

| Stage | Typical |
|---|---|
| Intent classification (LLM) | 0.5–1.5s (eliminated for high-confidence once shadow mode off) |
| Backend data fetch | 50–300ms (parallel via `asyncio.gather`) |
| RAG search | 20–80ms |
| LLM generation | 2–4s (streamed → perceived faster) |

## 4. Concurrency

- All LLM, Chroma, and backend calls are async; Chroma sync calls wrapped in `asyncio.to_thread`.
- Modules use `asyncio.gather()` to fetch multiple data sources concurrently.

## 5. Deploy Order (recommended)

1. Set `EMBEDDING_API_KEY` (or `OPENAI_API_KEY`) → redeploy → confirm `Fallback=False`.
2. Confirm `GET /health/rag` → healthy.
3. Set `EMBEDDING_CLASSIFIER_SHADOW_MODE=False`.
4. (Optional) Set `MODEL_COMPLEX=openai/gpt-4o`.
5. (Optional) Remove torch/ST from requirements + Dockerfile.
