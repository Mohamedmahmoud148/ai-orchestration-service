# Embedding System Report

> **File:** `app/services/embedding_service.py` | **Date:** 2026-06-22

---

## 1. The Problem (root cause of degraded RAG)

Production logs showed:
```
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.0+cpu
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6
EmbeddingService: sentence-transformers load failed — name 'nn' is not defined
EmbeddingService: mode=keyword_fallback — RAG semantic quality is DEGRADED
```

**Chain of failure:** `requirements.txt` allowed `numpy>=1.26.0` (no upper bound) → NumPy 2.4.6 installed → PyTorch 2.2.0 (built against NumPy 1.x) broke → `sentence-transformers` failed to import → EmbeddingService silently dropped to `keyword_fallback` (lexical overlap, **not** semantic).

This degraded: RAG search, intent classification, academic advisor, material explanation, regulation search, PDF understanding.

## 2. The Fix

### a) numpy pin (root cause)
```
numpy>=1.26,<2
```
This alone restores `sentence_transformers` mode even with **no API key**.

### b) Provider priority (Phase 1)
```
EMBEDDING_API_KEY  →  OPENAI_API_KEY  →  sentence-transformers  →  keyword_fallback
```
Set `EMBEDDING_API_KEY` (or `OPENAI_API_KEY`) on Railway for the best quality (`text-embedding-3-small`, 1536-dim).

### c) No silent fallback (Phase 15)
The constructor now logs the exact provider, mode, dimensions, and fallback status, and logs the **reason** whenever it degrades.

## 3. Provider Matrix

| Mode | Trigger | Quality | Dim | Deps |
|---|---|---|---|---|
| `openai_api` | `EMBEDDING_API_KEY`/`OPENAI_API_KEY` set | ⭐⭐⭐ Best | 1536 | none (API) |
| `sentence_transformers` | local model loads (numpy<2) | ⭐⭐ Good | 384 | torch + ST |
| `keyword_fallback` | nothing else available | ⭐ Degraded | configurable | none |

## 4. Expected Startup Log (success)

```
EmbeddingService: EmbeddingProvider=OpenAI EmbeddingMode=openai base=... model=text-embedding-3-small Dimensions=1536 Fallback=False
```

If degraded, the log is `ERROR` level with `Fallback=True` and an explicit reason — never silent.

## 5. Batching (Phase 11)

`embed_batch()` is implemented for **all** providers ([embedding_service.py:160](../app/services/embedding_service.py#L160)):
- OpenAI: single `/embeddings` call with a list input (one network round-trip for N chunks).
- sentence-transformers: `model.encode(list)` in a worker thread.
- Indexing uses the batch path → ~10x faster than per-chunk calls.

> **Action item:** verify every indexing caller uses `embed_batch`, not `embed_text` in a loop.

## 6. Public API

```python
await embedding_service.embed_text(text)        # → list[float]
await embedding_service.embed_batch(texts)       # → list[list[float]]
embedding_service.cosine_similarity(a, b)        # → float
embedding_service.is_using_real_embeddings()     # → bool
embedding_service.get_mode()                     # → "openai_api"|"sentence_transformers"|"keyword_fallback"
```

## 7. Deploy Checklist

1. Set `EMBEDDING_API_KEY` (or `OPENAI_API_KEY`) on Railway.
2. Redeploy (the `numpy<2` pin takes effect on rebuild).
3. Confirm startup log shows `Fallback=False`.
4. Hit `GET /health/rag` → `real_embeddings: true`.
