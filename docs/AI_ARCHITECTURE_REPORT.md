# AI Architecture Report

> **Service:** AI Orchestration Service (FastAPI) | **Version:** 3.0 | **Date:** 2026-06-22

---

## 1. Overview

The AI service is a FastAPI orchestration layer that sits between the React frontend / .NET backend and the LLM providers. It classifies user intent, fetches live academic data from the .NET backend, runs RAG retrieval over course materials and regulations, and produces grounded answers.

```
User → .NET Backend → FastAPI AI Service
                          │
                  ┌───────┴────────┐
                  │  Agent          │
                  │   ├ ReactAgent  │ (primary — function-calling loop)
                  │   └ Planner     │ (fallback — intent → module)
                  └───────┬────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
   ModelRouter     VectorStore        BackendClient
   (OpenRouter)    (ChromaDB)         (.NET REST)
          │               │
   OpenRouter LLMs   EmbeddingService
                     (OpenAI / ST / fallback)
```

## 2. Request Lifecycle

1. **Entry** — `/api/chat` or `/api/chat/stream` ([chat.py](../app/api/routes/chat.py)).
2. **Preprocess + safety** — input sanitisation, prompt-injection guard ([core/prompt_safety.py](../app/core/prompt_safety.py)).
3. **Classification** — Layer 1 embedding classifier (fast) → Layer 2 LLM function-call classifier (fallback).
4. **RBAC gate** — intent restricted by role ([core/rbac.py](../app/core/rbac.py), [agents/executor.py](../app/agents/executor.py)).
5. **Execution** — ReactAgent tool loop OR Planner → Module.
6. **Grounding** — modules fetch live data via BackendClient + optional RAG.
7. **Generation** — ModelRouter calls the tiered LLM with fallback chain.
8. **Response** — streamed (SSE) or returned as JSON; saved to conversation memory (Redis).

## 3. Core Components

| Component | File | Role |
|---|---|---|
| Agent | `agents/agent.py` | Top-level orchestrator |
| ReactAgent | `agents/react_agent.py` | gpt-4o-mini function-calling loop (max 4 iters) |
| Planner | `agents/planner.py` | Intent → ExecutionPlan fallback |
| ModelRouter | `agents/model_router.py` | LLM selection + fallback chain + **model tiering** |
| Executor | `agents/executor.py` | RBAC gate + module dispatch |
| EmbeddingService | `services/embedding_service.py` | OpenAI / sentence-transformers / keyword |
| VectorStore | `services/vector_store.py` | ChromaDB (with in-memory fallback) |
| BackendClient | `services/backend_client.py` | .NET REST + circuit breaker |
| MemoryStore | `services/memory_store.py` | Redis conversation memory (disk fallback) |

## 4. Reliability Layers

- **Circuit breaker** — opens after 5 backend failures, resets in 30s.
- **LLM fallback chain** — primary → `gemini-flash-1.5` → `mistral-7b-instruct` (Phase 14).
- **Graceful degradation** — Redis → disk → none; ChromaDB → in-memory; embeddings → keyword.
- **Timeouts** — per-LLM-call 45s, per-request 60s, backend 30s.

## 5. Design Principles

1. **Backend is source of truth** — AI never caches or mutates business data.
2. **Single orchestration entry-point** — assembled once in `main.py` lifespan, stored on `app.state`.
3. **No silent failures** — every degradation is logged with an explicit reason (Phase 15).
4. **Tiered models** — simple/complex/vision routing via `ModelRouter.pick_model()` (Phase 13).

## 6. Production Hardening Applied (2026-06-22)

| Phase | Change |
|---|---|
| 1 | EmbeddingService key priority (`EMBEDDING_API_KEY` → `OPENAI_API_KEY`) + explicit startup logging |
| 2 | `numpy<2` pin — root-cause fix for sentence-transformers load failure |
| 3 | Dockerfile `chroma_data` mkdir+chown + `CHROMADB_PATH` env |
| 13 | `ModelRouter.pick_model()` task→model tiering |
| 14 | Real fallback chain (gemini + mistral) |
| 15 | No-silent-fallback logging across embedding init |
| 16 | Health endpoint mode-string fix (`openai_api`) |

See [PERFORMANCE_OPTIMIZATION_REPORT.md](PERFORMANCE_OPTIMIZATION_REPORT.md) for the remaining roadmap.
