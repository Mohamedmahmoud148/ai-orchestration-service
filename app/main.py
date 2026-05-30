"""
main.py

Application entry point.

The Agent is assembled ONCE here at startup and stored in app.state.agent.
Every endpoint reads it from there via the app state —
no circular imports, no global singletons, no tight coupling.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from app.agents.agent import Agent
from app.agents.executor import PlanExecutor
from app.agents.model_router import ModelRouter
from app.agents.planner import PlannerAgent

from app.api.routes import chat, health, complaint_intelligence, rag as rag_routes, memory as memory_routes, ai_grading, exam_generation_api
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.core.middleware import CorrelationIDMiddleware, RequestTimingMiddleware
from app.core.rate_limiter import RateLimiter
from app.services.backend_client import tool_execution_client
from app.services.memory_store import get_memory_store
from app.services.model_service import local_model_service
from app.services.tool_registry import tool_registry


# ─────────────────────────────────────────────────────────────
#  Lifespan: validate config, build Agent, tear down cleanly
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; release them on shutdown."""
    setup_logging()
    logger.info("Starting AI Orchestration Service (env=%s)", settings.ENVIRONMENT)

    # ── 1. Startup Validation — fail fast, fail loud ──────────────────
    if not settings.BACKEND_BASE_URL:
        raise RuntimeError(
            "STARTUP FAILED: BACKEND_BASE_URL is not set. "
            "The AI service cannot communicate with the .NET backend. "
            "Set BACKEND_BASE_URL in your .env file (e.g. http://localhost:5000) "
            "and restart."
        )
    logger.info("Backend URL: %s", settings.BACKEND_BASE_URL)

    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError(
            "STARTUP FAILED: OPENROUTER_API_KEY is not set. "
            "The Planner requires a valid OpenRouter API key. "
            "Set OPENROUTER_API_KEY in your .env file and restart. "
            "Get one at https://openrouter.ai/keys"
        )
    logger.info("OpenRouter API key: configured.")

    # Confirm BackendClient singleton initialised correctly + open httpx pool.
    # (ToolExecutionClient.__init__ raises RuntimeError if URL is missing;
    #  this is a belt-and-suspenders log confirming it is alive.)
    await tool_execution_client.start()
    logger.info(
        "ToolExecutionClient ready (base_url=%s, pool=open, breaker=closed).",
        tool_execution_client.base_url,
    )

    # ── 1.5 Fetch and Cache Dynamic API Schema ────────────────────────
    from app.core import api_discovery
    logger.info("Fetching Dynamic API Schema from Backend...")
    await api_discovery.fetch_and_filter_schema()

    # ── 2. OpenRouter LLM client (OpenAI-compatible) ──────────────────
    # `timeout=` here is the SDK-level cap per request; OpenRouter sometimes
    # hangs on overloaded models. Without it, a single bad request would tie
    # up a worker indefinitely. Keep it under REQUEST_TIMEOUT_SECONDS so the
    # outer agent timeout can still fire and return a useful error.
    openai_client = AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        timeout=settings.OPENROUTER_TIMEOUT_SECONDS,
        max_retries=0,   # we own the retry loop via model_router fallback chain
        default_headers={
            "HTTP-Referer": "University AI System",
            "X-Title": "University AI Agent",
        },
    )
    logger.info(
        "OpenRouter client: timeout=%.1fs max_retries=0 (fallback chain owns retries)",
        settings.OPENROUTER_TIMEOUT_SECONDS,
    )
    app.state.openrouter_client = openai_client
    logger.info("OpenRouter client initialised (base_url=https://openrouter.ai/api/v1).")

    # ── 3. Component assembly ─────────────────────────────────────────
    model_router = ModelRouter(
        openai_client=openai_client,
        local_model_service=local_model_service,
    )

    memory_store = get_memory_store()  # process-wide singleton
    await memory_store.ping()          # disable permanently if auth fails
    app.state.memory_store = memory_store
    app.state.rate_limiter = RateLimiter()

    planner = PlannerAgent(
        model_router=model_router,
        memory=memory_store,
        ranker=None,
    )

    executor = PlanExecutor(
        backend_execution_func=tool_execution_client.execute_tool,
        model_router=model_router,
    )

    # ── 4. ReactAgent (gpt-4o-mini function calling loop) ────────────
    from app.agents.react_agent import ReactAgent
    react_agent = ReactAgent(
        openrouter_client=openai_client,
        model_router=model_router,
    )
    logger.info("ReactAgent ready — flow: User → ReactAgent → tools loop → Response")

    # ── 4.1 Agent (single instance for the lifetime of the app) ──────
    app.state.agent = Agent(
        planner=planner,
        tool_registry=tool_registry,
        model_router=model_router,
        executor=executor,
        react_agent=react_agent,
    )

    logger.info("Agent ready — flow: User → ReactAgent (smart) → fallback: Planner → Module")

    # ── 5. RAG services (EmbeddingService + VectorStore) ─────────────
    from app.services.embedding_service import embedding_service
    from app.services.vector_store import vector_store
    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    logger.info(
        "RAG services ready — embedding_fallback=%s, vector_store_available=%s.",
        embedding_service._fallback_mode,
        vector_store._available,
    )

    # ── 6. Auto-index academic regulations on first boot ─────────────
    # Non-blocking: scheduled as a background task so a slow .NET backend
    # or large PDF doesn't delay app startup. The AcademicAdvisor v2 and
    # RegulationModule both have on-demand fallbacks if this hasn't finished.
    if vector_store._available:
        import asyncio
        from app.services.regulation_indexer import (
            index_all_active_regulations,
            is_any_regulation_indexed,
        )

        async def _bootstrap_regulation_index() -> None:
            try:
                if await is_any_regulation_indexed():
                    logger.info("Regulation auto-index: already indexed — skipping.")
                    return
                logger.info("Regulation auto-index: starting background reindex.")
                result = await index_all_active_regulations(auth_header=None)
                logger.info(
                    "Regulation auto-index: done — status=%s total_chunks=%d",
                    result.get("status"), result.get("total_chunks", 0),
                )
            except Exception as exc:
                logger.warning(
                    "Regulation auto-index failed (non-fatal) — modules will fall back: %s",
                    exc,
                )

        asyncio.create_task(_bootstrap_regulation_index())

    yield

    # ── Shutdown ───────────────────────────────────────────────────────
    logger.info("Shutting down AI Orchestration Service.")
    app.state.agent = None
    try:
        await tool_execution_client.aclose()
    except Exception as exc:
        logger.warning("Shutdown: tool_execution_client.aclose failed — %s", exc)


# ─────────────────────────────────────────────────────────────
#  FastAPI application
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Orchestration Service",
    description="FastAPI service orchestrating AI pipelines to .NET backend execution.",
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS origins ───────────────────────────────────────────────────────────────
# Priority: ALLOWED_ORIGINS env var →  BACKEND_BASE_URL  → localhost only
def _build_cors_origins() -> list[str]:
    if settings.ALLOWED_ORIGINS.strip():
        origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
    elif settings.BACKEND_BASE_URL:
        origins = [settings.BACKEND_BASE_URL]
    else:
        origins = []

    # Always permit localhost in development so developers aren't blocked
    if settings.ENVIRONMENT == "development":
        for port in ("3000", "5000", "8000"):
            for scheme in ("http", "https"):
                candidate = f"{scheme}://localhost:{port}"
                if candidate not in origins:
                    origins.append(candidate)

    logger.info("CORS allowed origins: %s", origins)
    return origins


# Middleware is applied in reverse order (last added = outermost).
# Desired order: Correlation → Timing → CORS → route handler.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(CorrelationIDMiddleware)

app.include_router(health.router)
app.include_router(chat.router, prefix="/api")
app.include_router(complaint_intelligence.router, prefix="/api")
app.include_router(rag_routes.router)
app.include_router(memory_routes.router)
app.include_router(ai_grading.router)
app.include_router(exam_generation_api.router)


@app.get("/")
async def root():
    return {
        "service": "AI Orchestration Service",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=(settings.ENVIRONMENT == "development"),
    )
