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

from app.api.routes import chat, health
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.services.backend_client import tool_execution_client
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

    # Confirm BackendClient singleton initialised correctly.
    # (ToolExecutionClient.__init__ raises RuntimeError if URL is missing;
    #  this is a belt-and-suspenders log confirming it is alive.)
    logger.info(
        "ToolExecutionClient ready (base_url=%s).", tool_execution_client.base_url
    )

    # ── 2. OpenRouter LLM client (OpenAI-compatible) ──────────────────
    openai_client = AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "University AI System",
            "X-Title": "University AI Agent",
        },
    )
    logger.info("OpenRouter client initialised (base_url=https://openrouter.ai/api/v1).")

    # ── 3. Component assembly ─────────────────────────────────────────
    model_router = ModelRouter(
        openai_client=openai_client,
        local_model_service=local_model_service,
    )

    planner = PlannerAgent(
        model_router=model_router,
        ranker=None
    )

    executor = PlanExecutor(
        backend_execution_func=tool_execution_client.execute_tool,
        model_router=model_router,
    )

    # ── 4. Agent (single instance for the lifetime of the app) ────────
    app.state.agent = Agent(
        planner=planner,
        tool_registry=tool_registry,
        model_router=model_router,
        executor=executor,
    )

    logger.info("Agent ready — flow: User → Agent → Planner → Module → Response")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────
    logger.info("Shutting down AI Orchestration Service.")
    app.state.agent = None


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


app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router, prefix="/api")


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
