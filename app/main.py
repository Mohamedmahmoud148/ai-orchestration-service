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
from google import genai

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
#  Lifespan: build Agent once, tear down cleanly
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; release them on shutdown."""
    setup_logging()
    logger.info("Starting AI Orchestration Service (env=%s)", settings.ENVIRONMENT)

    # ── Cloud LLM clients ──────────────────────────────────────
    gemini_client = None
    if settings.GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info("Gemini client initialised.")
    else:
        logger.warning("GEMINI_API_KEY not set — cloud LLM calls will fail.")

    # ── Component assembly ────────────────────────────────────
    model_router = ModelRouter(
        gemini_client=gemini_client,
        local_model_service=local_model_service,   # HuggingFace local models
    )

    planner = PlannerAgent(model_router=model_router, ranker=None)

    executor = PlanExecutor(
        backend_execution_func=tool_execution_client.execute_tool,
        model_router=model_router,
    )

    # ── Agent (single instance for the lifetime of the app) ───
    app.state.agent = Agent(
        planner=planner,
        tool_registry=tool_registry,
        model_router=model_router,
        executor=executor,
    )

    logger.info("Agent ready — flow: User → Agent → Planner → Module → Response")
    yield

    # ── Shutdown ──────────────────────────────────────────────
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
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
