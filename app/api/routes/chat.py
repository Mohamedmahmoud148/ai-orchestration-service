"""
chat.py — /api/chat endpoint

Responsibilities (ONLY these):
  1. Authenticate the caller.
  2. Build an ExecutionContext from the request.
  3. Delegate everything to OrchestrationPipeline.run().
  4. Serialise context.result into a ChatResponse.

This file must NEVER:
  - Call Gemini / OpenAI / any LLM directly.
  - Call PlannerAgent, ToolRegistry, ModelRouter, or PlanExecutor directly.
  - Contain if/else logic based on intent, role, or tool.
"""

from fastapi import APIRouter, HTTPException, Request

from app.agents.execution_context import ExecutionContext
from app.agents.pipeline import OrchestrationPipeline, _PipelineStageError
from app.core.logging import logger
from app.models.chat import ChatRequest, ChatResponse

# ─────────────────────────────────────────────────────────────
#  Pipeline is assembled once at import time via the app
#  state (set in main.py).  The endpoint reads it from there.
# ─────────────────────────────────────────────────────────────
router = APIRouter()


def _get_pipeline(fastapi_request: Request) -> OrchestrationPipeline:
    """Retrieve the pre-built pipeline from FastAPI app state."""
    pipeline: OrchestrationPipeline = fastapi_request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Orchestration pipeline not initialised.")
    return pipeline


@router.post("/chat", response_model=ChatResponse, tags=["AI Chat"])
async def chat_endpoint(request: ChatRequest, fastapi_request: Request):
    """
    Unified chat entry-point.

    Delegates ALL orchestration logic to OrchestrationPipeline.run().
    This handler contains zero business logic.
    """
    # ── Auth ─────────────────────────────────────────────────
    auth_header = fastapi_request.headers.get("Authorization")
    if not auth_header:
        logger.warning(
            "Unauthorized chat attempt — missing Authorization header. user_id=%s",
            request.user_id,
        )
        raise HTTPException(status_code=401, detail="Authorization header missing.")

    logger.info("Chat request received. user_id=%s role=%s", request.user_id, request.role)

    # ── Build context ─────────────────────────────────────────
    context = ExecutionContext(
        user_id=request.user_id or "anonymous",
        role=request.role,
        message=request.message,
        conversation_id=request.conversation_id or "",   # pipeline generates UUID if empty
        history=request.history,
        academic_context=request.academic_context,
        metadata={"auth_header": auth_header},           # forwarded for backend_client inside executor
    )

    # ── Delegate to pipeline ──────────────────────────────────
    pipeline = _get_pipeline(fastapi_request)

    try:
        await pipeline.run(context)
    except _PipelineStageError as exc:
        logger.error(
            "Pipeline aborted. stage=%s conversation_id=%s detail=%s",
            exc.stage,
            context.conversation_id,
            exc.detail,
        )
        raise HTTPException(status_code=500, detail=exc.detail)

    # ── Serialise & return ────────────────────────────────────
    return ChatResponse(
        response=str(context.result or ""),
        conversation_id=context.conversation_id,
        intent_executed=context.intent,
        tool_used=context.selected_tool,
        model_used=context.selected_model,
        metadata=context.metadata,
    )
