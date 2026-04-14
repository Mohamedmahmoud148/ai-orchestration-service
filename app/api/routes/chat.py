"""
chat.py — /api/chat endpoint

Responsibilities (ONLY these):
  1. Authenticate the caller.
  2. Build an ExecutionContext from the request.
  3. Delegate everything to Agent.run().
  4. Serialise context.result into a ChatResponse.

This file must NEVER:
  - Call Gemini / OpenAI / any LLM directly.
  - Call PlannerAgent, ToolRegistry, ModelRouter, or PlanExecutor directly.
  - Contain if/else logic based on intent, role, or tool.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.security import HTTPBearer

from app.agents.agent import Agent
from app.agents.execution_context import ExecutionContext
from app.agents.pipeline import _PipelineStageError
from app.core.logging import logger
from app.models.chat import ChatRequest, ChatResponse

# ─────────────────────────────────────────────────────────────
#  Agent is assembled once at startup in main.py and stored
#  in app.state.agent.  The endpoint reads it from there.
# ─────────────────────────────────────────────────────────────
router = APIRouter()
security = HTTPBearer(auto_error=False)

def _get_agent(fastapi_request: Request) -> Agent:
    """Retrieve the pre-built Agent from FastAPI app state."""
    agent: Agent = fastapi_request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    return agent


@router.post("/chat", response_model=ChatResponse, tags=["AI Chat"])
async def chat_endpoint(
    request: ChatRequest, 
    fastapi_request: Request,
    token = Depends(security)
):
    """
    Unified chat entry-point.

    Delegates ALL orchestration logic to Agent.run().
    This handler contains zero business logic.
    """
    # ── Auth ──────────────────────────────────────────────────
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
        user_id=request.user_id,
        role=request.role,
        message=request.message,
        conversation_id=request.conversation_id or "",
        history=request.history,
        academic_context=request.academic_context,
        metadata={
            "auth_header": auth_header,
            "explain":     request.explain,
        },
    )

    # ── Delegate to Agent ─────────────────────────────────────
    agent = _get_agent(fastapi_request)

    try:
        await agent.run(context)
    except _PipelineStageError as exc:
        logger.error(
            "Agent aborted. stage=%s conversation_id=%s detail=%s",
            exc.stage,
            context.conversation_id,
            exc.detail,
        )
        # Executor/module failures (failed/forbidden) are user-facing messages,
        # NOT server crashes. Return them as 200 so the client can display them.
        if exc.stage == "executor":
            return ChatResponse(
                response=exc.detail,
                conversation_id=context.conversation_id,
                intent_executed=context.intent or "unknown",
                tool_used=context.selected_tool or "none",
                model_used=context.selected_model or "unknown",
                metadata=context.metadata,
                suggestions=[],
                actions_available=[],
            )
        # Planning/infrastructure failures are real 500s
        raise HTTPException(status_code=500, detail=exc.detail)

    # ── Serialise & return ────────────────────────────────────
    # Handle Clarification Disambiguation
    if context.metadata and context.metadata.get("clarification_needed"):
        options = context.metadata.get("clarification_options", [])
        prefix = str(context.result) + "\n\nتقصد أي واحد من دول؟" if context.result and context.result != "clarification_needed" else "تقصد أي واحد من دول؟"
        lines = [prefix]
        for i, opt in enumerate(options, 1):
            name = opt.get("title") or opt.get("name") or opt.get("subjectName") or "Unknown"
            id_val = opt.get("id") or opt.get("subjectOfferingId") or "?"
            lines.append(f"{i}. {name} ({id_val})")
            
        return ChatResponse(
            response="\n".join(lines),
            conversation_id=context.conversation_id,
            intent_executed=context.intent,
            tool_used=context.selected_tool,
            model_used=context.selected_model,
            metadata=context.metadata,
        )

    # Extract suggestions / actions injected by the executor
    executor_data    = (context.metadata or {}).get("executor_data", {}) or {}
    suggestions      = executor_data.get("suggestions",       [])
    actions_avail    = executor_data.get("actions_available", [])

    return ChatResponse(
        response=str(context.result or ""),
        conversation_id=context.conversation_id,
        intent_executed=context.intent,
        tool_used=context.selected_tool,
        model_used=context.selected_model,
        metadata=context.metadata,
        suggestions=suggestions,
        actions_available=actions_avail,
    )
