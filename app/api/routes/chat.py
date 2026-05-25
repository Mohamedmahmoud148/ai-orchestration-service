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

import json as _json

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer

from app.agents.agent import Agent
from app.agents.execution_context import ExecutionContext
from app.agents.pipeline import _PipelineStageError
from app.core.logging import logger, get_correlation_id, set_request_user_id
from app.core.emotion import detect_emotion
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

    # Propagate user_id into the logging context for this request
    set_request_user_id(request.user_id or "-")

    # ── Rate limiting ─────────────────────────────────────────
    rate_limiter = getattr(fastapi_request.app.state, "rate_limiter", None)
    if rate_limiter is not None:
        allowed = await rate_limiter.is_allowed(request.user_id or "")
        if not allowed:
            logger.warning(
                "Rate limit exceeded. user_id=%s correlation=%s",
                request.user_id,
                get_correlation_id(),
            )
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment before trying again.",
            )

    logger.info(
        "Chat request received. user_id=%s role=%s correlation=%s",
        request.user_id, request.role, get_correlation_id(),
    )

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

        clarification_text = "\n".join(lines)
        return ChatResponse(
            response=clarification_text,
            conversation_id=context.conversation_id,
            intent_executed=context.intent,
            tool_used=context.selected_tool,
            model_used=context.selected_model,
            metadata=context.metadata,
            emotion=detect_emotion(clarification_text),
        )

    # Extract suggestions / actions injected by the executor
    executor_data    = (context.metadata or {}).get("executor_data", {}) or {}
    suggestions      = executor_data.get("suggestions",       [])
    actions_avail    = executor_data.get("actions_available", [])

    response_text = str(context.result or "")
    return ChatResponse(
        response=response_text,
        conversation_id=context.conversation_id,
        intent_executed=context.intent,
        tool_used=context.selected_tool,
        model_used=context.selected_model,
        metadata=context.metadata,
        suggestions=suggestions,
        actions_available=actions_avail,
        emotion=detect_emotion(response_text),
    )


# ─────────────────────────────────────────────────────────────
#  Streaming endpoint — SSE (token-by-token like ChatGPT)
# ─────────────────────────────────────────────────────────────

@router.post("/chat/stream", tags=["AI Chat"])
async def chat_stream_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    token=Depends(security),
):
    """
    Streaming chat — returns Server-Sent Events.

    Frame format (one per line):
      data: {"type": "token",    "content": "..."}    # incremental text chunk
      data: {"type": "meta",     "intent": "...",     # final metadata
                                  "tool":  "...", "model": "...",
                                  "suggestions": [...] }
      data: {"type": "done"}                          # stream finished
      data: {"type": "error",    "message": "..."}    # on failure

    The frontend should EventSource-style accumulate `content` tokens to display.
    """
    auth_header = fastapi_request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header missing.")

    set_request_user_id(request.user_id or "-")

    rate_limiter = getattr(fastapi_request.app.state, "rate_limiter", None)
    if rate_limiter is not None:
        if not await rate_limiter.is_allowed(request.user_id or ""):
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment.",
            )

    agent = _get_agent(fastapi_request)
    model_router = getattr(agent, "_model_router", None)

    context = ExecutionContext(
        user_id=request.user_id,
        role=request.role,
        message=request.message,
        conversation_id=request.conversation_id or "",
        history=request.history,
        academic_context=request.academic_context,
        metadata={"auth_header": auth_header, "explain": request.explain, "stream": True},
    )

    async def event_generator():
        try:
            # Run the full pipeline first — planner, RBAC, modules, narration —
            # then stream the FINAL text token-by-token. The pipeline itself
            # isn't token-streamable because it involves multiple LLM hops and
            # backend tool calls; what the user sees as "streaming" is the
            # final narration replayed at typing speed.
            try:
                await agent.run(context)
            except _PipelineStageError as exc:
                yield f"data: {_json.dumps({'type': 'error', 'message': exc.detail})}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
                return

            final_text = str(context.result or "")
            executor_data = (context.metadata or {}).get("executor_data", {}) or {}

            # If the model_router is available AND the response came from a
            # general_chat / narrator path, re-issue the call in streaming mode
            # for a true ChatGPT-like effect. Otherwise we stream the cached
            # text in word-sized chunks (still feels live, no extra LLM cost).
            stream_via_llm = (
                model_router is not None
                and context.intent in ("general_chat", None, "")
                and len(final_text) < 4000  # safety cap
            )

            if stream_via_llm:
                # Re-stream from the model so the user sees token-level output.
                # Use the same system + history we'd use for fallback.
                from app.agents.executor import ROLE_SYSTEM_PROMPTS
                role = context.role or "student"
                sys_prompt = ROLE_SYSTEM_PROMPTS.get(role, ROLE_SYSTEM_PROMPTS["student"])
                messages = [{"role": "system", "content": sys_prompt}]
                for turn in (context.history or [])[-10:]:
                    tr = turn.get("role")
                    tc = turn.get("content", "")
                    if tr in ("user", "assistant") and tc:
                        messages.append({"role": tr, "content": str(tc)})
                messages.append({"role": "user", "content": context.message})

                full = []
                try:
                    async for chunk in model_router.stream_with_messages(
                        messages=messages,
                        model_id=context.selected_model or "openai/gpt-4o-mini",
                    ):
                        full.append(chunk)
                        yield f"data: {_json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    final_text = "".join(full) or final_text
                except Exception as exc:
                    logger.warning("chat/stream: LLM stream failed (%s) — falling back to chunked replay", exc)
                    # Fall back to chunked replay below
                    stream_via_llm = False

            if not stream_via_llm:
                # Replay the already-computed text in small chunks so the UI
                # still feels alive even for tool-result narrations.
                import asyncio
                words = final_text.split(" ")
                buf = []
                for i, w in enumerate(words):
                    buf.append(w)
                    if len(buf) >= 3 or i == len(words) - 1:
                        chunk = (" " if i >= 3 else "") + " ".join(buf)
                        yield f"data: {_json.dumps({'type': 'token', 'content': chunk})}\n\n"
                        buf = []
                        await asyncio.sleep(0.02)

            meta_frame = {
                "type": "meta",
                "intent": context.intent or "unknown",
                "tool": context.selected_tool or "none",
                "model": context.selected_model or "unknown",
                "conversation_id": context.conversation_id,
                "suggestions": executor_data.get("suggestions", []),
                "actions_available": executor_data.get("actions_available", []),
                "emotion": detect_emotion(final_text),
            }
            yield f"data: {_json.dumps(meta_frame, ensure_ascii=False)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as exc:
            logger.error("chat/stream unexpected error: %s", exc, exc_info=True)
            yield f"data: {_json.dumps({'type': 'error', 'message': 'internal error'})}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )
