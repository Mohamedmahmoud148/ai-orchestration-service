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


async def _run_orchestration(fastapi_request: Request, context: ExecutionContext) -> ExecutionContext:
    """
    Route through the LangGraph wrapper if enabled, else the direct
    Agent.run() path (default, unchanged behavior).

    See app/agents/graph.py — the graph-on path calls the exact same
    Agent.run(context) internally, just sequenced through a StateGraph.
    _PipelineStageError raised inside Agent.run() propagates through
    graph.ainvoke() unchanged, so the caller's existing except clause
    still catches it correctly either way.
    """
    graph = getattr(fastapi_request.app.state, "agent_graph", None)
    if graph is not None:
        result_state = await graph.ainvoke({"context": context})
        return result_state["context"]
    agent = _get_agent(fastapi_request)
    return await agent.run(context)


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

    # ── Delegate to Agent (direct, or via LangGraph wrapper if enabled) ──
    try:
        context = await _run_orchestration(fastapi_request, context)
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

    # Grab the ReactAgent instance if available (assembled at startup)
    react_agent = getattr(agent, "_react_agent", None)

    async def event_generator():
        import asyncio as _asyncio

        try:
            # ── Phase 1: Signal immediately that work has started ─────────
            yield f"data: {_json.dumps({'type': 'thinking'})}\n\n"

            # ── Phase 2: Decide streaming strategy ────────────────────────
            # ReactAgent path: run memory/context loading via agent (Stage 0),
            # then stream the final answer via ReactAgent.stream_run().
            # This gives true token-level streaming for the final LLM call
            # while keeping all the memory/RBAC/rate-limit logic intact.
            use_react_stream = react_agent is not None

            if use_react_stream:
                # Stage 0 only: load memory + academic profile into context.
                # We replicate the non-LLM setup stages from agent.run() so
                # memory is available but we drive the LLM ourselves below.
                memory_store = getattr(agent, "_memory_store", None)
                if memory_store:
                    try:
                        memory_key = (
                            f"{context.user_id}:{context.conversation_id}"
                            if context.conversation_id else context.user_id
                        )
                        memory = await memory_store.get_conversation(memory_key)
                        if memory:
                            context.add_metadata("memory", memory)
                        prefs = await memory_store.get_preferences(context.user_id)
                        if prefs:
                            context.add_metadata("preferences", prefs)
                        academic_profile = await memory_store.get_academic_profile(context.user_id)
                        if academic_profile:
                            context.add_metadata("academic_profile", academic_profile)
                    except Exception as mem_exc:
                        logger.warning("chat/stream: memory load failed — %s", mem_exc)

                # Stream via ReactAgent — tool calls run in parallel internally,
                # final answer is streamed token-by-token.
                collected: list[str] = []
                try:
                    async for token in react_agent.stream_run(context):
                        collected.append(token)
                        yield f"data: {_json.dumps({'type': 'token', 'content': token})}\n\n"
                    final_text = "".join(collected)
                    context.set_result(final_text)
                except Exception as exc:
                    logger.error("chat/stream: ReactAgent.stream_run failed — %s", exc, exc_info=True)
                    # Fall back to full pipeline
                    use_react_stream = False

                # Save memory after streaming
                if use_react_stream and memory_store:
                    try:
                        memory_key = (
                            f"{context.user_id}:{context.conversation_id}"
                            if context.conversation_id else context.user_id
                        )
                        await memory_store.save_conversation(memory_key, {
                            "last_intent": context.intent,
                            "last_result": context.result,
                            "entities": {},
                        })
                    except Exception:
                        pass

            if not use_react_stream:
                # ── Fallback: run full pipeline then replay result ─────────
                # Routed through _run_orchestration (same helper /api/chat
                # uses) so this path also goes through the LangGraph wrapper
                # when USE_LANGGRAPH_ORCHESTRATION is on — Phase 4 of the
                # agentic architecture upgrade. This fallback already
                # computes the full answer up front and fakes streaming by
                # replaying it in word chunks below, so there's no real
                # token-level streaming to preserve from astream_events —
                # routing through the same graph-or-direct dispatch as
                # /api/chat gives full parity more simply.
                #
                # Note: the result is NOT reassigned to `context` here —
                # ExecutionContext is mutated in place by both the direct
                # and graph paths (same object identity either way, see
                # tests/test_langgraph_wrapper.py), and `context` is a
                # closure variable read elsewhere in this nested generator;
                # reassigning it here would make Python treat it as local
                # to event_generator() throughout, breaking the earlier
                # reads above (UnboundLocalError).
                try:
                    await _run_orchestration(fastapi_request, context)
                except _PipelineStageError as exc:
                    yield f"data: {_json.dumps({'type': 'error', 'message': exc.detail})}\n\n"
                    yield "data: {\"type\": \"done\"}\n\n"
                    return

                final_text = str(context.result or "")

                # Replay the computed text in 3-word chunks at typing speed.
                # Only replay if we haven't already streamed it above.
                words = final_text.split()
                for i in range(0, len(words), 3):
                    chunk = " ".join(words[i:i + 3]) + " "
                    yield f"data: {_json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    await _asyncio.sleep(0.018)

            # ── Phase 3: Send metadata frame ──────────────────────────────
            executor_data = (context.metadata or {}).get("executor_data", {}) or {}
            meta_frame = {
                "type": "meta",
                "intent": context.intent or "unknown",
                "tool": context.selected_tool or "none",
                "model": context.selected_model or "unknown",
                "conversation_id": context.conversation_id,
                "suggestions": executor_data.get("suggestions", []),
                "actions_available": executor_data.get("actions_available", []),
                "emotion": detect_emotion(final_text if "final_text" in dir() else ""),
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
