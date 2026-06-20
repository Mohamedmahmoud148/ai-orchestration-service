"""
app/core/observability.py — Structured Logging & Observability — Section 9

Adds structured context to every request:
  - RequestId
  - CorrelationId
  - UserId
  - ConversationId
  - SelectedAgent (ReactAgent / LegacyPipeline)
  - EmbeddingProvider
  - ExecutionTime

Usage in FastAPI middleware (add to main.py):
    from app.core.observability import ObservabilityMiddleware
    app.add_middleware(ObservabilityMiddleware)
"""
from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import logger


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Adds CorrelationId + RequestId to every request/response.
    Logs structured request metadata after each response.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ── Correlation ID ────────────────────────────────────────────────────
        correlation_id = (
            request.headers.get("X-Correlation-Id")
            or str(uuid.uuid4())[:16]
        )
        request_id = str(uuid.uuid4())[:16]

        # Store on request state for access by route handlers
        request.state.correlation_id = correlation_id
        request.state.request_id     = request_id

        t0 = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── Add to response headers ───────────────────────────────────────────
        response.headers["X-Correlation-Id"] = correlation_id
        response.headers["X-Request-Id"]     = request_id

        # ── Structured log ────────────────────────────────────────────────────
        # Skip noise from health checks and static assets
        path = request.url.path
        if path not in ("/health", "/docs", "/openapi.json", "/favicon.ico"):
            logger.info(
                "HTTP %s %s → %d | duration=%sms correlation=%s request=%s",
                request.method,
                path,
                response.status_code,
                duration_ms,
                correlation_id,
                request_id,
            )

        return response


def log_ai_execution(
    user_id: str,
    conversation_id: str,
    selected_agent: str,
    tool_calls: int,
    duration_s: float,
    fallback_triggered: bool = False,
    fallback_reason: str = "",
    embedding_provider: str = "",
):
    """
    Structured log for AI execution telemetry.
    Called after each ReactAgent / LegacyPipeline execution.
    """
    logger.info(
        "AI_EXEC user_id=%s conversation_id=%s selected_agent=%s "
        "tool_calls=%d duration_s=%.3f fallback=%s fallback_reason=%r "
        "embedding_provider=%s",
        user_id,
        conversation_id,
        selected_agent,
        tool_calls,
        duration_s,
        fallback_triggered,
        fallback_reason,
        embedding_provider or "n/a",
    )
