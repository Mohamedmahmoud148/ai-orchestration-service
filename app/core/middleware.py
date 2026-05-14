"""
app/core/middleware.py

Production-grade ASGI middleware stack.

CorrelationIDMiddleware
    — Reads X-Request-ID from incoming headers (or generates a UUID4).
    — Stores it in the logging ContextVar so every log line for this
      request carries the same ID automatically.
    — Echoes it back in the response header.

RequestTimingMiddleware
    — Records request start time in a ContextVar.
    — Logs METHOD path → status_code in Xms at INFO level after the
      response is sent.
    — Adds X-Response-Time-Ms to the response.
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import (
    logger,
    set_correlation_id,
    set_request_user_id,
    mark_request_start,
)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Propagates / generates a unique X-Request-ID for every HTTP request.

    Priority:
      1. Use X-Request-ID header sent by the client (load-balancer / frontend).
      2. Use X-Correlation-ID header (alternative convention).
      3. Generate a fresh UUID4.

    The ID is stored in the `_correlation_id` ContextVar so it is
    automatically included in every log line for the duration of the request.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        cid = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or str(uuid.uuid4())
        )
        set_correlation_id(cid)

        # Best-effort: extract user_id from request body for log context.
        # We read it from the query string or JSON body header to avoid
        # consuming the body stream (which would break downstream handlers).
        uid = request.query_params.get("user_id", "-")
        set_request_user_id(uid)

        response = await call_next(request)
        response.headers["X-Request-ID"] = cid
        return response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Measures and logs end-to-end request duration.

    Skips /health to avoid log noise from container health checks.
    Adds X-Response-Time-Ms to every response for client-side monitoring.
    """

    _SKIP_PATHS = frozenset({"/health", "/", "/docs", "/openapi.json"})

    async def dispatch(self, request: Request, call_next) -> Response:
        mark_request_start()
        start = time.monotonic()

        response = await call_next(request)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

        if request.url.path not in self._SKIP_PATHS:
            logger.info(
                "%s %s → %d  (%dms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )

        return response
