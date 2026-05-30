"""
backend_client.py

Provides the ToolExecutionClient for the FastAPI AI orchestration service.

Methods
-------
execute_tool(route, parameters, auth_header, user_id)
    POST with the legacy {"parameters": …, "userId": …} envelope.
    Kept for any remaining legacy endpoints.
post(route, payload, auth_header)
    Direct POST — raw JSON body, no wrapper. Used by /api/Exams and
    other RESTful .NET endpoints that expect a plain DTO body.
fetch(route, auth_header, params)
    GET with optional query params. Used for resolve-offering, materials, etc.

Hardening (Phase 5 Critical fixes):
-----------------------------------
- C4: Single `httpx.AsyncClient` reused across requests (connection pool).
      Saves ~30ms TLS handshake per call AND keeps the worker from running
      out of sockets under load.
- C3: Circuit breaker — after N consecutive failures the client rejects
      calls immediately instead of waiting 30s on a dead .NET service.
      This prevents one outage from cascading into worker pool exhaustion.

Error contract
--------------
- RuntimeError        → configuration error at construction time only.
- HTTPException(502)  → any HTTP-level or network-level failure
                        OR circuit breaker open (fail-fast).

Logging contract
----------------
Every method logs (before the call):
    [METHOD] endpoint=<url>  (+ request_body or query_params)
Every method logs (after a successful call):
    [METHOD] endpoint=<url> status=<code> response=<first 500 chars>

Safety
------
- All JSON parsing is wrapped with a try/except to handle non-JSON bodies
  gracefully, so callers never receive a raw JSONDecodeError.
- HTTP error bodies are truncated to 300 chars in exception details to
  prevent credential or PII leaks in log aggregators.
"""

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import logger
from app.services.circuit_breaker import CircuitBreaker, CircuitOpenError

_BACKEND_UNAVAILABLE_MSG = (
    "The backend service is currently unavailable. Please try again later."
)

_CIRCUIT_OPEN_MSG = (
    "The backend service is temporarily unavailable due to repeated failures. "
    "Service will retry automatically in a few seconds."
)


def _safe_json(response: httpx.Response) -> Any:
    """Attempt to parse a response body as JSON; return raw text on failure."""
    try:
        return response.json()
    except Exception:
        return {"_raw_text": response.text}


class ToolExecutionClient:
    """
    HTTP client to securely call the .NET backend.

    The httpx client is created lazily on first use (or via `start()` from
    the FastAPI lifespan) and shared across all requests. Connection pooling
    keeps latency low under load.

    Raises
    ------
    RuntimeError
        If BACKEND_BASE_URL is not set when the client is constructed.
    HTTPException(502)
        On any HTTP-level or network-level failure during a request OR when
        the circuit breaker has tripped.
    """

    def __init__(self) -> None:
        base_url = (settings.BACKEND_BASE_URL or "").rstrip("/")
        if not base_url:
            raise RuntimeError(
                "ToolExecutionClient: BACKEND_BASE_URL is not configured. "
                "Set it in your .env file before starting the service."
            )
        self.base_url = base_url

        # Single shared client → connection pooling. Created on first use OR
        # by the lifespan via start(). The aclose() must run on shutdown.
        self._client: Optional[httpx.AsyncClient] = None

        self._breaker = CircuitBreaker(
            name="backend_dotnet",
            fail_threshold=settings.BACKEND_BREAKER_FAIL_THRESHOLD,
            reset_seconds=settings.BACKEND_BREAKER_RESET_SECONDS,
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks — called from main.py lifespan
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the shared httpx client. Idempotent."""
        if self._client is None:
            self._client = self._build_client()
            logger.info(
                "ToolExecutionClient: httpx pool initialised (base_url=%s, timeout=%.1fs)",
                self.base_url, settings.BACKEND_TIMEOUT_SECONDS,
            )

    async def aclose(self) -> None:
        """Close the shared httpx client. Safe to call multiple times."""
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None
            logger.info("ToolExecutionClient: httpx pool closed")

    def breaker_status(self) -> dict:
        """Return circuit breaker state — used by /health endpoints."""
        return self._breaker.status()

    # ------------------------------------------------------------------
    # Internal: ensure the client exists (lazy fallback)
    # ------------------------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        """Construct the shared client with sane pool + timeout settings."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(settings.BACKEND_TIMEOUT_SECONDS),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )

    async def _client_or_init(self) -> httpx.AsyncClient:
        """Lazy initialiser — guarantees a client even if start() wasn't called."""
        if self._client is None:
            self._client = self._build_client()
            logger.info("ToolExecutionClient: lazy-initialised httpx pool")
        return self._client

    # ------------------------------------------------------------------
    # POST (legacy envelope) — for old /api/ai/execute/* endpoints
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        route: str,
        parameters: Dict[str, Any],
        auth_header: Optional[str],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST with the legacy {parameters, userId} envelope."""
        url = route if route.startswith("http") else route
        payload = {"parameters": parameters, "userId": user_id}
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "BackendClient [POST-TOOL] endpoint=%s user_id=%s request_body=%s",
            url, user_id, parameters,
        )

        async def _do_call() -> Dict[str, Any]:
            client = await self._client_or_init()
            response = await client.post(url, json=payload, headers=headers)
            logger.info(
                "BackendClient [POST-TOOL] endpoint=%s status=%s response=%s",
                url, response.status_code, response.text[:500],
            )
            response.raise_for_status()
            return _safe_json(response)

        return await self._guarded(_do_call, url)

    # ------------------------------------------------------------------
    # POST (direct) — for RESTful endpoints like POST /api/Exams
    # ------------------------------------------------------------------

    async def post(
        self,
        route: str,
        payload: Dict[str, Any],
        auth_header: Optional[str],
    ) -> Dict[str, Any]:
        """Direct JSON POST — no envelope wrapper."""
        url = route
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "BackendClient [POST] endpoint=%s request_body=%s",
            url, payload,
        )

        async def _do_call() -> Dict[str, Any]:
            client = await self._client_or_init()
            response = await client.post(url, json=payload, headers=headers)
            logger.info(
                "BackendClient [POST] endpoint=%s status=%s response=%s",
                url, response.status_code, response.text[:500],
            )

            # Auth errors: return soft dict instead of raising so callers
            # can surface user-friendly messages.
            if response.status_code in (401, 403):
                logger.warning(
                    "BackendClient [POST] Auth error (%s) on %s — soft error",
                    response.status_code, url,
                )
                return {"_error": "unauthorized", "status_code": response.status_code}

            response.raise_for_status()
            if not response.content:
                return {"status": "created", "http_status": response.status_code}
            return _safe_json(response)

        return await self._guarded(_do_call, url)

    # ------------------------------------------------------------------
    # GET — read data from the backend
    # ------------------------------------------------------------------

    async def fetch(
        self,
        route: str,
        auth_header: Optional[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """HTTP GET — returns JSON dict or {_raw_bytes, content_type} for files."""
        url = route
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "BackendClient [GET] endpoint=%s query_params=%s",
            url, params,
        )

        async def _do_call() -> Dict[str, Any]:
            client = await self._client_or_init()
            response = await client.get(url, headers=headers, params=params or {})
            logger.info(
                "BackendClient [GET] endpoint=%s status=%s response=%s",
                url, response.status_code, response.text[:500],
            )

            # Soft 401/403: return structured dict instead of crashing,
            # so the module surfaces a friendlier message.
            if response.status_code in (401, 403):
                logger.warning(
                    "BackendClient [GET] Auth error (%s) on %s — soft error",
                    response.status_code, url,
                )
                return {"_error": "unauthorized", "status_code": response.status_code}

            # Soft 404: same reason.
            if response.status_code == 404:
                logger.warning(
                    "BackendClient [GET] 404 on %s — soft not_found", url,
                )
                return {"_error": "not_found", "status_code": 404}

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if not response.content:
                logger.warning(
                    "BackendClient [GET] endpoint=%s returned empty body", url,
                )
                return {}

            if "application/json" in content_type:
                return _safe_json(response)

            return {"_raw_bytes": response.content, "content_type": content_type}

        return await self._guarded(_do_call, url)

    # ------------------------------------------------------------------
    # Internal: shared guard wrapping every HTTP call
    # ------------------------------------------------------------------

    async def _guarded(self, do_call, url: str) -> Dict[str, Any]:
        """
        Run a call through the circuit breaker and translate errors uniformly.

        Soft auth/not-found responses are NOT counted as breaker failures —
        the backend is alive and answering, just denying the specific request.
        """
        try:
            result = await self._breaker.call(do_call)
            return result

        except CircuitOpenError:
            logger.error(
                "BackendClient: circuit OPEN — short-circuited call to %s", url,
            )
            raise HTTPException(
                status_code=503,
                detail=_CIRCUIT_OPEN_MSG,
            )

        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            status_code = exc.response.status_code
            logger.error(
                "BackendClient HTTP %s from %s: %s",
                status_code, url, body,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except httpx.RequestError as exc:
            logger.error("BackendClient network error at %s: %s", url, exc)
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except HTTPException:
            raise

        except Exception as exc:
            logger.error(
                "BackendClient unexpected error: %s",
                str(exc), exc_info=True,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc


# Singleton — constructed once at import time.
# Will raise RuntimeError immediately if BACKEND_BASE_URL is not set.
tool_execution_client = ToolExecutionClient()
