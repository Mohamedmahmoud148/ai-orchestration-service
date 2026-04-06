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

Error contract
--------------
- RuntimeError        → configuration error at construction time only.
- HTTPException(502)  → any HTTP-level or network-level failure.

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

_BACKEND_UNAVAILABLE_MSG = (
    "The backend service is currently unavailable. Please try again later."
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

    Raises
    ------
    RuntimeError
        If BACKEND_BASE_URL is not set when the client is constructed.
    HTTPException(502)
        On any HTTP-level or network-level failure during a request.
    """

    def __init__(self) -> None:
        base_url = (settings.BACKEND_BASE_URL or "").rstrip("/")
        if not base_url:
            raise RuntimeError(
                "ToolExecutionClient: BACKEND_BASE_URL is not configured. "
                "Set it in your .env file before starting the service."
            )
        self.base_url = base_url

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
        url = f"{self.base_url}{route}"
        payload = {"parameters": parameters, "userId": user_id}
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "BackendClient [POST-TOOL] endpoint=%s user_id=%s request_body=%s",
            url, user_id, parameters,
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, json=payload, headers=headers, timeout=30.0
                )
                logger.info(
                    "BackendClient [POST-TOOL] endpoint=%s status=%s response=%s",
                    url, response.status_code, response.text[:500],
                )
                response.raise_for_status()
                return _safe_json(response)

        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            logger.error(
                "BackendClient [POST-TOOL] HTTP %s from %s: %s",
                exc.response.status_code, url, body,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except httpx.RequestError as exc:
            logger.error(
                "BackendClient [POST-TOOL] network error at %s: %s", url, exc
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except HTTPException:
            raise  # re-raise without wrapping

        except Exception as exc:
            logger.error(
                "BackendClient [POST-TOOL] unexpected error: %s",
                str(exc), exc_info=True,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

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
        url = f"{self.base_url}{route}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "BackendClient [POST] endpoint=%s request_body=%s",
            url, payload,
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, json=payload, headers=headers, timeout=30.0
                )
                logger.info(
                    "BackendClient [POST] endpoint=%s status=%s response=%s",
                    url, response.status_code, response.text[:500],
                )
                response.raise_for_status()
                # 201 / 204 responses may have an empty body — handle safely
                if not response.content:
                    return {"status": "created", "http_status": response.status_code}
                return _safe_json(response)

        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            logger.error(
                "BackendClient [POST] HTTP %s from %s: %s",
                exc.response.status_code, url, body,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except httpx.RequestError as exc:
            logger.error(
                "BackendClient [POST] network error at %s: %s", url, exc
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except HTTPException:
            raise

        except Exception as exc:
            logger.error(
                "BackendClient [POST] unexpected error: %s",
                str(exc), exc_info=True,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

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
        url = f"{self.base_url}{route}"
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "BackendClient [GET] endpoint=%s query_params=%s",
            url, params,
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, headers=headers, params=params or {}, timeout=30.0
                )
                logger.info(
                    "BackendClient [GET] endpoint=%s status=%s response=%s",
                    url, response.status_code, response.text[:500],
                )
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if not response.content:
                    # Empty 200/204: return empty sentinel rather than crashing
                    logger.warning(
                        "BackendClient [GET] endpoint=%s returned empty body", url
                    )
                    return {}

                if "application/json" in content_type:
                    return _safe_json(response)

                # Non-JSON (e.g. PDF, binary): return raw bytes
                return {"_raw_bytes": response.content, "content_type": content_type}

        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            logger.error(
                "BackendClient [GET] HTTP %s from %s: %s",
                exc.response.status_code, url, body,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except httpx.RequestError as exc:
            logger.error(
                "BackendClient [GET] network error at %s: %s", url, exc
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc

        except HTTPException:
            raise

        except Exception as exc:
            logger.error(
                "BackendClient [GET] unexpected error: %s",
                str(exc), exc_info=True,
            )
            raise HTTPException(
                status_code=502,
                detail=_BACKEND_UNAVAILABLE_MSG,
            ) from exc


# Singleton — constructed once at import time.
# Will raise RuntimeError immediately if BACKEND_BASE_URL is not set.
tool_execution_client = ToolExecutionClient()
