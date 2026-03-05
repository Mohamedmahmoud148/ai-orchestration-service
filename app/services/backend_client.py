"""
backend_client.py

Provides the ToolExecutionClient for Version 2: Command Bridge architecture.
All tool execution is forwarded to the .NET backend. The FastAPI service
has no direct database access and contains zero tool business logic.

Error contract
--------------
- RuntimeError        → configuration error at construction time
- HTTPException(502)  → backend returned an error or a network failure occurred

The pipeline must NEVER silently degrade on backend failures.
"""

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import logger


class ToolExecutionClient:
    """
    HTTP client to securely call the .NET backend AI Execution Layer.

    Raises
    ------
    RuntimeError
        If ``BACKEND_BASE_URL`` is not set when the client is constructed.
    HTTPException(502)
        On any HTTP-level or network-level failure during a request.
    """

    def __init__(self) -> None:
        base_url = (settings.BACKEND_BASE_URL or "").rstrip("/")
        if not base_url:
            # Settings validator should have caught this; guard defensively.
            raise RuntimeError(
                "ToolExecutionClient: BACKEND_BASE_URL is not configured. "
                "Set it in your .env file before starting the service."
            )
        self.base_url = base_url

    # ------------------------------------------------------------------
    # POST — trigger backend actions
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        route: str,
        parameters: Dict[str, Any],
        auth_header: Optional[str],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a backend tool over HTTP POST.

        Parameters
        ----------
        route : str
            Backend path, e.g. ``/api/ai/execute/create-generated-exam``.
        parameters : dict
            Payload forwarded to the tool.
        auth_header : str | None
            Forwarded ``Authorization`` header (JWT).
        user_id : str | None
            Caller's user ID forwarded to the backend.

        Returns
        -------
        dict
            Parsed JSON response from the backend.

        Raises
        ------
        HTTPException(502)
            On any HTTP-level or network-level failure.
        """
        url = f"{self.base_url}{route}"

        payload = {
            "parameters": parameters,
            "userId": user_id,
        }

        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "BackendClient POST url=%s user_id=%s parameters=%s",
            url,
            user_id,
            parameters,
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as exc:
            text = exc.response.text
            logger.error(
                "BackendClient POST HTTP %s from %s: %s",
                exc.response.status_code,
                url,
                text,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Backend tool call failed with HTTP {exc.response.status_code}: "
                    f"{text[:300]}"
                ),
            ) from exc

        except httpx.RequestError as exc:
            logger.error("BackendClient POST network error at %s: %s", url, str(exc))
            raise HTTPException(
                status_code=502,
                detail=f"Network error communicating with backend at {url}: {exc}",
            ) from exc

        except Exception as exc:
            logger.error(
                "BackendClient POST unexpected error: %s", str(exc), exc_info=True
            )
            raise HTTPException(
                status_code=502,
                detail=f"Unexpected error executing backend tool: {exc}",
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
        """
        HTTP GET against a .NET backend route.

        Used by modules that need to *read* data (materials, results, files …)
        rather than trigger an action.

        Parameters
        ----------
        route : str
            Backend path, e.g. ``/api/Materials/by-offering/42``.
        auth_header : str | None
            Forwarded JWT so the backend can authorise the request.
        params : dict | None
            Optional query-string parameters.

        Returns
        -------
        dict
            Parsed JSON response, or
            ``{"_raw_bytes": bytes, "content_type": str}`` for non-JSON
            responses (e.g. file downloads).

        Raises
        ------
        HTTPException(502)
            On any HTTP-level or network-level failure.
        """
        url = f"{self.base_url}{route}"
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info("BackendClient GET url=%s params=%s", url, params)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=headers,
                    params=params or {},
                    timeout=30.0,
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    return response.json()
                # Return raw bytes under a known key for file-like responses
                return {"_raw_bytes": response.content, "content_type": content_type}

        except httpx.HTTPStatusError as exc:
            text = exc.response.text
            logger.error(
                "BackendClient GET HTTP %s from %s: %s",
                exc.response.status_code,
                url,
                text,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Backend GET failed with HTTP {exc.response.status_code}: "
                    f"{text[:300]}"
                ),
            ) from exc

        except httpx.RequestError as exc:
            logger.error("BackendClient GET network error at %s: %s", url, str(exc))
            raise HTTPException(
                status_code=502,
                detail=f"Network error fetching from backend at {url}: {exc}",
            ) from exc

        except Exception as exc:
            logger.error(
                "BackendClient GET unexpected error: %s", str(exc), exc_info=True
            )
            raise HTTPException(
                status_code=502,
                detail=f"Unexpected error in BackendClient.fetch: {exc}",
            ) from exc


# Singleton — constructed once at import time.
# Will raise RuntimeError immediately if BACKEND_BASE_URL is not set.
tool_execution_client = ToolExecutionClient()
