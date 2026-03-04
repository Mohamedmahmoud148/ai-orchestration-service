"""
backend_client.py

Provides the ToolExecutionClient for Version 2: Command Bridge architecture.
All tool execution is forwarded to the .NET backend. The FastAPI service
has no direct database access and contains zero tool business logic.
"""

from typing import Any, Dict, Optional

import httpx

from app.core.config import settings
from app.core.logging import logger


class ToolExecutionClient:
    """
    HTTP Client to securely call the .NET backend AI Execution Layer.
    """

    def __init__(self) -> None:
        self.base_url = (settings.BACKEND_BASE_URL or "").rstrip("/")

    async def execute_tool(
        self,
        route: str,
        parameters: Dict[str, Any],
        auth_header: Optional[str],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a backend tool over HTTP using the Command Bridge pattern.

        Parameters
        ----------
        route : str
            The specific backend route for this tool (e.g. "/api/ai/execute/schedule-meeting").
        parameters : dict
            The extracted parameters for the tool payload.
        auth_header : str | None
            The forwarded Authorization header to preserve the caller's identity.
        user_id : str | None
            The caller's user ID.

        Returns
        -------
        dict
            The structured response from the backend. Contains an "error" key
            if the execution failed at the HTTP or application layer.
        """
        if not self.base_url:
            logger.error("BACKEND_BASE_URL is not set. Cannot execute tools.")
            return {"error": "Backend URL configuration missing"}

        url = f"{self.base_url}{route}"

        # Model matches AiExecutionRequest in .NET
        payload = {
            "parameters": parameters,
            "userId": user_id,
        }

        headers = {}
        if auth_header:
            headers["Authorization"] = auth_header

        logger.info(
            "Executing remote tool. url=%s user_id=%s parameters=%s",
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

        except httpx.HTTPStatusError as e:
            text = e.response.text
            logger.error("Backend tool returned HTTP %s: %s", e.response.status_code, text)
            return {"error": f"Backend HTTP error {e.response.status_code}: {text}"}
        except httpx.RequestError as e:
            logger.error("Network error calling backend tool at %s: %s", url, str(e))
            return {"error": f"Network error calling backend: {str(e)}"}
        except Exception as e:
            logger.error("Unexpected error executing backend tool: %s", str(e), exc_info=True)
            return {"error": f"Unexpected execution error: {str(e)}"}

    async def fetch(
        self,
        route: str,
        auth_header: Optional[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        HTTP GET against a .NET backend route.

        Used by modules that need to *read* data (materials, results, etc.)
        rather than trigger an action.

        Parameters
        ----------
        route : str
            Backend path, e.g. "/api/Materials/by-offering/42".
        auth_header : str | None
            Forwarded JWT so the backend can authorise the request.
        params : dict | None
            Optional query-string parameters.
        """
        if not self.base_url:
            logger.error("BACKEND_BASE_URL is not set. Cannot fetch data.")
            return {"error": "Backend URL configuration missing"}

        url = f"{self.base_url}{route}"
        headers = {}
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
                # Some endpoints return plain text / bytes (e.g. file download)
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    return response.json()
                # Return raw bytes under a known key for file-like responses
                return {"_raw_bytes": response.content, "content_type": content_type}

        except httpx.HTTPStatusError as e:
            text = e.response.text
            logger.error("Backend GET returned HTTP %s: %s", e.response.status_code, text)
            return {"error": f"Backend HTTP error {e.response.status_code}: {text}"}
        except httpx.RequestError as e:
            logger.error("Network error fetching from backend at %s: %s", url, str(e))
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            logger.error("Unexpected error in BackendClient.fetch: %s", str(e), exc_info=True)
            return {"error": f"Unexpected error: {str(e)}"}


# Export a singleton instance.
tool_execution_client = ToolExecutionClient()
