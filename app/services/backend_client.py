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


# Export a singleton instance.
tool_execution_client = ToolExecutionClient()
