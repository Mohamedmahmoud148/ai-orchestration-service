import httpx
from typing import Any, Dict, Optional
from app.core.config import settings
from app.core.logging import logger

class BackendClient:
    """
    HTTP Client to securely call the .NET backend AI Execution Layer.
    """
    def __init__(self):
        self.base_url = settings.DOTNET_BACKEND_URL.rstrip("/")
        
    async def execute_tool(self, route: str, parameters: Dict[str, Any], auth_header: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calls the backend endpoint matching the tool route, forwarding the JWT token.
        """
        if not self.base_url:
            logger.warning("DOTNET_BACKEND_URL is not set.")
            return {"error": "Backend URL configuration missing"}
            
        url = f"{self.base_url}{route}"
        
        # Payload matching the AiExecutionRequest model defined in the .NET layer
        payload = {
            "parameters": parameters,
            "userId": user_id
        }
        
        headers = {
            "Authorization": auth_header
        }
        
        logger.info(f"Calling backend tool at {url} with parameters: {parameters}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=30.0)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Backend returned HTTP error: {e.response.status_code} - {e.response.text}")
            return {"error": f"Backend error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Error calling backend tool: {e}")
            return {"error": str(e)}

# Singleton instance
backend_client = BackendClient()
