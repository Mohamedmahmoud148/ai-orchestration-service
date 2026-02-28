import json
from google import genai
from google.genai import types
from app.core.logging import logger

class ModelRouter:
    """
    Abstracts LLM selection away from the agents, allowing future
    integration of OpenAI, Claude, or local specific fine-tunes.
    """
    def __init__(self, api_key: str | None = None):
        """Pass the API key or a pre-configured client via DI"""
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None
        
    async def generate_structured_json(self, prompt: str, system_instruction: str, model_id: str = "gemini-2.5-flash") -> dict | None:
        """Centralized method for prompting language models for strict JSON outputs."""
        if not self.gemini_client:
            logger.error("No valid Gemini configuration found in ModelRouter.")
            return None
            
        try:
            logger.debug(f"Routing request to model: {model_id}")
            response = await self.gemini_client.aio.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                ),
            )
            
            if response.text:
                return json.loads(response.text)
                
            return None
        except Exception as e:
            logger.error(f"Error in ModelRouter generation: {e}")
            return None

