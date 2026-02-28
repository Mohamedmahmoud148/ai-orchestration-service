import json
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.logging import logger
from app.models.chat import AiIntent

class AiService:
    def __init__(self):
        # Initialize Gemini Client if API key is provided
        if settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = "gemini-2.5-flash"
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY is not set. AiService will not function correctly.")

    async def determine_intent(self, message: str) -> AiIntent:
        if not self.client:
            logger.error("Attempted to call Gemini without API key.")
            return AiIntent(intent_name="Error", parameters={"error": "Gemini client not initialized. Check GEMINI_API_KEY."})

        system_instruction = (
            "You are an AI assistant routing user intents to backend tools.\n"
            "Identify the user's intent from their message and extract any relevant parameters.\n"
            "Available intents: ScheduleMeeting, GetUserStatus, DefaultChat.\n"
            "Format the output strictly as JSON matching this schema:\n"
            '{"intent_name": "NameOfIntent", "parameters": {"key1": "value1", ...}}\n'
            "If the intent is just general chat, use 'DefaultChat' and return parameters as empty."
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                ),
            )
            
            if response.text:
                data = json.loads(response.text)
                return AiIntent(
                    intent_name=data.get("intent_name", "DefaultChat"),
                    parameters=data.get("parameters", {})
                )
            
            return AiIntent(intent_name="DefaultChat", parameters={})
            
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return AiIntent(intent_name="Error", parameters={"error": str(e)})

# Singleton instance
ai_service = AiService()
