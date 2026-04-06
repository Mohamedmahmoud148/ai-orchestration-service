import json
from typing import Optional, Any
from app.core.logging import logger


class ModelRouter:
    """
    Abstracts LLM selection away from agents and modules.
    Supports Gemini, OpenAI, Anthropic (cloud), and HuggingFace (local) models.

    Local models are addressed with the 'hf/' prefix, e.g. 'hf/TinyLlama'.
    Any other model_id is dispatched to the appropriate cloud client.
    """
    def __init__(
        self,
        gemini_client: Optional[Any] = None,
        openai_client: Optional[Any] = None,
        anthropic_client: Optional[Any] = None,
        local_model_service: Optional[Any] = None,
    ):
        self.gemini_client = gemini_client
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.local_model_service = local_model_service

    async def generate_structured_json(
        self, 
        prompt: str, 
        system_instruction: str, 
        model_id: str = "gpt-4o-mini"
    ) -> dict | None:
        """Centralized method for prompting language models for strict JSON outputs."""
        logger.debug(f"Routing request to model: {model_id}")
        
        try:
            # ── HuggingFace (local) ──────────────────────────────────────────
            if model_id.startswith("hf/") and self.local_model_service:
                return await self.local_model_service.generate_structured_json(
                    prompt=prompt,
                    system_instruction=system_instruction,
                )

            # ── Gemini ──────────────────────────────────────────────────────
            elif "gemini" in model_id.lower() and self.gemini_client:
                from google.genai import types
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
                    
            elif ("gpt" in model_id.lower() or "o1" in model_id.lower() or "o3" in model_id.lower()) and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                if content:
                    return json.loads(content)
                    
            elif "claude" in model_id.lower() and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    system=system_instruction + "\nRespond ONLY with valid JSON.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.content[0].text
                if content:
                    return json.loads(content)
            else:
                logger.error(f"No configured client found for model_id: {model_id}")
                return None
                
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from {model_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in ModelRouter generation for {model_id}: {e}")
            return None
            
    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        model_id: str = "gpt-4o-mini"
    ) -> str | None:
        """Standard text generation — dispatches to cloud or local HuggingFace model."""
        logger.debug(f"Routing text generation to model: {model_id}")
        try:
            # ── HuggingFace (local) ────────────────────────────────────────
            if model_id.startswith("hf/") and self.local_model_service:
                return await self.local_model_service.generate_text(
                    prompt=prompt,
                    system_instruction=system_instruction,
                )

            # ── Gemini ─────────────────────────────────────────────────
            elif "gemini" in model_id.lower() and self.gemini_client:
                from google.genai import types
                response = await self.gemini_client.aio.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                    ),
                )
                return response.text
                
            elif ("gpt" in model_id.lower() or "o1" in model_id.lower() or "o3" in model_id.lower()) and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
                
            elif "claude" in model_id.lower() and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    system=system_instruction,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
                
            else:
                logger.error(f"No configured client found for model_id: {model_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error in ModelRouter text generation for {model_id}: {e}")
            return None

    async def summarize(
        self,
        text: str,
        model_id: Optional[str] = None,
        max_length: int = 300,
    ) -> str:
        """
        Summarise text using the provided model_id or a task-optimized default.
        """
        target_model = model_id or "hf/facebook/bart-large-cnn"
        logger.info("ModelRouter.summarize: using model=%s", target_model)

        if target_model.startswith("hf/") and self.local_model_service:
            return await self.local_model_service.summarize(text, max_length=max_length)

        # Build a summarization prompt for cloud models
        prompt = (
            "Provide a concise, bullet-point summary of the following text. "
            "Focus on the key points and ignore minor details.\n\n"
            f"TEXT:\n{text[:4000]}"
        )
        system_instruction = "You are a professional summarization assistant."

        result = await self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            model_id=target_model,
        )
        return result or ""

    async def generate_questions(
        self,
        text: str,
        num_questions: int = 5,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Generate questions from text using the provided model_id or a task-optimized default.
        """
        target_model = model_id or "hf/google/flan-t5-base"
        logger.info("ModelRouter.generate_questions: using model=%s", target_model)

        if target_model.startswith("hf/") and self.local_model_service:
            # Map standard model names to specific local service methods if needed
            return await self.local_model_service.generate_questions(
                text, num_questions=num_questions
            )

        # Build a question generation prompt for cloud models
        prompt = (
            f"Based on the text below, generate {num_questions} clear and challenging "
            "university-level exam questions.\n\n"
            f"TEXT:\n{text[:4000]}"
        )
        system_instruction = (
            "You are an expert educator. Generate academic questions based strictly on the provided text. "
            "Return the questions as a simple list."
        )

        result = await self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            model_id=target_model,
        )
        return result or ""
