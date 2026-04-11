import json
from typing import List, Optional, Any
from app.core.logging import logger


class ModelRouter:
    """
    Abstracts LLM selection away from agents and modules.

    Primary backend : OpenAI (gpt-4o, gpt-4o-mini, gpt-3.5-turbo …)
    Optional extras : Gemini, Anthropic, HuggingFace (local).

    Key methods
    -----------
    generate(prompt, system_instruction, model_id)
        Single-turn text generation — convenience wrapper.
    generate_with_messages(messages, model_id)
        Multi-turn structured generation.  Accepts a list of
        ``{"role": …, "content": …}`` dicts (OpenAI chat format).
    generate_structured_json(prompt, system_instruction, model_id)
        Returns a parsed dict; enforces JSON response format.
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
            
    async def generate_with_messages(
        self,
        messages: List[dict],
        model_id: str = "gpt-4o-mini",
    ) -> str | None:
        """
        Multi-turn generation using a pre-built messages list.

        ``messages`` must follow the OpenAI chat format::

            [
                {"role": "system",    "content": "…"},
                {"role": "user",      "content": "…"},
                {"role": "assistant", "content": "…"},
                …
            ]

        This is the preferred method for the fallback LLM call because it
        preserves conversation history without lossy string concatenation.
        """
        logger.debug("ModelRouter.generate_with_messages: model=%s messages=%d", model_id, len(messages))
        try:
            if model_id.startswith("hf/") and self.local_model_service:
                # Local models receive the last user message only
                user_msg = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
                )
                sys_msg = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                return await self.local_model_service.generate_text(
                    prompt=user_msg, system_instruction=sys_msg
                )

            if ("gpt" in model_id.lower() or "o1" in model_id.lower() or "o3" in model_id.lower()) and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                )
                return response.choices[0].message.content

            if "gemini" in model_id.lower() and self.gemini_client:
                # Flatten to a single combined prompt for Gemini
                combined = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
                )
                sys_msg = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                from google.genai import types
                response = await self.gemini_client.aio.models.generate_content(
                    model=model_id,
                    contents=combined,
                    config=types.GenerateContentConfig(system_instruction=sys_msg),
                )
                return response.text

            if "claude" in model_id.lower() and self.anthropic_client:
                sys_msg = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                non_sys = [m for m in messages if m["role"] != "system"]
                response = await self.anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    system=sys_msg,
                    messages=non_sys,
                )
                return response.content[0].text

            logger.error("ModelRouter.generate_with_messages: no client for model_id=%s", model_id)
            return None

        except Exception as exc:
            logger.error("ModelRouter.generate_with_messages error: %s", exc, exc_info=True)
            return None

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        model_id: str = "gpt-4o-mini",
    ) -> str | None:
        """Single-turn text generation — thin wrapper around generate_with_messages."""
        messages: List[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_with_messages(messages, model_id=model_id)

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
