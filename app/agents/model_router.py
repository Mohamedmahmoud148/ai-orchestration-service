"""
model_router.py

Routes LLM requests through OpenRouter (OpenAI-compatible API).

Primary model  : openai/gpt-4o-mini   (cheap, fast, smart enough for planning)
Fallback chain :
  1. settings.OPENROUTER_FALLBACK_MODEL_1  (default: openai/gpt-4o-mini)
  2. settings.OPENROUTER_FALLBACK_MODEL_2  (default: empty / skip)

OpenRouter accepts ANY model slug from https://openrouter.ai/models.
Gemini/Anthropic/HuggingFace local paths are still supported alongside it.
"""

import json
from typing import List, Optional, Any

from app.core.logging import logger


# ── Helper ────────────────────────────────────────────────────────────────────

def _is_openrouter_model(model_id: str) -> bool:
    """
    Return True for model IDs that should be sent to OpenRouter.

    OpenRouter accepts:
      - provider/model  slugs  e.g. "openai/gpt-4o-mini", "mistralai/mistral-7b-instruct"
      - legacy bare names      e.g. "gpt-4o-mini", "gpt-3.5-turbo"
    """
    if "/" in model_id:
        return True
    lower = model_id.lower()
    return any(
        lower.startswith(p)
        for p in ("gpt-", "o1-", "o3-", "o4-")
    )


class ModelRouter:
    """
    Abstracts LLM selection away from agents and modules.

    Primary backend : OpenRouter (all cloud models via a single endpoint)
    Optional extras : Gemini direct, Anthropic direct, HuggingFace local.

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

    # Default model served through OpenRouter
    DEFAULT_MODEL = "openai/gpt-4o-mini"

    # ── Task → model tier mapping (Phase 13) ─────────────────────────────────
    # Complex tasks route to MODEL_COMPLEX (e.g. gpt-4o), vision to MODEL_VISION,
    # everything else to MODEL_SIMPLE. Opt-in: callers pass the resolved model_id.
    _COMPLEX_TASKS = frozenset({
        "academic_advisor", "pdf_explanation", "material_explanation",
        "exam_generation", "study_plan", "regulation_analysis",
        "doctor_intelligence", "deep_pdf",
    })
    _VISION_TASKS = frozenset({"vision", "ocr", "scanned_pdf", "image"})

    @staticmethod
    def pick_model(task: str) -> str:
        """
        Resolve a task name to the configured model tier.

        Simple tasks  → MODEL_SIMPLE  (cheap, fast — classification, titles)
        Complex tasks → MODEL_COMPLEX (smart — advisor, PDF, exam gen)
        Vision tasks  → MODEL_VISION  (OCR, images)
        Unknown       → MODEL_SIMPLE
        """
        try:
            from app.core.config import settings
            t = (task or "").lower()
            if t in ModelRouter._VISION_TASKS:
                return settings.MODEL_VISION
            if t in ModelRouter._COMPLEX_TASKS:
                return settings.MODEL_COMPLEX
            return settings.MODEL_SIMPLE
        except Exception:
            return ModelRouter.DEFAULT_MODEL

    def __init__(
        self,
        gemini_client: Optional[Any] = None,
        openai_client: Optional[Any] = None,       # ← now the OpenRouter client
        anthropic_client: Optional[Any] = None,
        local_model_service: Optional[Any] = None,
    ):
        self.gemini_client = gemini_client
        self.openai_client = openai_client          # OpenRouter / OpenAI-compat
        self.anthropic_client = anthropic_client
        self.local_model_service = local_model_service

        # Pull fallback chain from settings (lazy import to avoid circular deps)
        try:
            from app.core.config import settings
            self._fallback_1: str = settings.OPENROUTER_FALLBACK_MODEL_1 or "openai/gpt-4o-mini"
            self._fallback_2: str = settings.OPENROUTER_FALLBACK_MODEL_2 or ""
        except Exception:
            self._fallback_1 = "openai/gpt-4o-mini"
            self._fallback_2 = ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Core internal: single OpenRouter call
    # ─────────────────────────────────────────────────────────────────────────

    async def _openrouter_json(
        self,
        model_id: str,
        messages: List[dict],
        max_tokens: Optional[int] = None,
    ) -> dict | None:
        """Call OpenRouter for a strict JSON response."""
        kwargs: dict = {
            "model": model_id,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        response = await self.openai_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content:
            return json.loads(content)
        return None

    async def _openrouter_text(
        self,
        model_id: str,
        messages: List[dict],
        max_tokens: Optional[int] = None,
    ) -> str | None:
        """Call OpenRouter for a plain-text response."""
        kwargs: dict = {"model": model_id, "messages": messages}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        response = await self.openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def stream_with_messages(
        self,
        messages: List[dict],
        model_id: str = "openai/gpt-4o-mini",
    ):
        """
        Stream tokens from OpenRouter as an async generator yielding text chunks.

        Falls back to the configured fallback chain on failure. Used by the SSE
        chat endpoint so the user sees the response materialise live.
        """
        if not self.openai_client:
            yield ""
            return

        models_to_try = [model_id]
        if self._fallback_1 and self._fallback_1 != model_id:
            models_to_try.append(self._fallback_1)
        if self._fallback_2 and self._fallback_2 not in models_to_try:
            models_to_try.append(self._fallback_2)

        last_exc: Optional[Exception] = None
        for mdl in models_to_try:
            try:
                stream = await self.openai_client.chat.completions.create(
                    model=mdl,
                    messages=messages,
                    stream=True,
                )
                async for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta.content
                    except (AttributeError, IndexError):
                        delta = None
                    if delta:
                        yield delta
                return
            except Exception as exc:
                last_exc = exc
                logger.warning("ModelRouter.stream: %s failed (%s) — trying next", mdl, exc)
                continue

        logger.error("ModelRouter.stream: all models exhausted — %s", last_exc)

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: generate_structured_json
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_structured_json(
        self,
        prompt: str,
        system_instruction: str,
        model_id: str = DEFAULT_MODEL,
    ) -> dict | None:
        """Centralised method for prompting LLMs for strict JSON outputs."""
        logger.debug("ModelRouter.generate_structured_json: model=%s", model_id)

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user",   "content": prompt},
        ]

        try:
            # ── HuggingFace (local) ─────────────────────────────────────────
            if model_id.startswith("hf/") and self.local_model_service:
                return await self.local_model_service.generate_structured_json(
                    prompt=prompt,
                    system_instruction=system_instruction,
                )

            # ── Gemini direct ───────────────────────────────────────────────
            if "gemini" in model_id.lower() and self.gemini_client:
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
                return None

            # ── Anthropic direct ────────────────────────────────────────────
            if "claude" in model_id.lower() and self.anthropic_client and not self.openai_client:
                response = await self.anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    system=system_instruction + "\nRespond ONLY with valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
                return json.loads(content) if content else None

            # ── OpenRouter (default) ────────────────────────────────────────
            if self.openai_client:
                result = await self._call_with_fallback_json(model_id, messages)
                return result

            logger.error("ModelRouter: no client configured for model_id=%s", model_id)
            return None

        except json.JSONDecodeError as exc:
            logger.error("ModelRouter: JSON parse error for %s — %s", model_id, exc)
            return None
        except Exception as exc:
            logger.error("ModelRouter: unexpected error for %s — %s", model_id, exc, exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: generate_with_messages
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_with_messages(
        self,
        messages: List[dict],
        model_id: str = DEFAULT_MODEL,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
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

        If ``response_format={"type": "json_object"}`` is passed, returns raw
        JSON string (suitable for json.loads by the caller).
        """
        logger.debug(
            "ModelRouter.generate_with_messages: model=%s messages=%d json_mode=%s",
            model_id, len(messages), response_format is not None,
        )
        try:
            # ── HuggingFace local ───────────────────────────────────────────
            if model_id.startswith("hf/") and self.local_model_service:
                user_msg = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
                )
                sys_msg = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                return await self.local_model_service.generate_text(
                    prompt=user_msg, system_instruction=sys_msg
                )

            # ── Gemini direct ───────────────────────────────────────────────
            if "gemini" in model_id.lower() and self.gemini_client:
                combined = "\n".join(
                    f"{m['role'].upper()}: {m['content']}"
                    for m in messages if m["role"] != "system"
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

            # ── Anthropic direct (only when no openai_client) ───────────────
            if "claude" in model_id.lower() and self.anthropic_client and not self.openai_client:
                sys_msg = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                non_sys = [m for m in messages if m["role"] != "system"]
                response = await self.anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens or 4096,
                    system=sys_msg,
                    messages=non_sys,
                )
                return response.content[0].text

            # ── OpenRouter (default — handles gpt-*, openai/*, mistralai/*, etc.) ──
            if self.openai_client:
                if response_format and response_format.get("type") == "json_object":
                    # JSON mode: call openrouter_json and return as string
                    result = await self._call_with_fallback_json(
                        model_id, messages, max_tokens=max_tokens
                    )
                    import json as _json
                    return _json.dumps(result, ensure_ascii=False) if result else None
                return await self._call_with_fallback_text(
                    model_id, messages, max_tokens=max_tokens
                )

            logger.error(
                "ModelRouter.generate_with_messages: no client for model_id=%s", model_id
            )
            return None

        except Exception as exc:
            logger.error(
                "ModelRouter.generate_with_messages error: %s", exc, exc_info=True
            )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: generate  (single-turn convenience wrapper)
    # ─────────────────────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        model_id: str = DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
    ) -> str | None:
        """Single-turn text generation — thin wrapper around generate_with_messages."""
        messages: List[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_with_messages(
            messages, model_id=model_id, max_tokens=max_tokens
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: summarize
    # ─────────────────────────────────────────────────────────────────────────

    async def summarize(
        self,
        text: str,
        model_id: Optional[str] = None,
        max_length: int = 300,
    ) -> str:
        """Summarise text using the provided model_id or a task-optimised default."""
        target_model = model_id or "hf/facebook/bart-large-cnn"
        logger.info("ModelRouter.summarize: using model=%s", target_model)

        if target_model.startswith("hf/") and self.local_model_service:
            return await self.local_model_service.summarize(text, max_length=max_length)

        prompt = (
            "Provide a concise, bullet-point summary of the following text. "
            "Focus on the key points and ignore minor details.\n\n"
            f"TEXT:\n{text[:4000]}"
        )
        result = await self.generate(
            prompt=prompt,
            system_instruction="You are a professional summarization assistant.",
            model_id=target_model,
        )
        return result or ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Public: generate_questions
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_questions(
        self,
        text: str,
        num_questions: int = 5,
        model_id: Optional[str] = None,
    ) -> str:
        """Generate exam questions from text using the provided model."""
        target_model = model_id or "hf/google/flan-t5-base"
        logger.info("ModelRouter.generate_questions: using model=%s", target_model)

        if target_model.startswith("hf/") and self.local_model_service:
            return await self.local_model_service.generate_questions(
                text, num_questions=num_questions
            )

        prompt = (
            f"Based on the text below, generate {num_questions} clear and challenging "
            "university-level exam questions.\n\n"
            f"TEXT:\n{text[:4000]}"
        )
        result = await self.generate(
            prompt=prompt,
            system_instruction=(
                "You are an expert educator. Generate academic questions based strictly "
                "on the provided text. Return the questions as a simple list."
            ),
            model_id=target_model,
        )
        return result or ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Private: fallback chains
    # ─────────────────────────────────────────────────────────────────────────

    async def _call_with_fallback_json(
        self,
        primary: str,
        messages: List[dict],
        max_tokens: Optional[int] = None,
    ) -> dict | None:
        """
        Try primary model → fallback_1 → fallback_2 for JSON responses.
        Returns the first successful non-None result.
        """
        candidates = self._build_fallback_chain(primary)
        for model in candidates:
            try:
                logger.debug("OpenRouter JSON attempt: model=%s", model)
                result = await self._openrouter_json(model, messages, max_tokens=max_tokens)
                if result is not None:
                    return result
                logger.warning("OpenRouter: empty JSON from %s — trying next", model)
            except Exception as exc:
                logger.warning("OpenRouter: %s failed (%s) — trying next", model, exc)
        logger.error("OpenRouter: all fallbacks exhausted for JSON call")
        return None

    async def _call_with_fallback_text(
        self,
        primary: str,
        messages: List[dict],
        max_tokens: Optional[int] = None,
    ) -> str | None:
        """
        Try primary model → fallback_1 → fallback_2 for text responses.
        Returns the first successful non-None result.
        """
        candidates = self._build_fallback_chain(primary)
        for model in candidates:
            try:
                logger.debug("OpenRouter text attempt: model=%s", model)
                result = await self._openrouter_text(model, messages, max_tokens=max_tokens)
                if result is not None:
                    return result
                logger.warning("OpenRouter: empty text from %s — trying next", model)
            except Exception as exc:
                logger.warning("OpenRouter: %s failed (%s) — trying next", model, exc)
        logger.error("OpenRouter: all fallbacks exhausted for text call")
        return None

    def _build_fallback_chain(self, primary: str) -> List[str]:
        """Return ordered list of unique models to try."""
        chain = [primary]
        if self._fallback_1 and self._fallback_1 != primary:
            chain.append(self._fallback_1)
        if self._fallback_2 and self._fallback_2 not in chain:
            chain.append(self._fallback_2)
        return chain
