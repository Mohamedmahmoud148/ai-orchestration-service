"""
app/services/model_service.py

Lazy-loaded HuggingFace inference service.

Three pipelines, created once on first use:
  text-generation       -- TinyLlama/TinyLlama-1.1B-Chat-v1.0
  summarization         -- facebook/bart-large-cnn
  text2text-generation  -- google/flan-t5-base

Public API
----------
generate_text(prompt, system_instruction, max_new_tokens) -> str
summarize(text, max_length)                               -> str
generate_questions(text, num_questions)                   -> str
"""
from __future__ import annotations
import asyncio
import logging

logger = logging.getLogger(__name__)

_TEXT_GEN_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_SUMMARIZE_MODEL = "facebook/bart-large-cnn"
_QUESTION_MODEL  = "google/flan-t5-base"


class ModelService:
    """Manages three HuggingFace pipelines with lazy initialisation."""

    def __init__(self) -> None:
        self._text_gen_pipe  = None
        self._summarize_pipe = None
        self._question_pipe  = None
        self._lock           = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    async def _get_text_gen(self):
        if self._text_gen_pipe is None:
            async with self._lock:
                if self._text_gen_pipe is None:
                    self._text_gen_pipe = await self._load(
                        "text-generation", _TEXT_GEN_MODEL
                    )
        return self._text_gen_pipe

    async def _get_summarize(self):
        if self._summarize_pipe is None:
            async with self._lock:
                if self._summarize_pipe is None:
                    self._summarize_pipe = await self._load(
                        "summarization", _SUMMARIZE_MODEL
                    )
        return self._summarize_pipe

    async def _get_question(self):
        if self._question_pipe is None:
            async with self._lock:
                if self._question_pipe is None:
                    self._question_pipe = await self._load(
                        "text2text-generation", _QUESTION_MODEL
                    )
        return self._question_pipe

    @staticmethod
    async def _load(task: str, model: str):
        """Download and build a pipeline inside a thread-pool worker."""
        def _create():
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            logger.info(
                "ModelService: loading task=%s model=%s device=%s",
                task, model, device,
            )
            return pipeline(task, model=model, device=device)

        loop = asyncio.get_event_loop()
        pipe = await loop.run_in_executor(None, _create)
        logger.info("ModelService: '%s' ready.", model)
        return pipe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_prompt(self, prompt: str, system_instruction: str) -> str:
        """Build a plain-text prompt. Uses tokenizer.apply_chat_template when available."""
        if not system_instruction:
            return prompt
        # Try the tokenizer's own chat template first
        return (
            "### System:\n"
            + system_instruction
            + "\n\n### User:\n"
            + prompt
            + "\n\n### Assistant:\n"
        )

    async def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        max_new_tokens: int = 512,
    ) -> str:
        """Generate free-form text with TinyLlama."""
        try:
            pipe        = await self._get_text_gen()
            full_prompt = self._build_prompt(prompt, system_instruction)

            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: pipe(
                    full_prompt,
                    max_new_tokens=max_new_tokens,
                    truncation=True,
                    return_full_text=False,
                ),
            )
            return result[0].get("generated_text", "").strip() if result else ""
        except Exception as exc:
            logger.error("ModelService.generate_text error: %s", exc)
            return ""

    async def summarize(self, text: str, max_length: int = 300) -> str:
        """Summarise text with BART-large-CNN."""
        if not text or not text.strip():
            return "No content to summarize."
        try:
            pipe      = await self._get_summarize()
            truncated = text[:3800]  # BART has a 1024-token limit

            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: pipe(
                    truncated,
                    max_length=max_length,
                    min_length=60,
                    do_sample=False,
                ),
            )
            return result[0].get("summary_text", "").strip() if result else text[:300]
        except Exception as exc:
            logger.error("ModelService.summarize error: %s", exc)
            return text[:300]

    async def generate_questions(self, text: str, num_questions: int = 5) -> str:
        """Generate exam questions with Flan-T5."""
        if not text or not text.strip():
            return "No content provided for question generation."
        try:
            pipe   = await self._get_question()
            prompt = (
                f"Generate {num_questions} university exam questions "
                f"from this text:\n\n{text[:1500]}"
            )

            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: pipe(prompt, max_length=512, do_sample=False),
            )
            return result[0].get("generated_text", "").strip() if result else ""
        except Exception as exc:
            logger.error("ModelService.generate_questions error: %s", exc)
            return ""

    async def generate_structured_json(
        self,
        prompt: str,
        system_instruction: str = "",
    ) -> dict | None:
        """Compatibility shim used by model_router for hf/ prefixed models."""
        import json
        raw = await self.generate_text(
            prompt,
            system_instruction=system_instruction,
            max_new_tokens=1024,
        )
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except Exception:
            return None


# Singleton — lazily loads models on first call
local_model_service = ModelService()
