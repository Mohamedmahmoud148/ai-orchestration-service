"""
app/services/tool_registry.py  —  v2.0

Centralised registry mapping AI intents → internal module identifiers.

IMPORTANT: Every intent listed in `planner.VALID_INTENTS` that has a
dedicated module MUST be registered here.  Missing entries cause the
executor to fall back to `module_name = "model_only"`, which bypasses
the backend-first execution policy and answers from general LLM knowledge.

Module identifiers MUST match the keys in executor._MODULE_CLASS_MAP.
"""
from typing import Dict, Optional

from app.core.logging import logger


class ToolRegistry:
    """
    Centralised registry mapping AI intents → module identifier string.

    The module identifier is passed to PlanExecutor._get_module() which
    uses it to look up the concrete module class to instantiate and run.

    Intents NOT registered here (e.g. general_chat) fall through to
    the LLM fallback path — this is intentional for chat-only intents.
    """

    def __init__(self) -> None:
        # Keys MUST match planner.VALID_INTENTS exactly.
        # Values MUST match executor._MODULE_CLASS_MAP keys exactly.
        self._registry: Dict[str, str] = {
            # ── Original modules ────────────────────────────────────────────
            "generate_exam":        "exam_generation",
            "summarization":        "summarization",
            "file_extraction":      "file_extraction",
            "result_query":         "result_query",
            # ── New modules (critical: missing = model_only fallback) ────────
            "complaint_submit":     "complaint_submit",
            "complaint_summary":    "complaint_summary",
            "file_processing":      "file_processing",
            "cv_analysis":          "cv_analysis",
            "academic_advice":      "academic_advice",
            "material_explanation": "material_explanation",
            "backend_api_query":    "dynamic_api_module",
            # ── general_chat is intentionally NOT registered:
            #    it routes directly to LLM fallback (no backend call needed)
        }

    def get_module_for_intent(self, intent_name: str) -> Optional[str]:
        """
        Return the module identifier for the given intent, or None if the
        intent should be handled by the LLM fallback (e.g. general_chat).
        """
        module = self._registry.get(intent_name)
        if module:
            logger.debug(
                "ToolRegistry: intent=%r → module=%r", intent_name, module
            )
        else:
            logger.debug(
                "ToolRegistry: intent=%r → no module (LLM fallback path)", intent_name
            )
        return module

    def register_module(self, intent_name: str, module_name: str) -> None:
        """Dynamically register a new intent → module mapping at runtime."""
        logger.info(
            "ToolRegistry: registering intent=%r → module=%r", intent_name, module_name
        )
        self._registry[intent_name] = module_name

    def all_registered_intents(self) -> list[str]:
        """Return a list of all registered intent names (for diagnostics)."""
        return list(self._registry.keys())


# Singleton instance — imported by agent.py and main.py
tool_registry = ToolRegistry()
