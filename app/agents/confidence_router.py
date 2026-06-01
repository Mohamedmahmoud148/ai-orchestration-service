"""
app/agents/confidence_router.py  —  Layer 3: Confidence-Based Router

Takes the combined output of the embedding classifier (Layer 1) and the
LLM function-call classifier (Layer 2) and decides what to do:

  action = "execute"   → confidence is high enough, proceed to execution
  action = "clarify"   → confidence is in the grey zone, ask the user
  action = "fallback"  → too uncertain, treat as general_chat

Default thresholds (tune based on production logs):

  Source        EXECUTE     CLARIFY range    FALLBACK
  ─────────────────────────────────────────────────────
  embedding     ≥ 0.82      0.60–0.82        < 0.60
  llm           ≥ 0.78      0.55–0.78        < 0.55

Safe intents (general_chat, academic_advice, study_plan) skip the
clarification gate because degrading to a general answer is harmless.

Critical intents (action_execute, complaint_submit) apply a HIGHER
execute threshold to reduce false-positive write operations.  Their
confirmed execution path goes through the ActionGuard (Layer 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Intent risk tiers ─────────────────────────────────────────────────────────

# Always safe to execute even with low confidence (answer quality degrades,
# but no irreversible action is taken).
_SAFE_LOW_CONFIDENCE: frozenset[str] = frozenset({
    "general_chat",
    "academic_advice",
    "study_plan",
    "material_explanation",
    "result_query",
    "regulation",
})

# Write operations — require higher confidence before executing AND must
# pass through ActionGuard for user confirmation.
_CRITICAL_WRITE_INTENTS: frozenset[str] = frozenset({
    "action_execute",
    "complaint_submit",
    "file_processing",
})

# Default clarification questions per intent (Arabic first, English fallback)
_DEFAULT_CLARIFICATIONS: dict[str, str] = {
    "generate_exam":      "هتعمل امتحان لأنهي مادة؟ وكام سؤال تقريباً؟",
    "action_execute":     "عايزني أعمل ايه بالظبط؟ تسجيل في مواد؟",
    "complaint_submit":   "الشكوى عن ايه؟ دكتور، امتحان، ولا درجة؟",
    "material_explanation": "تريد شرح أنهي مادة أو موضوع بالضبط؟",
    "material_qa":        "سؤالك ده من أي محاضرة أو مادة؟",
    "regulation":         "بتسأل عن ايه في اللائحة؟ مواد سنة معينة ولا متطلبات التخرج؟",
    "backend_api_query":  "بتسأل عن بياناتك إنت ولا بيانات السيستم؟",
    "assignment_query":   "بتسأل عن أنهي واجب بالظبط؟",
    "summarization":      "عايز ألخص أنهي نص أو مستند؟",
    "file_extraction":    "ممكن ترفعلي الملف اللي عايز أستخرج منه؟",
    "cv_analysis":        "ممكن ترفعلي الـ CV بتاعك؟",
}


@dataclass(frozen=True)
class RoutingDecision:
    """
    Output of the ConfidenceRouter.

    action                — "execute" | "clarify" | "fallback"
    intent                — the (possibly adjusted) intent to execute
    confidence            — the confidence score that drove this decision
    source                — "embedding" | "llm" | "pronoun" | "keyword"
    clarification_question — question to show the user (action=="clarify" only)
    goal_summary          — human-readable description of what we think the user wants
    """
    action: str
    intent: str
    confidence: float
    source: str
    goal_summary: str
    clarification_question: Optional[str] = None


class ConfidenceRouter:
    """
    Stateless routing logic.  All thresholds are passed in at construction
    time so they can be driven by settings without coupling this module to
    the config layer.
    """

    def __init__(
        self,
        embedding_execute_threshold: float = 0.82,
        llm_execute_threshold:       float = 0.78,
        llm_clarify_threshold:       float = 0.55,
        critical_execute_threshold:  float = 0.88,
    ) -> None:
        self._emb_exec    = embedding_execute_threshold
        self._llm_exec    = llm_execute_threshold
        self._llm_clarify = llm_clarify_threshold
        self._crit_exec   = critical_execute_threshold

    def route(
        self,
        intent: str,
        confidence: float,
        source: str,
        goal_summary: str,
        clarification_question: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Decide what to do with a classified intent.

        Parameters:
          intent       — classified intent label
          confidence   — numeric score (0.0–1.0)
          source       — "embedding" | "llm" | "pronoun" | "keyword"
          goal_summary — one-line description of the user's goal
          clarification_question — suggested question (LLM may provide one)
        """
        # ── Pronoun / keyword overrides always execute ────────────────────────
        if source in ("pronoun", "keyword"):
            return RoutingDecision(
                action="execute",
                intent=intent,
                confidence=confidence,
                source=source,
                goal_summary=goal_summary,
            )

        # ── Safe intents: always execute regardless of confidence ────────────
        if intent in _SAFE_LOW_CONFIDENCE:
            return RoutingDecision(
                action="execute",
                intent=intent,
                confidence=confidence,
                source=source,
                goal_summary=goal_summary,
            )

        # ── Critical write intents: higher execute threshold ─────────────────
        if intent in _CRITICAL_WRITE_INTENTS:
            exec_threshold = self._crit_exec
        elif source == "embedding":
            exec_threshold = self._emb_exec
        else:
            exec_threshold = self._llm_exec

        if confidence >= exec_threshold:
            return RoutingDecision(
                action="execute",
                intent=intent,
                confidence=confidence,
                source=source,
                goal_summary=goal_summary,
            )

        # ── Clarification zone ───────────────────────────────────────────────
        clarify_threshold = (
            self._llm_clarify if source == "llm"
            else self._llm_clarify  # same floor for embedding
        )

        if confidence >= clarify_threshold:
            q = (
                clarification_question
                or _DEFAULT_CLARIFICATIONS.get(intent)
                or "ممكن توضح أكتر عايز ايه بالظبط؟"
            )
            return RoutingDecision(
                action="clarify",
                intent=intent,
                confidence=confidence,
                source=source,
                goal_summary=goal_summary,
                clarification_question=q,
            )

        # ── Fallback ─────────────────────────────────────────────────────────
        return RoutingDecision(
            action="fallback",
            intent="general_chat",
            confidence=confidence,
            source=source,
            goal_summary=goal_summary,
        )

    def is_critical(self, intent: str) -> bool:
        """True when the intent is a write operation requiring ActionGuard."""
        return intent in _CRITICAL_WRITE_INTENTS
