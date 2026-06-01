"""
app/agents/action_guard.py  —  Layer 4: Critical Action Guard

Intercepts write operations (enrollment, complaint submission, bulk uploads)
before they reach the Executor.

Two-step flow:
  1. Guard detects a critical intent and builds a human-readable preview.
  2. Returns status="confirmation_required" with a confirmation message.
  3. The chat endpoint returns this to the user.
  4. The user replies "نعم" / "yes" / "تأكيد" / "confirm".
  5. MemoryStore.get_pending_action() retrieves the saved action.
  6. Agent resumes execution with the confirmed plan.

The guard uses a fast second LLM call to independently verify
that the classified intent is correct.  This costs ~100ms but
prevents accidental enrollment from an ambiguous message like
"سجل في كل حاجة" misclassified from "my schedule".

Stored in Redis: user:{id}:pending_action  TTL: 5 min  (MemoryStore already has this)

Confirmation keywords (accept):
  Arabic:  نعم، أيوه، تمام، موافق، اتفضل، نعم تأكيد
  English: yes, confirm, ok, go ahead, proceed, do it

Cancellation keywords:
  Arabic:  لأ، لا، الغي، مش عايز، cancel
  English: no, cancel, stop, never mind, abort
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from app.core.logging import logger

if TYPE_CHECKING:
    from app.agents.model_router import ModelRouter

# ── Intent risk tiers ─────────────────────────────────────────────────────────

CRITICAL_INTENTS: frozenset[str] = frozenset({
    "action_execute",    # enrollment, admin create operations
    "complaint_submit",  # irreversible complaint record
    "file_processing",   # bulk grade / student upload
})

# Intents that need a preview but NOT a LLM verification step
# (they are structurally obvious enough to skip the extra call)
_SKIP_VERIFICATION: frozenset[str] = frozenset({
    "complaint_submit",
    "file_processing",
})

# ── Confirmation / cancellation keyword sets ──────────────────────────────────

_CONFIRM_KEYWORDS: frozenset[str] = frozenset({
    # Arabic
    "نعم", "أيوه", "ايوه", "تمام", "موافق", "اتفضل", "تأكيد",
    "اكمل", "أكمل", "نعم تأكيد", "ماشي", "أوك", "اوك",
    # English
    "yes", "confirm", "ok", "okay", "go ahead", "proceed", "do it",
    "sure", "absolutely", "affirmative", "approve",
})

_CANCEL_KEYWORDS: frozenset[str] = frozenset({
    # Arabic
    "لأ", "لا", "الغي", "الغ", "مش عايز", "مش عايزه", "cancel", "وقف",
    "توقف", "مش كده", "غلط",
    # English
    "no", "cancel", "stop", "never mind", "nevermind", "abort", "nope",
    "negative", "don't", "dont",
})


def is_confirmation(message: str) -> bool:
    """Return True if the user's message is a confirmation response."""
    msg = message.strip().lower()
    return any(kw in msg for kw in _CONFIRM_KEYWORDS)


def is_cancellation(message: str) -> bool:
    """Return True if the user's message is a cancellation response."""
    msg = message.strip().lower()
    return any(kw in msg for kw in _CANCEL_KEYWORDS)


# ── Guard result ──────────────────────────────────────────────────────────────

class ActionGuardResult:
    __slots__ = ("should_confirm", "confirmation_message", "verified")

    def __init__(
        self,
        should_confirm: bool,
        confirmation_message: str = "",
        verified: bool = True,
    ) -> None:
        self.should_confirm = should_confirm
        self.confirmation_message = confirmation_message
        self.verified = verified


# ── Main guard function ───────────────────────────────────────────────────────

async def check_critical_action(
    intent: str,
    goal_summary: str,
    extracted_params: dict[str, Any],
    user_language: str,
    model_router: Optional["ModelRouter"] = None,
    run_verification: bool = True,
) -> ActionGuardResult:
    """
    Check whether a critical intent needs user confirmation.

    Returns ActionGuardResult:
      - should_confirm=False → proceed directly (intent verified + safe)
      - should_confirm=True  → show confirmation_message to user and pause

    The LLM verification step (run_verification=True) is skipped for
    intents in _SKIP_VERIFICATION or when model_router is None.
    """
    if intent not in CRITICAL_INTENTS:
        return ActionGuardResult(should_confirm=False)

    # ── Optional LLM verification ─────────────────────────────────────────
    verified = True
    if (
        run_verification
        and intent not in _SKIP_VERIFICATION
        and model_router is not None
    ):
        verified = await _verify_intent(
            intent=intent,
            goal_summary=goal_summary,
            params=extracted_params,
            model_router=model_router,
        )

    if not verified:
        # Verification rejected the intent — don't execute, don't confirm
        # (silently fallback; the caller will treat this as general_chat)
        logger.warning(
            "ActionGuard: LLM verification REJECTED intent=%r goal=%r",
            intent, goal_summary,
        )
        return ActionGuardResult(should_confirm=False, verified=False)

    # ── Build confirmation message ─────────────────────────────────────────
    msg = _build_confirmation_message(
        intent=intent,
        goal_summary=goal_summary,
        params=extracted_params,
        lang=user_language,
    )

    logger.info(
        "ActionGuard: confirmation required for intent=%r user_lang=%s",
        intent, user_language,
    )

    return ActionGuardResult(
        should_confirm=True,
        confirmation_message=msg,
        verified=True,
    )


async def _verify_intent(
    intent: str,
    goal_summary: str,
    params: dict[str, Any],
    model_router: "ModelRouter",
) -> bool:
    """
    Run a lightweight LLM call to confirm that the classification is correct.

    Returns True (proceed) or False (reject / treat as general_chat).
    On LLM error, defaults to True (proceed) — fail open for non-destructive
    actions.  For destructive actions, the confirmation step still acts
    as a second safety net.
    """
    import json

    prompt = (
        f"A user request was classified as intent='{intent}'.\n"
        f"Goal: {goal_summary}\n"
        f"Extracted params: {json.dumps(params, ensure_ascii=False)}\n\n"
        "Is this classification correct? Does the user genuinely want to "
        "perform this action (not just query information about it)?\n"
        'Answer with valid JSON only: {"confirmed": true} or {"confirmed": false, "reason": "..."}'
    )

    try:
        raw = await model_router.generate_structured_json(
            prompt=prompt,
            system_instruction=(
                "You are a classification verifier for a university AI system. "
                "Only confirm if you are highly confident the intent is correct. "
                "Return JSON only."
            ),
        )
        if isinstance(raw, dict):
            return bool(raw.get("confirmed", True))
        return True
    except Exception as exc:
        logger.warning("ActionGuard._verify_intent error (defaulting to proceed): %s", exc)
        return True


def _build_confirmation_message(
    intent: str,
    goal_summary: str,
    params: dict[str, Any],
    lang: str,
) -> str:
    """Build a bilingual confirmation prompt shown to the user."""

    # Intent-specific details
    detail_lines: list[str] = []
    if intent == "action_execute":
        if params.get("studentId") or params.get("userId"):
            detail_lines.append(f"• الطالب: {params.get('studentId') or params.get('userId')}")
        if params.get("batchId"):
            detail_lines.append(f"• الدفعة: {params.get('batchId')}")
        detail_en = "Auto-enroll in all available subjects for this semester."
        detail_ar = "تسجيل تلقائي في كل المواد المتاحة للترم الحالي."
    elif intent == "complaint_submit":
        target = params.get("targetType", "Other")
        detail_en = f"Submit a complaint — Target: {target}."
        detail_ar = f"تقديم شكوى — الموجهة لـ: {target}."
    elif intent == "file_processing":
        detail_en = "Process and upload the file to the system (bulk operation)."
        detail_ar = "معالجة ورفع الملف للسيستم (عملية جماعية)."
    else:
        detail_en = goal_summary
        detail_ar = goal_summary

    details_block = "\n".join(detail_lines) if detail_lines else ""
    details_sep = f"\n{details_block}" if details_block else ""

    if lang in ("ar", "mixed"):
        return (
            f"هأعمل: **{detail_ar}**{details_sep}\n\n"
            "اكتب **نعم** للمتابعة أو **لأ** للإلغاء."
        )
    return (
        f"I'm about to: **{detail_en}**{details_sep}\n\n"
        "Type **yes** to proceed or **no** to cancel."
    )
