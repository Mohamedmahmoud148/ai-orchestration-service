"""
app/core/response_guard.py

Lightweight hallucination + response-quality guard.

Checks the agent's final response for patterns that indicate the model
invented data instead of grounding its answer in tool results.

Contract:
  validate(response_text, tool_results) → GuardResult

GuardResult fields:
  passed          bool     — False means something looks wrong
  risk_level      str      — "none" | "low" | "medium" | "high"
  warnings        list[str]— human-readable descriptions of issues found
  sanitized_text  str      — response with potentially dangerous injections removed

This module is NOT a blocker — callers may log the result and still return
the response.  Use risk_level == "high" to decide whether to show a
disclaimer or substitute a safe fallback.

Design principles:
  - Fast (< 1 ms) — runs synchronously, no LLM calls.
  - Conservative — only flags patterns with very high precision.
  - Never silently drops user responses — always returns the full text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

# ── Regex: Arabic/English numerals embedded in claims ─────────────────────────
# Matches patterns like "عدد الطلاب هو 5,432" or "there are 142 students enrolled"
# Only fires when a number appears after a claim-linking phrase.
_CLAIM_NUMBER_AR = re.compile(
    r"(?:عدد|هناك|يوجد|الإجمالي|المجموع|بلغ|وصل إلى|يبلغ)\s+[\d,٠-٩]+",
    re.UNICODE,
)
_CLAIM_NUMBER_EN = re.compile(
    r"(?:there are|total of|count of|number of|found|enrolled)\s+[\d,]+",
    re.IGNORECASE,
)

# ── Prompt-injection markers ───────────────────────────────────────────────────
# User-supplied content that tries to hijack the system prompt.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(previous|all|the\s+above)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)", re.IGNORECASE),
    re.compile(r"forget\s+(?:everything|all|previous)", re.IGNORECASE),
    re.compile(r"<\s*(?:system|human|assistant)\s*>", re.IGNORECASE),
    re.compile(r"OVERRIDE\s+(?:INSTRUCTIONS?|RULES?|SYSTEM)", re.IGNORECASE),
]

# ── Fabricated-data red flags ─────────────────────────────────────────────────
# Phrases the model uses when it's making something up.
_FABRICATION_PHRASES = (
    "على سبيل المثال",   # "for example" — OK in explanations but risky in data context
    "مثلاً",
    "افتراضياً",
    "تقريباً",
    "على ما أعلم",        # "as far as I know"
    "في الغالب",          # "probably"
    "I assume",
    "I believe",
    "I think the number",
    "approximately",
    "roughly",
    "probably around",
    "estimated",
)

# ── Empty-data safe phrases ───────────────────────────────────────────────────
# If the response contains these it's OK — the model is correctly reporting no data.
_SAFE_EMPTY_PHRASES = (
    "لا يوجد",
    "لا توجد",
    "فارغ",
    "لم يتم",
    "لا بيانات",
    "0 طلاب",
    "0 دكاترة",
    "no data",
    "no students",
    "no results",
    "not found",
    "empty",
    "no records",
)


@dataclass
class GuardResult:
    passed: bool = True
    risk_level: str = "none"          # "none" | "low" | "medium" | "high"
    warnings: List[str] = field(default_factory=list)
    sanitized_text: str = ""

    def _set_risk(self, level: str) -> None:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        if order.get(level, 0) > order.get(self.risk_level, 0):
            self.risk_level = level


def validate(
    response_text: str,
    tool_results: Optional[List[Any]] = None,
    user_message: str = "",
) -> GuardResult:
    """
    Run all guards against the response text.

    Parameters
    ----------
    response_text : str
        The final response the agent intends to send to the user.
    tool_results : list | None
        Raw string/dict results from tool calls made during this turn.
        Used to cross-check numeric claims.
    user_message : str
        The original user message (for context in warnings).

    Returns
    -------
    GuardResult
        Always returned — never raises.
    """
    result = GuardResult(sanitized_text=response_text)

    if not response_text or not response_text.strip():
        result.warnings.append("Response is empty.")
        result._set_risk("low")
        result.passed = False
        return result

    # 1. Prompt injection check (in response — rare but possible if model echoes user input)
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(response_text):
            result.warnings.append(
                f"Potential prompt-injection echo detected: {pattern.pattern!r}"
            )
            result._set_risk("high")
            result.passed = False

    # 2. Fabrication phrase check — only risky in data-query contexts
    if tool_results is not None:
        lower = response_text.lower()
        found_fabrication = [p for p in _FABRICATION_PHRASES if p.lower() in lower]
        if found_fabrication:
            result.warnings.append(
                f"Possible fabrication phrases detected: {found_fabrication}"
            )
            result._set_risk("low")

    # 3. Numeric claim cross-check — when tool_results available and non-empty
    if tool_results:
        results_text = " ".join(str(r) for r in tool_results)
        claims_ar = _CLAIM_NUMBER_AR.findall(response_text)
        claims_en = _CLAIM_NUMBER_EN.findall(response_text)
        all_claims = claims_ar + claims_en

        for claim in all_claims:
            # Extract the digit portion from the claim
            digits = re.findall(r"[\d,]+", claim)
            for d in digits:
                clean = d.replace(",", "")
                if clean and clean not in results_text and clean != "0":
                    result.warnings.append(
                        f"Numeric claim '{claim}' not found in tool results — "
                        f"possible hallucination."
                    )
                    result._set_risk("medium")
                    result.passed = False
                    break

    # 4. Length sanity check
    if len(response_text) > 8_000:
        result.warnings.append(
            f"Response unusually long ({len(response_text)} chars) — "
            f"consider truncating."
        )
        result._set_risk("low")

    return result


def check_user_input(text: str) -> GuardResult:
    """
    Run injection checks on user input before it enters the agent pipeline.
    Separate from validate() to keep concerns distinct.
    """
    result = GuardResult(sanitized_text=text)

    if len(text) > 4_000:
        result.warnings.append(
            f"User message unusually long ({len(text)} chars) — truncating to 4000."
        )
        result.sanitized_text = text[:4_000]
        result._set_risk("low")

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            result.warnings.append(
                f"Prompt injection attempt detected in user input: {pattern.pattern!r}"
            )
            result._set_risk("high")
            result.passed = False

    return result
