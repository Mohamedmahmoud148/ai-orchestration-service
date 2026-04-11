"""
app/core/rbac.py  —  v2.0

Role-Based Access Control (RBAC) for the AI orchestration pipeline.

This is the SINGLE SOURCE OF TRUTH for intent-level permissions.
The PlanExecutor enforces this gate BEFORE calling any module or backend tool.

Roles & Permissions
-------------------
  student  — personal queries, complaints, CV, advice, material study
  doctor   — exam generation, student data, summaries, materials, complaints
  admin    — all operations including bulk file processing

Architecture rules
------------------
- AI enforces intent-level RBAC (what you're ALLOWED to ask for).
- Backend enforces data-level permissions (what records you can see).
- Both layers are required — AI RBAC is NOT a substitute for backend auth.
- `agent.py` NO LONGER duplicates this gate; executor.py is the sole enforcer.

Denied attempts are logged via `log_blocked_attempt()` for monitoring.

Permission matrix
-----------------
  intent                student  doctor  admin
  ─────────────────────────────────────────────
  general_chat            ✅       ✅      ✅
  summarization           ✅       ✅      ✅
  result_query            ✅       ✅      ✅
  file_extraction         ✅       ✅      ✅
  complaint_submit        ✅       ❌      ✅
  complaint_summary       ❌       ✅      ✅
  file_processing         ❌       ❌      ✅
  cv_analysis             ✅       ✅      ✅
  academic_advice         ✅       ✅      ✅
  material_explanation    ✅       ✅      ✅
  generate_exam           ❌       ✅      ✅
"""
from __future__ import annotations

import datetime
from typing import FrozenSet, Literal, Union

from app.core.logging import logger

# ── Sentinel: admin gets every intent ─────────────────────────────────────────
_ALL = "*"

RoleType = Literal["student", "doctor", "admin"]

# ── Intent sets per role ───────────────────────────────────────────────────────
_STUDENT_ALLOWED: FrozenSet[str] = frozenset({
    "general_chat",
    "summarization",
    "result_query",
    "file_extraction",
    "complaint_submit",
    "cv_analysis",
    "academic_advice",
    "material_explanation",
})

_DOCTOR_ALLOWED: FrozenSet[str] = frozenset({
    "general_chat",
    "summarization",
    "result_query",
    "generate_exam",
    "complaint_summary",
    "material_explanation",
    "academic_advice",
    "file_extraction",
    "cv_analysis",
})

# admin → _ALL (checked dynamically — no explicit list needed)

# ── Permission map ─────────────────────────────────────────────────────────────
ROLE_PERMISSIONS: dict[str, Union[FrozenSet[str], str]] = {
    "student": _STUDENT_ALLOWED,
    "doctor":  _DOCTOR_ALLOWED,
    "admin":   _ALL,
}

# ── Human-readable intent labels for error messages ────────────────────────────
_INTENT_LABELS: dict[str, str] = {
    "general_chat":         "General Chat",
    "summarization":        "Document Summarization",
    "result_query":         "Academic Results Query",
    "file_extraction":      "File Extraction",
    "complaint_submit":     "Complaint Submission",
    "complaint_summary":    "Complaint Summary",
    "file_processing":      "Bulk File Processing",
    "cv_analysis":          "CV Analysis",
    "academic_advice":      "Academic Advice",
    "material_explanation": "Material Explanation",
    "generate_exam":        "Exam Generation",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def is_allowed(intent: str, role: str) -> bool:
    """
    Return True if the given role is permitted to trigger this intent.

    Unknown roles default to student permissions (most restrictive).
    """
    permissions = ROLE_PERMISSIONS.get(role, _STUDENT_ALLOWED)
    if permissions is _ALL:
        return True
    return intent in permissions  # type: ignore[operator]


def get_denial_message(intent: str, role: str) -> str:
    """
    Return the standardised denial message for a blocked intent.

    Per security spec: never reveals internal role logic or system structure.
    Format: "You are not authorized to perform this action. Please contact your administrator."
    """
    label      = _INTENT_LABELS.get(intent, intent.replace("_", " ").title())
    role_label = role.capitalize()
    return (
        f"You are not authorized to perform '{label}' with your current role ({role_label}). "
        "Please contact your administrator if you believe this is a mistake."
    )


def log_blocked_attempt(
    intent: str,
    role: str,
    user_id: str | None = None,
    extra: dict | None = None,
) -> None:
    """
    Emit a structured WARNING log entry for every blocked RBAC attempt.

    Fields logged (for SIEM / log aggregation):
      event, timestamp, user_id, role, intent, intent_label, extra
    """
    label = _INTENT_LABELS.get(intent, intent)
    logger.warning(
        "RBAC_BLOCKED event=rbac_denied ts=%s user_id=%s role=%r "
        "intent=%r intent_label=%r extra=%s",
        datetime.datetime.utcnow().isoformat(),
        user_id or "unknown",
        role,
        intent,
        label,
        extra or {},
    )


def get_intent_label(intent: str) -> str:
    """Return a human-readable label for an intent (used in logs/UI)."""
    return _INTENT_LABELS.get(intent, intent.replace("_", " ").title())
