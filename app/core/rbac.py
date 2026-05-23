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
  backend_api_query       ✅       ✅      ✅
  material_qa             ✅       ✅      ✅
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
    "backend_api_query",
    "action_execute",     # Students can enroll, submit — JWT enforces data-level RBAC
    "material_qa",
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
    "backend_api_query",
    "action_execute",     # Doctors can create exams, grade submissions
    "material_qa",
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
    "backend_api_query":    "Data Query",
    "action_execute":       "System Action",
    "material_qa":          "Course Material Q&A",
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
    Return a bilingual, user-friendly denial message.

    Never reveals internal role logic, intent names, or system structure.
    Students and doctors receive actionable guidance, not technical jargon.
    """
    if role == "student":
        if intent == "complaint_summary":
            return (
                "عرض ملخصات الشكاوى متاح للدكاترة والإدارة فقط.\n"
                "Complaint summaries are available to doctors and admins only."
            )
        if intent == "generate_exam":
            return (
                "إنشاء الامتحانات متاح للدكاترة فقط.\n"
                "Exam generation is available to doctors only."
            )
        if intent == "file_processing":
            return (
                "معالجة الملفات المجمّعة متاحة للإدارة فقط.\n"
                "Bulk file processing is available to admins only."
            )
    if role == "doctor":
        if intent == "complaint_submit":
            return (
                "تقديم الشكاوى متاح للطلاب فقط. إذا كان لديك مشكلة يُرجى التواصل مع الإدارة.\n"
                "Complaint submission is available to students only. "
                "Please contact the administration if you have an issue."
            )
        if intent == "file_processing":
            return (
                "معالجة الملفات المجمّعة متاحة للإدارة فقط.\n"
                "Bulk file processing is available to admins only."
            )

    # Generic fallback — no technical details exposed
    return (
        "عذراً، ليس لديك صلاحية للقيام بهذا الإجراء. "
        "تواصل مع الإدارة إذا كنت تعتقد أن هذا خطأ.\n"
        "Sorry, you don't have permission to perform this action. "
        "Please contact the administration if you believe this is a mistake."
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
