"""
app/modules/academic_advisor.py

Academic Advisor Module — intent: academic_advice

Provides personalised academic advice based on:
  - GPA and enrolled courses from academic_context (always checked first)
  - Optional: GET /api/ai-tools/student-academic-summary (if context is sparse)

Detects:
  - Weak subjects (low GPA, failed attempts)

Recommends:
  - Courses to take / retake
  - Study plan tailored to identified weaknesses

Output: structured bullet-point advice (personalized, not generic).

Architecture rules
------------------
- AI reads academic_context, optionally enriches from backend, then advises.
- Backend is the single source of truth — AI never invents grade values.
- No direct DB access.
"""
from __future__ import annotations

import json as _json
from typing import Any, Dict, List, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger

_DEFAULT_MODEL = "openai/gpt-4o-mini"

# GPA thresholds
_GPA_WEAK_THRESHOLD      = 2.0   # below this → weak subject
_GPA_AVERAGE_THRESHOLD   = 2.75  # below this → needs improvement
_GPA_GOOD_THRESHOLD      = 3.5   # above this → strong

# Safe context keys to read from academic_context
_SAFE_CONTEXT_KEYS = [
    "gpa", "GPA",
    "enrolledCourses", "courses",
    "failedCourses", "failedSubjects",
    "passedCourses",
    "currentSemester", "semester",
    "departmentName", "collegeName",
    "studentName", "userId", "studentId",
    "creditHoursCompleted", "creditHoursRequired",
]


def _extract_academic_data(academic_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Pull and normalise academic data from context."""
    gpa = (
        academic_ctx.get("gpa")
        or academic_ctx.get("GPA")
        or academic_ctx.get("cgpa")
        or academic_ctx.get("CGPA")
    )
    try:
        gpa = float(gpa) if gpa is not None else None
    except (ValueError, TypeError):
        gpa = None

    enrolled = (
        academic_ctx.get("enrolledCourses")
        or academic_ctx.get("courses")
        or []
    )
    failed = (
        academic_ctx.get("failedCourses")
        or academic_ctx.get("failedSubjects")
        or []
    )

    return {
        "gpa":          gpa,
        "enrolled":     enrolled if isinstance(enrolled, list) else [],
        "failed":       failed   if isinstance(failed, list) else [],
        "semester":     academic_ctx.get("currentSemester") or academic_ctx.get("semester"),
        "department":   academic_ctx.get("departmentName"),
        "student_name": academic_ctx.get("studentName"),
    }


def _classify_gpa(gpa: Optional[float]) -> str:
    if gpa is None:
        return "unknown"
    if gpa < _GPA_WEAK_THRESHOLD:
        return "at_risk"
    if gpa < _GPA_AVERAGE_THRESHOLD:
        return "needs_improvement"
    if gpa < _GPA_GOOD_THRESHOLD:
        return "average"
    return "strong"


def _build_advisor_prompt(
    data: Dict[str, Any],
    enriched: Optional[Dict[str, Any]],
    user_message: str,
) -> str:
    """Build a rich, context-aware prompt for the advisor LLM."""
    lines: List[str] = [
        f"Student request: {user_message}\n",
        "=== Academic Profile ===",
    ]

    if data["student_name"]:
        lines.append(f"Name:        {data['student_name']}")
    if data["department"]:
        lines.append(f"Department:  {data['department']}")
    if data["semester"]:
        lines.append(f"Semester:    {data['semester']}")
    if data["gpa"] is not None:
        lines.append(f"Current GPA: {data['gpa']:.2f} ({_classify_gpa(data['gpa'])})")
    else:
        lines.append("Current GPA: not available")

    if data["enrolled"]:
        lines.append(f"\nEnrolled courses ({len(data['enrolled'])}):")
        for c in data["enrolled"][:10]:  # cap to avoid token overflow
            if isinstance(c, dict):
                name  = c.get("name") or c.get("subjectName") or c.get("courseName") or str(c)
                grade = c.get("grade") or c.get("score") or ""
                lines.append(f"  - {name}" + (f" (grade: {grade})" if grade else ""))
            else:
                lines.append(f"  - {c}")

    if data["failed"]:
        lines.append(f"\nFailed/at-risk subjects:")
        for c in data["failed"][:10]:
            if isinstance(c, dict):
                lines.append(f"  ⚠ {c.get('name') or c.get('subjectName') or c}")
            else:
                lines.append(f"  ⚠ {c}")

    if enriched:
        lines.append(f"\n=== Academic Summary (from backend) ===")
        lines.append(_json.dumps(enriched, ensure_ascii=False)[:1_500])

    return "\n".join(lines)


_ADVISOR_SYSTEM_PROMPT = """\
You are a personalised academic advisor at a university management system.
Your goal is to help students improve their academic performance with
specific, actionable advice — NOT generic platitudes.

IMPORTANT rules:
- Base recommendations STRICTLY on the data provided. Do NOT invent GPA values or course names.
- If GPA is below 2.0, prioritise urgent recovery advice.
- If GPA is between 2.0-2.75, focus on specific improvement tactics.
- If GPA is above 3.5, suggest paths for further excellence (honours, research, etc.).
- Always recommend specific, named courses or study methods — never vague advice.

Output format (use markdown):

**📊 Academic Status Assessment**
(one paragraph evaluating the student's current standing)

**⚠️ Weak Areas Identified**
(bullet list — based ONLY on the data provided)

**📚 Course Recommendations**
(3-5 specific courses to take, retake, or focus on — with brief reason)

**🗓️ Study Plan (Next 4 Weeks)**
(concrete weekly plan tailored to their weaknesses)

**💡 Additional Tips**
(2-3 short, specific tips based on their profile)

Respond in the same language as the student's message.\
"""


class AcademicAdvisorModule:
    """
    Generates personalised academic advice using academic_context as primary
    data source. Optionally enriches from backend if context is sparse.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        ctx      = agent_input.context or {}
        model_id = ctx.get("selected_model") or _DEFAULT_MODEL

        academic_ctx: Dict[str, Any] = ctx.get("academic_context", {}) or {}

        # ── 1. Extract academic data from context ─────────────────────────────
        data = _extract_academic_data(academic_ctx)

        logger.info(
            "AcademicAdvisorModule: gpa=%s enrolled=%d failed=%d",
            data["gpa"], len(data["enrolled"]), len(data["failed"]),
        )

        # ── 2. Optionally enrich from backend if context is sparse ────────────
        enriched: Optional[Dict[str, Any]] = None
        context_is_sparse = (
            data["gpa"] is None
            and not data["enrolled"]
            and not data["failed"]
        )

        if context_is_sparse and self.backend_client:
            user_id = (
                agent_input.user_id
                or academic_ctx.get("userId")
                or academic_ctx.get("studentId")
            )
            if user_id:
                logger.info(
                    "AcademicAdvisorModule: sparse context — fetching from backend "
                    "for userId=%s", user_id
                )
                try:
                    enriched = await self.backend_client.fetch(
                        route="/api/ai-tools/student-academic-summary",
                        auth_header=agent_input.auth_header,
                        params={"userId": user_id},
                    )
                    if enriched and "error" in enriched:
                        logger.warning(
                            "AcademicAdvisorModule: backend enrichment error — %s",
                            enriched.get("error"),
                        )
                        enriched = None
                    # Try to re-parse GPA from enriched if still None
                    if enriched and data["gpa"] is None:
                        data = _extract_academic_data({**academic_ctx, **enriched})
                except Exception as exc:
                    logger.warning(
                        "AcademicAdvisorModule: enrichment failed (non-fatal) — %s", exc
                    )
                    enriched = None

        # ── 3. Build analysis prompt ──────────────────────────────────────────
        prompt = _build_advisor_prompt(data, enriched, agent_input.message)

        # ── 4. LLM advice generation ──────────────────────────────────────────
        advice = await self.model_router.generate(
            prompt=prompt,
            system_instruction=_ADVISOR_SYSTEM_PROMPT,
            model_id=model_id,
        )

        if not advice:
            return AgentOutput(
                status="failed",
                response=(
                    "I couldn't generate personalised advice at this time. "
                    "Please try again in a moment."
                ),
            )

        gpa_status = _classify_gpa(data["gpa"])
        return AgentOutput(
            status="success",
            response=advice,
            data={
                "module":          "AcademicAdvisorModule",
                "gpa":             data["gpa"],
                "gpa_status":      gpa_status,
                "enrolled_count":  len(data["enrolled"]),
                "failed_count":    len(data["failed"]),
                "enriched_from_backend": enriched is not None,
                "model_used":      model_id,
            },
        )
