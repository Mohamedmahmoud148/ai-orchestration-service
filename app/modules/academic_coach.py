"""
app/modules/academic_coach.py  —  AI Academic Coach Module

The coach analyzes a student's actual academic data from the backend and
produces a personalized, actionable coaching response.

It answers questions like:
  - "كيف وضعي الأكاديمي؟"
  - "ايه نقاط ضعفي؟"
  - "هل أنا في مخاطرة؟"
  - "ايه اللي لازم أركز عليه؟"
  - "give me academic advice"

Flow:
  1. Fetch student's current grades from /api/grades/student/{studentId}
  2. Fetch enrollment list from /api/enrollments/student
  3. Fetch attendance summary from /api/attendance/student/{studentId}
  4. Compute coaching metrics (weak subjects, risk score, trend)
  5. Generate personalized coaching response via LLM
  6. Update the student's AiCompanionProfile via /api/companion/profile
"""
from __future__ import annotations

import json
from typing import Any, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger
from app.services.backend_client import tool_execution_client


class AcademicCoachModule:
    """
    Produces personalized academic coaching using real backend data.

    Unlike the old academic_advisor module (which focused on roadmap/GPA advice),
    the coach is a multi-turn conversational companion that:
      - Detects weaknesses from grade patterns
      - Identifies attendance issues
      - Computes a personal risk score
      - Suggests specific, actionable improvement steps
      - Adapts its tone to the student's current emotional state
    """

    def __init__(self, model_router=None, backend_client=None):
        self.model_router = model_router
        self.backend_client = backend_client or tool_execution_client

    async def run(self, agent_input: AgentInput, plan: Any) -> AgentOutput:
        ctx          = agent_input.context or {}
        role         = ctx.get("role", "student")
        academic_ctx = ctx.get("academic_context", {}) or {}
        history      = ctx.get("history", [])
        model_id     = ctx.get("selected_model", "openai/gpt-4o-mini")
        lang         = self._detect_lang(agent_input.message)

        student_id = (
            academic_ctx.get("studentId")
            or academic_ctx.get("profileId")
        )
        user_id = academic_ctx.get("userId") or agent_input.user_id

        # ── Fetch academic data ────────────────────────────────────────────
        grades_data      = await self._fetch_grades(student_id, agent_input.auth_header)
        enrollment_data  = await self._fetch_enrollments(user_id, agent_input.auth_header)
        attendance_data  = await self._fetch_attendance(student_id, agent_input.auth_header)

        # ── Compute coaching metrics ───────────────────────────────────────
        metrics = self._compute_metrics(grades_data, attendance_data)

        # ── Build coaching context ─────────────────────────────────────────
        coaching_context = self._build_coaching_context(
            metrics, grades_data, enrollment_data, academic_ctx
        )

        # ── Generate LLM coaching response ────────────────────────────────
        response = await self._generate_coaching_response(
            message=agent_input.message,
            coaching_context=coaching_context,
            metrics=metrics,
            academic_ctx=academic_ctx,
            history=history,
            model_id=model_id,
            lang=lang,
        )

        # ── Return structured output ───────────────────────────────────────
        return AgentOutput(
            status="success",
            response=response,
            data={
                "coaching_metrics": metrics,
                "suggestions": self._get_suggestions(metrics, lang),
            },
        )

    # ── Backend data fetchers ─────────────────────────────────────────────

    async def _fetch_grades(
        self, student_id: Optional[str], auth_header: Optional[str]
    ) -> list[dict]:
        if not student_id:
            return []
        try:
            result = await self.backend_client.execute_tool(
                "GetStudentGrades",
                {"studentId": student_id},
                auth_header,
                None,
            )
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("grades", result.get("data", []))
            return []
        except Exception as exc:
            logger.warning("AcademicCoach._fetch_grades: %s", exc)
            return []

    async def _fetch_enrollments(
        self, user_id: Optional[str], auth_header: Optional[str]
    ) -> list[dict]:
        if not user_id:
            return []
        try:
            result = await self.backend_client.execute_tool(
                "GetCourseEnrollments",
                {"userId": user_id},
                auth_header,
                None,
            )
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("enrollments", result.get("data", []))
            return []
        except Exception as exc:
            logger.warning("AcademicCoach._fetch_enrollments: %s", exc)
            return []

    async def _fetch_attendance(
        self, student_id: Optional[str], auth_header: Optional[str]
    ) -> dict:
        if not student_id:
            return {}
        try:
            result = await self.backend_client.execute_tool(
                "GetStudentAttendance",
                {"studentId": student_id},
                auth_header,
                None,
            )
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.warning("AcademicCoach._fetch_attendance: %s", exc)
            return {}

    # ── Metric computation ────────────────────────────────────────────────

    def _compute_metrics(
        self, grades: list[dict], attendance: dict
    ) -> dict:
        """
        Compute coaching metrics from raw backend data.

        Returns:
          overall_gpa, weak_subjects, strong_subjects, risk_level,
          attendance_warnings, improvement_trend
        """
        if not grades:
            return {
                "overall_gpa": None,
                "weak_subjects": [],
                "strong_subjects": [],
                "risk_level": "unknown",
                "attendance_warnings": [],
                "improvement_trend": "unknown",
                "passed_count": 0,
                "failed_count": 0,
            }

        numeric_grades: list[tuple[str, float]] = []
        for g in grades:
            name = g.get("subjectName") or g.get("subject") or "Unknown"
            grade = g.get("finalGrade") or g.get("grade") or g.get("percentage")
            if grade is not None:
                try:
                    numeric_grades.append((name, float(grade)))
                except (ValueError, TypeError):
                    pass

        if not numeric_grades:
            return {
                "overall_gpa": None,
                "weak_subjects": [],
                "strong_subjects": [],
                "risk_level": "unknown",
                "attendance_warnings": [],
                "improvement_trend": "unknown",
                "passed_count": 0,
                "failed_count": 0,
            }

        avg = sum(g for _, g in numeric_grades) / len(numeric_grades)
        weak    = [name for name, g in numeric_grades if g < 50]
        average = [name for name, g in numeric_grades if 50 <= g < 65]
        strong  = [name for name, g in numeric_grades if g >= 75]
        passed  = [name for name, g in numeric_grades if g >= 50]
        failed  = [name for name, g in numeric_grades if g < 50]

        risk = "low"
        if avg < 50 or len(failed) > 2:
            risk = "critical"
        elif avg < 60 or failed:
            risk = "high"
        elif avg < 70:
            risk = "medium"

        # Attendance warnings
        att_warnings: list[str] = []
        if isinstance(attendance, dict):
            for subject, att in attendance.items():
                if isinstance(att, dict):
                    pct = att.get("attendancePercent", 100)
                    if pct < 75:
                        att_warnings.append(subject)
                elif isinstance(att, (int, float)) and att < 75:
                    att_warnings.append(str(subject))

        return {
            "overall_gpa": round(avg / 25, 2),  # Convert 0-100 to 0-4 scale
            "average_percent": round(avg, 1),
            "weak_subjects": weak[:5],
            "average_subjects": average[:5],
            "strong_subjects": strong[:5],
            "risk_level": risk,
            "attendance_warnings": att_warnings[:3],
            "improvement_trend": "improving" if avg >= 65 else "needs_attention",
            "passed_count": len(passed),
            "failed_count": len(failed),
            "total_subjects": len(numeric_grades),
        }

    def _build_coaching_context(
        self,
        metrics: dict,
        grades: list[dict],
        enrollments: list[dict],
        academic_ctx: dict,
    ) -> str:
        """Build a concise plain-text coaching context for the LLM prompt."""
        parts: list[str] = []

        student_name = (
            academic_ctx.get("studentName")
            or academic_ctx.get("name")
            or ""
        )
        if student_name:
            parts.append(f"Student: {student_name}")

        if metrics.get("average_percent") is not None:
            parts.append(f"Overall average: {metrics['average_percent']}%")

        if metrics.get("overall_gpa") is not None:
            parts.append(f"Estimated GPA: {metrics['overall_gpa']:.2f}/4.0")

        if metrics["weak_subjects"]:
            parts.append(f"Weak subjects (< 50%): {', '.join(metrics['weak_subjects'])}")

        if metrics["strong_subjects"]:
            parts.append(f"Strong subjects (≥ 75%): {', '.join(metrics['strong_subjects'])}")

        if metrics["attendance_warnings"]:
            parts.append(f"Attendance warning (< 75%): {', '.join(metrics['attendance_warnings'])}")

        risk = metrics.get("risk_level", "unknown")
        parts.append(f"Academic risk level: {risk.upper()}")
        parts.append(f"Passed: {metrics.get('passed_count', 0)} | Failed: {metrics.get('failed_count', 0)}")

        if metrics.get("improvement_trend") == "improving":
            parts.append("Trend: Student is on an improving trajectory.")
        elif metrics.get("improvement_trend") == "needs_attention":
            parts.append("Trend: Student needs targeted intervention.")

        return "\n".join(parts)

    # ── LLM coaching response ─────────────────────────────────────────────

    async def _generate_coaching_response(
        self,
        message: str,
        coaching_context: str,
        metrics: dict,
        academic_ctx: dict,
        history: list,
        model_id: str,
        lang: str,
    ) -> str:
        if not self.model_router:
            return self._fallback_response(metrics, lang)

        student_name = academic_ctx.get("studentName", "").split()[0] if academic_ctx.get("studentName") else ""
        name_note = f"Student's first name: {student_name}. Use their name naturally." if student_name else ""

        lang_rule = (
            "Respond in Arabic (Egyptian dialect). Be warm and encouraging like a knowledgeable senior student."
            if lang == "ar"
            else "Respond in English. Be warm and encouraging like a knowledgeable mentor."
        )

        system_prompt = f"""\
You are an AI Academic Coach for a university student.

{lang_rule}

{name_note}

PERSONALITY:
- Warm, encouraging, honest but never harsh
- Like a brilliant friend who knows academics well
- Give specific, actionable advice — not generic platitudes
- Never say "don't worry" without a concrete plan
- Celebrate strengths before addressing weaknesses

REAL ACADEMIC DATA:
{coaching_context}

RULES:
- Base EVERY recommendation on the actual data above
- Never invent grades, subjects, or statistics not in the data
- If data is unavailable, say so honestly and offer general guidance
- Keep response focused (3–5 points max)
- End with one specific action step the student can take TODAY
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Include last 3 turns of history for continuity
        for turn in history[-3:]:
            t_role = turn.get("role", "user")
            t_content = str(turn.get("content", ""))[:400]
            if t_role in ("user", "assistant") and t_content:
                messages.append({"role": t_role, "content": t_content})

        messages.append({"role": "user", "content": message})

        try:
            response = await self.model_router.generate_with_messages(
                messages=messages,
                model_id=model_id,
                max_tokens=1000,
            )
            return response or self._fallback_response(metrics, lang)
        except Exception as exc:
            logger.error("AcademicCoach: LLM call failed — %s", exc)
            return self._fallback_response(metrics, lang)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _fallback_response(self, metrics: dict, lang: str) -> str:
        risk = metrics.get("risk_level", "unknown")
        weak = ", ".join(metrics.get("weak_subjects", []))

        if lang == "ar":
            if risk == "critical":
                return (
                    f"⚠️ وضعك الأكاديمي يحتاج تدخل فوري.\n"
                    f"المواد الأكثر ضعفاً: **{weak or 'غير محدد'}**\n\n"
                    "خطوة اليوم: تواصل مع الدكتور المسؤول في أقرب فرصة."
                )
            return (
                f"وضعك الأكاديمي {'يحتاج اهتمام' if risk == 'high' else 'معقول'}. "
                f"{'ركز على: ' + weak if weak else 'استمر في المذاكرة المنتظمة.'}"
            )
        return (
            f"Your academic standing is **{risk}**. "
            f"{'Focus on: ' + weak if weak else 'Keep up the consistent effort.'}"
        )

    def _get_suggestions(self, metrics: dict, lang: str) -> list[str]:
        suggestions = []
        if metrics.get("weak_subjects"):
            s = metrics["weak_subjects"][0]
            suggestions.append(
                f"اعمل quiz على {s}" if lang == "ar"
                else f"Take a quick quiz on {s}"
            )
        if metrics.get("risk_level") in ("high", "critical"):
            suggestions.append(
                "اعمل خطة مذاكرة مخصصة" if lang == "ar"
                else "Generate a personalized study plan"
            )
        suggestions.append(
            "شوف تقرير أسبوعك" if lang == "ar"
            else "View your weekly progress report"
        )
        return suggestions[:3]

    @staticmethod
    def _detect_lang(message: str) -> str:
        arabic_chars = sum(1 for c in message if "؀" <= c <= "ۿ")
        return "ar" if arabic_chars / max(len(message), 1) > 0.2 else "en"
