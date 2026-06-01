"""
app/modules/progress_intelligence.py  —  Progress Intelligence Module

Generates personalized academic progress reports and insights.

Handles intents:
  - weekly_report    → "ايه اللي عملته الأسبوع ده؟"
  - monthly_report   → "تقرير الشهر"
  - progress_check   → "هل بتحسن؟"
  - study_analysis   → "حلل مذاكرتي"

This module combines:
  1. Real grade data from the backend
  2. Learning session history from the companion API
  3. Engagement metrics from the profile
  4. LLM-generated narrative report
"""
from __future__ import annotations

import json
from typing import Any, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger
from app.services.backend_client import tool_execution_client


class ProgressIntelligenceModule:
    """
    Generates intelligent, personalized academic progress reports.

    The report is data-driven (real grades + session history) and
    narrated by the LLM in the student's language.
    """

    def __init__(self, model_router=None, backend_client=None):
        self.model_router = model_router
        self.backend_client = backend_client or tool_execution_client

    async def run(self, agent_input: AgentInput, plan: Any) -> AgentOutput:
        ctx          = agent_input.context or {}
        academic_ctx = ctx.get("academic_context", {}) or {}
        model_id     = ctx.get("selected_model", "openai/gpt-4o-mini")
        lang         = self._detect_lang(agent_input.message)

        student_id = academic_ctx.get("studentId") or academic_ctx.get("profileId")
        user_id    = academic_ctx.get("userId") or agent_input.user_id
        period     = self._detect_period(agent_input.message)

        # ── Fetch data ─────────────────────────────────────────────────────
        grades     = await self._fetch_grades(student_id, agent_input.auth_header)
        gpa_data   = await self._fetch_gpa(user_id, agent_input.auth_header)
        companion_summary = await self._fetch_companion_summary(
            user_id, agent_input.auth_header
        )

        # ── Build report data ──────────────────────────────────────────────
        report_data = self._compile_report_data(
            grades, gpa_data, companion_summary, academic_ctx, period
        )

        # ── Generate LLM narrative ─────────────────────────────────────────
        narrative = await self._generate_report_narrative(
            message=agent_input.message,
            report_data=report_data,
            period=period,
            academic_ctx=academic_ctx,
            model_id=model_id,
            lang=lang,
        )

        return AgentOutput(
            status="success",
            response=narrative,
            data={
                "report_data": report_data,
                "period": period,
                "suggestions": self._get_suggestions(report_data, lang),
            },
        )

    # ── Data fetchers ─────────────────────────────────────────────────────

    async def _fetch_grades(
        self, student_id: Optional[str], auth_header: Optional[str]
    ) -> list[dict]:
        if not student_id:
            return []
        try:
            result = await self.backend_client.execute_tool(
                "GetStudentGrades", {"studentId": student_id}, auth_header, None
            )
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("grades", [])
            return []
        except Exception as exc:
            logger.warning("ProgressIntelligence._fetch_grades: %s", exc)
            return []

    async def _fetch_gpa(
        self, user_id: Optional[str], auth_header: Optional[str]
    ) -> dict:
        if not user_id:
            return {}
        try:
            result = await self.backend_client.execute_tool(
                "GetGPASummary", {"userId": user_id}, auth_header, None
            )
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.warning("ProgressIntelligence._fetch_gpa: %s", exc)
            return {}

    async def _fetch_companion_summary(
        self, user_id: Optional[str], auth_header: Optional[str]
    ) -> dict:
        """Fetch study session stats from companion memory (Redis)."""
        if not user_id:
            return {}
        try:
            result = await self.backend_client.execute_tool(
                "GetCompanionSummary", {"userId": user_id}, auth_header, None
            )
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}  # Non-fatal — companion data is optional

    # ── Report compilation ────────────────────────────────────────────────

    def _compile_report_data(
        self,
        grades: list[dict],
        gpa_data: dict,
        companion: dict,
        academic_ctx: dict,
        period: str,
    ) -> dict:
        """Compile all data sources into a unified report dict."""

        # Grade analysis
        numeric_grades = []
        for g in grades:
            val = g.get("finalGrade") or g.get("grade") or g.get("percentage")
            name = g.get("subjectName") or g.get("subject") or "Unknown"
            if val is not None:
                try:
                    numeric_grades.append({"subject": name, "grade": float(val)})
                except (ValueError, TypeError):
                    pass

        avg_grade = (
            sum(g["grade"] for g in numeric_grades) / len(numeric_grades)
            if numeric_grades else None
        )

        best_subject = max(numeric_grades, key=lambda x: x["grade"], default=None)
        worst_subject = min(numeric_grades, key=lambda x: x["grade"], default=None)

        return {
            "period": period,
            "student_name": academic_ctx.get("studentName", ""),
            "gpa": gpa_data.get("gpa") or gpa_data.get("currentGpa"),
            "average_grade": round(avg_grade, 1) if avg_grade else None,
            "total_subjects": len(numeric_grades),
            "passed_subjects": sum(1 for g in numeric_grades if g["grade"] >= 50),
            "failed_subjects": sum(1 for g in numeric_grades if g["grade"] < 50),
            "best_subject": best_subject,
            "worst_subject": worst_subject,
            "grade_distribution": {
                "A (85+)": sum(1 for g in numeric_grades if g["grade"] >= 85),
                "B (70-84)": sum(1 for g in numeric_grades if 70 <= g["grade"] < 85),
                "C (60-69)": sum(1 for g in numeric_grades if 60 <= g["grade"] < 70),
                "D (50-59)": sum(1 for g in numeric_grades if 50 <= g["grade"] < 60),
                "F (<50)": sum(1 for g in numeric_grades if g["grade"] < 50),
            },
            # Companion/study data
            "study_sessions": companion.get("sessions_count", 0),
            "study_minutes": companion.get("total_minutes", 0),
            "avg_quiz_accuracy": companion.get("avg_accuracy"),
            "current_streak": companion.get("streak_days", 0),
            "flashcards_reviewed": companion.get("flashcards_reviewed", 0),
        }

    # ── LLM narrative generation ──────────────────────────────────────────

    async def _generate_report_narrative(
        self,
        message: str,
        report_data: dict,
        period: str,
        academic_ctx: dict,
        model_id: str,
        lang: str,
    ) -> str:
        if not self.model_router:
            return self._fallback_report(report_data, period, lang)

        student_name = report_data.get("student_name", "").split()[0] if report_data.get("student_name") else ""

        lang_rule = (
            "Write in Arabic. Use warm Egyptian dialect. Start with the student's name if available."
            if lang == "ar" else
            "Write in English. Be encouraging and data-driven."
        )

        # Serialize report data for LLM
        data_json = json.dumps(report_data, ensure_ascii=False, indent=2)

        system_prompt = f"""\
You are an AI Academic Coach generating a personalized progress report for a student.

{lang_rule}

REPORT DATA:
{data_json}

TASK: Generate a {period} progress report that:
1. Opens with the student's name and a brief overall assessment
2. Highlights the BEST achievement first (positive reinforcement)
3. Addresses weak areas with specific, actionable advice
4. Mentions study habits if data is available
5. Ends with 1-2 specific goals for the next {period}

TONE: Like a mentor who genuinely cares. Data-driven but human.
LENGTH: 150-250 words.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        try:
            return await self.model_router.generate_with_messages(
                messages=messages,
                model_id=model_id,
                max_tokens=500,
            ) or self._fallback_report(report_data, period, lang)
        except Exception as exc:
            logger.error("ProgressIntelligence: LLM failed — %s", exc)
            return self._fallback_report(report_data, period, lang)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _detect_period(self, message: str) -> str:
        msg = message.lower()
        if any(k in msg for k in ["month", "شهر", "شهري", "monthly"]):
            return "monthly"
        return "weekly"

    def _fallback_report(self, data: dict, period: str, lang: str) -> str:
        avg = data.get("average_grade")
        passed = data.get("passed_subjects", 0)
        failed = data.get("failed_subjects", 0)
        sessions = data.get("study_sessions", 0)

        if lang == "ar":
            return (
                f"📊 **تقرير {period}**\n\n"
                f"• المتوسط الكلي: **{avg or 'غير متاح'}%**\n"
                f"• مواد ناجحة: {passed} | راسب في: {failed}\n"
                f"• جلسات مذاكرة: {sessions}\n\n"
                "استمر في المذاكرة المنتظمة! 💪"
            )
        return (
            f"📊 **{period.title()} Report**\n\n"
            f"• Average: **{avg or 'N/A'}%**\n"
            f"• Passed: {passed} | Failed: {failed}\n"
            f"• Study sessions: {sessions}\n\n"
            "Keep up the consistent effort! 💪"
        )

    def _get_suggestions(self, data: dict, lang: str) -> list[str]:
        suggestions = []
        if data.get("failed_subjects", 0) > 0:
            s = "وضع خطة لمواد الرسوب" if lang == "ar" else "Create a recovery plan for failed subjects"
            suggestions.append(s)
        if data.get("study_sessions", 0) < 3:
            s = "ابدأ جلسة مذاكرة جديدة" if lang == "ar" else "Start a new study session"
            suggestions.append(s)
        s = "شوف نصيحة أكاديمية" if lang == "ar" else "Get academic advice"
        suggestions.append(s)
        return suggestions[:3]

    @staticmethod
    def _detect_lang(message: str) -> str:
        arabic_chars = sum(1 for c in message if "؀" <= c <= "ۿ")
        return "ar" if arabic_chars / max(len(message), 1) > 0.2 else "en"
