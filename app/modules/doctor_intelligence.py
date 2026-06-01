"""
app/modules/doctor_intelligence.py  —  AI Doctor Intelligence Module

Handles teaching intelligence requests from doctors via the AI chat interface.

Supported intents:
  - doctor_analytics       → class performance overview
  - doctor_risk_students   → at-risk student list with explanations
  - doctor_weak_topics     → topic analysis
  - doctor_recommendations → AI teaching recommendations

This module calls the .NET backend's Teaching Intelligence APIs
and narrates results in the doctor's language.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger
from app.services.backend_client import tool_execution_client


class DoctorIntelligenceModule:
    """
    Doctor-facing intelligence module.

    Fetches pre-computed analytics from the Teaching Intelligence APIs
    and generates an intelligent narrative for the doctor.
    """

    def __init__(self, model_router=None, backend_client=None):
        self.model_router = model_router
        self.backend_client = backend_client or tool_execution_client

    async def run(self, agent_input: AgentInput, plan: Any) -> AgentOutput:
        ctx          = agent_input.context or {}
        academic_ctx = ctx.get("academic_context", {}) or {}
        model_id     = ctx.get("selected_model", "openai/gpt-4o-mini")
        lang         = self._detect_lang(agent_input.message)
        history      = ctx.get("history", [])

        intent   = getattr(plan, "intent", "doctor_analytics")
        user_id  = academic_ctx.get("userId") or agent_input.user_id
        offering_id = academic_ctx.get("subjectOfferingId")

        # ── Fetch data ─────────────────────────────────────────────────────
        if intent == "doctor_risk_students":
            data = await self._fetch_at_risk(user_id, agent_input.auth_header)
            response = await self._narrate_risk_students(
                data, agent_input.message, history, model_id, lang
            )
            return AgentOutput(
                status="success",
                response=response,
                data={"at_risk_students": data, "suggestions": self._risk_suggestions(lang)},
            )

        if intent == "doctor_weak_topics" and offering_id:
            data = await self._fetch_weak_topics(offering_id, agent_input.auth_header)
            response = await self._narrate_weak_topics(
                data, agent_input.message, model_id, lang
            )
            return AgentOutput(
                status="success",
                response=response,
                data={"weak_topics": data, "suggestions": self._topic_suggestions(lang)},
            )

        # Default: dashboard overview
        dashboard = await self._fetch_dashboard(user_id, agent_input.auth_header)
        response = await self._narrate_dashboard(
            dashboard, agent_input.message, history, model_id, lang
        )
        return AgentOutput(
            status="success",
            response=response,
            data={
                "dashboard_summary": dashboard,
                "suggestions": self._dashboard_suggestions(lang),
            },
        )

    # ── Backend fetchers ──────────────────────────────────────────────────

    async def _fetch_dashboard(
        self, user_id: Optional[str], auth_header: Optional[str]
    ) -> dict:
        try:
            result = await self.backend_client.execute_tool(
                "GetTeachingDashboard", {"userId": user_id}, auth_header, None
            )
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.warning("DoctorIntelligence._fetch_dashboard: %s", exc)
            return {}

    async def _fetch_at_risk(
        self, user_id: Optional[str], auth_header: Optional[str]
    ) -> list:
        try:
            result = await self.backend_client.execute_tool(
                "GetAtRiskStudents", {"userId": user_id, "minRiskLevel": "medium"},
                auth_header, None
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.warning("DoctorIntelligence._fetch_at_risk: %s", exc)
            return []

    async def _fetch_weak_topics(
        self, offering_id: str, auth_header: Optional[str]
    ) -> list:
        try:
            result = await self.backend_client.execute_tool(
                "GetWeakTopics", {"offeringId": offering_id}, auth_header, None
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.warning("DoctorIntelligence._fetch_weak_topics: %s", exc)
            return []

    # ── LLM narration ─────────────────────────────────────────────────────

    async def _narrate_dashboard(
        self, dashboard: dict, message: str, history: list,
        model_id: str, lang: str
    ) -> str:
        if not self.model_router or not dashboard:
            return self._fallback_dashboard(dashboard, lang)

        stats = dashboard.get("overallStats", {})
        offerings = dashboard.get("offerings", [])[:3]
        at_risk = dashboard.get("atRiskStudents", [])[:5]
        recs = dashboard.get("aiRecommendations", [])

        context_block = (
            f"Total students: {stats.get('totalStudents', 0)}\n"
            f"Total offerings: {stats.get('totalOfferings', 0)}\n"
            f"Critical risk: {stats.get('criticalRiskCount', 0)}\n"
            f"High risk: {stats.get('highRiskCount', 0)}\n"
            f"Overall average grade: {stats.get('overallAverageGrade', 0):.1f}%\n"
            f"Overall attendance: {stats.get('overallAttendanceRate', 0):.1f}%\n"
            f"AI recommendations: {'; '.join(recs[:3])}\n"
        )

        if offerings:
            context_block += "\nTop offerings:\n"
            for o in offerings:
                context_block += (
                    f"  - {o.get('subjectName', '')} ({o.get('batchName', '')}) "
                    f"[{o.get('totalStudents', 0)} students, "
                    f"{o.get('atRiskCount', 0)} at risk, "
                    f"health: {o.get('overallHealth', '')}]\n"
                )

        lang_rule = (
            "Respond in Arabic (Egyptian dialect). Be professional yet warm. "
            "Structure the response clearly with bullet points."
            if lang == "ar" else
            "Respond in English. Be professional and data-driven."
        )

        system_prompt = f"""\
You are an AI Teaching Assistant presenting a performance dashboard to a doctor.

{lang_rule}

DASHBOARD DATA:
{context_block}

TASK: Provide a concise teaching intelligence briefing that:
1. Opens with overall class health (1 sentence)
2. Highlights the most urgent issue requiring attention
3. Mentions the most positive achievement
4. Gives 2-3 specific, actionable next steps
LENGTH: 120-180 words.
"""

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-2:]:
            r, c = turn.get("role"), str(turn.get("content", ""))[:200]
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": message})

        try:
            return await self.model_router.generate_with_messages(
                messages=messages, model_id=model_id, max_tokens=500
            ) or self._fallback_dashboard(dashboard, lang)
        except Exception as exc:
            logger.error("DoctorIntelligence._narrate_dashboard: %s", exc)
            return self._fallback_dashboard(dashboard, lang)

    async def _narrate_risk_students(
        self, students: list, message: str, history: list,
        model_id: str, lang: str
    ) -> str:
        if not students:
            return (
                "لا يوجد طلاب في خطر حالياً — وضع الفصل جيد! ✅"
                if lang == "ar" else
                "No at-risk students found — your class is doing well! ✅"
            )

        if not self.model_router:
            return self._fallback_risk(students, lang)

        critical = [s for s in students if s.get("riskLevel") == "Critical"]
        high     = [s for s in students if s.get("riskLevel") == "High"]

        summary_lines = []
        for s in students[:8]:
            name = s.get("studentName", "Unknown")
            score = s.get("riskScore", 0)
            level = s.get("riskLevel", "")
            factors = s.get("riskFactors", [])
            if isinstance(factors, str):
                try: factors = json.loads(factors)
                except: factors = [factors]
            summary_lines.append(
                f"  - {name}: Risk {score:.0f}/100 ({level}) — {', '.join(factors[:2])}"
            )

        lang_rule = (
            "Respond in Arabic (Egyptian dialect). Address each critical case specifically."
            if lang == "ar" else
            "Respond in English. Be specific and actionable."
        )

        system_prompt = f"""\
You are an AI Teaching Assistant presenting at-risk student data to a doctor.

{lang_rule}

AT-RISK STUDENTS ({len(students)} total, {len(critical)} critical, {len(high)} high):
{chr(10).join(summary_lines)}

TASK:
1. Summarize the risk situation clearly
2. Name the top 2-3 most critical students and WHY
3. Give specific, actionable intervention steps for each
4. End with a prioritized action list
LENGTH: 150-200 words.
"""
        try:
            return await self.model_router.generate_with_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                model_id=model_id, max_tokens=500
            ) or self._fallback_risk(students, lang)
        except Exception as exc:
            logger.error("DoctorIntelligence._narrate_risk_students: %s", exc)
            return self._fallback_risk(students, lang)

    async def _narrate_weak_topics(
        self, topics: list, message: str, model_id: str, lang: str
    ) -> str:
        if not topics:
            return (
                "لا توجد مواضيع ضعيفة — طلابك يفهمون المنهج بشكل جيد! 📚"
                if lang == "ar" else
                "No weak topics detected — students are grasping the material well! 📚"
            )

        if not self.model_router:
            return self._fallback_topics(topics, lang)

        topic_lines = "\n".join([
            f"  - {t.get('topicName', '')}: "
            f"{t.get('errorRate', 0):.0f}% error rate, "
            f"{t.get('affectedStudents', 0)} students affected "
            f"(severity: {t.get('severity', '')})"
            for t in topics[:6]
        ])

        lang_rule = (
            "Respond in Arabic. Give pedagogical advice."
            if lang == "ar" else
            "Respond in English. Be pedagogically specific."
        )

        system_prompt = f"""\
You are an AI Teaching Assistant analyzing topic performance for a doctor.

{lang_rule}

WEAK TOPICS:
{topic_lines}

TASK: Provide a teaching analysis that:
1. Names the most critical topics (highest error rates)
2. Suggests WHY students are struggling (common misconceptions)
3. Recommends specific teaching interventions (revision exercises, different explanation approach)
LENGTH: 120-160 words.
"""
        try:
            return await self.model_router.generate_with_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                model_id=model_id, max_tokens=400
            ) or self._fallback_topics(topics, lang)
        except Exception as exc:
            return self._fallback_topics(topics, lang)

    # ── Fallbacks ─────────────────────────────────────────────────────────

    def _fallback_dashboard(self, data: dict, lang: str) -> str:
        stats = data.get("overallStats", {})
        if lang == "ar":
            return (
                f"📊 **ملخص التدريس**\n"
                f"• إجمالي الطلاب: {stats.get('totalStudents', '—')}\n"
                f"• في خطر عالٍ/حرج: {(stats.get('criticalRiskCount', 0)) + (stats.get('highRiskCount', 0))}\n"
                f"• متوسط الدرجات: {stats.get('overallAverageGrade', 0):.1f}%\n"
                f"• متوسط الحضور: {stats.get('overallAttendanceRate', 0):.1f}%"
            )
        return (
            f"📊 **Teaching Summary**\n"
            f"• Total students: {stats.get('totalStudents', '—')}\n"
            f"• At risk: {(stats.get('criticalRiskCount', 0)) + (stats.get('highRiskCount', 0))}\n"
            f"• Avg grade: {stats.get('overallAverageGrade', 0):.1f}%\n"
            f"• Avg attendance: {stats.get('overallAttendanceRate', 0):.1f}%"
        )

    def _fallback_risk(self, students: list, lang: str) -> str:
        count = len(students)
        if lang == "ar":
            return f"⚠️ يوجد **{count} طالب** في خطر أكاديمي. يُنصح بمراجعة بياناتهم والتواصل معهم فوراً."
        return f"⚠️ **{count} students** are at academic risk. Review their profiles and reach out immediately."

    def _fallback_topics(self, topics: list, lang: str) -> str:
        if not topics:
            return "No weak topics data available." if lang == "en" else "لا توجد بيانات مواضيع متاحة."
        top = topics[0]
        if lang == "ar":
            return f"📚 أكثر موضوع صعوبة: **{top.get('topicName', '')}** بنسبة خطأ {top.get('errorRate', 0):.0f}%."
        return f"📚 Most difficult topic: **{top.get('topicName', '')}** with {top.get('errorRate', 0):.0f}% error rate."

    # ── Suggestions ───────────────────────────────────────────────────────

    def _dashboard_suggestions(self, lang: str) -> list[str]:
        if lang == "ar":
            return ["عرض الطلاب في خطر", "تحليل موضوع ضعيف", "مقارنة الفصول"]
        return ["Show at-risk students", "Analyze weak topics", "Compare classes"]

    def _risk_suggestions(self, lang: str) -> list[str]:
        if lang == "ar":
            return ["تصفية حسب المستوى الحرج فقط", "مقارنة مع الفصل السابق", "تصدير Excel"]
        return ["Filter critical risk only", "Compare with last semester", "Export to Excel"]

    def _topic_suggestions(self, lang: str) -> list[str]:
        if lang == "ar":
            return ["اقتراح تمارين تصحيحية", "جدولة جلسة مراجعة", "تحليل نتائج الامتحان"]
        return ["Suggest remedial exercises", "Schedule revision session", "Analyze exam results"]

    @staticmethod
    def _detect_lang(message: str) -> str:
        arabic_chars = sum(1 for c in message if "؀" <= c <= "ۿ")
        return "ar" if arabic_chars / max(len(message), 1) > 0.2 else "en"
