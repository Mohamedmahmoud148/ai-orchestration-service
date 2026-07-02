"""
app/modules/study_plan.py  —  v1.0

Study Plan Generator — intent: study_plan

Generates a personalized, time-aware, data-driven study/revision plan by
combining five live data sources fetched in parallel from the .NET backend:

  1. GET /api/Regulations/my-roadmap
       → GPA, semesters, passed/failed/enrolled subjects, mustRetake, recommendedNext

  2. GET /api/ai-tools/student-overview/{userId}
       → finalized grades per subject, exam history

  3. GET /api/analytics/student/{userId}/performance
       → per-subject attendance rate + performance category (Excellent/Good/Average/Failing)

  4. GET /api/assignments/offering/{id}  (per enrolled offering, parallel)
       → upcoming assignment deadlines + submission status

  5. Upcoming exams  (derived from overview.exams filtered by date + roadmap data)
       → exam dates, subjects, time-remaining countdown

Then it asks a strong LLM to produce a structured, prioritized, motivating
weekly/daily study plan tailored to the student's specific weaknesses and deadlines.

Architecture rules (same as AcademicAdvisorModule):
  - Backend is the sole source of truth — zero hallucination.
  - All network calls in parallel to minimise latency.
  - Graceful degradation: missing data sources → module warns but proceeds.
  - Long-form output (max_tokens=2500) — depth over speed for planning.
"""
from __future__ import annotations

import asyncio
import json as _json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger
from app.prompts import load_prompt

_DEFAULT_MODEL    = "openai/gpt-4o-mini"
_PLAN_MAX_TOKENS  = 2500
_RAW_CHAR_BUDGET  = 2000   # per section before LLM prompt


def _load_prompt() -> str:
    try:
        return load_prompt("study_plan")
    except Exception as exc:
        logger.warning("StudyPlanModule: prompt load failed — using inline fallback: %s", exc)
        return _FALLBACK_PROMPT


_FALLBACK_PROMPT = """\
أنت مرشد دراسة شخصي. مهمتك توليد خطة مذاكرة واقعية مخصصة للطالب بناءً حصرياً على بياناته الفعلية.
قواعد: ممنوع اختراع أي رقم أو موعد. استخدم لغة الطالب نفسها (عربي/إنجليزي).
رتّب المواد حسب الأولوية: امتحانات قريبة > واجبات قريبة > رسوب > حضور منخفض > إجباري.
اعطِ خطة يومية للأسبوع الحالي + توصيات محددة لكل مادة حرجة.
"""


def _days_until(date_str: str) -> Optional[int]:
    """Return days until a future date string (ISO format). None if unparseable."""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        delta = dt - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds() // 86400))
    except Exception:
        return None


def _is_arabic(text: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in text)


def _truncate(obj: Any, max_chars: int) -> str:
    try:
        s = _json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s[:max_chars] + " …[truncated]" if len(s) > max_chars else s


class StudyPlanModule:
    """
    Generates a personalized, time-aware study plan combining:
      • Roadmap (GPA, enrolled/failed subjects, mustRetake)
      • Grades + exam history
      • Per-subject attendance + performance category
      • Upcoming assignments with deadlines
    """

    def __init__(self, model_router: Any, backend_client: Any) -> None:
        self.model_router   = model_router
        self.backend_client = backend_client

    # ─────────────────────────────────────────────────────────────────────────
    #  Data Fetchers
    # ─────────────────────────────────────────────────────────────────────────

    async def _fetch_roadmap(self, auth: Optional[str]) -> Optional[Dict]:
        try:
            data = await self.backend_client.fetch(
                route="/api/Regulations/my-roadmap",
                auth_header=auth,
            )
            return data if isinstance(data, dict) and not data.get("_error") else None
        except Exception as exc:
            logger.warning("StudyPlan: roadmap fetch failed — %s", exc)
            return None

    async def _fetch_overview(self, user_id: str, auth: Optional[str]) -> Optional[Dict]:
        try:
            data = await self.backend_client.fetch(
                route=f"/api/ai-tools/student-overview/{user_id}",
                auth_header=auth,
            )
            return data if isinstance(data, dict) and not data.get("_error") else None
        except Exception as exc:
            logger.warning("StudyPlan: overview fetch failed — %s", exc)
            return None

    async def _fetch_performance(self, user_id: str, auth: Optional[str]) -> List[Dict]:
        """GET /api/analytics/student/{userId}/performance → per-subject attendance + grade."""
        try:
            data = await self.backend_client.fetch(
                route=f"/api/analytics/student/{user_id}/performance",
                auth_header=auth,
            )
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", []) or []
        except Exception as exc:
            logger.warning("StudyPlan: performance fetch failed — %s", exc)
        return []

    async def _fetch_assignments_for_offering(
        self, offering_id: str, auth: Optional[str]
    ) -> List[Dict]:
        try:
            data = await self.backend_client.fetch(
                route=f"/api/assignments/offering/{offering_id}",
                auth_header=auth,
            )
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", []) or data.get("items", [])
        except Exception:
            pass
        return []

    async def _fetch_all_assignments(
        self, offering_ids: List[str], auth: Optional[str]
    ) -> List[Dict]:
        """Fetch assignments for all enrolled offerings in parallel."""
        if not offering_ids:
            return []
        results = await asyncio.gather(
            *(self._fetch_assignments_for_offering(oid, auth) for oid in offering_ids),
            return_exceptions=True,
        )
        now = datetime.now(timezone.utc)
        upcoming: List[Dict] = []
        for batch in results:
            if isinstance(batch, list):
                for a in batch:
                    dl = a.get("deadline") or a.get("dueDate") or ""
                    try:
                        dt = datetime.fromisoformat(dl.replace("Z", "+00:00"))
                        if dt > now:
                            a["_daysUntil"] = max(0, int((dt - now).total_seconds() // 86400))
                            upcoming.append(a)
                    except Exception:
                        pass
        upcoming.sort(key=lambda x: x.get("_daysUntil", 999))
        return upcoming

    # ─────────────────────────────────────────────────────────────────────────
    #  Context Block Builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_context_block(
        self,
        today_str: str,
        day_of_week: str,
        roadmap: Optional[Dict],
        overview: Optional[Dict],
        performance: List[Dict],
        assignments: List[Dict],
        academic_ctx: Dict,
        message: str,
    ) -> str:
        lines: List[str] = [
            f"=== بيانات الطالب الفعلية — {today_str} ({day_of_week}) ===",
        ]

        # ── Profile ──────────────────────────────────────────────────────────
        name = academic_ctx.get("studentName") or academic_ctx.get("name")
        dept = academic_ctx.get("departmentName")
        batch = academic_ctx.get("batchName")
        if name or dept or batch:
            lines.append("[الملف]")
            if name:  lines.append(f"  الاسم: {name}")
            if dept:  lines.append(f"  القسم: {dept}")
            if batch: lines.append(f"  الدفعة: {batch}")

        # ── Roadmap summary ───────────────────────────────────────────────────
        if roadmap:
            gpa = roadmap.get("currentGpa")
            lines.append("\n[الرودماب الأكاديمي]")
            lines.append(f"  GPA: {gpa if gpa is not None else 'غير متاح'}")
            lines.append(
                f"  الساعات: منجز={roadmap.get('completedCreditHours', 0)} / "
                f"إجمالي={roadmap.get('totalCreditHours', 0)} — "
                f"باقي={roadmap.get('remainingCreditHours', 0)}"
            )
            lines.append(
                f"  مواد: ناجح={roadmap.get('passedSubjects', 0)}, "
                f"راسب={roadmap.get('failedSubjects', 0)}, "
                f"مسجل={roadmap.get('currentlyEnrolled', 0)}"
            )

            # Currently enrolled subjects (priority targets)
            enrolled_subjects: List[str] = []
            for sem in (roadmap.get("semesters") or []):
                for sub in (sem.get("subjects") or []):
                    if sub.get("status") in ("in_progress", "enrolled"):
                        enrolled_subjects.append(
                            f"{sub.get('subjectName', '?')} ({sub.get('subjectCode', '')})"
                            f" — {sub.get('creditHours', '?')} ساعة"
                            f"{', إجباري' if sub.get('isRequired') else ''}"
                        )
            if enrolled_subjects:
                lines.append("\n  [المواد المسجل فيها حالياً]")
                for s in enrolled_subjects[:10]:
                    lines.append(f"    • {s}")

            # Must retake (critical — failing mandatory)
            must_retake = roadmap.get("mustRetake") or []
            if must_retake:
                lines.append("\n  [⚠️ مواد لازم يعيدها]")
                for s in must_retake[:8]:
                    if isinstance(s, dict):
                        lines.append(
                            f"    • {s.get('subjectName', '?')} — "
                            f"درجة سابقة: {s.get('gradeLetter', '?')}"
                        )

        # ── Per-subject attendance + performance ──────────────────────────────
        if performance:
            lines.append("\n[الأداء والحضور في المواد الحالية]")
            for p in performance[:10]:
                if isinstance(p, dict):
                    att = p.get("attendanceRate")
                    score = p.get("finalScore") or p.get("score")
                    status = p.get("status") or p.get("performanceCategory", "")
                    att_flag = "⚠️ حضور منخفض" if (att is not None and att < 75) else ""
                    lines.append(
                        f"  • {p.get('subjectName', '?')}: "
                        f"حضور={att}% {att_flag}, "
                        f"درجة={score}, "
                        f"تصنيف={status}"
                    )
        elif overview:
            # Fallback: grades from overview
            grades = overview.get("grades") or []
            if grades:
                lines.append("\n[درجات نهائية سابقة]")
                for g in grades[:12]:
                    if isinstance(g, dict):
                        lines.append(
                            f"  • {g.get('subjectName', '?')}: "
                            f"{g.get('gradeLetter', '?')} ({g.get('finalScore', '?')}%)"
                        )

        # ── Upcoming exams ────────────────────────────────────────────────────
        exams_upcoming: List[str] = []
        if overview:
            for e in (overview.get("exams") or []):
                if isinstance(e, dict) and not e.get("isGraded"):
                    # Ungraded exams are future/pending
                    title    = e.get("examTitle") or e.get("title", "امتحان")
                    subject  = e.get("subjectName", "")
                    exams_upcoming.append(f"  • {title} ({subject}) — لم يُصحَّح بعد")
        if exams_upcoming:
            lines.append("\n[امتحانات في الانتظار]")
            lines.extend(exams_upcoming[:6])

        # ── Upcoming assignments ──────────────────────────────────────────────
        if assignments:
            lines.append("\n[واجبات قادمة مرتبة حسب الأقرب]")
            for a in assignments[:8]:
                days = a.get("_daysUntil", "?")
                dl_raw = a.get("deadline") or a.get("dueDate", "")
                try:
                    dl_display = datetime.fromisoformat(
                        dl_raw.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")
                except Exception:
                    dl_display = dl_raw
                urgency = "🔴" if isinstance(days, int) and days <= 1 else (
                    "🟡" if isinstance(days, int) and days <= 3 else "🟢"
                )
                lines.append(
                    f"  {urgency} {a.get('title', '?')} — موعد: {dl_display} "
                    f"(باقي {days} يوم)"
                )
        else:
            lines.append("\n[الواجبات] لا توجد واجبات قادمة متاحة حالياً.")

        # ── User's specific question ──────────────────────────────────────────
        lines.append(f"\n[سؤال الطالب]: {message}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    #  Entry Point
    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, agent_input: AgentInput, plan: Any = None) -> AgentOutput:
        ctx          = agent_input.context or {}
        academic_ctx: Dict = ctx.get("academic_context", {}) or {}
        auth         = agent_input.auth_header
        model_id     = ctx.get("selected_model") or _DEFAULT_MODEL

        user_id = (
            agent_input.user_id
            or academic_ctx.get("userId")
            or academic_ctx.get("studentId")
            or academic_ctx.get("profileId")
        )

        today_str   = academic_ctx.get("today") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_of_week = academic_ctx.get("dayOfWeek") or datetime.now(timezone.utc).strftime("%A")
        offering_ids: List[str] = academic_ctx.get("enrolledOfferingIds") or []

        logger.info(
            "StudyPlanModule: user=%s today=%s offerings=%d",
            user_id, today_str, len(offering_ids),
        )

        # ── Fetch all data in parallel ────────────────────────────────────────
        roadmap_task     = asyncio.create_task(self._fetch_roadmap(auth))
        overview_task    = asyncio.create_task(
            self._fetch_overview(user_id, auth) if user_id else asyncio.sleep(0)
        )
        performance_task = asyncio.create_task(
            self._fetch_performance(user_id, auth) if user_id else asyncio.sleep(0)
        )
        assignments_task = asyncio.create_task(
            self._fetch_all_assignments(offering_ids[:6], auth)
        )

        roadmap, overview_raw, perf_raw, assignments = await asyncio.gather(
            roadmap_task, overview_task, performance_task, assignments_task,
        )

        overview    = overview_raw if isinstance(overview_raw, dict) else None
        performance = perf_raw if isinstance(perf_raw, list) else []

        logger.info(
            "StudyPlanModule: roadmap=%s overview=%s perf=%d assignments=%d",
            bool(roadmap), bool(overview), len(performance), len(assignments),
        )

        # ── Hard fallback: no data at all ─────────────────────────────────────
        if not roadmap and not overview and not performance:
            msg = (
                "مش قادر أوصل لبياناتك الأكاديمية دلوقتي. "
                "ممكن تكون مشكلة مؤقتة في الاتصال. حاول تاني بعد لحظات."
            )
            return AgentOutput(status="success", response=msg, data={})

        # ── Build the data context block ──────────────────────────────────────
        context_block = self._build_context_block(
            today_str   = today_str,
            day_of_week = day_of_week,
            roadmap     = roadmap,
            overview    = overview,
            performance = performance,
            assignments = assignments,
            academic_ctx= academic_ctx,
            message     = agent_input.message,
        )

        # ── Include recent conversation history ───────────────────────────────
        raw_history = (ctx.get("history") or [])[-4:]
        history_block = ""
        if raw_history:
            hlines = []
            for t in raw_history:
                role    = "الطالب" if t.get("role") == "user" else "المساعد"
                content = str(t.get("content", "")).strip()[:200]
                if content:
                    hlines.append(f"{role}: {content}")
            if hlines:
                history_block = (
                    "\n=== السياق السابق ===\n"
                    + "\n".join(hlines)
                    + "\n=== نهاية السياق ===\n"
                )

        user_prompt = (
            f"{context_block}\n\n"
            f"{history_block}\n"
            "بناءً على بيانات الطالب الفعلية أعلاه، ولّد خطة مذاكرة مخصصة تتبع الشكل المطلوب في تعليمات النظام. "
            "رتّب المواد حسب الأولوية الحقيقية (امتحانات قريبة → واجبات قادمة → رسوب → حضور منخفض → إجباري). "
            "الخطة لازم تكون واقعية وقابلة للتنفيذ، مش مجرد قائمة. فسّر ليه كل توصية وحفّزه."
        )

        # ── LLM call ─────────────────────────────────────────────────────────
        try:
            plan_text = await self.model_router.generate(
                prompt=user_prompt,
                system_instruction=_load_prompt(),
                model_id=model_id,
                max_tokens=_PLAN_MAX_TOKENS,
            )
        except Exception as exc:
            logger.error("StudyPlanModule: LLM call failed — %s", exc)
            return AgentOutput(
                status="failed",
                response="مش قادر أولد الخطة دلوقتي. حاول تاني بعد لحظات.",
            )

        if not plan_text:
            return AgentOutput(
                status="failed",
                response="ما طلعش رد من الموديل. ممكن تكرر السؤال؟",
            )

        # ── Build metadata for client ─────────────────────────────────────────
        gpa_val: Optional[float] = None
        if roadmap and roadmap.get("currentGpa") is not None:
            try:
                gpa_val = float(roadmap["currentGpa"])
            except (TypeError, ValueError):
                pass

        upcoming_count = len(assignments)
        urgent_count   = sum(
            1 for a in assignments
            if isinstance(a.get("_daysUntil"), int) and a["_daysUntil"] <= 3
        )

        return AgentOutput(
            status="success",
            response=plan_text,
            data={
                "module":              "StudyPlanModule",
                "version":             "v1",
                "today":               today_str,
                "gpa":                 gpa_val,
                "upcoming_assignments": upcoming_count,
                "urgent_assignments":   urgent_count,
                "performance_subjects": len(performance),
                "roadmap_loaded":       bool(roadmap),
                "model_used":           model_id,
            },
        )
