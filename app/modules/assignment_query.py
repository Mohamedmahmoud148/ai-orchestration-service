"""
app/modules/assignment_query.py

Assignment Query Module — shows students (or doctors) their assignments,
deadlines, and submission status by fetching live data from the .NET backend.

Intent: assignment_query

Flow:
  1. Extract subjectOfferingId from academic_context.
  2. Call GET /api/assignments/offering/{offeringId}.
  3. For each assignment, fetch submission status via GET /api/assignments/{id}/my-submission.
  4. Format results with deadline countdown and submission status.

Module is registered in:
  - executor._MODULE_CLASS_MAP    as "assignment_query"
  - planner.VALID_INTENTS         as "assignment_query"
  - rbac._STUDENT_ALLOWED etc.    as "assignment_query"
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


def _is_arabic(text: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in text)


def _time_until_ar(deadline_str: str) -> str:
    """Arabic time-until string."""
    try:
        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = deadline - now
        if delta.total_seconds() < 0:
            return "انتهى الموعد"
        hours = int(delta.total_seconds() // 3600)
        if hours < 2:
            minutes = int(delta.total_seconds() // 60)
            return f"باقي {minutes} دقيقة"
        if hours < 24:
            return f"باقي {hours} ساعة"
        return f"باقي {hours // 24} يوم"
    except Exception:
        return deadline_str


def _time_until_en(deadline_str: str) -> str:
    """English time-until string."""
    try:
        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = deadline - now
        if delta.total_seconds() < 0:
            return "overdue"
        hours = int(delta.total_seconds() // 3600)
        if hours < 2:
            return f"{int(delta.total_seconds()//60)} min left"
        if hours < 24:
            return f"{hours}h left"
        return f"{hours // 24}d left"
    except Exception:
        return deadline_str


class AssignmentQueryModule:
    """Fetches and narrates assignment information for the authenticated user."""

    intent = "assignment_query"

    def __init__(self, model_router: Any, backend_client: Any) -> None:
        self.model_router = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan: Any = None) -> AgentOutput:
        ctx = agent_input.context or {}
        academic_ctx: dict = ctx.get("academic_context", {})
        role: str = ctx.get("role", "student").lower()
        auth_header: str | None = ctx.get("auth_header")
        arabic = _is_arabic(agent_input.message)

        offering_id: str | None = academic_ctx.get("subjectOfferingId")

        # ── Step 1: Fetch assignments for the offering ────────────────────
        assignments: list[dict] = []
        if offering_id:
            try:
                data = await self.backend_client.fetch(
                    route=f"/api/assignments/offering/{offering_id}",
                    auth_header=auth_header,
                    params={},
                )
                if isinstance(data, list):
                    assignments = data
                elif isinstance(data, dict):
                    assignments = data.get("data", []) or data.get("items", [])
            except Exception as exc:
                logger.warning("AssignmentQueryModule: fetch assignments failed — %s", exc)

        if not assignments:
            if arabic:
                if not offering_id:
                    return AgentOutput(
                        status="success",
                        response="📚 محتاج تحدد الكورس المحدد عشان أجيب واجباته. ممكن تقولي اسم المادة؟",
                        data={},
                    )
                return AgentOutput(
                    status="success",
                    response="✅ مفيش واجبات متاحة دلوقتي في هذا الكورس.",
                    data={},
                )
            else:
                if not offering_id:
                    return AgentOutput(
                        status="success",
                        response="📚 Please specify which course you'd like to see assignments for.",
                        data={},
                    )
                return AgentOutput(
                    status="success",
                    response="✅ No assignments found for this course.",
                    data={},
                )

        # ── Step 2: For students, fetch submission statuses concurrently ──
        submission_map: dict[str, dict | None] = {}
        if role == "student" and auth_header:
            async def _fetch_submission(aid: str) -> tuple[str, dict | None]:
                try:
                    data = await self.backend_client.fetch(
                        route=f"/api/assignments/{aid}/my-submission",
                        auth_header=auth_header,
                        params={},
                    )
                    return aid, data
                except Exception:
                    return aid, None

            results = await asyncio.gather(*[
                _fetch_submission(a["id"])
                for a in assignments
                if "id" in a
            ])
            submission_map = dict(results)

        # ── Step 3: Build formatted response ─────────────────────────────
        now = datetime.now(timezone.utc)
        lines: list[str] = []

        for a in assignments:
            title = a.get("title", "—")
            desc = a.get("description", "")
            deadline_raw = a.get("deadline") or a.get("dueDate") or ""
            max_grade = a.get("maxGrade", 100)
            allow_late = a.get("allowLateSubmission", False)

            is_overdue = False
            deadline_display = deadline_raw
            time_str = ""
            try:
                dl = datetime.fromisoformat(deadline_raw.replace("Z", "+00:00"))
                is_overdue = dl < now
                deadline_display = dl.strftime("%Y-%m-%d %H:%M UTC")
                time_str = _time_until_ar(deadline_raw) if arabic else _time_until_en(deadline_raw)
            except Exception:
                pass

            sub = submission_map.get(a.get("id", ""))

            if arabic:
                if sub:
                    grade = sub.get("grade")
                    status_icon = "✅"
                    status_text = f"تم التسليم"
                    if grade is not None:
                        status_text += f" | الدرجة: **{grade}/{max_grade}**"
                    feedback = sub.get("feedback") or sub.get("aiFeedback", "")
                    if feedback:
                        status_text += f"\n  💬 {feedback[:120]}..."
                elif is_overdue:
                    status_icon = "⚠️"
                    status_text = "متأخر — لم يُسلَّم"
                    if allow_late:
                        status_text += " (التسليم المتأخر مقبول)"
                else:
                    status_icon = "📝"
                    status_text = f"لم يُسلَّم بعد — {time_str}"

                entry = f"{status_icon} **{title}**"
                if desc:
                    entry += f"\n  📋 {desc[:100]}"
                entry += f"\n  📅 الموعد: {deadline_display}"
                entry += f"\n  {status_text}"
            else:
                if sub:
                    grade = sub.get("grade")
                    status_icon = "✅"
                    status_text = "Submitted"
                    if grade is not None:
                        status_text += f" | Grade: **{grade}/{max_grade}**"
                    feedback = sub.get("feedback") or sub.get("aiFeedback", "")
                    if feedback:
                        status_text += f"\n  💬 {feedback[:120]}..."
                elif is_overdue:
                    status_icon = "⚠️"
                    status_text = "Overdue — not submitted"
                    if allow_late:
                        status_text += " (late submission allowed)"
                else:
                    status_icon = "📝"
                    status_text = f"Not submitted — {time_str}"

                entry = f"{status_icon} **{title}**"
                if desc:
                    entry += f"\n  📋 {desc[:100]}"
                entry += f"\n  📅 Due: {deadline_display}"
                entry += f"\n  {status_text}"

            lines.append(entry)

        count = len(assignments)
        if arabic:
            header = f"عندك **{count}** واجب في هذا الكورس:\n\n"
        else:
            header = f"You have **{count}** assignment(s) in this course:\n\n"

        return AgentOutput(
            status="success",
            response=header + "\n\n---\n\n".join(lines),
            data={"assignments": assignments, "submission_map": submission_map},
        )
