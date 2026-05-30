"""
app/modules/result_query.py  —  v2.0

Result Query Module — personalized, empathetic, with follow-up suggestions.

Flow:
  1. Resolve exam/student identifiers from context.
  2. Fetch results from backend (exam-specific OR GPA summary).
  3. Build a personalized explanation using role + academic profile.
  4. Return with contextually-aware follow-up suggestions.

Upgrade from v1:
  - Role-specific system prompts (student vs doctor vs admin).
  - Academic profile injection (weak subjects, GPA trend).
  - Follow-up suggestion generation via entity_tracker.
  - Better empty-data handling (explains why, suggests next step).
  - Personalization using student name from academic_context.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


# ── Role-aware system prompts ─────────────────────────────────────────────────

_SYSTEM_STUDENT = """\
أنت مرشد أكاديمي ودود ومتفهم تتكلم مع طالب جامعي.
- اشرح النتائج بوضوح وبلغة بسيطة.
- ابرز نقاط القوة أولاً، ثم مجالات التحسين.
- لو النتيجة ضعيفة → كن داعماً وصادقاً في نفس الوقت. اقترح خطوات عملية محددة.
- لو النتيجة ممتازة → بارك واقترح كيف يحافظ عليها.
- ممنوع جمل روبوتية مثل "يسعدني" أو "لا تتردد في طلب المساعدة".
- استخدم اسم الطالب لو متوفر.
- اللغة: نفس لغة الطالب (عربي بعربي، إنجليزي بإنجليزي).
"""

_SYSTEM_DOCTOR = """\
أنت مساعد تحليلي أكاديمي لدكتور جامعي يراجع نتائج طلابه.
- قدّم تحليلاً مختصراً للنتائج مع الأرقام الرئيسية.
- إذا كانت النتائج ضعيفة جماعياً، اقترح تحليلاً تشخيصياً (هل الأسئلة صعبة؟ هل الطلاب مستعدون؟).
- قدّم insights عملية وليس فقط أرقام.
- نبرة مهنية كزميل أكاديمي.
"""

_SYSTEM_ADMIN = """\
أنت محلل أكاديمي لإداري جامعي يراجع بيانات النتائج.
- عرض ملخص تنفيذي مع المؤشرات الرئيسية.
- ابرز أي anomalies أو تنبيهات تستحق الاهتمام.
- نبرة مختصرة واحترافية.
"""


def _get_system_prompt(role: str) -> str:
    role = (role or "student").lower()
    if role == "doctor":
        return _SYSTEM_DOCTOR
    if role in ("admin", "superadmin"):
        return _SYSTEM_ADMIN
    return _SYSTEM_STUDENT


def _build_result_prompt(
    message: str,
    result_data: Dict[str, Any],
    academic_ctx: Dict[str, Any],
    role: str,
    personalized_context: str = "",
) -> str:
    """Build a rich, personalized prompt from result data + academic context."""
    student_name = (
        academic_ctx.get("studentName")
        or academic_ctx.get("fullName")
        or academic_ctx.get("name")
        or ""
    )

    lines = [f"سؤال المستخدم: {message}"]

    if student_name and role == "student":
        lines.append(f"اسم الطالب: {student_name}")

    if personalized_context:
        lines.append(f"\nالملف الأكاديمي للطالب:\n{personalized_context}")

    if result_data:
        try:
            data_str = json.dumps(result_data, ensure_ascii=False, default=str, indent=2)
            if len(data_str) > 3000:
                data_str = data_str[:3000] + "\n... [truncated]"
        except Exception:
            data_str = str(result_data)[:3000]
        lines.append(f"\nبيانات النتائج من السيستم:\n{data_str}")
    else:
        lines.append(
            "\nملاحظة: لم تُسترجع بيانات نتائج محددة من السيستم حالياً. "
            "اشرح هذا للمستخدم بوضوح واقترح كيف يحصل على بياناته."
        )

    lines.append(
        "\nبناءً على البيانات أعلاه، قدّم تحليلاً شخصياً ومفيداً. "
        "كل رقم تذكره يجب أن يأتي من البيانات، لا من تخمين."
    )

    return "\n".join(lines)


class ResultQueryModule:
    """
    Fetches exam/grade results from the .NET backend and explains them
    in a personalized, role-appropriate, empathetic way.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("ResultQueryModule v2: starting.")

        context      = agent_input.context or {}
        role         = context.get("role") or "student"
        academic_ctx = context.get("academic_context") or {}
        personalized = context.get("preferences", {}).get("personalized_context", "")

        # -- 1. Resolve identifiers -------------------------------------------
        exam_id    = context.get("examId") or context.get("exam_id")
        student_id = agent_input.user_id

        # -- 2. Fetch results from backend ------------------------------------
        result_data: Dict[str, Any] = {}

        if exam_id:
            result = await self.backend_client.fetch(
                route=f"/api/Exams/{exam_id}/results",
                auth_header=agent_input.auth_header,
                params={"studentId": student_id} if student_id else None,
            )
            if not result.get("_error"):
                result_data = result
            else:
                logger.warning("ResultQueryModule: exam results error — %s", result)
        else:
            # Fetch GPA summary as fallback
            result = await self.backend_client.fetch(
                route="/api/Gpa/my-gpa",
                auth_header=agent_input.auth_header,
                params={},
            )
            if result and not result.get("_error"):
                result_data = result

        # -- 3. Build personalized explanation with LLM ----------------------
        selected_model = context.get("selected_model") or "openai/gpt-4o-mini"

        prompt = _build_result_prompt(
            message=agent_input.message,
            result_data=result_data,
            academic_ctx=academic_ctx,
            role=role,
            personalized_context=personalized,
        )

        system_prompt = _get_system_prompt(role)

        explanation = await self.model_router.generate(
            prompt=prompt,
            system_instruction=system_prompt,
            model_id=selected_model,
        )

        # Fallback: TinyLlama if cloud fails
        if not explanation and not selected_model.startswith("hf/"):
            logger.info("ResultQueryModule: cloud generation failed, trying hf fallback")
            explanation = await self.model_router.generate(
                prompt=prompt,
                system_instruction=system_prompt,
                model_id="hf/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            )

        if not explanation:
            explanation = "تعذّر استرجاع شرح النتائج. يمكنك مراجعة نتائجك مباشرة من لوحة التحكم."

        # -- 4. Generate follow-up suggestions --------------------------------
        suggestions = []
        try:
            from app.core.entity_tracker import infer_followup_suggestions
            entities = context.get("academic_context", {}).get("conversation_entities", {})
            last_intent = context.get("last_intent") or "result_query"
            suggestions = infer_followup_suggestions(entities, last_intent, role)
        except Exception as exc:
            logger.debug("ResultQueryModule: followup suggestions failed (non-fatal) — %s", exc)

        return AgentOutput(
            status="success",
            response=explanation,
            data={
                "module": "ResultQueryModule",
                "version": "v2",
                "exam_id": exam_id,
                "has_data": bool(result_data),
                "model_used": selected_model,
                "suggestions": suggestions,
            },
        )
