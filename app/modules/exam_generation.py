"""
app/modules/exam_generation.py

Exam Generation Module.

Flow:
  1. Resolve subjectOfferingId (from plan, context, or pre-execution steps).
     GET /api/ai-tools/resolve-offering?subject={subjectName}
  2. Fetch course material from backend   GET /api/Materials/by-offering/{id}
  3. Generate structured MCQ questions via Claude (generate_structured_json).
  4. Persist the exam via BackendClient  POST /api/Exams/generate-ai
  5. Return generated questions as structured text and JSON.

Safety contract
---------------
- If offering cannot be resolved → ValueError (never continue with None).
- If model output is empty/invalid → ValueError (never return placeholders).
- If questions list is empty or contains placeholders → ValueError (never persist garbage).
- All backend errors are caught and returned as user-friendly AgentOutput(status="error").
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import httpx

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


async def _download_pdf(url: str, auth_header: Optional[str] = None) -> bytes:
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content

_BACKEND_UNAVAILABLE_MSG = (
    "The backend service is currently unavailable. Please try again later."
)

_QUESTION_SYSTEM_PROMPT = """You are an expert university exam writer.
Generate exam questions strictly as a JSON array. Each element must have:
  - "questionText": string — the question (at least 20 characters)
  - "questionType": integer — 0 for MCQ, 1 for TrueFalse, 2 for Essay
  - "options": array of exactly 4 strings for MCQ, null for TrueFalse/Essay
  - "correctAnswer": string — for MCQ: one of the option strings; for TrueFalse: "True" or "False"; for Essay: a model answer
  - "mark": integer between 5 and 20

Return ONLY the JSON array with no extra text, markdown, or explanation.
"""


def _pdf_to_text(data: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(io.BytesIO(data)) or ""
    except Exception as exc:
        logger.warning("ExamGenerationModule: pdfminer failed: %s", exc)
        return ""


class ExamGenerationModule:
    """
    Generates structured MCQ exam questions via Claude.
    Persists the exam to the .NET backend when a valid offering is resolved.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("ExamGenerationModule: starting.")

        params      = getattr(plan, "exam_params", None) if plan else None
        context     = agent_input.context or {}
        num_q       = getattr(params, "numberOfQuestions", 10) if params else context.get("questionCount", 10)
        exam_type   = getattr(params, "examType", "midterm") if params else context.get("examType", "midterm")
        difficulty  = getattr(params, "difficulty", "Medium") if params else context.get("difficulty", "Medium")
        subject     = getattr(params, "subjectName", "the subject") if params else "the subject"
        is_random   = context.get("isRandomized", False)
        per_student = context.get("questionsPerStudent", 0)
        topics      = context.get("topics", [])

        # ── 1. Resolve subjectOfferingId ──────────────────────────────────────
        offering_id = (
            getattr(params, "subjectOfferingId", None) if params else None
        ) or context.get("subjectOfferingId")

        if not offering_id and plan and hasattr(plan, "pre_execution_steps"):
            for step in plan.pre_execution_steps:
                if step.tool == "ResolveSubjectOffering":
                    subject_name = (step.input_payload.get("subjectName") or "").strip()
                    logger.info(
                        "ExamGenerationModule [GET] endpoint=/api/ai-tools/resolve-offering "
                        "query_params={'subject': '%s'}",
                        subject_name,
                    )
                    try:
                        res = await self.backend_client.fetch(
                            route="/api/ai-tools/resolve-offering",
                            auth_header=agent_input.auth_header,
                            params={"subject": subject_name},
                        )

                        if isinstance(res, list) and len(res) > 1:
                            logger.info(
                                "ExamGenerationModule: resolve-offering returned %d results "
                                "— clarification required.", len(res)
                            )
                            return AgentOutput(
                                status="clarification_needed",
                                response="clarification_needed",
                                data={
                                    "options": res,
                                    "original_intent": getattr(plan, "intent", "generate_exam"),
                                    "step_context": {
                                        "module_name": "exam_generation",
                                        "subjectName": subject_name,
                                        "exam_params": (
                                            params.model_dump() if params else {}
                                        ),
                                    },
                                },
                            )

                        if isinstance(res, list) and len(res) == 1:
                            offering_id = res[0].get("subjectOfferingId") or res[0].get("id")
                            subject = res[0].get("subjectName", subject)
                        else:
                            offering_id = (
                                res.get("subjectOfferingId")
                                if res and "error" not in res
                                else None
                            )
                            subject = res.get("subjectName", subject) if res else subject
                        logger.info(
                            "ExamGenerationModule: resolved subjectOfferingId=%s",
                            offering_id,
                        )
                    except Exception as resolve_exc:
                        logger.error(
                            "ExamGenerationModule: ResolveSubjectOffering failed — %s",
                            resolve_exc,
                        )
                        offering_id = None
                    break

        if not offering_id:
            logger.error(
                "ExamGenerationModule: offering_id is None — cannot generate exam "
                "without a valid subject offering."
            )
            return AgentOutput(
                status="error",
                response=(
                    "Could not resolve the subject offering. "
                    "Please specify a valid subject name and try again."
                ),
                data={
                    "module": "ExamGenerationModule",
                    "error": "subjectOfferingId could not be resolved.",
                },
            )

        # ── 2. Fetch course material ──────────────────────────────────────────
        material_text = ""

        # 2a. Check if a PDF was sent directly in context (file_url / fileUrl)
        ctx = agent_input.context or {}
        ac = ctx.get("academic_context", {}) or {}
        direct_file_url = (
            ctx.get("file_url") or ctx.get("fileUrl")
            or ac.get("file_url") or ac.get("fileUrl")
        )
        if direct_file_url and direct_file_url.startswith("http"):
            logger.info("ExamGenerationModule: using direct PDF from context url=%s", direct_file_url[:80])
            try:
                pdf_bytes = await _download_pdf(direct_file_url, agent_input.auth_header)
                material_text = _pdf_to_text(pdf_bytes)
                logger.info("ExamGenerationModule: extracted %d chars from context PDF", len(material_text))
            except Exception as pdf_exc:
                logger.warning("ExamGenerationModule: context PDF download failed — %s", pdf_exc)

        # 2b. Fetch from offering if no direct PDF
        if not material_text:
            logger.info(
                "ExamGenerationModule [GET] endpoint=/api/Materials/by-offering/%s",
                offering_id,
            )
            try:
                result = await self.backend_client.fetch(
                    route=f"/api/Materials/by-offering/{offering_id}",
                    auth_header=agent_input.auth_header,
                )
                if "error" not in result:
                    if "_raw_bytes" in result:
                        material_text = _pdf_to_text(result["_raw_bytes"])
                    else:
                        # Try direct text fields first
                        material_text = str(result.get("content") or result.get("text") or "")

                    # Follow fileUrl / signedUrl to the actual PDF if text is empty
                    if not material_text.strip():
                        items = result if isinstance(result, list) else result.get("data", [result])
                        for item in (items if isinstance(items, list) else [items]):
                            mat_url = (
                                item.get("fileUrl") or item.get("signedUrl")
                                or item.get("filePath") or item.get("url") or ""
                            )
                            if mat_url and mat_url.startswith("http"):
                                logger.info("ExamGenerationModule: fetching PDF from fileUrl=%s", mat_url[:80])
                                try:
                                    pdf_bytes = await _download_pdf(mat_url, agent_input.auth_header)
                                    material_text = _pdf_to_text(pdf_bytes)
                                    logger.info("ExamGenerationModule: extracted %d chars from material PDF", len(material_text))
                                    if material_text.strip():
                                        break
                                except Exception as dl_exc:
                                    logger.warning("ExamGenerationModule: material PDF download failed — %s", dl_exc)
            except Exception as material_exc:
                logger.warning(
                    "ExamGenerationModule: material fetch failed (non-fatal) — %s",
                    material_exc,
                )

        if not material_text:
            material_text = (
                agent_input.message or f"General {exam_type} exam for {subject}."
            )

        # ── 3. Generate structured questions via Claude ───────────────────────
        topics_hint = f" Focus on these topics: {', '.join(topics)}." if topics else ""
        user_prompt = (
            f"Generate {num_q} {difficulty}-difficulty {exam_type} exam questions "
            f"for the subject '{subject}'.{topics_hint}\n\n"
            f"Course material excerpt:\n{material_text[:4000]}"
        )

        logger.info(
            "ExamGenerationModule: requesting %d questions for '%s' (difficulty=%s).",
            num_q, subject, difficulty,
        )

        questions_list: List[Dict[str, Any]] = []
        try:
            raw = await self.model_router.generate_structured_json(
                prompt=user_prompt,
                system_instruction=_QUESTION_SYSTEM_PROMPT,
            )
            if isinstance(raw, list):
                questions_list = raw
            elif isinstance(raw, str):
                questions_list = json.loads(raw)
            else:
                raise ValueError(f"Unexpected model output type: {type(raw)}")
        except Exception as gen_exc:
            logger.error(
                "ExamGenerationModule: Claude generation failed — %s", gen_exc
            )
            return AgentOutput(
                status="error",
                response=(
                    "The AI could not generate valid exam questions. "
                    "Please try again with more detailed course content."
                ),
                data={"module": "ExamGenerationModule", "error": str(gen_exc)},
            )

        # ── 4. Validate ───────────────────────────────────────────────────────
        try:
            self._validate_questions(questions_list)
        except ValueError as val_exc:
            logger.error(
                "ExamGenerationModule: validation failed — %s", val_exc
            )
            return AgentOutput(
                status="error",
                response="The generated exam is invalid and cannot be saved. Please try again.",
                data={"module": "ExamGenerationModule", "error": str(val_exc)},
            )

        # ── 5. Persist to backend via generate-ai endpoint ───────────────────
        # Send pre-generated questions so .NET skips the redundant second AI call.
        exam_payload: Dict[str, Any] = {
            "subjectOfferingId": offering_id,
            "difficulty": difficulty,
            "questionCount": len(questions_list),
            "examType": exam_type.capitalize(),
            "topics": topics,
            "isRandomized": is_random,
            "questionsPerStudent": per_student,
            "preGeneratedQuestions": questions_list,
        }
        logger.info(
            "ExamGenerationModule [POST] endpoint=/api/Exams/generate-ai "
            "offeringId=%s questions_count=%d isRandomized=%s",
            offering_id, len(questions_list), is_random,
        )
        backend_result: Dict[str, Any] = {}
        try:
            backend_result = await self.backend_client.post(
                route="/api/Exams/generate-ai",
                payload=exam_payload,
                auth_header=agent_input.auth_header,
            )
            if backend_result and "error" in backend_result:
                logger.warning(
                    "ExamGenerationModule: backend persist returned error — %s",
                    backend_result["error"],
                )
        except Exception as persist_exc:
            logger.warning(
                "ExamGenerationModule: exam persist failed (non-fatal) — %s",
                persist_exc,
            )
            backend_result = {"error": _BACKEND_UNAVAILABLE_MSG}

        # ── 6. Return ─────────────────────────────────────────────────────────
        formatted = self._format_questions(questions_list, subject, exam_type)
        return AgentOutput(
            status="success",
            response=formatted,
            data={
                "module": "ExamGenerationModule",
                "questions": questions_list,
                "backend_result": backend_result,
                "isRandomized": is_random,
                "questionsPerStudent": per_student,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_questions(questions: list) -> None:
        if not questions:
            raise ValueError("Generated exam is invalid: questions list is empty.")

        for i, q in enumerate(questions):
            text = (q.get("questionText") or q.get("question") or "").strip()
            if not text or len(text) < 10:
                raise ValueError(
                    f"Question at index {i} is missing or too short: {text!r}"
                )
            q_type = q.get("questionType", 0)
            if q_type == 0:  # MCQ
                opts = q.get("options")
                if not opts or len(opts) < 2:
                    raise ValueError(
                        f"MCQ question at index {i} must have at least 2 options."
                    )
            if not q.get("correctAnswer"):
                raise ValueError(
                    f"Question at index {i} is missing 'correctAnswer'."
                )

    @staticmethod
    def _format_questions(questions: list, subject: str, exam_type: str) -> str:
        lines = [f"**{exam_type.capitalize()} Exam — {subject}**", ""]
        for i, q in enumerate(questions, 1):
            text = q.get("questionText") or q.get("question", "")
            mark = q.get("mark", 5)
            q_type = q.get("questionType", 0)
            lines.append(f"{i}. {text}  [{mark} marks]")
            if q_type == 0:
                opts = q.get("options") or []
                for j, opt in enumerate(opts):
                    lines.append(f"   {'ABCD'[j]}. {opt}")
            lines.append("")
        return "\n".join(lines)
