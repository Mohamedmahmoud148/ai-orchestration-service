"""
app/modules/exam_generation.py

Exam Generation Module.

Flow:
  1. Resolve subjectOfferingId (from plan, context, or pre-execution steps).
     GET /api/ai-tools/resolve-offering?subject={subjectName}
  2. Fetch course material from backend   GET /api/Materials/by-offering/{id}
  3. Generate questions via Flan-T5  (model_service.generate_questions).
  4. Persist the exam via BackendClient  POST /api/Exams
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
import re
from typing import Any, Dict, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger

# Patterns that identify placeholder / fake question text that must be rejected.
_PLACEHOLDER_PATTERNS = re.compile(
    r"""
    ^                               # start of line
    (question\s+\d+                 # "Question 1", "question 2 about ..."
    |additional\s+question          # "Additional question 3."
    |q\s*\d+\s*[.:\-]              # "Q1:", "Q 2 -"
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_BACKEND_UNAVAILABLE_MSG = (
    "The backend service is currently unavailable. Please try again later."
)


def _pdf_to_text(data: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(io.BytesIO(data)) or ""
    except Exception as exc:
        logger.warning("ExamGenerationModule: pdfminer failed: %s", exc)
        return ""


class ExamGenerationModule:
    """
    Generates exam questions from course material using Flan-T5.
    Persists the exam to the .NET backend when a valid offering is resolved.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("ExamGenerationModule: starting.")

        params    = getattr(plan, "exam_params", None) if plan else None
        context   = agent_input.context or {}
        num_q     = getattr(params, "numberOfQuestions", 5) if params else 5
        exam_type = getattr(params, "examType", "midterm") if params else "midterm"
        subject   = getattr(params, "subjectName", "the subject") if params else "the subject"

        # ── 1. Resolve subjectOfferingId ──────────────────────────────────────
        offering_id = (
            getattr(params, "subjectOfferingId", None) if params else None
        ) or context.get("subjectOfferingId")

        if not offering_id and plan and hasattr(plan, "pre_execution_steps"):
            for step in plan.pre_execution_steps:
                if step.tool == "ResolveSubjectOffering":
                    subject_name = step.input_payload.get("subjectName", "").strip()
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

                        # ── Disambiguation: backend returned multiple matches ──────────
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

                        # ── Single result or dict ────────────────────────────────────
                        if isinstance(res, list) and len(res) == 1:
                            offering_id = res[0].get("subjectOfferingId") or res[0].get("id")
                        else:
                            offering_id = (
                                res.get("subjectOfferingId")
                                if res and "error" not in res
                                else None
                            )
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

        # CRITICAL: Never continue without a valid offering ID.
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
                    material_text = str(
                        result.get("content") or result.get("text") or ""
                    )
        except Exception as material_exc:
            # Material fetch is non-fatal — fall back to the user message.
            logger.warning(
                "ExamGenerationModule: material fetch failed (non-fatal) — %s",
                material_exc,
            )

        if not material_text:
            material_text = (
                agent_input.message or f"General {exam_type} exam for {subject}."
            )

        # ── 3. Generate questions via ModelRouter ─────────────────────────────
        selected_model = context.get("selected_model")
        target_model = selected_model
        if not selected_model or not selected_model.startswith("hf/"):
            target_model = "hf/google/flan-t5-base"
            logger.info(
                "ExamGenerationModule: overriding pipeline model '%s' "
                "with task-optimized 'hf/google/flan-t5-base'.",
                selected_model or "default",
            )

        logger.info(
            "ExamGenerationModule: generating %d questions for '%s' using %s.",
            num_q, subject, target_model,
        )
        raw_questions = await self.model_router.generate_questions(
            material_text, num_questions=num_q, model_id=target_model
        )

        # ── Parse raw model output — raises ValueError on any invalid output ──
        try:
            questions_list = self._parse_questions(raw_questions, num_q, subject)
        except ValueError as parse_exc:
            logger.error(
                "ExamGenerationModule: question parsing failed — %s", parse_exc
            )
            return AgentOutput(
                status="error",
                response=(
                    "The AI model could not generate valid exam questions from "
                    "the provided material. Please try again with more detailed "
                    "course content."
                ),
                data={"module": "ExamGenerationModule", "error": str(parse_exc)},
            )

        # ── 4. Pre-persist validation ─────────────────────────────────────────
        try:
            self._validate_questions(questions_list)
        except ValueError as val_exc:
            logger.error(
                "ExamGenerationModule: pre-persist validation failed — %s", val_exc
            )
            return AgentOutput(
                status="error",
                response=(
                    "The generated exam is invalid or empty and cannot be saved. "
                    "Please try again."
                ),
                data={"module": "ExamGenerationModule", "error": str(val_exc)},
            )

        # ── 5. Persist to backend ─────────────────────────────────────────────
        exam_payload: Dict[str, Any] = {
            "subjectOfferingId": offering_id,
            "examData": {
                "title": f"{exam_type.capitalize()} Exam — {subject}",
                "questions": questions_list,
            },
            "handleStudentRandomization": False,
        }
        logger.info(
            "ExamGenerationModule [POST] endpoint=/api/Exams "
            "offeringId=%s questions_count=%d",
            offering_id,
            len(questions_list),
        )
        backend_result: Dict[str, Any] = {}
        try:
            backend_result = await self.backend_client.post(
                route="/api/Exams",
                payload=exam_payload,
                auth_header=agent_input.auth_header,
            )
            logger.info(
                "ExamGenerationModule [POST] /api/Exams response=%s",
                backend_result,
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
                "model_used": target_model,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_questions(raw: str, num_q: int, subject: str) -> list:
        """Parse raw model output into a validated list of question dicts.

        Raises
        ------
        ValueError
            - If ``raw`` is None, empty, or whitespace-only.
            - If no non-empty lines can be extracted after stripping.
            - If every extracted line looks like a placeholder.
            Fake / placeholder questions are NEVER returned.
        """
        if not raw or not raw.strip():
            raise ValueError(
                "Model returned an empty response. "
                "Cannot generate exam questions without valid model output."
            )

        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        if not lines:
            raise ValueError(
                "No question lines could be extracted from model output. "
                "The response contained only blank lines."
            )

        # Filter out lines that are too short to be real questions (< 10 chars)
        valid_lines = [l for l in lines if len(l) >= 10]
        if not valid_lines:
            raise ValueError(
                f"All extracted lines are too short to be valid questions "
                f"(minimum 10 characters). Got: {lines[:3]}"
            )

        # Reject if every line matches a known placeholder pattern
        placeholder_count = sum(
            1 for l in valid_lines if _PLACEHOLDER_PATTERNS.match(l)
        )
        if placeholder_count == len(valid_lines):
            raise ValueError(
                "Model output consists entirely of placeholder text. "
                "Cannot persist fake questions to the backend."
            )

        result = []
        for line in valid_lines[:num_q]:
            result.append({"question": line, "answer": "", "marks": 5})

        return result

    @staticmethod
    def _validate_questions(questions: list) -> None:
        """Strict pre-persist gate — raises ValueError on any violation.

        Checks
        ------
        - List must not be empty.
        - Every item must have a non-empty ``question`` field.
        - No item may match the placeholder pattern.

        Raises
        ------
        ValueError
            If any validation rule is violated.
        """
        if not questions:
            raise ValueError(
                "Generated exam is invalid: questions list is empty."
            )

        for i, q in enumerate(questions):
            text = q.get("question", "").strip()
            if not text:
                raise ValueError(
                    f"Question at index {i} has an empty 'question' field."
                )
            if len(text) < 10:
                raise ValueError(
                    f"Question at index {i} is too short to be valid: {text!r}"
                )
            if _PLACEHOLDER_PATTERNS.match(text):
                raise ValueError(
                    f"Question at index {i} looks like a placeholder: {text!r}"
                )

    @staticmethod
    def _format_questions(questions: list, subject: str, exam_type: str) -> str:
        lines = [f"**{exam_type.capitalize()} Exam — {subject}**", ""]
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q['question']}  [{q['marks']} marks]")
        return "\n".join(lines)
