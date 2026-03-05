"""
app/modules/exam_generation.py

Exam Generation Module.

Flow:
  1. Resolve subjectOfferingId (from plan, context, or pre-execution steps).
  2. Fetch course material from backend   GET /api/Materials/by-offering/{id}
  3. Generate questions via Flan-T5  (model_service.generate_questions).
  4. Persist the exam via BackendClient  POST /api/ai/execute/create-generated-exam
  5. Return generated questions as structured text and JSON.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


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
    Optionally persists the exam to the .NET backend.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("ExamGenerationModule: starting.")

        params      = getattr(plan, "exam_params", None) if plan else None
        context     = agent_input.context or {}
        num_q       = getattr(params, "numberOfQuestions", 5) if params else 5
        exam_type   = getattr(params, "examType", "midterm") if params else "midterm"
        subject     = getattr(params, "subjectName", "the subject") if params else "the subject"

        # -- 1. Resolve subjectOfferingId -------------------------------------
        offering_id = (
            getattr(params, "subjectOfferingId", None) if params else None
        ) or context.get("subjectOfferingId")

        if not offering_id and plan and hasattr(plan, "pre_execution_steps"):
            for step in plan.pre_execution_steps:
                if step.tool == "ResolveSubjectOffering":
                    res = await self.backend_client.execute_tool(
                        route="/api/ai/execute/resolve-subject-offering",
                        parameters=step.input_payload,
                        auth_header=agent_input.auth_header,
                        user_id=agent_input.user_id,
                    )
                    offering_id = res.get("subjectOfferingId") if res and "error" not in res else None
                    break

        # -- 2. Fetch course material ------------------------------------------
        material_text = ""
        if offering_id:
            result = await self.backend_client.fetch(
                route=f"/api/Materials/by-offering/{offering_id}",
                auth_header=agent_input.auth_header,
            )
            if "error" not in result:
                if "_raw_bytes" in result:
                    material_text = _pdf_to_text(result["_raw_bytes"])
                else:
                    material_text = str(result.get("content") or result.get("text") or "")

        if not material_text:
            material_text = agent_input.message or f"General {exam_type} exam for {subject}."

        # -- 3. Generate questions via ModelRouter ----------------------------
        selected_model = context.get("selected_model")

        # Guard: Prefer task-optimized Flan-T5 for exam generation unless explicitly requested otherwise
        target_model = selected_model
        if not selected_model or not selected_model.startswith("hf/"):
            target_model = "hf/google/flan-t5-base"
            logger.info(
                "ExamGenerationModule: overriding pipeline model '%s' with task-optimized 'hf/google/flan-t5-base'.",
                selected_model or "default"
            )

        logger.info(
            "ExamGenerationModule: generating %d questions for '%s' using %s.", 
            num_q, subject, target_model
        )
        raw_questions = await self.model_router.generate_questions(
            material_text, num_questions=num_q, model_id=target_model
        )

        # Build structured list from the raw output
        questions_list = self._parse_questions(raw_questions, num_q, subject)

        # -- 4. Persist to backend (optional) ---------------------------------
        backend_result: Dict[str, Any] = {}
        if offering_id:
            exam_payload: Dict[str, Any] = {
                "subjectOfferingId": offering_id,
                "examData": {
                    "title": f"{exam_type.capitalize()} Exam — {subject}",
                    "questions": questions_list,
                },
                "handleStudentRandomization": False,
            }
            backend_result = await self.backend_client.execute_tool(
                route="/api/ai/execute/create-generated-exam",
                parameters=exam_payload,
                auth_header=agent_input.auth_header,
                user_id=agent_input.user_id,
            )
            if backend_result and "error" in backend_result:
                logger.warning(
                    "ExamGenerationModule: backend persist failed — %s",
                    backend_result["error"],
                )

        # -- 5. Return ---------------------------------------------------------
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
    def _parse_questions(raw: str, num_q: int, subject: str):
        """Turn the model's raw text into a list of question dicts."""
        if not raw:
            return [
                {"question": f"Question {i + 1} about {subject}.", "answer": "N/A", "marks": 5}
                for i in range(num_q)
            ]
        lines  = [l.strip() for l in raw.split("\n") if l.strip()]
        result = []
        for i, line in enumerate(lines[:num_q]):
            result.append({"question": line, "answer": "", "marks": 5})
        # Pad if needed
        while len(result) < num_q:
            result.append(
                {"question": f"Additional question {len(result)+1}.", "answer": "", "marks": 5}
            )
        return result[:num_q]

    @staticmethod
    def _format_questions(questions: list, subject: str, exam_type: str) -> str:
        lines = [f"**{exam_type.capitalize()} Exam — {subject}**", ""]
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q['question']}  [{q['marks']} marks]")
        return "\n".join(lines)
