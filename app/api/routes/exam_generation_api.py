"""
exam_generation_api.py — POST /generate-exam

Called directly by the .NET AiService.GenerateExamAsync to generate exam questions.
Standalone endpoint — bypasses the agent pipeline for speed and simplicity.

Request  (mirrors C# AiGenerateExamRequestDto):
  { subject, department, batch, difficulty, questionCount, examType, topics[] }

Response (mirrors C# List<CreateExamQuestionDto>):
  [{ questionText, options, correctAnswer, questionType, mark }]
  questionType: 0=MCQ, 1=TrueFalse, 2=Essay
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.logging import logger

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class GenerateExamRequest(BaseModel):
    # Accept both camelCase (from C# SnakeCaseLower serializer → question_count)
    # and fallback aliases
    subject: str = "General"
    department: str = ""
    batch: str = ""
    difficulty: str = "Medium"
    question_count: int = 10   # C# sends snake_case
    exam_type: str = "Final"
    topics: List[str] = []

    # camelCase aliases for forward compatibility
    questionCount: Optional[int] = None
    examType: Optional[str] = None

    def resolved_count(self) -> int:
        return self.questionCount if self.questionCount is not None else self.question_count

    def resolved_type(self) -> str:
        return self.examType if self.examType is not None else self.exam_type


# Response uses snake_case to match C# AiService._jsonOptions (SnakeCaseLower)
class ExamQuestionOut(BaseModel):
    question_text: str
    options: Optional[List[str]] = None
    correct_answer: str
    question_type: int = 0   # 0=MCQ, 1=TrueFalse, 2=Essay
    mark: int = 5


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert university exam writer.
Generate exam questions strictly as a JSON array. Each element must have:
  - "questionText": string (at least 20 characters)
  - "questionType": integer — 0 for MCQ, 1 for TrueFalse, 2 for Essay
  - "options": array of exactly 4 strings for MCQ, null for TrueFalse/Essay
  - "correctAnswer": for MCQ: one of the exact option strings; for TrueFalse: "True" or "False"; for Essay: a model answer
  - "mark": integer between 5 and 20

Return ONLY the JSON array with no extra text, markdown, or code fences."""


def _build_prompt(req: GenerateExamRequest) -> str:
    context_parts = []
    if req.department:
        context_parts.append(f"Department: {req.department}")
    if req.batch:
        context_parts.append(f"Batch/Level: {req.batch}")
    if req.topics:
        context_parts.append(f"Topics: {', '.join(req.topics)}")
    context = "\n".join(context_parts)

    count = req.resolved_count()
    etype = req.resolved_type()

    return (
        f"Generate {count} {req.difficulty}-difficulty {etype} "
        f"exam questions for the subject '{req.subject}'.\n"
        f"{context}\n\n"
        f"Return a valid JSON array only."
    )


def _get(q: dict, *keys, default=""):
    """Try multiple key names (camelCase + snake_case) and return first non-empty value."""
    for k in keys:
        v = q.get(k)
        if v is not None and v != "":
            return v
    return default


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/generate-exam", response_model=List[ExamQuestionOut])
async def generate_exam(req: GenerateExamRequest, request: Request) -> List[ExamQuestionOut]:
    client = getattr(request.app.state, "openrouter_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="AI client not available.")

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(req)},
            ],
            temperature=0.4,
            max_tokens=4000,
        )

        raw = (response.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        questions: list = json.loads(raw)
        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError("Empty or invalid questions list returned by model.")

        result = []
        for q in questions:
            # Handle both camelCase and snake_case from different AI models
            text    = str(_get(q, "questionText", "question_text", "question")).strip()
            if not text:
                continue
            q_type  = int(_get(q, "questionType", "question_type") or 0)
            options = q.get("options") if q_type == 0 else None
            correct = str(_get(q, "correctAnswer", "correct_answer")).strip()
            mark    = max(1, min(int(q.get("mark", 5)), 100))

            result.append(ExamQuestionOut(
                question_text=text,
                options=options,
                correct_answer=correct,
                question_type=q_type,
                mark=mark,
            ))

        if not result:
            raise ValueError("All generated questions were empty after filtering.")

        logger.info(
            "generate-exam: generated %d questions for subject='%s' difficulty='%s'",
            len(result), req.subject, req.difficulty,
        )
        return result

    except json.JSONDecodeError as e:
        logger.error("generate-exam: model returned invalid JSON — %s | raw=%s", e, raw[:200])
        raise HTTPException(status_code=502, detail="AI returned invalid JSON response.")
    except Exception as e:
        logger.error("generate-exam: failed — %s", e)
        raise HTTPException(status_code=500, detail=f"Exam generation failed: {str(e)}")
