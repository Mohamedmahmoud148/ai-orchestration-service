"""
ai_grading.py — POST /api/ai/grade-submission

Called by the .NET AssignmentService to AI-grade essay submissions.
Completely standalone — does NOT go through the Agent pipeline.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.logging import logger

router = APIRouter()

# ── Request / Response models ─────────────────────────────────
class GradingRequest(BaseModel):
    submission_text: str
    assignment_title: str
    assignment_description: str
    rubric: Optional[str] = None
    max_grade: float = 100.0

class GradingResponse(BaseModel):
    score: float
    feedback: str
    strengths: str
    weaknesses: str
    confidence: float

# ── Grading prompt ────────────────────────────────────────────
_GRADING_SYSTEM_PROMPT = """You are an expert university professor grading student assignments.
Your job is to evaluate the student's submission fairly and provide detailed, constructive feedback.

Rules:
- Be objective and consistent
- Base your grade ONLY on the provided rubric and assignment description
- Never give a score above max_grade
- Provide specific, actionable feedback
- Respond ONLY with valid JSON — no explanation outside the JSON block

Response format (strict JSON):
{
  "score": <number between 0 and max_grade>,
  "feedback": "<overall assessment, 2-3 sentences>",
  "strengths": "<what the student did well>",
  "weaknesses": "<areas needing improvement>",
  "confidence": <0.0 to 1.0 — how confident you are in this grade>
}"""

def _build_grading_prompt(req: GradingRequest) -> str:
    rubric_section = f"\nGrading Rubric:\n{req.rubric}" if req.rubric else ""
    return f"""Assignment: {req.assignment_title}
Description: {req.assignment_description}{rubric_section}
Maximum Grade: {req.max_grade}

Student Submission:
---
{req.submission_text}
---

Grade this submission and return JSON only."""


@router.post("/api/ai/grade-submission", response_model=GradingResponse)
async def grade_submission(req: GradingRequest, request: Request) -> GradingResponse:
    """AI-grade an essay/text assignment submission."""
    if not req.submission_text or len(req.submission_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Submission text is too short to grade.")

    try:
        client = getattr(request.app.state, "openrouter_client", None)
        if client is None:
            raise HTTPException(status_code=503, detail="AI client not available.")

        import os
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _GRADING_SYSTEM_PROMPT},
                {"role": "user", "content": _build_grading_prompt(req)},
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        score = float(data.get("score", 0))
        score = max(0.0, min(score, req.max_grade))

        return GradingResponse(
            score=score,
            feedback=data.get("feedback", "No feedback provided."),
            strengths=data.get("strengths", ""),
            weaknesses=data.get("weaknesses", ""),
            confidence=float(data.get("confidence", 0.7)),
        )

    except json.JSONDecodeError:
        logger.error("AI grading returned invalid JSON")
        raise HTTPException(status_code=502, detail="AI returned invalid grading response.")
    except Exception as e:
        logger.error(f"AI grading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")
