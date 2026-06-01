"""
app/api/routes/companion.py  —  AI Companion FastAPI Routes

Endpoints called by the .NET backend (AiService.cs) for companion features.

Route prefix: /api/companion

Endpoints:
  POST /api/companion/generate-flashcards  — generate flashcard JSON for a topic
  POST /api/companion/quick-prompt         — one-shot LLM prompt
  POST /api/companion/study-plan           — generate personalized study plan
  POST /api/companion/progress-report      — generate progress report narrative
  POST /api/companion/analyze-academic     — academic coaching analysis
  GET  /api/companion/profile/{user_id}    — get cached companion profile
  POST /api/companion/profile/{user_id}    — update cached companion profile
  POST /api/companion/record-session       — record a completed learning session

These routes are internal — called by .NET, not directly by the frontend.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.logging import logger

router = APIRouter(prefix="/api/companion", tags=["companion"])


# ── Request/Response Models ───────────────────────────────────────────────────

class GenerateFlashcardsRequest(BaseModel):
    topic: str
    card_count: int = Field(default=15, ge=3, le=30)
    difficulty: str = "mixed"   # easy | medium | hard | mixed


class QuickPromptRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=200, ge=50, le=800)
    lang: str = "ar"


class StudyPlanRequest(BaseModel):
    student_name: str = ""
    weak_subjects: list[str] = []
    enrolled_subjects: list[str] = []
    gpa: float = 0.0
    goal: str = ""
    learning_style: str = "mixed"
    days_until_exam: int = 14


class ProgressReportRequest(BaseModel):
    student_name: str = ""
    sessions_this_week: int = 0
    avg_accuracy: float = 0.0
    study_minutes: int = 0
    streak_days: int = 0
    weak_subjects: list[str] = []
    strong_subjects: list[str] = []
    period: str = "weekly"


class RecordSessionRequest(BaseModel):
    user_id: str
    topic: str
    session_type: str
    duration_minutes: int = 0
    accuracy_percent: float = 0.0
    completed: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate-flashcards")
async def generate_flashcards(request: Request, body: GenerateFlashcardsRequest):
    """
    Generate AI flashcards for a topic.
    Returns a JSON array of flashcard objects.
    """
    model_router = getattr(request.app.state, "agent", None)
    if model_router:
        model_router = getattr(model_router, "_model_router", None) or \
                       getattr(model_router, "model_router", None)
    if not model_router:
        from app.agents.model_router import ModelRouter
        # Attempt to get from app state directly
        model_router = getattr(request.app.state, "model_router", None)

    if not model_router:
        # Fallback: return template flashcards
        return _template_flashcards(body.topic, body.card_count)

    prompt = _build_flashcard_prompt(body.topic, body.card_count, body.difficulty)

    try:
        raw = await model_router.generate(
            prompt=prompt,
            system_instruction=(
                "You are an expert tutor. Generate flashcards as a JSON array only. "
                "No markdown, no extra text — pure JSON array."
            ),
            model_id="openai/gpt-4o-mini",
        )

        if not raw:
            return _template_flashcards(body.topic, body.card_count)

        # Extract JSON array
        import re
        match = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
        if match:
            cards = json.loads(match.group(0))
            logger.info("Companion: generated %d flashcards for topic=%r", len(cards), body.topic)
            return cards

        return _template_flashcards(body.topic, body.card_count)

    except Exception as exc:
        logger.error("companion/generate-flashcards: %s", exc)
        return _template_flashcards(body.topic, body.card_count)


@router.post("/quick-prompt")
async def quick_prompt(request: Request, body: QuickPromptRequest):
    """
    One-shot LLM prompt for feedback generation, recommendations, etc.
    Returns: {"response": "..."}
    """
    model_router = _get_model_router(request)
    if not model_router:
        return {"response": None}

    try:
        lang_instruction = (
            "Respond in Arabic (Egyptian dialect)." if body.lang == "ar"
            else "Respond in English."
        )
        response = await model_router.generate(
            prompt=body.prompt,
            system_instruction=(
                f"You are an AI academic assistant. {lang_instruction} "
                "Be concise, accurate, and encouraging."
            ),
            model_id="openai/gpt-4o-mini",
        )
        return {"response": response}
    except Exception as exc:
        logger.error("companion/quick-prompt: %s", exc)
        return {"response": None}


@router.post("/study-plan")
async def generate_study_plan(request: Request, body: StudyPlanRequest):
    """
    Generate a personalized study plan.
    Returns AiStudyPlanDto-compatible JSON.
    """
    model_router = _get_model_router(request)
    if not model_router:
        return _fallback_study_plan(body)

    weak_str   = ", ".join(body.weak_subjects) if body.weak_subjects else "none identified"
    enroll_str = ", ".join(body.enrolled_subjects) if body.enrolled_subjects else "all subjects"

    prompt = f"""\
Create a personalized {body.days_until_exam}-day study plan in JSON format.

Student: {body.student_name or 'Student'}
GPA: {body.gpa:.2f}
Weak subjects: {weak_str}
Enrolled: {enroll_str}
Goal: {body.goal or 'exam preparation'}
Learning style: {body.learning_style}

Return JSON matching this schema exactly:
{{
  "plan_title": "string",
  "daily_tasks": [
    {{"day": "Day 1", "subject": "string", "task": "string", "duration_min": 30}}
  ],
  "focus_areas": ["string"],
  "motivational_note": "string"
}}

Language: Arabic. Return ONLY valid JSON.
"""

    try:
        raw = await model_router.generate(
            prompt=prompt,
            system_instruction="You are an academic planner. Return ONLY valid JSON, no markdown.",
            model_id="openai/gpt-4o-mini",
        )
        if raw:
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception as exc:
        logger.error("companion/study-plan: %s", exc)

    return _fallback_study_plan(body)


@router.post("/progress-report")
async def generate_progress_report(request: Request, body: ProgressReportRequest):
    """
    Generate a narrative progress report.
    Returns: {"report": "markdown text"}
    """
    model_router = _get_model_router(request)
    if not model_router:
        return {"report": _fallback_progress_text(body)}

    prompt = f"""\
Generate a {body.period} academic progress report for {body.student_name or 'the student'}.

Data:
- Study sessions: {body.sessions_this_week}
- Study time: {body.study_minutes} minutes
- Average quiz accuracy: {body.avg_accuracy:.0f}%
- Study streak: {body.streak_days} days
- Weak subjects: {', '.join(body.weak_subjects) or 'none'}
- Strong subjects: {', '.join(body.strong_subjects) or 'none'}

Write in Arabic, 100-150 words, warm and encouraging tone.
Use the student's name if provided. End with a specific action recommendation.
"""

    try:
        report = await model_router.generate(
            prompt=prompt,
            system_instruction="You are an AI academic coach. Write in Arabic. Be encouraging and specific.",
            model_id="openai/gpt-4o-mini",
        )
        return {"report": report or _fallback_progress_text(body)}
    except Exception as exc:
        logger.error("companion/progress-report: %s", exc)
        return {"report": _fallback_progress_text(body)}


@router.post("/record-session")
async def record_session(request: Request, body: RecordSessionRequest):
    """
    Record a completed learning session in Redis memory.
    Called by AiCompanionService after CompleteSessionAsync.
    """
    try:
        from app.services.companion_memory import get_companion_memory
        memory = get_companion_memory()
        await memory.record_study_session(
            user_id=body.user_id,
            topic=body.topic,
            session_type=body.session_type,
            duration_minutes=body.duration_minutes,
            accuracy_percent=body.accuracy_percent,
        )
        if body.completed:
            await memory.infer_and_update_learning_style(
                user_id=body.user_id,
                session_type=body.session_type,
                completed=True,
                duration_minutes=body.duration_minutes,
            )
        return {"status": "recorded"}
    except Exception as exc:
        logger.error("companion/record-session: %s", exc)
        return {"status": "error", "detail": str(exc)}


@router.get("/profile/{user_id}")
async def get_companion_profile(user_id: str):
    """Return cached companion profile from Redis."""
    try:
        from app.services.companion_memory import get_companion_memory
        profile = await get_companion_memory().get_companion_profile(user_id)
        return profile
    except Exception as exc:
        logger.error("companion/profile GET: %s", exc)
        return {}


@router.post("/profile/{user_id}")
async def update_companion_profile(user_id: str, body: dict):
    """Update companion profile in Redis."""
    try:
        from app.services.companion_memory import get_companion_memory
        updated = await get_companion_memory().update_companion_profile(user_id, body)
        return updated
    except Exception as exc:
        logger.error("companion/profile POST: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_model_router(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent:
        mr = getattr(agent, "_model_router", None) or getattr(agent, "model_router", None)
        if mr:
            return mr
    return getattr(request.app.state, "model_router", None)


def _build_flashcard_prompt(topic: str, count: int, difficulty: str) -> str:
    diff_note = {
        "easy": "Focus on basic definitions and simple recall.",
        "hard": "Focus on complex application and analysis questions.",
        "mixed": "Mix easy (40%), medium (40%), and hard (20%) cards.",
    }.get(difficulty, "Mix different difficulty levels.")

    return f"""\
Generate {count} flashcards for the topic: "{topic}"
{diff_note}

Return ONLY a JSON array:
[
  {{"front": "question/term", "back": "answer/definition", "hint": "memory tip or null", "difficulty": "easy|medium|hard"}},
  ...
]

Make each card educational, clear, and university-level appropriate.
"""


def _template_flashcards(topic: str, count: int) -> list[dict]:
    """Fallback: return a minimal template set."""
    return [
        {
            "front": f"What is the definition of {topic}?",
            "back": f"[Definition of {topic} — AI service unavailable, please try again]",
            "hint": None,
            "difficulty": "medium",
        }
    ]


def _fallback_study_plan(body: StudyPlanRequest) -> dict:
    return {
        "plan_title": f"خطة مذاكرة {body.days_until_exam} يوم",
        "daily_tasks": [
            {
                "day": f"اليوم {i+1}",
                "subject": body.weak_subjects[0] if body.weak_subjects else "المادة",
                "task": "مراجعة المحاضرات وحل تمارين",
                "duration_min": 45,
            }
            for i in range(min(body.days_until_exam, 7))
        ],
        "focus_areas": body.weak_subjects[:3] or ["المواد الأساسية"],
        "motivational_note": "كل يوم مذاكرة هو خطوة نحو النجاح! 💪",
    }


def _fallback_progress_text(body: ProgressReportRequest) -> str:
    name = body.student_name.split()[0] if body.student_name else "طالبنا"
    return (
        f"تقرير {body.period} لـ{name}:\n"
        f"• جلسات المذاكرة: {body.sessions_this_week}\n"
        f"• وقت المذاكرة: {body.study_minutes} دقيقة\n"
        f"• دقة الـ quizzes: {body.avg_accuracy:.0f}%\n"
        "استمر في المذاكرة المنتظمة! 💪"
    )
