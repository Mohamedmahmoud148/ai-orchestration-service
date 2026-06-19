"""
app/api/routes/lecture.py  —  Lecture Recording Intelligence

Called by the .NET backend for AI-powered audio analysis.

Routes:
  POST /api/lecture/transcribe  — audio bytes → transcript (Whisper)
  POST /api/lecture/analyze     — transcript text → summary + flashcards + quiz + timeline
  POST /api/lecture/ask         — transcript + question → AI answer

These are internal routes called by .NET — not directly by the frontend.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.logging import logger

router = APIRouter(prefix="/api/lecture", tags=["lecture"])


# ── Request/Response Models ───────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    transcript: str

class AskRequest(BaseModel):
    transcript: str
    message: str


# ── Helper: get model router ──────────────────────────────────────────────────

def _get_model_router(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent:
        mr = getattr(agent, "_model_router", None) or getattr(agent, "model_router", None)
        if mr:
            return mr
    return getattr(request.app.state, "model_router", None)


# ── POST /api/lecture/transcribe ──────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(request: Request):
    """
    Accepts multipart audio file → returns transcript via Whisper API.

    Falls back to OpenRouter multimodal model if OPENAI_API_KEY is not set.
    The .NET WhisperSpeechToTextService calls this endpoint.
    """
    try:
        form = await request.form()
        audio_file = form.get("file")
        if audio_file is None:
            return JSONResponse(status_code=400, content={"detail": "No audio file provided."})

        filename: str = getattr(audio_file, "filename", "recording.mp3")
        audio_bytes: bytes = await audio_file.read()

        if not audio_bytes:
            return JSONResponse(status_code=400, content={"detail": "Empty audio file."})

        logger.info("lecture/transcribe: received %d bytes for %s", len(audio_bytes), filename)

        _WHISPER_MAX_BYTES = 24 * 1024 * 1024  # 24MB (Whisper limit is 25MB)

        # ── Try OpenAI Whisper first ──────────────────────────────────────────
        from app.core.config import settings
        if settings.OPENAI_API_KEY:
            try:
                import httpx

                # Whisper has 25MB limit — compress if needed
                whisper_bytes = audio_bytes
                if len(audio_bytes) > _WHISPER_MAX_BYTES:
                    logger.warning(
                        "lecture/transcribe: file %d bytes > 24MB limit — attempting compression",
                        len(audio_bytes)
                    )
                    try:
                        # Try to compress MP3 using pydub (lower bitrate)
                        import io
                        from pydub import AudioSegment
                        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                        buf = io.BytesIO()
                        # Export at 32kbps mono which reduces size ~8x
                        seg.export(buf, format="mp3", bitrate="32k", parameters=["-ac", "1"])
                        whisper_bytes = buf.getvalue()
                        logger.info(
                            "lecture/transcribe: compressed %d → %d bytes",
                            len(audio_bytes), len(whisper_bytes)
                        )
                    except Exception as compress_ex:
                        logger.warning("lecture/transcribe: compression failed — %s", compress_ex)
                        # If still too large, chunk and transcribe first 24MB
                        if len(audio_bytes) > _WHISPER_MAX_BYTES:
                            whisper_bytes = audio_bytes[:_WHISPER_MAX_BYTES]
                            logger.info("lecture/transcribe: truncated to first 24MB for Whisper")

                async with httpx.AsyncClient(timeout=300.0) as client:
                    files = {"file": (filename, whisper_bytes, _mime_from_name(filename))}
                    data = {"model": "whisper-1", "response_format": "verbose_json"}
                    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
                    resp = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers=headers, files=files, data=data
                    )
                    resp.raise_for_status()
                    whisper_data = resp.json()
                    transcript = whisper_data.get("text", "")
                    duration = int(whisper_data.get("duration", 0)) if "duration" in whisper_data else None

                    logger.info("lecture/transcribe: Whisper returned %d chars", len(transcript))
                    return {
                        "transcript":       transcript,
                        "duration_seconds": duration,
                        "provider":         "whisper-1"
                    }
            except Exception as ex:
                logger.warning("lecture/transcribe: Whisper API failed — %s. Trying fallback.", ex)

        # ── Fallback: OpenRouter audio analysis ───────────────────────────────
        model_router = _get_model_router(request)
        if model_router:
            try:
                import base64
                audio_b64 = base64.b64encode(audio_bytes).decode()
                transcript = await model_router.generate(
                    prompt=(
                        "Please transcribe the following audio recording completely and accurately. "
                        "Return only the transcript text, no additional commentary."
                    ),
                    system_instruction="You are a speech-to-text transcription service.",
                    model_id="openai/whisper-large-v3",
                )
                if transcript:
                    return {"transcript": transcript, "duration_seconds": None, "provider": "openrouter-whisper"}
            except Exception as ex:
                logger.error("lecture/transcribe: fallback also failed — %s", ex)

        return JSONResponse(status_code=503, content={
            "detail": "Speech-to-text service unavailable. Please configure OPENAI_API_KEY."
        })

    except Exception as ex:
        logger.error("lecture/transcribe: unexpected error — %s", ex)
        return JSONResponse(status_code=500, content={"detail": str(ex)})


# ── POST /api/lecture/analyze ─────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_transcript(request: Request, body: AnalyzeRequest):
    """
    Takes lecture transcript → returns:
      summary, key_concepts, timeline, flashcards (15), quiz (10), suggested_questions
    All generated strictly from the transcript — no external knowledge.
    """
    model_router = _get_model_router(request)
    transcript = body.transcript[:10_000]  # safe token limit

    system = (
        "You are an AI academic assistant. Your task is to analyze a lecture transcript. "
        "Use ONLY the content of the transcript provided — do not add external information. "
        "Always respond in the same language as the transcript."
    )

    # ── Summary + Key Concepts + Timeline ────────────────────────────────────
    summary_prompt = f"""Analyze this lecture transcript and return a JSON object with exactly this structure:
{{
  "summary": "A clear 3-5 paragraph summary of the lecture",
  "key_concepts": ["concept1", "concept2", "concept3", ...],
  "timeline": [
    {{"title": "Section Name", "start": 0, "end": 300}},
    ...
  ],
  "suggested_questions": [
    {{"question": "Exam question?", "difficulty": "Easy|Medium|Hard"}},
    ...
  ]
}}

TRANSCRIPT:
{transcript}

Return ONLY the JSON object. No markdown, no extra text."""

    summary_data = {"summary": "", "key_concepts": [], "timeline": [], "suggested_questions": []}
    if model_router:
        try:
            raw = await model_router.generate(
                prompt=summary_prompt,
                system_instruction=system,
                model_id="openai/gpt-4o-mini"
            )
            if raw:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    summary_data = json.loads(match.group(0))
        except Exception as ex:
            logger.error("lecture/analyze: summary generation failed — %s", ex)

    # ── Flashcards ────────────────────────────────────────────────────────────
    flashcard_prompt = f"""Based on this lecture transcript, generate 15 educational flashcards as a JSON array:
[
  {{"front": "Question or concept", "back": "Answer or explanation"}},
  ...
]

TRANSCRIPT:
{transcript[:5_000]}

Return ONLY the JSON array. No markdown."""

    flashcards = []
    if model_router:
        try:
            raw_fc = await model_router.generate(
                prompt=flashcard_prompt,
                system_instruction=system,
                model_id="openai/gpt-4o-mini"
            )
            if raw_fc:
                match = re.search(r'\[.*\]', raw_fc, re.DOTALL)
                if match:
                    flashcards = json.loads(match.group(0))
        except Exception as ex:
            logger.error("lecture/analyze: flashcard generation failed — %s", ex)

    # ── Quiz ──────────────────────────────────────────────────────────────────
    quiz_prompt = f"""Based on this lecture transcript, generate 10 multiple-choice quiz questions as a JSON array:
[
  {{
    "question": "Question text?",
    "option_a": "Option A",
    "option_b": "Option B",
    "option_c": "Option C",
    "option_d": "Option D",
    "correct_answer": "A",
    "explanation": "Why this is correct"
  }},
  ...
]

TRANSCRIPT:
{transcript[:5_000]}

Return ONLY the JSON array. No markdown."""

    quiz = []
    if model_router:
        try:
            raw_quiz = await model_router.generate(
                prompt=quiz_prompt,
                system_instruction=system,
                model_id="openai/gpt-4o-mini"
            )
            if raw_quiz:
                match = re.search(r'\[.*\]', raw_quiz, re.DOTALL)
                if match:
                    quiz = json.loads(match.group(0))
        except Exception as ex:
            logger.error("lecture/analyze: quiz generation failed — %s", ex)

    return {
        "summary":             summary_data.get("summary", ""),
        "key_concepts":        summary_data.get("key_concepts", []),
        "timeline":            summary_data.get("timeline", []),
        "suggested_questions": summary_data.get("suggested_questions", []),
        "flashcards":          flashcards,
        "quiz":                quiz
    }


# ── POST /api/lecture/ask ─────────────────────────────────────────────────────

@router.post("/ask")
async def ask_about_lecture(request: Request, body: AskRequest):
    """
    Student asks a question about the lecture.
    AI answers using ONLY the transcript content.
    """
    model_router = _get_model_router(request)
    if not model_router:
        return {"answer": "خدمة الـ AI غير متاحة مؤقتاً."}

    prompt = (
        f"=== LECTURE TRANSCRIPT ===\n{body.transcript[:8_000]}\n=== END ===\n\n"
        f"Student question: {body.message}\n\n"
        "Answer the student's question using ONLY the lecture transcript above. "
        "If the answer is not in the transcript, say so clearly. "
        "Be concise and educational."
    )

    try:
        answer = await model_router.generate(
            prompt=prompt,
            system_instruction=(
                "You are an AI tutor helping a student understand their lecture recording. "
                "Answer only from the provided transcript. "
                "Respond in the same language as the student's question."
            ),
            model_id="openai/gpt-4o-mini"
        )
        return {"answer": answer or "لم أتمكن من الإجابة. حاول مرة أخرى."}
    except Exception as ex:
        logger.error("lecture/ask: failed — %s", ex)
        return {"answer": "حدث خطأ. حاول مرة أخرى."}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mime_from_name(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
        "ogg": "audio/ogg",
    }.get(ext, "audio/mpeg")
