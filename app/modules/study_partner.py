"""
app/modules/study_partner.py  —  AI Study Partner Module

The study partner turns the AI into an interactive learning companion.

Handles intents:
  - quiz_me            → generate and run a quiz on a topic
  - active_recall      → Socratic Q&A: ask the student, don't just explain
  - concept_check      → "do you understand this?" follow-up question
  - explain_and_quiz   → explain a concept then immediately quiz it

The module is conversational — it can run multi-turn quiz sessions
using the conversation history to track which questions were already asked.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


class StudyPartnerModule:
    """
    Interactive study partner module.

    Core behaviors:
      1. Generate topic-specific quiz questions at the right difficulty
      2. Evaluate student answers using LLM (not just keyword match)
      3. Provide immediate, educational feedback
      4. Track session progress and adjust difficulty
      5. Use active recall (ask questions, don't lecture)
    """

    def __init__(self, model_router=None, backend_client=None):
        self.model_router = model_router

    async def run(self, agent_input: AgentInput, plan: Any) -> AgentOutput:
        ctx          = agent_input.context or {}
        history      = ctx.get("history", [])
        academic_ctx = ctx.get("academic_context", {}) or {}
        model_id     = ctx.get("selected_model", "openai/gpt-4o-mini")
        lang         = self._detect_lang(agent_input.message)

        # Determine the quiz mode from the message + plan
        mode = self._detect_mode(agent_input.message, plan)
        topic = self._extract_topic(agent_input.message, academic_ctx)

        # Detect if the user is answering a question from the previous turn
        pending_question = self._get_pending_question(history)

        if pending_question and self._is_answer(agent_input.message, history):
            # Evaluate the student's answer
            response = await self._evaluate_answer(
                question=pending_question,
                student_answer=agent_input.message,
                topic=topic,
                history=history,
                model_id=model_id,
                lang=lang,
            )
        else:
            # Generate a new quiz/active recall session
            response = await self._start_study_session(
                message=agent_input.message,
                mode=mode,
                topic=topic,
                history=history,
                academic_ctx=academic_ctx,
                model_id=model_id,
                lang=lang,
            )

        return AgentOutput(
            status="success",
            response=response,
            data={
                "mode": mode,
                "topic": topic,
                "suggestions": self._get_suggestions(topic, lang),
            },
        )

    # ── Session management ────────────────────────────────────────────────

    async def _start_study_session(
        self,
        message: str,
        mode: str,
        topic: str,
        history: list,
        academic_ctx: dict,
        model_id: str,
        lang: str,
    ) -> str:
        if not self.model_router:
            return self._fallback_quiz(topic, lang)

        student_name = academic_ctx.get("studentName", "").split()[0] if academic_ctx.get("studentName") else ""

        lang_rule = (
            "Use Egyptian Arabic dialect. Short, punchy questions. Be energetic and fun."
            if lang == "ar" else
            "Use clear English. Be engaging and educational."
        )

        mode_instruction = {
            "quiz":         "Generate 1 multiple-choice question with 4 options (A/B/C/D). Wait for the answer.",
            "active_recall": "Ask 1 open-ended question that requires the student to recall from memory. DO NOT give the answer. Wait for their response.",
            "concept_check": "Ask 1 short question to check understanding of the topic just discussed.",
            "flashcard":    "Show 1 flashcard: front side only (term or question). Wait for the answer.",
        }.get(mode, "Generate 1 educational question about the topic.")

        system_prompt = f"""\
You are an AI Study Partner — an energetic, smart learning companion.

{lang_rule}

TASK: {mode_instruction}

Topic: {topic or 'the subject being studied'}
{'Student name: ' + student_name + '. Use their name to make it personal.' if student_name else ''}

RULES:
- Ask ONLY 1 question per turn. Never ask multiple questions at once.
- Make questions specific and educational, not trivial.
- For MCQ: label options clearly (A) (B) (C) (D).
- End with "جاهز؟" or "Ready?" to invite the answer.
- NEVER give the answer in the same message as the question.
- Match the student's language (Arabic/English).
"""

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-4:]:
            t_role = turn.get("role", "user")
            t_content = str(turn.get("content", ""))[:300]
            if t_role in ("user", "assistant") and t_content:
                messages.append({"role": t_role, "content": t_content})
        messages.append({"role": "user", "content": message})

        try:
            return await self.model_router.generate_with_messages(
                messages=messages,
                model_id=model_id,
                max_tokens=400,
            ) or self._fallback_quiz(topic, lang)
        except Exception as exc:
            logger.error("StudyPartner._start_session: %s", exc)
            return self._fallback_quiz(topic, lang)

    async def _evaluate_answer(
        self,
        question: str,
        student_answer: str,
        topic: str,
        history: list,
        model_id: str,
        lang: str,
    ) -> str:
        if not self.model_router:
            return "إجابة جيدة! هل تريد سؤالاً آخر؟" if lang == "ar" else "Good answer! Want another question?"

        lang_rule = (
            "Respond in Egyptian Arabic. Short, encouraging, specific."
            if lang == "ar" else
            "Respond in English. Short, encouraging, specific."
        )

        system_prompt = f"""\
You are an AI Study Partner evaluating a student's answer.

{lang_rule}

TASK:
1. Evaluate if the answer is correct, partially correct, or incorrect.
2. Give specific, educational feedback (not just "correct" or "wrong").
3. If wrong: explain WHY briefly and give the correct answer.
4. If correct: reinforce with a brief explanation of WHY it's correct.
5. Then IMMEDIATELY ask a follow-up question to deepen understanding.

Topic: {topic or 'general'}
Keep the entire response under 5 sentences.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": question},
            {"role": "user", "content": student_answer},
        ]

        try:
            return await self.model_router.generate_with_messages(
                messages=messages,
                model_id=model_id,
                max_tokens=350,
            ) or ("ممتاز! هل عايز تكمل؟" if lang == "ar" else "Great! Want to continue?")
        except Exception as exc:
            logger.error("StudyPartner._evaluate_answer: %s", exc)
            return "حاول تاني!" if lang == "ar" else "Good attempt! Let's try another."

    # ── Detection helpers ─────────────────────────────────────────────────

    def _detect_mode(self, message: str, plan: Any) -> str:
        msg = message.lower()
        if any(kw in msg for kw in ["quiz", "اختبار", "امتحاني", "سألني", "quiz me"]):
            return "quiz"
        if any(kw in msg for kw in ["active recall", "تذكر", "استرجاع"]):
            return "active_recall"
        if any(kw in msg for kw in ["flashcard", "فلاش كارد", "بطاقة"]):
            return "flashcard"
        return "quiz"  # default

    def _extract_topic(self, message: str, academic_ctx: dict) -> str:
        # Try to extract topic from message
        patterns = [
            r"(?:in|on|about|على|في|عن|لـ|لـ)\s+(.{3,40}?)(?:\s*$|\?|؟|،|,)",
            r"(?:topic|موضوع|مادة)\s+(.{3,40})",
        ]
        for pat in patterns:
            m = re.search(pat, message, re.IGNORECASE | re.UNICODE)
            if m:
                return m.group(1).strip()

        # Fall back to current subject from context
        return (
            academic_ctx.get("subjectName")
            or academic_ctx.get("currentSubject")
            or ""
        )

    def _get_pending_question(self, history: list) -> Optional[str]:
        """Return the last assistant message if it looks like a question."""
        for turn in reversed(history[-3:]):
            if turn.get("role") == "assistant":
                content = turn.get("content", "")
                if "?" in content or "؟" in content or "جاهز" in content or "Ready" in content.lower():
                    return content
        return None

    def _is_answer(self, message: str, history: list) -> bool:
        """Heuristic: message is likely an answer if it's short and follows a question."""
        return len(message.split()) <= 15

    def _fallback_quiz(self, topic: str, lang: str) -> str:
        t = topic or ("المادة" if lang == "ar" else "the subject")
        if lang == "ar":
            return f"حسناً! سؤال في {t}: ما أهم مفهوم تعلمته في هذا الموضوع؟ 🤔"
        return f"Alright! Question about {t}: What's the most important concept you've learned so far? 🤔"

    def _get_suggestions(self, topic: str, lang: str) -> list[str]:
        t = topic or ("الموضوع" if lang == "ar" else "this topic")
        if lang == "ar":
            return [
                f"سؤال آخر في {t}",
                "اعمل flashcards",
                "اشرحلي المفهوم ده",
            ]
        return [
            f"Another question on {t}",
            "Generate flashcards",
            "Explain this concept",
        ]

    @staticmethod
    def _detect_lang(message: str) -> str:
        arabic_chars = sum(1 for c in message if "؀" <= c <= "ۿ")
        return "ar" if arabic_chars / max(len(message), 1) > 0.2 else "en"
