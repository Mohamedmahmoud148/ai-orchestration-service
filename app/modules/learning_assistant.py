"""
app/modules/learning_assistant.py  —  AI Learning Assistant Module

Handles all content-generation learning requests:
  - generate_flashcards  → Create a set of flashcards for a topic
  - generate_examples    → Generate practical examples for a concept
  - generate_exercises   → Create practice exercises
  - generate_summary     → Create a study-ready summary
  - explain_concept      → Deep concept explanation with analogies
  - generate_mnemonics   → Memory tricks for hard topics

These are standalone AI-generation tasks (no backend data needed).
The generated content is also returned in structured format for
storage via the /api/companion/flashcards/generate endpoint.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


# Content type → prompt instruction mapping
_CONTENT_INSTRUCTIONS = {
    "flashcards": """\
Generate {count} flashcards as a JSON array. Each card:
{{"front": "question or term", "back": "answer or definition", "hint": "memory tip or null", "difficulty": "easy|medium|hard"}}
Return ONLY the JSON array, no markdown.
""",
    "examples": """\
Generate {count} practical, real-world examples for this concept.
Format: numbered list with a brief explanation for each example.
Make examples relatable to university students.
""",
    "exercises": """\
Generate {count} practice exercises.
For each: state the problem, then on a new line "Answer: [answer]".
Vary difficulty from easy to hard.
""",
    "summary": """\
Generate a structured study summary for this topic.
Format:
## Key Concepts
- [bullet points]
## Important Formulas/Rules
- [if applicable]
## Common Mistakes
- [what students get wrong]
## One-Line Memory Trick
[a memorable way to remember the core idea]
""",
    "mnemonics": """\
Create {count} creative memory tricks (mnemonics, acronyms, or analogies)
for remembering this topic. Make them funny or vivid — memorable!
""",
}


class LearningAssistantModule:
    """
    Generates educational content on demand.

    No backend data needed — pure LLM generation.
    Optimized for Arabic/English bilingual output.
    """

    def __init__(self, model_router=None, backend_client=None):
        self.model_router = model_router

    async def run(self, agent_input: AgentInput, plan: Any) -> AgentOutput:
        ctx      = agent_input.context or {}
        model_id = ctx.get("selected_model", "openai/gpt-4o-mini")
        lang     = self._detect_lang(agent_input.message)
        academic_ctx = ctx.get("academic_context", {}) or {}

        content_type = self._detect_content_type(agent_input.message)
        topic        = self._extract_topic(agent_input.message, academic_ctx)
        count        = self._extract_count(agent_input.message, content_type)

        response, structured_data = await self._generate_content(
            content_type=content_type,
            topic=topic,
            count=count,
            message=agent_input.message,
            model_id=model_id,
            lang=lang,
        )

        return AgentOutput(
            status="success",
            response=response,
            data={
                "content_type": content_type,
                "topic": topic,
                "structured_data": structured_data,
                "suggestions": self._get_suggestions(content_type, topic, lang),
            },
        )

    # ── Content generation ────────────────────────────────────────────────

    async def _generate_content(
        self,
        content_type: str,
        topic: str,
        count: int,
        message: str,
        model_id: str,
        lang: str,
    ) -> tuple[str, Any]:
        """Generate the requested content type. Returns (text_response, structured_data)."""
        if not self.model_router:
            return self._fallback(content_type, topic, lang), None

        instruction = _CONTENT_INSTRUCTIONS.get(content_type, _CONTENT_INSTRUCTIONS["examples"])
        instruction = instruction.format(count=count)

        lang_rule = (
            "Respond in Arabic. Use Egyptian dialect for conversational parts, Modern Standard Arabic for technical content."
            if lang == "ar" else
            "Respond in English. Be clear and academic."
        )

        system_prompt = f"""\
You are an expert AI tutor creating educational content for a university student.

{lang_rule}

TOPIC: {topic or 'as specified by the student'}

TASK: {instruction}

RULES:
- Make content accurate, clear, and university-level appropriate
- Examples and exercises should be practical and relatable
- For flashcards: return ONLY the JSON array (no extra text)
- For other content: use clean markdown formatting
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        try:
            response_text = await self.model_router.generate_with_messages(
                messages=messages,
                model_id=model_id,
                max_tokens=1800,
            )

            if not response_text:
                return self._fallback(content_type, topic, lang), None

            # Try to parse structured data for flashcards
            structured = None
            if content_type == "flashcards":
                structured = self._parse_flashcard_json(response_text)
                if structured:
                    # Format nicely for display
                    display = self._format_flashcards_display(structured, lang)
                    return display, structured

            return response_text, structured

        except Exception as exc:
            logger.error("LearningAssistant._generate_content: %s", exc)
            return self._fallback(content_type, topic, lang), None

    # ── Detection helpers ─────────────────────────────────────────────────

    def _detect_content_type(self, message: str) -> str:
        msg = message.lower()
        if any(k in msg for k in ["flashcard", "فلاش", "بطاقة", "بطاقات", "flash"]):
            return "flashcards"
        if any(k in msg for k in ["example", "مثال", "أمثلة", "امثله", "examples"]):
            return "examples"
        if any(k in msg for k in ["exercise", "تمرين", "تمارين", "drill", "practice", "مسألة"]):
            return "exercises"
        if any(k in msg for k in ["summary", "ملخص", "لخص", "summarize", "مراجعة سريعة"]):
            return "summary"
        if any(k in msg for k in ["mnemonic", "حيلة", "حيل حفظ", "trick", "how to remember", "ازاي أحفظ"]):
            return "mnemonics"
        return "examples"  # default

    def _extract_topic(self, message: str, academic_ctx: dict) -> str:
        patterns = [
            r"(?:for|about|on|على|في|عن|لـ|لموضوع|لمادة)\s+(.{3,50}?)(?:\s*$|\?|؟|،|,)",
            r"(?:topic|موضوع|مادة|لـ)\s+(.{3,50})",
        ]
        for pat in patterns:
            m = re.search(pat, message, re.IGNORECASE | re.UNICODE)
            if m:
                return m.group(1).strip()
        return academic_ctx.get("subjectName") or academic_ctx.get("currentSubject") or ""

    def _extract_count(self, message: str, content_type: str) -> int:
        defaults = {
            "flashcards": 15,
            "examples": 5,
            "exercises": 5,
            "summary": 1,
            "mnemonics": 3,
        }
        m = re.search(r"\b(\d+)\b", message)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 30:
                return n
        return defaults.get(content_type, 5)

    def _parse_flashcard_json(self, text: str) -> list[dict] | None:
        """Extract JSON array from LLM response."""
        try:
            # Find JSON array in the response
            match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            # Try parsing the whole thing
            return json.loads(text.strip())
        except Exception:
            return None

    def _format_flashcards_display(self, cards: list[dict], lang: str) -> str:
        """Format parsed flashcards as readable markdown."""
        header = f"تم إنشاء **{len(cards)} بطاقة** 🗂️\n\n" if lang == "ar" else f"Generated **{len(cards)} flashcards** 🗂️\n\n"
        lines = [header]
        for i, card in enumerate(cards[:5], 1):
            front = card.get("front", "")
            back = card.get("back", "")
            lines.append(f"**{i}. {front}**\n→ {back}\n")
        if len(cards) > 5:
            remaining = len(cards) - 5
            more_text = f"\n... و{remaining} بطاقة أخرى محفوظة." if lang == "ar" else f"\n... and {remaining} more cards saved."
            lines.append(more_text)
        return "".join(lines)

    def _fallback(self, content_type: str, topic: str, lang: str) -> str:
        t = topic or ("الموضوع" if lang == "ar" else "this topic")
        if lang == "ar":
            return f"جاري إنشاء {content_type} لموضوع {t}... حاول مرة أخرى بعد لحظة."
        return f"Generating {content_type} for {t}... Please try again in a moment."

    def _get_suggestions(self, content_type: str, topic: str, lang: str) -> list[str]:
        t = topic or ("الموضوع" if lang == "ar" else "this topic")
        if lang == "ar":
            return [
                f"سألني على {t}",
                f"اشرح {t} بأمثلة",
                f"اعمل ملخص لـ{t}",
            ]
        return [
            f"Quiz me on {t}",
            f"Explain {t} with examples",
            f"Summarize {t}",
        ]

    @staticmethod
    def _detect_lang(message: str) -> str:
        arabic_chars = sum(1 for c in message if "؀" <= c <= "ۿ")
        return "ar" if arabic_chars / max(len(message), 1) > 0.2 else "en"
