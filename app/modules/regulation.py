"""
regulation.py — Academic Regulation & Curriculum Advisor

Fetches the university regulation document (دليل الطالب / اللائحة) from the backend,
reads the PDF, and answers student/admin questions from the actual document content.

This is the core academic advisor feature: students ask about curriculum, subjects
per year, credit hours, graduation requirements — AI answers ONLY from the real document.

Flow:
  1. GET /api/Regulations  → get regulation list with fileUrl
  2. Download & parse the PDF
  3. LLM answers the user's question strictly from the document text
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import httpx

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger

_DEFAULT_MODEL = "openai/gpt-4o-mini"

_SYSTEM_PROMPT = """\
أنت "مرشد" — مرشد أكاديمي ودود وذكي في الجامعة. بتساعد الطلاب يفهموا اللائحة الأكاديمية الرسمية بسهولة، كأنك زميل أكبر بيشرحلهم.

🎯 الأسلوب:
- ودود وطبيعي مع لمسة احترافية. ممنوع الجفاف الروبوتي.
- نفس لغة السائل: عربي مع عربي، إنجليزي مع إنجليزي. ممنوع الخلط.
- استخدم 📚 أو ✅ أو 💡 emoji محسوبة لما تضيف معنى — مش في كل سطر.

📋 قواعد المحتوى (صارمة):
1. أجب فقط من محتوى اللائحة المُقدَّم — ممنوع تستخدم معرفتك العامة.
2. لو السؤال مش في اللائحة → قول "المعلومة دي مش موجودة في اللائحة المتاحة عندي" بدل اختلاق إجابة.
3. نظّم إجابتك: عنوان قصير، نقاط واضحة، أمثلة من النص لو متاح.
4. لو السؤال عن مواد سنة معينة → اذكر الأسماء والساعات والـ prerequisites لو موجودة.
5. لو السؤال عام ("لخص اللائحة") → قدّم ملخص شامل ومنظم: الأقسام الرئيسية، عدد الساعات، شروط التخرج، نظام التقدير.
6. اختم بسؤال صغير لو مناسب: "تحب أركّز على جزء معين؟" — مش كل مرة.
7. الدقة أهم من الإطالة. ممنوع اختلاق أي بيانات.\
"""


async def _fetch_pdf_text(file_url: str, auth_header: Optional[str] = None) -> str:
    """Download and extract text from a PDF URL."""
    if not file_url or not file_url.startswith("http"):
        return ""
    try:
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header
        async with httpx.AsyncClient(timeout=40.0, follow_redirects=True) as client:
            resp = await client.get(file_url, headers=headers)
            resp.raise_for_status()

        # Try pdfminer first, pypdf as fallback
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(io.BytesIO(resp.content)) or ""
        except Exception:
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(resp.content))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                logger.warning("RegulationModule: PDF parse failed — %s", exc)
                return ""

        text = text.strip()
        logger.info("RegulationModule: extracted %d chars from regulation PDF", len(text))
        return text[:10_000]  # cap to avoid token overflow

    except Exception as exc:
        logger.warning("RegulationModule: failed to fetch PDF %s — %s", file_url[:80], exc)
        return ""


class RegulationModule:
    """
    Academic regulation advisor — fetches the official regulation PDF from the
    backend and answers questions strictly from its content.
    """

    def __init__(self, model_router, backend_client) -> None:
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        ctx        = agent_input.context or {}
        model_id   = ctx.get("selected_model") or _DEFAULT_MODEL
        auth       = agent_input.auth_header

        logger.info("RegulationModule: fetching regulations for user=%s", agent_input.user_id)

        # ── 1. Fetch regulation list ───────────────────────────────────────────
        try:
            regs_raw = await self.backend_client.fetch(
                route="/api/Regulations",
                auth_header=auth,
                params={"page": 1, "size": 10},
            )
        except Exception as exc:
            logger.error("RegulationModule: backend fetch failed — %s", exc)
            return AgentOutput(
                status="failed",
                response="مش قادر أجيب اللائحة دلوقتي. حاول تاني بعد ثوانٍ.",
            )

        # Unwrap ApiResponse envelope  { success, data: [...] }
        if isinstance(regs_raw, dict):
            regs_list: List[Any] = regs_raw.get("data") or regs_raw.get("items") or []
        elif isinstance(regs_raw, list):
            regs_list = regs_raw
        else:
            regs_list = []

        if not regs_list:
            return AgentOutput(
                status="success",
                response="مفيش لوائح أكاديمية متاحة في السيستم دلوقتي.",
            )

        # ── 2. Read each regulation's PDF ──────────────────────────────────────
        all_text_parts: List[str] = []
        for reg in regs_list[:3]:
            title   = reg.get("title") or "لائحة"
            content = reg.get("content") or ""
            fileurl = reg.get("fileUrl") or reg.get("filePath") or ""

            if fileurl:
                logger.info("RegulationModule: reading PDF for '%s'", title)
                pdf_text = await _fetch_pdf_text(fileurl, auth)
                if pdf_text:
                    all_text_parts.append(f"=== {title} ===\n{pdf_text}")
                    continue

            # Fallback: use inline content field if no PDF
            if content:
                all_text_parts.append(f"=== {title} ===\n{content}")

        if not all_text_parts:
            return AgentOutput(
                status="success",
                response=(
                    "اللائحة موجودة في السيستم بس مش قادر أقراها دلوقتي.\n"
                    "اسم اللائحة: " + (regs_list[0].get("title") or "غير محدد")
                ),
            )

        regulation_text = "\n\n".join(all_text_parts)

        # ── 3. LLM answers from document only ─────────────────────────────────
        # Inject the last 4 conversation turns so vague follow-ups like
        # "اشرحهالي / لخصها / explain it" resolve naturally against earlier context.
        raw_history = (ctx.get("history") or [])[-4:]
        history_block = ""
        if raw_history:
            lines = []
            for t in raw_history:
                tr = t.get("role", "user")
                tc = str(t.get("content", "")).strip()
                if tc:
                    speaker = "المستخدم" if tr == "user" else "المساعد"
                    lines.append(f"{speaker}: {tc[:400]}")
            if lines:
                history_block = (
                    "\n=== السياق السابق للمحادثة (استخدمه لفهم الإشارات والضمائر) ===\n"
                    + "\n".join(lines)
                    + "\n=== نهاية السياق ===\n\n"
                )

        user_prompt = (
            f"{history_block}"
            f"سؤال المستخدم الحالي: {agent_input.message}\n\n"
            f"=== محتوى اللائحة الأكاديمية الرسمية ===\n"
            f"{regulation_text}\n"
            f"=== نهاية اللائحة ===\n\n"
            "بناءً على محتوى اللائحة أعلاه فقط، أجب على سؤال المستخدم بشكل واضح ومنظم. "
            "لو السؤال قصير أو فيه ضمير (زي 'اشرحهالي' أو 'لخصها')، افهم المقصود من السياق السابق "
            "وقدّم ملخصاً شاملاً للائحة بدلاً من سؤال المستخدم عن المادة."
        )

        try:
            answer = await self.model_router.generate(
                prompt=user_prompt,
                system_instruction=_SYSTEM_PROMPT,
                model_id=model_id,
            )
        except Exception as exc:
            logger.error("RegulationModule: LLM call failed — %s", exc)
            return AgentOutput(
                status="failed",
                response="تعذّر توليد الإجابة. حاول مرة أخرى.",
            )

        if not answer:
            return AgentOutput(
                status="failed",
                response="مش قادر أولد إجابة دلوقتي. حاول تاني.",
            )

        return AgentOutput(
            status="success",
            response=answer,
            data={
                "module":           "RegulationModule",
                "regulations_read": len(all_text_parts),
                "text_chars":       len(regulation_text),
                "model_used":       model_id,
            },
        )
