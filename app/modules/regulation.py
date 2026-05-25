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
أنت مرشد أكاديمي ذكي في جامعة. مهمتك الإجابة على أسئلة الطلاب والإداريين بناءً على اللائحة الأكاديمية الرسمية للجامعة.

قواعد صارمة:
1. أجب فقط من محتوى اللائحة المُقدَّم لك — لا تستخدم معرفتك العامة.
2. لو السؤال مش موجود في اللائحة → قول بوضوح "هذه المعلومة غير موجودة في اللائحة المتاحة."
3. نظّم إجابتك: عناوين + نقاط + أمثلة من النص.
4. أجب بنفس لغة السؤال (عربي أو إنجليزي).
5. لو السؤال عن مواد سنة معينة → اذكر أسماء المواد وساعاتها بوضوح.
6. لا تخترع أي بيانات — الدقة أهم من الإطالة.\
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
        user_prompt = (
            f"سؤال المستخدم: {agent_input.message}\n\n"
            f"=== محتوى اللائحة الأكاديمية الرسمية ===\n"
            f"{regulation_text}\n"
            f"=== نهاية اللائحة ===\n\n"
            "بناءً على محتوى اللائحة أعلاه فقط، أجب على سؤال المستخدم بشكل واضح ومنظم."
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
