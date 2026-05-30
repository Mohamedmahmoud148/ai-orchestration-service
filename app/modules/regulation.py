"""
regulation.py — Academic Regulation & Curriculum Advisor (v2: RAG-powered)

Answers questions about the official academic regulation (دليل الطالب / اللائحة)
using semantic search over indexed regulation chunks.

Architecture:
  Primary path  → semantic search via app.services.regulation_indexer (fast, accurate)
  Fallback path → live PDF download + extract (slow, used when RAG is empty)

The student/admin asks a question, we retrieve the top-k relevant chunks
from ChromaDB, and the LLM answers strictly from those chunks. No re-downloading
the entire PDF on every turn, no 10K-character truncation.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import httpx

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger
from app.prompts import load_prompt
from app.services.regulation_indexer import (
    index_all_active_regulations,
    is_any_regulation_indexed,
    search_regulation,
)

_DEFAULT_MODEL = "openai/gpt-4o-mini"
_REGULATION_MAX_TOKENS = 2200          # allow detailed multi-paragraph answers
_RAG_TOP_K = 8                         # diverse chunks → richer answers
_PDF_FALLBACK_CAP = 30_000             # 3x the old cap; only hit when RAG fails


def _get_system_prompt() -> str:
    """Load from app/prompts/regulation_advisor.md; fall back to inline if missing."""
    try:
        return load_prompt("regulation_advisor")
    except Exception as exc:
        logger.warning("RegulationModule: prompt load failed — using inline fallback: %s", exc)
        return _SYSTEM_PROMPT_FALLBACK


_SYSTEM_PROMPT_FALLBACK = """\
أنت "مرشد" — مرشد أكاديمي ودود وذكي في الجامعة. بتساعد الطلاب يفهموا اللائحة الأكاديمية الرسمية بسهولة، كأنك زميل أكبر بيشرحلهم.

🎯 الأسلوب:
- ودود وطبيعي مع لمسة احترافية. ممنوع الجفاف الروبوتي.
- نفس لغة السائل: عربي مع عربي، إنجليزي مع إنجليزي. ممنوع الخلط.
- استخدم 📚 أو ✅ أو 💡 emoji محسوبة لما تضيف معنى — مش في كل سطر.

📋 قواعد المحتوى (صارمة):
1. أجب فقط من المقاطع المُقدَّمة من اللائحة — ممنوع تستخدم معرفتك العامة عن جامعات تانية.
2. لو السؤال مش متغطى في المقاطع المتاحة → قول "الجزء ده مش موجود في المقاطع اللي قدرت أوصلها من اللائحة" بدل اختلاق إجابة.
3. ردك لازم يكون مفصّل ومنظم: عنوان قصير، نقاط واضحة، أمثلة من النص لما متاحة.
4. لو السؤال عن مواد سنة معينة → اذكر الأسماء والساعات والـ prerequisites لو موجودة.
5. لو السؤال عام ("لخص اللائحة") → قدّم ملخص شامل من المقاطع المتاحة: الأقسام، عدد الساعات، شروط التخرج، نظام التقدير.
6. لو في رقم/شرط/نسبة في النص → اقتبسها حرفياً مع ذكر السطر اللي جاية منه.
7. لما يناسب، اختم بسؤال صغير: "تحب أركّز على جزء معين؟" — مش كل مرة.
8. الدقة أهم من الإطالة، لكن مكنش بخيل في التفاصيل لو الطالب طلب شرح كامل.\
"""


# ── Legacy PDF fallback (used only when RAG is empty) ────────────────────────


async def _fetch_pdf_text(file_url: str, auth_header: Optional[str] = None) -> str:
    if not file_url or not file_url.startswith("http"):
        return ""
    try:
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header
        async with httpx.AsyncClient(timeout=40.0, follow_redirects=True) as client:
            resp = await client.get(file_url, headers=headers)
            resp.raise_for_status()

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

        return (text or "").strip()[:_PDF_FALLBACK_CAP]

    except Exception as exc:
        logger.warning("RegulationModule: PDF fetch failed %s — %s", file_url[:80], exc)
        return ""


# ── Module ────────────────────────────────────────────────────────────────────


class RegulationModule:
    """
    RAG-first academic regulation advisor.

    Pipeline:
      1. Ensure regulations are indexed (auto-trigger first-time index if not).
      2. Semantic search top-k chunks for the user's question.
      3. LLM answers from those chunks with a detailed, structured response.
      4. If RAG returns nothing usable → fallback to the legacy live PDF read.
    """

    def __init__(self, model_router, backend_client) -> None:
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        ctx      = agent_input.context or {}
        model_id = ctx.get("selected_model") or _DEFAULT_MODEL
        auth     = agent_input.auth_header

        logger.info(
            "RegulationModule: user=%s message=%r",
            agent_input.user_id, agent_input.message[:100],
        )

        # ── 0. Auto-trigger first-time indexing if vector store has no regulations
        try:
            indexed = await is_any_regulation_indexed()
        except Exception:
            indexed = False

        if not indexed:
            logger.info("RegulationModule: no regulations indexed yet — triggering index")
            try:
                await index_all_active_regulations(auth_header=auth)
            except Exception as exc:
                logger.warning("RegulationModule: auto-index failed (non-fatal) — %s", exc)

        # ── 1. RAG semantic search ────────────────────────────────────────────
        chunks = await search_regulation(
            query=agent_input.message,
            top_k=_RAG_TOP_K,
        )

        regulation_text = ""
        source = "rag"
        if chunks:
            # Order by score, build a numbered passage block with citation metadata
            chunks_sorted = sorted(chunks, key=lambda c: c["score"], reverse=True)
            passage_lines: List[str] = []
            for i, ch in enumerate(chunks_sorted, 1):
                meta  = ch.get("metadata") or {}
                title = meta.get("materialTitle") or "اللائحة الأكاديمية"
                score = ch.get("score", 0)
                chunk_idx = meta.get("chunkIndex", "")
                # Include citation reference so LLM can cite it accurately
                citation = f"[مقطع {i} — المصدر: {title}"
                if chunk_idx != "":
                    citation += f", الفقرة {chunk_idx}"
                citation += f" (صلة: {score:.2f})]"
                passage_lines.append(f"{citation}\n{ch['content']}")
            regulation_text = "\n\n".join(passage_lines)
            logger.info(
                "RegulationModule: RAG returned %d chunks (top score=%.3f)",
                len(chunks), chunks_sorted[0]["score"],
            )

        # ── 2. Fallback: live PDF read if RAG is empty ────────────────────────
        if not regulation_text:
            source = "pdf_fallback"
            logger.info("RegulationModule: RAG empty — falling back to live PDF read")
            regulation_text = await self._fallback_live_pdf(auth)

        if not regulation_text:
            return AgentOutput(
                status="success",
                response=(
                    "📄 معنديش حالياً نص اللائحة الأكاديمية متاح. ممكن الملف مش مرفوع "
                    "في السيستم أو محمي. ابعتلي رسالة تاني بعد ما الإدارة ترفع اللائحة."
                ),
            )

        # ── 3. Build conversation-aware prompt ────────────────────────────────
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
            f"=== مقاطع مأخوذة من اللائحة الأكاديمية الرسمية ===\n"
            f"{regulation_text}\n"
            f"=== نهاية المقاطع ===\n\n"
            "تعليمات الإجابة:\n"
            "1. أجب فقط مما هو مكتوب في المقاطع أعلاه.\n"
            "2. لكل رقم أو شرط تذكره، اذكر مصدره: 'وفقاً لـ [اسم المصدر من تعريف المقطع]'.\n"
            "3. لو السؤال يحتوي ضمير ('اشرحها'، 'لخصها') → افهم المقصود من السياق السابق.\n"
            "4. لو المعلومة غير موجودة في المقاطع → قل ذلك بصراحة ولا تخمّن.\n"
            "5. الإجابة المنظمة: عنوان قصير → نقاط → خلاصة.\n"
        )

        # ── 4. LLM call with long max_tokens ──────────────────────────────────
        try:
            answer = await self.model_router.generate(
                prompt=user_prompt,
                system_instruction=_get_system_prompt(),
                model_id=model_id,
                max_tokens=_REGULATION_MAX_TOKENS,
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
                "module":            "RegulationModule",
                "source":            source,
                "chunks_returned":   len(chunks) if chunks else 0,
                "text_chars":        len(regulation_text),
                "model_used":        model_id,
                "max_tokens":        _REGULATION_MAX_TOKENS,
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Fallback path: live PDF download (only used when RAG is empty)
    # ──────────────────────────────────────────────────────────────────────

    async def _fallback_live_pdf(self, auth: Optional[str]) -> str:
        """Legacy path — pull regulations from .NET and read up to N PDFs."""
        try:
            regs_raw = await self.backend_client.fetch(
                route="/api/Regulations",
                auth_header=auth,
                params={"page": 1, "size": 10},
            )
        except Exception as exc:
            logger.error("RegulationModule._fallback_live_pdf: backend fetch failed — %s", exc)
            return ""

        if isinstance(regs_raw, dict):
            regs_list: List[Any] = regs_raw.get("data") or regs_raw.get("items") or []
        elif isinstance(regs_raw, list):
            regs_list = regs_raw
        else:
            regs_list = []

        if not regs_list:
            return ""

        parts: List[str] = []
        for reg in regs_list[:3]:
            if not isinstance(reg, dict):
                continue
            title    = reg.get("title") or "لائحة"
            content  = reg.get("content") or ""
            file_url = reg.get("fileUrl") or reg.get("filePath") or ""
            if file_url:
                pdf_text = await _fetch_pdf_text(file_url, auth)
                if pdf_text:
                    parts.append(f"=== {title} ===\n{pdf_text}")
                    continue
            if content:
                parts.append(f"=== {title} (وصف مختصر) ===\n{content}")

        return "\n\n".join(parts)
