"""
app/modules/deep_pdf_study.py  —  Deep PDF Study Mode

Handles large PDFs (50-200 pages) properly using hierarchical summarization.

Strategy:
  1. Extract ALL pages individually (never truncate)
  2. Get real page count from PDF metadata
  3. For short PDFs (≤ ~15K chars):  send full text to LLM in one call
  4. For long PDFs (> 15K chars):    hierarchical summarization:
       - Split into chunks of 5 pages
       - Summarize each chunk → chunk summaries
       - Combine chunk summaries → section summaries
       - Combine section summaries → final answer
  5. For "explain page N" → extract only that page

This module is called by MaterialExplanationModule when the full text is too large.
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import logger

# ── Limits ────────────────────────────────────────────────────────────────────
_SINGLE_CALL_CHAR_LIMIT  = 15_000   # below this: send all text in one call
_CHUNK_PAGES             = 5        # pages per chunk for hierarchical mode
_CHUNK_CHAR_LIMIT        = 12_000   # max chars per chunk sent to LLM
_MAX_TOTAL_CHARS         = 300_000  # hard ceiling on extraction (very large PDFs)


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION — PAGE LEVEL
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(data: bytes) -> List[str]:
    """
    Extract text from every page of a PDF.
    Returns a list where index 0 = page 1.
    Never truncates.
    """
    # Try pdfminer page-by-page (best quality)
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer

        pages: List[str] = []
        for page_layout in extract_pages(io.BytesIO(data)):
            page_text = ""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text += element.get_text()
            pages.append(page_text.strip())
        if pages and any(p for p in pages):
            logger.info("deep_pdf: pdfminer extracted %d pages", len(pages))
            return pages
    except Exception as e:
        logger.debug("deep_pdf: pdfminer page extraction failed: %s", e)

    # Fallback: pypdf page-by-page
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        logger.info("deep_pdf: pypdf extracted %d pages", len(pages))
        return pages
    except Exception as e:
        logger.debug("deep_pdf: pypdf failed: %s", e)

    return []


def get_pdf_page_count(data: bytes) -> int:
    """Get the real page count from PDF metadata. Never guesses."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        return len(reader.pages)
    except Exception:
        pass
    try:
        from pdfminer.pdfpage import PDFPage
        count = sum(1 for _ in PDFPage.get_pages(io.BytesIO(data)))
        return count
    except Exception:
        return 0


def get_pdf_stats(data: bytes) -> Dict[str, Any]:
    """Extract PDF stats: page count, total chars, total words."""
    pages = extract_pdf_pages(data)
    total_chars = sum(len(p) for p in pages)
    total_words = sum(len(p.split()) for p in pages)
    return {
        "page_count":    len(pages),
        "total_chars":   total_chars,
        "total_words":   total_words,
        "pages":         pages,
        "chars_per_page": [len(p) for p in pages],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL SUMMARIZATION
# ─────────────────────────────────────────────────────────────────────────────

async def summarize_large_pdf(
    pages: List[str],
    user_question: str,
    model_router: Any,
    model_id: str = "openai/gpt-4o-mini",
    subject_name: str = "",
    student_name: str = "",
) -> str:
    """
    Summarize a large PDF using hierarchical chunking.

    Flow:
      pages → chunk groups of 5 → summarize each chunk
           → combine summaries → final answer
    """
    total_pages = len(pages)
    total_chars = sum(len(p) for p in pages)

    logger.info(
        "deep_pdf: hierarchical mode — %d pages, %d chars, question=%r",
        total_pages, total_chars, user_question[:80]
    )

    # ── Detect if user asked about a specific page ────────────────────────────
    page_match = re.search(r"(?:صفحة|page|pg\.?)\s*(\d+)", user_question, re.IGNORECASE)
    if page_match:
        page_num = int(page_match.group(1))
        if 1 <= page_num <= total_pages:
            page_text = pages[page_num - 1]
            if page_text.strip():
                return await _explain_single_page(
                    page_num, page_text, user_question, model_router, model_id, student_name
                )

    # ── Short PDF: send all in one call ──────────────────────────────────────
    if total_chars <= _SINGLE_CALL_CHAR_LIMIT:
        full_text = "\n\n".join(
            f"[صفحة {i+1}]\n{p}" for i, p in enumerate(pages) if p.strip()
        )
        return await _single_call_explain(
            full_text, total_pages, user_question, model_router, model_id, subject_name, student_name
        )

    # ── Large PDF: hierarchical summarization ─────────────────────────────────
    chunk_summaries: List[str] = []

    # Group pages into chunks of _CHUNK_PAGES
    for chunk_start in range(0, total_pages, _CHUNK_PAGES):
        chunk_end = min(chunk_start + _CHUNK_PAGES, total_pages)
        chunk_pages = pages[chunk_start:chunk_end]

        chunk_text = "\n\n".join(
            f"[صفحة {chunk_start + i + 1}]\n{p}"
            for i, p in enumerate(chunk_pages)
            if p.strip()
        )

        if not chunk_text.strip():
            continue

        # Truncate chunk if too large (shouldn't happen with 5 pages but safety)
        chunk_text = chunk_text[:_CHUNK_CHAR_LIMIT]

        prompt = (
            f"هذه صفحات {chunk_start+1}–{chunk_end} من ملف '{subject_name or 'المادة'}' "
            f"(إجمالي {total_pages} صفحة).\n\n"
            f"{chunk_text}\n\n"
            f"قدم ملخصاً شاملاً لهذه الصفحات يشمل:\n"
            f"- النقاط الرئيسية\n- المفاهيم الأساسية\n- أي معادلات أو تعريفات مهمة\n"
            f"استخدم نفس لغة المستخدم."
        )

        try:
            summary = await model_router.generate(
                prompt=prompt,
                system_instruction=(
                    "أنت مساعد أكاديمي متخصص. لخّص المحتوى بدقة وشمولية. "
                    "لا تحذف معلومات مهمة."
                ),
                model_id=model_id,
            )
            if summary:
                chunk_summaries.append(
                    f"**صفحات {chunk_start+1}–{chunk_end}:**\n{summary}"
                )
                logger.info(
                    "deep_pdf: chunk %d-%d summarized (%d chars)",
                    chunk_start+1, chunk_end, len(summary)
                )
        except Exception as e:
            logger.warning("deep_pdf: chunk %d-%d summary failed: %s", chunk_start+1, chunk_end, e)
            chunk_summaries.append(
                f"**صفحات {chunk_start+1}–{chunk_end}:** (تعذّر ملخص هذا الجزء)"
            )

    if not chunk_summaries:
        return "تعذّر تحليل محتوى الملف. يرجى المحاولة مرة أخرى."

    # ── Final synthesis call ──────────────────────────────────────────────────
    all_summaries = "\n\n".join(chunk_summaries)

    greeting = f"{student_name}، " if student_name else ""
    final_prompt = (
        f"{greeting}فيما يلي ملخصات كل أجزاء الملف '{subject_name or 'المادة'}' "
        f"({total_pages} صفحة):\n\n"
        f"{all_summaries[:14_000]}\n\n"
        f"سؤال الطالب: {user_question}\n\n"
        f"بناءً على هذه الملخصات الشاملة، أجب على سؤال الطالب بشكل مفصّل ومنظّم. "
        f"إذا طلب شرحاً كاملاً، قدّم ملخصاً تفصيلياً لكل الملف مقسّماً بوضوح."
    )

    final_answer = await model_router.generate(
        prompt=final_prompt,
        system_instruction=(
            "أنت مدرّس أكاديمي محترف. قدّم إجابة شاملة ومفيدة تعتمد على ملخصات الملف المقدّمة. "
            "نظّم الإجابة بعناوين وفقرات واضحة. استخدم نفس لغة الطالب."
        ),
        model_id=model_id,
    )

    return final_answer or "\n\n".join(chunk_summaries)


async def _single_call_explain(
    full_text: str,
    page_count: int,
    user_question: str,
    model_router: Any,
    model_id: str,
    subject_name: str,
    student_name: str,
) -> str:
    greeting = f"{student_name}، " if student_name else ""
    prompt = (
        f"{greeting}فيما يلي المحتوى الكامل لملف '{subject_name or 'المادة'}' "
        f"({page_count} صفحة):\n\n"
        f"{full_text}\n\n"
        f"سؤال الطالب: {user_question}\n\n"
        "أجب بشكل مفصّل ومنظّم. لو سأل عن كم صفحة، اذكر العدد الصحيح. "
        "استخدم نفس لغة الطالب."
    )
    result = await model_router.generate(
        prompt=prompt,
        system_instruction=(
            "أنت مساعد أكاديمي. أجب بدقة وشمولية بناءً على المحتوى المقدّم فقط. "
            "نظّم الإجابة بعناوين وفقرات."
        ),
        model_id=model_id,
    )
    return result or ""


async def _explain_single_page(
    page_num: int,
    page_text: str,
    user_question: str,
    model_router: Any,
    model_id: str,
    student_name: str,
) -> str:
    greeting = f"{student_name}، " if student_name else ""
    prompt = (
        f"{greeting}هذا محتوى الصفحة {page_num}:\n\n"
        f"{page_text[:8_000]}\n\n"
        f"سؤال الطالب: {user_question}\n\n"
        "اشرح محتوى هذه الصفحة بالتفصيل."
    )
    result = await model_router.generate(
        prompt=prompt,
        system_instruction="أنت مدرّس أكاديمي. اشرح المحتوى بوضوح وتفصيل. استخدم لغة الطالب.",
        model_id=model_id,
    )
    return result or ""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: check if question is about the full document
# ─────────────────────────────────────────────────────────────────────────────

def is_full_doc_request(question: str) -> bool:
    """Returns True if the user wants a full explanation/summary of the document."""
    patterns = [
        r"اشرح.*(كل|كامل|الملف|الكتاب|المادة)",
        r"لخص.*(كل|كامل|الملف)",
        r"اقرأ.*(كل|كامل|الملف)",
        r"(كم|عدد)\s*(صفح|page)",
        r"explain.*(entire|whole|full|all|complete)",
        r"summarize.*(entire|whole|full|all)",
        r"كم صفحة",
        r"how many pages",
    ]
    q_lower = question.lower()
    return any(re.search(p, q_lower) for p in patterns)
