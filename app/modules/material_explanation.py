"""
app/modules/material_explanation.py  —  v2.0

Material Explanation Module — intent: material_explanation

STRICT DATA-FIRST policy:
  - NEVER generate an explanation from general LLM knowledge if real materials exist.
  - ALWAYS fetch materials from the backend before responding.
  - If NO materials found → return a specific, actionable bilingual message.

Flow:
  1. Resolve subjectOfferingId from academic_context (mandatory).
  2. GET /api/Materials/by-offering/{subjectOfferingId}
  3. If materials returned:
       a. Extract text from each material item (content field, fileUrl fetch, or PDF bytes).
       b. Pass material text to LLM with strict instruction:
          "Explain/summarize ONLY using the provided material. Do NOT use general knowledge."
       c. Tailor framing to role:
          - student → "Here's a simplified explanation of your course material..."
          - doctor  → "Here's an academic summary of the course material for your review..."
  4. If NO materials:
       → Return bilingual message (Arabic + English) with actionable suggestion.

Trigger keywords (planner routes these):
  - English: "explain course", "summarize material", "explain subject", "summarize course",
             "explain this topic", "what does this material say", "understand this subject"
  - Arabic:  "شرح مادة", "لخص المادة", "اشرح المادة", "ملخص المادة",
             "فهم المادة", "شرح الموضوع", "اشرح موضوع", "ما محتوى المادة"

Architecture rules
------------------
- Backend is the single source of truth for material content.
- LLM is only used for explaining/summarizing — never for inventing facts.
- fileUrl fields in the material response are optionally fetched as fallback text.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import httpx

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger

_DEFAULT_MODEL = "openai/gpt-4o-mini"

# ── Bilingual "no materials" messages ─────────────────────────────────────────
_NO_MATERIALS_EN = (
    "No materials were found for this subject. "
    "Please ask your instructor to upload course materials, then try again."
)
_NO_MATERIALS_AR = (
    "لم يتم العثور على مواد لهذه المادة الدراسية. "
    "يرجى مطالبة المحاضر برفع المواد التعليمية والمحاولة مرة أخرى."
)
_NO_MATERIALS_BILINGUAL = f"{_NO_MATERIALS_EN}\n\n{_NO_MATERIALS_AR}"

# ── Role-aware system prompts ─────────────────────────────────────────────────
_STUDENT_SYSTEM_PROMPT = """\
You are an academic tutor explaining university course material to a student.

CRITICAL RULES — follow exactly:
1. Use ONLY the course material provided below. Do NOT add information from general knowledge.
2. If the material does not contain enough information to answer the question, say:
   "The uploaded material does not cover this topic. Please ask your instructor for more details."
3. Structure your explanation clearly: use headings, bullet points, and examples from the text.
4. Be concise but thorough. Target length: 200-400 words.
5. Use simple, friendly language appropriate for a student.
6. Respond in the same language as the student's question.\
"""

_DOCTOR_SYSTEM_PROMPT = """\
You are an academic assistant helping a faculty member review or summarize course material.

CRITICAL RULES — follow exactly:
1. Use ONLY the course material provided. Do NOT add information from general knowledge.
2. Structure your summary professionally: key topics, learning objectives, assessment points.
3. Be precise and concise \u2014 the faculty member needs actionable academic insight.
4. If the material is insufficient, state: "The uploaded material is limited. Consider supplementing."
5. Respond in the same language as the faculty member's request.\
"""

_ADMIN_SYSTEM_PROMPT = """\
You are an AI assistant summarizing university course material for administrative review.

CRITICAL RULES:
1. Use ONLY the course material provided.
2. Provide a structured overview: title, key topics, volume (number of items).
3. Flag any concerns (e.g., missing content, very short material).
4. Respond in the same language as the request.\
"""


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from PDF bytes using pdfminer (preferred) or pypdf as fallback."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(io.BytesIO(data)) or ""
    except Exception:
        pass
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        logger.warning("MaterialExplanationModule: PDF extraction failed — %s", exc)
        return ""


async def _fetch_file_url_text(file_url: str, auth_header: Optional[str]) -> str:
    """
    Optionally fetch a fileUrl returned in the materials response.

    If the URL points to a PDF, extract its text.  If it's plain text,
    return as-is.  Returns empty string on any failure (non-fatal).
    """
    if not file_url or not file_url.startswith("http"):
        return ""
    try:
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header
        async with httpx.AsyncClient() as client:
            resp = await client.get(file_url, headers=headers, timeout=20.0)
            resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type or file_url.lower().endswith(".pdf"):
            return _extract_pdf_text(resp.content)
        if "text" in content_type or "json" in content_type:
            return resp.text[:5_000]
        return ""
    except Exception as exc:
        logger.warning(
            "MaterialExplanationModule: fileUrl fetch failed (%s) — %s", file_url[:80], exc
        )
        return ""


async def _collect_material_text(
    materials_data: Any,
    auth_header: Optional[str],
) -> str:
    """
    Collect usable text from the backend material response.

    Backend may return:
      - A list of material objects: [{content, title, fileUrl, ...}, ...]
      - A single dict with content/fileUrl
      - Raw bytes (PDF)

    Attempts in order:
      1. content / text / description field (fastest)
      2. fileUrl fetch (fallback — only if content is empty)
    """
    if not materials_data:
        return ""

    texts: List[str] = []

    # ── List of material objects ──────────────────────────────────────────
    if isinstance(materials_data, list):
        for item in materials_data[:5]:  # cap to avoid token overflow
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or ""
            text = (
                item.get("content")
                or item.get("text")
                or item.get("description")
                or ""
            )
            if text and isinstance(text, str):
                texts.append(f"[{title}]\n{text}" if title else text)
                continue

            # Fallback: fetch fileUrl if content field is empty
            file_url = (
                item.get("fileUrl")
                or item.get("url")
                or item.get("filePath")
                or ""
            )
            if file_url:
                fetched = await _fetch_file_url_text(file_url, auth_header)
                if fetched:
                    texts.append(f"[{title}]\n{fetched}" if title else fetched)

    # ── Single dict ────────────────────────────────────────────────────────
    elif isinstance(materials_data, dict):
        if "_raw_bytes" in materials_data:
            pdf_text = _extract_pdf_text(materials_data["_raw_bytes"])
            if pdf_text:
                texts.append(pdf_text)
        else:
            text = (
                materials_data.get("content")
                or materials_data.get("text")
                or ""
            )
            if text:
                texts.append(str(text))
            elif materials_data.get("fileUrl"):
                fetched = await _fetch_file_url_text(
                    materials_data["fileUrl"], auth_header
                )
                if fetched:
                    texts.append(fetched)

    # ── Raw bytes (PDF returned directly) ─────────────────────────────────
    elif isinstance(materials_data, bytes):
        pdf_text = _extract_pdf_text(materials_data)
        if pdf_text:
            texts.append(pdf_text)

    combined = "\n\n".join(texts).strip()
    # Safe token limit (~4 000 words ≈ 5 000 chars)
    return combined[:5_000]


class MaterialExplanationModule:
    """
    Fetches real course materials from the backend and uses the LLM to
    explain or summarize them based on the user's question.

    This module NEVER answers from general LLM knowledge when materials exist.
    Backend data is always fetched first; LLM is strictly the presentation layer.
    """

    def __init__(self, model_router, backend_client) -> None:
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        ctx          = agent_input.context or {}
        model_id     = ctx.get("selected_model") or _DEFAULT_MODEL
        role         = ctx.get("role", "student")
        academic_ctx: Dict[str, Any] = ctx.get("academic_context", {}) or {}

        # ── 1. Resolve subjectOfferingId ──────────────────────────────────────
        offering_id: Optional[str] = (
            academic_ctx.get("subjectOfferingId")
            or academic_ctx.get("courseId")
            or ctx.get("subjectOfferingId")
        )

        # Try from plan steps/params if not in context
        if not offering_id and plan:
            if hasattr(plan, "exam_params") and plan.exam_params:
                offering_id = getattr(plan.exam_params, "subjectOfferingId", None)
            if not offering_id and hasattr(plan, "steps"):
                for step in (plan.steps or []):
                    offering_id = (step.input_payload or {}).get("subjectOfferingId")
                    if offering_id:
                        break

        if not offering_id:
            logger.warning(
                "MaterialExplanationModule: subjectOfferingId not found in context"
            )
            enrolled = academic_ctx.get("enrolledCourses") or academic_ctx.get("courses") or []
            if enrolled:
                course_list = ", ".join(
                    (c.get("name") or c.get("subjectName") or str(c))
                    for c in (enrolled[:5] if isinstance(enrolled, list) else [])
                )
                suggestion = (
                    f"I can see you're enrolled in: {course_list}. "
                    "Please specify which subject you'd like explained "
                    "(e.g., 'Explain Data Structures')."
                    "\n\n"
                    f"يمكنني رؤية المواد المسجّلة لديك: {course_list}. "
                    "يرجى تحديد المادة التي تريد شرحها "
                    "(مثال: 'اشرح مادة هياكل البيانات')."
                )
            else:
                suggestion = (
                    "Please specify the subject name "
                    "(e.g., 'Explain Machine Learning - Level 3')."
                    "\n\n"
                    "يرجى تحديد اسم المادة الدراسية "
                    "(مثال: 'اشرح مادة تعلم الآلة - المستوى الثالث')."
                )

            return AgentOutput(
                status="failed",
                response=(
                    "I need to know which subject you're asking about before I can "
                    f"fetch the materials.\n\n{suggestion}"
                ),
            )

        logger.info(
            "MaterialExplanationModule: fetching materials for subjectOfferingId=%s role=%s",
            offering_id, role,
        )

        # ── 2. Fetch materials from backend ───────────────────────────────────
        try:
            materials_data = await self.backend_client.fetch(
                route=f"/api/Materials/by-offering/{offering_id}",
                auth_header=agent_input.auth_header,
            )
            logger.info(
                "MaterialExplanationModule: backend response type=%s",
                type(materials_data).__name__,
            )
        except Exception as exc:
            logger.error(
                "MaterialExplanationModule: backend fetch failed — %s", exc
            )
            return AgentOutput(
                status="failed",
                response=(
                    "I couldn't retrieve the course materials at this time. "
                    "Please try again in a moment.\n\n"
                    "تعذّر الوصول إلى مواد المادة الدراسية. يرجى المحاولة مرة أخرى."
                ),
            )

        # ── 3. Extract usable text (including fileUrl fallback) ───────────────
        material_text = await _collect_material_text(
            materials_data, agent_input.auth_header
        )

        if not material_text:
            logger.info(
                "MaterialExplanationModule: no material text found for offering=%s",
                offering_id,
            )
            return AgentOutput(
                status="success",
                response=_NO_MATERIALS_BILINGUAL,
                data={
                    "module":       "MaterialExplanationModule",
                    "offering_id":  offering_id,
                    "has_material": False,
                },
            )

        logger.info(
            "MaterialExplanationModule: %d chars of material extracted — sending to LLM",
            len(material_text),
        )

        # ── 4. LLM explanation — STRICT: only use provided material ───────────
        # Select role-appropriate system prompt
        if role == "doctor":
            system_prompt = _DOCTOR_SYSTEM_PROMPT
            role_framing  = (
                "The following course material is from your subject offering. "
                "Please provide an academic summary for faculty review."
            )
        elif role == "admin":
            system_prompt = _ADMIN_SYSTEM_PROMPT
            role_framing  = "The following course material is being reviewed for administrative purposes."
        else:
            system_prompt = _STUDENT_SYSTEM_PROMPT
            role_framing  = "The student has asked about the following course material."

        # Personalization using academic_context
        student_name = academic_ctx.get("studentName") or ""
        subject_name = (
            academic_ctx.get("subjectName")
            or academic_ctx.get("courseName")
            or "this subject"
        )
        department   = academic_ctx.get("departmentName") or ""

        greeting = f"For {student_name}, " if student_name and role == "student" else ""
        dept_info = f" ({department})" if department else ""

        user_prompt = (
            f"{greeting}{role_framing}\n\n"
            f"Subject: {subject_name}{dept_info}\n"
            f"Subject Offering ID: {offering_id}\n"
            f"User question: \"{agent_input.message}\"\n\n"
            f"=== COURSE MATERIAL (USE ONLY THIS — DO NOT USE GENERAL KNOWLEDGE) ===\n"
            f"{material_text}\n"
            f"=== END OF MATERIAL ===\n\n"
            "Using ONLY the course material above, provide a clear, structured "
            "explanation or summary that directly answers the user's question."
        )

        explanation = await self.model_router.generate(
            prompt=user_prompt,
            system_instruction=system_prompt,
            model_id=model_id,
        )

        if not explanation:
            return AgentOutput(
                status="failed",
                response=(
                    "The course material was retrieved but the explanation could "
                    "not be generated. Please try again.\n\n"
                    "تم جلب المواد الدراسية بنجاح لكن تعذّر إنشاء الشرح. يرجى المحاولة مرة أخرى."
                ),
            )

        return AgentOutput(
            status="success",
            response=explanation,
            data={
                "module":           "MaterialExplanationModule",
                "offering_id":      offering_id,
                "subject_name":     subject_name,
                "role":             role,
                "has_material":     True,
                "material_chars":   len(material_text),
                "model_used":       model_id,
            },
        )
