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
أنت أستاذ جامعي متخصص في شرح المواد الأكاديمية للطلاب بعمق وتفصيل.

القواعد الأساسية:
1. استخدم المحتوى المقدم فقط — لا تضف معلومات من خارج الملف.
2. قدّم شرحاً مفصلاً وشاملاً — لا تختصر أبداً.
3. نظّم الشرح بعناوين واضحة وفقرات منظمة.
4. اشرح كل مفهوم بمثال عملي إن أمكن.
5. اذكر النقاط المهمة للامتحانات.
6. أجب بنفس لغة سؤال الطالب (عربي أو إنجليزي).
7. الطول المطلوب: شرح كامل ومفصل بدون تقليص.\
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


async def _fetch_file_url_bytes(file_url: str, auth_header: Optional[str]) -> Optional[bytes]:
    """Download raw bytes from a URL."""
    if not file_url or not file_url.startswith("http"):
        return None
    try:
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header
        async with httpx.AsyncClient() as client:
            resp = await client.get(file_url, headers=headers, timeout=60.0, follow_redirects=True)
            resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("MaterialExplanationModule: byte fetch failed (%s) — %s", file_url[:80], exc)
        return None


async def _fetch_file_url_text(file_url: str, auth_header: Optional[str], filename: str = "") -> str:
    """
    Fetch file from URL and extract text.
    Supports PDF, Excel (xlsx/xls), DOCX, CSV, and plain text.
    """
    if not file_url or not file_url.startswith("http"):
        return ""
    try:
        headers: Dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header
        async with httpx.AsyncClient() as client:
            resp = await client.get(file_url, headers=headers, timeout=60.0, follow_redirects=True)
            resp.raise_for_status()

        # Determine file type from URL or content-type
        url_path = file_url.split("?")[0].lower()
        fname = filename.lower() or url_path.split("/")[-1]
        content_type = resp.headers.get("content-type", "")

        # Import extraction functions from file_extraction module
        from app.modules.file_extraction import extract_text_from_bytes

        if any(fname.endswith(ext) for ext in (".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt")):
            # No truncation here — caller handles large text
            return extract_text_from_bytes(resp.content, fname)

        if "pdf" in content_type:
            return _extract_pdf_text(resp.content)
        if "spreadsheet" in content_type or "excel" in content_type:
            from app.modules.file_extraction import _extract_excel
            return _extract_excel(resp.content, fname)[:6_000]
        if "text" in content_type:
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

    Backend returns a paginated envelope:
      { "items": [{id, fileName, contentType, fileSize, uploadedAt, fileUrl}, ...],
        "totalCount": N, "pageNumber": 1, "pageSize": 10 }

    For each item:
      1. Try content / text / description field (fastest — usually empty for file-only uploads)
      2. Try fileUrl → download → extract text from PDF / read plain text

    Raw bytes and bare-list shapes are kept for backward compatibility.
    """
    if not materials_data:
        return ""

    texts: List[str] = []

    # ── Unwrap paginated envelope ─────────────────────────────────────────
    # The .NET backend returns { "items": [...], "totalCount": N }.
    # Unwrap so the rest of the function always operates on a plain list.
    if isinstance(materials_data, dict) and "items" in materials_data:
        materials_data = materials_data["items"] or []

    # ── List of material objects ──────────────────────────────────────────
    if isinstance(materials_data, list):
        for item in materials_data[:5]:  # cap to avoid token overflow
            if not isinstance(item, dict):
                continue
            title = item.get("fileName") or item.get("title") or item.get("name") or ""

            # 1. Inline text field (usually absent for pure file uploads)
            text = (
                item.get("content")
                or item.get("text")
                or item.get("description")
                or ""
            )
            if text and isinstance(text, str):
                texts.append(f"[{title}]\n{text}" if title else text)
                continue

            # 2. Download via fileUrl (public CDN URL built by backend)
            file_url = (
                item.get("fileUrl")
                or item.get("signedUrl")
                or item.get("url")
                or item.get("filePath")
                or ""
            )
            if file_url:
                fetched = await _fetch_file_url_text(file_url, auth_header, title)
                if fetched:
                    label = f"[{title}]\n{fetched}" if title else fetched
                    texts.append(label)
                    logger.info(
                        "MaterialExplanationModule: extracted %d chars from '%s'",
                        len(fetched), title,
                    )

    # ── Single dict (non-paginated legacy shape) ──────────────────────────
    elif isinstance(materials_data, dict):
        if "_raw_bytes" in materials_data:
            pdf_text = _extract_pdf_text(materials_data["_raw_bytes"])
            if pdf_text:
                texts.append(pdf_text)
        else:
            text = materials_data.get("content") or materials_data.get("text") or ""
            if text:
                texts.append(str(text))
            else:
                file_url = (
                    materials_data.get("fileUrl")
                    or materials_data.get("signedUrl")
                    or ""
                )
                if file_url:
                    fetched = await _fetch_file_url_text(file_url, auth_header)
                    if fetched:
                        texts.append(fetched)

    # ── Raw bytes (PDF returned directly) ─────────────────────────────────
    elif isinstance(materials_data, bytes):
        pdf_text = _extract_pdf_text(materials_data)
        if pdf_text:
            texts.append(pdf_text)

    combined = "\n\n".join(texts).strip()
    # No hard truncation — MaterialExplanationModule handles large text via deep_pdf_study
    return combined


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

        # ── 1b. No subjectOfferingId — try smarter fallbacks ─────────────────
        if not offering_id:
            logger.warning("MaterialExplanationModule: subjectOfferingId not found — trying fallbacks")

            # Fallback A: file_url in academic_context (restored from memory or passed directly)
            direct_url = (
                academic_ctx.get("file_url") or academic_ctx.get("fileUrl")
                or ctx.get("file_url") or ctx.get("fileUrl")
            )
            if direct_url:
                logger.info("MaterialExplanationModule: using direct file_url from context")
                material_text = await _fetch_file_url_text(direct_url, agent_input.auth_header)
                if material_text:
                    return await self._explain(material_text, agent_input, model_id, role, ctx)

            # Fallback B: extract file URL from recent conversation history
            history = ctx.get("history", []) or []
            import re
            for turn in reversed(history[-6:]):
                content = turn.get("content", "")
                urls = re.findall(
                    r'https?://[^\s\)\]"\']+\.(?:pdf|xlsx|xls|docx|csv)',
                    content, re.IGNORECASE
                )
                if urls:
                    logger.info("MaterialExplanationModule: found file URL in history: %s", urls[0][:60])
                    material_text = await _fetch_file_url_text(urls[0], agent_input.auth_header)
                    if material_text:
                        return await self._explain(material_text, agent_input, model_id, role, ctx)

            # Fallback C: search for the file by name across all student's enrolled offerings
            student_id = academic_ctx.get("studentId") or academic_ctx.get("student_id")
            if student_id and agent_input.auth_header:
                try:
                    enrollments = await self.backend_client.fetch(
                        route="/api/SubjectOfferings/my-enrollments",
                        auth_header=agent_input.auth_header,
                    )
                    items = enrollments if isinstance(enrollments, list) else \
                            (enrollments.get("items") or enrollments.get("data") or [])
                    # Extract file name from user message to search for
                    import re
                    fname_match = re.search(
                        r'([^\s/\\]+\.(?:pdf|docx|xlsx|xls|csv|txt|png|jpg))',
                        agent_input.message or "", re.IGNORECASE
                    )
                    fname_query = fname_match.group(1).lower() if fname_match else ""

                    for offering in (items[:8] if isinstance(items, list) else []):
                        oid = (offering.get("id") or offering.get("offeringId") or
                               offering.get("subjectOfferingId") or "")
                        if not oid:
                            continue
                        try:
                            mats = await self.backend_client.fetch(
                                route=f"/api/Materials/by-offering/{oid}",
                                auth_header=agent_input.auth_header,
                            )
                            mat_items = mats if isinstance(mats, list) else \
                                        (mats.get("items") or [])
                            for mat in (mat_items if isinstance(mat_items, list) else []):
                                mat_fname = (mat.get("fileName") or mat.get("title") or "").lower()
                                if fname_query and fname_query not in mat_fname:
                                    continue  # skip if looking for specific file
                                mat_url = (mat.get("fileUrl") or mat.get("signedUrl")
                                           or mat.get("url") or "")
                                if mat_url:
                                    material_text = await _fetch_file_url_text(
                                        mat_url, agent_input.auth_header,
                                        mat.get("fileName") or ""
                                    )
                                    if material_text:
                                        logger.info(
                                            "MaterialExplanationModule: found file '%s' in offering %s",
                                            mat.get("fileName"), oid
                                        )
                                        # Store offering_id so deep PDF mode works
                                        offering_id = oid
                                        # Use deep PDF if file is large
                                        if mat_fname.endswith(".pdf") and len(material_text) > 8_000:
                                            from app.modules.deep_pdf_study import extract_pdf_pages, summarize_large_pdf
                                            url_bytes = await _fetch_file_url_bytes(mat_url, agent_input.auth_header)
                                            if url_bytes:
                                                pages = extract_pdf_pages(url_bytes)
                                                if pages:
                                                    explanation = await summarize_large_pdf(
                                                        pages=pages,
                                                        user_question=agent_input.message or "اشرح الملف",
                                                        model_router=self.model_router,
                                                        model_id=model_id,
                                                        subject_name=mat.get("title") or mat.get("fileName") or "",
                                                    )
                                                    return AgentOutput(
                                                        status="success",
                                                        response=explanation,
                                                        data={"module": "MaterialExplanationModule",
                                                              "mode": "cross_offering_deep_pdf",
                                                              "page_count": len(pages)},
                                                    )
                                        return await self._explain(material_text, agent_input, model_id, role, ctx)
                        except Exception:
                            continue
                except Exception as fe:
                    logger.warning("MaterialExplanationModule: enrollment search failed — %s", fe)

            # Fallback D: give a helpful response
            enrolled = academic_ctx.get("enrolledCourses") or academic_ctx.get("courses") or []
            if enrolled:
                course_list = ", ".join(
                    (c.get("name") or c.get("subjectName") or str(c))
                    for c in (enrolled[:5] if isinstance(enrolled, list) else [])
                )
                return AgentOutput(
                    status="failed",
                    response=f"عندك المواد دي: {course_list}\nقولي أي مادة عايز تعرف تفاصيلها؟",
                )
            return AgentOutput(
                status="failed",
                response="قولي اسم المادة أو اللائحة اللي عايز أقرأها.",
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

        # ── 3. Handle soft auth error ─────────────────────────────────────────
        if isinstance(materials_data, dict) and materials_data.get("_error") == "unauthorized":
            logger.warning(
                "MaterialExplanationModule: 401/403 from backend for offering=%s", offering_id
            )
            return AgentOutput(
                status="failed",
                response=(
                    "⚠️ مفيش صلاحية للوصول لمواد المادة دي. "
                    "تأكد إنك مسجل في المادة دي.\n\n"
                    "Unable to access materials for this subject. "
                    "Ensure you are enrolled in this offering."
                ),
            )

        # ── 4. Extract usable text (including fileUrl fallback) ───────────────
        material_text = await _collect_material_text(
            materials_data, agent_input.auth_header
        )

        # ── Vision retry for scanned/image PDFs ─────────────────────────────────
        if not material_text or len(material_text.strip()) < 30:
            # Try vision model before giving up
            try:
                items_v = materials_data if isinstance(materials_data, list) \
                    else (materials_data.get("items") or [])
                for item_v in (items_v[:2] if isinstance(items_v, list) else []):
                    if not isinstance(item_v, dict): continue
                    vurl = (item_v.get("fileUrl") or item_v.get("signedUrl") or "")
                    vname = (item_v.get("fileName") or "").lower()
                    if not vurl: continue
                    vbytes = await _fetch_file_url_bytes(vurl, agent_input.auth_header)
                    if not vbytes: continue
                    from app.modules.file_extraction import _vision_extract
                    vision_text = await _vision_extract(
                        vbytes, vname,
                        agent_input.message or "اشرح محتوى الملف",
                        self.model_router, agent_input.auth_header
                    )
                    if vision_text and len(vision_text.strip()) > 50:
                        material_text = vision_text
                        logger.info("MaterialExplanationModule: vision rescued %d chars", len(vision_text))
                        break
            except Exception as ve:
                logger.warning("MaterialExplanationModule: vision rescue failed — %s", ve)

        if not material_text or len(material_text.strip()) < 30:
            logger.info("MaterialExplanationModule: no readable text for offering=%s", offering_id)
            # Don't say "problem" — ask what they want instead
            subject = academic_ctx.get("subjectName") or academic_ctx.get("courseName") or "المادة"
            return AgentOutput(
                status="success",
                response=(
                    f"الملف في مادة **{subject}** مش قابل للقراءة كـ نص عادي "
                    f"(ممكن يكون PDF مسحوب ضوئياً أو slides بالصور).\n\n"
                    f"ممكن تساعدني وتقولي:\n"
                    f"- ايه الموضوع اللي عايز تفهمه؟\n"
                    f"- ولا عايز أشرحلك أي مفهوم معين في {subject}؟"
                ),
                data={"module": "MaterialExplanationModule", "offering_id": offering_id, "has_material": False},
            )

        logger.info(
            "MaterialExplanationModule: %d chars of material extracted",
            len(material_text),
        )

        # ── 3b. Deep PDF mode for large documents ─────────────────────────────
        # If material is a PDF and text is large, use hierarchical summarization.
        # This gives accurate page count and full document coverage.
        from app.modules.deep_pdf_study import (
            extract_pdf_pages, get_pdf_page_count, summarize_large_pdf,
            is_full_doc_request, _SINGLE_CALL_CHAR_LIMIT
        )

        student_name = academic_ctx.get("studentName") or ""
        subject_name = (
            academic_ctx.get("subjectName")
            or academic_ctx.get("courseName")
            or "this subject"
        )

        # Check if we have a PDF URL to download for deep mode
        pdf_bytes: Optional[bytes] = None
        if len(material_text) > _SINGLE_CALL_CHAR_LIMIT or \
           any(kw in (agent_input.message or "").lower() for kw in
               ["كم صفحة", "كم عدد", "how many page", "اشرح كل", "اشرح الكامل",
                "summarize", "لخص", "اقرأ", "شرح تفصيلي", "explain everything",
                "explain all", "page", "صفحة"]):
            # Try to get PDF bytes for page-level analysis
            try:
                materials_data_raw = await self.backend_client.fetch(
                    route=f"/api/Materials/by-offering/{offering_id}",
                    auth_header=agent_input.auth_header,
                )
                items = materials_data_raw if isinstance(materials_data_raw, list) \
                    else materials_data_raw.get("items", [])
                for item in (items[:3] if isinstance(items, list) else []):
                    if not isinstance(item, dict): continue
                    furl = (item.get("fileUrl") or item.get("signedUrl")
                            or item.get("url") or "")
                    fname_item = (item.get("fileName") or "").lower()
                    if furl and fname_item.endswith(".pdf"):
                        pdf_bytes = await _fetch_file_url_bytes(furl, agent_input.auth_header)
                        if pdf_bytes:
                            logger.info(
                                "MaterialExplanationModule: downloaded PDF bytes (%d) for deep mode",
                                len(pdf_bytes)
                            )
                            break
            except Exception as e:
                logger.warning("MaterialExplanationModule: deep mode PDF fetch failed — %s", e)

        if pdf_bytes:
            pages = extract_pdf_pages(pdf_bytes)
            real_page_count = len(pages)
            total_chars     = sum(len(p) for p in pages)

            logger.info(
                "MaterialExplanationModule: deep PDF mode — %d pages, %d chars",
                real_page_count, total_chars
            )

            explanation = await summarize_large_pdf(
                pages         = pages,
                user_question = agent_input.message or "اشرح محتوى الملف",
                model_router  = self.model_router,
                model_id      = model_id,
                subject_name  = subject_name,
                student_name  = student_name,
            )

            return AgentOutput(
                status="success",
                response=explanation,
                data={
                    "module":         "MaterialExplanationModule",
                    "offering_id":    offering_id,
                    "subject_name":   subject_name,
                    "role":           role,
                    "has_material":   True,
                    "page_count":     real_page_count,
                    "total_chars":    total_chars,
                    "model_used":     model_id,
                    "mode":           "deep_pdf",
                },
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

        # Use full text (no truncation) — but cap at 30K chars for single-call mode
        # Deep PDF mode already handled above for large PDFs
        material_for_prompt = material_text[:30_000] if len(material_text) > 30_000 else material_text

        user_prompt = (
            f"{greeting}{role_framing}\n\n"
            f"Subject: {subject_name}{dept_info}\n"
            f"Subject Offering ID: {offering_id}\n"
            f"User question: \"{agent_input.message}\"\n\n"
            f"=== COURSE MATERIAL (USE ONLY THIS — DO NOT USE GENERAL KNOWLEDGE) ===\n"
            f"{material_for_prompt}\n"
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

    async def _explain(
        self,
        material_text: str,
        agent_input: AgentInput,
        model_id: str,
        role: str,
        ctx: dict,
    ) -> AgentOutput:
        """Direct explanation from already-fetched material text."""
        if role == "doctor":
            system_prompt = _DOCTOR_SYSTEM_PROMPT
        elif role == "admin":
            system_prompt = _ADMIN_SYSTEM_PROMPT
        else:
            system_prompt = _STUDENT_SYSTEM_PROMPT

        user_prompt = (
            f"سؤال/طلب المستخدم: {agent_input.message}\n\n"
            f"=== محتوى الملف/اللائحة ===\n{material_text[:6000]}\n=== نهاية المحتوى ===\n\n"
            "بناءً على المحتوى أعلاه فقط، أجب على سؤال المستخدم بشكل واضح ومنظم."
        )

        explanation = await self.model_router.generate(
            prompt=user_prompt,
            system_instruction=system_prompt,
            model_id=model_id,
        )

        return AgentOutput(
            status="success",
            response=explanation or "تعذّر إنشاء الشرح. حاول مرة أخرى.",
            data={"module": "MaterialExplanationModule", "has_material": True},
        )
