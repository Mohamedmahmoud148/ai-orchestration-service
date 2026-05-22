"""
app/modules/file_extraction.py

File Extraction Module.

Extracts text from uploaded files (PDF, DOCX, TXT).
The file is either:
  a) Provided as raw bytes in context["file_bytes"] + context["file_name"], or
  b) Fetched from the .NET backend via a stored reference.

Libraries used:
  - pdfminer.six  — PDF text extraction
  - python-docx   — DOCX extraction
  - built-in      — TXT decode
"""
from __future__ import annotations

import io
from typing import Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


def _extract_pdf(data: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(io.BytesIO(data)) or ""
    except Exception as exc:
        logger.warning("FileExtractionModule: pdfminer error: %s", exc)
        # Fallback: pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc2:
            logger.warning("FileExtractionModule: pypdf fallback failed: %s", exc2)
            return ""


def _extract_docx(data: bytes) -> str:
    try:
        import docx
        doc   = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception as exc:
        logger.warning("FileExtractionModule: docx error: %s", exc)
        return ""


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def extract_text(data: bytes, filename: str) -> str:
    """Dispatch extraction based on file extension."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf(data)
    if name.endswith(".docx"):
        return _extract_docx(data)
    # Default: treat as plain text
    return _extract_txt(data)


class FileExtractionModule:
    """
    Extracts structured text from PDF, DOCX, or TXT files and optionally
    passes the result through the LLM for cleanup.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("FileExtractionModule: starting.")

        context        = agent_input.context or {}
        file_bytes: Optional[bytes] = context.get("file_bytes")
        file_name: str              = context.get("file_name", "file.pdf")
        file_reference: Optional[str] = context.get("file_reference")
        file_url: Optional[str]     = (
            context.get("file_url")
            or context.get("fileUrl")
            or context.get("signedUrl")
            or context.get("url")
        )

        # -- 1. Get raw bytes — try all sources --------------------------------

        # Source A: direct file_reference → backend API
        if not file_bytes and file_reference:
            result = await self.backend_client.fetch(
                route=f"/api/Files/{file_reference}",
                auth_header=agent_input.auth_header,
            )
            if "error" not in result:
                file_bytes   = result.get("_raw_bytes")
                file_name    = result.get("fileName", file_name)
                content_type = result.get("content_type", "")
                if not file_name.endswith((".pdf", ".docx", ".txt")):
                    if "pdf" in content_type:
                        file_name += ".pdf"
                    elif "word" in content_type or "docx" in content_type:
                        file_name += ".docx"
                    else:
                        file_name += ".txt"

        # Source B: direct URL (from material response or previous AI turn)
        if not file_bytes and file_url and file_url.startswith("http"):
            try:
                import httpx
                headers = {}
                if agent_input.auth_header:
                    headers["Authorization"] = agent_input.auth_header
                async with httpx.AsyncClient() as client:
                    resp = await client.get(file_url, headers=headers, timeout=30.0)
                    resp.raise_for_status()
                file_bytes = resp.content
                # derive filename from URL if not set
                url_path = file_url.split("?")[0]
                if "/" in url_path:
                    file_name = url_path.split("/")[-1] or file_name
                logger.info("FileExtractionModule: fetched %d bytes from URL", len(file_bytes))
            except Exception as exc:
                logger.warning("FileExtractionModule: URL fetch failed — %s", exc)

        # Source C: try fetching materials from the backend using subjectOfferingId
        if not file_bytes:
            offering_id = (
                context.get("subjectOfferingId")
                or context.get("academic_context", {}).get("subjectOfferingId")
            )
            if offering_id:
                result = await self.backend_client.fetch(
                    route=f"/api/Materials/by-offering/{offering_id}",
                    auth_header=agent_input.auth_header,
                )
                items = result.get("items") or (result if isinstance(result, list) else [])
                if items:
                    first = items[0] if isinstance(items, list) else {}
                    mat_url = (
                        first.get("fileUrl") or first.get("signedUrl")
                        or first.get("url") or first.get("filePath") or ""
                    )
                    file_name = first.get("fileName") or first.get("title") or file_name
                    if mat_url and mat_url.startswith("http"):
                        try:
                            import httpx
                            headers = {}
                            if agent_input.auth_header:
                                headers["Authorization"] = agent_input.auth_header
                            async with httpx.AsyncClient() as client:
                                resp = await client.get(mat_url, headers=headers, timeout=30.0)
                                resp.raise_for_status()
                            file_bytes = resp.content
                            logger.info("FileExtractionModule: fetched material from offering — %d bytes", len(file_bytes))
                        except Exception as exc:
                            logger.warning("FileExtractionModule: material URL fetch failed — %s", exc)

        if not file_bytes:
            return AgentOutput(
                status="failed",
                response=(
                    "لم أتمكن من الوصول إلى محتوى الملف. "
                    "تأكد من رفع المواد الدراسية أولاً عبر الدكتور، ثم حاول مجدداً.\n\n"
                    "FileExtractionModule: no file provided. "
                    "Supply file_bytes+file_name or file_reference or file_url in context."
                ),
            )

        # -- 2. Extract text ---------------------------------------------------
        raw_text = extract_text(file_bytes, file_name)
        if not raw_text.strip():
            return AgentOutput(
                status="failed",
                response="FileExtractionModule: no text could be extracted from the file.",
            )

        logger.info("FileExtractionModule: extracted %d chars from '%s'.", len(raw_text), file_name)

        # -- 3. Optional LLM cleanup ------------------------------------------
        selected_model = context.get("selected_model") or "gpt-4o-mini"

        cleaned = await self.model_router.generate(
            prompt=(
                "Clean and structure the following extracted document text. "
                "Remove repeated headers, page numbers, and artefacts. "
                "Return well-formed paragraphs:\n\n"
                + raw_text[:4000]
            ),
            system_instruction="You are a document processing assistant.",
            model_id=selected_model,
        )

        return AgentOutput(
            status="success",
            response=cleaned or raw_text[:2000],
            data={
                "module": "FileExtractionModule",
                "file_name": file_name,
                "char_count": len(raw_text),
                "raw_preview": raw_text[:500],
                "model_used": selected_model
            },
        )

