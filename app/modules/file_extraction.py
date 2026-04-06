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

        # -- 1. Get raw bytes --------------------------------------------------
        if not file_bytes and file_reference:
            result = await self.backend_client.fetch(
                route=f"/api/Files/{file_reference}",
                auth_header=agent_input.auth_header,
            )
            if "error" in result:
                return AgentOutput(
                    status="failed",
                    response=f"FileExtractionModule: could not fetch file — {result['error']}",
                )
            file_bytes   = result.get("_raw_bytes")
            file_name    = result.get("fileName", file_name)
            content_type = result.get("content_type", "")
            # Detect type from content-type header if no extension
            if not file_name.endswith((".pdf", ".docx", ".txt")):
                if "pdf" in content_type:
                    file_name += ".pdf"
                elif "word" in content_type or "docx" in content_type:
                    file_name += ".docx"
                else:
                    file_name += ".txt"

        if not file_bytes:
            return AgentOutput(
                status="failed",
                response=(
                    "FileExtractionModule: no file provided. "
                    "Supply file_bytes+file_name or file_reference in context."
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

