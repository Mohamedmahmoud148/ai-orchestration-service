"""
app/modules/file_processor.py

File Processor Module — handles bulk Excel + PDF uploads via fileUrl.

Intent: file_processing

Detection strategy (HYBRID):
  A) Primary — message keywords:
       "students" → BulkCreateStudents → POST /api/ai-tools/bulk-create-students
       "grades"   → BulkUploadGrades  → POST /api/ai-tools/bulk-upload-grades
  B) Fallback — Excel column detection:
       name, email, student_code          → students file
       UniversityStudentId, SubjectOfferingId, FinalScore … → grades file
  C) If still ambiguous → ask user for clarification.

PDF:
  Extract text via pdfminer → LLM summarization.

Returns:
  { success_count, failed_count, summary }

Architecture rules
------------------
- AI downloads file, parses it, decides operation type, calls backend.
- Backend validates records, enforces permissions, persists data.
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional

import httpx

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger

_DEFAULT_MODEL = "openai/gpt-4o-mini"

# ── Column fingerprints ────────────────────────────────────────────────────────
_STUDENT_COLUMNS: frozenset[str] = frozenset({"name", "email", "student_code"})
_GRADE_COLUMNS:   frozenset[str] = frozenset({
    "universityStudentId", "SubjectOfferingId",
    "FinalScore", "GradeLetter", "GradePoints",
})

# ── Keyword patterns ───────────────────────────────────────────────────────────
_STUDENT_KEYWORDS = re.compile(r"\bstudents?\b|\benroll\b|\bregist", re.I)
_GRADE_KEYWORDS   = re.compile(r"\bgrades?\b|\bmarks?\b|\bscores?\b|\bresults?\b", re.I)


def _detect_file_type_from_columns(columns: List[str]) -> Optional[str]:
    """
    Infer upload type from column names.
    Returns 'students', 'grades', or None.
    """
    lower_cols = {c.lower() for c in columns}
    student_matches = sum(1 for c in _STUDENT_COLUMNS if c.lower() in lower_cols)
    grade_matches   = sum(1 for c in {"universityStudentId", "FinalScore", "GradeLetter"} if c.lower() in lower_cols)

    if student_matches >= 2:
        return "students"
    if grade_matches >= 2:
        return "grades"
    return None


async def _download_file(url: str, auth_header: Optional[str]) -> bytes:
    """Download a file from a URL, forwarding auth if provided."""
    headers: Dict[str, str] = {}
    if auth_header:
        headers["Authorization"] = auth_header
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.content


def _extract_file_url(message: str) -> Optional[str]:
    """Pull the first HTTP(S) URL from the user message."""
    match = re.search(r"https?://\S+", message)
    return match.group(0).rstrip(".,)>\"'") if match else None


def _parse_excel(data: bytes) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse Excel bytes → (records list, columns list).
    Returns ([], []) on failure.
    """
    try:
        import pandas as pd
        df = pd.read_excel(io.BytesIO(data))
        df = df.where(df.notna(), None)  # replace NaN with None for JSON
        return df.to_dict(orient="records"), list(df.columns)
    except Exception as exc:
        logger.error("FileProcessorModule: pandas Excel parse failed — %s", exc)
        return [], []


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from PDF bytes using pdfminer, fallback to pypdf."""
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
        logger.warning("FileProcessorModule: PDF extraction failed — %s", exc)
        return ""


class FileProcessorModule:
    """
    Downloads and processes Excel (.xlsx/.xls) or PDF files from a URL.

    Excel: bulk-create students OR bulk-upload grades depending on hybrid detection.
    PDF:   extracts text and summarizes via LLM.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        ctx      = agent_input.context or {}
        model_id = ctx.get("selected_model") or _DEFAULT_MODEL
        message  = agent_input.message or ""

        # ── 1. Locate fileUrl ─────────────────────────────────────────────────
        file_url = (
            ctx.get("fileUrl")
            or ctx.get("file_url")
            or _extract_file_url(message)
        )

        if not file_url:
            logger.warning("FileProcessorModule: no fileUrl found in context or message")
            return AgentOutput(
                status="failed",
                response=(
                    "I couldn't find a file URL in your message. "
                    "Please include the file URL and try again."
                ),
            )

        logger.info("FileProcessorModule: downloading file from %s", file_url)

        # ── 2. Download ───────────────────────────────────────────────────────
        try:
            file_bytes = await _download_file(file_url, agent_input.auth_header)
        except Exception as exc:
            logger.error("FileProcessorModule: download failed — %s", exc)
            return AgentOutput(
                status="failed",
                response="Could not download the file. Please check the URL and your permissions.",
            )

        # ── 3. Route by file type ─────────────────────────────────────────────
        url_lower = file_url.lower()
        if url_lower.endswith((".xlsx", ".xls")):
            return await self._process_excel(
                file_bytes, message, agent_input, model_id
            )
        elif url_lower.endswith(".pdf"):
            return await self._process_pdf(file_bytes, message, model_id)
        else:
            # Attempt Excel parse first, then PDF, then fail
            records, columns = _parse_excel(file_bytes)
            if records:
                return await self._process_excel(
                    file_bytes, message, agent_input, model_id, pre_parsed=(records, columns)
                )
            text = _extract_pdf_text(file_bytes)
            if text.strip():
                return await self._summarize_pdf_text(text, model_id)
            return AgentOutput(
                status="failed",
                response=(
                    "The file format could not be determined. "
                    "Please upload an Excel (.xlsx) or PDF file."
                ),
            )

    # ──────────────────────────────────────────────────────────────────────────
    #  Excel processing
    # ──────────────────────────────────────────────────────────────────────────

    async def _process_excel(
        self,
        file_bytes: bytes,
        message: str,
        agent_input: AgentInput,
        model_id: str,
        pre_parsed: Optional[tuple] = None,
    ) -> AgentOutput:
        records, columns = pre_parsed if pre_parsed else _parse_excel(file_bytes)

        if not records:
            return AgentOutput(
                status="failed",
                response="The Excel file appears to be empty or could not be parsed.",
            )

        logger.info(
            "FileProcessorModule: parsed %d rows, columns=%s", len(records), columns
        )

        # ── Hybrid detection ──────────────────────────────────────────────────
        # A) Primary: message keywords
        if _STUDENT_KEYWORDS.search(message):
            operation = "students"
        elif _GRADE_KEYWORDS.search(message):
            operation = "grades"
        else:
            # B) Fallback: column fingerprint
            operation = _detect_file_type_from_columns(columns)

        # C) Still ambiguous → ask for clarification
        if not operation:
            return AgentOutput(
                status="clarification_needed",
                response=(
                    "I've parsed your Excel file and found the following columns: "
                    f"{', '.join(columns[:10])}.\n\n"
                    "Could you clarify what this file contains?\n"
                    "- Reply **\"students\"** to bulk-create student accounts\n"
                    "- Reply **\"grades\"** to bulk-upload grade records"
                ),
                data={
                    "module":   "FileProcessorModule",
                    "columns":  columns,
                    "row_count": len(records),
                },
            )

        # ── Call backend ──────────────────────────────────────────────────────
        if operation == "students":
            route   = "/api/ai-tools/bulk-create-students"
            payload = {"students": records}
        else:
            route   = "/api/ai-tools/bulk-upload-grades"
            payload = {"grades": records}

        logger.info(
            "FileProcessorModule [POST] route=%s records=%d", route, len(records)
        )

        try:
            result = await self.backend_client.post(
                route=route,
                payload=payload,
                auth_header=agent_input.auth_header,
            )
            logger.info("FileProcessorModule: backend response=%s", str(result)[:300])
        except Exception as exc:
            logger.error("FileProcessorModule: backend call failed — %s", exc)
            return AgentOutput(
                status="failed",
                response=(
                    "The file was parsed successfully but could not be uploaded "
                    "to the backend. Please try again."
                ),
            )

        # ── Extract counts from response ──────────────────────────────────────
        success_count = (
            result.get("successCount")
            or result.get("success_count")
            or result.get("created")
            or len(records)
        )
        failed_count  = (
            result.get("failedCount")
            or result.get("failed_count")
            or result.get("failed")
            or 0
        )

        noun = "students" if operation == "students" else "grade records"
        response_text = (
            f"✅ **Bulk upload complete**\n\n"
            f"- **Total rows processed**: {len(records)}\n"
            f"- **Successfully created**: {success_count} {noun}\n"
            f"- **Failed**: {failed_count}\n"
        )
        if failed_count > 0:
            response_text += (
                "\n⚠️ Some records failed. "
                "Check the backend logs for details on which rows were rejected."
            )

        return AgentOutput(
            status="success",
            response=response_text,
            data={
                "module":        "FileProcessorModule",
                "operation":     operation,
                "row_count":     len(records),
                "success_count": success_count,
                "failed_count":  failed_count,
                "backend_response": result,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  PDF processing
    # ──────────────────────────────────────────────────────────────────────────

    async def _process_pdf(self, file_bytes: bytes, message: str, model_id: str) -> AgentOutput:
        text = _extract_pdf_text(file_bytes)
        if not text.strip():
            return AgentOutput(
                status="failed",
                response="No text could be extracted from the PDF file.",
            )
        return await self._summarize_pdf_text(text, model_id)

    async def _summarize_pdf_text(self, text: str, model_id: str) -> AgentOutput:
        summary = await self.model_router.generate(
            prompt=(
                "Summarize the key information in the following document. "
                "Focus on facts, numbers, and actionable data.\n\n"
                f"DOCUMENT:\n{text[:4_000]}"
            ),
            system_instruction=(
                "You are a professional document analyst. "
                "Produce a clear, concise summary in bullet points."
            ),
            model_id=model_id,
        )
        return AgentOutput(
            status="success",
            response=summary or text[:1_500],
            data={
                "module":     "FileProcessorModule",
                "operation":  "pdf_summary",
                "char_count": len(text),
            },
        )
