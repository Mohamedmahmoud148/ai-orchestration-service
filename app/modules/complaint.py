"""
app/modules/complaint.py

Complaint Module — handles two intents:

  complaint_submit  (student only)
      Constructs the full complaint DTO and POSTs it to:
          POST /api/ai-tools/create-complaint

      Required DTO fields:
          userId           — from academic_context or agent_input.user_id
          targetType       — inferred from message ("Doctor"|"Exam"|"Grade"|"Other")
          subjectOfferingId— from academic_context
          message          — from user message

  complaint_summary (admin / doctor only)
      GETs all complaints then uses the LLM to produce a human-readable
      summary: most-common issues, counts, trends.
          GET /api/ai-tools/get-complaints

Architecture rules
------------------
- AI decides intent, constructs payload, calls backend.
- Backend validates DTO, enforces permissions, resolves DoctorId.
- No direct DB access from here.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger

# ── targetType inference ───────────────────────────────────────────────────────
_TARGET_PATTERNS = [
    (re.compile(r"\bdoctor\b|\bdr\.?\b|\bprofessor\b|\bteacher\b|\binstructor\b", re.I), "Doctor"),
    (re.compile(r"\bexam\b|\btest\b|\bquiz\b", re.I), "Exam"),
    (re.compile(r"\bgrade\b|\bmark\b|\bscore\b|\bresult\b", re.I), "Grade"),
]

_DEFAULT_MODEL = "openai/gpt-4o-mini"


def _infer_target_type(message: str) -> str:
    """Infer complaint targetType from free-form text."""
    for pattern, target in _TARGET_PATTERNS:
        if pattern.search(message):
            return target
    return "Other"


class ComplaintModule:
    """
    Handles complaint submission (student) and complaint summarization
    (admin / doctor) via the .NET backend AI-tools API.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        ctx    = agent_input.context or {}
        role   = ctx.get("role", "student")
        intent = getattr(plan, "intent", None) or ctx.get("intent", "complaint_submit")
        model_id = ctx.get("selected_model") or _DEFAULT_MODEL

        if intent == "complaint_submit":
            return await self._submit(agent_input, ctx, model_id)
        elif intent == "complaint_summary":
            return await self._summarize(agent_input, ctx, role, model_id)
        else:
            return AgentOutput(
                status="failed",
                response="Unknown complaint intent. Expected complaint_submit or complaint_summary.",
            )

    # ──────────────────────────────────────────────────────────────────────────
    #  Submit
    # ──────────────────────────────────────────────────────────────────────────

    async def _submit(
        self, agent_input: AgentInput, ctx: Dict[str, Any], model_id: str
    ) -> AgentOutput:
        """
        Build the full complaint DTO and POST to /api/ai-tools/create-complaint.

        DTO:  { userId, targetType, subjectOfferingId, message }
        DoctorId is resolved by the backend — do NOT send it.
        """
        academic_ctx: Dict[str, Any] = ctx.get("academic_context", {}) or {}

        # ── Resolve required fields ───────────────────────────────────────────
        user_id            = agent_input.user_id or academic_ctx.get("userId") or academic_ctx.get("studentId")
        subject_offering_id = academic_ctx.get("subjectOfferingId") or academic_ctx.get("courseId")
        complaint_message  = agent_input.message.strip()
        target_type        = _infer_target_type(complaint_message)

        logger.info(
            "ComplaintModule [submit]: userId=%s targetType=%s subjectOfferingId=%s",
            user_id, target_type, subject_offering_id,
        )

        # ── Validation — fail fast with helpful message ───────────────────────
        missing: list[str] = []
        if not user_id:
            missing.append("userId")
        if not subject_offering_id:
            missing.append("subjectOfferingId")
        if not complaint_message:
            missing.append("message")

        if missing:
            logger.warning("ComplaintModule [submit]: missing fields — %s", missing)
            return AgentOutput(
                status="failed",
                response=(
                    f"I couldn't submit your complaint because the following "
                    f"information is missing: {', '.join(missing)}. "
                    "Please make sure your academic context is loaded and try again."
                ),
            )

        payload: Dict[str, Any] = {
            "userId":            user_id,
            "targetType":        target_type,
            "subjectOfferingId": subject_offering_id,
            "message":           complaint_message,
        }

        # ── Call backend ──────────────────────────────────────────────────────
        try:
            result = await self.backend_client.post(
                route="/api/ai-tools/create-complaint",
                payload=payload,
                auth_header=agent_input.auth_header,
            )
            logger.info("ComplaintModule [submit]: backend response=%s", result)
        except Exception as exc:
            logger.error("ComplaintModule [submit]: backend call failed — %s", exc)
            return AgentOutput(
                status="failed",
                response=(
                    "Your complaint could not be submitted at this time due to a "
                    "server issue. Please try again in a moment."
                ),
            )

        # ── Build human-readable confirmation ────────────────────────────────
        confirmation = await self.model_router.generate(
            prompt=(
                f"A student just submitted a complaint about a '{target_type}' "
                f"with the following message:\n\n\"{complaint_message}\"\n\n"
                "Write a brief, warm confirmation (2-3 sentences) that:\n"
                "1. Acknowledges the complaint was received.\n"
                "2. Mentions it will be reviewed.\n"
                "3. Encourages them.\n"
                "Do NOT repeat the full complaint text."
            ),
            system_instruction=(
                "You are a friendly university assistant. "
                "Respond in the same language as the student's message."
            ),
            model_id=model_id,
        )

        return AgentOutput(
            status="success",
            response=confirmation or "Your complaint has been submitted successfully. Our team will review it shortly.",
            data={
                "module":   "ComplaintModule",
                "action":   "submit",
                "targetType": target_type,
                "payload":  payload,
                "backend_response": result,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Summary (admin / doctor)
    # ──────────────────────────────────────────────────────────────────────────

    async def _summarize(
        self,
        agent_input: AgentInput,
        ctx: Dict[str, Any],
        role: str,
        model_id: str,
    ) -> AgentOutput:
        """
        Fetch all complaints and produce an LLM summary:
        most-common issues, counts, trends.
        """
        logger.info("ComplaintModule [summary]: role=%s", role)

        # ── Fetch complaints from backend ─────────────────────────────────────
        try:
            complaints_data = await self.backend_client.fetch(
                route="/api/ai-tools/get-complaints",
                auth_header=agent_input.auth_header,
            )
            logger.info(
                "ComplaintModule [summary]: fetched %s complaints",
                len(complaints_data) if isinstance(complaints_data, list) else "data",
            )
        except Exception as exc:
            logger.error("ComplaintModule [summary]: fetch failed — %s", exc)
            return AgentOutput(
                status="failed",
                response="Could not retrieve complaints at this time. Please try again.",
            )

        if not complaints_data or (isinstance(complaints_data, dict) and "error" in complaints_data):
            return AgentOutput(
                status="success",
                response="No complaints have been submitted yet.",
                data={"module": "ComplaintModule", "action": "summary", "count": 0},
            )

        # Compact representation for LLM (cap at 3 000 chars to stay in token budget)
        import json as _json
        complaints_str = _json.dumps(complaints_data, ensure_ascii=False)[:3_000]
        count = len(complaints_data) if isinstance(complaints_data, list) else "unknown"

        # ── LLM summarization ─────────────────────────────────────────────────
        summary = await self.model_router.generate(
            prompt=(
                f"Below are {count} university complaints submitted by students.\n\n"
                f"DATA:\n{complaints_str}\n\n"
                "Produce a structured summary that includes:\n"
                "1. **Total complaints**: exact count.\n"
                "2. **Most common issues**: group by targetType (Doctor, Exam, Grade, Other) with counts.\n"
                "3. **Key themes**: 3-5 bullet points describing recurring problems.\n"
                "4. **Trend**: any noticeable pattern (e.g. spike in a specific area).\n"
                "5. **Recommendation**: one actionable suggestion for administration.\n"
                "Format as a clear, professional report. Do NOT list individual complaints."
            ),
            system_instruction=(
                "You are an academic data analyst preparing a complaint report "
                "for university administration. Be concise, factual, and professional."
            ),
            model_id=model_id,
        )

        return AgentOutput(
            status="success",
            response=summary or "Complaints have been retrieved. Please review them in the admin panel.",
            data={
                "module":  "ComplaintModule",
                "action":  "summary",
                "count":   count,
                "raw":     complaints_data,
            },
        )
