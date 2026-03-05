"""
app/modules/result_query.py

Result Query Module.

Flow:
  1. Extract exam/student identifiers from plan context.
  2. Call backend  GET /api/Exams/{exam_id}/results
  3. Use Gemini (or local model) to explain the results in plain language.

Example output:
  "You scored 78/100, which corresponds to a B+. You performed well on
   the theory sections but lost marks on the practical questions."
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


class ResultQueryModule:
    """
    Fetches exam results from the .NET backend and explains them
    to the student (or faculty) in plain, friendly language.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("ResultQueryModule: starting.")

        context = agent_input.context or {}

        # -- 1. Resolve identifiers -------------------------------------------
        exam_id    = context.get("examId") or context.get("exam_id")
        student_id = agent_input.user_id

        # -- 2. Fetch results from backend ------------------------------------
        result_data: Dict[str, Any] = {}

        if exam_id:
            result = await self.backend_client.fetch(
                route=f"/api/Exams/{exam_id}/results",
                auth_header=agent_input.auth_header,
                params={"studentId": student_id} if student_id else None,
            )
            if "error" in result:
                logger.warning("ResultQueryModule: backend error — %s", result["error"])
                result_data = {}
            else:
                result_data = result
        else:
            # No exam ID: fall back to a general results query
            result = await self.backend_client.execute_tool(
                route="/api/ai/execute/query-results",
                parameters={
                    "userId":  student_id,
                    "query":   agent_input.message,
                    **{
                        k: context[k]
                        for k in ("subjectId", "semesterId", "batchId")
                        if k in context
                    },
                },
                auth_header=agent_input.auth_header,
                user_id=student_id,
            )
            if result and "error" not in result:
                result_data = result

        # -- 3. Build explanation with LLM ------------------------------------
        selected_model = context.get("selected_model") or "gemini-2.5-flash"
        
        if result_data:
            data_str = str(result_data)
            prompt   = (
                f"Student asked: {agent_input.message}\n\n"
                f"Their result data from the system:\n{data_str}\n\n"
                "Explain these results clearly and helpfully."
            )
        else:
            prompt = (
                f"A student asked: {agent_input.message}\n\n"
                "There are no specific results available in the system at this time. "
                "Provide a helpful, friendly response."
            )

        explanation = await self.model_router.generate(
            prompt=prompt,
            system_instruction=(
                "You are a friendly academic advisor. "
                "Explain exam results clearly, highlight strengths and areas for improvement, "
                "and give encouragement. Be concise (2-4 sentences)."
            ),
            model_id=selected_model,
        )

        # Automatic fallback to local TinyLlama if generation failed or returned empty
        if not explanation and not selected_model.startswith("hf/"):
            logger.info("ResultQueryModule: cloud generation failed, falling back to hf/TinyLlama")
            explanation = await self.model_router.generate(
                prompt=prompt,
                system_instruction="You are a friendly academic advisor explaining exam results.",
                model_id="hf/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )

        return AgentOutput(
            status="success",
            response=explanation or "Your results have been retrieved. Please contact your instructor for details.",
            data={
                "module": "ResultQueryModule",
                "exam_id": exam_id,
                "raw_result": result_data,
                "model_used": selected_model
            },
        )

