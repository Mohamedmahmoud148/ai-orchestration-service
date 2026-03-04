"""
app/agents/planner.py

PlannerAgent — uses Gemini to classify the user's intent and produce
a validated ExecutionPlan that the Agent pipeline can consume.

Gemini is called via the official google-generativeai SDK and the API key
is read from the GEMINI_API_KEY environment variable at import time,
matching the pattern requested in the project spec.
"""

import json
import os
from typing import Optional, Protocol

import google.generativeai as genai
from pydantic import ValidationError

from app.agents.base_agent import BaseAgent
from app.agents.schemas import (
    AgentInput,
    AgentOutput,
    ExecutionPlan,
    ExamParams,
    PreExecutionStep,
)
from app.core.logging import logger

# ── Gemini SDK setup ──────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ── Valid intent catalogue ────────────────────────────────────────────────────
VALID_INTENTS = {
    "general_chat",
    "summarization",
    "generate_exam",
    "result_query",
    "file_extraction",
}

# ── Fallback plan when Gemini cannot produce valid JSON ───────────────────────
_FALLBACK_INTENT = "general_chat"

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an AI Planning Agent.

Your ONLY job is to read the user's message and return a single JSON object that
classifies the request.

Valid intents:
- general_chat       — casual conversation, questions, or anything else
- summarization      — the user wants a summary of a text or document
- generate_exam      — the user (a doctor/educator) wants an exam or quiz generated
- result_query       — the user wants to query academic results or grades
- file_extraction    — the user wants to extract or parse information from a file

Rules:
1. Reply with ONLY a raw JSON object. No markdown, no code fences, no extra text.
2. Use exactly this schema:

{
  "intent": "<one of the valid intents above>",
  "goal_summary": "<one short sentence describing what the user wants>",
  "is_executable": true,
  "exam_params": null,
  "pre_execution_steps": [],
  "steps": []
}

3. If the intent is generate_exam AND the user supplied exam details, populate
   exam_params with as many of these fields as you can extract:
     collegeName, departmentName, batchName, subjectName,
     numberOfQuestions (integer), examType ("midterm"|"final"),
     variationMode ("same_for_all"|"different_per_student"),
     subjectOfferingId (string | null)

4. If intent is anything other than generate_exam, keep exam_params as null.
5. Never output anything outside the JSON object.
"""


class MemoryStore(Protocol):
    """Protocol defining how the Planner retrieves historical context."""

    async def get_context(self, user_id: str | None) -> str: ...


class PlannerAgent(BaseAgent):
    """
    Generates an ExecutionPlan by asking Gemini to classify the user's intent.

    The Gemini API key is read from GEMINI_API_KEY at module load.
    The model_router dependency is kept for backward compatibility with
    main.py's DI setup but is NOT used for the planning call — Gemini is
    called directly so that planning is always available regardless of
    which cloud clients the ModelRouter has configured.
    """

    def __init__(self, model_router, ranker=None, memory: Optional[MemoryStore] = None):
        self.model_router = model_router   # kept for DI compatibility
        self.ranker = ranker
        self.memory = memory

    # ─────────────────────────────────────────────────────────────────────
    #  Public interface (required by Agent pipeline)
    # ─────────────────────────────────────────────────────────────────────

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        1. Build a prompt from the user message + optional memory context.
        2. Call Gemini for a JSON plan.
        3. Parse + validate → ExecutionPlan.
        4. Inject ResolveSubjectOffering pre-step when exam_params lacks an ID.
        5. Return AgentOutput(status="success", data={"plan": plan}).
           On any failure / bad JSON → fallback to a general_chat plan.
        """
        logger.info("PlannerAgent: starting for user_id=%s", agent_input.user_id)

        # ── Optional memory context ───────────────────────────────────────
        memory_prefix = ""
        if self.memory:
            try:
                past = await self.memory.get_context(agent_input.user_id)
                if past:
                    memory_prefix = f"[Past context]: {past}\n\n"
            except Exception as mem_exc:
                logger.warning("PlannerAgent: memory lookup failed — %s", mem_exc)

        # ── Build user prompt ─────────────────────────────────────────────
        role = agent_input.context.get("role", "user") if agent_input.context else "user"
        prompt = (
            f"{memory_prefix}"
            f"User role: {role}\n"
            f"User message: {agent_input.message}"
        )

        # ── Call Gemini ───────────────────────────────────────────────────
        raw_json = await self._call_gemini(prompt)

        # ── Parse response → ExecutionPlan ────────────────────────────────
        plan = self._parse_plan(raw_json, agent_input)

        # ── Deterministic guard for generate_exam ─────────────────────────
        plan = self._ensure_resolve_step(plan)

        logger.info(
            "PlannerAgent: intent=%r goal=%r", plan.intent, plan.goal_summary
        )

        return AgentOutput(
            status="success",
            response=plan.goal_summary,
            data={"plan": plan},
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    async def _call_gemini(self, prompt: str) -> dict | None:
        """
        Send the prompt to Gemini and return the parsed JSON dict,
        or None on any error.
        """
        try:
            response = await _gemini_model.generate_content_async(
                contents=[
                    {"role": "user", "parts": [_SYSTEM_PROMPT + "\n\n" + prompt]}
                ],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )

            text = response.text or ""
            # Strip accidental markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            logger.debug("PlannerAgent: Gemini raw plan = %s", parsed)
            return parsed

        except json.JSONDecodeError as exc:
            logger.error("PlannerAgent: Gemini returned invalid JSON — %s", exc)
            return None
        except Exception as exc:
            logger.error("PlannerAgent: Gemini call failed — %s", exc, exc_info=True)
            return None

    def _parse_plan(self, raw: dict | None, agent_input: AgentInput) -> ExecutionPlan:
        """
        Convert raw Gemini output into a validated ExecutionPlan.
        Falls back to a general_chat plan on any parse/validation error.
        """
        if not raw:
            return self._fallback_plan(agent_input.message)

        # Normalise intent
        intent = raw.get("intent", _FALLBACK_INTENT)
        if intent not in VALID_INTENTS:
            logger.warning(
                "PlannerAgent: unknown intent %r — falling back to %s",
                intent,
                _FALLBACK_INTENT,
            )
            intent = _FALLBACK_INTENT
            raw["intent"] = intent

        try:
            plan = ExecutionPlan(**raw)
            return plan
        except (ValidationError, TypeError) as exc:
            logger.error(
                "PlannerAgent: ExecutionPlan validation failed — %s", exc
            )
            return self._fallback_plan(agent_input.message)

    @staticmethod
    def _fallback_plan(message: str) -> ExecutionPlan:
        """Return a minimal, always-valid general_chat plan."""
        return ExecutionPlan(
            intent=_FALLBACK_INTENT,
            goal_summary=f"Handle the user's request: {message[:120]}",
            is_executable=True,
        )

    @staticmethod
    def _ensure_resolve_step(plan: ExecutionPlan) -> ExecutionPlan:
        """
        If the plan targets generate_exam but lacks subjectOfferingId,
        inject the ResolveSubjectOffering pre-execution step so the
        ExamGenerationModule never receives an incomplete plan.
        """
        if (
            plan.intent == "generate_exam"
            and plan.exam_params is not None
            and plan.exam_params.subjectOfferingId is None
        ):
            already_there = any(
                s.tool == "ResolveSubjectOffering"
                for s in plan.pre_execution_steps
            )
            if not already_there:
                logger.info(
                    "PlannerAgent: injecting ResolveSubjectOffering pre-step "
                    "(subjectOfferingId not supplied by user)"
                )
                plan.pre_execution_steps.append(
                    PreExecutionStep(
                        tool="ResolveSubjectOffering",
                        reason=(
                            "subjectOfferingId is required to generate the exam "
                            "but was not provided by the user"
                        ),
                        input_payload={
                            "collegeName": plan.exam_params.collegeName,
                            "departmentName": plan.exam_params.departmentName,
                            "batchName": plan.exam_params.batchName,
                            "subjectName": plan.exam_params.subjectName,
                        },
                    )
                )
        return plan
