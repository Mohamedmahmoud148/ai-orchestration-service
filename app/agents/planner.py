"""
planner.py

PlannerAgent — uses the LLM to classify the user's intent, extract academic
parameters, and optionally produce multi-step ExecutionPlans for tool-bound
requests.

Key upgrades (v4.0):
  - Injects academic_context into the classification prompt so the model can
    auto-fill tool parameters (e.g. user_id, subjectOfferingId) without
    follow-up questions. (Context-Aware Reasoning)
  - Unlocks multi-step plans for tool-bound intents. general_chat is strictly
    locked to steps=[]. (Multi-Step Planning)
  - Updated system prompt with consistent rule numbering and university domain
    framing. (Prompt Engineering)
  - Uses structured messages[] instead of concatenated history strings. (fixed)
"""

import json
from typing import Optional, Protocol

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

# ── Valid intent catalogue ────────────────────────────────────────────────────
VALID_INTENTS = {
    "general_chat",
    "summarization",
    "generate_exam",
    "result_query",
    "file_extraction",
    # ── New modules ──
    "complaint_submit",
    "complaint_summary",
    "file_processing",
    "cv_analysis",
    "academic_advice",
}

# ── Fallback intent ───────────────────────────────────────────────────────────
_FALLBACK_INTENT = "general_chat"

# ── Available backend tools (referenced in system prompt) ─────────────────────
_AVAILABLE_TOOLS = [
    "ResolveSubjectOffering",
    "GetStudentResults",
    "GetStudentGrades",
    "GetGPASummary",
    "GetTranscript",
    "GetSchedule",
    "GetSubjectOfferings",
    "GetCourseEnrollments",
    "GenerateExam",
    "DistributeExam",
    # ── New ──
    "SubmitComplaint",
    "GetComplaints",
    "GetStudentAcademicSummary",
    "BulkCreateStudents",
    "BulkUploadGrades",
]

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an AI Planning Agent for a university management system.

Your job is to classify the user's request and return a structured JSON plan.

## Valid Intents
- general_chat       — conversation, questions, greetings, anything not needing backend data
- summarization      — summarise a document or text
- generate_exam      — generate a university exam (doctor/admin only)
- result_query       — query academic results, grades, GPA, transcripts, schedules
- file_extraction    — extract information from an uploaded file (no bulk ops)
- complaint_submit   — student submitting a complaint or feedback about a doctor/exam/grade
- complaint_summary  — admin/doctor requesting a summary of submitted complaints
- file_processing    — bulk upload of Excel (students/grades) or PDF summarization via fileUrl
- cv_analysis        — analyzing a student CV to extract skills and give recommendations
- academic_advice    — personalized academic recommendations based on GPA and enrolled courses

## Output Schema (return ONLY this JSON, no markdown, no extra text)
{{
  "intent": "<one of the valid intents>",
  "goal_summary": "<one clear sentence describing what the user wants>",
  "is_executable": true,
  "exam_params": null,
  "pre_execution_steps": [],
  "steps": []
}}

## Rules

### 1. general_chat
- steps MUST be [] (empty array). Never add steps for general_chat.
- exam_params MUST be null.
- Use this intent for greetings, explanations, advice, and any question
  that does not require fetching real student/exam data from the backend.

### 2. Tool-bound intents (summarization, result_query, file_extraction, generate_exam)
- You MAY include steps when multiple sequential backend calls are needed.
- Available tools: {tools}
- Step format:
  {{
    "step_id": <int>,
    "action": "tool",
    "tool_name": "<one of the available tools>",
    "input_payload": {{...}},
    "depends_on": []
  }}
- Use {{{{step_N.output}}}} to reference the output of step N in a later step.
- If only one tool call is needed, leave steps=[].

### 3. generate_exam
- Populate exam_params with: collegeName, departmentName, batchName,
  subjectName, numberOfQuestions (int), examType ("midterm"|"final"),
  variationMode ("same_for_all"|"different_per_student"),
  subjectOfferingId (string|null).
- If subjectOfferingId is unknown, add ResolveSubjectOffering to pre_execution_steps.

### 4. Context-aware auto-fill (MANDATORY)
- The caller has already authenticated and their academic record is embedded in
  the request under academic_context.
- You MUST extract userId, studentId, courseId, subjectOfferingId,
  departmentId, batchId, collegeName, departmentName, batchName from
  academic_context and inject them into the relevant tool input_payload fields.
- NEVER ask the user for parameters already present in academic_context.
- NEVER leave userId or studentId blank when they exist in academic_context.
- If a required field is absent from both the user message AND academic_context,
  only then flag it as missing in goal_summary.

### 5. complaint_submit (student only)
- Use intent=complaint_submit when a student reports a problem, complains,
  or gives negative feedback about a doctor, exam, grade, or the system.
- Extract from user message: the complaint content (for "message" field).
- Required payload fields (MUST be populated from academic_context):
    userId, subjectOfferingId
- targetType MUST be one of: "Doctor" | "Exam" | "Grade" | "Other"
  Infer it from the message (e.g. "doctor" → "Doctor", "exam" → "Exam",
  "grade" / "mark" → "Grade", anything else → "Other").
- DoctorId is resolved server-side — do NOT include it in the payload.
- If role is NOT "student" → use general_chat instead.

### 6. complaint_summary (admin/doctor only)
- Use intent=complaint_summary when an admin or doctor asks to see, review,
  or summarize complaints.
- If role is "student" → use general_chat instead.

### 7. file_processing
- Use intent=file_processing when the user message contains a fileUrl
  OR mentions uploading/processing a file for bulk operations.
- Do NOT use this for single-file text extraction (use file_extraction).

### 8. cv_analysis
- Use intent=cv_analysis when the user wants their CV reviewed, analyzed,
  or feedback on skills, experience, or job readiness.

### 9. academic_advice
- Use intent=academic_advice when a student asks for study advice, course
  recommendations, or wants to know how to improve their GPA.

### 10. When in doubt → use general_chat with steps=[].

### Multi-step example (result_query — grades then GPA):
{{
  "intent": "result_query",
  "goal_summary": "Fetch student grades and calculate GPA",
  "is_executable": true,
  "exam_params": null,
  "pre_execution_steps": [],
  "steps": [
    {{"step_id": 1, "action": "tool", "tool_name": "GetStudentGrades",
      "input_payload": {{"userId": "<from context>"}}, "depends_on": []}},
    {{"step_id": 2, "action": "tool", "tool_name": "GetGPASummary",
      "input_payload": {{"gradeData": "{{{{step_1.output}}}}"}}, "depends_on": [1]}}
  ]
}}
""".format(tools=", ".join(_AVAILABLE_TOOLS))


class MemoryStore(Protocol):
    """Protocol defining how the Planner retrieves historical context."""

    async def get_context(self, user_id: str | None) -> str: ...


class PlannerAgent(BaseAgent):
    """
    Generates an ExecutionPlan by asking the LLM to classify the user's intent.

    v4.0 upgrades:
      - Injects academic_context for context-aware parameter resolution.
      - Allows multi-step plans for tool-bound intents.
      - Uses structured messages[] history.
    """

    def __init__(
        self,
        model_router,
        ranker=None,
        memory: Optional[MemoryStore] = None,
    ):
        self.model_router = model_router
        self.ranker = ranker
        self.memory = memory

    # ─────────────────────────────────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────────────────────────────────

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        1. Pull optional memory summary for long-term context.
        2. Inject academic_context to allow context-aware auto-filling.
        3. Build structured messages[] from history + current message.
        4. Call LLM for classification JSON.
        5. Validate, sanitise, and enrich the resulting ExecutionPlan.
        6. Return AgentOutput(status="success", data={"plan": plan}).
        """
        logger.info("PlannerAgent: starting for user_id=%s", agent_input.user_id)

        # ── Optional memory context ───────────────────────────────────────
        memory_prefix = ""
        if self.memory:
            try:
                past = await self.memory.get_context(agent_input.user_id)
                if past:
                    memory_prefix = f"[Conversation summary]: {past}\n\n"
            except Exception as mem_exc:
                logger.warning("PlannerAgent: memory lookup failed — %s", mem_exc)

        # ── Extract context components ────────────────────────────────────
        ctx = agent_input.context or {}
        role = ctx.get("role", "user")
        raw_history: list[dict] = ctx.get("history", [])
        academic_ctx: dict = ctx.get("academic_context", {})

        # Compact summary of academic context for auto-filling parameters
        auto_fill_note = ""
        if academic_ctx:
            # Only expose safe, useful fields — never passwords or tokens
            safe_keys = [
                "userId", "studentId", "courseId", "subjectOfferingId",
                "departmentId", "batchId", "collegeName", "departmentName",
            ]
            relevant = {k: v for k, v in academic_ctx.items() if k in safe_keys and v}
            if relevant:
                auto_fill_note = (
                    f"\nAvailable context for auto-filling parameters: "
                    f"{json.dumps(relevant, ensure_ascii=False)}"
                )

        # ── Build structured history turns (last 3 pairs = 6 messages) ────
        history_turns: list[dict] = []
        for turn in raw_history[-6:]:
            turn_role = turn.get("role", "user")
            turn_content = str(turn.get("content", ""))
            if turn_role in ("user", "assistant") and turn_content:
                history_turns.append({"role": turn_role, "content": turn_content})

        # ── Compose user classification request ───────────────────────────
        user_content = (
            f"{memory_prefix}"
            f"User role: {role}\n"
            f"User message: {agent_input.message}"
            f"{auto_fill_note}"
        )

        # ── Call LLM ──────────────────────────────────────────────────────
        raw_json = await self._call_planner_model(history_turns, user_content)

        # ── Parse + validate → ExecutionPlan ──────────────────────────────
        plan = self._parse_plan(raw_json, agent_input)

        # ── Deterministic guard for generate_exam ─────────────────────────
        plan = self._ensure_resolve_step(plan)

        logger.info(
            "PlannerAgent: intent=%r steps=%d goal=%r",
            plan.intent, len(plan.steps), plan.goal_summary,
        )

        return AgentOutput(
            status="success",
            response=plan.goal_summary,
            data={"plan": plan},
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    async def _call_planner_model(
        self, history_turns: list[dict], user_content: str
    ) -> dict | None:
        """
        Send the planning request to OpenAI via structured messages[].

        Message order:
          [system]  _SYSTEM_PROMPT
          [prior turns from history…]
          [user]    role + message + academic_context note

        Falls back to gpt-3.5-turbo if gpt-4o-mini returns nothing.
        """
        try:
            logger.debug("PlannerAgent: requesting JSON from openai/gpt-4o-mini")
            parsed = await self.model_router.generate_structured_json(
                prompt=user_content,
                system_instruction=_SYSTEM_PROMPT,
                model_id="openai/gpt-4o-mini",
            )

            if not parsed:
                logger.warning(
                    "PlannerAgent: openai/gpt-4o-mini returned empty — fallback chain will handle it"
                )

            logger.debug("PlannerAgent: raw plan = %s", parsed)
            return parsed

        except Exception as exc:
            logger.error("PlannerAgent: model call failed — %s", exc, exc_info=True)
            return None

    def _parse_plan(self, raw: dict | None, agent_input: AgentInput) -> ExecutionPlan:
        """
        Validate the raw LLM dict into an ExecutionPlan with safety guards:
          - Invalid intent → downgrade to general_chat
          - general_chat   → force steps=[], exam_params=None
          - tool_name in steps → validated by executor (not here)
        """
        if not raw:
            return self._fallback_plan(agent_input.message)

        # Normalise intent
        intent = raw.get("intent", _FALLBACK_INTENT)
        if intent not in VALID_INTENTS:
            logger.warning(
                "PlannerAgent: unknown intent %r — falling back to %s",
                intent, _FALLBACK_INTENT,
            )
            intent = _FALLBACK_INTENT
            raw["intent"] = intent

        # HARD RULE: general_chat must never have steps or exam context
        if intent == "general_chat":
            raw["steps"] = []
            raw["exam_params"] = None

        # HARD RULE: steps must always be an empty list regardless of what
        # the planner returned (planner is advisory; steps come from planner
        # only for tool-bound intents, handled via pre_execution_steps or
        # the module architecture)
        # NOTE: We allow non-empty steps for non-chat intents (multi-step unlock)
        # but sanitise any non-list value to an empty list.
        if not isinstance(raw.get("steps"), list):
            raw["steps"] = []

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
                            "subjectName": plan.exam_params.subjectName,
                        },
                    )
                )
        return plan
