"""
app/agents/executor.py  —  v4.1 Production-Grade Dispatcher

Capabilities in this version:
  1. Context-Aware Reasoning    — user_id + academic_context auto-injected into tool payloads
  2. Multi-Step Planning        — sequential tool chaining with {{step_N.output}} interpolation
  3. Tool Result Reasoning      — LLM narrates raw tool data in natural, role-appropriate language
  4. Memory                     — preferences injected into system prompt (set by Agent)
  5. Explainability Mode        — optional transparency block appended to responses
  6. Role-Specific Behaviour    — three distinct system prompts (student / doctor / admin)
  7. Error Recovery             — 1 automatic retry; graceful LLM fallback on persistent failure
  8. Smart Follow-Up Suggestions— deterministic suggestions per intent+role, no LLM cost
  9. Hybrid Model Usage         — model_id selected by Agent routing, passed in via context
 10. Structured Response        — AgentOutput.data carries {suggestions, actions_available}
 11. Strict RBAC               — single source of truth via app/core/rbac.py:
                                  log_blocked_attempt() emits structured WARNING for monitoring

Routing order in execute() (strictly enforced):
  0. RBAC gate              → immediate forbidden if role not permitted
                              (calls rbac.log_blocked_attempt() for audit trail)
  1. Dedicated module       → _run_module()
  2. general_chat OR empty  → _fallback_model_call()  [NO ECHO PATH]
  3. Non-empty steps        → _run_plan()
  4. No plan                → _fallback_model_call()
"""

from __future__ import annotations

import asyncio
import html
import importlib
import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Awaitable, TYPE_CHECKING

from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan
from app.core.logging import logger

if TYPE_CHECKING:
    from app.agents.model_router import ModelRouter


# ── Module class registry ─────────────────────────────────────────────────────
_MODULE_CLASS_MAP: Dict[str, tuple[str, str]] = {
    "exam_generation": ("app.modules.exam_generation", "ExamGenerationModule"),
    "summarization":   ("app.modules.summarization",   "SummarizationModule"),
    "file_extraction": ("app.modules.file_extraction",  "FileExtractionModule"),
    "result_query":    ("app.modules.result_query",      "ResultQueryModule"),
    # ── New modules ──
    "complaint_submit":   ("app.modules.complaint",           "ComplaintModule"),
    "complaint_summary":  ("app.modules.complaint",           "ComplaintModule"),
    "file_processing":    ("app.modules.file_processor",      "FileProcessorModule"),
    "cv_analysis":        ("app.modules.cv_analysis",         "CVAnalysisModule"),
    "academic_advice":    ("app.modules.academic_advisor",    "AcademicAdvisorModule"),
    "material_explanation": ("app.modules.material_explanation", "MaterialExplanationModule"),
    "dynamic_api_module": ("app.modules.dynamic_api",         "DynamicApiModule"),
}

# ── Tool allowlist ─────────────────────────────────────────────────────────────
# ONLY these tool names may be dispatched to the backend from _run_plan().
ALLOWED_TOOL_NAMES: frozenset[str] = frozenset({
    "ResolveSubjectOffering",
    "GetStudentResults",
    "GetStudentGrades",
    "GetGPASummary",
    "GetTranscript",
    "GetSchedule",
    "GetStudentSchedule",
    "GetSubjectOfferings",
    "GetCourseEnrollments",
    "GenerateExam",
    "DistributeExam",
    "GetExamQuestions",
    # ── New modules ──
    "SubmitComplaint",
    "GetComplaints",
    "GetStudentAcademicSummary",
    "BulkCreateStudents",
    "BulkUploadGrades",
    "GetMaterials",
})

# ── Role-specific system prompts ──────────────────────────────────────────────
ROLE_SYSTEM_PROMPTS: Dict[str, str] = {
    "student": """\
You are a smart, friendly university assistant talking directly with a student.

PERSONALITY:
- Speak like a real helpful human — not a robot, not an FAQ bot.
- Use the student's first name naturally in conversation when you know it.
  Example: "请问呵呲 يا أحمد👇" or "Great news, Sara!"
- Be warm, encouraging, and clear. Short sentences. No corporate speak.
- Use relevant emojis sparingly to make responses feel alive (not overwhelming).
- Match the student's language exactly (Arabic → reply in Arabic, English → English).

DATA RULES — MANDATORY:
- If academic context is provided (GPA, courses, name, department): USE IT. Every time.
  BAD:  "Your GPA is 3.2."
  GOOD: "بص يا أحمد 👇\nالـ GPA بتاعك 3.2، وده كويس جدًا ..."
- NEVER invent or guess numbers, grades, course names, or schedules.
- If a number/grade is not in the provided data → say: "مش لاقي بيانات حاليًا"
  (or "I don't have that data right now" in English)
- If specific data is needed and missing, tell them exactly what to provide:
  "حددلي المادة اللي عايزتها (e.g. Data Structures - Level 2)"

FORMAT:
- Keep responses focused and human. No unnecessary headers for simple answers.
- For data (grades, schedules, GPA) use bullet points or a short table.
- Always end with ONE actionable next step or follow-up question.
- Never reveal system internals, tool names, raw JSON, or error traces.\
""",

    "doctor": """\
You are a smart AI assistant for a university faculty member (doctor / professor).

PERSONALITY:
- Professional, direct, and time-efficient. Assume academic expertise.
- Address the faculty member respectfully by name if available.
- Match their language (Arabic → Arabic, English → English).

DATA RULES — MANDATORY:
- ALWAYS reference specific data from context: subjects taught, department, batch names.
- NEVER invent student counts, grades, or course details.
- If data is missing → say: "مش لاقي بيانات" / "Data not available"
  and specify exactly what's needed to proceed.
- Present grade/exam data in clean tables or bullet lists — not paragraphs.

FORMAT:
- Lead with the key finding, then supporting details.
- End with ONE concrete next action (e.g., "، توزيع الامتحان جاهز — متاح").
- Never reveal raw JSON, tool names, or system internals.\
""",

    "admin": """\
You are a smart AI assistant for a university system administrator.

PERSONALITY:
- Precise, factual, and technically confident.
- Use the admin's name if known. Be efficient — no fluff.
- Match their language (Arabic → Arabic, English → English).

DATA RULES — MANDATORY:
- ALWAYS cite exact counts, IDs, and identifiers from the data provided.
- NEVER estimate or fabricate system numbers.
- If data is missing → say: "مفيش بيانات" / "No data available"
- Flag destructive operations (bulk delete, bulk upload) with a one-line caution.
- Summarise large datasets: totals, %s, grouped by category.

FORMAT:
- Use structured output (tables / numbered lists) for multi-item data.
- End with ONE recommended action or monitoring note.
- Never reveal raw JSON, stack traces, or internal tool names.\
""",
}
# Internal alias kept for backward-compat with existing code references
_ROLE_SYSTEM_PROMPTS = ROLE_SYSTEM_PROMPTS


# ── Follow-up suggestion map ──────────────────────────────────────────────────
# Deterministic (no LLM cost). Key: (intent, role) → list[str]
_SUGGESTIONS_MAP: Dict[str, Dict[str, List[str]]] = {
    "general_chat": {
        "student": ["Check my grades", "What's my schedule today?", "Explain a course topic"],
        "doctor":  ["Generate an exam", "View course enrollments", "Check student results"],
        "admin":   ["View system stats", "List all departments", "Check pending actions"],
    },
    "result_query": {
        "student": ["View full transcript", "Calculate my GPA", "Which courses did I pass?"],
        "doctor":  ["Export results report", "Compare batch results", "View top students"],
        "admin":   ["Export all results", "View department summary", "Filter by GPA range"],
    },
    "generate_exam": {
        "student": ["View exam schedule", "Ask about exam topics", "Get study tips"],
        "doctor":  ["Distribute this exam", "Preview questions", "Generate for another subject"],
        "admin":   ["Review all exams", "Export exam report", "View submitted exams"],
    },
    "summarization": {
        "student": ["Ask questions about this content", "Get key points only", "Explain in simpler terms"],
        "doctor":  ["Generate questions from this", "Create exam from content", "Save as notes"],
        "admin":   ["Archive this summary", "Share with department", "Generate report"],
    },
    "file_extraction": {
        "student": ["Summarize this file", "Ask questions about it", "Extract key topics"],
        "doctor":  ["Generate exam from file", "Extract syllabus topics", "Summarise for students"],
        "admin":   ["Process all pending files", "Extract and archive", "Generate department report"],
    },
    # ── New modules ──
    "complaint_submit": {
        "student": ["Track my complaint", "Submit another complaint", "Ask about my grades"],
        "doctor":  ["View complaints about me", "Generate exam", "Check course enrollments"],
        "admin":   ["View all complaints", "Complaint summary", "Export complaint report"],
    },
    "complaint_summary": {
        "student": ["Check my grades", "View my schedule", "Ask general question"],
        "doctor":  ["Filter by subject", "View recent complaints", "Export report"],
        "admin":   ["Filter by department", "View trend analysis", "Resolve complaint"],
    },
    "file_processing": {
        "student": ["Check my grades", "View my profile", "Ask a question"],
        "doctor":  ["Generate exam", "View enrollments", "Check results"],
        "admin":   ["Upload another file", "View student list", "Check upload status"],
    },
    "cv_analysis": {
        "student": ["Improve my CV further", "Recommend courses", "Get academic advice"],
        "doctor":  ["Analyse student CV", "Generate exam", "View enrollments"],
        "admin":   ["Bulk CV analysis", "Department skill summary", "Generate report"],
    },
    "academic_advice": {
        "student": ["View my grades", "Check my GPA", "Analyse my CV"],
        "doctor":  ["View class performance", "Generate exam", "Check enrollments"],
        "admin":   ["Department GPA summary", "At-risk students", "Generate report"],
    },
    "material_explanation": {
        "student": ["Ask a question about this topic", "Get study tips", "View my grades"],
        "doctor":  ["Generate exam from this material", "Summarize for students", "View enrollments"],
        "admin":   ["View all materials", "Export material report", "Check upload status"],
    },
}

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_HISTORY_TURNS = 10
_MAX_INPUT_LENGTH  = 4_000
_TOOL_RETRY_DELAY  = 1.0    # seconds between retry attempts

# ── Data-sensitive intents — MUST have backend data before LLM responds ────────
# If any of these arrive at _fallback_model_call() without a tool having run,
# the AI is BLOCKED from answering from its training memory.
_DATA_SENSITIVE_INTENTS: frozenset[str] = frozenset({
    "backend_api_query",  # grades, stats, records dynamically read from Swagger
    "complaint_summary",  # complaint records from DB
    "file_processing",    # bulk DB operations
    "generate_exam",      # must invoke backend exam service
    "academic_advice",    # requires real GPA / grade data
})

# Bilingual blocked-data message (shown to user when gate fires).
_NO_BACKEND_DATA_MSG = (
    "مش لاقي بيانات من السيستم حالياً، حاول تاني. "
    "لو المشكلة فاضلت، تواصل مع الدعم التقني.\n"
    "(No data retrieved from the system right now — please try again.)"
)



# ── Input sanitisation ────────────────────────────────────────────────────────

def _sanitise(text: str) -> str:
    """
    Sanitise user content before injecting into LLM prompts.
      1. Strip whitespace.
      2. Truncate to _MAX_INPUT_LENGTH characters.
      3. HTML-escape special characters to defuse common prompt-injection tricks.
    """
    if not isinstance(text, str):
        text = str(text)
    return html.escape(text.strip()[:_MAX_INPUT_LENGTH])


def _build_explain_block(steps_narrative: List[str], intent: str) -> str:
    """Build a brief transparency note for explainability mode."""
    if not steps_narrative:
        return "\n\nℹ️ *I used my general knowledge to answer this question.*"
    steps_text = "; ".join(steps_narrative)
    return (
        f"\n\nℹ️ *Transparency: To answer your request about **{intent.replace('_', ' ')}**, "
        f"I performed the following: {steps_text}. "
        f"The response above is based on the data retrieved from your academic records.*"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PlanExecutor
# ─────────────────────────────────────────────────────────────────────────────

class PlanExecutor:
    """
    Routes each orchestration request to the correct execution path and
    enriches responses with natural-language reasoning, follow-up
    suggestions, and optional explainability notes.
    """

    def __init__(
        self,
        backend_execution_func: Callable[
            [str, Dict[str, Any], Optional[str], Optional[str]],
            Awaitable[Dict[str, Any]],
        ],
        model_router: Optional["ModelRouter"] = None,
    ) -> None:
        self.backend_execution_func = backend_execution_func
        self.model_router = model_router
        self._module_cache: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    async def execute(
        self,
        plan: Any,
        input_context: AgentInput,
        module_name: str = "model_only",
    ) -> AgentOutput:
        """
        Dispatch to the correct handler.

        Priority order (strictly enforced):
          0. RBAC gate              → immediate forbidden if role not permitted
          1. Dedicated module       → _run_module()
          2. general_chat / empty   → _fallback_model_call()  [NO ECHO PATH]
          3. Non-empty steps        → _run_plan()
          4. No plan                → _fallback_model_call()
        """
        ctx    = input_context.context or {}
        role   = ctx.get("role", "student")
        # Resolve intent: explicit module_name takes priority, else read from plan
        intent = module_name if module_name in _MODULE_CLASS_MAP else (
            getattr(plan, "intent", None) or "general_chat"
        )

        # 0. RBAC gate ─────────────────────────────────────────────────────────
        from app.core.rbac import is_allowed, get_denial_message, log_blocked_attempt
        if not is_allowed(intent, role):
            # Emit structured audit log entry for monitoring / SIEM integration
            log_blocked_attempt(
                intent=intent,
                role=role,
                user_id=input_context.user_id,
                extra={"module": module_name},
            )
            return AgentOutput(
                status="forbidden",
                response=get_denial_message(intent, role),
                data={"blocked_intent": intent, "role": role},
            )

        # 1. Module dispatch ────────────────────────────────────────────────────
        if module_name in _MODULE_CLASS_MAP:
            logger.info("PlanExecutor: module route → '%s'", module_name)
            return await self._run_module(module_name, plan, input_context)

        # 2. general_chat OR empty steps → LLM (no echo path) ─────────────────
        if plan and isinstance(plan, ExecutionPlan):
            is_chat   = plan.intent == "general_chat"
            has_steps = bool(plan.steps)

            if is_chat or not has_steps:
                logger.info(
                    "PlanExecutor: intent=%r steps=0 → _fallback_model_call",
                    plan.intent,
                )
                return await self._fallback_model_call(input_context, intent=plan.intent)

            # 3. Multi-step plan → step runner ─────────────────────────────────
            logger.info(
                "PlanExecutor: intent=%r steps=%d → _run_plan",
                plan.intent, len(plan.steps),
            )
            return await self._run_plan(plan, input_context)

        # 4. No plan → LLM ─────────────────────────────────────────────────────
        logger.info("PlanExecutor: no plan → _fallback_model_call")
        return await self._fallback_model_call(input_context, intent=intent)


    # ──────────────────────────────────────────────────────────────────────
    #  Module dispatch
    # ──────────────────────────────────────────────────────────────────────

    def _get_module(self, module_name: str) -> Any:
        """Lazily import and cache the module class instance."""
        if module_name not in self._module_cache:
            mod_path, class_name = _MODULE_CLASS_MAP[module_name]
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, class_name)

            init_params = inspect.signature(cls.__init__).parameters
            kwargs: Dict[str, Any] = {}
            if "model_router" in init_params:
                kwargs["model_router"] = self.model_router
            if "backend_client" in init_params:
                from app.services.backend_client import tool_execution_client
                kwargs["backend_client"] = tool_execution_client

            self._module_cache[module_name] = cls(**kwargs)
            logger.info("PlanExecutor: loaded module '%s'.", module_name)

        return self._module_cache[module_name]

    async def _run_module(
        self,
        module_name: str,
        plan: Any,
        input_context: AgentInput,
    ) -> AgentOutput:
        """Fetch (or load) and run the named module."""
        logger.info("PlanExecutor: dispatching to module '%s'.", module_name)
        try:
            module = self._get_module(module_name)
            result = await module.run(input_context, plan)
            # Inject suggestions if not already present
            if result.status == "success":
                role   = (input_context.context or {}).get("role", "student")
                intent = getattr(plan, "intent", module_name) or module_name
                suggestions = self._get_suggestions(intent, role)
                data = result.data or {}
                data.setdefault("suggestions", suggestions)
                data.setdefault("actions_available", suggestions)
                return AgentOutput(
                    status=result.status,
                    response=result.response,
                    data=data,
                )
            return result

        except Exception as exc:
            logger.error(
                "PlanExecutor: module '%s' raised an error: %s",
                module_name, exc, exc_info=True,
            )
            role     = (input_context.context or {}).get("role", "student")
            model_id = (input_context.context or {}).get("selected_model", "gpt-4o-mini")
            return await self._graceful_error_response(
                module_name, role, model_id,
            )

    # ──────────────────────────────────────────────────────────────────────
    #  Multi-step plan runner
    # ──────────────────────────────────────────────────────────────────────

    async def _run_plan(
        self, plan: ExecutionPlan, input_context: AgentInput
    ) -> AgentOutput:
        """
        Execute an ExecutionPlan step-by-step with:
          - Allowlist validation per tool step
          - Automatic 1-retry on transient tool failures
          - Tool result narration via LLM
          - Follow-up suggestions
          - Optional explainability block
        """
        if not plan.is_executable:
            return AgentOutput(
                status="failed",
                response="This plan is marked as non-executable.",
            )

        ctx        = input_context.context or {}
        role       = ctx.get("role", "student")
        model_id   = ctx.get("selected_model", "openai/gpt-4o-mini")
        explain    = ctx.get("explain", False)
        intent     = plan.intent or "general_chat"
        user_id    = input_context.user_id
        academic_ctx: dict = ctx.get("academic_context", {}) or {}

        execution_results: Dict[int, Any] = {}
        successful_steps = 0
        steps_narrative: List[str] = []
        ordered_steps = sorted(plan.steps, key=lambda s: s.step_id)

        for step in ordered_steps:
            logger.info(
                "PlanExecutor: step_id=%s action=%s", step.step_id, step.action
            )
            payload = self._interpolate(step.input_payload, execution_results)

            # ── Context-aware auto-injection ────────────────────────────
            # Priority: academic_context values first, then fall back to
            # top-level user_id.  Values already set in input_payload are
            # NEVER overridden (planner is authoritative for explicit values).
            _CONTEXT_FIELD_MAP = {
                "userId":            ["userId", "studentId"],
                "studentId":         ["studentId", "userId"],
                "courseId":          ["courseId"],
                "subjectOfferingId": ["subjectOfferingId"],
                "departmentId":      ["departmentId"],
                "batchId":           ["batchId"],
            }
            for payload_key, ctx_keys in _CONTEXT_FIELD_MAP.items():
                if payload_key not in payload:   # don't overwrite explicit values
                    for ck in ctx_keys:
                        val = academic_ctx.get(ck)
                        if val:
                            payload[payload_key] = val
                            break

            # Fallback: inject top-level user_id if still missing
            if user_id and "userId" not in payload:
                payload["userId"] = user_id

            try:
                if step.action == "tool":
                    # ── Allowlist validation ─────────────────────────────
                    if step.tool_name not in ALLOWED_TOOL_NAMES:
                        logger.warning(
                            "PlanExecutor: BLOCKED disallowed tool '%s' at step %s",
                            step.tool_name, step.step_id,
                        )
                        return AgentOutput(
                            status="failed",
                            response=(
                                "I'm unable to perform that action — "
                                "the requested operation is not permitted."
                            ),
                        )

                    result = await self._execute_tool_with_retry(
                        step.tool_name, payload,
                        input_context.auth_header, user_id,
                    )
                    steps_narrative.append(f"retrieved data via {step.tool_name}")
                    logger.info(
                        "PlanExecutor: step %s → tool='%s' success",
                        step.step_id, step.tool_name,
                    )

                elif step.action == "model":
                    prompt  = _sanitise(payload.get("prompt", str(payload)))
                    sys_ins = payload.get("system_instruction", "")
                    text    = await self.model_router.generate(
                        prompt=prompt,
                        system_instruction=sys_ins,
                        model_id=model_id,
                    )
                    result = {"output": text}
                    steps_narrative.append("generated AI text")

                elif step.action == "agent_module":
                    result_out = await self._run_module(
                        step.module_name or "", plan, input_context
                    )
                    if result_out.status == "clarification_needed":
                        return result_out
                    result = {"output": result_out.response, "data": result_out.data}
                    steps_narrative.append(f"processed via {step.module_name} module")

                else:
                    result = {"skipped": True, "reason": f"Unknown action '{step.action}'"}

                execution_results[step.step_id] = result

                if isinstance(result, dict) and "error" in result:
                    logger.warning(
                        "PlanExecutor: step %s returned error in result", step.step_id
                    )
                    return await self._graceful_error_response(intent, role, model_id)

                successful_steps += 1

            except Exception as exc:
                logger.error(
                    "PlanExecutor: step %s failed after retry: %s",
                    step.step_id, exc, exc_info=True,
                )
                return await self._graceful_error_response(intent, role, model_id)

        # ── Narrate results via LLM ───────────────────────────────────────
        narrative = await self._reason_about_results(
            execution_results, intent, role, model_id
        )

        # ── Explainability block ──────────────────────────────────────────
        if explain and steps_narrative:
            narrative += _build_explain_block(steps_narrative, intent)

        suggestions = self._get_suggestions(intent, role)

        return AgentOutput(
            status="success",
            response=narrative,
            data={
                "results":           execution_results,
                "suggestions":       suggestions,
                "actions_available": suggestions,
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Tool retry
    # ──────────────────────────────────────────────────────────────────────

    async def _execute_tool_with_retry(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        auth_header: Optional[str],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Execute a backend tool call with one automatic retry on any failure.

        Attempt 1 → failure → sleep 1 s → Attempt 2 → failure → raise.
        """
        last_exc: Exception = RuntimeError("Tool did not execute.")
        for attempt in range(2):
            try:
                return await self.backend_execution_func(
                    tool_name, payload, auth_header, user_id
                )
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "PlanExecutor: tool '%s' failed (attempt 1), retrying in %.1fs: %s",
                        tool_name, _TOOL_RETRY_DELAY, exc,
                    )
                    await asyncio.sleep(_TOOL_RETRY_DELAY)
        raise last_exc

    # ──────────────────────────────────────────────────────────────────────
    #  Tool result reasoning (LLM narration)
    # ──────────────────────────────────────────────────────────────────────

    async def _reason_about_results(
        self,
        execution_results: Dict[int, Any],
        intent: str,
        role: str,
        model_id: str,
    ) -> str:
        """
        Ask the LLM to narrate raw tool results in natural, role-appropriate language.

        Raw JSON is never shown to the user — the LLM extracts the meaningful
        information and presents it conversationally.
        """
        if not self.model_router or not execution_results:
            return "The operation completed successfully."

        # Build compact, safe result representation
        result_parts: List[str] = []
        for step_id, result in sorted(execution_results.items()):
            if isinstance(result, dict):
                # Strip binary/internal keys before sending to LLM
                clean = {
                    k: v for k, v in result.items()
                    if k not in ("_raw_bytes", "skipped", "content_type")
                }
                if clean:
                    result_parts.append(
                        f"Step {step_id} data:\n"
                        + json.dumps(clean, ensure_ascii=False)[:1_200]
                    )
            elif isinstance(result, str) and result.strip():
                result_parts.append(f"Step {step_id} result: {result[:600]}")

        if not result_parts:
            return "The operation completed successfully."

        results_str = "\n\n".join(result_parts)
        role_prompt = _ROLE_SYSTEM_PROMPTS.get(role, _ROLE_SYSTEM_PROMPTS["student"])

        # Build a personalisation note to inject into the narration request
        # so the LLM knows WHO it's talking to and can use their name + GPA.
        ctx        = {}
        student_name = ""
        gpa_note     = ""
        courses_note = ""
        # execution_results shares scope — pull academic_ctx from input_context if accessible
        # We encode it via a helper: narration always runs inside _run_plan which has access.
        # The personalization context is passed via result metadata if available.
        for _step_result in execution_results.values():
            if isinstance(_step_result, dict):
                if _step_result.get("studentName"):
                    student_name = _step_result["studentName"]
                if _step_result.get("gpa") or _step_result.get("GPA"):
                    gpa_note = str(_step_result.get("gpa") or _step_result.get("GPA"))
                if _step_result.get("enrolledCourses") or _step_result.get("courses"):
                    c_list = _step_result.get("enrolledCourses") or _step_result.get("courses") or []
                    if isinstance(c_list, list) and c_list:
                        courses_note = ", ".join(
                            (c.get("name") or c.get("subjectName") or str(c))
                            if isinstance(c, dict) else str(c)
                            for c in c_list[:4]
                        )

        personalisation = ""
        if student_name:
            personalisation += f"The student's name is {student_name}. "
        if gpa_note:
            personalisation += f"Their GPA is {gpa_note}. "
        if courses_note:
            personalisation += f"They are enrolled in: {courses_note}. "
        if personalisation:
            personalisation = (
                f"\n\nPersonalisation context: {personalisation}"
                "Use this naturally in your response (e.g. use their name, reference their GPA). "
                "NEVER invent data not in the result."
            )

        narration_instruction = (
            f"You retrieved the following real university data for a {role}. "
            "Present it in a warm, human, conversational way. "
            "NEVER show raw JSON, field names, or technical identifiers. "
            "If the data contains numbers (grades, GPA, counts): state them clearly and contextually. "
            "If no useful data is present, say: ’مش لاقي بيانات حاليًا‘ (or 'No data found right now' in English). "
            "Be concise but complete."
            + personalisation
            + f"\n\nRetrieved data:\n{results_str}"
        )

        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user",   "content": narration_instruction},
        ]

        try:
            response = await self.model_router.generate_with_messages(
                messages=messages, model_id=model_id
            )
            return response or "The operation completed successfully."
        except Exception as exc:
            logger.error("PlanExecutor._reason_about_results error: %s", exc)
            return "The operation completed successfully."

    # ──────────────────────────────────────────────────────────────────────
    #  Graceful error recovery
    # ──────────────────────────────────────────────────────────────────────

    async def _graceful_error_response(
        self,
        intent: str,
        role: str,
        model_id: str = "openai/gpt-4o-mini",
    ) -> AgentOutput:
        """
        Generate a user-friendly error message via LLM.

        NEVER exposes raw error strings, stack traces, or backend responses
        to the end user. Always offers a constructive alternative.
        """
        logger.info(
            "PlanExecutor: graceful error recovery → intent=%s role=%s", intent, role
        )

        if not self.model_router:
            return AgentOutput(
                status="failed",
                response=(
                    "I'm sorry, I encountered a temporary issue processing your request. "
                    "Please try again in a moment."
                ),
            )

        role_prompt = _ROLE_SYSTEM_PROMPTS.get(role, _ROLE_SYSTEM_PROMPTS["student"])
        intent_label = intent.replace("_", " ")

        messages = [
            {"role": "system", "content": role_prompt},
            {
                "role": "user",
                "content": (
                    f"The system encountered a temporary issue while completing a "
                    f"'{intent_label}' request. Please generate a friendly, helpful "
                    "message that:\n"
                    "1. Apologises briefly without being excessive.\n"
                    "2. Suggests a concrete alternative the user can try.\n"
                    "3. Does NOT reveal any technical details, error codes, or system internals.\n"
                    "Respond in the same language the user would expect (Arabic or English)."
                ),
            },
        ]

        try:
            response = await self.model_router.generate_with_messages(
                messages=messages, model_id="openai/gpt-4o-mini"
            )
            return AgentOutput(
                status="failed",
                response=(
                    response
                    or "I'm sorry, I encountered a temporary issue. Please try again in a moment."
                ),
            )
        except Exception as exc:
            logger.error("PlanExecutor._graceful_error_response LLM call failed: %s", exc)
            return AgentOutput(
                status="failed",
                response=(
                    "I'm sorry, I encountered a temporary issue. "
                    "Please try again in a moment."
                ),
            )

    # ──────────────────────────────────────────────────────────────────────
    #  Fallback LLM call (general_chat + no-plan paths)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_raw_data_request(message: str) -> bool:
        """
        CRITICAL GLOBAL DATA GUARD:
        Scans the raw user message for data-seeking keywords.
        If the user asks for numbers, grades, schedules, or system facts,
        this override prevents the LLM from fabricating an answer, regardless
        of what the planner classified the intent as.
        """
        msg = message.lower()
        keywords = {
            "كم", "عدد", "total", "count",          # numbers requests
            "درجات", "درجة", "gpa", "result",       # student data
            "جدول", "مواعيد", "timetable",          # schedules
            "مين", "ايه البيانات", "بيانات"           # system queries
        }
        # We do simple substring matching to catch prefixed Arabic words
        # (e.g. "درجاتي", "الجدول")
        return any(kw in msg for kw in keywords)

    async def _fallback_model_call(
        self,
        input_context: AgentInput,
        intent: str = "general_chat",
    ) -> AgentOutput:
        """
        Direct LLM call for general_chat and all cases without a step plan.

        DATA-FIRST GATE (Step 0)
        ────────────────────────
        If `intent` is in _DATA_SENSITIVE_INTENTS and this method is called
        (meaning no backend tool was executed), we BLOCK the response.
        The AI is FORBIDDEN from fabricating grades, GPA, counts, or any
        student/system data from its training memory.

        Builds structured messages[] with:
          - Role-specific system prompt (student / doctor / admin)
          - Language preference injected if stored in user preferences
          - Up to _MAX_HISTORY_TURNS of sanitised conversation history
          - Sanitised current user message

        Appends explainability block if explain=True.
        Attaches deterministic follow-up suggestions.
        """
        # ── Step 0: Data-sensitive intent gate ───────────────────────────────────────
        if intent in _DATA_SENSITIVE_INTENTS or self._is_raw_data_request(input_context.message):
            logger.warning(
                "PlanExecutor [GLOBAL DATA-GATE]: blocked fallback with no backend data — "
                "intent=%r, message=%r, user_id=%s",
                intent, input_context.message, input_context.user_id,
            )
            role = (input_context.context or {}).get("role", "student")
            suggestions = self._get_suggestions(intent, role)
            return AgentOutput(
                status="no_data",
                response=_NO_BACKEND_DATA_MSG,
                data={
                    "blocked_intent":   intent,
                    "reason":           "global_data_guard_no_backend_call",
                    "suggestions":      suggestions,
                    "actions_available": suggestions,
                },
            )

        if not self.model_router:

            return AgentOutput(status="failed", response="No model router available.")

        ctx          = input_context.context or {}
        role         = ctx.get("role", "student")
        raw_history  = ctx.get("history", []) or []
        model_id     = ctx.get("selected_model", "openai/gpt-4o-mini")
        explain      = ctx.get("explain", False)
        prefs        = ctx.get("preferences", {}) or {}
        academic_ctx: dict = ctx.get("academic_context", {}) or {}

        # ── Role-specific system prompt ───────────────────────────────────
        base_prompt = _ROLE_SYSTEM_PROMPTS.get(role, _ROLE_SYSTEM_PROMPTS["student"])

        # Inject language preference if stored
        lang_pref = prefs.get("language", "")
        if lang_pref:
            base_prompt += (
                f"\n\nUser language preference: {lang_pref}. "
                "Respond in that language whenever possible."
            )

        # Inject any interests for personalisation
        interests = prefs.get("interests", [])
        if interests:
            base_prompt += (
                f"\nUser academic interests: {', '.join(interests[:5])}. "
                "Tailor responses to these interests when relevant."
            )

        # ── Academic context injection ───────────────────────────────────────────────
        # Build a human-readable profile block instead of raw JSON.
        # This tells the LLM WHO is asking and what we know about them,
        # so it can personalise the response naturally (name, GPA, courses).
        if academic_ctx:
            profile_lines: List[str] = []

            name = academic_ctx.get("studentName") or academic_ctx.get("name") or ""
            dept = academic_ctx.get("departmentName") or ""
            batch = academic_ctx.get("batchName") or ""
            college = academic_ctx.get("collegeName") or ""
            gpa = academic_ctx.get("gpa") or academic_ctx.get("GPA") or ""
            courses = academic_ctx.get("enrolledCourses") or academic_ctx.get("courses") or []
            subject = academic_ctx.get("subjectName") or ""

            if name:    profile_lines.append(f"Student name: {name}")
            if dept:    profile_lines.append(f"Department: {dept}")
            if batch:   profile_lines.append(f"Batch/Level: {batch}")
            if college: profile_lines.append(f"College: {college}")
            if gpa:     profile_lines.append(f"Current GPA: {gpa}")
            if subject: profile_lines.append(f"Current subject: {subject}")
            if isinstance(courses, list) and courses:
                course_names = [
                    (c.get("name") or c.get("subjectName") or str(c))
                    if isinstance(c, dict) else str(c)
                    for c in courses[:6]
                ]
                profile_lines.append(f"Enrolled courses: {', '.join(course_names)}")

            if profile_lines:
                base_prompt += (
                    "\n\n⚠\ufe0f VERIFIED STUDENT PROFILE — use this in your response:"
                    "\n" + "\n".join(profile_lines) + "\n"
                    "Use the student's name naturally. Reference their GPA and courses "
                    "when relevant. NEVER invent data not listed above."
                )

        # ── Build messages[] ──────────────────────────────────────────────
        messages: List[dict] = [{"role": "system", "content": base_prompt}]

        for turn in raw_history[-_MAX_HISTORY_TURNS:]:
            turn_role    = turn.get("role", "user")
            turn_content = _sanitise(turn.get("content", ""))
            if turn_role in ("user", "assistant") and turn_content:
                messages.append({"role": turn_role, "content": turn_content})

        messages.append({"role": "user", "content": _sanitise(input_context.message)})

        logger.info(
            "PlanExecutor._fallback_model_call: role=%r model=%r history=%d explain=%s",
            role, model_id, len(raw_history), explain,
        )

        # ── LLM call ─────────────────────────────────────────────────────
        response = await self.model_router.generate_with_messages(
            messages=messages, model_id=model_id
        )
        final_response = (
            response
            or "I'm sorry, I could not generate a response. Please try again."
        )

        # ── Explainability block ──────────────────────────────────────────
        if explain:
            final_response += (
                "\n\nℹ️ *I answered this using my general knowledge "
                "and your conversation history — no backend data was retrieved.*"
            )

        suggestions = self._get_suggestions(intent, role)

        return AgentOutput(
            status="success",
            response=final_response,
            data={
                "suggestions":       suggestions,
                "actions_available": suggestions,
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Follow-up suggestion helper
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_suggestions(intent: str, role: str) -> List[str]:
        """Return deterministic follow-up suggestions for the given intent+role."""
        intent_map = _SUGGESTIONS_MAP.get(intent)
        if not intent_map:
            intent_map = _SUGGESTIONS_MAP.get("general_chat", {})
        return intent_map.get(role) or intent_map.get("student", [])

    # ──────────────────────────────────────────────────────────────────────
    #  Payload interpolation helper
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _interpolate(
        payload: Dict[str, Any], results: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Replace {{step_X.output}} tokens in payload string values."""
        pattern = re.compile(r"\{\{step_(\d+)\.output\}\}")
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, str):
                match = pattern.search(v)
                if match:
                    step_id = int(match.group(1))
                    sub = results.get(step_id, v)
                    out[k] = (
                        sub
                        if v.strip() == match.group(0)
                        else v.replace(match.group(0), str(sub))
                    )
                else:
                    out[k] = v
            else:
                out[k] = v
        return out
