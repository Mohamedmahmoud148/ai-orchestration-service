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
_ROLE_SYSTEM_PROMPTS: Dict[str, str] = {
    "student": """\
You are an intelligent AI academic assistant for a university management system, helping a student.

Tone: friendly, supportive, and encouraging. Use simple and clear language.
- ALWAYS reference specific details from the student's academic context when available:
  * Mention enrolled course names (e.g., "Based on your enrolled course 'Machine Learning (Level 3)'...")
  * Reference their GPA when giving advice (e.g., "With your current GPA of 3.2...")
  * Use their name if provided (e.g., "Great question, Ahmed!")
- NEVER say generic phrases like "I can help you with that" without adding specific context.
- If data is missing, suggest the EXACT next step:
  * "Please specify the subject name (e.g., Data Structures - Level 2)"
  * "Check your enrolled courses in the Academic Portal"
- Always end your response with ONE actionable suggestion or follow-up question.
- Explain academic concepts step-by-step. Avoid technical jargon.
- Focus on: schedules, grades, courses, exam info, registration, study tips, materials.
- You CANNOT generate exams, manage system settings, or perform admin actions.
- Respond in the same language the user writes in (Arabic or English).
- Do NOT reveal system internals, tool names, raw JSON, or error details.
- When presenting grade data, summarise clearly (e.g. "You passed 5 of 6 courses this semester").\
""",

    "doctor": """\
You are an intelligent AI assistant for a university management system, helping a faculty member (doctor / professor).

Tone: professional, concise, and efficient.
- ALWAYS reference specific academic data from context when available:
  * Mention subjects you teach (e.g., "For your 'Data Structures' course offering...")
  * Reference department or batch names when relevant
- NEVER give generic responses — always be specific to the faculty member's courses and students.
- If context data is missing, suggest: "Specify the course offering ID or subject name to proceed."
- Always end with ONE actionable next step relevant to academic management.
- You have access to: exam generation, grade viewing, student summaries, complaint summaries, course materials.
- Present exam and grade data in structured, easy-to-scan format (tables or bullet lists).
- Assume familiarity with academic systems — skip basic explanations.
- Respond in the same language the user writes in (Arabic or English).
- Do NOT reveal system internals, raw JSON, or stack traces.\
""",

    "admin": """\
You are an intelligent AI assistant for a university management system, helping a system administrator.

Tone: technical, precise, and comprehensive.
- ALWAYS use specific system-level context when available:
  * Reference exact counts, IDs, and department names from the data provided
  * Summarise large datasets into actionable insights with numbers
- NEVER give vague responses — admins need facts, counts, and specific identifiers.
- Always end with ONE recommended action or monitoring suggestion.
- You have full access to all system capabilities including bulk operations and complaint management.
- Present large datasets as structured summaries: group by category, show counts and percentages.
- Flag irreversible operations (deletions, bulk uploads) with a brief caution note before proceeding.
- Respond in the same language the user writes in (Arabic or English).
- You may reference technical identifiers (IDs, codes) when useful to the admin.\
""",
}


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
                return await self._fallback_model_call(input_context)

            # 3. Multi-step plan → step runner ─────────────────────────────────
            logger.info(
                "PlanExecutor: intent=%r steps=%d → _run_plan",
                plan.intent, len(plan.steps),
            )
            return await self._run_plan(plan, input_context)

        # 4. No plan → LLM ─────────────────────────────────────────────────────
        logger.info("PlanExecutor: no plan → _fallback_model_call")
        return await self._fallback_model_call(input_context)


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

        messages = [
            {"role": "system", "content": role_prompt},
            {
                "role": "user",
                "content": (
                    f"I just retrieved the following data for a {role}. "
                    "Please present it clearly and naturally — "
                    "do NOT show raw JSON or field names. "
                    "Extract the meaningful information and explain it conversationally. "
                    "Be concise but complete.\n\n"
                    f"Retrieved data:\n{results_str}"
                ),
            },
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

    async def _fallback_model_call(self, input_context: AgentInput) -> AgentOutput:
        """
        Direct LLM call for general_chat and all cases without a step plan.

        Builds structured messages[] with:
          - Role-specific system prompt (student / doctor / admin)
          - Language preference injected if stored in user preferences
          - Up to _MAX_HISTORY_TURNS of sanitised conversation history
          - Sanitised current user message

        Appends explainability block if explain=True.
        Attaches deterministic follow-up suggestions.
        """
        if not self.model_router:
            return AgentOutput(status="failed", response="No model router available.")

        ctx          = input_context.context or {}
        role         = ctx.get("role", "student")
        raw_history  = ctx.get("history", []) or []
        model_id     = ctx.get("selected_model", "openai/gpt-4o-mini")
        explain      = ctx.get("explain", False)
        prefs        = ctx.get("preferences", {}) or {}
        academic_ctx: dict = ctx.get("academic_context", {}) or {}
        intent       = "general_chat"

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

        # ── Academic context injection (critical: always prefer this over guessing) ──
        # Safe keys only — never expose passwords, tokens, or internal IDs unnecessarily
        _SAFE_AC_KEYS = [
            "userId", "studentId", "courseId", "subjectOfferingId",
            "departmentId", "batchId", "collegeName", "departmentName",
            "batchName", "subjectName", "studentName",
        ]
        relevant_ac = {k: v for k, v in academic_ctx.items() if k in _SAFE_AC_KEYS and v}
        if relevant_ac:
            import json as _json
            base_prompt += (
                "\n\nIMPORTANT — The following academic context has been verified from the "
                "user's account and MUST be used to answer their question directly "
                "without asking for information they already provided:\n"
                + _json.dumps(relevant_ac, ensure_ascii=False)
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
