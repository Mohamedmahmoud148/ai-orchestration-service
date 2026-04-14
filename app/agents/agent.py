"""
app/agents/agent.py

The Agent — the single orchestration entry-point for the entire system.

Flow:
    User Request
        → Agent.run(context)
            → Planner.run()        — produce an ExecutionPlan + intent
            → ToolRegistry         — resolve intent → module name
            → Executor.execute()   — load & run the matching Module
        → Return enriched ExecutionContext

The Agent replaces the old OrchestrationPipeline while maintaining
backward compatibility with the ExecutionContext contract expected by
the existing chat route.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from app.agents.execution_context import ExecutionContext
from app.agents.schemas import AgentInput
from app.core.logging import logger
from app.services.memory_store import MemoryStore

if TYPE_CHECKING:
    from app.agents.planner import PlannerAgent
    from app.agents.executor import PlanExecutor
    from app.agents.model_router import ModelRouter
    from app.services.tool_registry import ToolRegistry


# ── RBAC Note ─────────────────────────────────────────────────────────────────
# Intent-level access control is enforced EXCLUSIVELY in:
#   app/core/rbac.py         → is_allowed() / log_blocked_attempt()
#   app/agents/executor.py   → Step 0 RBAC gate in execute()
#
# The old ROLE_PERMITTED_INTENTS dict has been REMOVED to eliminate the
# stale duplicate gate that was blocking all new intent modules before
# they could reach the executor.  A single source of truth prevents
# silent permission mismatches when new intents are added.

# Number of conversation turns that triggers background summarisation
_SUMMARY_THRESHOLD = 12


class Agent:
    """
    Main AI orchestrator.  Accepts an ExecutionContext, drives the pipeline,
    and writes results back onto the context for the endpoint layer to serialise.
    """

    def __init__(
        self,
        planner: "PlannerAgent",
        tool_registry: "ToolRegistry",
        model_router: "ModelRouter",
        executor: "PlanExecutor",
    ) -> None:
        self._planner = planner
        self._tool_registry = tool_registry
        self._model_router = model_router
        self._executor = executor
        self._memory_store = MemoryStore()

    # ──────────────────────────────────────────────────────────────────────
    #  Public entry-point
    # ──────────────────────────────────────────────────────────────────────

    async def run(self, context: ExecutionContext) -> ExecutionContext:
        """
        Execute the full agent pipeline for one user request.

        Writes results back to *context* so the caller (chat route) can
        serialise them without touching any agent internals.
        """
        logger.info(
            "[Agent] START conversation_id=%s user_id=%s role=%s message=%.80r",
            context.conversation_id,
            context.user_id,
            context.role,
            context.message,
        )
        pipeline_start = time.perf_counter()

        # ── Stage 0: Load Memory + User Preferences ─────────────────────
        memory = await self._memory_store.get_conversation(context.user_id)
        prefs  = await self._memory_store.get_preferences(context.user_id)
        if memory:
            context.add_metadata("memory", memory)
        if prefs:
            context.add_metadata("preferences", prefs)
            logger.info(
                "[Agent] Loaded preferences for user_id=%s keys=%s",
                context.user_id, list(prefs.keys()),
            )

        # ── Stage 0.5: Clarification Disambiguation ───────────────────────
        clarification = await self._memory_store.get_clarification(context.user_id)
        plan = None
        module_name = None

        if clarification:
            msg = context.message.strip().lower()
            options = clarification.get("options", [])
            selected_option = None

            if msg.isdigit() and 1 <= int(msg) <= len(options):
                selected_option = options[int(msg) - 1]
            else:
                for opt in options:
                    name = str(opt.get("name") or opt.get("title") or opt.get("subjectName") or opt.get("id") or "")
                    if name and msg in name.lower():
                        selected_option = opt
                        break

            if selected_option:
                logger.info("[Agent] Clarification resolved user_id=%s", context.user_id)
                await self._memory_store.delete_clarification(context.user_id)

                step_context = clarification.get("step_context", {})
                exam_params = step_context.get("exam_params", {})
                offering_id = selected_option.get("subjectOfferingId") or selected_option.get("id")

                context.academic_context["subjectOfferingId"] = offering_id
                if exam_params:
                    exam_params["subjectOfferingId"] = offering_id

                from app.agents.schemas import ExecutionPlan, ExamParams
                intent = clarification.get("original_intent", "unknown")
                plan = ExecutionPlan(
                    goal_summary="Clarification resolved execution.",
                    intent=intent,
                    is_executable=True
                )
                if intent == "generate_exam" and exam_params:
                    plan.exam_params = ExamParams(**exam_params)

                context.set_intent(intent)
                module_name = step_context.get("module_name", "model_only")
                context.set_tool(module_name)
                self._route_model(context)
            else:
                logger.warning("[Agent] Invalid clarification choice user_id=%s msg=%r", context.user_id, msg)
                context.add_metadata("clarification_needed", True)
                context.add_metadata("clarification_options", options)
                context.set_result("عذراً، هذا الاختيار غير صحيح.")
                return context

        if not plan:
            # ── Stage 1: Planning ─────────────────────────────────────────────
            plan = await self._plan(context)

            # ── Stage 2: Model Routing ────────────────────────────────────────
            # NOTE: RBAC validation is enforced downstream in executor.execute()
            # (Step 0) via app/core/rbac.py — NOT here.  This removes the stale
            # duplicate gate that blocked new intents before reaching modules.
            self._route_model(context)

            # ── Stage 3: Tool / Module Selection ─────────────────────────────
            module_name = self._select_module(context)

        # ── Stage 4: Execution ────────────────────────────────────────────
        await self._execute(context, plan, module_name)

        # ── Clarification Handling or Save Memory ────────────────────────
        if context.metadata.get("clarification_needed"):
            options = context.metadata.get("clarification_options", [])
            data = {
                "options": options,
                "original_intent": context.intent,
                "step_context": {
                    "module_name": module_name,
                    "exam_params": getattr(plan, "exam_params", None).model_dump(exclude_none=True) if getattr(plan, "exam_params", None) else {}
                }
            }
            await self._memory_store.save_clarification(context.user_id, data)
        else:
            # ── Stage 5: Save Memory + async summarisation ───────────────────
            entities = {}
            if plan and getattr(plan, "exam_params", None):
                entities = plan.exam_params.model_dump(exclude_none=True)

            memory_data = {
                "last_intent": context.intent,
                "last_result": context.result,
                "entities":    entities,
            }
            await self._memory_store.save_conversation(context.user_id, memory_data)

            # Fire-and-forget: compress long conversations in the background
            if len(context.history) >= _SUMMARY_THRESHOLD:
                asyncio.create_task(self._summarize_and_save(context))

        elapsed = round(time.perf_counter() - pipeline_start, 4)
        context.add_metadata("agent_duration_seconds", elapsed)

        logger.info(
            "[Agent] END conversation_id=%s status=%s duration_s=%s",
            context.conversation_id,
            "success" if context.result else "empty",
            elapsed,
        )
        return context

    # _validate_role_permissions() REMOVED (2026-04-12)
    # ─────────────────────────────────────────────────────────────────────
    # RBAC is now enforced exclusively at the executor level:
    #   PlanExecutor.execute() → Step 0 → app/core/rbac.is_allowed()
    # The stale ROLE_PERMITTED_INTENTS dict listed only 4 old intents and
    # silently blocked all new modules (material_explanation, complaint,
    # cv_analysis, academic_advice) before they could run.
    # ─────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────
    #  Internal stages
    # ──────────────────────────────────────────────────────────────────────

    async def _plan(self, context: ExecutionContext):
        """Stage 1 — call Planner; populate context.intent and store the plan."""
        logger.info("[Agent] stage=planner conversation_id=%s", context.conversation_id)
        t0 = time.perf_counter()

        agent_input = AgentInput(
            message=context.message,
            user_id=context.user_id,
            auth_header=context.metadata.get("auth_header"),
            context={
                "role": context.role,
                "history": context.history,
                "academic_context": context.academic_context,
            },
        )

        from app.agents.pipeline import _PipelineStageError   # keep same error sentinel

        planner_output = await self._planner.run(agent_input)

        elapsed = round(time.perf_counter() - t0, 4)
        context.add_metadata("planner_duration_seconds", elapsed)

        if planner_output.status != "success" or not planner_output.data:
            context.set_result(planner_output.response)
            raise _PipelineStageError("planner", planner_output.response)

        plan = planner_output.data.get("plan")
        context.set_intent(plan.intent or plan.goal_summary if plan else "unknown")
        context.add_metadata("plan", plan)

        logger.info(
            "[Agent] stage=planner intent=%r duration_s=%s",
            context.intent,
            elapsed,
        )
        return plan

    def _route_model(self, context: ExecutionContext) -> None:
        """
        Stage 2 — intent + role aware model selection.

        Routing table:
          summarization                                → HuggingFace BART (local, free)
          generate_exam        + doctor/admin          → gpt-4o  (quality-critical)
          material_explanation + doctor                → gpt-4o  (faculty-grade summary)
          file_processing      + admin                 → gpt-4o  (bulk operations)
          admin role           (any intent)            → gpt-4o
          everything else                              → gpt-4o-mini
        """
        intent = context.intent or "general_chat"
        role   = context.role   or "student"

        if intent == "summarization":
            # Route to local HuggingFace BART — free, no API cost
            model = "hf/facebook/bart-large-cnn"
        elif intent == "generate_exam" and role in ("doctor", "admin"):
            model = "openai/gpt-4o"
        elif intent == "material_explanation" and role == "doctor":
            # Doctors need high-quality summaries for academic use
            model = "openai/gpt-4o"
        elif intent == "file_processing" and role == "admin":
            # Bulk data operations benefit from stronger reasoning
            model = "openai/gpt-4o"
        elif role == "admin":
            model = "openai/gpt-4o"
        else:
            model = "openai/gpt-4o-mini"

        context.set_model(model)
        logger.info(
            "[Agent] stage=model_routing intent=%r role=%r selected_model=%r",
            intent, role, model,
        )

    def _select_module(self, context: ExecutionContext) -> str:
        """Stage 3 — resolve intent → module name via ToolRegistry."""
        intent = context.intent or ""
        module_name = self._tool_registry.get_module_for_intent(intent) or "model_only"
        context.set_tool(module_name)
        logger.info("[Agent] stage=tool_selection module=%r", module_name)
        return module_name

    async def _execute(
        self, context: ExecutionContext, plan, module_name: str
    ) -> None:
        """Stage 4 — delegate execution to PlanExecutor (module dispatcher)."""
        from app.agents.pipeline import _PipelineStageError
        from app.agents.schemas import AgentInput

        logger.info(
            "[Agent] stage=executor module=%r model=%r",
            module_name,
            context.selected_model,
        )
        t0 = time.perf_counter()

        agent_input = AgentInput(
            message=context.message,
            user_id=context.user_id,
            auth_header=context.metadata.get("auth_header"),
            context={
                "role":             context.role,
                "selected_tool":    context.selected_tool,
                "selected_model":   context.selected_model,
                "explain":          context.metadata.get("explain", False),
                "preferences":      context.metadata.get("preferences", {}),
                "academic_context": context.academic_context,
                "history":          context.history,
                # Forward auth header into context so DynamicApiModule can use it
                "auth_header":      context.metadata.get("auth_header"),
            },
        )

        try:
            executor_output = await self._executor.execute(
                plan=plan,
                input_context=agent_input,
                module_name=module_name,
            )
        except Exception as exc:
            logger.error("[Agent] stage=executor error=%s", exc, exc_info=True)
            raise _PipelineStageError("executor", str(exc)) from exc

        elapsed = round(time.perf_counter() - t0, 4)
        context.add_metadata("executor_duration_seconds", elapsed)
        context.add_metadata("executor_status", executor_output.status)
        # Store executor data (suggestions, actions_available, raw results)
        context.add_metadata("executor_data", executor_output.data or {})

        if executor_output.status in ("failed", "partial_failure", "forbidden"):
            context.set_result(executor_output.response)
            raise _PipelineStageError("executor", executor_output.response)

        if executor_output.status == "clarification_needed":
            context.add_metadata("clarification_needed", True)
            context.add_metadata("clarification_options", (executor_output.data or {}).get("options", []))
            context.set_result(executor_output.response)
            return

        context.set_result(executor_output.response)

    # ──────────────────────────────────────────────────────────────────────
    #  Background summarisation (fire-and-forget)
    # ──────────────────────────────────────────────────────────────────────

    async def _summarize_and_save(self, context: ExecutionContext) -> None:
        """
        Compress long conversation history into a concise summary and
        persist it in Redis with a 24-hour TTL.

        Called as an asyncio background task — never blocks the response.
        Uses the local HuggingFace BART model (free, no OpenAI cost).
        """
        try:
            history_text = "\n".join(
                f"{t.get('role', 'user')}: {t.get('content', '')}"
                for t in context.history[-20:]
            )
            if not history_text.strip():
                return

            summary = await self._model_router.summarize(
                text=history_text,
                model_id="hf/facebook/bart-large-cnn",
            )

            if summary:
                await self._memory_store.save_summary(context.user_id, summary)
                logger.info(
                    "[Agent] Background summarisation complete: "
                    "user_id=%s turns=%d summary_chars=%d",
                    context.user_id, len(context.history), len(summary),
                )
        except Exception as exc:
            # Never propagate — this is best-effort background work
            logger.warning(
                "[Agent] Background summarisation failed for user_id=%s: %s",
                context.user_id, exc,
            )
