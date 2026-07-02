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
import re
import time
from typing import TYPE_CHECKING, Optional

from app.agents.execution_context import ExecutionContext
from app.agents.pipeline import _PipelineStageError
from app.agents.schemas import AgentInput, ExamParams, ExecutionPlan
from app.core.logging import logger
from app.services.memory_store import MemoryStore, get_memory_store

if TYPE_CHECKING:
    from app.agents.planner import PlannerAgent
    from app.agents.executor import PlanExecutor
    from app.agents.model_router import ModelRouter
    from app.agents.react_agent import ReactAgent
    from app.services.tool_registry import ToolRegistry


# Compile the file-URL detection regex once at import time instead of on
# every response. Bounded character class + bounded path segment length
# protects against catastrophic backtracking on hostile inputs (M2).
_FILE_URL_PATTERN = re.compile(
    r"https?://[A-Za-z0-9._~:/?#@!$&'()*+,;=%\-]{1,1024}?"
    r"\.(?:pdf|xlsx|xls|docx|csv|png|jpg)",
    re.IGNORECASE,
)


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
        react_agent: Optional["ReactAgent"] = None,
    ) -> None:
        self._planner = planner
        self._tool_registry = tool_registry
        self._model_router = model_router
        self._executor = executor
        self._react_agent = react_agent          # None → falls back to old pipeline
        # Use the shared singleton — main.py already ran ping() against it.
        # If main.py hasn't constructed it yet (e.g. in a test), get_memory_store()
        # builds it lazily and the next ping() (or no-op fallback) handles it.
        self._memory_store = get_memory_store()

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

        memory_key, plan, module_name, should_return_early = await self._load_memory(context)
        if should_return_early:
            # An in-progress clarification was answered with an invalid
            # choice. context.result is already set (Arabic error message);
            # matches the original inline `return context` exactly — no
            # background tasks fire, no Stage 5 save happens.
            return context

        await self._execute_core(context, plan, module_name)

        await self._save_memory(context, memory_key, plan, module_name)

        elapsed = round(time.perf_counter() - pipeline_start, 4)
        context.add_metadata("agent_duration_seconds", elapsed)

        logger.info(
            "[Agent] END conversation_id=%s status=%s duration_s=%s",
            context.conversation_id,
            "success" if context.result else "empty",
            elapsed,
        )
        return context

    # ──────────────────────────────────────────────────────────────────────
    #  Stage 0 / Stage 1-4 / Stage 5 — extracted from run() (Phases 2-3 of
    #  the agentic architecture upgrade, see docs/AGENTIC_UPGRADE_ROADMAP.md).
    #
    #  Pure extract-method refactor — no logic changed from the original
    #  inline blocks. Exists as real, independently callable methods so the
    #  LangGraph multi-node graph (app/agents/graph.py) can call each one
    #  as a separate node instead of running the whole run() method.
    # ──────────────────────────────────────────────────────────────────────

    async def _load_memory(
        self, context: ExecutionContext
    ) -> tuple[str, Optional[ExecutionPlan], Optional[str], bool]:
        """
        Stage 0 — load memory, preferences, entities, user goal, academic
        profile, personalized context, and last-seen file URL; resolve any
        in-progress clarification; fire background language-detection and
        entity-extraction tasks.

        Returns (memory_key, plan, module_name, should_return_early).

        should_return_early is True only when an in-progress clarification
        was answered with an invalid choice — context.result is already set
        to the Arabic "invalid choice" message in that case, and the caller
        must return context immediately without firing the background tasks
        below or running Stage 5, exactly matching the original inline
        `return context` early-exit.
        """
        # Use composite key so different conversations don't share memory.
        memory_key = f"{context.user_id}:{context.conversation_id}" if context.conversation_id else context.user_id
        memory = await self._memory_store.get_conversation(memory_key)
        prefs  = await self._memory_store.get_preferences(context.user_id)
        if memory:
            context.add_metadata("memory", memory)
        if prefs:
            context.add_metadata("preferences", prefs)
            logger.info(
                "[Agent] Loaded preferences for user_id=%s keys=%s",
                context.user_id, list(prefs.keys()),
            )

        # ── Stage 0.1b: Load conversation entities + user goal ──────────
        entities = await self._memory_store.get_entities(context.user_id)
        if entities:
            context.add_metadata("conversation_entities", entities)
            context.academic_context["conversation_entities"] = entities
        user_goal = await self._memory_store.get_user_goal(context.user_id)
        if user_goal:
            context.add_metadata("user_goal", user_goal)
            context.academic_context["user_goal"] = user_goal

        # ── Stage 0.2: Load Academic Profile (Phase 5) ──────────────────
        academic_profile = await self._memory_store.get_academic_profile(context.user_id)
        if academic_profile:
            context.add_metadata("academic_profile", academic_profile)

        personalized_context = await self._memory_store.get_personalized_context(context.user_id)
        if personalized_context:
            context.add_metadata("personalized_context", personalized_context)
            # Also inject into academic_context so modules can use it directly
            context.academic_context["personalized_context"] = personalized_context
            logger.info(
                "[Agent] Loaded personalized context for user_id=%s context=%.80r",
                context.user_id, personalized_context,
            )

        # ── Stage 0.1: Restore last seen file URL into academic_context ──
        file_ctx = await self._memory_store.get_file_context(context.user_id)
        if file_ctx and file_ctx.get("file_url"):
            # Inject into academic_context so FileExtractionModule can find it
            if not context.academic_context.get("file_url"):
                context.academic_context["file_url"]   = file_ctx["file_url"]
                context.academic_context["file_name"]  = file_ctx.get("file_name", "")
                logger.info(
                    "[Agent] Restored last file URL for user_id=%s url=%s",
                    context.user_id, file_ctx["file_url"][:60],
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
                return memory_key, None, None, True

        # Detect + persist user language early (fire-and-forget, never blocks)
        asyncio.create_task(
            self._memory_store.detect_and_save_language(context.user_id, context.message)
        )

        # Extract + persist conversation entities (courses, doctors, goals…)
        asyncio.create_task(self._extract_and_save_entities(context))

        return memory_key, plan, module_name, False

    async def _execute_core(
        self,
        context: ExecutionContext,
        plan: Optional[ExecutionPlan],
        module_name: Optional[str],
    ) -> None:
        """
        Stage 1-4 (combined) — run either the ReactAgent smart path (when
        clarification resolution did not produce a plan — the normal case)
        or the legacy PlanExecutor path (when clarification resolution did
        produce a plan). Writes context.result / context.metadata in place.

        Note: this never needs to return an updated plan/module_name —
        the ReactAgent branch's `plan = None` reassignment below is a
        no-op (plan is already None on entry to that branch), so whatever
        `plan` the caller passed in is exactly what should be passed to
        _save_memory() afterwards, unchanged.
        """
        if not plan:
            if self._react_agent is not None:
                # ── ReactAgent path (smart, iterative, no keywords) ───────
                logger.info(
                    "[Agent] stage=react user_id=%s conversation_id=%s",
                    context.user_id, context.conversation_id,
                )
                t0 = time.perf_counter()
                try:
                    result = await self._react_agent.run(context)
                    context.set_result(result)
                except Exception as exc:
                    logger.error("[Agent] ReactAgent failed — %s", exc, exc_info=True)
                    # fall through to old pipeline below
                    context.set_result(None)

                if context.result:
                    elapsed = round(time.perf_counter() - t0, 4)
                    context.add_metadata("react_duration_seconds", elapsed)
                    logger.info(
                        "[Agent] stage=react done duration_s=%s", elapsed
                    )
                    # skip old pipeline stages 1-4
                    plan = None
                else:
                    # ReactAgent returned no result — respond gracefully
                    logger.error("[Agent] ReactAgent produced no result for user_id=%s", context.user_id)
                    context.set_result(
                        "عذراً، تعذّر معالجة طلبك في الوقت الحالي. حاول مرة أخرى."
                        "\n(Request could not be processed — please try again.)"
                    )
        else:
            # Clarification path — always uses old executor
            await self._execute(context, plan, module_name)

    async def _save_memory(
        self,
        context: ExecutionContext,
        memory_key: str,
        plan: Optional[ExecutionPlan],
        module_name: Optional[str],
    ) -> None:
        """
        Stage 5 — save a fresh clarification if the executor determined one
        is needed this turn, otherwise persist conversation memory, extract
        and persist any file URL found in the response, and fire background
        summarization for long conversations.
        """
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

            # Build richer memory: include intent, result summary, role, and message topic
            result_text = str(context.result or "")
            memory_data = {
                "last_intent":  context.intent or "general_chat",
                "last_result":  result_text[:400] if result_text else "",
                "last_message": context.message[:120],
                "role":         context.role or "student",
                "entities":     entities,
            }
            await self._memory_store.save_conversation(memory_key, memory_data)

            # ── Extract and persist any file URL found in the response ───
            # Bounded regex compiled at module load (see _FILE_URL_PATTERN).
            # Cap the search input length defensively so a 10MB response can't
            # turn into a regex-engine DoS even though the pattern is bounded.
            result_text = str(context.result or "")[:200_000]
            urls_found = _FILE_URL_PATTERN.findall(result_text)
            if urls_found:
                best_url = urls_found[0]
                # extract filename from URL
                fname = best_url.split("?")[0].split("/")[-1]
                await self._memory_store.save_file_context(context.user_id, best_url, fname)
                logger.info("[Agent] Saved file URL to memory for user_id=%s url=%s", context.user_id, best_url[:60])

                # Also persist as ActiveDocumentContext so ReactAgent can inject it
                # into the system prompt for follow-up requests ("لخصه", "اقراه", etc.)
                active_doc = {
                    "document_type": "material",
                    "file_url": best_url,
                    "title": fname,
                    "material_id": "",
                    "subject_name": "",
                }
                # Try to enrich from academic_context (subject name may be known)
                acad = context.academic_context or {}
                if acad.get("subjectName"):
                    active_doc["subject_name"] = acad["subjectName"]
                elif acad.get("subject_name"):
                    active_doc["subject_name"] = acad["subject_name"]
                await self._memory_store.set_active_document(context.user_id, active_doc)
                logger.info(
                    "[Agent] Saved active_doc (material) for user_id=%s title=%s",
                    context.user_id, fname,
                )

            # Fire-and-forget: compress long conversations in the background
            if len(context.history) >= _SUMMARY_THRESHOLD:
                asyncio.create_task(self._summarize_and_save(context))

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
            model = "hf/facebook/bart-large-cnn"
        elif intent == "generate_exam" and role in ("doctor", "admin", "superadmin"):
            model = "openai/gpt-4o"
        elif intent in ("material_explanation", "material_qa", "file_extraction"):
            model = "openai/gpt-4o"
        elif intent == "file_processing" and role in ("admin", "superadmin"):
            model = "openai/gpt-4o"
        else:
            # gpt-4o-mini for everything else including admin general_chat —
            # fast, cheap, good enough. gpt-4o reserved for quality-critical tasks above.
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

    async def _extract_and_save_entities(self, context: ExecutionContext) -> None:
        """
        Extract academic entities from the user message and save them for
        cross-turn coreference. Background task — never blocks the response.
        """
        try:
            from app.core.entity_tracker import extract_entities
            entities = extract_entities(context.message)

            # Also extract from history last turn to catch pronouns ("ها"/"it")
            if context.history:
                last_user = next(
                    (t.get("content", "") for t in reversed(context.history)
                     if t.get("role") == "user"),
                    "",
                )
                if last_user:
                    prev_entities = extract_entities(str(last_user))
                    from app.core.entity_tracker import merge_entities
                    entities = merge_entities(entities, prev_entities)

            await self._memory_store.save_entities(context.user_id, entities)

            # Persist primary user goal if detected
            goals = entities.get("goals", [])
            if goals:
                await self._memory_store.save_user_goal(context.user_id, goals[0])
        except Exception as exc:
            logger.warning("[Agent] _extract_and_save_entities failed: %s", exc)

    async def _summarize_and_save(self, context: ExecutionContext) -> None:
        """
        Compress long conversation history into a concise summary and
        persist it in Redis with a 24-hour TTL.

        Called as an asyncio background task — never blocks the response.
        Uses gpt-4o-mini for Arabic-capable summarization.
        """
        try:
            history_text = "\n".join(
                f"{t.get('role', 'user')}: {t.get('content', '')}"
                for t in context.history[-20:]
            )
            if not history_text.strip():
                return

            # Use gpt-4o-mini via model_router for Arabic-capable summarization
            summary_prompt = (
                "Summarize this conversation in 2-3 sentences, preserving key topics, "
                "entities (names, course IDs, departments), and unresolved requests. "
                "Reply in the same language as the conversation.\n\n"
                f"{history_text[:3000]}"
            )
            try:
                summary = await self._model_router.generate(
                    prompt=summary_prompt,
                    model_id="openai/gpt-4o-mini",
                )
            except Exception:
                # Fallback: BART if gpt-4o-mini call fails
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
