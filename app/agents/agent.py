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

import time
from typing import TYPE_CHECKING

from app.agents.execution_context import ExecutionContext
from app.agents.schemas import AgentInput
from app.core.logging import logger

if TYPE_CHECKING:
    from app.agents.planner import PlannerAgent
    from app.agents.executor import PlanExecutor
    from app.agents.model_router import ModelRouter
    from app.services.tool_registry import ToolRegistry


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

        # ── Stage 1: Planning ─────────────────────────────────────────────
        plan = await self._plan(context)

        # ── Stage 2: Model Routing ────────────────────────────────────────
        self._route_model(context)

        # ── Stage 3: Tool / Module Selection ─────────────────────────────
        module_name = self._select_module(context)

        # ── Stage 4: Execution ────────────────────────────────────────────
        await self._execute(context, plan, module_name)

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
        """Stage 2 — pick the target LLM based on role."""
        role_map = {
            "admin": "gemini-2.5-pro",
            "doctor": "gemini-2.5-flash",
            "student": "gemini-2.5-flash",
        }
        model = role_map.get(context.role, "gemini-2.5-flash")
        context.set_model(model)
        logger.info("[Agent] stage=model_routing selected_model=%r", model)

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
                "role": context.role,
                "selected_tool": context.selected_tool,
                "selected_model": context.selected_model,
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

        if executor_output.status in ("failed", "partial_failure"):
            context.set_result(executor_output.response)
            raise _PipelineStageError("executor", executor_output.response)

        context.set_result(executor_output.response)
