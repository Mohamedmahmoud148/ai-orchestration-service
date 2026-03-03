"""
pipeline.py

OrchestrationPipeline — the single, authoritative entry-point for all AI
orchestration work.  No stage knows about any other stage; each speaks
only to the shared ExecutionContext.

Stage order (enforced, non-negotiable):
    1. Planner          → resolves intent & creates an ExecutionPlan
    2. Tool Selection   → maps intent → concrete tool via ToolRegistry
    3. Model Routing    → selects the best model for this intent/role
    4. Executor         → runs the plan and stores the final result
"""

from __future__ import annotations

import time
import logging
from typing import Any, Protocol, runtime_checkable

from app.agents.execution_context import ExecutionContext
from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


# ────────────────────────────────────────────────────────────────────────────
#  Structural Protocols — depend on behaviour, not concrete classes
# ────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Planner(Protocol):
    """Any object that can produce an AgentOutput (containing a plan) from AgentInput."""

    async def run(self, agent_input: AgentInput) -> AgentOutput: ...


@runtime_checkable
class ToolRegistry(Protocol):
    """Any object that maps an intent name to a backend route string."""

    def get_route_for_intent(self, intent_name: str) -> str | None: ...


@runtime_checkable
class ModelRouter(Protocol):
    """Any object that can route text-generation requests to a concrete LLM."""

    async def generate(
        self,
        prompt: str,
        system_instruction: str,
        model_id: str,
    ) -> str | None: ...


@runtime_checkable
class Executor(Protocol):
    """Any object that can execute an ExecutionPlan from an AgentInput."""

    async def execute(
        self,
        plan: Any,
        input_context: AgentInput,
    ) -> AgentOutput: ...


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _stage_log(stage: str, context: ExecutionContext, extra: str = "") -> None:
    """Emit a structured INFO log entry for a pipeline stage."""
    logger.info(
        "[Pipeline] stage=%s conversation_id=%s user_id=%s %s",
        stage,
        context.conversation_id,
        context.user_id,
        extra,
    )


def _stage_error(stage: str, context: ExecutionContext, error: Exception) -> None:
    """Emit a structured ERROR log entry for a failed pipeline stage."""
    logger.error(
        "[Pipeline] stage=%s conversation_id=%s user_id=%s error=%s",
        stage,
        context.conversation_id,
        context.user_id,
        str(error),
        exc_info=True,
    )


# ────────────────────────────────────────────────────────────────────────────
#  OrchestrationPipeline
# ────────────────────────────────────────────────────────────────────────────

class OrchestrationPipeline:
    """
    Unified orchestration engine.

    Responsibilities
    ----------------
    - Accept an :class:`ExecutionContext` from the endpoint layer.
    - Drive each stage in strict, documented order.
    - Persist per-stage timing in ``context.metadata``.
    - Return the enriched context — the endpoint layer only serialises it.

    Non-responsibilities (clean separation)
    ----------------------------------------
    - No HTTP, no database, no model SDK calls.
    - No business-logic branching (that belongs in the stages themselves).
    - No if-else routing scattered across the codebase.
    """

    def __init__(
        self,
        planner: Planner,
        tool_registry: ToolRegistry,
        model_router: ModelRouter,
        executor: Executor,
    ) -> None:
        self._planner = planner
        self._tool_registry = tool_registry
        self._model_router = model_router
        self._executor = executor

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(self, context: ExecutionContext) -> ExecutionContext:
        """
        Execute the full pipeline for a single orchestration request.

        Parameters
        ----------
        context:
            A freshly-built :class:`ExecutionContext` from the endpoint.

        Returns
        -------
        ExecutionContext
            The same object, enriched by each stage.  The endpoint layer
            reads ``context.result`` (and optionally ``context.metadata``)
            to build its HTTP response.
        """
        logger.info(
            "[Pipeline] START conversation_id=%s user_id=%s role=%s message=%.80r",
            context.conversation_id,
            context.user_id,
            context.role,
            context.message,
        )

        pipeline_start = time.perf_counter()

        await self._run_planner(context)
        await self._run_tool_selection(context)
        await self._run_model_routing(context)
        await self._run_executor(context)

        elapsed = round(time.perf_counter() - pipeline_start, 4)
        context.add_metadata("pipeline_duration_seconds", elapsed)

        logger.info(
            "[Pipeline] END conversation_id=%s status=%s duration_s=%s",
            context.conversation_id,
            "success" if context.result else "empty",
            elapsed,
        )

        return context

    # ------------------------------------------------------------------ #
    #  Stage 1 — Planner                                                    #
    # ------------------------------------------------------------------ #

    async def _run_planner(self, context: ExecutionContext) -> None:
        """
        Call the Planner with the user message + history.

        Writes
        ------
        context.intent          → high-level intent label
        context.metadata["plan"] → raw ExecutionPlan object for the executor
        """
        _stage_log("planner", context, f"message_len={len(context.message)}")
        t0 = time.perf_counter()

        agent_input = AgentInput(
            message=context.message,
            user_id=context.user_id,
            context={
                "role": context.role,
                "history": context.history,
                "academic_context": context.academic_context,
            },
        )

        planner_output: AgentOutput = await self._planner.run(agent_input)

        elapsed = round(time.perf_counter() - t0, 4)
        context.add_metadata("planner_duration_seconds", elapsed)

        if planner_output.status != "success" or not planner_output.data:
            logger.warning(
                "[Pipeline] stage=planner conversation_id=%s planner_failed=%r",
                context.conversation_id,
                planner_output.response,
            )
            context.set_result(planner_output.response)
            raise _PipelineStageError("planner", planner_output.response)

        plan = planner_output.data.get("plan")
        context.set_intent(plan.goal_summary if plan else "unknown")
        context.add_metadata("plan", plan)

        _stage_log("planner", context, f"intent={context.intent!r} duration_s={elapsed}")

    # ------------------------------------------------------------------ #
    #  Stage 2 — Tool Selection                                             #
    # ------------------------------------------------------------------ #

    async def _run_tool_selection(self, context: ExecutionContext) -> None:
        """
        Resolve the intent to a concrete backend tool route via ToolRegistry.

        Writes
        ------
        context.selected_tool → route string or None (model-only flows)
        """
        _stage_log("tool_selection", context, f"intent={context.intent!r}")
        t0 = time.perf_counter()

        route = self._tool_registry.get_route_for_intent(context.intent or "")
        context.set_tool(route or "model_only")

        elapsed = round(time.perf_counter() - t0, 4)
        context.add_metadata("tool_selection_duration_seconds", elapsed)

        _stage_log(
            "tool_selection",
            context,
            f"selected_tool={context.selected_tool!r} duration_s={elapsed}",
        )

    # ------------------------------------------------------------------ #
    #  Stage 3 — Model Routing                                             #
    # ------------------------------------------------------------------ #

    async def _run_model_routing(self, context: ExecutionContext) -> None:
        """
        Choose the best model for this request based on role, intent, and
        any runtime quality/cost signals.

        Writes
        ------
        context.selected_model → model identifier string
        """
        _stage_log("model_routing", context, f"role={context.role!r}")
        t0 = time.perf_counter()

        model = self._resolve_model(context)
        context.set_model(model)

        elapsed = round(time.perf_counter() - t0, 4)
        context.add_metadata("model_routing_duration_seconds", elapsed)

        _stage_log(
            "model_routing",
            context,
            f"selected_model={context.selected_model!r} duration_s={elapsed}",
        )

    def _resolve_model(self, context: ExecutionContext) -> str:
        """
        Encapsulates model-selection logic in one deterministic place.

        Extend this method — or replace it with an ML-based ranker — as
        your model catalogue grows.  The pipeline itself never changes.
        """
        role_model_map: dict[str, str] = {
            "admin": "gemini-2.5-pro",
            "doctor": "gemini-2.5-flash",
            "student": "gemini-2.5-flash",
        }
        return role_model_map.get(context.role, "gemini-2.5-flash")

    # ------------------------------------------------------------------ #
    #  Stage 4 — Executor                                                   #
    # ------------------------------------------------------------------ #

    async def _run_executor(self, context: ExecutionContext) -> None:
        """
        Execute the plan produced by the Planner with the chosen model and tool.

        Writes
        ------
        context.result → final answer / payload returned to the API layer
        """
        _stage_log(
            "executor",
            context,
            f"tool={context.selected_tool!r} model={context.selected_model!r}",
        )
        t0 = time.perf_counter()

        plan = context.metadata.get("plan")
        agent_input = AgentInput(
            message=context.message,
            user_id=context.user_id,
            context={
                "role": context.role,
                "selected_tool": context.selected_tool,
                "selected_model": context.selected_model,
            },
        )

        try:
            executor_output: AgentOutput = await self._executor.execute(plan, agent_input)
        except Exception as exc:
            _stage_error("executor", context, exc)
            raise _PipelineStageError("executor", str(exc)) from exc

        elapsed = round(time.perf_counter() - t0, 4)
        context.add_metadata("executor_duration_seconds", elapsed)
        context.add_metadata("executor_status", executor_output.status)

        if executor_output.status in ("failed", "partial_failure"):
            context.set_result(executor_output.response)
            raise _PipelineStageError("executor", executor_output.response)

        context.set_result(executor_output.response)

        _stage_log(
            "executor",
            context,
            f"status={executor_output.status!r} duration_s={elapsed}",
        )


# ────────────────────────────────────────────────────────────────────────────
#  Internal sentinel — keeps pipeline.run() clean
# ────────────────────────────────────────────────────────────────────────────

class _PipelineStageError(Exception):
    """
    Raised internally to short-circuit the pipeline when a stage fails
    fatally.  Caught by the endpoint layer; never surfaces to the end-user
    as an unhandled exception.
    """

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"Pipeline aborted at stage '{stage}': {detail}")
