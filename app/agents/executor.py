"""
app/agents/executor.py

PlanExecutor — the dynamic module dispatcher.

Responsibilities:
  - Accept an ExecutionPlan + a module_name resolved by the Agent.
  - Look up (or lazily instantiate) the correct Module from app.modules.
  - Call module.run(agent_input, plan) and return an AgentOutput.
  - Fall back to a generic step-by-step executor for plans without a
    dedicated module (model-only or multi-tool flows).
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Dict, Optional, Awaitable, TYPE_CHECKING

from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan
from app.core.logging import logger

if TYPE_CHECKING:
    from app.agents.model_router import ModelRouter


# Map module_name strings → fully qualified class names inside app.modules
_MODULE_CLASS_MAP: Dict[str, tuple[str, str]] = {
    "exam_generation":  ("app.modules.exam_generation",  "ExamGenerationModule"),
    "summarization":    ("app.modules.summarization",    "SummarizationModule"),
    "file_extraction":  ("app.modules.file_extraction",  "FileExtractionModule"),
    "result_query":     ("app.modules.result_query",      "ResultQueryModule"),
}


class PlanExecutor:
    """
    Iterates through an ExecutionPlan and dispatches work to the
    appropriate Module, model call, or backend tool call.
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
        # Cache for lazily instantiated module objects
        self._module_cache: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    #  Public API (called by Agent)
    # ──────────────────────────────────────────────────────────────────────

    async def execute(
        self,
        plan: Any,
        input_context: AgentInput,
        module_name: str = "model_only",
    ) -> AgentOutput:
        """
        Dispatch to the correct Module if one is registered, otherwise
        fall back to the generic step-by-step plan runner.
        """
        # ── 1. Dedicated module flow ───────────────────────────────────
        if module_name in _MODULE_CLASS_MAP:
            return await self._run_module(module_name, plan, input_context)

        # ── 2. Generic plan step-loop (model-only or multi-tool) ──────
        if plan and isinstance(plan, ExecutionPlan):
            return await self._run_plan(plan, input_context)

        # ── 3. Bare LLM call (no plan, no module) ─────────────────────
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

            # Inject only the dependencies that the class accepts
            import inspect
            init_params = inspect.signature(cls.__init__).parameters
            kwargs: Dict[str, Any] = {}
            if "model_router" in init_params:
                kwargs["model_router"] = self.model_router
            if "backend_client" in init_params:
                # Manufacture a thin wrapper that matches the module interface
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
        """Instantiate (or fetch from cache) and run the named module."""
        logger.info("PlanExecutor: dispatching to module '%s'.", module_name)
        try:
            module = self._get_module(module_name)
            return await module.run(input_context, plan)
        except Exception as exc:
            logger.error(
                "PlanExecutor: module '%s' raised an error: %s",
                module_name,
                exc,
                exc_info=True,
            )
            return AgentOutput(
                status="failed",
                response=f"Module '{module_name}' failed: {exc}",
            )

    # ──────────────────────────────────────────────────────────────────────
    #  Generic step-loop (fallback for multi-tool / model-only plans)
    # ──────────────────────────────────────────────────────────────────────

    async def _run_plan(
        self, plan: ExecutionPlan, input_context: AgentInput
    ) -> AgentOutput:
        """Execute a generic ExecutionPlan step by step."""
        if not plan.is_executable:
            return AgentOutput(
                status="failed",
                response="Plan is marked as non-executable.",
            )

        execution_results: Dict[int, Any] = {}
        successful_steps = 0
        ordered_steps = sorted(plan.steps, key=lambda s: s.step_id)

        for step in ordered_steps:
            logger.info(
                "PlanExecutor: step_id=%s action=%s", step.step_id, step.action
            )
            payload = self._interpolate(step.input_payload, execution_results)

            try:
                if step.action == "tool":
                    result = await self.backend_execution_func(
                        step.tool_name,
                        payload,
                        input_context.auth_header,
                        input_context.user_id,
                    )
                elif step.action == "model":
                    prompt = payload.get("prompt", str(payload))
                    sys_inst = payload.get("system_instruction", "")
                    text = await self.model_router.generate(
                        prompt=prompt, system_instruction=sys_inst
                    )
                    result = {"output": text}
                elif step.action == "agent_module":
                    result_out = await self._run_module(
                        step.module_name or "", plan, input_context
                    )
                    result = {"output": result_out.response, "data": result_out.data}
                else:
                    result = {"skipped": True, "reason": f"Unknown action '{step.action}'"}

                execution_results[step.step_id] = result

                if isinstance(result, dict) and "error" in result:
                    return AgentOutput(
                        status="partial_failure",
                        response=f"Halted at step {step.step_id}: {result['error']}",
                        data={"results": execution_results},
                    )
                successful_steps += 1

            except Exception as exc:
                return AgentOutput(
                    status="failed",
                    response=f"Step {step.step_id} failed: {exc}",
                    data={"error": str(exc)},
                )

        return AgentOutput(
            status="success",
            response=plan.goal_summary or f"Executed {successful_steps} steps.",
            data={"results": execution_results},
        )

    async def _fallback_model_call(self, input_context: AgentInput) -> AgentOutput:
        """Pure LLM call when there's no plan and no module."""
        if not self.model_router:
            return AgentOutput(status="failed", response="No model router available.")

        response = await self.model_router.generate(
            prompt=input_context.message,
            system_instruction="You are a helpful AI assistant.",
        )
        return AgentOutput(
            status="success",
            response=response or "I could not generate a response.",
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Helpers
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
                    out[k] = sub if v.strip() == match.group(0) else v.replace(match.group(0), str(sub))
                else:
                    out[k] = v
            else:
                out[k] = v
        return out
