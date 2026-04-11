"""
app/agents/pipeline.py

Structural protocols and shared sentinel exception for the orchestration pipeline.

The OrchestrationPipeline class has been removed — the Agent class in agent.py
is the authoritative orchestration entry-point as of v3.0.

What remains:
  - Protocol interfaces (Planner, ToolRegistry, ModelRouter, Executor)
      Kept for type-checking and forward-compatibility.
  - _PipelineStageError
      Imported by agent.py and chat.py to signal stage failures cleanly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.agents.schemas import AgentInput, AgentOutput


# ─────────────────────────────────────────────────────────────────────────────
#  Structural Protocols — depend on behaviour, not concrete classes
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Shared sentinel exception
# ─────────────────────────────────────────────────────────────────────────────

class _PipelineStageError(Exception):
    """
    Raised inside the Agent pipeline when a stage fails fatally.

    Caught by the endpoint layer (chat.py); never surfaces to the end-user
    as an unhandled exception.
    """

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"Pipeline aborted at stage '{stage}': {detail}")
