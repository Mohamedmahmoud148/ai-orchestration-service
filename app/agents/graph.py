"""
app/agents/graph.py

LangGraph orchestration wrapper — Phases 1-3 of the agentic architecture
upgrade (see docs/AGENTIC_UPGRADE_ROADMAP.md).

Phase 1 wrapped the whole Agent.run() call as a single opaque node. Phase 2
extracted Agent.run()'s inline stages into real, independently callable
methods (_load_memory / _execute_core / _save_memory). Phase 3 (this
version) gives each of those three stages its own LangGraph node, calling
them directly instead of the monolithic Agent.run() — so per-node timing
and structure are now real, not cosmetic.

Nothing about ReactAgent, PlanExecutor, or core/rbac.py is touched here or
by the Phase 2 extraction — this file only sequences the same three method
calls Agent.run() already makes, in the same order, with the same
early-return semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.execution_context import ExecutionContext
from app.agents.schemas import ExecutionPlan
from app.core.logging import logger

if TYPE_CHECKING:
    from app.agents.agent import Agent


class AgentGraphState(TypedDict):
    """
    LangGraph state for the 3-node agent pipeline.

    `context` is the live ExecutionContext, held by reference (not
    field-mapped) for the same reason as Phase 1: it's the universal
    mutate-in-place carrier every other module already assumes object
    identity for. `memory_key`/`plan`/`module_name` are the values
    Agent._load_memory() produces and Agent._execute_core()/_save_memory()
    consume — threaded through state exactly as they're threaded through
    local variables in Agent.run().
    """
    context: ExecutionContext
    memory_key: str
    plan: Optional[ExecutionPlan]
    module_name: Optional[str]
    should_return_early: bool


def _make_load_memory_node(agent: "Agent"):
    async def _load_memory_node(state: AgentGraphState) -> AgentGraphState:
        context = state["context"]
        memory_key, plan, module_name, should_return_early = await agent._load_memory(context)
        logger.debug(
            "LangGraph load_memory: user_id=%s conversation_id=%s should_return_early=%s",
            context.user_id, context.conversation_id, should_return_early,
        )
        return {
            "context": context,
            "memory_key": memory_key,
            "plan": plan,
            "module_name": module_name,
            "should_return_early": should_return_early,
        }

    return _load_memory_node


def _make_agent_core_node(agent: "Agent"):
    async def _agent_core_node(state: AgentGraphState) -> AgentGraphState:
        context = state["context"]
        await agent._execute_core(context, state["plan"], state["module_name"])
        return {"context": context}

    return _agent_core_node


def _make_save_memory_node(agent: "Agent"):
    async def _save_memory_node(state: AgentGraphState) -> AgentGraphState:
        context = state["context"]
        await agent._save_memory(context, state["memory_key"], state["plan"], state["module_name"])
        return {"context": context}

    return _save_memory_node


def _route_after_load_memory(state: AgentGraphState) -> str:
    """
    Conditional edge — matches Agent.run()'s early `return context` when an
    in-progress clarification was answered with an invalid choice. In that
    case context.result is already set and neither agent_core nor
    save_memory should run (no background tasks, no memory save), exactly
    as in the original inline code.
    """
    return END if state["should_return_early"] else "agent_core"


def build_agent_graph(agent: "Agent"):
    """
    Build and compile the 3-node LangGraph wrapper around Agent's extracted
    stages.

    Takes the already-constructed Agent instance (the same one main.py
    assembles and stores on app.state.agent) so there is no import-order
    coupling to main.py and the graph is trivially unit-testable with a
    fake/mock Agent.

    Graph shape:
        START → load_memory ─┬─(should_return_early)─→ END
                              └─(else)──→ agent_core → save_memory → END
    """
    graph = StateGraph(AgentGraphState)
    graph.add_node("load_memory", _make_load_memory_node(agent))
    graph.add_node("agent_core", _make_agent_core_node(agent))
    graph.add_node("save_memory", _make_save_memory_node(agent))

    graph.add_edge(START, "load_memory")
    graph.add_conditional_edges(
        "load_memory",
        _route_after_load_memory,
        {"agent_core": "agent_core", END: END},
    )
    graph.add_edge("agent_core", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
