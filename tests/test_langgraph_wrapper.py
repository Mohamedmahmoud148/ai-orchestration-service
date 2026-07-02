"""
tests/test_langgraph_wrapper.py

Regression tests for the LangGraph orchestration wrapper (app/agents/graph.py),
covering Phases 1-3 of the agentic architecture upgrade.

Goal: prove the wrapper changes nothing about observable behavior.
  1. Graph construction doesn't eagerly execute anything.
  2. Flag-off path (app.state.agent_graph is None) is unchanged.
  3. Flag-on path produces an identical ExecutionContext to calling
     agent.run() directly — including the should_return_early conditional
     edge, which must skip agent_core/save_memory exactly like the inline
     early `return context` in Agent.run() does.
  4. _PipelineStageError propagates through graph.ainvoke() unchanged.
  5. Object identity is preserved (same ExecutionContext instance back).
  6. langgraph is never imported into the process when the flag is off.
  7. tests/test_pipeline.py and tests/test_agent_memory_stages.py stay green
     unmodified — not re-asserted here, they're the regression gate for the
     real Agent methods this graph calls.
"""
import sys
import subprocess
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.agent import Agent
from app.agents.execution_context import ExecutionContext
from app.agents.pipeline import _PipelineStageError


def _make_context(**overrides) -> ExecutionContext:
    defaults = dict(
        user_id="user_123",
        role="student",
        message="كام ساعة خلصت؟",
        conversation_id="conv_abc",
    )
    defaults.update(overrides)
    return ExecutionContext(**defaults)


class _FakeAgent:
    """
    Stand-in for Agent, mirroring the real _load_memory / _execute_core /
    _save_memory / run() contracts (see app/agents/agent.py). run() calls
    the three sub-methods in the same order the real Agent.run() does,
    so the flag-on-vs-off equivalence tests exercise genuinely the same
    control flow as production, not a hand-waved shortcut.
    """

    def __init__(self, result="ok", raise_exc=None, should_return_early=False):
        self._result = result
        self._raise_exc = raise_exc
        self._should_return_early = should_return_early
        self.load_memory_calls = 0
        self.execute_core_calls = 0
        self.save_memory_calls = 0

    async def _load_memory(self, context: ExecutionContext):
        self.load_memory_calls += 1
        if self._should_return_early:
            context.set_result("عذراً، هذا الاختيار غير صحيح.")
            context.add_metadata("clarification_needed", True)
            return "mem_key", None, None, True
        return "mem_key", None, None, False

    async def _execute_core(self, context: ExecutionContext, plan, module_name) -> None:
        self.execute_core_calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        context.set_result(self._result)
        context.set_intent("general_chat")
        context.set_tool("react_agent")
        context.set_model("openai/gpt-4o-mini")

    async def _save_memory(self, context: ExecutionContext, memory_key, plan, module_name) -> None:
        self.save_memory_calls += 1
        context.add_metadata("agent_duration_seconds", 0.01)

    async def run(self, context: ExecutionContext) -> ExecutionContext:
        memory_key, plan, module_name, should_return_early = await self._load_memory(context)
        if should_return_early:
            return context
        await self._execute_core(context, plan, module_name)
        await self._save_memory(context, memory_key, plan, module_name)
        return context


class TestGraphConstruction:
    def test_build_does_not_execute_any_stage(self):
        from app.agents.graph import build_agent_graph

        fake_agent = _FakeAgent()
        build_agent_graph(fake_agent)

        assert fake_agent.load_memory_calls == 0
        assert fake_agent.execute_core_calls == 0
        assert fake_agent.save_memory_calls == 0


class TestFlagOnEquivalence:
    @pytest.mark.asyncio
    async def test_graph_result_matches_direct_agent_run(self):
        from app.agents.graph import build_agent_graph

        # Two independent contexts, one per path, so mutation on one can't
        # bleed into the other and mask a real divergence.
        ctx_direct = _make_context()
        ctx_graph = _make_context()

        agent_for_direct = _FakeAgent(result="hello from agent")
        agent_for_graph = _FakeAgent(result="hello from agent")

        direct_result = await agent_for_direct.run(ctx_direct)

        graph = build_agent_graph(agent_for_graph)
        state = await graph.ainvoke({"context": ctx_graph})
        graph_result = state["context"]

        assert graph_result.result == direct_result.result
        assert graph_result.intent == direct_result.intent
        assert graph_result.selected_tool == direct_result.selected_tool
        assert graph_result.selected_model == direct_result.selected_model
        assert graph_result.metadata == direct_result.metadata

        # All three stages ran exactly once on both paths.
        assert agent_for_graph.load_memory_calls == 1
        assert agent_for_graph.execute_core_calls == 1
        assert agent_for_graph.save_memory_calls == 1

    @pytest.mark.asyncio
    async def test_object_identity_preserved(self):
        from app.agents.graph import build_agent_graph

        ctx = _make_context()
        agent = _FakeAgent()
        graph = build_agent_graph(agent)

        state = await graph.ainvoke({"context": ctx})

        assert state["context"] is ctx


class TestShouldReturnEarlyConditionalEdge:
    @pytest.mark.asyncio
    async def test_early_return_skips_agent_core_and_save_memory(self):
        """
        The key behavior Phase 3 had to preserve: Agent.run() returns
        immediately (no _execute_core, no _save_memory) when
        _load_memory signals should_return_early=True. The graph's
        conditional edge must reproduce this exactly.
        """
        from app.agents.graph import build_agent_graph

        agent_direct = _FakeAgent(should_return_early=True)
        agent_graph = _FakeAgent(should_return_early=True)

        direct_result = await agent_direct.run(_make_context())
        assert agent_direct.execute_core_calls == 0
        assert agent_direct.save_memory_calls == 0

        graph = build_agent_graph(agent_graph)
        state = await graph.ainvoke({"context": _make_context()})
        graph_result = state["context"]

        assert agent_graph.load_memory_calls == 1
        assert agent_graph.execute_core_calls == 0
        assert agent_graph.save_memory_calls == 0
        assert graph_result.result == direct_result.result == "عذراً، هذا الاختيار غير صحيح."


class TestPipelineStageErrorPropagation:
    @pytest.mark.asyncio
    async def test_pipeline_stage_error_propagates_through_graph(self):
        from app.agents.graph import build_agent_graph

        exc = _PipelineStageError(stage="executor", detail="forbidden: role not allowed")
        agent = _FakeAgent(raise_exc=exc)
        graph = build_agent_graph(agent)

        with pytest.raises(_PipelineStageError) as excinfo:
            await graph.ainvoke({"context": _make_context()})

        assert excinfo.value.stage == "executor"
        assert excinfo.value.detail == "forbidden: role not allowed"


class TestFlagOffPath:
    @pytest.mark.asyncio
    async def test_run_orchestration_uses_direct_agent_when_graph_is_none(self):
        """
        Mirrors chat.py's _run_orchestration: when app.state.agent_graph is
        None (default), the direct agent.run(context) path must be used,
        unchanged from pre-Phase-1 behavior.
        """
        from fastapi import FastAPI
        from app.api.routes.chat import _run_orchestration

        app = FastAPI()
        fake_agent = _FakeAgent(result="direct path result")
        app.state.agent = fake_agent
        app.state.agent_graph = None

        class _FakeRequest:
            def __init__(self, app):
                self.app = app

        ctx = _make_context()
        result = await _run_orchestration(_FakeRequest(app), ctx)

        assert result.result == "direct path result"
        assert fake_agent.load_memory_calls == 1
        assert fake_agent.execute_core_calls == 1
        assert fake_agent.save_memory_calls == 1


class TestNoImportWhenFlagOff:
    def test_langgraph_not_imported_when_flag_off(self):
        """
        Process-level check: with USE_LANGGRAPH_ORCHESTRATION unset/false,
        importing app.main must not pull `langgraph` into sys.modules.
        This is the concrete proof that cold start / memory footprint on
        Railway are unaffected when the flag is off (default). Run in a
        subprocess since langgraph may already be imported by earlier
        tests in this same pytest session.
        """
        script = textwrap.dedent(
            """
            import os
            os.environ["BACKEND_BASE_URL"] = "http://localhost:5000"
            os.environ["OPENROUTER_API_KEY"] = "test-key-not-used"
            os.environ["ENVIRONMENT"] = "development"
            os.environ.pop("USE_LANGGRAPH_ORCHESTRATION", None)

            import sys
            import app.main  # noqa: F401 — import side effect is what's under test

            assert "langgraph" not in sys.modules, (
                "langgraph was imported even though USE_LANGGRAPH_ORCHESTRATION is off"
            )
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 and "ModuleNotFoundError" in result.stderr and "langgraph" not in result.stderr:
            # `import app.main` pulls in the full dependency stack (torch,
            # transformers, chromadb, ...) unrelated to this test's actual
            # invariant. If some *other* module is missing in this local
            # environment (e.g. a torch wheel unavailable for this Python
            # version — production pins torch>=2.2.0,<2.3 and runs Python
            # 3.11 per the Dockerfile), that's an environment provisioning
            # gap, not a regression this test is meant to catch. Skip rather
            # than false-fail; a fully-provisioned environment (CI, Docker)
            # will still catch a real langgraph-import regression.
            pytest.skip(
                f"app.main import failed for an unrelated dependency reason, "
                f"cannot verify langgraph-import isolation in this environment: "
                f"{result.stderr.strip().splitlines()[-1]}"
            )
        assert result.returncode == 0, (
            f"subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "OK" in result.stdout


class TestRealAgentThroughRealGraph:
    """
    Closes the loop: everywhere above uses _FakeAgent (a hand-written
    stand-in) to isolate graph.py's own logic. This test instead builds the
    real Agent class and runs it through the real graph, to prove the two
    modules actually compose correctly against Agent's true method
    signatures — not just against a stand-in that might drift from them.
    """

    def _make_real_agent(self, react_agent) -> Agent:
        agent = Agent(
            planner=MagicMock(),
            tool_registry=MagicMock(),
            model_router=MagicMock(),
            executor=MagicMock(),
            react_agent=react_agent,
        )
        store = AsyncMock()
        for method_name, return_value in dict(
            get_conversation=None,
            get_preferences=None,
            get_entities={},
            get_user_goal="",
            get_academic_profile={},
            get_personalized_context="",
            get_file_context=None,
            get_clarification=None,
        ).items():
            getattr(store, method_name).return_value = return_value
        agent._memory_store = store
        return agent

    @pytest.mark.asyncio
    async def test_real_agent_via_graph_matches_real_agent_via_run(self):
        from app.agents.graph import build_agent_graph

        react_agent_for_direct = AsyncMock()
        react_agent_for_direct.run.return_value = "real answer"
        react_agent_for_graph = AsyncMock()
        react_agent_for_graph.run.return_value = "real answer"

        agent_direct = self._make_real_agent(react_agent_for_direct)
        agent_graph = self._make_real_agent(react_agent_for_graph)

        direct_result = await agent_direct.run(_make_context())

        graph = build_agent_graph(agent_graph)
        state = await graph.ainvoke({"context": _make_context()})
        graph_result = state["context"]

        assert graph_result.result == direct_result.result == "real answer"
        assert graph_result.selected_tool == direct_result.selected_tool
        assert graph_result.selected_model == direct_result.selected_model
        react_agent_for_graph.run.assert_awaited_once()


class _FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeStreamRequest:
    def __init__(self, app):
        self.app = app
        self.headers = _FakeHeaders({"Authorization": "Bearer test-token"})


class TestStreamEndpointFallbackRoutesThroughGraph:
    """
    Phase 4: /api/chat/stream's non-streaming fallback path (fires when no
    ReactAgent is available) is now routed through _run_orchestration, the
    same helper /api/chat uses — so it goes through the graph when the flag
    is on, exactly like the main endpoint.
    """

    @pytest.mark.asyncio
    async def test_fallback_uses_graph_when_agent_graph_is_set(self):
        from app.api.routes.chat import chat_stream_endpoint
        from app.agents.graph import build_agent_graph
        from app.models.chat import ChatRequest
        from fastapi import FastAPI

        app = FastAPI()
        fake_agent = _FakeAgent(result="graph streamed answer")
        app.state.agent = fake_agent
        app.state.agent_graph = build_agent_graph(fake_agent)
        app.state.rate_limiter = None

        chat_request = ChatRequest(message="hi", user_id="user_123", role="student")
        response = await chat_stream_endpoint(chat_request, _FakeStreamRequest(app), token=None)

        frames = [chunk async for chunk in response.body_iterator]
        full = "".join(frames)

        assert "graph streamed answer" in full
        assert fake_agent.load_memory_calls == 1
        assert fake_agent.execute_core_calls == 1
        assert fake_agent.save_memory_calls == 1

    @pytest.mark.asyncio
    async def test_fallback_uses_direct_agent_when_agent_graph_is_none(self):
        from app.api.routes.chat import chat_stream_endpoint
        from app.models.chat import ChatRequest
        from fastapi import FastAPI

        app = FastAPI()
        fake_agent = _FakeAgent(result="direct streamed answer")
        app.state.agent = fake_agent
        app.state.agent_graph = None
        app.state.rate_limiter = None

        chat_request = ChatRequest(message="hi", user_id="user_123", role="student")
        response = await chat_stream_endpoint(chat_request, _FakeStreamRequest(app), token=None)

        frames = [chunk async for chunk in response.body_iterator]
        full = "".join(frames)

        assert "direct streamed answer" in full
        assert fake_agent.load_memory_calls == 1
        assert fake_agent.execute_core_calls == 1
        assert fake_agent.save_memory_calls == 1
