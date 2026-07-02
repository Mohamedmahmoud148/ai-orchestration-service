"""
tests/test_agent_memory_stages.py

Characterization tests for Agent._load_memory() / Agent._save_memory() —
Phase 2 of the agentic architecture upgrade (docs/AGENTIC_UPGRADE_ROADMAP.md).

These were extracted verbatim (no logic changes) from Agent.run()'s inline
Stage 0 / Stage 5 blocks. These tests lock in the exact current behavior,
including the one subtle control-flow edge case: an invalid clarification
choice must short-circuit *before* Stage 5 and before the background
language/entity tasks fire.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.agent import Agent
from app.agents.execution_context import ExecutionContext


def _make_agent(memory_store: AsyncMock) -> Agent:
    """Build an Agent with all collaborators mocked except the memory store,
    which is swapped in after construction (Agent.__init__ always resolves
    the process-wide singleton via get_memory_store())."""
    agent = Agent(
        planner=MagicMock(),
        tool_registry=MagicMock(),
        model_router=MagicMock(),
        executor=MagicMock(),
        react_agent=None,
    )
    agent._memory_store = memory_store
    return agent


def _make_memory_store(**overrides) -> AsyncMock:
    store = AsyncMock()
    defaults = dict(
        get_conversation=None,
        get_preferences=None,
        get_entities={},
        get_user_goal="",
        get_academic_profile={},
        get_personalized_context="",
        get_file_context=None,
        get_clarification=None,
    )
    defaults.update(overrides)
    for method_name, return_value in defaults.items():
        getattr(store, method_name).return_value = return_value
    return store


def _make_context(**overrides) -> ExecutionContext:
    defaults = dict(
        user_id="user_123",
        role="student",
        message="كام ساعة خلصت؟",
        conversation_id="conv_abc",
    )
    defaults.update(overrides)
    return ExecutionContext(**defaults)


class TestLoadMemoryNoClarification:
    @pytest.mark.asyncio
    async def test_returns_memory_key_and_no_plan(self):
        store = _make_memory_store()
        agent = _make_agent(store)
        context = _make_context()

        memory_key, plan, module_name, should_return_early = await agent._load_memory(context)

        assert memory_key == "user_123:conv_abc"
        assert plan is None
        assert module_name is None
        assert should_return_early is False

    @pytest.mark.asyncio
    async def test_memory_key_falls_back_to_user_id_without_conversation(self):
        store = _make_memory_store()
        agent = _make_agent(store)
        context = _make_context(conversation_id="")

        memory_key, _, _, _ = await agent._load_memory(context)

        assert memory_key == "user_123"

    @pytest.mark.asyncio
    async def test_loaded_data_populates_context_metadata(self):
        store = _make_memory_store(
            get_conversation={"last_intent": "general_chat"},
            get_preferences={"language": "ar"},
            get_entities={"courses": ["CS101"]},
            get_user_goal="graduate on time",
            get_academic_profile={"gpa": 3.4},
            get_personalized_context="likes concise answers",
            get_file_context={"file_url": "https://x/y.pdf", "file_name": "y.pdf"},
        )
        agent = _make_agent(store)
        context = _make_context()

        await agent._load_memory(context)

        assert context.metadata["memory"] == {"last_intent": "general_chat"}
        assert context.metadata["preferences"] == {"language": "ar"}
        assert context.metadata["conversation_entities"] == {"courses": ["CS101"]}
        assert context.academic_context["conversation_entities"] == {"courses": ["CS101"]}
        assert context.metadata["user_goal"] == "graduate on time"
        assert context.metadata["academic_profile"] == {"gpa": 3.4}
        assert context.metadata["personalized_context"] == "likes concise answers"
        assert context.academic_context["file_url"] == "https://x/y.pdf"
        assert context.academic_context["file_name"] == "y.pdf"

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing_academic_context_file_url(self):
        """A caller-provided file_url in academic_context must win over the
        restored one from memory — matches the original `if not
        context.academic_context.get("file_url")` guard."""
        store = _make_memory_store(
            get_file_context={"file_url": "https://memory/old.pdf", "file_name": "old.pdf"},
        )
        agent = _make_agent(store)
        context = _make_context(academic_context={"file_url": "https://caller/new.pdf"})

        await agent._load_memory(context)

        assert context.academic_context["file_url"] == "https://caller/new.pdf"

    @pytest.mark.asyncio
    async def test_fires_background_language_and_entity_tasks(self):
        store = _make_memory_store()
        agent = _make_agent(store)
        context = _make_context()

        await agent._load_memory(context)
        # Background tasks are fire-and-forget asyncio.create_task calls;
        # give the event loop a tick to run them.
        await asyncio.sleep(0)

        store.detect_and_save_language.assert_awaited_once_with(
            context.user_id, context.message
        )


class TestLoadMemoryClarificationResolved:
    @pytest.mark.asyncio
    async def test_numeric_choice_builds_plan_and_deletes_clarification(self):
        store = _make_memory_store(
            get_clarification={
                "options": [{"id": "off-1", "subjectOfferingId": "off-1", "name": "Section A"}],
                "original_intent": "result_query",
                "step_context": {"module_name": "result_query"},
            },
        )
        agent = _make_agent(store)
        context = _make_context(message="1")

        memory_key, plan, module_name, should_return_early = await agent._load_memory(context)

        assert should_return_early is False
        assert plan is not None
        assert plan.intent == "result_query"
        assert module_name == "result_query"
        assert context.intent == "result_query"
        assert context.selected_tool == "result_query"
        assert context.academic_context["subjectOfferingId"] == "off-1"
        store.delete_clarification.assert_awaited_once_with(context.user_id)

    @pytest.mark.asyncio
    async def test_generate_exam_clarification_populates_exam_params(self):
        store = _make_memory_store(
            get_clarification={
                "options": [{"id": "off-1", "subjectOfferingId": "off-1", "name": "Section A"}],
                "original_intent": "generate_exam",
                "step_context": {
                    "module_name": "exam_generation",
                    "exam_params": {"question_count": 10, "difficulty": "medium"},
                },
            },
        )
        agent = _make_agent(store)
        context = _make_context(message="Section A")

        _, plan, _, _ = await agent._load_memory(context)

        assert plan.exam_params is not None
        assert plan.exam_params.subjectOfferingId == "off-1"

    @pytest.mark.asyncio
    async def test_invalid_choice_short_circuits_before_background_tasks(self):
        store = _make_memory_store(
            get_clarification={
                "options": [{"id": "off-1", "name": "Section A"}],
                "original_intent": "result_query",
                "step_context": {"module_name": "result_query"},
            },
        )
        agent = _make_agent(store)
        context = _make_context(message="not a valid option at all")

        memory_key, plan, module_name, should_return_early = await agent._load_memory(context)

        assert should_return_early is True
        assert plan is None
        assert module_name is None
        assert context.result == "عذراً، هذا الاختيار غير صحيح."
        assert context.metadata["clarification_needed"] is True

        # Background tasks must NOT have fired — the original inline code
        # `return context`s before reaching that point.
        await asyncio.sleep(0)
        store.detect_and_save_language.assert_not_awaited()


class TestSaveMemoryNormalPath:
    @pytest.mark.asyncio
    async def test_saves_conversation_with_expected_shape(self):
        store = _make_memory_store()
        agent = _make_agent(store)
        context = _make_context()
        context.set_intent("general_chat")
        context.set_result("hello there")

        await agent._save_memory(context, "user_123:conv_abc", plan=None, module_name=None)

        store.save_conversation.assert_awaited_once()
        call_args = store.save_conversation.await_args
        assert call_args.args[0] == "user_123:conv_abc"
        memory_data = call_args.args[1]
        assert memory_data["last_intent"] == "general_chat"
        assert memory_data["last_result"] == "hello there"
        assert memory_data["role"] == "student"

    @pytest.mark.asyncio
    async def test_extracts_and_saves_file_url_and_active_document(self):
        store = _make_memory_store()
        agent = _make_agent(store)
        context = _make_context()
        context.set_result("هنا الملف: https://cdn.example.com/materials/lecture1.pdf")

        await agent._save_memory(context, "user_123:conv_abc", plan=None, module_name=None)

        store.save_file_context.assert_awaited_once()
        url_arg = store.save_file_context.await_args.args[1]
        assert url_arg == "https://cdn.example.com/materials/lecture1.pdf"

        store.set_active_document.assert_awaited_once()
        active_doc = store.set_active_document.await_args.args[1]
        assert active_doc["file_url"] == "https://cdn.example.com/materials/lecture1.pdf"
        assert active_doc["title"] == "lecture1.pdf"

    @pytest.mark.asyncio
    async def test_no_file_url_in_response_skips_file_save(self):
        store = _make_memory_store()
        agent = _make_agent(store)
        context = _make_context()
        context.set_result("plain text answer, no links here")

        await agent._save_memory(context, "user_123:conv_abc", plan=None, module_name=None)

        store.save_file_context.assert_not_awaited()
        store.set_active_document.assert_not_awaited()


class TestSaveMemoryClarificationNeeded:
    @pytest.mark.asyncio
    async def test_saves_clarification_instead_of_conversation(self):
        store = _make_memory_store()
        agent = _make_agent(store)
        context = _make_context()
        context.add_metadata("clarification_needed", True)
        context.add_metadata("clarification_options", [{"id": "1", "name": "A"}])
        context.set_intent("generate_exam")

        await agent._save_memory(
            context, "user_123:conv_abc", plan=None, module_name="exam_generation"
        )

        store.save_clarification.assert_awaited_once()
        store.save_conversation.assert_not_awaited()
        data = store.save_clarification.await_args.args[1]
        assert data["original_intent"] == "generate_exam"
        assert data["step_context"]["module_name"] == "exam_generation"


class TestRunOrchestratesBothStages:
    @pytest.mark.asyncio
    async def test_run_calls_load_then_save_in_order(self):
        """End-to-end check that run() still wires the two extracted stages
        together correctly, with the ReactAgent producing the result."""
        store = _make_memory_store()
        react_agent = AsyncMock()
        react_agent.run.return_value = "final answer"

        agent = Agent(
            planner=MagicMock(),
            tool_registry=MagicMock(),
            model_router=MagicMock(),
            executor=MagicMock(),
            react_agent=react_agent,
        )
        agent._memory_store = store
        context = _make_context()

        result_context = await agent.run(context)

        assert result_context is context
        assert result_context.result == "final answer"
        react_agent.run.assert_awaited_once_with(context)
        store.save_conversation.assert_awaited_once()
        assert "agent_duration_seconds" in result_context.metadata
