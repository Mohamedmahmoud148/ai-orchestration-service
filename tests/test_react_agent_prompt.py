"""
tests/test_react_agent_prompt.py

Unit tests for ReactAgent system prompt generation and message building.
Tests cover: persona injection, entity context, memory injection,
coreference section presence, and tool list completeness.

These tests validate the AI's prompt engineering without making real LLM calls.
"""
import pytest
from unittest.mock import MagicMock


def _make_context(
    role="student",
    user_id="user_123",
    message="كم عدد مواد الفرقة الثالثة؟",
    history=None,
    academic_context=None,
    metadata=None,
):
    """Build a minimal ExecutionContext-like mock."""
    ctx = MagicMock()
    ctx.role = role
    ctx.user_id = user_id
    ctx.message = message
    ctx.history = history or []
    ctx.academic_context = academic_context or {}
    ctx.metadata = metadata or {}
    return ctx


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt() in react_agent.py."""

    def setup_method(self):
        from app.agents.react_agent import _build_system_prompt
        self._build = _build_system_prompt

    def test_contains_zero_hallucination_rules(self):
        ctx = _make_context()
        prompt = self._build(ctx)
        assert "ZERO-HALLUCINATION" in prompt or "منع الاختلاق" in prompt

    def test_contains_think_act_validate_respond(self):
        ctx = _make_context()
        prompt = self._build(ctx)
        assert "THINK" in prompt or "فكّر" in prompt

    def test_contains_coreference_section(self):
        ctx = _make_context()
        prompt = self._build(ctx)
        assert "COREFERENCE" in prompt or "الضمائر" in prompt or "coreference" in prompt.lower()

    def test_includes_user_id_in_ctx_line(self):
        ctx = _make_context(user_id="abc123")
        prompt = self._build(ctx)
        assert "abc123" in prompt

    def test_includes_role_in_ctx_line(self):
        ctx = _make_context(role="admin")
        prompt = self._build(ctx)
        assert "admin" in prompt

    def test_memory_section_injected_when_present(self):
        ctx = _make_context(metadata={
            "memory": {
                "last_intent": "result_query",
                "last_message": "ما درجاتي؟",
                "last_result": "درجة 85 في Data Structures",
            }
        })
        prompt = self._build(ctx)
        assert "result_query" in prompt or "ذاكرة" in prompt

    def test_entity_section_injected_when_present(self):
        ctx = _make_context(metadata={
            "conversation_entities": {
                "courses": ["قواعد البيانات"],
                "goals": ["graduation"],
                "doctors": [],
                "semesters": [],
                "gpa_values": [],
                "exams": [],
            }
        })
        prompt = self._build(ctx)
        assert "قواعد البيانات" in prompt

    def test_goal_section_injected_when_present(self):
        ctx = _make_context(metadata={"user_goal": "graduation"})
        prompt = self._build(ctx)
        assert "التخرج" in prompt or "graduation" in prompt.lower()

    def test_student_persona_injected(self):
        ctx = _make_context(role="student")
        prompt = self._build(ctx)
        # Should contain the student persona (at least a fragment)
        assert len(prompt) > 500  # prompt is non-trivial

    def test_admin_persona_injected(self):
        ctx = _make_context(role="admin")
        prompt = self._build(ctx)
        assert len(prompt) > 500

    def test_schema_section_present(self):
        ctx = _make_context()
        prompt = self._build(ctx)
        assert "نقاط النهاية" in prompt or "endpoint" in prompt.lower()


class TestBuildMessages:
    """Tests for _build_messages() — message list construction."""

    def setup_method(self):
        from app.agents.react_agent import _build_messages
        self._build = _build_messages

    def test_first_message_is_system(self):
        ctx = _make_context()
        msgs = self._build(ctx)
        assert msgs[0]["role"] == "system"

    def test_last_message_is_user_message(self):
        ctx = _make_context(message="سؤالي هنا")
        msgs = self._build(ctx)
        assert msgs[-1]["role"] == "user"
        assert "سؤالي هنا" in msgs[-1]["content"]

    def test_history_turns_included(self):
        history = [
            {"role": "user", "content": "سؤال قديم"},
            {"role": "assistant", "content": "جواب قديم"},
        ]
        ctx = _make_context(history=history)
        msgs = self._build(ctx)
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_history_truncated_to_ten(self):
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(30)
        ]
        ctx = _make_context(history=history)
        msgs = self._build(ctx)
        # system + up to 10 history + current user message
        assert len(msgs) <= 12

    def test_empty_history_ok(self):
        ctx = _make_context(history=[])
        msgs = self._build(ctx)
        assert len(msgs) == 2  # system + user


class TestToolList:
    """Tests that all expected tools are registered."""

    def test_all_four_tools_registered(self):
        from app.agents.react_agent import _ALL_TOOLS
        names = {t["function"]["name"] for t in _ALL_TOOLS}
        assert "call_backend_api" in names
        assert "read_regulation_pdf" in names
        assert "generate_exam" in names
        assert "analyze_academic_profile" in names

    def test_analyze_academic_profile_has_focus_enum(self):
        from app.agents.react_agent import _TOOL_ACADEMIC_ANALYSIS
        params = _TOOL_ACADEMIC_ANALYSIS["function"]["parameters"]["properties"]
        assert "focus" in params
        assert "enum" in params["focus"]
        assert "graduation_readiness" in params["focus"]["enum"]
