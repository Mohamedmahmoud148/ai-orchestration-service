"""
tests/test_active_document_context.py

Integration tests for the ActiveDocumentContext feature.

Verifies:
1. test_material_listed_then_summarize  — after listing materials (which saves active_doc),
   a follow-up "لخصه" message gets the CONTEXT prefix injected and the prompt contains
   the active document section pointing to read_material_pdf.
2. test_material_listed_then_quiz       — "اعمل امتحان" follow-up also gets the prefix.
3. test_regulation_no_crossover         — when active_doc has document_type="regulation",
   follow-up messages are NOT injected with read_material_pdf context.
4. test_memory_store_active_document    — MemoryStore set/get active_doc round-trip.
5. test_inject_followup_context         — unit-test the _inject_followup_context helper.
6. test_is_followup_message             — unit-test the _is_followup_message helper.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

FAKE_MATERIAL_URL = "https://cdn.example.com/uploads/DL.pdf"
FAKE_MATERIAL_DOC = {
    "document_type": "material",
    "file_url": FAKE_MATERIAL_URL,
    "title": "Lecture One (DL.pdf)",
    "material_id": "mat-001",
    "subject_name": "Data Structure",
}
FAKE_REGULATION_DOC = {
    "document_type": "regulation",
    "file_url": "https://cdn.example.com/regulations/handbook.pdf",
    "title": "Student Handbook",
    "material_id": "",
    "subject_name": "",
}


# ── MemoryStore round-trip ─────────────────────────────────────────────────────

class TestMemoryStoreActiveDocument:
    """Test the new set_active_document / get_active_document methods."""

    def _make_store(self):
        """Create a MemoryStore with Redis disabled (no-op / disk fallback)."""
        import os
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("REDIS_PUBLIC_URL", None)
        os.environ.pop("REDISHOST", None)

        from app.services.memory_store import MemoryStore
        store = MemoryStore.__new__(MemoryStore)
        store.redis_client = None
        store._disabled = True
        store._disk_store = None
        store.redis_url = None
        return store

    @pytest.mark.asyncio
    async def test_set_and_get_active_document_noop(self):
        """set_active_document in no-op mode doesn't raise; get returns None."""
        store = self._make_store()
        # Should not raise
        await store.set_active_document("user-1", FAKE_MATERIAL_DOC)
        result = await store.get_active_document("user-1")
        # In no-op mode (no Redis, no disk) there is nothing to retrieve
        assert result is None

    @pytest.mark.asyncio
    async def test_set_active_document_empty_user_id(self):
        """Empty user_id is ignored gracefully."""
        store = self._make_store()
        await store.set_active_document("", FAKE_MATERIAL_DOC)  # should not raise

    @pytest.mark.asyncio
    async def test_get_active_document_empty_user_id(self):
        store = self._make_store()
        result = await store.get_active_document("")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_via_mocked_redis(self):
        """With a mocked Redis client, set/get round-trip works correctly."""
        import json
        from app.services.memory_store import MemoryStore

        store = MemoryStore.__new__(MemoryStore)
        store._disabled = False
        store._disk_store = None
        store.redis_url = "redis://fake"

        stored: dict = {}

        async def fake_setex(key, ttl, value):
            stored[key] = value

        async def fake_get(key):
            return stored.get(key)

        mock_redis = MagicMock()
        mock_redis.setex = AsyncMock(side_effect=fake_setex)
        mock_redis.get = AsyncMock(side_effect=fake_get)
        store.redis_client = mock_redis

        await store.set_active_document("user-42", FAKE_MATERIAL_DOC)
        result = await store.get_active_document("user-42")

        assert result is not None
        assert result["document_type"] == "material"
        assert result["file_url"] == FAKE_MATERIAL_URL
        assert result["title"] == "Lecture One (DL.pdf)"
        assert result["subject_name"] == "Data Structure"


# ── _inject_followup_context helper ───────────────────────────────────────────

class TestInjectFollowupContext:
    """Unit-test the _inject_followup_context function."""

    def _fn(self):
        from app.agents.react_agent import _inject_followup_context
        return _inject_followup_context

    def test_arabic_summarize_injected(self):
        fn = self._fn()
        result = fn("لخصه", FAKE_MATERIAL_DOC)
        assert FAKE_MATERIAL_URL in result
        assert "read_material_pdf" in result
        assert "لخصه" in result

    def test_arabic_read_injected(self):
        fn = self._fn()
        result = fn("اقراه", FAKE_MATERIAL_DOC)
        assert FAKE_MATERIAL_URL in result

    def test_arabic_exam_injected(self):
        fn = self._fn()
        result = fn("اعمل امتحان", FAKE_MATERIAL_DOC)
        assert FAKE_MATERIAL_URL in result

    def test_english_summarize_injected(self):
        fn = self._fn()
        result = fn("summarize it", FAKE_MATERIAL_DOC)
        assert FAKE_MATERIAL_URL in result

    def test_english_quiz_injected(self):
        fn = self._fn()
        result = fn("make a quiz", FAKE_MATERIAL_DOC)
        assert FAKE_MATERIAL_URL in result

    def test_non_followup_not_injected(self):
        fn = self._fn()
        # Normal question — should not be modified
        result = fn("ايه المواد اللي عندي؟", FAKE_MATERIAL_DOC)
        assert result == "ايه المواد اللي عندي؟"

    def test_no_active_doc_not_injected(self):
        fn = self._fn()
        result = fn("لخصه", None)
        assert result == "لخصه"

    def test_regulation_doc_not_injected(self):
        """Follow-up on regulation doc should NOT inject read_material_pdf."""
        fn = self._fn()
        result = fn("لخصه", FAKE_REGULATION_DOC)
        assert result == "لخصه"

    def test_no_file_url_not_injected(self):
        fn = self._fn()
        doc_no_url = {**FAKE_MATERIAL_DOC, "file_url": ""}
        result = fn("لخصه", doc_no_url)
        assert result == "لخصه"


# ── _is_followup_message helper ───────────────────────────────────────────────

class TestIsFollowupMessage:
    def _fn(self):
        from app.agents.react_agent import _is_followup_message
        return _is_followup_message

    @pytest.mark.parametrize("msg", [
        "لخصه", "اقراه", "اشرحه", "اشرح الملف", "اشرح المحتوى",
        "اعمل quiz", "اعمل امتحان", "استخرج العناوين", "ملخص",
        "summarize it", "read it", "explain it", "make a quiz", "generate exam",
    ])
    def test_followup_detected(self, msg):
        assert self._fn()(msg) is True

    @pytest.mark.parametrize("msg", [
        "ايه المواد اللي عندي؟",
        "كم مادة عندي هذا الترم؟",
        "اعرض لي الجدول",
        "what subjects do I have?",
        "show me the schedule",
    ])
    def test_non_followup_not_detected(self, msg):
        assert self._fn()(msg) is False


# ── System prompt injection ────────────────────────────────────────────────────

class TestBuildActiveDocSection:
    """Test that _build_active_doc_section generates the correct prompt text."""

    def _fn(self):
        from app.agents.react_agent import _build_active_doc_section
        return _build_active_doc_section

    def test_material_section_contains_url(self):
        fn = self._fn()
        section = fn(FAKE_MATERIAL_DOC)
        assert FAKE_MATERIAL_URL in section
        assert "read_material_pdf" in section
        assert "Data Structure" in section
        assert "Lecture One (DL.pdf)" in section

    def test_material_section_warns_against_regulation(self):
        fn = self._fn()
        section = fn(FAKE_MATERIAL_DOC)
        assert "read_regulation_pdf" in section
        # The rule says "لا تستخدم" (don't use) read_regulation_pdf
        assert "لا تستخدم" in section

    def test_regulation_section(self):
        fn = self._fn()
        section = fn(FAKE_REGULATION_DOC)
        assert "regulation" in section
        assert "read_regulation_pdf" in section

    def test_empty_doc_returns_empty(self):
        fn = self._fn()
        assert fn({}) == ""
        assert fn(None) == ""

    def test_material_without_url_returns_empty(self):
        fn = self._fn()
        doc = {**FAKE_MATERIAL_DOC, "file_url": ""}
        assert fn(doc) == ""


# ── Integration: material listed then summarize ───────────────────────────────

class TestMaterialListedThenSummarize:
    """
    Simulate the full flow:
    1. User asks "ايه المواد الي عندي؟" → agent lists materials, DL.pdf saved as active_doc
    2. User says "لخصه" → message should have CONTEXT prefix injected pointing to DL.pdf
    """

    @pytest.mark.asyncio
    async def test_followup_gets_file_url_prefix(self):
        from app.agents.react_agent import _inject_followup_context

        # Step 1: active_doc was already saved (simulated by having the dict in memory)
        # Step 2: follow-up message arrives
        result = _inject_followup_context("لخصه", FAKE_MATERIAL_DOC)

        assert FAKE_MATERIAL_URL in result
        assert "read_material_pdf" in result
        assert "لخصه" in result
        # Prefix should come BEFORE the original message
        assert result.index(FAKE_MATERIAL_URL) < result.index("لخصه")

    @pytest.mark.asyncio
    async def test_followup_quiz_gets_file_url_prefix(self):
        from app.agents.react_agent import _inject_followup_context

        result = _inject_followup_context("اعمل امتحان", FAKE_MATERIAL_DOC)
        assert FAKE_MATERIAL_URL in result
        assert "read_material_pdf" in result

    @pytest.mark.asyncio
    async def test_regulation_context_does_not_bleed_into_material(self):
        """
        If active_doc is a regulation, follow-up "لخصه" should NOT inject
        a read_material_pdf context (no crossover).
        """
        from app.agents.react_agent import _inject_followup_context

        result = _inject_followup_context("لخصه", FAKE_REGULATION_DOC)
        # Should be unchanged — no injection for regulation doc type
        assert result == "لخصه"
        assert "read_material_pdf" not in result
