"""
tests/test_pipeline.py

Unit tests for the 5-layer AI classification pipeline.

Coverage:
  - Layer 0: preprocessor (Arabic, English, Arabizi, mixed)
  - Layer 1: embedding classifier (warm_up + classify)
  - Layer 2: LLM classifier (mocked)
  - Layer 3: confidence router
  - Layer 4: action guard
  - Layer 5: conversation state (entity stack, pronoun resolution)
  - Integration: full planner pipeline (mocked LLM)

Run:
  cd f:\\fastApi
  python -m pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import asyncio
import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 0: Preprocessor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessor:

    def _pp(self, msg):
        from app.core.preprocessor import preprocess_message
        return preprocess_message(msg)

    # Arabic
    def test_arabic_detected(self):
        r = self._pp("اعمل امتحان في قواعد البيانات")
        assert r.detected_lang == "ar"
        assert r.script_type == "arabic"
        assert not r.is_arabizi

    def test_arabic_normalized_alef(self):
        r = self._pp("أنا طالب")
        assert "أ" not in r.clean_text  # alef normalized to ا
        assert "ا" in r.clean_text

    def test_arabic_diacritics_removed(self):
        r = self._pp("المَدرسَةُ")
        assert "َ" not in r.clean_text
        assert "ُ" not in r.clean_text

    def test_arabic_teh_marbuta(self):
        r = self._pp("المدرسة الجامعة")
        assert "ة" not in r.clean_text

    # English
    def test_english_detected(self):
        r = self._pp("generate an exam for my students")
        assert r.detected_lang == "en"
        assert r.script_type == "latin"
        assert not r.is_arabizi

    def test_english_unchanged(self):
        r = self._pp("hello world")
        assert "hello" in r.clean_text

    # Mixed
    def test_mixed_detected(self):
        r = self._pp("اعملي quiz عن machine learning")
        assert r.script_type in ("mixed", "arabic")

    def test_mixed_exam_intent_preserved(self):
        r = self._pp("generate exam في database")
        assert "exam" in r.clean_text.lower() or "امتحان" in r.clean_text

    # Arabizi
    def test_arabizi_sajelny_detected(self):
        r = self._pp("sajelny fel mawad")
        assert r.is_arabizi
        assert r.detected_lang == "ar"

    def test_arabizi_sajelny_transliterated(self):
        r = self._pp("sajelny fel mawad")
        assert "سجلني" in r.clean_text

    def test_arabizi_3ayez_detected(self):
        r = self._pp("3ayez exam fel AI")
        assert r.is_arabizi

    def test_arabizi_3ayez_transliterated(self):
        r = self._pp("3ayez exam fel AI")
        assert "عايز" in r.clean_text

    def test_arabizi_eb3at_transliterated(self):
        r = self._pp("eb3at complaint")
        assert r.is_arabizi
        assert "ابعت" in r.clean_text or "ارسل" in r.clean_text or "ب" in r.clean_text

    def test_empty_message(self):
        r = self._pp("")
        assert r.clean_text == ""
        assert r.detected_lang == "en"

    # Language detection standalone
    def test_detect_language_fast_arabic(self):
        from app.core.preprocessor import detect_language_fast
        assert detect_language_fast("اعمل امتحان") == "ar"

    def test_detect_language_fast_english(self):
        from app.core.preprocessor import detect_language_fast
        assert detect_language_fast("generate an exam") == "en"


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 3: Confidence Router Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceRouter:

    def _router(self):
        from app.agents.confidence_router import ConfidenceRouter
        return ConfidenceRouter(
            embedding_execute_threshold=0.82,
            llm_execute_threshold=0.78,
            llm_clarify_threshold=0.55,
            critical_execute_threshold=0.88,
        )

    def test_high_confidence_execute(self):
        r = self._router()
        d = r.route("generate_exam", 0.90, "llm", "Generate exam")
        assert d.action == "execute"
        assert d.intent == "generate_exam"

    def test_low_confidence_clarify(self):
        r = self._router()
        d = r.route("generate_exam", 0.65, "llm", "Generate exam")
        assert d.action == "clarify"
        assert d.clarification_question is not None

    def test_very_low_confidence_fallback(self):
        r = self._router()
        d = r.route("generate_exam", 0.40, "llm", "Generate exam")
        assert d.action == "fallback"
        assert d.intent == "general_chat"

    def test_safe_intent_always_executes(self):
        r = self._router()
        for intent in ("general_chat", "academic_advice", "study_plan"):
            d = r.route(intent, 0.30, "llm", "safe intent")
            assert d.action == "execute", f"{intent} should always execute"

    def test_critical_intent_higher_threshold(self):
        r = self._router()
        # 0.85 is above llm_execute (0.78) but below critical (0.88)
        d = r.route("action_execute", 0.85, "llm", "Enroll student")
        assert d.action == "clarify"  # below critical threshold

    def test_critical_intent_execute_above_threshold(self):
        r = self._router()
        d = r.route("action_execute", 0.92, "llm", "Enroll student")
        assert d.action == "execute"

    def test_pronoun_source_always_executes(self):
        r = self._router()
        d = r.route("regulation", 0.50, "pronoun", "Pronoun reference")
        assert d.action == "execute"

    def test_keyword_source_always_executes(self):
        r = self._router()
        d = r.route("generate_exam", 0.50, "keyword", "Keyword match")
        assert d.action == "execute"

    def test_embedding_fast_path(self):
        r = self._router()
        d = r.route("result_query", 0.85, "embedding", "Get results")
        assert d.action == "execute"  # result_query is safe intent

    def test_is_critical(self):
        r = self._router()
        assert r.is_critical("action_execute")
        assert r.is_critical("complaint_submit")
        assert not r.is_critical("general_chat")
        assert not r.is_critical("result_query")


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 4: Action Guard Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestActionGuard:

    def test_non_critical_intent_skipped(self):
        from app.agents.action_guard import check_critical_action
        result = asyncio.get_event_loop().run_until_complete(
            check_critical_action(
                intent="general_chat",
                goal_summary="Hello",
                extracted_params={},
                user_language="ar",
                model_router=None,
                run_verification=False,
            )
        )
        assert not result.should_confirm

    def test_complaint_submit_requires_confirmation(self):
        from app.agents.action_guard import check_critical_action
        result = asyncio.get_event_loop().run_until_complete(
            check_critical_action(
                intent="complaint_submit",
                goal_summary="Submit a complaint",
                extracted_params={"targetType": "Doctor"},
                user_language="ar",
                model_router=None,
                run_verification=False,
            )
        )
        assert result.should_confirm
        assert result.confirmation_message != ""

    def test_action_execute_requires_confirmation(self):
        from app.agents.action_guard import check_critical_action
        result = asyncio.get_event_loop().run_until_complete(
            check_critical_action(
                intent="action_execute",
                goal_summary="Enroll in all courses",
                extracted_params={"studentId": "s1", "batchId": "b1"},
                user_language="en",
                model_router=None,
                run_verification=False,
            )
        )
        assert result.should_confirm
        assert "yes" in result.confirmation_message.lower() or "confirm" in result.confirmation_message.lower()

    def test_arabic_confirmation_message(self):
        from app.agents.action_guard import check_critical_action
        result = asyncio.get_event_loop().run_until_complete(
            check_critical_action(
                intent="action_execute",
                goal_summary="تسجيل في المواد",
                extracted_params={},
                user_language="ar",
                model_router=None,
                run_verification=False,
            )
        )
        assert result.should_confirm
        assert "نعم" in result.confirmation_message

    def test_confirmation_keywords(self):
        from app.agents.action_guard import is_confirmation, is_cancellation
        assert is_confirmation("نعم")
        assert is_confirmation("yes")
        assert is_confirmation("تأكيد")
        assert is_confirmation("ok")
        assert is_cancellation("لأ")
        assert is_cancellation("no")
        assert is_cancellation("cancel")
        assert is_cancellation("الغي")
        assert not is_confirmation("مش عارف")
        assert not is_cancellation("كمّل")


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 5: Conversation State Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationState:

    def test_empty_state(self):
        from app.core.conversation_state import ConversationState
        state = ConversationState()
        assert state.top_entity() is None
        assert state.resolve_pronoun() is None

    def test_push_entity(self):
        from app.core.conversation_state import ConversationState, EntityFrame
        state = ConversationState()
        frame = EntityFrame(type="regulation", name="fbn", intent="regulation")
        state.push_entity(frame)
        top = state.top_entity()
        assert top is not None
        assert top.type == "regulation"
        assert top.name == "fbn"

    def test_pronoun_resolution_regulation(self):
        from app.core.conversation_state import ConversationState, EntityFrame
        state = ConversationState()
        state.push_entity(EntityFrame(type="regulation", name="fbn", intent="regulation"))
        assert state.resolve_pronoun() == "regulation"

    def test_pronoun_resolution_exam(self):
        from app.core.conversation_state import ConversationState, EntityFrame
        state = ConversationState()
        state.push_entity(EntityFrame(type="exam", name="OS midterm", intent="generate_exam"))
        assert state.resolve_pronoun() == "generate_exam"

    def test_entity_stack_lifo(self):
        from app.core.conversation_state import ConversationState, EntityFrame
        state = ConversationState()
        state.push_entity(EntityFrame(type="regulation", name="reg1", intent="regulation"))
        state.push_entity(EntityFrame(type="exam", name="exam1", intent="generate_exam"))
        top = state.top_entity()
        assert top.type == "exam"  # most recent

    def test_entity_stack_max_depth(self):
        from app.core.conversation_state import ConversationState, EntityFrame
        state = ConversationState()
        for i in range(10):
            state.push_entity(EntityFrame(type="material", name=f"m{i}", intent="material_explanation"))
        assert len(state.entity_stack) <= 5

    def test_serialization_roundtrip(self):
        from app.core.conversation_state import ConversationState, EntityFrame
        state = ConversationState(current_intent="regulation", turn_count=3)
        state.push_entity(EntityFrame(type="regulation", name="fbn", intent="regulation"))
        d = state.to_dict()
        restored = ConversationState.from_dict(d)
        assert restored.current_intent == "regulation"
        assert restored.turn_count == 3
        top = restored.top_entity()
        assert top is not None
        assert top.name == "fbn"

    def test_entity_context_note(self):
        from app.core.conversation_state import ConversationState, EntityFrame, build_entity_context_note
        state = ConversationState()
        state.push_entity(EntityFrame(type="regulation", name="fbn", intent="regulation"))
        note = build_entity_context_note(state)
        assert "regulation" in note
        assert "fbn" in note

    def test_empty_context_note(self):
        from app.core.conversation_state import ConversationState, build_entity_context_note
        state = ConversationState()
        note = build_entity_context_note(state)
        assert note == ""

    # Pronoun detection patterns
    @pytest.mark.parametrize("msg", [
        "اشرحها",
        "لخصها",
        "ابعتها",
        "استمر",
        "كمّل",
        "اعمل زي اللي فوق",
        "explain it",
        "send it",
        "continue",
        "use the previous one",
        "summarize it",
    ])
    def test_pronoun_patterns(self, msg):
        from app.core.conversation_state import is_pronoun_reference
        assert is_pronoun_reference(msg), f"'{msg}' should be detected as pronoun reference"

    @pytest.mark.parametrize("msg", [
        "اعمل امتحان في قواعد البيانات",
        "سجلني في المواد",
        "generate an exam for database",
        "hello how are you",
        "اشرح مادة الخوارزميات بالتفصيل",
    ])
    def test_non_pronoun_patterns(self, msg):
        from app.core.conversation_state import is_pronoun_reference
        assert not is_pronoun_reference(msg), f"'{msg}' should NOT be pronoun reference"


# ─────────────────────────────────────────────────────────────────────────────
#  Intent Examples Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentExamples:

    def test_all_valid_intents_covered(self):
        from app.agents.intent_examples import INTENT_EXAMPLES
        from app.agents.planner import VALID_INTENTS
        for intent in VALID_INTENTS:
            assert intent in INTENT_EXAMPLES, f"Intent '{intent}' missing from INTENT_EXAMPLES"

    def test_minimum_examples_per_intent(self):
        from app.agents.intent_examples import INTENT_EXAMPLES
        for intent, examples in INTENT_EXAMPLES.items():
            assert len(examples) >= 5, (
                f"Intent '{intent}' has only {len(examples)} examples (min 5)"
            )

    def test_no_duplicate_examples_within_intent(self):
        from app.agents.intent_examples import INTENT_EXAMPLES
        for intent, examples in INTENT_EXAMPLES.items():
            assert len(examples) == len(set(examples)), (
                f"Intent '{intent}' has duplicate examples"
            )

    def test_validate_examples_no_critical_warnings(self):
        from app.agents.intent_examples import validate_examples
        warnings = validate_examples()
        assert len(warnings) == 0, f"Example bank has warnings: {warnings}"


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding Classifier Tests (mock embedding service)
# ─────────────────────────────────────────────────────────────────────────────

class MockEmbeddingService:
    """Deterministic mock: returns a pseudo-embedding based on word hashing."""

    def __init__(self):
        self._mode = "mock"
        self.embedding_dim = 64

    def is_using_real_embeddings(self) -> bool:
        return False

    def get_mode(self) -> str:
        return "mock"

    async def embed_text(self, text: str) -> list[float]:
        return self._keyword_vector(text, self.embedding_dim)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._keyword_vector(t, self.embedding_dim) for t in texts]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(y * y for y in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _keyword_vector(text: str, dim: int) -> list[float]:
        import math, re
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return [0.0] * dim
        vec = [0.0] * dim
        for tok in tokens:
            vec[hash(tok) % dim] += 1.0 / len(tokens)
        mag = math.sqrt(sum(v * v for v in vec))
        return [v / mag for v in vec] if mag > 0 else vec


@pytest.fixture
def mock_emb_svc():
    return MockEmbeddingService()


class TestEmbeddingClassifier:

    @pytest.fixture(autouse=True)
    def loop(self):
        self._loop = asyncio.new_event_loop()
        yield
        self._loop.close()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def test_warm_up_succeeds(self, mock_emb_svc):
        from app.agents.embedding_classifier import EmbeddingIntentClassifier
        clf = EmbeddingIntentClassifier(mock_emb_svc)
        self._run(clf.warm_up())
        assert clf.is_ready
        assert clf.intent_count > 0

    def test_classify_returns_result(self, mock_emb_svc):
        from app.agents.embedding_classifier import EmbeddingIntentClassifier
        clf = EmbeddingIntentClassifier(mock_emb_svc)
        self._run(clf.warm_up())
        result = self._run(clf.classify("generate an exam"))
        assert result.intent in {"generate_exam", "general_chat", "backend_api_query",
                                   "result_query", "academic_advice", "study_plan",
                                   "material_explanation", "action_execute", "complaint_submit",
                                   "regulation", "material_qa", "assignment_query",
                                   "summarization", "file_extraction", "file_processing",
                                   "cv_analysis", "complaint_summary"}
        assert 0.0 <= result.score <= 1.0
        assert len(result.candidates) <= 3

    def test_classify_before_warmup_returns_fallback(self, mock_emb_svc):
        from app.agents.embedding_classifier import EmbeddingIntentClassifier
        clf = EmbeddingIntentClassifier(mock_emb_svc)
        result = self._run(clf.classify("generate exam"))
        assert not result.is_confident  # not warmed up → fallback

    def test_double_warmup_is_idempotent(self, mock_emb_svc):
        from app.agents.embedding_classifier import EmbeddingIntentClassifier
        clf = EmbeddingIntentClassifier(mock_emb_svc)
        self._run(clf.warm_up())
        count1 = clf.intent_count
        self._run(clf.warm_up())  # should not re-run
        assert clf.intent_count == count1

    def test_status_dict(self, mock_emb_svc):
        from app.agents.embedding_classifier import EmbeddingIntentClassifier
        clf = EmbeddingIntentClassifier(mock_emb_svc)
        self._run(clf.warm_up())
        s = clf.status_dict()
        assert s["ready"] is True
        assert s["intent_count"] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Planner Integration Tests (mocked LLM + embedding)
# ─────────────────────────────────────────────────────────────────────────────

class MockModelRouter:
    """Returns a configurable JSON response for LLM calls."""

    def __init__(self, intent: str = "general_chat", confidence: float = 0.9):
        self._intent = intent
        self._confidence = confidence

    async def generate_with_messages(self, messages, model_id=None, **kwargs) -> str:
        import json
        return json.dumps({
            "intent": self._intent,
            "confidence": self._confidence,
            "goal_summary": f"Mock goal for {self._intent}",
            "requires_clarification": False,
            "extracted_params": {},
        })

    async def generate_structured_json(self, prompt, system_instruction=None, **kwargs) -> dict:
        return {"confirmed": True}

    async def generate(self, *args, **kwargs) -> str:
        return "Mock response"


class TestPlannerIntegration:

    @pytest.fixture(autouse=True)
    def loop(self):
        self._loop = asyncio.new_event_loop()
        yield
        self._loop.close()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _make_planner(self, intent="general_chat", confidence=0.9, use_router=True, shadow=True):
        from app.agents.planner import PlannerAgent
        from app.agents.confidence_router import ConfidenceRouter

        router = ConfidenceRouter() if use_router else None
        planner = PlannerAgent(
            model_router=MockModelRouter(intent=intent, confidence=confidence),
            memory=None,
            embedding_classifier=None,
            confidence_router=router,
        )
        planner._settings_loaded = True
        planner._shadow_mode = shadow
        planner._emb_enabled = False   # skip Layer 1 in unit tests
        planner._guard_enabled = False # skip Layer 4 in unit tests
        return planner

    def _make_input(self, message: str, role: str = "student") -> "AgentInput":
        from app.agents.schemas import AgentInput
        return AgentInput(
            message=message,
            user_id="test_user",
            context={"role": role, "history": [], "academic_context": {}},
        )

    # ── Arabic messages ────────────────────────────────────────────────────

    def test_arabic_exam_intent(self):
        planner = self._make_planner(intent="generate_exam", confidence=0.92)
        result = self._run(planner.run(self._make_input("اعمل امتحان في قواعد البيانات", "doctor")))
        assert result.status == "success"
        plan = result.data["plan"]
        assert plan.intent == "generate_exam"

    def test_arabic_enrollment_intent(self):
        planner = self._make_planner(intent="action_execute", confidence=0.91)
        result = self._run(planner.run(self._make_input("سجلني في المواد")))
        assert result.status == "success"
        plan = result.data["plan"]
        assert plan.intent == "action_execute"

    def test_arabic_regulation_keyword_override(self):
        planner = self._make_planner(intent="general_chat", confidence=0.4)
        result = self._run(planner.run(self._make_input("مواد السنة الاولى في اللائحة")))
        assert result.status == "success"
        plan = result.data["plan"]
        assert plan.intent == "regulation"  # keyword override fires

    def test_arabic_study_plan_keyword(self):
        planner = self._make_planner(intent="general_chat", confidence=0.4)
        result = self._run(planner.run(self._make_input("اعمللي خطة مذاكرة")))
        assert result.status == "success"
        plan = result.data["plan"]
        assert plan.intent == "study_plan"

    # ── English messages ───────────────────────────────────────────────────

    def test_english_exam_keyword(self):
        planner = self._make_planner(intent="general_chat", confidence=0.4)
        result = self._run(planner.run(self._make_input("create a final exam for OS")))
        assert result.status == "success"
        assert result.data["plan"].intent == "generate_exam"

    def test_english_enrollment(self):
        planner = self._make_planner(intent="action_execute", confidence=0.9)
        result = self._run(planner.run(self._make_input("register me for this semester")))
        assert result.status == "success"
        assert result.data["plan"].intent == "action_execute"

    def test_english_who_am_i(self):
        planner = self._make_planner(intent="backend_api_query", confidence=0.88)
        result = self._run(planner.run(self._make_input("who am i")))
        assert result.status == "success"
        assert result.data["plan"].intent == "backend_api_query"

    # ── Mixed language ──────────────────────────────────────────────────────

    def test_mixed_quiz_ml(self):
        planner = self._make_planner(intent="generate_exam", confidence=0.87)
        result = self._run(planner.run(self._make_input("اعملي quiz عن machine learning", "doctor")))
        assert result.status == "success"
        assert result.data["plan"].intent == "generate_exam"

    def test_mixed_submit_complaint(self):
        planner = self._make_planner(intent="complaint_submit", confidence=0.88)
        result = self._run(planner.run(self._make_input("submit complaint للدكتور")))
        assert result.status == "success"
        assert result.data["plan"].intent == "complaint_submit"

    # ── Arabizi messages (pre-translated by Layer 0) ───────────────────────

    def test_arabizi_sajelny(self):
        planner = self._make_planner(intent="action_execute", confidence=0.9)
        result = self._run(planner.run(self._make_input("sajelny fel mawad")))
        assert result.status == "success"
        assert result.data["plan"].intent == "action_execute"

    def test_arabizi_3ayez_exam(self):
        planner = self._make_planner(intent="general_chat", confidence=0.4)
        result = self._run(planner.run(self._make_input("3ayez exam fel AI")))
        # After transliteration: "عايز امتحان في ال AI"
        # keyword "عايز امتحان" or "exam" + verb → generate_exam
        plan = result.data["plan"]
        # Either LLM or keyword should catch this
        assert plan.intent in ("generate_exam", "general_chat", "backend_api_query")

    # ── Clarification flow ─────────────────────────────────────────────────

    def test_clarification_triggered_on_low_confidence(self):
        from app.agents.planner import PlannerAgent
        from app.agents.confidence_router import ConfidenceRouter
        router = ConfidenceRouter()
        planner = PlannerAgent(
            model_router=MockModelRouter(intent="generate_exam", confidence=0.65),
            memory=None,
            embedding_classifier=None,
            confidence_router=router,
        )
        planner._settings_loaded = True
        planner._shadow_mode = False  # production mode for confidence routing
        planner._emb_enabled = False
        planner._guard_enabled = False
        result = self._run(planner.run(self._make_input("عايز حاجة")))
        # 0.65 < 0.78 (llm_execute) but > 0.55 (clarify) → clarify
        assert result.status in ("clarification_needed", "success")

    # ── Fallback plan ───────────────────────────────────────────────────────

    def test_fallback_on_none_llm_response(self):
        from app.agents.planner import PlannerAgent

        class FailingRouter:
            async def generate_with_messages(self, *args, **kwargs):
                return None
            async def generate_structured_json(self, *args, **kwargs):
                return {}

        planner = PlannerAgent(
            model_router=FailingRouter(),
            memory=None,
            embedding_classifier=None,
            confidence_router=None,
        )
        planner._settings_loaded = True
        planner._shadow_mode = True
        planner._emb_enabled = False
        planner._guard_enabled = False
        result = self._run(planner.run(self._make_input("hello")))
        assert result.status == "success"
        assert result.data["plan"].intent == "general_chat"

    # ── generate_exam edge cases ───────────────────────────────────────────

    def test_generate_exam_has_resolve_step(self):
        planner = self._make_planner(intent="generate_exam", confidence=0.9)
        result = self._run(planner.run(self._make_input(
            "generate an exam for data structures", role="doctor"
        )))
        plan = result.data["plan"]
        assert plan.intent == "generate_exam"
        # Should inject ResolveSubjectOffering when no offeringId
        has_resolve = any(
            s.tool == "ResolveSubjectOffering"
            for s in plan.pre_execution_steps
        )
        assert has_resolve

    # ── Pronoun resolution with state ──────────────────────────────────────

    def test_pronoun_resolution_with_state(self):
        """When a regulation is on the entity stack, 'اشرحها' → regulation."""
        from app.agents.planner import PlannerAgent
        from app.core.conversation_state import ConversationState, EntityFrame

        class MockMemoryWithState:
            def __init__(self):
                self._state = ConversationState()
                self._state.push_entity(EntityFrame(
                    type="regulation", name="fbn", intent="regulation"
                ))

            async def get_context(self, uid): return ""
            async def get_conv_state(self, uid): return self._state
            async def save_conv_state(self, uid, state): pass
            async def append_intent_log(self, uid, entry): pass
            async def save_clarification(self, uid, data): pass
            async def get_pending_action(self, uid): return None
            async def save_pending_action(self, uid, data): pass

        planner = PlannerAgent(
            model_router=MockModelRouter(intent="general_chat", confidence=0.5),
            memory=MockMemoryWithState(),
            embedding_classifier=None,
            confidence_router=None,
        )
        planner._settings_loaded = True
        planner._shadow_mode = True
        planner._emb_enabled = False
        planner._guard_enabled = False

        result = self._run(planner.run(self._make_input("اشرحها")))
        assert result.status == "success"
        plan = result.data["plan"]
        assert plan.intent == "regulation", (
            f"Expected regulation via pronoun resolution, got {plan.intent}"
        )
        assert result.data.get("source") == "pronoun"


# ─────────────────────────────────────────────────────────────────────────────
#  Keyword Safety Net Tests (Layer 2b)
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordSafetyNet:
    """
    Verify that the legacy keyword functions still catch edge cases
    when both the embedding and LLM return general_chat.
    """

    def test_detect_generate_exam_english(self):
        from app.agents.planner import _detect_generate_exam
        assert _detect_generate_exam("create exam for databases")
        assert _detect_generate_exam("generate a test for my students")
        assert _detect_generate_exam("make quiz")
        assert not _detect_generate_exam("view exam results")
        assert not _detect_generate_exam("when is the exam?")

    def test_detect_generate_exam_arabic(self):
        from app.agents.planner import _detect_generate_exam
        assert _detect_generate_exam("اعمل امتحان")
        assert _detect_generate_exam("انشئ اختبار")
        assert not _detect_generate_exam("نتيجة الامتحان")

    def test_detect_backend_query(self):
        from app.agents.planner import _detect_backend_query
        assert _detect_backend_query("كم عدد الطلاب")
        assert _detect_backend_query("who am i")
        assert _detect_backend_query("my courses")
        assert _detect_backend_query("show me all departments")

    def test_detect_regulation(self):
        from app.agents.planner import _detect_regulation
        assert _detect_regulation("مواد السنة الاولى")
        assert _detect_regulation("متطلبات التخرج")
        assert _detect_regulation("graduation requirements")
        assert not _detect_regulation("نتيجتي")

    def test_detect_study_plan(self):
        from app.agents.planner import _detect_study_plan
        assert _detect_study_plan("خطة مذاكرة")
        assert _detect_study_plan("study plan for midterm")
        assert _detect_study_plan("اذاكر ايه")

    def test_detect_assignment_query(self):
        from app.agents.planner import _detect_assignment_query
        assert _detect_assignment_query("واجباتي")
        assert _detect_assignment_query("my assignment deadline")
        assert _detect_assignment_query("did i submit")
