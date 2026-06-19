"""
planner.py  —  PlannerAgent v5.0

5-Layer intent classification pipeline:

  Layer 0  Pre-Processor         — language detect, Arabizi transliterate, normalize
  Layer 1  Embedding Classifier  — cosine similarity against intent centroid bank
  Layer 2  LLM Function-Call     — compact prompt + JSON schema (confident score)
  Layer 3  Confidence Router     — execute / clarify / fallback
  Layer 4  Critical Action Guard — user confirmation for write operations
  Layer 5  Conversation State    — entity stack update, pronoun resolution

Backward-compatible changes:
  - PlannerAgent.__init__() accepts two new optional parameters:
      embedding_classifier  (EmbeddingIntentClassifier | None)
      confidence_router     (ConfidenceRouter | None)
    When None the corresponding layer is bypassed — old behavior preserved.
  - All existing keyword detection functions (_detect_generate_exam, etc.)
    are kept and now act as a final safety net (Layer 2b) AFTER the new
    LLM function-call classifier, replacing their former role as the only
    fallback.
  - Shadow mode: EMBEDDING_CLASSIFIER_SHADOW_MODE=true → embedding runs
    and logs but does NOT change the execution path.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional, Protocol

from pydantic import ValidationError

from app.agents.base_agent import BaseAgent
from app.agents.schemas import (
    AgentInput,
    AgentOutput,
    ExecutionPlan,
    ExamParams,
    PreExecutionStep,
)
from app.core.logging import logger
from app.prompts import render_prompt

# ── Valid intent catalogue (unchanged) ───────────────────────────────────────
VALID_INTENTS = {
    "general_chat",
    "summarization",
    "generate_exam",
    "result_query",
    "file_extraction",
    "complaint_submit",
    "complaint_summary",
    "file_processing",
    "cv_analysis",
    "academic_advice",
    "material_explanation",
    "backend_api_query",
    "material_qa",
    "regulation",
    "action_execute",
    "assignment_query",
    "study_plan",
    # ── AI Companion Platform intents ───────────────
    "academic_coach",      # personalized academic coaching with grade analysis
    "quiz_me",             # interactive quiz session
    "generate_flashcards", # create flashcard deck for a topic
    "generate_examples",   # generate practical examples
    "generate_exercises",  # generate practice exercises
    "progress_report",     # weekly/monthly progress report
    "learning_assistant",  # concept explanation with examples/exercises
    "doctor_analytics",    # doctor: class performance analytics
    "doctor_risk_students",   # doctor: list at-risk students
    "doctor_weak_topics",     # doctor: topic performance analysis
    "doctor_recommendations", # doctor: AI teaching recommendations
}

_FALLBACK_INTENT = "general_chat"

# ─────────────────────────────────────────────────────────────────────────────
#  Legacy Layer-2b keyword sets
#  These are preserved as a final safety net but their role has changed:
#  they now fire ONLY when BOTH the embedding and LLM classifiers returned
#  general_chat.  They are no longer the primary classification path.
# ─────────────────────────────────────────────────────────────────────────────

_EXAM_KEYWORDS_EN: frozenset[str] = frozenset({
    "generate exam", "create exam", "make exam", "build exam", "write exam",
    "prepare exam", "prepare test", "create test", "generate test",
    "make test", "build test", "write test", "exam for subject",
    "exam for course", "new exam", "draft exam", "design exam", "set exam",
    "set a test", "produce exam", "develop exam",
})
_EXAM_ACTION_VERBS_EN: frozenset[str] = frozenset({
    "create", "generate", "make", "build", "write",
    "prepare", "draft", "design", "produce", "develop", "compose", "set",
})
_EXAM_TARGET_WORDS_EN: frozenset[str] = frozenset({"exam", "test", "quiz", "assessment"})
_EXAM_KEYWORDS_AR: frozenset[str] = frozenset({
    "اعمل امتحان", "انشئ امتحان", "سوي امتحان", "حضّر امتحان",
    "اكتب امتحان", "امتحان لمادة", "عمل امتحان", "أنشئ امتحان",
    "صمّم امتحان", "عايز امتحان", "نعمل امتحان", "جهّز امتحان",
    "جهّز اختبار", "عمل اختبار", "انشئ اختبار", "طوّر امتحان",
    "اكتب اختبار", "حضر امتحان", "صمم امتحان",
})


def _detect_generate_exam(message: str) -> bool:
    import re
    msg = message.strip().lower()
    for kw in _EXAM_KEYWORDS_EN:
        if kw in msg:
            return True
    for kw in _EXAM_KEYWORDS_AR:
        if kw in msg:
            return True
    words = set(re.findall(r"\b\w+\b", msg))
    return bool(words & _EXAM_ACTION_VERBS_EN) and bool(words & _EXAM_TARGET_WORDS_EN)


def _detect_backend_query(message: str) -> bool:
    msg = message.strip().lower()
    _BACKEND_KEYWORDS = {
        "كم عدد", "كام بدرس", "كام طالب", "كام دكتور", "كام مادة",
        "عدد الدكاترة", "عدد الطلاب", "عدد المواد", "عدد الاقسام",
        "نسبة", "احصائيات", "إحصائيات", "تحليل", "تقرير",
        "مين هم", "قائمة", "اللي بيدرس", "اللي مسجل",
        "كليات", "الكليات", "دكاترة", "الدكاترة",
        "قسم", "اقسام", "الأقسام", "الاقسام",
        "طلاب", "الطلاب", "مواد", "المواد",
        "اسمي", "اسم", "انا مين", "أنا مين", "من انا", "من أنا",
        "معلوماتي", "بياناتي", "بروفايلي",
        "how many", "count of", "number of", "total students",
        "statistics", "analytics", "list of", "show me", "what are",
        "who am i", "my name", "my profile", "my info",
        "my courses", "my subjects", "my schedule", "my grades",
        "my gpa", "my results",
        # Schedule-specific keywords — must NOT fall to general_chat or study_plan
        "جدولي", "جدول دراسي", "جدول المحاضرات", "جدول الأسبوع", "جدول الاسبوع",
        "الجدول الدراسي", "محاضرات النهارده", "محاضرات بكرا",
        "موعد المحاضرة", "المحاضرة الجاية",
        "schedule this week", "class schedule", "lecture schedule",
        "timetable", "weekly schedule", "today's classes", "tomorrow's classes",
        "next lecture", "class today", "classes tomorrow",
    }
    return any(kw in msg for kw in _BACKEND_KEYWORDS)


_REGULATION_KEYWORDS: frozenset[str] = frozenset({
    "اشرح الليحه", "اشرح اللائحه", "اشرح اللائحة",
    "ايه الليحه", "ايه اللائحه", "ايه اللائحة",
    "محتوى اللائحة", "مواد سنة اولى", "مواد السنة الاولى",
    "مواد سنة تانية", "مواد السنة التانية", "مواد سنة ثانية",
    "مواد سنة تالتة", "مواد السنة التالتة", "مواد سنة ثالثة",
    "مواد سنة رابعة", "مواد السنة الرابعة",
    "مواد الترم الاول", "مواد الترم الثاني",
    "ساعات اللائحة", "متطلبات التخرج", "شروط التخرج",
    "الخطة الدراسية", "خطة الدراسة", "دليل الطالب",
    "الاليخه", "اللايحه", "اللايحة", "هذه اللائحه",
    "في الليحه", "في اللايحه", "في اللائحه",
    "جوه اللايحه", "جوا اللائحه", "مواد اللائحه",
    "explain the regulation", "what is in the regulation",
    "subjects in year one", "subjects in first year",
    "graduation requirements", "credit hours required",
    "study plan subjects", "curriculum subjects", "student handbook",
})


def _detect_regulation(message: str) -> bool:
    msg = message.strip().lower()
    return any(kw in msg for kw in _REGULATION_KEYWORDS)


_MATERIAL_QA_KEYWORDS: frozenset[str] = frozenset({
    "explain from lecture", "from the material", "what does the lecture say",
    "according to the course", "course material", "lecture notes",
    "from the lecture", "from the book", "from the textbook",
    "in the lecture", "in the material", "in the book",
    "من المحاضرة", "من المادة", "اشرح من", "في الكتاب",
    "في المحاضرة", "من الملزمة", "من الكتاب",
})


def _detect_material_qa(message: str) -> bool:
    msg = message.strip().lower()
    return any(kw in msg for kw in _MATERIAL_QA_KEYWORDS)


_ASSIGNMENT_KEYWORDS: frozenset[str] = frozenset({
    "واجب", "الواجب", "واجباتي", "واجبات", "تسليم", "التسليم",
    "موعد التسليم", "الموعد النهائي", "deadline",
    "سلمت", "لسه ما سلمتش", "هل سلمت", "حالة الواجب",
    "درجة الواجب", "الواجب الجاي", "الواجبات المتأخرة",
    "ايه الواجبات", "تفاصيل الواجب",
    "my assignment", "my assignments", "assignment deadline", "due date",
    "submitted assignment", "did i submit", "assignment status",
    "pending assignment", "overdue assignment", "late assignment",
    "assignment grade", "assignment feedback", "assignment requirements",
    "show me assignments", "list assignments",
})


def _detect_assignment_query(message: str) -> bool:
    msg = message.strip().lower()
    return any(kw in msg for kw in _ASSIGNMENT_KEYWORDS)


_STUDY_PLAN_KEYWORDS: frozenset[str] = frozenset({
    "خطة مذاكرة", "خطة دراسة", "خطة دراسية", "خطة للمذاكرة",
    "اعمللي خطة", "اعمل لي خطة", "عمل خطة", "خطة للأسبوع",
    "جدول مذاكرة", "جدول دراسة", "جدول للأسبوع", "جدول يومي",
    "كيف أذاكر", "كيف اذاكر", "ازاي اذاكر", "ازاي أذاكر",
    "أولوياتي", "اولوياتي", "رتب أولوياتي",
    "كيف أرفع معدلي", "كيف ارفع معدلي", "أرفع المعدل",
    "ارفع المعدل", "كيف أحسن معدلي",
    "كيف أنجح", "كيف انجح", "ماذا أذاكر",
    "خطة للامتحانات", "خطة الامتحانات", "خطة للميدتيرم",
    "خطة للاختبار", "استعداد للامتحان", "استعداد للميدتيرم",
    "كيف أستعد للامتحان", "كيف استعد للامتحان",
    "أذاكر إيه النهارده", "أذاكر ايه النهارده", "اذاكر ايه",
    "وقت المذاكرة", "توزيع وقت", "توزيع المذاكرة",
    "ساعدني في المذاكرة",
    "study plan", "study schedule", "study timetable",
    "revision plan", "revision schedule",
    "how to study", "how should i study", "what should i study",
    "study for midterm", "study for final", "study for exam",
    "prioritize my subjects", "raise my gpa", "improve my gpa",
    "how to pass", "study tips", "what to study today",
    "weekly plan", "daily plan", "exam prep", "exam preparation",
    "prepare for exam",
})


def _detect_study_plan(message: str) -> bool:
    msg = message.strip().lower()
    return any(kw in msg for kw in _STUDY_PLAN_KEYWORDS)


# ── Available backend tools (for system prompt) ───────────────────────────────
_AVAILABLE_TOOLS = [
    "ResolveSubjectOffering", "GetStudentResults", "GetStudentGrades",
    "GetGPASummary", "GetTranscript", "GetSchedule", "GetSubjectOfferings",
    "GetCourseEnrollments", "GenerateExam", "DistributeExam",
    "SubmitComplaint", "GetComplaints", "GetStudentAcademicSummary",
    "BulkCreateStudents", "BulkUploadGrades", "GetMaterials",
]

# ── Compact LLM classifier system prompt (Layer 2) ───────────────────────────
# Much shorter than the original 600-line prompt — the LLM is guided by
# the function schema to return a valid intent from the enum.
_COMPACT_CLASSIFICATION_PROMPT = """\
You are an intent classifier for a university AI assistant.

LANGUAGE RULE: Detect the user's language. Arabic/Egyptian dialect → use Arabic everywhere.
English → use English. Mixed → use Arabic. NEVER mix languages in a single response.

INTENTS (pick exactly one):
- general_chat: greeting, general question, no backend data needed
- backend_api_query: any data retrieval (grades, lists, my profile, counts, schedule, who am I)
- generate_exam: CREATE/GENERATE an exam or test — regardless of user role
- result_query: student asking about their GPA, grades, transcript, results
- complaint_submit: student submitting a complaint about a doctor/exam/grade
- complaint_summary: admin/doctor reviewing or summarizing complaints
- academic_advice: study advice, course recommendations, general standing queries
- study_plan: study schedule, revision plan, exam prep, what to study today/this week
- material_explanation: explain/summarize a course or subject's material content
- material_qa: answer a question grounded in specific course material / lecture notes
- regulation: questions about the academic regulation PDF (subjects per year, graduation)
- action_execute: user wants to DO something (enroll, register, create entity)
- assignment_query: questions about assignments, deadlines, submission status
- file_extraction: extract info from an uploaded file (single file)
- summarization: summarize a document or text block
- file_processing: bulk upload (Excel grades/students)
- cv_analysis: analyze/review a student's CV or resume

KEY RULES:
1. backend_api_query: use for identity ("who am I", "my name") — NEVER answer from training data.
2. generate_exam: use whenever user wants to CREATE an exam. RBAC is enforced elsewhere.
3. action_execute: enrollment triggers ("سجلني", "register me") — NOT backend_api_query.
4. study_plan vs academic_advice: "make a study schedule" = study_plan; "how am I doing?" = academic_advice.
5. Pronoun resolution: if the message is short ("اشرحها", "continue") and prior context shows
   an active topic, match the intent of that topic.
6. confidence < 0.70 → set requires_clarification=true.

Return ONLY valid JSON matching the function schema. No markdown, no extra text.
"""

# ── LLM function-call schema ──────────────────────────────────────────────────
_CLASSIFICATION_FUNCTION = {
    "name": "classify_intent",
    "description": "Classify the user's intent and extract parameters",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": sorted(VALID_INTENTS),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in this classification (0.0–1.0)",
            },
            "goal_summary": {
                "type": "string",
                "description": "One sentence: what does the user want?",
            },
            "requires_clarification": {
                "type": "boolean",
                "description": "True if the request is ambiguous",
            },
            "clarification_question": {
                "type": "string",
                "description": "Question to ask if requires_clarification=true",
            },
            "extracted_params": {
                "type": "object",
                "description": "Any parameters: subjectName, targetType, etc.",
            },
        },
        "required": ["intent", "confidence", "goal_summary"],
    },
}

# ── Legacy inline system prompt (fallback when file is missing) ───────────────
def _get_system_prompt() -> str:
    """Load planner system prompt; fall back to compact inline version."""
    try:
        return render_prompt("planner_system", tools=", ".join(_AVAILABLE_TOOLS))
    except Exception as exc:
        logger.warning("PlannerAgent: prompt file failed — using compact inline: %s", exc)
        return _COMPACT_CLASSIFICATION_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
#  MemoryStore protocol (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryStore(Protocol):
    async def get_context(self, user_id: str | None) -> str: ...
    async def get_conv_state(self, user_id: str) -> Optional[object]: ...
    async def save_conv_state(self, user_id: str, state: object) -> None: ...
    async def append_intent_log(self, user_id: str, entry: dict) -> None: ...
    async def save_clarification(self, user_id: str, data: dict) -> None: ...
    async def get_pending_action(self, user_id: str) -> Optional[dict]: ...
    async def save_pending_action(self, user_id: str, data: dict) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
#  PlannerAgent v5.0
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    5-layer intent classification pipeline.

    New optional parameters (backward compatible — None = bypass layer):
      embedding_classifier  — EmbeddingIntentClassifier (Layer 1)
      confidence_router     — ConfidenceRouter (Layer 3)

    Settings-driven behavior:
      EMBEDDING_CLASSIFIER_SHADOW_MODE = True  → log only, don't change routing
      EMBEDDING_CLASSIFIER_SHADOW_MODE = False → full fast path active
      ACTION_GUARD_ENABLED = True              → Layer 4 active
    """

    def __init__(
        self,
        model_router,
        ranker=None,
        memory: Optional[MemoryStore] = None,
        embedding_classifier=None,
        confidence_router=None,
    ):
        self.model_router = model_router
        self.ranker = ranker
        self.memory = memory
        self.embedding_classifier = embedding_classifier  # Layer 1
        self.confidence_router = confidence_router        # Layer 3

        # Load settings lazily
        self._settings_loaded = False
        self._shadow_mode = True
        self._emb_enabled = True
        self._guard_enabled = True

    def _ensure_settings(self) -> None:
        if self._settings_loaded:
            return
        try:
            from app.core.config import settings
            self._shadow_mode  = settings.EMBEDDING_CLASSIFIER_SHADOW_MODE
            self._emb_enabled  = settings.EMBEDDING_CLASSIFIER_ENABLED
            self._guard_enabled = settings.ACTION_GUARD_ENABLED
        except Exception:
            pass
        self._settings_loaded = True

    # ─────────────────────────────────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────────────────────────────────

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Full 5-layer classification pipeline.

        Layer 0  Pre-process message (language, Arabizi, normalize)
        Layer 1  Embedding classifier (fast path if confident)
        Layer 2  LLM function-call classifier (when embedding uncertain)
        Layer 2b Legacy keyword safety net (when both above return general_chat)
        Layer 3  Confidence router (execute / clarify / fallback)
        Layer 4  Critical action guard (confirmation for write operations)
        Layer 5  Conversation state update + entity stack

        Returns AgentOutput with data={"plan": ExecutionPlan, ...}
        """
        self._ensure_settings()
        t0 = time.perf_counter()
        logger.info("PlannerAgent v5: user_id=%s", agent_input.user_id)

        ctx            = agent_input.context or {}
        role           = ctx.get("role", "user")
        raw_history    = ctx.get("history", [])
        academic_ctx   = ctx.get("academic_context", {})

        # ── Layer 0: Pre-process ──────────────────────────────────────────
        from app.core.preprocessor import preprocess_message
        preprocessed = preprocess_message(agent_input.message)
        logger.debug(
            "PlannerAgent [L0]: lang=%s script=%s arabizi=%s",
            preprocessed.detected_lang, preprocessed.script_type, preprocessed.is_arabizi,
        )

        # ── Layer 5a: Load conversation state (before classification) ────
        conv_state = None
        if self.memory:
            try:
                conv_state = await self.memory.get_conv_state(agent_input.user_id)
            except Exception as exc:
                logger.warning("PlannerAgent: conv_state load failed — %s", exc)

        # Build entity context note for pronoun resolution
        entity_context_note = ""
        if conv_state is not None:
            from app.core.conversation_state import build_entity_context_note, is_pronoun_reference
            entity_context_note = build_entity_context_note(conv_state)

            # ── Pronoun / coreference fast path ───────────────────────────
            if is_pronoun_reference(agent_input.message):
                resolved_intent = conv_state.resolve_pronoun()
                if resolved_intent:
                    logger.info(
                        "PlannerAgent [L5-pronoun]: '%s' → intent=%r",
                        agent_input.message, resolved_intent,
                    )
                    plan = self._build_plan_from_params(
                        intent=resolved_intent,
                        goal_summary=f"Continuing: {conv_state.current_topic or resolved_intent}",
                        extracted_params={},
                        agent_input=agent_input,
                    )
                    await self._update_conv_state(
                        agent_input.user_id, conv_state, resolved_intent,
                        {}, "pronoun",
                    )
                    return AgentOutput(
                        status="success",
                        response=plan.goal_summary,
                        data={"plan": plan, "source": "pronoun", "confidence": 1.0},
                    )

        # ── Layer 1: Embedding classifier ────────────────────────────────
        emb_intent: Optional[str] = None
        emb_score:  float = 0.0
        emb_confident = False

        if self._emb_enabled and self.embedding_classifier is not None:
            try:
                emb_result = await self.embedding_classifier.classify(
                    message=agent_input.message,
                    clean_text=preprocessed.clean_text,
                )
                emb_intent    = emb_result.intent
                emb_score     = emb_result.score
                emb_confident = emb_result.is_confident

                logger.info(
                    "PlannerAgent [L1-emb]: intent=%r score=%.3f confident=%s latency=%.0fms",
                    emb_intent, emb_score, emb_confident, emb_result.latency_ms,
                )
            except Exception as exc:
                logger.warning("PlannerAgent [L1-emb]: error — %s", exc)

        # ── Decide whether to use fast path or call LLM ──────────────────
        # Fast path conditions:
        #   - Embedding is confident
        #   - Shadow mode is OFF
        #   - Intent is NOT critical (critical intents always go through LLM
        #     for parameter extraction before ActionGuard)
        from app.agents.action_guard import CRITICAL_INTENTS
        use_embedding_fast_path = (
            emb_confident
            and not self._shadow_mode
            and emb_intent not in CRITICAL_INTENTS
        )

        if use_embedding_fast_path:
            logger.info(
                "PlannerAgent [L1-fastpath]: skipping LLM → intent=%r score=%.3f",
                emb_intent, emb_score,
            )
            intent         = emb_intent
            confidence     = emb_score
            goal_summary   = f"User wants to: {(intent or '').replace('_', ' ')}"
            classification_source = "embedding"
            requires_clarification = False
            clarification_question = None
            extracted_params = {}
        else:
            # ── Layer 2: LLM function-call classifier ────────────────────
            intent, confidence, goal_summary, requires_clarification, \
                clarification_question, extracted_params = \
                await self._call_llm_classifier(
                    agent_input=agent_input,
                    preprocessed=preprocessed,
                    entity_context_note=entity_context_note,
                    history=raw_history,
                    role=role,
                    academic_ctx=academic_ctx,
                )
            classification_source = "llm"

        # ── Shadow-mode logging ──────────────────────────────────────────
        if self._shadow_mode and emb_intent is not None:
            await self._log_shadow_comparison(
                agent_input=agent_input,
                emb_intent=emb_intent,
                emb_score=emb_score,
                llm_intent=intent,
                llm_confidence=confidence,
                final_intent=intent,
                source=classification_source,
            )

        # ── Layer 2b: Legacy keyword safety net ──────────────────────────
        # Only fires when both Layer 1 + Layer 2 returned general_chat
        # AND the message matches a critical deterministic pattern.
        if intent == "general_chat":
            intent, classification_source = self._keyword_safety_net(
                message=agent_input.message,
                current_intent=intent,
                current_source=classification_source,
            )
            if classification_source == "keyword":
                goal_summary = f"Keyword-detected intent: {intent.replace('_', ' ')}"

        # Special: regulation override (deterministic, always fires for regulation questions)
        if _detect_regulation(agent_input.message) and intent not in ("regulation",):
            logger.info("PlannerAgent [L2b-keyword]: regulation override")
            intent = "regulation"
            classification_source = "keyword"
            goal_summary = "Read and answer from the academic regulation PDF."

        # ── Layer 3: Confidence router ───────────────────────────────────
        routing_decision = None
        if self.confidence_router is not None and not self._shadow_mode:
            routing_decision = self.confidence_router.route(
                intent=intent,
                confidence=confidence,
                source=classification_source,
                goal_summary=goal_summary,
                clarification_question=clarification_question,
            )
            logger.info(
                "PlannerAgent [L3-router]: action=%s intent=%r confidence=%.3f",
                routing_decision.action, routing_decision.intent, routing_decision.confidence,
            )

            if routing_decision.action == "clarify":
                # Save pending intent → user answers → resumes in agent.py
                if self.memory:
                    try:
                        await self.memory.save_clarification(agent_input.user_id, {
                            "pending_intent": routing_decision.intent,
                            "confidence": routing_decision.confidence,
                            "goal_summary": routing_decision.goal_summary,
                            "original_message": agent_input.message,
                        })
                    except Exception as exc:
                        logger.warning("PlannerAgent: save_clarification failed — %s", exc)
                return AgentOutput(
                    status="clarification_needed",
                    response=routing_decision.clarification_question,
                    data={
                        "pending_intent": routing_decision.intent,
                        "confidence": routing_decision.confidence,
                    },
                )

            if routing_decision.action == "fallback":
                intent = "general_chat"

            # Use router's (possibly adjusted) intent
            intent = routing_decision.intent

        # ── Layer 4: Critical action guard ───────────────────────────────
        if self._guard_enabled and intent in CRITICAL_INTENTS and not self._shadow_mode:
            guard_result = await self._run_action_guard(
                intent=intent,
                goal_summary=goal_summary,
                extracted_params=extracted_params,
                user_language=preprocessed.detected_lang,
                user_id=agent_input.user_id,
            )
            if guard_result is not None:
                # Confirmation required — save pending action and return
                return guard_result

        # ── Build ExecutionPlan ──────────────────────────────────────────
        plan = self._build_plan(
            intent=intent,
            goal_summary=goal_summary,
            extracted_params=extracted_params,
            agent_input=agent_input,
        )

        # ── Layer 5b: Update conversation state ──────────────────────────
        await self._update_conv_state(
            user_id=agent_input.user_id,
            conv_state=conv_state,
            intent=intent,
            extracted_params=extracted_params,
            source=classification_source,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "PlannerAgent v5: DONE intent=%r source=%s confidence=%.3f total=%.0fms",
            intent, classification_source, confidence, elapsed,
        )

        return AgentOutput(
            status="success",
            response=plan.goal_summary,
            data={
                "plan": plan,
                "source": classification_source,
                "confidence": confidence,
                "emb_intent": emb_intent,
                "emb_score": emb_score,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Layer 2: LLM function-call classifier
    # ─────────────────────────────────────────────────────────────────────

    async def _call_llm_classifier(
        self,
        agent_input: AgentInput,
        preprocessed,
        entity_context_note: str,
        history: list,
        role: str,
        academic_ctx: dict,
    ) -> tuple:
        """
        Call the LLM with a compact system prompt + function schema.

        Falls back to legacy free-form JSON generation when function calling
        is not available on the selected model.

        Returns:
          (intent, confidence, goal_summary, requires_clarification,
           clarification_question, extracted_params)
        """
        # Pull memory context
        memory_prefix = ""
        if self.memory:
            try:
                past = await self.memory.get_context(agent_input.user_id)
                if past:
                    memory_prefix = f"[Memory context]: {past}\n\n"
            except Exception as exc:
                logger.warning("PlannerAgent: memory lookup failed — %s", exc)

        # Build academic context injection for parameter auto-fill
        auto_fill_note = ""
        if academic_ctx:
            safe_keys = [
                "userId", "studentId", "courseId", "subjectOfferingId",
                "departmentId", "batchId", "collegeName", "departmentName", "profileId",
            ]
            relevant = {k: v for k, v in academic_ctx.items() if k in safe_keys and v}
            if relevant:
                auto_fill_note = (
                    f"\n[Context for auto-fill]: {json.dumps(relevant, ensure_ascii=False)}"
                )

        # Arabizi note (from pre-processor)
        arabizi_note = ""
        if preprocessed.is_arabizi:
            arabizi_note = f"\n{preprocessed.arabizi_note}"

        user_content = (
            f"{memory_prefix}"
            f"User role: {role}\n"
            f"User message: {agent_input.message}"
            f"{arabizi_note}"
            f"{auto_fill_note}"
            f"{entity_context_note}"
        )

        # Build structured history (last 4 pairs)
        history_turns: list[dict] = []
        for turn in history[-4:]:
            t_role = turn.get("role", "user")
            t_content = str(turn.get("content", ""))
            if t_role in ("user", "assistant") and t_content:
                history_turns.append({"role": t_role, "content": t_content})

        messages = [
            {"role": "system", "content": _get_system_prompt()},
            *history_turns,
            {"role": "user", "content": user_content},
        ]

        # Try function calling first
        try:
            raw = await self._call_with_function_schema(messages)
            if raw:
                return self._parse_function_result(raw, agent_input)
        except Exception as exc:
            logger.warning("PlannerAgent [L2-function-call]: failed — %s", exc)

        # Fallback: legacy JSON generation (old behavior)
        raw_json = await self._call_planner_model(history_turns, user_content)
        plan = self._parse_plan(raw_json, agent_input)
        return (
            plan.intent or _FALLBACK_INTENT,
            0.75,          # legacy path has no confidence score
            plan.goal_summary,
            False,
            None,
            {},
        )

    async def _call_with_function_schema(self, messages: list[dict]) -> Optional[dict]:
        """
        Attempt function-call style JSON generation.

        Uses generate_with_messages with response_format json_object and
        injects the function schema description into the system prompt.
        Falls back gracefully if the model doesn't support function calling.
        """
        try:
            from app.core.config import settings
            model_id = getattr(settings, "OPENROUTER_MODEL", "openai/gpt-4o-mini")
        except Exception:
            model_id = "openai/gpt-4o-mini"

        # Inject function schema description into system message
        schema_desc = json.dumps(_CLASSIFICATION_FUNCTION, ensure_ascii=False, indent=2)
        enhanced_messages = list(messages)
        enhanced_messages[0] = {
            "role": "system",
            "content": (
                messages[0]["content"]
                + f"\n\nReturn a JSON object matching this schema:\n{schema_desc}"
            ),
        }

        raw = await self.model_router.generate_with_messages(
            messages=enhanced_messages,
            model_id=model_id,
            response_format={"type": "json_object"},
        )
        if not raw:
            return None
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else None

    def _parse_function_result(
        self, raw: dict, agent_input: AgentInput
    ) -> tuple:
        """Parse the structured LLM response into classification fields."""
        intent = raw.get("intent", _FALLBACK_INTENT)
        if intent not in VALID_INTENTS:
            logger.warning("PlannerAgent [L2]: unknown intent %r → fallback", intent)
            intent = _FALLBACK_INTENT

        confidence = float(raw.get("confidence", 0.75))
        confidence = max(0.0, min(1.0, confidence))
        goal_summary = raw.get("goal_summary", f"Handle user request: {agent_input.message[:80]}")
        requires_clarification = bool(raw.get("requires_clarification", False))
        clarification_question = raw.get("clarification_question")
        extracted_params = raw.get("extracted_params") or {}
        if not isinstance(extracted_params, dict):
            extracted_params = {}

        logger.debug(
            "PlannerAgent [L2-parse]: intent=%r conf=%.3f clarify=%s",
            intent, confidence, requires_clarification,
        )
        return (
            intent, confidence, goal_summary,
            requires_clarification, clarification_question, extracted_params,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Layer 2b: Legacy keyword safety net
    # ─────────────────────────────────────────────────────────────────────

    def _keyword_safety_net(
        self, message: str, current_intent: str, current_source: str
    ) -> tuple[str, str]:
        """
        Apply legacy keyword detection ONLY when both L1 and L2 returned general_chat.

        Returns (intent, source) where source="keyword" if overridden.
        """
        if current_intent != "general_chat":
            return current_intent, current_source

        if _detect_generate_exam(message):
            logger.info("PlannerAgent [L2b-keyword]: generate_exam override")
            return "generate_exam", "keyword"
        if _detect_study_plan(message):
            logger.info("PlannerAgent [L2b-keyword]: study_plan override")
            return "study_plan", "keyword"
        if _detect_assignment_query(message):
            logger.info("PlannerAgent [L2b-keyword]: assignment_query override")
            return "assignment_query", "keyword"
        if _detect_material_qa(message):
            logger.info("PlannerAgent [L2b-keyword]: material_qa override")
            return "material_qa", "keyword"
        if _detect_backend_query(message):
            logger.info("PlannerAgent [L2b-keyword]: backend_api_query override")
            return "backend_api_query", "keyword"

        return current_intent, current_source

    # ─────────────────────────────────────────────────────────────────────
    #  Layer 4: Action Guard
    # ─────────────────────────────────────────────────────────────────────

    async def _run_action_guard(
        self,
        intent: str,
        goal_summary: str,
        extracted_params: dict,
        user_language: str,
        user_id: Optional[str],
    ) -> Optional[AgentOutput]:
        """
        Run the critical action guard.  Returns an AgentOutput if confirmation
        is required, or None if execution should proceed.
        """
        try:
            from app.agents.action_guard import check_critical_action
            guard = await check_critical_action(
                intent=intent,
                goal_summary=goal_summary,
                extracted_params=extracted_params,
                user_language=user_language,
                model_router=self.model_router,
                run_verification=(intent == "action_execute"),
            )

            if not guard.verified:
                # Verification rejected — treat as general_chat
                return None

            if guard.should_confirm:
                # Save pending action so agent can resume on confirmation
                if self.memory and user_id:
                    try:
                        await self.memory.save_pending_action(user_id, {
                            "intent": intent,
                            "goal_summary": goal_summary,
                            "params": extracted_params,
                        })
                    except Exception as exc:
                        logger.warning("PlannerAgent: save_pending_action failed — %s", exc)

                return AgentOutput(
                    status="confirmation_required",
                    response=guard.confirmation_message,
                    data={"pending_intent": intent, "awaiting_confirmation": True},
                )
        except Exception as exc:
            logger.error("PlannerAgent [L4-guard]: error — %s", exc)

        return None

    # ─────────────────────────────────────────────────────────────────────
    #  Layer 5: Conversation state update
    # ─────────────────────────────────────────────────────────────────────

    async def _update_conv_state(
        self,
        user_id: Optional[str],
        conv_state,
        intent: str,
        extracted_params: dict,
        source: str,
    ) -> None:
        """Push the resolved intent/entity onto the conversation state stack."""
        if not self.memory or not user_id:
            return
        try:
            from app.core.conversation_state import ConversationState, EntityFrame, ENTITY_INTENT_MAP

            if conv_state is None:
                conv_state = ConversationState()

            conv_state.current_intent = intent
            conv_state.turn_count += 1

            # Push to entity stack when a concrete artifact is produced
            entity_type = None
            entity_name = ""

            if intent == "regulation":
                entity_type = "regulation"
                entity_name = extracted_params.get("regulationName", "regulation")
            elif intent == "generate_exam":
                entity_type = "exam"
                entity_name = extracted_params.get("subjectName", "exam")
            elif intent in ("material_explanation", "material_qa"):
                entity_type = "material"
                entity_name = extracted_params.get("subjectName", "material")
            elif intent == "complaint_submit":
                entity_type = "complaint"
                entity_name = "complaint"
            elif intent == "study_plan":
                entity_type = "study_plan"
                entity_name = "study plan"
            elif intent == "result_query":
                entity_type = "result"
                entity_name = "results"
            elif intent == "assignment_query":
                entity_type = "assignment"
                entity_name = "assignment"

            if entity_type:
                frame = EntityFrame(
                    type=entity_type,
                    name=entity_name,
                    intent=ENTITY_INTENT_MAP.get(entity_type, intent),
                    params=extracted_params,
                )
                conv_state.push_entity(frame)
                conv_state.current_topic = f"{entity_type}:{entity_name}"

            await self.memory.save_conv_state(user_id, conv_state)
        except Exception as exc:
            logger.warning("PlannerAgent [L5]: conv_state update failed — %s", exc)

    # ─────────────────────────────────────────────────────────────────────
    #  Shadow mode logging
    # ─────────────────────────────────────────────────────────────────────

    async def _log_shadow_comparison(
        self,
        agent_input: AgentInput,
        emb_intent: str,
        emb_score: float,
        llm_intent: str,
        llm_confidence: float,
        final_intent: str,
        source: str,
    ) -> None:
        """Log embedding vs LLM comparison for calibration."""
        match = emb_intent == llm_intent
        if not match:
            logger.warning(
                "PlannerAgent [SHADOW]: MISMATCH emb=%r(%.3f) llm=%r(%.3f) final=%r "
                "msg=%.80r",
                emb_intent, emb_score, llm_intent, llm_confidence, final_intent,
                agent_input.message,
            )
        else:
            logger.info(
                "PlannerAgent [SHADOW]: MATCH emb=%r(%.3f) llm=%r(%.3f)",
                emb_intent, emb_score, llm_intent, llm_confidence,
            )

        if self.memory and agent_input.user_id:
            entry = {
                "ts":          datetime.now(timezone.utc).isoformat(),
                "message":     agent_input.message[:120],
                "emb_intent":  emb_intent,
                "emb_score":   round(emb_score, 4),
                "llm_intent":  llm_intent,
                "llm_conf":    round(llm_confidence, 4),
                "final":       final_intent,
                "source":      source,
                "match":       match,
            }
            try:
                await self.memory.append_intent_log(agent_input.user_id, entry)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    #  Plan builders
    # ─────────────────────────────────────────────────────────────────────

    def _build_plan(
        self,
        intent: str,
        goal_summary: str,
        extracted_params: dict,
        agent_input: AgentInput,
    ) -> ExecutionPlan:
        """Build an ExecutionPlan from classification results."""
        # Build plan using legacy method for backward compat
        raw = {
            "intent": intent,
            "goal_summary": goal_summary,
            "is_executable": True,
            "exam_params": None,
            "pre_execution_steps": [],
            "steps": [],
        }

        # Populate exam_params for generate_exam (always — extracted_params may be empty)
        if intent == "generate_exam":
            exam_fields = {
                "subjectName": extracted_params.get("subjectName"),
                "subjectOfferingId": extracted_params.get("subjectOfferingId"),
                "numberOfQuestions": extracted_params.get("numberOfQuestions", 10),
                "examType": extracted_params.get("examType", "midterm"),
                "variationMode": extracted_params.get("variationMode", "same_for_all"),
                "collegeName": extracted_params.get("collegeName"),
                "departmentName": extracted_params.get("departmentName"),
                "batchName": extracted_params.get("batchName"),
            }
            try:
                raw["exam_params"] = ExamParams(**{k: v for k, v in exam_fields.items() if v is not None})
            except Exception:
                raw["exam_params"] = ExamParams(numberOfQuestions=10, examType="midterm")

        plan = self._parse_plan(raw, agent_input)
        plan = self._ensure_resolve_step(plan)
        return plan

    def _build_plan_from_params(
        self,
        intent: str,
        goal_summary: str,
        extracted_params: dict,
        agent_input: AgentInput,
    ) -> ExecutionPlan:
        return self._build_plan(intent, goal_summary, extracted_params, agent_input)

    # ─────────────────────────────────────────────────────────────────────
    #  Legacy helpers (preserved for backward compat)
    # ─────────────────────────────────────────────────────────────────────

    async def _call_planner_model(
        self, history_turns: list[dict], user_content: str
    ) -> dict | None:
        """Legacy: free-form JSON generation (used as final fallback)."""
        messages = [
            {"role": "system", "content": _get_system_prompt()},
            *history_turns,
            {"role": "user", "content": user_content},
        ]
        try:
            from app.core.config import settings
            model_id = getattr(settings, "OPENROUTER_MODEL", "openai/gpt-4o-mini")
        except Exception:
            model_id = "openai/gpt-4o-mini"

        try:
            raw = await self.model_router.generate_with_messages(
                messages=messages,
                model_id=model_id,
                response_format={"type": "json_object"},
            )
            if not raw:
                return None
            return json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            logger.error("PlannerAgent._call_planner_model JSON error: %s", exc)
            return None
        except Exception as exc:
            logger.error("PlannerAgent._call_planner_model error: %s", exc)
            return None

    def _parse_plan(self, raw: dict | None, agent_input: AgentInput) -> ExecutionPlan:
        """Parse raw dict into ExecutionPlan with safety guards."""
        if not raw:
            return self._fallback_plan(agent_input.message)

        intent = raw.get("intent", _FALLBACK_INTENT)
        if intent not in VALID_INTENTS:
            logger.warning("PlannerAgent: unknown intent %r → fallback", intent)
            intent = _FALLBACK_INTENT
            raw["intent"] = intent

        if intent == "general_chat":
            raw["steps"] = []
            raw["exam_params"] = None

        if not isinstance(raw.get("steps"), list):
            raw["steps"] = []

        try:
            return ExecutionPlan(**raw)
        except (ValidationError, TypeError) as exc:
            logger.error("PlannerAgent: ExecutionPlan validation failed — %s", exc)
            return self._fallback_plan(agent_input.message)

    @staticmethod
    def _fallback_plan(message: str) -> ExecutionPlan:
        return ExecutionPlan(
            intent=_FALLBACK_INTENT,
            goal_summary=f"Handle the user's request: {message[:120]}",
            is_executable=True,
        )

    @staticmethod
    def _ensure_resolve_step(plan: ExecutionPlan) -> ExecutionPlan:
        """Inject ResolveSubjectOffering pre-step when subjectOfferingId is missing."""
        if (
            plan.intent == "generate_exam"
            and plan.exam_params is not None
            and plan.exam_params.subjectOfferingId is None
        ):
            already_there = any(
                s.tool == "ResolveSubjectOffering"
                for s in plan.pre_execution_steps
            )
            if not already_there:
                logger.info("PlannerAgent: injecting ResolveSubjectOffering pre-step")
                plan.pre_execution_steps.append(
                    PreExecutionStep(
                        tool="ResolveSubjectOffering",
                        reason="subjectOfferingId required but not provided",
                        input_payload={"subjectName": plan.exam_params.subjectName},
                    )
                )
        return plan
