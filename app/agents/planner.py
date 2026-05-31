"""
planner.py

PlannerAgent — uses the LLM to classify the user's intent, extract academic
parameters, and optionally produce multi-step ExecutionPlans for tool-bound
requests.

Key upgrades (v4.0):
  - Injects academic_context into the classification prompt so the model can
    auto-fill tool parameters (e.g. user_id, subjectOfferingId) without
    follow-up questions. (Context-Aware Reasoning)
  - Unlocks multi-step plans for tool-bound intents. general_chat is strictly
    locked to steps=[]. (Multi-Step Planning)
  - Updated system prompt with consistent rule numbering and university domain
    framing. (Prompt Engineering)
  - Uses structured messages[] instead of concatenated history strings. (fixed)
"""

import json
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

# ── Valid intent catalogue ────────────────────────────────────────────────────
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
    "action_execute",   # enrollment, submissions — routed to DynamicApiModule
    "assignment_query", # student asks about their assignments, deadlines, submissions
    "study_plan",       # student asks for a personalized study/revision plan
}


# ── Fallback intent ───────────────────────────────────────────────────────────
_FALLBACK_INTENT = "general_chat"

# ── Deterministic exam-intent keyword sets (Layer-2 LLM override) ─────────────
# Matched AFTER LLM classification so any LLM misclassification of exam-creation
# requests is caught and corrected deterministically. Case-insensitive substring match.
_EXAM_KEYWORDS_EN: frozenset[str] = frozenset({
    "generate exam",    "create exam",     "make exam",
    "build exam",       "write exam",      "prepare exam",
    "prepare test",     "create test",     "generate test",
    "make test",        "build test",      "write test",
    "exam for subject", "exam for course", "new exam",
    "draft exam",       "design exam",     "set exam",
    "set a test",       "produce exam",    "develop exam",
})

# Action verbs that, combined with "exam" or "test" anywhere in the message,
# confirm exam-creation intent  (covers "create a new ... exam" patterns)
_EXAM_ACTION_VERBS_EN: frozenset[str] = frozenset({
    "create", "generate", "make", "build", "write",
    "prepare", "draft", "design", "produce", "develop",
    "compose", "set",
})
_EXAM_TARGET_WORDS_EN: frozenset[str] = frozenset({"exam", "test", "quiz", "assessment"})

_EXAM_KEYWORDS_AR: frozenset[str] = frozenset({
    "اعمل امتحان",  "انشئ امتحان",  "سوي امتحان",
    "حضّر امتحان",   "اكتب امتحان",  "امتحان لمادة",
    "عمل امتحان",    "أنشئ امتحان",  "صمّم امتحان",
    "عايز امتحان",   "نعمل امتحان",  "جهّز امتحان",
    "جهّز اختبار",  "عمل اختبار",   "انشئ اختبار",
    "طوّر امتحان",  "اكتب اختبار",  "حضر امتحان",
    "انشئ امتحان",  "صمم امتحان",
})


def _detect_generate_exam(message: str) -> bool:
    """
    Two-pass deterministic scan for exam-creation intent.

    Pass 1 — direct phrase match: catches "generate exam", "create test", etc.
    Pass 2 — loose match: catches "create a new introduction to ML exam"
             by checking if BOTH an action verb AND an exam target word
             appear anywhere in the message (order-independent).

    Never fires on passive phrases like "view exam", "exam results",
    "I failed the exam", or "when is the exam?" because those lack
    a creation action verb.
    """
    msg = message.strip().lower()

    # Pass 1: direct phrase substring match
    for kw in _EXAM_KEYWORDS_EN:
        if kw in msg:
            return True
    for kw in _EXAM_KEYWORDS_AR:
        if kw in msg:
            return True

    # Pass 2: verb + target word loose match (handles inserted adjectives)
    import re
    words = set(re.findall(r"\b\w+\b", msg))
    has_action = bool(words & _EXAM_ACTION_VERBS_EN)
    has_target = bool(words & _EXAM_TARGET_WORDS_EN)
    if has_action and has_target:
        return True

    return False

def _detect_backend_query(message: str) -> bool:
    """Detect if the user is asking a data query like counts, lists, analytics, or system stats."""
    msg = message.strip().lower()

    _BACKEND_KEYWORDS = {
        # ── Arabic: regulation / roadmap queries ─────────────────────────
        "لائحة", "لوائح", "خطة دراسية", "خارطة طريق",
        "مواد الترم", "مواد الفصل", "مواد السنة",
        "المواد اللي هسجلها", "المواد المقترحة", "ايه المواد",
        "كام ساعة خلصت", "ساعات معتمدة", "الساعات الباقية",
        "رسبت في", "مواد راسب", "مواد باقية", "مواد خلصتها",
        "تقدمي الأكاديمي", "وضعي الأكاديمي", "هل انا في المسار",
        "الترم الجاي", "المواد القادمة", "ايه اللي باقيلي",
        "roadmap", "academic plan", "study plan", "academic progress",
        "credit hours", "remaining subjects", "passed subjects",
        "failed subjects", "next semester subjects", "what subjects",
        # ── Arabic: enrollment ACTIONS (register/enroll me) ──────────────
        "سجلني", "سجل لي", "سجل لى", "اسجلني", "عايز أسجل",
        "عايز اسجل", "ابدأ التسجيل", "ابدا التسجيل",
        "سجلني في المواد", "سجل في كل المواد", "سجلني في الترم",
        "تسجيل المواد", "تسجيل في المواد", "اعملي تسجيل",
        "register me", "enroll me", "sign me up", "auto enroll",
        "register for courses", "enroll in courses",
        # ── Arabic: count / analytics ─────────────────────────────────────
        "كم عدد", "كام بدرس", "كام طالب", "كام دكتور", "كام مادة",
        "عدد الدكاترة", "عدد الطلاب", "عدد المواد", "عدد الاقسام",
        "نسبة", "احصائيات", "إحصائيات", "احصاء", "إحصاء",
        "تحليل", "تقرير", "ملخص الشكاوى", "ملخص النتائج",
        "مين هم", "قائمة", "اللي بيدرس", "اللي مسجل",
        "اعلى", "أعلى", "اقل", "أقل", "افضل", "أفضل",
        # ── Arabic: filter / relationship queries ─────────────────────────
        "دكاترة في", "طلاب في", "مواد في", "عرض في",
        "في قسم", "في كلية", "في الفرقة", "في الدفعة",
        "بيدرس في", "مسجل في", "تابع ل",
        # ── Arabic: entity names ──────────────────────────────────────────
        "كليات", "الكليات", "دكاترة", "الدكاترة",
        "قسم", "اقسام", "الأقسام", "الاقسام",
        "طلاب", "الطلاب", "مواد", "المواد",
        "فرقة", "دفعة", "الفرقة", "الدفعة",
        "عروض", "العروض", "التسجيلات",
        # ── Arabic: identity / profile ────────────────────────────────────
        "اسمي", "اسم", "انا مين", "أنا مين",
        "من انا", "من أنا", "مين انا", "مين أنا",
        "معلوماتي", "بياناتي", "بروفايلي", "حسابي",
        "كليتي", "قسمي", "دفعتي", "فرقتي",
        # ── Arabic: system data ───────────────────────────────────────────
        "بيانات", "جامعه", "جامعة", "السيستم",
        # ── English: count / analytics ────────────────────────────────────
        "how many", "count of", "number of", "total students",
        "total doctors", "total courses", "how much",
        "statistics", "analytics", "distribution", "breakdown",
        "top students", "at risk", "failing students",
        "most enrolled", "most popular", "average gpa",
        # ── English: filter / relationship queries ────────────────────────
        "doctors in", "students in", "courses in", "offerings in",
        "in department", "in college", "in batch", "in year",
        "who teaches", "enrolled in", "assigned to",
        "list doctors", "list students", "list courses",
        # ── English: list / show ──────────────────────────────────────────
        "list of", "show me", "what are", "give me",
        "students list", "doctors list", "departments",
        # ── English: my data ──────────────────────────────────────────────
        "my courses", "my subjects", "my schedule", "my grades",
        "my gpa", "my results", "my profile", "my info",
        "my college", "my department", "my batch",
        "who am i", "my name", "my account", "my details",
        "profile", "courses i have", "subjects i have",
    }
    for kw in _BACKEND_KEYWORDS:
        if kw in msg:
            return True
    return False


# ── Deterministic regulation keyword override ─────────────────────────────────
_REGULATION_KEYWORDS: frozenset[str] = frozenset({
    # Arabic — explicit regulation/curriculum content questions
    "اشرح الليحه", "اشرح اللائحه", "اشرح اللائحة",
    "ايه الليحه", "ايه اللائحه", "ايه اللائحة",
    "محتوى اللائحة", "محتوى الليحه",
    "مواد سنة اولى", "مواد السنة الاولى", "مواد سنة اولي",
    "مواد سنة تانية", "مواد السنة التانية", "مواد سنة ثانية",
    "مواد سنة تالتة", "مواد السنة التالتة", "مواد سنة ثالثة",
    "مواد سنة رابعة", "مواد السنة الرابعة",
    "مواد الترم الاول", "مواد الترم الثاني",
    "مواد جوه الليحه", "مواد في اللائحة",
    "ساعات اللائحة", "متطلبات التخرج", "شروط التخرج",
    "الخطة الدراسية", "خطة الدراسة",
    "دليل الطالب", "الدليل الاكاديمي",
    # Egyptian dialect variants of اللائحة (common misspellings / colloquial forms)
    "الاليخه", "اللايحه", "اللايحة",
    "هذه الليحه", "هذه اللايحه", "هذه اللائحه", "هذه اللائحة",
    "في الليحه", "في اللايحه", "في اللائحه",
    "جوه اللايحه", "جوا اللائحه", "جوا اللايحه",
    "اليخه", "الليحة",
    "كم مادة في", "كم ماده في",
    "مواد اللائحه", "مواد اللايحه", "مواد الاليخه",
    # English
    "explain the regulation", "what is in the regulation",
    "subjects in year one", "subjects in first year",
    "subjects in second year", "subjects in third year",
    "graduation requirements", "credit hours required",
    "study plan subjects", "curriculum subjects",
    "student handbook", "academic handbook",
})


def _detect_regulation(message: str) -> bool:
    """Detect if the user wants to know content FROM the regulation document."""
    msg = message.strip().lower()
    for kw in _REGULATION_KEYWORDS:
        if kw in msg:
            return True
    return False


# ── Deterministic material_qa keyword override ────────────────────────────────
_MATERIAL_QA_KEYWORDS: frozenset[str] = frozenset({
    # English
    "explain from lecture", "from the material", "what does the lecture say",
    "according to the course", "course material", "lecture notes",
    "from the lecture", "from the book", "from the textbook",
    "in the lecture", "in the material", "in the book",
    # Arabic
    "من المحاضرة", "من المادة", "اشرح من", "في الكتاب",
    "في المحاضرة", "من الملزمة", "من الكتاب",
})


def _detect_material_qa(message: str) -> bool:
    """Detect if the user is asking a question grounded in course material."""
    msg = message.strip().lower()
    for kw in _MATERIAL_QA_KEYWORDS:
        if kw in msg:
            return True
    return False


# ── Deterministic assignment keyword override ─────────────────────────────────
_ASSIGNMENT_KEYWORDS: frozenset[str] = frozenset({
    # Arabic — ask about assignments / deadlines
    "واجب", "الواجب", "واجباتي", "واجبات", "تسليم", "التسليم",
    "موعد التسليم", "الموعد النهائي", "deadline",
    "سلمت", "لسه ما سلمتش", "هل سلمت", "حالة الواجب",
    "درجة الواجب", "الواجب الجاي", "مش عارف الواجب",
    "الواجبات المتأخرة", "واجب متأخر", "فاتني الواجب",
    "ايه الواجبات", "اشرحلي الواجب", "شرح الواجب",
    "تفاصيل الواجب", "متطلبات الواجب",
    # English
    "my assignment", "my assignments", "assignment deadline", "due date",
    "submitted assignment", "did i submit", "assignment status",
    "pending assignment", "overdue assignment", "late assignment",
    "assignment grade", "assignment feedback", "explain assignment",
    "assignment requirements", "assignment details", "what is the assignment",
    "show me assignments", "list assignments",
})


def _detect_assignment_query(message: str) -> bool:
    """Detect if the user is asking about their assignments."""
    msg = message.strip().lower()
    for kw in _ASSIGNMENT_KEYWORDS:
        if kw in msg:
            return True
    return False


# ── Deterministic study-plan keyword override ─────────────────────────────────
_STUDY_PLAN_KEYWORDS: frozenset[str] = frozenset({
    # Arabic — study plan / schedule requests
    "خطة مذاكرة", "خطة دراسة", "خطة دراسية", "خطة للمذاكرة",
    "اعمللي خطة", "اعمل لي خطة", "عمل خطة", "خطة للأسبوع",
    "جدول مذاكرة", "جدول دراسة", "جدول للأسبوع", "جدول يومي",
    "كيف أذاكر", "كيف اذاكر", "ازاي اذاكر", "ازاي أذاكر",
    "أولوياتي", "اولوياتي", "رتب أولوياتي", "رتب اولوياتي",
    "ما المواد التي أركز", "أركز على إيه", "أركز على ايه",
    "كيف أرفع معدلي", "كيف ارفع معدلي", "أرفع المعدل",
    "ارفع المعدل", "كيف أحسن معدلي", "كيف احسن معدلي",
    "كيف أنجح", "كيف انجح", "كيف أكمل", "كيف اكمل الترم",
    "ماذا أذاكر", "ماذا اذاكر", "ايه اللي أذاكره", "إيه اللي أذاكره",
    "خطة للامتحانات", "خطة الامتحانات", "خطة للميدتيرم", "خطة للفاينل",
    "خطة للاختبار", "استعداد للامتحان", "استعداد للميدتيرم",
    "كيف أستعد للامتحان", "كيف استعد للامتحان",
    "أذاكر إيه النهارده", "أذاكر ايه النهارده", "اذاكر ايه", "اذاكر إيه",
    "وقت المذاكرة", "توزيع وقت", "توزيع المذاكرة",
    "مواد محتاجة تركيز", "مواد محتاجة اهتمام", "المواد الصعبة",
    "تقدر تساعدني أذاكر", "ساعدني في المذاكرة",
    # English
    "study plan", "study schedule", "study timetable",
    "revision plan", "revision schedule",
    "how to study", "how should i study", "what should i study",
    "study for midterm", "study for final", "study for exam",
    "prioritize my subjects", "my priorities this week",
    "raise my gpa", "improve my gpa", "improve my grades",
    "how to pass", "study tips", "focus on what",
    "what to study today", "study today",
    "weekly plan", "daily plan", "study this week",
    "exam prep", "exam preparation", "prepare for exam",
})


def _detect_study_plan(message: str) -> bool:
    """Detect if the user is asking for a study or revision plan."""
    msg = message.strip().lower()
    for kw in _STUDY_PLAN_KEYWORDS:
        if kw in msg:
            return True
    return False


# ── Available backend tools (referenced in system prompt) ─────────────────────

_AVAILABLE_TOOLS = [
    "ResolveSubjectOffering",
    "GetStudentResults",
    "GetStudentGrades",
    "GetGPASummary",
    "GetTranscript",
    "GetSchedule",
    "GetSubjectOfferings",
    "GetCourseEnrollments",
    "GenerateExam",
    "DistributeExam",
    # ── New ──
    "SubmitComplaint",
    "GetComplaints",
    "GetStudentAcademicSummary",
    "BulkCreateStudents",
    "BulkUploadGrades",
    "GetMaterials",
]

# ── System prompt ─────────────────────────────────────────────────────────────

def _get_system_prompt() -> str:
    """Load planner system prompt from app/prompts/planner_system.md; fall back to inline."""
    try:
        return render_prompt("planner_system", tools=", ".join(_AVAILABLE_TOOLS))
    except Exception as exc:
        logger.warning("PlannerAgent: prompt load failed — using inline fallback: %s", exc)
        return _SYSTEM_PROMPT_FALLBACK.format(tools=", ".join(_AVAILABLE_TOOLS))


_SYSTEM_PROMPT_FALLBACK = """\
You are an AI Planning Agent for a university management system.

⚠️ CRITICAL LANGUAGE RULE — apply before everything else:
- Detect the user's language from their message.
- If the user writes in Arabic or Egyptian dialect → ALL your responses and goal_summary MUST be in Arabic.
- If the user writes in English → respond in English.
- NEVER mix languages. NEVER reply in English to an Arabic-speaking user.
- This applies to goal_summary, clarification questions, and all text fields.

Your job is to classify the user's request and return a structured JSON plan.

## Valid Intents
- general_chat       — conversation, questions, greetings, anything not needing backend data
- backend_api_query  — MANDATORY for querying system stats, counting numbers (كم عدد), user lists, or any database retrieval.
- summarization      — summarise a document or text
- generate_exam      — generate a university exam (doctor/admin only)
- result_query       — query academic results, grades, GPA, transcripts, schedules
- file_extraction    — extract information from an uploaded file (no bulk ops)
- complaint_submit   — student submitting a complaint or feedback about a doctor/exam/grade
- complaint_summary  — admin/doctor requesting a summary of submitted complaints
- file_processing    — bulk upload of Excel (students/grades) or PDF summarization via fileUrl
- cv_analysis        — analyzing a student CV to extract skills and give recommendations
- academic_advice    — personalized academic recommendations based on GPA and enrolled courses
- material_explanation — explain or summarize real course material fetched from the backend
- material_qa        — answer a student question grounded ONLY in indexed course material (RAG)
- action_execute     — execute a write action in the system (enroll student, create entity, etc.)
- assignment_query   — student asks about their assignments, deadlines, submission status, or requirements
- study_plan         — student asks for a personalized study/revision plan, schedule, weekly priorities, or exam prep strategy

## Output Schema (return ONLY this JSON, no markdown, no extra text)
{{
  "intent": "<one of the valid intents>",
  "goal_summary": "<one clear sentence describing what the user wants>",
  "is_executable": true,
  "exam_params": null,
  "pre_execution_steps": [],
  "steps": []
}}

## Rules

### 1. general_chat
- steps MUST be [] (empty array). Never add steps for general_chat.
- exam_params MUST be null.
- Use this intent for greetings, explanations, advice, and any question
  that does not require fetching real student/exam data from the backend.

### 2. Tool-bound intents (summarization, result_query, file_extraction, generate_exam)
- You MAY include steps when multiple sequential backend calls are needed.
- Available tools: {tools}
- Step format:
  {{
    "step_id": <int>,
    "action": "tool",
    "tool_name": "<one of the available tools>",
    "input_payload": {{...}},
    "depends_on": []
  }}
- Use {{{{step_N.output}}}} to reference the output of step N in a later step.
- If only one tool call is needed, leave steps=[].

### 3. generate_exam — HIGHEST PRIORITY INTENT FOR EXAM CREATION

⚠️ CRITICAL: The following phrases ALWAYS mean intent = "generate_exam".
NEVER classify them as general_chat. The executor enforces role permissions separately.

English triggers (ANY of these = generate_exam):
  "create exam", "generate exam", "make exam", "build exam",
  "write exam",  "prepare exam",  "prepare test", "new exam",
  "draft exam",  "design exam",  "set exam",    "produce exam",
  "exam for subject", "exam for course", "create test", "generate test"

Arabic triggers (ANY of these = generate_exam):
  "اعمل امتحان", "انشئ امتحان", "سوي امتحان", "حضّر امتحان",
  "اكتب امتحان", "امتحان لمادة", "عمل امتحان", "جهّز امتحان"

Rules for generate_exam:
- Use intent=generate_exam whenever the user wants to CREATE or GENERATE any exam/test.
- Role does NOT affect intent classification. Even if role=student, use generate_exam.
  The executor will enforce the RBAC denial if the role is not permitted.
- If subject details are missing, STILL use intent=generate_exam and fill exam_params
  with whatever IS available. Leave missing fields as null.
- Populate exam_params with:
    collegeName, departmentName, batchName, subjectName,
    numberOfQuestions (int, default 10 if not specified),
    examType ("midterm"|"final", default "midterm" if not specified),
    variationMode ("same_for_all"|"different_per_student"),
    subjectOfferingId (string|null)
- If subjectOfferingId is unknown, add ResolveSubjectOffering to pre_execution_steps.
  pre_execution_steps format (use "tool" NOT "tool_name"):
  {{"tool": "ResolveSubjectOffering", "reason": "Need subjectOfferingId to create exam", "input_payload": {{"departmentId": "<from context>", "subjectName": "<from user>"}}}}
- NEVER leave intent=general_chat when the user's action is exam creation.

### 4. Context-aware auto-fill (MANDATORY)
- The caller has already authenticated and their academic record is embedded in
  the request under academic_context.
- You MUST extract userId, studentId, courseId, subjectOfferingId,
  departmentId, batchId, collegeName, departmentName, batchName from
  academic_context and inject them into the relevant tool input_payload fields.
- NEVER ask the user for parameters already present in academic_context.
- NEVER leave userId or studentId blank when they exist in academic_context.
- If a required field is absent from both the user message AND academic_context,
  only then flag it as missing in goal_summary.

### 5. complaint_submit (student only)
- Use intent=complaint_submit when a student reports a problem, complains,
  or gives negative feedback about a doctor, exam, grade, or the system.
- Extract from user message: the complaint content (for "message" field).
- Required payload fields (MUST be populated from academic_context):
    userId, subjectOfferingId
- targetType MUST be one of: "Doctor" | "Exam" | "Grade" | "Other"
  Infer it from the message (e.g. "doctor" → "Doctor", "exam" → "Exam",
  "grade" / "mark" → "Grade", anything else → "Other").
- DoctorId is resolved server-side — do NOT include it in the payload.
- If role is NOT "student" → use general_chat instead.

### 6. complaint_summary (admin/doctor only)
- Use intent=complaint_summary when an admin or doctor asks to see, review,
  or summarize complaints.
- If role is "student" → use general_chat instead.

### 7. file_processing
- Use intent=file_processing when the user message contains a fileUrl
  OR mentions uploading/processing a file for bulk operations.
- Do NOT use this for single-file text extraction (use file_extraction).

### 7b. Checking/reading course material content → ALWAYS use material_explanation
- When user wants to "check", "read", "verify", "show content of" a course file/material:
  Use intent=material_explanation (NOT file_extraction).
- Triggers: "تحقق من محتوى الملف", "اقرأ الملف", "show file content",
  "check the material", "read the file", "ما محتوى الملف", "اعرض محتوى المادة"
- The material_explanation module handles file fetching internally using subjectOfferingId.

### 8. cv_analysis
- Use intent=cv_analysis when the user wants their CV reviewed, analyzed,
  or feedback on skills, experience, or job readiness.

### 9. academic_advice
- Use intent=academic_advice when a student asks for study advice, course
  recommendations, or wants to know how to improve their GPA.

### 9b. study_plan — PERSONALIZED STUDY SCHEDULE GENERATION

⚠️ ALWAYS use intent=study_plan when the student asks for:
- A study plan / revision plan / weekly schedule ("اعمللي خطة مذاكرة", "study plan")
- How to study for an exam ("كيف أذاكر للميدتيرم", "how to study for final")
- What to prioritize this week ("أولوياتي الأسبوع ده", "priorities this week")
- How to raise/improve their GPA ("ازاي أرفع معدلي", "how to raise my GPA")
- What to study today ("اذاكر ايه النهاردة", "what should I study today")
- Exam preparation strategy ("كيف أستعد للامتحان", "exam prep")

Rules for study_plan:
- Use intent=study_plan for ALL schedule/planning/prioritization requests.
- Steps MUST be [] — the StudyPlanModule fetches all data internally.
- exam_params MUST be null.
- NEVER use academic_advice for study scheduling requests — academic_advice is for
  general standing queries ("how am I doing?"), study_plan is for actionable schedules.

### 10. material_explanation (STRICT DATA-FIRST — HIGHEST PRIORITY INTENT)
- ALWAYS use intent=material_explanation when the user asks to EXPLAIN, SUMMARIZE,
  DESCRIBE, UNDERSTAND, or REVIEW a specific subject, course, topic, or lecture.

- English triggers (non-exhaustive):
    "explain course", "explain subject", "explain this topic",
    "summarize material", "summarize course", "summarize this subject",
    "what does this material say", "what is this course about",
    "understand this subject", "review the material", "study material",
    "give me a summary of", "help me understand", "break down this course"

- Arabic triggers (non-exhaustive):
    "شرح مادة", "اشرح المادة", "شرح الموضوع", "اشرح موضوع",
    "لخص المادة", "ملخص المادة", "لخص هذه المادة",
    "فهم المادة", "ما محتوى المادة", "عايز أفهم المادة",
    "ساعدني أفهم", "شرح الدرس", "اشرح لي", "ما هو محتوى",
    "راجع المادة", "شرح موضوع الامتحان", "عايز ملخص"

- MANDATORY: This intent triggers a real backend fetch of course materials.
    * The subjectOfferingId MUST be injected from academic_context.
    * If subjectOfferingId is NOT available in academic_context, set:
      goal_summary = "Need to clarify which subject offering to fetch materials for."
      and leave steps=[] — the module will prompt the user.
    * NEVER use general_chat for these triggers — always use material_explanation.
    * NEVER add tool steps for this intent — the MaterialExplanationModule handles
      the backend fetch internally.

### 11. backend_api_query (Dynamic API Fetching)
- Use intent=backend_api_query for ANY question requesting data from the university system.
- Triggers include asking about: users, names, colleges, departments, subjects, students, doctors, counts, lists.
- Examples: "ما هي الكليات", "من هم الدكاترة", "انا اسمي ايه", "كم عدد الطلاب", "what are the colleges"

- Doctor-specific triggers → ALWAYS backend_api_query:
  "علمت كام امتحان", "عندي كام امتحان", "امتحاناتي", "شوفلي امتحاناتي",
  "درجات الطلاب", "نتايج الطلاب", "نتيجة المادة", "درجات المادة",
  "كام طالب في المادة", "الطلاب اللي رسبوا", "من أعلى درجة",
  "my exams", "how many exams", "student grades", "exam results",
  "students who failed", "grade summary"
  → For these, use GET /api/Exams/my-exams or GET /api/Exams/{{id}}/results

### 12. Identity & Profile Queries → ALWAYS backend_api_query
⚠️ CRITICAL: Questions about the user's own identity, name, or profile data MUST use backend_api_query.
NEVER answer identity questions from LLM knowledge — always fetch from the backend.

Arabic triggers (ANY of these = backend_api_query):
  "انا مين", "أنا مين", "مين انا", "مين أنا",
  "من انا", "من أنا", "اسمي ايه", "اسمي إيه",
  "معلوماتي", "بياناتي", "بروفايلي", "حسابي"

English triggers (ANY of these = backend_api_query):
  "who am i", "what is my name", "my profile", "my info",
  "my account", "my details"

IMPORTANT: The userId is available in academic_context — ALWAYS inject it in the request.

### 13. action_execute — TAKE ACTION IN THE SYSTEM (POST endpoints)

⚠️ CRITICAL: Use intent=action_execute when the user wants the AI to DO something in the system,
not just query data. The AI must call the correct POST endpoint automatically.

Arabic student triggers → auto-enroll:
  "سجلني", "سجل لي", "سجل لى", "اسجلني", "عايز أسجل", "اعملي تسجيل",
  "سجلني في المواد", "سجل في كل المواد", "سجلني في الترم",
  "تسجيل المواد", "ابدأ التسجيل"

English student triggers → auto-enroll:
  "register me", "enroll me", "sign me up", "auto enroll",
  "register for courses", "enroll in courses", "enroll me in subjects"

For enrollment actions:
- endpoint: POST /api/enrollments/auto-enroll
- payload: {{ "studentId": "<from academic_context>", "batchId": "<from academic_context>" }}

Admin action triggers → create/add entities:
  "أضف طالب", "سجل طالب جديد", "أضف دكتور", "أضف مادة", "انشئ كلية",
  "add student", "create student", "add doctor", "create doctor",
  "add subject", "create subject", "add college", "create college"

For admin creation actions:
- Use the relevant POST endpoint based on the entity type
- Extract all required fields from the message
- Inject any IDs available from academic_context

Rules for action_execute:
- The AI EXECUTES the action immediately — no confirmation needed for safe actions.
- After execution, narrate the result clearly (what was done, what succeeded/failed).
- NEVER use general_chat when the user is asking to perform an action.
- If required parameters are missing → ask for them specifically before executing.

### 14. When in doubt → use general_chat with steps=[].

### 15. CONTEXT & PRONOUNS — use conversation history (CRITICAL)
You ARE given the previous turns of the conversation. USE THEM. Never treat the current message in isolation when the user clearly refers to something said earlier.

Rules:
- If the current message contains a pronoun ("ها"، "ه"، "it"، "this"، "them") or is very short (≤6 words) without a clear subject, resolve it from the PREVIOUS turns.
- If the prior turns mention the regulation / اللائحة / دليل / curriculum and the user now says "اشرحها / لخصها / explain it / summarize it / اقراها / show me" → intent = **regulation**.
- If the prior turns mention a specific subject material / محاضرة / lecture and the user says "اشرحها / explain it" → intent = **material_explanation**.
- If the prior turns mention an exam and the user says "ابعتها / send it / distribute it" → intent = **action_execute**.
- NEVER reply with "اشرح إيه؟" or "which one?" when the answer is sitting in the previous turn.

### Few-shot examples (study these — they show language variety + coreference)

Example A — Egyptian dialect + pronoun referring to regulation (from previous turn):
  [prior turn] user: "عندي لائحة اسمها fbn ضيفتها هنا"
  [prior turn] assistant: "تمام، اللائحة 'fbn' متاحة. تحب أعمل إيه فيها؟"
  current user: "طيب اقرا اللائحه وقولي الملخص بتاعها"
  → intent = "regulation"

Example B — Even shorter pronoun, same context:
  [prior turn] assistant: "موجود ملف اللائحة fbn..."
  current user: "لخصهالي"
  → intent = "regulation"   (NOT general_chat, NOT material_explanation)

Example C — Variant of "create exam" the keyword list might miss:
  current user: "ممكن تجهزلي امتحان نص الترم في الـ data structures على السريع"
  → intent = "generate_exam"

Example D — Pronoun referring to material (lecture) from prior turn:
  [prior turn] user: "اشرحلي محاضرة الـ binary trees"
  [prior turn] assistant: "تمام، فيها 4 أقسام: ..."
  current user: "اشرحلي الجزء الأخير تاني"
  → intent = "material_explanation"

Example E — User uses dialect "ايه اللي فيها":
  [prior turn] assistant: "اللائحة الأكاديمية موجودة..."
  current user: "ايه اللي فيها؟"
  → intent = "regulation"

Example F — When NOT to assume coreference:
  current user: "ازيك" (greeting, no pronoun, no prior topic relevant)
  → intent = "general_chat"

### Multi-step example (result_query — grades then GPA):
{{
  "intent": "result_query",
  "goal_summary": "Fetch student grades and calculate GPA",
  "is_executable": true,
  "exam_params": null,
  "pre_execution_steps": [],
  "steps": [
    {{"step_id": 1, "action": "tool", "tool_name": "GetStudentGrades",
      "input_payload": {{"userId": "<from context>"}}, "depends_on": []}},
    {{"step_id": 2, "action": "tool", "tool_name": "GetGPASummary",
      "input_payload": {{"gradeData": "{{{{step_1.output}}}}"}}, "depends_on": [1]}}
  ]
}}
"""


class MemoryStore(Protocol):
    """Protocol defining how the Planner retrieves historical context."""

    async def get_context(self, user_id: str | None) -> str: ...


class PlannerAgent(BaseAgent):
    """
    Generates an ExecutionPlan by asking the LLM to classify the user's intent.

    v4.0 upgrades:
      - Injects academic_context for context-aware parameter resolution.
      - Allows multi-step plans for tool-bound intents.
      - Uses structured messages[] history.
    """

    def __init__(
        self,
        model_router,
        ranker=None,
        memory: Optional[MemoryStore] = None,
    ):
        self.model_router = model_router
        self.ranker = ranker
        self.memory = memory

    # ─────────────────────────────────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────────────────────────────────

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        1. Pull optional memory summary for long-term context.
        2. Inject academic_context to allow context-aware auto-filling.
        3. Build structured messages[] from history + current message.
        4. Call LLM for classification JSON.
        5. Validate, sanitise, and enrich the resulting ExecutionPlan.
        6. Return AgentOutput(status="success", data={"plan": plan}).
        """
        logger.info("PlannerAgent: starting for user_id=%s", agent_input.user_id)

        # ── Optional memory context ───────────────────────────────────────
        memory_prefix = ""
        if self.memory:
            try:
                past = await self.memory.get_context(agent_input.user_id)
                if past:
                    memory_prefix = f"[Conversation summary]: {past}\n\n"
            except Exception as mem_exc:
                logger.warning("PlannerAgent: memory lookup failed — %s", mem_exc)

        # ── Extract context components ────────────────────────────────────
        ctx = agent_input.context or {}
        role = ctx.get("role", "user")
        raw_history: list[dict] = ctx.get("history", [])
        academic_ctx: dict = ctx.get("academic_context", {})

        # Compact summary of academic context for auto-filling parameters
        auto_fill_note = ""
        if academic_ctx:
            # Only expose safe, useful fields — never passwords or tokens
            safe_keys = [
                "userId", "studentId", "courseId", "subjectOfferingId",
                "departmentId", "batchId", "collegeName", "departmentName",
                "profileId",  # Admin/Doctor profile ID (different from userId)
            ]
            relevant = {k: v for k, v in academic_ctx.items() if k in safe_keys and v}
            if relevant:
                auto_fill_note = (
                    f"\nAvailable context for auto-filling parameters: "
                    f"{json.dumps(relevant, ensure_ascii=False)}"
                )

        # ── Build structured history turns (last 3 pairs = 6 messages) ────
        history_turns: list[dict] = []
        for turn in raw_history[-6:]:
            turn_role = turn.get("role", "user")
            turn_content = str(turn.get("content", ""))
            if turn_role in ("user", "assistant") and turn_content:
                history_turns.append({"role": turn_role, "content": turn_content})

        # ── Compose user classification request ───────────────────────────
        user_content = (
            f"{memory_prefix}"
            f"User role: {role}\n"
            f"User message: {agent_input.message}"
            f"{auto_fill_note}"
        )

        # ── Call LLM ──────────────────────────────────────────────────────
        raw_json = await self._call_planner_model(history_turns, user_content)

        # ── Parse + validate → ExecutionPlan ──────────────────────────────
        plan = self._parse_plan(raw_json, agent_input)

        # ── Layer 2: Deterministic exam-intent override ────────────────────
        # Fires ONLY when the LLM returned general_chat but the message matches
        # a confirmed exam-creation keyword.  Zero false-positive risk because
        # the keyword list is restricted to create/generate action verbs.
        if plan.intent == "general_chat" and _detect_generate_exam(agent_input.message):
            logger.warning(
                "PlannerAgent [Layer-2 override]: LLM misclassified exam request as "
                "general_chat — correcting to generate_exam (message=%.100r)",
                agent_input.message,
            )
            plan.intent = "generate_exam"
            plan.goal_summary = (
                f"Generate an exam for: {agent_input.message[:120]}"
            )
            # Bootstrap exam_params if the LLM didn't populate them
            if plan.exam_params is None:
                plan.exam_params = ExamParams(
                    subjectName=None,
                    numberOfQuestions=10,
                    examType="midterm",
                    variationMode="same_for_all",
                )

        if plan.intent == "general_chat" and _detect_backend_query(agent_input.message):
            logger.warning(
                "PlannerAgent [Layer-2 override]: Correcting general_chat to backend_api_query for data request."
            )
            plan.intent = "backend_api_query"
            plan.goal_summary = "Query dynamic backend APIs to answer the user request."

        if plan.intent == "general_chat" and _detect_material_qa(agent_input.message):
            logger.warning(
                "PlannerAgent [Layer-2 override]: Correcting general_chat to material_qa "
                "for course-material-grounded question."
            )
            plan.intent = "material_qa"
            plan.goal_summary = "Answer question grounded in indexed course material."

        # ── Layer 2: Regulation document override ──────────────────────────
        # Fires when user is asking about content INSIDE the regulation PDF
        # (subjects per year, graduation requirements, study plan, etc.)
        if _detect_regulation(agent_input.message):
            if plan.intent not in ("regulation",):
                logger.warning(
                    "PlannerAgent [Layer-2 override]: Correcting %r to regulation "
                    "for regulation content question.", plan.intent
                )
            plan.intent = "regulation"
            plan.goal_summary = "Read and answer from the official academic regulation PDF."
            plan.is_executable = True

        # ── Layer 2: Study plan override (highest priority after exam/regulation) ─
        # Fires before assignment_query — study plan is more specific
        if plan.intent in ("general_chat", "academic_advice", "backend_api_query") \
                and _detect_study_plan(agent_input.message):
            logger.warning(
                "PlannerAgent [Layer-2 override]: Correcting %r to study_plan.", plan.intent
            )
            plan.intent = "study_plan"
            plan.goal_summary = (
                "توليد خطة مذاكرة شخصية مبنية على بيانات الطالب الأكاديمية الفعلية."
            )

        # ── Layer 2: Assignment query override ────────────────────────────
        if plan.intent in ("general_chat", "backend_api_query") and _detect_assignment_query(agent_input.message):
            logger.warning(
                "PlannerAgent [Layer-2 override]: Correcting %r to assignment_query.", plan.intent
            )
            plan.intent = "assignment_query"
            plan.goal_summary = "Show the student their assignments, deadlines, and submission status."

        # ── Deterministic guard: ensure ResolveSubjectOffering pre-step ───
        plan = self._ensure_resolve_step(plan)

        logger.info(
            "PlannerAgent: intent=%r steps=%d goal=%r",
            plan.intent, len(plan.steps), plan.goal_summary,
        )

        return AgentOutput(
            status="success",
            response=plan.goal_summary,
            data={"plan": plan},
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    async def _call_planner_model(
        self, history_turns: list[dict], user_content: str
    ) -> dict | None:
        """
        Send the planning request via structured messages[] that include history.

        Message order:
          [system]  _SYSTEM_PROMPT
          [prior turns from history…]   ← gives the planner conversation context
          [user]    role + message + academic_context note

        Using generate_with_messages + json_object response_format instead of
        generate_structured_json so that history_turns are forwarded to the model
        (generate_structured_json is a single-turn helper with no history support).
        """
        messages = [
            {"role": "system", "content": _get_system_prompt()},
            *history_turns,
            {"role": "user", "content": user_content},
        ]
        try:
            logger.debug(
                "PlannerAgent: requesting JSON from openai/gpt-4o-mini "
                "(history_turns=%d)", len(history_turns),
            )
            raw = await self.model_router.generate_with_messages(
                messages=messages,
                model_id="openai/gpt-4o-mini",
                response_format={"type": "json_object"},
            )

            if not raw:
                logger.warning(
                    "PlannerAgent: openai/gpt-4o-mini returned empty — fallback chain will handle it"
                )
                return None

            parsed = json.loads(raw) if isinstance(raw, str) else raw
            logger.debug("PlannerAgent: raw plan = %s", parsed)
            return parsed

        except json.JSONDecodeError as exc:
            logger.error("PlannerAgent: JSON parse failed — %s", exc)
            return None
        except Exception as exc:
            logger.error("PlannerAgent: model call failed — %s", exc, exc_info=True)
            return None

    def _parse_plan(self, raw: dict | None, agent_input: AgentInput) -> ExecutionPlan:
        """
        Validate the raw LLM dict into an ExecutionPlan with safety guards:
          - Invalid intent → downgrade to general_chat
          - general_chat   → force steps=[], exam_params=None
          - tool_name in steps → validated by executor (not here)
        """
        if not raw:
            return self._fallback_plan(agent_input.message)

        # Normalise intent
        intent = raw.get("intent", _FALLBACK_INTENT)
        if intent not in VALID_INTENTS:
            logger.warning(
                "PlannerAgent: unknown intent %r — falling back to %s",
                intent, _FALLBACK_INTENT,
            )
            intent = _FALLBACK_INTENT
            raw["intent"] = intent

        # HARD RULE: general_chat must never have steps or exam context
        if intent == "general_chat":
            raw["steps"] = []
            raw["exam_params"] = None

        # HARD RULE: steps must always be an empty list regardless of what
        # the planner returned (planner is advisory; steps come from planner
        # only for tool-bound intents, handled via pre_execution_steps or
        # the module architecture)
        # NOTE: We allow non-empty steps for non-chat intents (multi-step unlock)
        # but sanitise any non-list value to an empty list.
        if not isinstance(raw.get("steps"), list):
            raw["steps"] = []

        try:
            plan = ExecutionPlan(**raw)
            return plan
        except (ValidationError, TypeError) as exc:
            logger.error(
                "PlannerAgent: ExecutionPlan validation failed — %s", exc
            )
            return self._fallback_plan(agent_input.message)

    @staticmethod
    def _fallback_plan(message: str) -> ExecutionPlan:
        """Return a minimal, always-valid general_chat plan."""
        return ExecutionPlan(
            intent=_FALLBACK_INTENT,
            goal_summary=f"Handle the user's request: {message[:120]}",
            is_executable=True,
        )

    @staticmethod
    def _ensure_resolve_step(plan: ExecutionPlan) -> ExecutionPlan:
        """
        If the plan targets generate_exam but lacks subjectOfferingId,
        inject the ResolveSubjectOffering pre-execution step so the
        ExamGenerationModule never receives an incomplete plan.
        """
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
                logger.info(
                    "PlannerAgent: injecting ResolveSubjectOffering pre-step "
                    "(subjectOfferingId not supplied by user)"
                )
                plan.pre_execution_steps.append(
                    PreExecutionStep(
                        tool="ResolveSubjectOffering",
                        reason=(
                            "subjectOfferingId is required to generate the exam "
                            "but was not provided by the user"
                        ),
                        input_payload={
                            "subjectName": plan.exam_params.subjectName,
                        },
                    )
                )
        return plan
