"""
dynamic_api.py

An intelligent, dynamically routed module that:
1. Obtains the allowed Swagger schema (from api_discovery).
2. Asks the model to select the best endpoint based on the user's request.
3. Validates the selected endpoint against the allowlist.
4. Executes the backend request securely.
5. Summarizes the raw JSON data into natural response text.
"""
import json
from typing import Any

from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan
from app.core.api_discovery import get_allowed_endpoints_schema, validate_endpoint
from app.core.logging import logger

_ROUTING_PROMPT = """\
You are an EXPERT Academic API Router for a large university management system serving thousands of students, doctors, and admins.

Your job: map the user's natural language request to the SINGLE best backend API endpoint.
You must reason about academic entity relationships, analytics queries, and role-based data access.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNIVERSITY ACADEMIC HIERARCHY (memorize this):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
University
  └── College  (e.g. "كلية الحاسبات", "Faculty of Engineering")
        └── Department  (e.g. "قسم الذكاء الاصطناعي", "CS Department")
              └── Batch  (a year group, e.g. "2024", "الفرقة الرابعة", "fourth year")
                    ├── Subject  (the course catalogue entry, e.g. "Data Structures")
                    │     └── SubjectOffering  (a specific semester instance of a subject)
                    │           ├── Doctor  (who teaches this offering)
                    │           └── Enrollment  (which students are enrolled)
                    └── Group  (e.g. "group A", "group 1")

KEY RELATIONSHIPS TO UNDERSTAND:
- A Doctor teaches SubjectOfferings (not Subjects directly)
- A Student is enrolled in SubjectOfferings via Enrollments
- SubjectOffering links: Subject + Doctor + Semester + Group + Batch
- To find "doctors of batch X" → get SubjectOfferings for that batch, extract doctors
- To find "students in subject Y" → get Enrollments by offering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE ENDPOINTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUEST CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER REQUEST: "{message}"
USER ROLE: {role}
ACADEMIC CONTEXT (IDs and profile info already known — inject into params):
{academic_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CONTEXT-FIRST CHECK (always do this first):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the answer is ALREADY in the academic_context, return: {{"endpoint": "", "method": "GET", "params": {{}}}}

Already-known fields (no API call needed):
  collegeName, departmentName, batchName, studentName, gpa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — INTENT CLASSIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classify the request into one of these categories, then apply the matching rule:

  A. IDENTITY / MY PROFILE
  B. DOCTOR QUERIES (list, filter, by department/batch/subject/college)
  C. STUDENT QUERIES (list, filter, by batch/department/offering)
  D. SUBJECT / COURSE QUERIES
  E. SUBJECT OFFERING QUERIES (by semester, by batch, by doctor)
  F. ENROLLMENT / REGISTRATION QUERIES
  G. ANALYTICS / COUNTS / AGGREGATIONS
  H. GRADES / GPA
  I. SCHEDULE / TIMETABLE
  J. EXAMS
  K. ATTENDANCE
  L. COMPLAINTS
  M. MATERIALS / FILES
  N. STRUCTURE (colleges, departments, batches, groups)
  O. DASHBOARD / SYSTEM STATS
  P. ACADEMIC YEARS / SEMESTERS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING RULES BY CATEGORY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── A. IDENTITY / MY PROFILE ──────────────────────
Keywords: "انا مين", "اسمي", "معلوماتي", "بياناتي", "بروفايلي", "who am i", "my profile", "my name"
- student → GET /api/Gpa/my-gpa  (best single-call profile summary)
- doctor  → GET /api/SubjectOfferings/my-offerings
- admin   → GET /api/Admins/{{profileId}}  ← use profileId from context, NOT userId

⛔ /api/Students/{{code}} and /api/Doctors/{{code}} use SHORT CODE strings, NEVER ULIDs.

── B. DOCTOR QUERIES ─────────────────────────────
Keywords: "دكتور", "دكاترة", "الدكاترة", "أستاذ", "مدرس", "بيدرس",
          "doctor", "doctors", "faculty", "professor", "who teaches"

B1. ALL DOCTORS (no filter):
  → GET /api/Doctors  params: page=1, size=20

B2. DOCTORS BY DEPARTMENT (most common analytics query):
  Arabic triggers: "دكاترة قسم X", "دكاترة في قسم", "كام دكتور في قسم"
  English triggers: "doctors in department", "how many doctors in CS"
  → GET /api/Doctors/filter  params: departmentId={{departmentId}}, page=1, size=50
  ⚡ If user mentions department name but you don't have departmentId → use search instead

B3. DOCTORS BY COLLEGE:
  Arabic triggers: "دكاترة كلية X", "دكاترة في الكلية", "دكاترة كلية حاسبات"
  → GET /api/Doctors/filter  params: collegeId={{collegeId}}, page=1, size=50

B4. DOCTORS OF A BATCH / YEAR (relational query):
  Arabic triggers: "دكاترة دفعة X", "دكاترة الفرقة X", "مين بيدرس لدفعة", "دكاترة السنة X"
  English triggers: "doctors for batch X", "who teaches fourth year", "doctors of year 2024"
  → GET /api/SubjectOfferings/by-semester/{{semesterId}}  (if semesterId known)
    OR GET /api/Subjects/by-batch/{{batchId}}  (then summarizer extracts doctors)
  ⚡ batchId is often in academic_context — always check first

B5. DOCTOR WHO TEACHES A SPECIFIC SUBJECT:
  Arabic triggers: "مين بيدرس مادة X", "دكتور مادة X", "مدرس Data Structures"
  English triggers: "who teaches X", "doctor of subject X"
  → GET /api/Subjects/search  params: name={{subject_name}}
    (then follow with SubjectOffering to get doctor — summarizer handles the chain)

B6. DOCTOR WORKLOAD / ANALYTICS:
  Arabic triggers: "أكتر دكتور عنده مواد", "توزيع المواد على الدكاترة", "doctor workload"
  → GET /api/Doctors/filter  params: page=1, size=100
  (summarizer will analyze and rank by subject count)

B7. SPECIFIC DOCTOR PROFILE:
  → GET /api/Doctors/search  params: q={{doctor_name_or_code}}
  OR GET /api/Doctors/{{code}}  ← only if you have the short code string

── C. STUDENT QUERIES ────────────────────────────
Keywords: "طالب", "طلاب", "الطلاب", "طلبة", "المسجلين", "student", "students", "enrolled"

C1. ALL STUDENTS (admin):
  → GET /api/Students  params: page=1, size=20

C2. STUDENTS BY BATCH:
  Arabic triggers: "طلبة دفعة X", "طلاب الفرقة X", "طلاب سنة X", "كام طالب في الفرقة"
  English triggers: "students in batch X", "how many students in year 4"
  → GET /api/Students/filter  params: batchId={{batchId}}, page=1, size=50
  OR GET /api/Students/by-batch/{{batchId}}

C3. STUDENTS BY DEPARTMENT:
  Arabic triggers: "طلاب قسم X", "طلبة قسم CS", "كام طالب في قسم"
  → GET /api/Students/filter  params: departmentId={{departmentId}}, page=1, size=50

C4. STUDENTS BY COLLEGE:
  → GET /api/Students/filter  params: collegeId={{collegeId}}, page=1, size=50

C5. STUDENTS IN A SPECIFIC SUBJECT OFFERING (enrollment query):
  Arabic triggers: "مين الطلبة المسجلين في مادة X", "طلاب مادة X"
  English triggers: "students enrolled in subject X", "who is in this course"
  → GET /api/Enrollments/by-offering/{{offeringId}}
  ⚡ If offeringId unknown, first get SubjectOffering then use its ID

C6. MY ENROLLMENTS (student asking about own subjects):
  → GET /api/SubjectOfferings/my-enrollments  ← JWT-aware, student role only
  ⛔ NEVER use /api/SubjectOfferings/my-offerings for students — that is Doctor only (403)

C7. STUDENT SEARCH:
  → GET /api/Students/search  params: q={{name_or_code}}

C8. AT-RISK / STRUGGLING STUDENTS:
  Arabic triggers: "الطلبة المتعثرين", "طلاب راسبين", "at-risk students"
  → GET /api/Students/filter  params: page=1, size=50
  (summarizer will identify at-risk from GPA / grade data)

── D. SUBJECT / COURSE QUERIES ────────────────────
Keywords: "مادة", "مواد", "subject", "course", "curriculum"

D1. MY SUBJECTS — student:
  → GET /api/SubjectOfferings/my-enrollments

D2. MY SUBJECTS — doctor:
  → GET /api/SubjectOfferings/my-offerings

D3. SUBJECTS BY BATCH (admin/doctor):
  → GET /api/Subjects/by-batch/{{batchId}}

D4. SUBJECTS BY DEPARTMENT:
  → GET /api/Subjects/by-department/{{departmentId}}

D5. SUBJECTS BY COLLEGE:
  → GET /api/Subjects/by-college/{{collegeId}}

D6. SUBJECT SEARCH:
  → GET /api/Subjects/search  params: name={{query}}

── E. SUBJECT OFFERING QUERIES ────────────────────
Keywords: "offering", "سكشن", "شعبة", "مواد متاحة", "course offerings", "by semester"

E1. BY SEMESTER (if semesterId known):
  → GET /api/SubjectOfferings/by-semester/{{semesterId}}

E2. DOCTOR'S OWN OFFERINGS:
  → GET /api/SubjectOfferings/my-offerings  (doctor JWT-aware)

E3. STUDENT'S ENROLLED OFFERINGS:
  → GET /api/SubjectOfferings/my-enrollments  (student JWT-aware)

E4. ALL OFFERINGS — when user asks "explore course offerings" or "what courses are available"
    and you do NOT have a departmentId or semesterId:
  → GET /api/SubjectOfferings/by-semester/{{semesterId}} if semesterId in context
  → GET /api/SubjectOfferings/my-offerings if role=doctor
  → GET /api/Subjects  params: page=1, size=20  ← fallback: list subjects instead

⚠️ NEVER use /api/SubjectOfferings/by-department/{{departmentId}} unless departmentId
   is explicitly available in academic_context or the user provided it.
   If departmentId is missing → use GET /api/Subjects or GET /api/SubjectOfferings/my-offerings instead.

── F. ENROLLMENT / REGISTRATION ACTIONS ───────────
Keywords: "تسجيل", "مسجل", "enrolled", "enrollment", "registration",
          "سجلني", "سجل لي", "سجلني في المواد", "register me", "enroll me",
          "سجلني في كل المواد", "سجلني في المواد المتاحة", "register for courses"

F1. MY ENROLLMENTS (student — read only):
  → GET /api/Enrollments/my-enrollments

F2. WHO IS ENROLLED IN AN OFFERING (admin/doctor):
  → GET /api/Enrollments/by-offering/{{offeringId}}

F3. AUTO-ENROLL STUDENT IN ALL AVAILABLE SUBJECTS (student action):
  ⚡ USE THIS when the student wants to REGISTER / ENROLL themselves in courses.
  Arabic triggers (any of these): "سجلني في المواد", "سجلني في المواد المتاحة",
    "سجل لي كل المواد", "عايز أسجل مواد", "سجلني", "اعملي تسجيل",
    "ابدأ التسجيل", "سجل في كل المواد", "سجلني في الترم ده"
  English triggers: "register me for courses", "enroll me in subjects",
    "sign me up for courses", "auto enroll", "register for available courses"
  → POST /api/Enrollments/auto-enroll  (NO body needed — JWT identifies the student)
  method: "POST"
  params: {{}}
  ⛔ STUDENT ROLE ONLY. Never call this for doctor or admin.
  ✅ Backend automatically finds all open offerings for the student's batch/dept/group.

── G. ANALYTICS / COUNTS / AGGREGATIONS ───────────
Keywords: "كام", "عدد", "كم", "توزيع", "أكتر", "تحليل", "إحصاء",
          "how many", "count", "total", "distribution", "most", "analytics",
          "أكتر مادة عليها تسجيل", "doctor workload", "student distribution"

G1. SYSTEM-WIDE SUMMARY (best for "how many X in total" + top departments + top subjects):
  → GET /api/Analytics/summary

G2. STUDENT COUNT BY BATCH (all batches ranked):
  → GET /api/Analytics/student-count-by-batch

G3. STUDENT COUNT BY DEPARTMENT (all departments ranked):
  → GET /api/Analytics/student-count-by-department

G4. DOCTOR WORKLOAD (offering count + student count per doctor):
  Arabic triggers: "workload الدكاترة", "توزيع المواد على الدكاترة", "أكتر دكتور عنده مواد"
  → GET /api/Analytics/doctor-workload  params: departmentId={{departmentId}} (optional)

G5. MOST ENROLLED SUBJECTS (top N):
  → GET /api/Analytics/top-enrolled-subjects  params: top=10

G6. OFFERING ENROLLMENT STATS (fill-rate, avg grade per offering):
  Arabic triggers: "احصائيات المواد المفتوحة", "نسبة امتلاء الشعب"
  → GET /api/Analytics/offering-enrollment-stats  params: departmentId=, batchId=, page=1, size=20

G7. FULL STRUCTURE / HIERARCHY:
  → GET /api/Colleges/full-structure  (complete university hierarchy with counts)

G8. SIMPLE DASHBOARD TOTALS:
  → GET /api/Dashboard

── H. GRADES / GPA ────────────────────────────────
Keywords: "درجة", "درجات", "GPA", "نتيجة", "grade", "result", "transcript"
- student own GPA  → GET /api/Gpa/my-gpa
- specific student → GET /api/Gpa/student/{{studentId}}

── I. SCHEDULE / TIMETABLE ─────────────────────────
Keywords: "جدول", "محاضرة", "موعد", "schedule", "timetable", "today", "النهارده", "بكرا"

FOR STUDENT:
  Today → GET /api/Schedule/batch/{{batchId}}/today
  Tomorrow (dayNum=(today+1)%7, Sun=0) → GET /api/Schedule/batch/{{batchId}}/day/{{dayNum}}
  Full week → GET /api/Schedule/batch/{{batchId}}

FOR DOCTOR:
  Today → GET /api/Schedule/my-today
  Full week → GET /api/Schedule/my-schedule
  ⛔ NEVER use batch schedule for doctor role

── J. EXAMS ────────────────────────────────────────
Keywords: "امتحان", "اختبار", "exam", "quiz"
  My exams → GET /api/Exams/my-exams
  By offering → GET /api/Exams/by-offering/{{offeringId}}
  Enrolled exams → GET /api/Exams/my-enrolled-exams

── K. ATTENDANCE ───────────────────────────────────
Keywords: "حضور", "غياب", "attendance", "absent"
  → GET /api/Attendance/student/{{studentId}}/report

── L. COMPLAINTS ───────────────────────────────────
Keywords: "شكوى", "شكاوى", "complaint"
  Admin view → GET /api/Complaints/all
  Doctor view → GET /api/Complaints/my-reports
  My complaints (student) → GET /api/Complaints/my-complaints

── M. MATERIALS / LECTURE FILES ────────────────────
Keywords: "ملف", "محاضرة", "material", "lecture file", "مادة تعليمية"
  → GET /api/Materials/by-offering/{{offeringId}}

── N. STRUCTURE (COLLEGES / DEPARTMENTS / BATCHES) ─
  All colleges → GET /api/Colleges
  Full structure → GET /api/Colleges/full-structure
  Departments → GET /api/Departments
  Departments by college → GET /api/Departments/by-college/{{collegeId}}
  Batches → GET /api/Batches
  Batches by department → GET /api/Batches/by-department/{{departmentId}}
  Groups → GET /api/Groups
  Groups by batch → GET /api/Groups/by-batch/{{batchId}}

  ⚠️ /api/Colleges/by-code/{{code}} expects SHORT STRING CODE (e.g. "ENG"), NEVER a ULID.

── O. DASHBOARD / SYSTEM STATS ─────────────────────
Keywords: "dashboard", "احصائيات", "overview", "stats", "نظرة عامة"
  → GET /api/Dashboard

── P. ACADEMIC YEARS / SEMESTERS ───────────────────
Keywords: "سنة دراسية", "فصل دراسي", "academic year", "semester"
  → GET /api/AcademicYears
  By academic year → GET /api/Semesters/by-academic-year/{{academicYearId}}

── Q. REGULATION / ACADEMIC ROADMAP (HIGHEST PRIORITY FOR STUDENTS) ─
Keywords: "لائحة", "لوائح", "خطة دراسية", "خارطة طريق", "regulation",
          "roadmap", "academic plan", "study plan",
          "مواد الترم", "مواد الفصل", "مواد السنة",
          "المواد اللي هسجلها", "المواد المقترحة", "ايه المواد",
          "كام ساعة خلصت", "ساعات معتمدة", "الساعات الباقية",
          "رسبت في", "مواد راسب", "مواد باقية", "مواد خلصتها",
          "هل انا في المسار", "تقدمي الأكاديمي", "وضعي الأكاديمي",
          "الترم الجاي", "المواد القادمة", "ايه اللي باقيلي",
          "credit hours", "remaining subjects", "passed subjects",
          "academic progress", "what subjects", "next semester subjects",
          "failed subjects", "must retake"

Q1. STUDENT'S FULL ACADEMIC ROADMAP (use for almost ALL regulation questions):
  ⚡ THIS IS THE PRIMARY ENDPOINT FOR ANY STUDENT QUESTION ABOUT:
    - their subjects by semester
    - their academic progress
    - credit hours completed/remaining
    - recommended next semester
    - failed/passed subjects
    - their regulation/لائحة details
  → GET /api/Regulations/my-roadmap
  method: "GET", params: {{}}
  ✅ JWT-aware — no params needed. Returns full personalized roadmap.
  ⛔ Student role ONLY.

Q2. SPECIFIC STUDENT'S REGULATION (Admin/Doctor viewing a student):
  → GET /api/Regulations/student/{{studentId}}

Q3. REGULATIONS BY DEPARTMENT (Admin):
  → GET /api/Regulations/by-department/{{departmentId}}

Q4. ALL REGULATIONS (Admin):
  → GET /api/Regulations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER INJECTION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always inject known IDs from academic_context into params (batchId, departmentId, collegeId, etc.)
- For filter endpoints: put IDs in "params" (query string), NOT path.
- For path params: substitute the real value directly into the URL string.
- NEVER leave a {{placeholder}} in the final endpoint string.
- NEVER use a ULID where a short code is expected.
- For paginated lists: always include page=1 and size=20 (or size=50 for analytics).
- Omit params that have no value — never send empty strings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R. WRITE ACTIONS (POST — execute something in the system):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ These are ACTION requests — user wants the system to DO something, not just fetch data.
Always use method: "POST" for these.

R1. STUDENT AUTO-ENROLL (student enrolling in their batch subjects):
  Arabic triggers: "سجلني", "سجل لي المواد", "اسجلني", "عايز أسجل", "اعملي تسجيل", "تسجيل المواد"
  English triggers: "enroll me", "register me", "sign me up for courses", "auto enroll"
  → POST /api/enrollments/auto-enroll
  params: {{"studentId": "{{studentId}}", "batchId": "{{batchId}}"}}
  ✅ JWT-aware — inject studentId and batchId from academic_context.

R2. CREATE EXAM (doctor generating AI exam):
  Arabic triggers: "اعمل امتحان", "انشئ امتحان", "حضّر امتحان"
  → POST /api/exams/generate-ai
  params: {{"subjectOfferingId": "{{offeringId}}", ...}}

R3. SUBMIT COMPLAINT (student submitting a complaint):
  Arabic triggers: "اشتكي", "عايز أشتكي", "قدم شكوى"
  → POST /api/ai-tools/create-complaint
  params: inject from context

R4. ADMIN: ADD/CREATE ENTITIES:
  "أضف طالب" / "add student"  → POST /api/students
  "أضف دكتور" / "add doctor"  → POST /api/doctors
  ⚠️ Always confirm required fields are in the message before executing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAIL SAFE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If no rule matches, return: {{"endpoint": "", "method": "GET", "params": {{}}}}
- NEVER hallucinate an endpoint not in the AVAILABLE ENDPOINTS list above.
- NEVER include markdown, explanations, or extra text — JSON only.

OUTPUT FORMAT (return ONLY this JSON):
{{
    "endpoint": "<full path with real substituted values>",
    "method": "GET or POST",
    "params": {{"key": "value"}}
}}
"""

_SUMMARY_PROMPT = """\
You are a university AI assistant. Real data was just fetched from the university system to answer the user's question.

USER MESSAGE: "{user_message}"
HTTP METHOD: {method}
USER ROLE: {role}
ACADEMIC CONTEXT: {academic_context}

FETCHED DATA:
{raw_response}

YOUR TASK:
Take this data and respond to the user naturally — exactly as a brilliant, knowledgeable human assistant would.
Do NOT follow a template. Do NOT use scripted phrases. Just answer the question intelligently using the data.

HARD RULES (never break these):
- Match the user's language exactly. Arabic → Arabic. English → English. Mixed → match the dominant language.
- Never show raw JSON, field names, or technical IDs to the user.
- Never invent numbers, names, or facts not present in the fetched data.
- If the data is empty → say so honestly and suggest what to try next.
- If the method was POST (an action was executed) → confirm what happened clearly and warmly.
- If the data contains numbers or lists → present them clearly, not as a paragraph of text.

OUTPUT FORMAT (return ONLY this JSON, no markdown):
{{
    "narrative": "<your natural, intelligent response to the user>",
    "suggestions": ["<logical follow-up 1>", "<logical follow-up 2>", "<logical follow-up 3>"],
    "explain_text": "<one sentence: where this data came from, no tech details>"
}}
"""


class DynamicApiModule:
    """
    Executes a dynamic endpoint selection against the allowed Swagger API,
    fetches the data, and summarizes it.

    Retry architecture (up to MAX_ATTEMPTS):
      Each failed attempt records WHY it failed and passes that history back
      to the LLM so it picks a DIFFERENT endpoint on the next try.
      Failure reasons: 403 (wrong role endpoint), 404 (not found),
      empty result (try broader endpoint), validation blocked (not in allowlist).
    """

    MAX_ATTEMPTS = 5

    def __init__(self, model_router: Any, backend_client: Any) -> None:
        self.model_router = model_router
        self.backend_client = backend_client

    # ── Public entry-point ────────────────────────────────────────────────

    async def run(
        self, input_context: AgentInput, plan: ExecutionPlan
    ) -> AgentOutput:
        import time
        start_time = time.time()

        ctx            = input_context.context or {}
        role           = ctx.get("role", "student")
        selected_model = ctx.get("selected_model", "openai/gpt-4o-mini")
        explain_mode   = ctx.get("explain", False)
        debug_mode     = ctx.get("debug", False)
        raw_ac         = ctx.get("academic_context", {})
        academic_ctx   = json.dumps(raw_ac, ensure_ascii=False)
        message        = input_context.message
        intent         = plan.intent or "backend_api_query"
        auth_header    = ctx.get("auth_header")
        schema_text    = get_allowed_endpoints_schema()

        failed_attempts: list[dict] = []   # [{endpoint, method, reason}]
        last_raw_data  = None
        winning_endpoint = ""
        winning_method   = ""

        # ── Retry loop ────────────────────────────────────────────────────
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            logger.info(
                "DynamicApiModule: attempt %d/%d user=%s message=%.60r",
                attempt, self.MAX_ATTEMPTS, input_context.user_id, message,
            )

            # 1. Route ─────────────────────────────────────────────────────
            route = await self._pick_endpoint(
                message, role, academic_ctx, schema_text,
                selected_model, failed_attempts,
            )
            if route is None:
                # LLM JSON parse failed — stop retrying
                break

            endpoint = route.get("endpoint", "")
            method   = route.get("method", "GET").upper()
            params   = route.get("params", {})

            # 2. Context-only shortcut ─────────────────────────────────────
            if not endpoint:
                logger.info("DynamicApiModule: context-only answer (attempt %d)", attempt)
                return await self._answer_from_context(message, academic_ctx, selected_model)

            # 3. Placeholder substitution ──────────────────────────────────
            endpoint, sub_error = self._substitute_placeholders(endpoint, raw_ac)
            if sub_error:
                failed_attempts.append({
                    "endpoint": endpoint, "method": method,
                    "reason": f"unresolved placeholder in URL: {sub_error}",
                })
                logger.warning("DynamicApiModule: placeholder error on attempt %d: %s", attempt, sub_error)
                continue

            # 4. Allowlist validation ───────────────────────────────────────
            if not validate_endpoint(method, endpoint):
                failed_attempts.append({
                    "endpoint": endpoint, "method": method,
                    "reason": "endpoint not in allowed list — pick a different one",
                })
                logger.warning("DynamicApiModule: blocked endpoint on attempt %d: %s %s", attempt, method, endpoint)
                continue

            # 5. Clean params + pagination defaults ────────────────────────
            clean_params = {k: v for k, v in params.items() if v not in ("", None)}
            if method == "GET":
                clean_params.setdefault("page", 1)
                clean_params.setdefault("size", 10)

            # 6. Execute backend call ──────────────────────────────────────
            logger.info("DynamicApiModule: executing %s %s params=%s", method, endpoint, clean_params)
            try:
                if method == "GET":
                    raw_data = await self.backend_client.fetch(
                        route=endpoint, auth_header=auth_header, params=clean_params
                    )
                else:
                    raw_data = await self.backend_client.post(
                        route=endpoint, payload=clean_params, auth_header=auth_header
                    )
            except Exception as exc:
                failed_attempts.append({
                    "endpoint": endpoint, "method": method,
                    "reason": f"network/backend exception: {exc}",
                })
                logger.error("DynamicApiModule: backend exception attempt %d: %s", attempt, exc)
                continue

            # 7. Evaluate response ─────────────────────────────────────────
            if isinstance(raw_data, dict) and raw_data.get("_error") == "unauthorized":
                failed_attempts.append({
                    "endpoint": endpoint, "method": method,
                    "reason": "403 Forbidden — this endpoint requires a higher role. Use a JWT-aware endpoint instead.",
                })
                logger.warning("DynamicApiModule: 403 on attempt %d: %s", attempt, endpoint)
                continue

            if isinstance(raw_data, dict) and raw_data.get("_error") == "not_found":
                failed_attempts.append({
                    "endpoint": endpoint, "method": method,
                    "reason": "404 Not Found — resource does not exist, try a list endpoint instead.",
                })
                logger.warning("DynamicApiModule: 404 on attempt %d: %s", attempt, endpoint)
                continue

            if not raw_data:
                # Empty result — record and try a broader endpoint
                failed_attempts.append({
                    "endpoint": endpoint, "method": method,
                    "reason": "returned empty result — try a broader or different endpoint",
                })
                logger.info("DynamicApiModule: empty result attempt %d: %s", attempt, endpoint)
                # Keep as candidate but try for a richer endpoint next
                last_raw_data    = raw_data
                winning_endpoint = endpoint
                winning_method   = method
                continue

            # ✅ Got real data — break out of loop
            last_raw_data    = raw_data
            winning_endpoint = endpoint
            winning_method   = method
            logger.info(
                "DynamicApiModule: success on attempt %d — %s %s",
                attempt, method, endpoint,
            )
            break

        # ── Post-loop handling ────────────────────────────────────────────
        duration = round(time.time() - start_time, 4)

        if last_raw_data is None:
            # All attempts failed
            logger.warning(
                "[AI] Intent: %s | All %d attempts failed | Duration: %ss | Failures: %s",
                intent, self.MAX_ATTEMPTS, duration, failed_attempts,
            )
            return await self._graceful_failure(
                message, role, academic_ctx, selected_model, failed_attempts
            )

        if not last_raw_data:
            # Only empty results found
            logger.info("[AI] Intent: %s | Empty data after %d attempts", intent, self.MAX_ATTEMPTS)
            return AgentOutput(
                status="success",
                response="مش لاقي أي بيانات مطابقة لطلبك في السيستم حالياً.",
                data={"endpoint_called": winning_endpoint},
            )

        # ── Summarize successful data ──────────────────────────────────────
        narrative, suggestions, explain_text = await self._summarize(
            message, winning_method, winning_endpoint,
            role, academic_ctx, last_raw_data, selected_model,
        )

        if explain_mode and explain_text:
            narrative += f"\n\nℹ️ *{explain_text}*"

        logger.info(
            "[AI] Intent: %s | Endpoint: %s | Method: %s | Attempts: %d | Duration: %ss | Status: Success",
            intent, winning_endpoint, winning_method, len(failed_attempts) + 1, duration,
        )

        metadata = {}
        if debug_mode:
            metadata = {
                "endpoint": winning_endpoint,
                "method": winning_method,
                "attempts": len(failed_attempts) + 1,
                "failed_attempts": failed_attempts,
                "execution_time_seconds": duration,
                "intent_detected": intent,
            }

        return AgentOutput(
            status="success",
            response=narrative,
            data={
                "endpoint_called": winning_endpoint,
                "method_called": winning_method,
                "raw_backend_data": last_raw_data,
                "suggestions": suggestions,
                "actions_available": suggestions,
                "debug_info": metadata,
            }
        )

    # ── Private helpers ───────────────────────────────────────────────────

    async def _pick_endpoint(
        self,
        message: str,
        role: str,
        academic_ctx: str,
        schema_text: str,
        model_id: str,
        failed_attempts: list[dict],
    ) -> dict | None:
        """Ask the LLM to pick the best endpoint, injecting failure history so it avoids repeats."""
        failure_note = ""
        if failed_attempts:
            lines = "\n".join(
                f"  - {a['method']} {a['endpoint']} → FAILED: {a['reason']}"
                for a in failed_attempts
            )
            failure_note = (
                f"\n\n⚠️ PREVIOUS ATTEMPTS THAT FAILED (do NOT repeat these):\n{lines}\n"
                "Pick a DIFFERENT endpoint that avoids the same failure modes."
            )

        prompt_content = (
            _ROUTING_PROMPT
            .replace("{schema}", schema_text)
            .replace("{message}", message)
            .replace("{role}", role)
            .replace("{academic_context}", academic_ctx)
            + failure_note
        )

        try:
            raw = await self.model_router.generate_with_messages(
                messages=[{"role": "system", "content": prompt_content}],
                model_id=model_id,
                response_format={"type": "json_object"},
            )
            return json.loads(raw)
        except Exception as exc:
            logger.error("DynamicApiModule._pick_endpoint parse error: %s", exc)
            return None

    def _substitute_placeholders(self, endpoint: str, raw_ac: dict) -> tuple[str, str]:
        """
        Replace {placeholder} tokens in endpoint with real values from academic_context.
        Returns (resolved_endpoint, error_string).  error_string is empty on success.
        """
        if "{" not in endpoint:
            return endpoint, ""

        user_code = (
            raw_ac.get("userCode") or raw_ac.get("doctorCode")
            or raw_ac.get("studentCode") or raw_ac.get("universityId")
            or raw_ac.get("staffId") or ""
        )
        substitutions = {
            "{userId}":    raw_ac.get("userId", ""),
            "{profileId}": raw_ac.get("profileId", ""),
            "{studentId}": raw_ac.get("studentId") or raw_ac.get("userId", ""),
            "{doctorId}":  raw_ac.get("doctorId") or raw_ac.get("profileId", ""),
            "{batchId}":   raw_ac.get("batchId", ""),
            "{offeringId}": raw_ac.get("subjectOfferingId", ""),
            "{id}":        raw_ac.get("profileId") or raw_ac.get("userId", ""),
            "{code}":      user_code or raw_ac.get("profileId") or raw_ac.get("userId", ""),
            "{userCode}":  user_code,
        }

        original = endpoint
        for placeholder, value in substitutions.items():
            if value:
                endpoint = endpoint.replace(placeholder, value)

        if "{" in endpoint:
            return endpoint, f"could not resolve placeholders in '{original}'"
        return endpoint, ""

    async def _answer_from_context(
        self, message: str, academic_ctx: str, model_id: str
    ) -> AgentOutput:
        """Answer directly from academic_context when no API call is needed."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful university AI assistant. "
                    "Answer the student's question using ONLY the academic context provided. "
                    "Be warm, concise, and natural. Match the student's language (Arabic/English). "
                    "NEVER invent data not present in the context."
                ),
            },
            {
                "role": "user",
                "content": f"Student question: {message}\n\nAcademic context: {academic_ctx}",
            },
        ]
        answer = await self.model_router.generate_with_messages(messages=messages, model_id=model_id)
        return AgentOutput(
            status="success",
            response=answer or "معلوماتك موجودة في الـ profile بتاعك.",
            data={"source": "academic_context"},
        )

    async def _summarize(
        self,
        message: str,
        method: str,
        endpoint: str,
        role: str,
        academic_ctx: str,
        raw_data: Any,
        model_id: str,
    ) -> tuple[str, list, str]:
        """Narrate raw backend data into natural language. Returns (narrative, suggestions, explain_text)."""
        summary_messages = [
            {
                "role": "system",
                "content": (
                    _SUMMARY_PROMPT
                    .replace("{user_message}", message)
                    .replace("{method}", method)
                    .replace("{endpoint}", endpoint)
                    .replace("{role}", role)
                    .replace("{academic_context}", academic_ctx)
                    .replace("{raw_response}", json.dumps(raw_data, ensure_ascii=False)[:3000])
                ),
            }
        ]
        try:
            payload = await self.model_router.generate_with_messages(
                messages=summary_messages,
                model_id=model_id,
                response_format={"type": "json_object"},
            )
            out = json.loads(payload)
            return (
                out.get("narrative", "تم جلب البيانات بنجاح."),
                out.get("suggestions", []),
                out.get("explain_text", ""),
            )
        except Exception:
            return "تمت العملية بنجاح.", [], ""

    async def _graceful_failure(
        self,
        message: str,
        role: str,
        academic_ctx: str,
        model_id: str,
        failed_attempts: list[dict],
    ) -> AgentOutput:
        """
        All attempts failed. Try to give a partial answer from academic_context,
        otherwise return a helpful bilingual message.
        """
        # Last resort: try to answer from context or general knowledge
        context_messages = [
            {
                "role": "system",
                "content": (
                    "أنت مساعد جامعي ذكي. الـ API مش متاح مؤقتًا.\n"
                    "المطلوب:\n"
                    "1. لو في academic_context يحتوي على إجابة للسؤال → استخدمه وأجب مباشرة.\n"
                    "2. لو مفيش context كافٍ → حاول تجاوب من معرفتك العامة بالأنظمة الجامعية.\n"
                    "3. لو السؤال محتاج بيانات حقيقية من السيستم فقط → اعتذر بجملة واحدة وقول للمستخدم يعيد المحاولة بعد ثانية.\n"
                    "لا تعتذر أكثر من جملة واحدة. لا تكشف أسباب تقنية. تكلم بلغة المستخدم."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"سؤال المستخدم: {message}\n\n"
                    f"السياق الأكاديمي المتاح: {academic_ctx}"
                ),
            },
        ]
        try:
            fallback = await self.model_router.generate_with_messages(
                messages=context_messages, model_id=model_id
            )
            return AgentOutput(
                status="partial",
                response=fallback or "في مشكلة مؤقتة — حاول تاني بعد ثواني.",
                data={"failed_attempts": failed_attempts},
            )
        except Exception:
            return AgentOutput(
                status="failed",
                response="في مشكلة مؤقتة — حاول تاني بعد ثواني.",
            )
