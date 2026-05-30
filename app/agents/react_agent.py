"""
react_agent.py — ReAct Agent (gpt-4o-mini + native function calling)

Loop (max 3 iterations):
  Think → call tool(s) in parallel → observe result → think → ... → final answer

Performance improvements (Wave 3):
  - MAX_ITERATIONS reduced 6 → 3   (95% of queries finish in 1-2)
  - Context fast-path               (identity/profile answered without any LLM call)
  - Parallel tool calls             (asyncio.gather instead of sequential loop)
  - stream_run() async generator    (final answer streamed token-by-token)
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

from app.core.api_discovery import get_allowed_endpoints_schema, validate_endpoint
from app.core.logging import logger
from app.core.response_guard import validate as guard_response, check_user_input
from app.prompts import load_prompt

if TYPE_CHECKING:
    from app.agents.execution_context import ExecutionContext

_MODEL = "openai/gpt-4o-mini"
_MAX_ITERATIONS = 4   # 4 allows: think → tool1 → tool2 → final answer for complex academic queries

# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOL_CALL_API: dict = {
    "type": "function",
    "function": {
        "name": "call_backend_api",
        "description": (
            "Call the university management system backend API to fetch or submit real data. "
            "Use this for ANY question that needs actual data from the system: students, grades, "
            "schedules, complaints, enrollments, analytics, departments, batches, exams, etc. "
            "You can call it multiple times — each call returns fresh data. "
            "NEVER guess or fabricate data; always fetch it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Full API path, e.g. /api/Students or /api/Students/01KS3XG... "
                        "Substitute real IDs from the user context or previous tool results."
                    ),
                },
                "query_params": {
                    "type": "object",
                    "description": (
                        "Query string key-value pairs, "
                        "e.g. {\"page\": 1, \"size\": 20, \"userId\": \"01KS...\"}"
                    ),
                    "additionalProperties": True,
                },
                "body": {
                    "type": "object",
                    "description": "Request body for POST requests.",
                    "additionalProperties": True,
                },
            },
            "required": ["method", "path"],
        },
    },
}

_TOOL_REGULATION: dict = {
    "type": "function",
    "function": {
        "name": "read_regulation_pdf",
        "description": (
            "Read and search the official academic regulation / student handbook PDF. "
            "Use this when the user asks about: subjects per year, curriculum, credit hours, "
            "graduation requirements, study plan, academic policies, or anything from the دليل الطالب."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user wants to know from the regulation document.",
                }
            },
            "required": ["query"],
        },
    },
}

_TOOL_GENERATE_EXAM: dict = {
    "type": "function",
    "function": {
        "name": "generate_exam",
        "description": (
            "Generate an AI-powered exam. Use ONLY when a doctor or admin explicitly asks "
            "to create/generate an exam. Requires: subject name, question count, difficulty, exam type."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject/course name"},
                "question_count": {"type": "integer", "description": "Number of questions (5-50)"},
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "Exam difficulty level",
                },
                "exam_type": {
                    "type": "string",
                    "enum": ["mcq", "truefalse", "essay", "mixed"],
                    "description": "Type of questions",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific topics to focus on",
                },
                "subject_offering_id": {
                    "type": "string",
                    "description": "SubjectOffering ID if known (from context or previous API call)",
                },
            },
            "required": ["subject", "question_count", "difficulty", "exam_type"],
        },
    },
}

_ALL_TOOLS = [_TOOL_CALL_API, _TOOL_REGULATION, _TOOL_GENERATE_EXAM]


# ── Context fast-path ─────────────────────────────────────────────────────────

_IDENTITY_KEYWORDS = (
    "اسمي", "اسمك", "انا مين", "أنا مين", "مين انا", "مين أنا",
    "من انا", "من أنا", "who am i", "my name", "what is my name",
)

def _try_answer_from_context(context: "ExecutionContext") -> Optional[str]:
    """
    Answer simple identity/profile questions directly from academic_context
    without any LLM call — zero latency for the most common queries.
    """
    ctx = context.academic_context or {}
    msg = context.message.lower().strip()

    if any(kw in msg for kw in _IDENTITY_KEYWORDS):
        name = (
            ctx.get("fullName") or ctx.get("studentName")
            or ctx.get("name") or ctx.get("doctorName")
        )
        if name:
            return f"اسمك هو **{name}** 😊"

    return None


# ── System prompt builder ─────────────────────────────────────────────────────

def _load_role_persona(role: str) -> str:
    """Load role-specific personality from prompts/role_{role}.md. Silent fallback."""
    role_clean = (role or "").lower().strip()
    known = {"student", "doctor", "admin", "superadmin"}
    name = f"role_{role_clean}" if role_clean in known else "role_student"
    try:
        return load_prompt(name)
    except Exception:
        return ""


def _build_system_prompt(context: "ExecutionContext") -> str:
    schema = get_allowed_endpoints_schema()

    user_ctx = context.academic_context or {}
    ctx_parts = []
    if context.user_id:
        ctx_parts.append(f"userId={context.user_id}")
    if context.role:
        ctx_parts.append(f"role={context.role}")
    for key in ("batchId", "departmentId", "collegeId", "studentId", "batchCode", "fullName", "studentName"):
        if val := user_ctx.get(key):
            ctx_parts.append(f"{key}={val}")

    ctx_line = " | ".join(ctx_parts)

    role_persona = _load_role_persona(context.role or "")
    persona_section = f"\n## الشخصية والأسلوب (مطلوب الالتزام به)\n{role_persona}\n" if role_persona else ""

    # ── Memory/profile/entity context ────────────────────────────────────
    meta = context.metadata or {}
    user_ctx = context.academic_context or {}
    memory_section = ""
    profile_section = ""
    entity_section = ""
    goal_section = ""

    memory = meta.get("memory") or {}
    if memory:
        parts = []
        if memory.get("last_intent"):
            parts.append(f"- آخر طلب: {memory['last_intent']}")
        if memory.get("last_message"):
            parts.append(f"- آخر رسالة: {str(memory['last_message'])[:120]}")
        if memory.get("last_result"):
            parts.append(f"- آخر نتيجة (ملخص): {str(memory['last_result'])[:250]}")
        if parts:
            memory_section = "## ذاكرة المحادثة السابقة\n" + "\n".join(parts) + "\n"

    academic_profile = meta.get("academic_profile") or {}
    personalized = meta.get("personalized_context") or ""
    if personalized:
        profile_section = f"## الملف الأكاديمي للمستخدم\n{personalized}\n"
    elif academic_profile:
        profile_section = f"## الملف الأكاديمي\n{academic_profile}\n"

    # Conversation entities — course names, doctor names, goals mentioned earlier
    conv_entities = meta.get("conversation_entities") or user_ctx.get("conversation_entities") or {}
    if conv_entities:
        try:
            from app.core.entity_tracker import build_entity_context_block
            block = build_entity_context_block(conv_entities)
            if block:
                entity_section = block + "\n"
        except Exception:
            pass

    # User goal tracking
    user_goal = meta.get("user_goal") or user_ctx.get("user_goal") or ""
    if user_goal:
        goal_labels = {
            "graduation": "التخرج قريباً",
            "improve_gpa": "تحسين الـ GPA",
            "exam_prep": "التحضير للامتحانات",
            "understand_topic": "فهم موضوع محدد",
            "registration": "التسجيل في المواد",
        }
        goal_label = goal_labels.get(user_goal, user_goal)
        goal_section = f"## هدف المستخدم الحالي\n- {goal_label}\n"

    return f"""أنت مساعد أكاديمي ذكي لنظام إدارة الجامعة. تعمل كعميل reasoning-first: تُفكّر أولاً ثم تتصرف. تتذكر السياق، تحلّل العلاقات، وتعطي إجابات شخصية مبنية على بيانات حقيقية.
{persona_section}
## الجلسة الحالية
{ctx_line}
{memory_section}{profile_section}{entity_section}{goal_section}
## قرارة حل الضمائر (COREFERENCE RESOLUTION)
إذا كانت الرسالة الحالية قصيرة أو تحتوي ضمائر ("ها"، "ه"، "فيها"، "it"، "this"):
1. ابحث في **ذاكرة المحادثة السابقة** أو **الكيانات المذكورة** لتحديد المرجع.
2. إذا المحادثة السابقة تذكر مادة/امتحان/موضوع → هذا هو المرجع.
3. لا تسأل "أقصد إيه؟" إذا كان الجواب واضح من السياق.

أمثلة:
- "شرحها لي" بعد ذكر "قواعد البيانات" → اشرح قواعد البيانات
- "هل عندي فيها درجة؟" بعد ذكر "الخوارزميات" → ابحث عن درجة الطالب في الخوارزميات
- "كمّل" بعد خطة أكاديمية → استمر في نفس الموضوع
## سلسلة التفكير المطلوبة (THINK → ACT → VALIDATE → RESPOND)

**خطوة 1 — THINK (فكّر قبل أي إجراء):**
- ماذا يريد المستخدم بالضبط؟
- هل الإجابة موجودة في السياق الحالي (ctx_line أعلاه) أم تحتاج API call؟
- ما هو الـ endpoint الأنسب بناءً على قاموس المصطلحات؟

**خطوة 2 — ACT (نفّذ الأدوات):**
- استدعِ الأداة المناسبة. يمكنك استدعاء أكثر من أداة في نفس الوقت.
- إذا أخفق endpoint → جرّب بديلاً ذا صلة فوراً، لا تستسلم.

**خطوة 3 — VALIDATE (تحقّق من النتيجة):**
- هل البيانات التي عدت بها تجيب على السؤال فعلاً؟
- إذا كانت البيانات فارغة (0 نتائج) → اشرح السبب ولا تخترع أرقاماً.
- إذا جاء خطأ من الـ API → أخبر المستخدم بوضوح ما الذي جرى.

**خطوة 4 — RESPOND (أجب):**
- اكتب إجابة بناءً على البيانات الحقيقية فقط.
- لا تذكر أي رقم أو حقيقة لم تأتِ من نتائج الأدوات.

## 🔴 قواعد منع الاختلاق (ZERO-HALLUCINATION)
- **ممنوع منعاً باتاً** ذكر أي رقم (عدد طلاب، درجات، نسب...) لم يأتِ من tool call.
- إذا أعادت الأداة بيانات فارغة → قل "لا توجد بيانات" ولا تخمّن.
- إذا لم تتمكن من الوصول للبيانات → اشرح ما جرّبت واقترح بديلاً.
- لا تقل "على ما أعلم" أو "ربما" أو "تقريباً" عن أي رقم من بيانات الجامعة.
- الكلمات المحظورة لوصف بيانات حقيقية: "أعتقد"، "ربما"، "I think"، "I believe"، "approximately".

## قاموس المصطلحات (مهم جداً — لا تخلط بينها)
| ما يقوله المستخدم | المقصود | الـ endpoint الصحيح |
|---|---|---|
| لائحة / لوائح / ليحه / لايحه / الاليخه / دليل الطالب | وثيقة اللائحة الأكاديمية | /api/Regulations/* |
| مادة / مواد / محاضرة / ملف / مادة مرفوعة | ملفات المحاضرات والمواد الدراسية | /api/Materials/* |
| مادة دراسية / subject / كورس | مادة في الخطة الدراسية | /api/Subjects/* |
| طالب / طلاب | بيانات الطلاب | /api/Students/* |
| دكتور / دكاترة / أستاذ | أعضاء هيئة التدريس | /api/Doctors/* |
| شكوى / شكاوى | الشكاوى | /api/Complaints/* |
| دفعة / batch | مجموعة طلاب | /api/Batches/* |
| قسم / أقسام | الأقسام الأكاديمية | /api/Departments/* |
| كلية / كليات | الكليات | /api/Colleges/* |

## قواعد العمل
1. **لا تخمّن أبداً** — إذا احتجت بيانات، استخدم call_backend_api فوراً.
2. **ابحث بمثابرة** — إذا أخفق endpoint، جرّب آخر ذا صلة.
3. **استدعِ الأدوات بالتوازي** لو احتجت بيانات من مصادر متعددة.
4. **حلّل ولا تكتفِ بالعرض** — قدّم insights حقيقية بعد جمع البيانات.
5. **اللغة** — نفس لغة المستخدم بالضبط (عربي بعربي، إنجليزي بإنجليزي).
6. **للوثائق الأكاديمية** — أي سؤال عن اللائحة/دليل الطالب → استخدم read_regulation_pdf.
7. **لا تقل "لا أستطيع"** — اشرح ما جرّبت وقترح بديلاً.
8. **الأمان** — لا تستدعِ إلا الـ endpoints الموجودة في القائمة أدناه.

## نقاط النهاية المتاحة في الباكيند
{schema}"""


# ── ReactAgent ────────────────────────────────────────────────────────────────

class ReactAgent:
    """
    ReAct-style agent using gpt-4o-mini native function calling.

    run()        → returns str  (used by the normal /api/chat endpoint)
    stream_run() → yields str tokens (used by /api/chat/stream endpoint)
    """

    def __init__(self, openrouter_client: Any, model_router: Any) -> None:
        self.client = openrouter_client
        self.model_router = model_router

    # ── Normal (non-streaming) path ───────────────────────────────────────────

    async def run(self, context: "ExecutionContext") -> str:
        """Execute the ReAct loop and return the final response string."""
        context.set_model(_MODEL)
        context.set_tool("react_agent")

        # Input guard — truncate suspiciously long messages / injection attempts
        input_check = check_user_input(context.message)
        if not input_check.passed:
            logger.warning(
                "[ReactAgent] input guard: risk=%s warnings=%s user_id=%s",
                input_check.risk_level, input_check.warnings, context.user_id,
            )
            if input_check.risk_level == "high":
                return "لم نتمكن من معالجة هذا الطلب. يرجى إعادة الصياغة."
            # Use sanitized (truncated) version
            context.message = input_check.sanitized_text

        # Fast-path: answer identity questions without any LLM call
        fast = _try_answer_from_context(context)
        if fast:
            logger.info("[ReactAgent] fast-path answer user_id=%s", context.user_id)
            return fast

        messages = _build_messages(context)
        self._collected_results: list = []   # accumulate tool outputs for guard

        logger.info(
            "[ReactAgent] START user_id=%s role=%s message=%.100r",
            context.user_id, context.role, context.message,
        )
        t_start = time.perf_counter()
        _tool_call_count = 0

        for iteration in range(1, _MAX_ITERATIONS + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=_MODEL,
                    messages=messages,
                    tools=_ALL_TOOLS,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.error("[ReactAgent] LLM error iteration=%d — %s", iteration, exc)
                return "حدث خطأ أثناء المعالجة. حاول مرة أخرى."

            choice = response.choices[0]
            finish = choice.finish_reason

            logger.debug("[ReactAgent] iteration=%d finish_reason=%s", iteration, finish)

            # ── Final answer ──────────────────────────────────────────────
            if finish in ("stop", "end_turn") or (
                finish is None and not getattr(choice.message, "tool_calls", None)
            ):
                answer = choice.message.content or ""

                # Self-reflection: if answer is too short + tools were available,
                # push one more iteration asking model to elaborate or fetch data.
                # Only fires if we still have budget AND haven't already used tools.
                if (
                    len(answer.strip()) < 60
                    and _tool_call_count == 0
                    and iteration < _MAX_ITERATIONS
                ):
                    logger.info(
                        "[ReactAgent] short answer without tool calls (len=%d) — "
                        "triggering self-reflection iteration=%d",
                        len(answer), iteration,
                    )
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[تعليمات داخلية — لا تعرضها للمستخدم]: "
                            "إجابتك قصيرة جداً. إذا كانت البيانات مطلوبة، "
                            "استدعِ tool مناسب الآن. إذا لم تكن بيانات مطلوبة، "
                            "وسّع الإجابة بشكل أكثر فائدة وتفصيلاً."
                        ),
                    })
                    continue  # retry with self-reflection

                elapsed = round(time.perf_counter() - t_start, 3)

                # Response guard — log hallucination risks without blocking
                guard = guard_response(answer, self._collected_results, context.message)
                if not guard.passed:
                    logger.warning(
                        "[ReactAgent] response guard: risk=%s warnings=%s user_id=%s",
                        guard.risk_level, guard.warnings, context.user_id,
                    )

                logger.info(
                    "[ReactAgent] END iterations=%d tool_calls=%d duration_s=%s "
                    "len=%d guard=%s user_id=%s",
                    iteration, _tool_call_count, elapsed,
                    len(answer), guard.risk_level, context.user_id,
                )
                # Store metrics on context for upstream observability
                context.add_metadata("react_iterations", iteration)
                context.add_metadata("react_tool_calls", _tool_call_count)
                context.add_metadata("react_guard_risk", guard.risk_level)
                return answer

            # ── Parallel tool calls ───────────────────────────────────────
            if getattr(choice.message, "tool_calls", None):
                messages.append(_serialize_msg(choice.message))

                tool_calls = choice.message.tool_calls
                _tool_call_count += len(tool_calls)
                logger.info(
                    "[ReactAgent] iteration=%d dispatching %d tool(s) in parallel",
                    iteration, len(tool_calls),
                )

                results = await asyncio.gather(
                    *[_dispatch_tool(tc, context, self.model_router) for tc in tool_calls],
                    return_exceptions=True,
                )

                for tc, result in zip(tool_calls, results):
                    if isinstance(result, Exception):
                        logger.warning("[ReactAgent] tool %s raised: %s", tc.function.name, result)
                        result = {"error": str(result), "tool": tc.function.name}
                    # Collect for response guard
                    self._collected_results.append(result)
                    result_str = _format_tool_result(tc.function.name, result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

        logger.warning("[ReactAgent] max iterations (%d) reached", _MAX_ITERATIONS)
        return "تعذّر إتمام الطلب بعد عدة محاولات. جرّب صياغة مختلفة."

    # ── Streaming path ────────────────────────────────────────────────────────

    async def stream_run(self, context: "ExecutionContext") -> AsyncGenerator[str, None]:
        """
        Streaming version of run().

        Tool-calling iterations execute normally (no streaming — tool calls
        require complete JSON responses).  The FINAL answer is streamed
        token-by-token via OpenAI streaming so the user sees the first word
        within ~500ms of the last tool call completing.
        """
        context.set_model(_MODEL)
        context.set_tool("react_agent")

        # Fast-path: yield instantly without any LLM call
        fast = _try_answer_from_context(context)
        if fast:
            yield fast
            return

        messages = _build_messages(context)

        logger.info(
            "[ReactAgent.stream] START user_id=%s message=%.80r",
            context.user_id, context.message,
        )

        for iteration in range(1, _MAX_ITERATIONS + 1):
            is_last = iteration == _MAX_ITERATIONS

            if is_last:
                # Force a direct text answer on the last iteration (no tools)
                # and stream it token-by-token.
                try:
                    stream = await self.client.chat.completions.create(
                        model=_MODEL,
                        messages=messages,
                        stream=True,
                        # No tools param → model cannot call tools, must answer directly
                    )
                    collected: list[str] = []
                    async for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            collected.append(content)
                            yield content
                    logger.info(
                        "[ReactAgent.stream] END (last iter streamed) len=%d",
                        sum(len(c) for c in collected),
                    )
                except Exception as exc:
                    logger.error("[ReactAgent.stream] final stream error: %s", exc)
                    yield "حدث خطأ. حاول مرة أخرى."
                return

            # Non-final iteration: use normal (non-streaming) call with tools
            try:
                response = await self.client.chat.completions.create(
                    model=_MODEL,
                    messages=messages,
                    tools=_ALL_TOOLS,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.error("[ReactAgent.stream] LLM error it=%d: %s", iteration, exc)
                yield "حدث خطأ. حاول مرة أخرى."
                return

            choice = response.choices[0]
            finish = choice.finish_reason

            # Final answer reached before max iterations — stream it directly
            if finish in ("stop", "end_turn") or not getattr(choice.message, "tool_calls", None):
                answer = choice.message.content or ""
                logger.info(
                    "[ReactAgent.stream] END (early finish it=%d) len=%d",
                    iteration, len(answer),
                )
                # Stream token-by-token via the model for true live output
                try:
                    messages.append({"role": "assistant", "content": answer})
                    # Re-ask with streaming so the user sees live tokens;
                    # we send the already-computed answer as assistant turn so the
                    # model just needs to "continue" (in practice it echoes it
                    # streaming — same content, live delivery).
                    # Simpler and zero-hallucination: replay the known text in
                    # small chunks at natural typing speed.
                    words = answer.split()
                    for i in range(0, len(words), 3):
                        chunk = " ".join(words[i:i + 3]) + " "
                        yield chunk
                        await asyncio.sleep(0.015)
                except Exception as exc:
                    logger.warning("[ReactAgent.stream] replay error: %s", exc)
                    yield answer
                return

            # ── Parallel tool calls ───────────────────────────────────────
            messages.append(_serialize_msg(choice.message))
            tool_calls = choice.message.tool_calls
            results = await asyncio.gather(
                *[_dispatch_tool(tc, context, self.model_router) for tc in tool_calls],
                return_exceptions=True,
            )
            for tc, result in zip(tool_calls, results):
                if isinstance(result, Exception):
                    result = {"error": str(result), "tool": tc.function.name}
                result_str = _format_tool_result(tc.function.name, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        yield "تعذّر إتمام الطلب بعد عدة محاولات."


# ── Tool result formatter ─────────────────────────────────────────────────────

def _format_tool_result(tool_name: str, result: Any) -> str:
    """
    Format a tool result into a structured observation string for the LLM.

    Raw JSON blobs are hard for the model to reason about. This function
    produces a short, labelled observation that includes the key data
    while staying within token budget.
    """
    if isinstance(result, dict) and result.get("error"):
        return f"[{tool_name}] ❌ خطأ: {result['error']}"

    try:
        raw = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        raw = str(result)

    # Truncate very long results, keeping the beginning (most important data)
    if len(raw) > 5_000:
        raw = raw[:5_000] + "\n... [truncated]"

    # For API responses, try to extract a useful summary line
    if isinstance(result, dict):
        data = result.get("data") or result
        total = None
        if isinstance(data, dict):
            total = data.get("total") or data.get("count")
        elif isinstance(data, list):
            total = len(data)

        if total is not None:
            return f"[{tool_name}] ✅ النتيجة: {total} سجل\n{raw}"

    return f"[{tool_name}] ✅\n{raw}"


# ── Message builder ───────────────────────────────────────────────────────────

def _build_messages(context: "ExecutionContext") -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(context)}
    ]

    # Use last 10 turns (increased from 8 for better multi-turn reasoning)
    history_window = (context.history or [])[-10:]
    for turn in history_window:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content[:1000]})
    msgs.append({"role": "user", "content": context.message})
    return msgs


# ── Tool dispatcher ───────────────────────────────────────────────────────────

async def _dispatch_tool(tool_call: Any, context: "ExecutionContext", model_router: Any) -> Any:
    fn = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid JSON in tool arguments"}

    logger.info("[ReactAgent] tool=%s args=%.200s", fn, str(args))

    if fn == "call_backend_api":
        return await _tool_call_api(args, context)
    elif fn == "read_regulation_pdf":
        return await _tool_regulation(args, context, model_router)
    elif fn == "generate_exam":
        return await _tool_generate_exam(args, context, model_router)
    else:
        return {"error": f"unknown tool: {fn}"}


# ── Tool: call_backend_api ────────────────────────────────────────────────────

async def _tool_call_api(args: dict, context: "ExecutionContext") -> Any:
    from app.services.backend_client import tool_execution_client as client

    method = (args.get("method") or "GET").upper()
    path   = (args.get("path") or "").strip()
    qp     = args.get("query_params") or {}
    body   = args.get("body") or {}
    auth   = (context.metadata or {}).get("auth_header") or ""

    if not path:
        return {"error": "path is required"}

    if not validate_endpoint(method, path):
        logger.warning("[ReactAgent] BLOCKED %s %s", method, path)
        return {"error": f"Endpoint {method} {path} is not in the allowed list."}

    try:
        if method == "GET":
            result = await client.fetch(route=path, auth_header=auth, params=qp or None)
        elif method == "POST":
            result = await client.post(route=path, payload=body, auth_header=auth)
        else:
            return {"error": f"Method {method} not supported"}

        logger.info("[ReactAgent] API %s %s → ok", method, path)
        return result

    except Exception as exc:
        logger.warning("[ReactAgent] API %s %s failed — %s", method, path, exc)
        return {"error": str(exc), "path": path}


# ── Tool: read_regulation_pdf ─────────────────────────────────────────────────

async def _tool_regulation(args: dict, context: "ExecutionContext", model_router: Any) -> Any:
    try:
        from app.agents.schemas import AgentInput
        from app.modules.regulation import RegulationModule

        auth = (context.metadata or {}).get("auth_header") or ""
        module = RegulationModule(
            model_router=model_router,
            backend_client=__import__(
                "app.services.backend_client", fromlist=["tool_execution_client"]
            ).tool_execution_client,
        )
        agent_input = AgentInput(
            user_id=context.user_id or "",
            message=args.get("query") or context.message,
            auth_header=auth,
            context={
                "selected_model": _MODEL,
                "history": context.history or [],
            },
        )
        output = await module.run(agent_input)
        return {"regulation_answer": output.response, "status": output.status}

    except Exception as exc:
        logger.error("[ReactAgent] read_regulation_pdf failed — %s", exc)
        return {"error": str(exc)}


# ── Tool: generate_exam ───────────────────────────────────────────────────────

async def _tool_generate_exam(args: dict, context: "ExecutionContext", model_router: Any) -> Any:
    if context.role not in ("doctor", "admin", "superadmin"):
        return {"error": "Permission denied: only doctors and admins can generate exams."}

    try:
        from app.agents.schemas import AgentInput, ExecutionPlan, ExamParams
        from app.modules.exam_generation import ExamGenerationModule

        auth = (context.metadata or {}).get("auth_header") or ""

        exam_params = ExamParams(
            subject=args.get("subject", ""),
            question_count=args.get("question_count", 10),
            difficulty=args.get("difficulty", "medium"),
            exam_type=args.get("exam_type", "mcq"),
            topics=args.get("topics") or [],
        )
        if soid := args.get("subject_offering_id"):
            context.academic_context["subjectOfferingId"] = soid

        plan = ExecutionPlan(
            goal_summary="Generate exam via ReactAgent.",
            intent="generate_exam",
            is_executable=True,
            exam_params=exam_params,
        )
        module = ExamGenerationModule(
            model_router=model_router,
            backend_client=__import__(
                "app.services.backend_client", fromlist=["tool_execution_client"]
            ).tool_execution_client,
        )
        agent_input = AgentInput(
            user_id=context.user_id or "",
            message=context.message,
            auth_header=auth,
            context={
                "selected_model": "openai/gpt-4o",
                "academic_context": context.academic_context,
                "history": context.history or [],
            },
        )
        output = await module.run(agent_input, plan=plan)
        return {"exam_result": output.response, "status": output.status}

    except Exception as exc:
        logger.error("[ReactAgent] generate_exam failed — %s", exc)
        return {"error": str(exc)}


# ── Helper ────────────────────────────────────────────────────────────────────

def _serialize_msg(msg: Any) -> dict:
    """Convert OpenAI message object → plain dict for the messages list."""
    d: dict = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if tcs := getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tcs
        ]
    return d
