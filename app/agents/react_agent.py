"""
react_agent.py — ReAct Agent (gpt-4o-mini + native function calling)

Replaces the rigid Planner → keyword-overrides → Module pipeline.

Loop (max 6 iterations):
  Think → call tool(s) → observe result → think → ... → final answer

The model autonomously picks which backend endpoints to call, how many
times, and in what order.  No keyword matching.  No intent classification.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.core.api_discovery import get_allowed_endpoints_schema, validate_endpoint
from app.core.logging import logger

if TYPE_CHECKING:
    from app.agents.execution_context import ExecutionContext

_MODEL = "openai/gpt-4o-mini"
_MAX_ITERATIONS = 6

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


# ── System prompt builder ─────────────────────────────────────────────────────

def _build_system_prompt(context: "ExecutionContext") -> str:
    schema = get_allowed_endpoints_schema()

    user_ctx = context.academic_context or {}
    ctx_parts = []
    if context.user_id:
        ctx_parts.append(f"userId={context.user_id}")
    if context.role:
        ctx_parts.append(f"role={context.role}")
    for key in ("batchId", "departmentId", "collegeId", "studentId", "batchCode"):
        if val := user_ctx.get(key):
            ctx_parts.append(f"{key}={val}")

    ctx_line = " | ".join(ctx_parts)

    return f"""أنت مساعد ذكي ومتخصص لنظام إدارة الجامعة. مهمتك تجيب بدقة تامة بناءً على البيانات الحقيقية.

## الجلسة الحالية
{ctx_line}

## قواعد العمل (صارمة)
1. **لا تخمّن أبداً** — إذا احتجت بيانات، استخدم call_backend_api فوراً.
2. **ابحث بذكاء ومثابرة** — إذا أخفق endpoint ما، جرّب endpoint آخر ذا صلة.
3. **يمكنك استدعاء الأدوات عدة مرات** — جمّع كل البيانات اللازمة قبل الإجابة.
4. **حلّل النتائج** — لا تكتفِ بعرض البيانات الخام، قدّم تحليلاً واضحاً ومفيداً.
5. **اللغة** — استخدم نفس لغة المستخدم بالضبط (عربي بعربي، إنجليزي بإنجليزي).
6. **للوثائق الأكاديمية** — أي سؤال عن اللائحة/دليل الطالب/المناهج → استخدم read_regulation_pdf.
7. **لا تقل "لا أستطيع"** — إذا لم تجد بيانات، اشرح بالضبط ماذا حاولت وما البديل.
8. **الأمان** — لا تستدعِ إلا الـ endpoints الموجودة في القائمة أدناه.

## نقاط النهاية المتاحة في الباكيند
{schema}"""


# ── ReactAgent ────────────────────────────────────────────────────────────────

class ReactAgent:
    """
    ReAct-style agent using gpt-4o-mini native function calling.

    Replaces: Planner → ToolRegistry → Executor pipeline.
    Keeps   : memory loading/saving in agent.py (infrastructure layer).
    """

    def __init__(self, openrouter_client: Any, model_router: Any) -> None:
        self.client = openrouter_client          # AsyncOpenAI (OpenRouter)
        self.model_router = model_router         # for non-tool LLM calls

    async def run(self, context: "ExecutionContext") -> str:
        """
        Execute the ReAct loop and return the final response string.
        Writes selected_model and selected_tool onto context for the response envelope.
        """
        context.set_model(_MODEL)
        context.set_tool("react_agent")

        messages = _build_messages(context)

        logger.info(
            "[ReactAgent] START user_id=%s role=%s message=%.100r",
            context.user_id, context.role, context.message,
        )
        t_start = time.perf_counter()

        for iteration in range(1, _MAX_ITERATIONS + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=_MODEL,
                    messages=messages,
                    tools=_ALL_TOOLS,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.error(
                    "[ReactAgent] LLM error iteration=%d — %s", iteration, exc
                )
                return "حدث خطأ أثناء المعالجة. حاول مرة أخرى."

            choice = response.choices[0]
            finish = choice.finish_reason

            logger.debug(
                "[ReactAgent] iteration=%d finish_reason=%s",
                iteration, finish,
            )

            # ── Final answer ──────────────────────────────────────────────
            if finish in ("stop", "end_turn") or (
                finish is None and not getattr(choice.message, "tool_calls", None)
            ):
                answer = choice.message.content or ""
                elapsed = round(time.perf_counter() - t_start, 3)
                logger.info(
                    "[ReactAgent] END iterations=%d duration_s=%s len=%d",
                    iteration, elapsed, len(answer),
                )
                return answer

            # ── Tool calls ────────────────────────────────────────────────
            if getattr(choice.message, "tool_calls", None):
                messages.append(_serialize_msg(choice.message))

                for tc in choice.message.tool_calls:
                    result = await _dispatch_tool(tc, context, self.model_router)
                    # Truncate large payloads to prevent context overflow
                    result_str = json.dumps(
                        result, ensure_ascii=False, default=str
                    )[:6000]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

        logger.warning("[ReactAgent] max iterations (%d) reached", _MAX_ITERATIONS)
        return "تعذّر إتمام الطلب بعد عدة محاولات. جرّب صياغة مختلفة."


# ── Message builder ───────────────────────────────────────────────────────────

def _build_messages(context: "ExecutionContext") -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(context)}
    ]
    for turn in (context.history or [])[-8:]:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content[:800]})
    msgs.append({"role": "user", "content": context.message})
    return msgs


# ── Tool dispatcher ───────────────────────────────────────────────────────────

async def _dispatch_tool(tool_call: Any, context: "ExecutionContext", model_router: Any) -> Any:
    fn = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid JSON in tool arguments"}

    logger.info(
        "[ReactAgent] tool=%s args=%.200s", fn, str(args)
    )

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
    # RBAC: only doctors and admins can generate exams
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
