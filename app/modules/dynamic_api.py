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
from typing import Any, Dict, Optional, Type

from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan
from app.core.api_discovery import get_allowed_endpoints_schema, validate_endpoint
from app.core.logging import logger

_ROUTING_PROMPT = """\
You are a STRICT API Router for a university management system.
Map the user's natural language request to the SINGLE best backend API endpoint.

AVAILABLE ENDPOINTS:
{schema}

USER REQUEST: "{message}"
USER ROLE: {role}
USER ACADEMIC CONTEXT (use these IDs to fill path/query params):
{academic_context}

════════════════════════════════════════════════════
SCENARIO MAPPING RULES (apply in ORDER, stop at first match)
════════════════════════════════════════════════════

RULE 0 — IDENTITY / PROFILE (HIGHEST PRIORITY):
Keywords: "انا مين", "أنا مين", "مين انا", "من انا", "اسمي ايه", "اسمي إيه",
          "معلوماتي", "بياناتي", "بروفايلي", "who am i", "my name", "my profile"
→ Select the profile/details endpoint for the user's role, using userId from context:
  - admin   → GET /api/Admins/{{userId}}
  - doctor/professor → GET /api/Doctors/{{code_or_userId}}
  - student → GET /api/Students/{{code_or_userId}}
→ Replace the path parameter with the actual userId from academic_context.

RULE 1 — DASHBOARD / STATISTICS:
Keywords: "dashboard", "احصائيات", "إحصائيات", "نظرة عامة", "overview", "stats"
→ GET /api/Dashboard

RULE 2 — COLLEGES:
Keywords: "كلية", "كليات", "الكليات", "colleges", "faculties"
→ GET /api/Colleges

RULE 3 — DEPARTMENTS:
Keywords: "قسم", "أقسام", "اقسام", "departments"
→ GET /api/Departments  (or /api/Departments/by-college/{{id}} if collegeId known)

RULE 4 — STUDENTS:
Keywords: "طالب", "طلاب", "الطلاب", "students", "كام طالب", "عدد الطلاب"
→ GET /api/Students  (list)
→ GET /api/Students/{{code}} (individual, if userId/code known)
→ GET /api/Students/by-batch/{{batchId}} (if batchId known)

RULE 5 — DOCTORS / FACULTY:
Keywords: "دكتور", "دكاترة", "الدكاترة", "أستاذ", "doctor", "doctors", "faculty"
→ GET /api/Doctors  (list)
→ GET /api/Doctors/{{code}} (individual)
→ GET /api/Doctors/{{code}}/subjects (their subjects)

RULE 6 — SUBJECTS / COURSES:
Keywords: "مادة", "مواد", "المواد", "subject", "subjects", "course", "courses"
→ GET /api/Subjects/by-batch/{{batchId}}  (if batchId known)
→ GET /api/Subjects/{{code}}  (individual)
→ GET /api/SubjectOfferings/my-offerings  (student's offerings)

RULE 7 — GRADES / RESULTS:
Keywords: "درجة", "درجات", "نتيجة", "نتايج", "grades", "results", "marks"
→ GET /api/Gpa/my-gpa  (student's own GPA)
→ GET /api/Gpa/student/{{studentId}}  (specific student GPA)
→ GET /api/Exams/{{id}}/results  (exam results, if examId known)

RULE 8 — EXAMS:
Keywords: "امتحان", "امتحانات", "اختبار", "exam", "exams", "quiz"
→ GET /api/Exams/my-exams  (student's own exams)
→ GET /api/Exams/by-offering/{{offeringId}}  (exams for a subject)
→ GET /api/Exams/my-enrolled-exams  (enrolled exams)

RULE 9 — ATTENDANCE:
Keywords: "حضور", "غياب", "attendance", "absent", "present"
→ GET /api/Attendance/student/{{studentId}}/report

RULE 10 — COMPLAINTS:
Keywords: "شكوى", "شكاوى", "complaint", "complaints"
→ GET /api/ai-tools/get-complaints  (admin/doctor view)

RULE 11 — MATERIALS:
Keywords: "ملف", "ملفات", "مادة تعليمية", "material", "materials", "lecture"
→ GET /api/Materials/by-offering/{{offeringId}}

RULE 12 — BATCHES / GROUPS:
Keywords: "دفعة", "دفعات", "batch", "batches", "group", "groups"
→ GET /api/Batches
→ GET /api/Groups

RULE 13 — ACADEMIC YEARS / SEMESTERS:
Keywords: "سنة دراسية", "فصل", "فصول", "academic year", "semester"
→ GET /api/academic-years
→ GET /api/Semesters/by-academic-year/{{academicYearId}}

════════════════════════════════════════════════════
PARAMETERS RULES:
════════════════════════════════════════════════════
- ALWAYS inject IDs from academic_context into path parameters.
- NEVER return empty string "" for any parameter — omit it entirely if unknown.
- For list endpoints, inject: page=1, size=20 as defaults.
- For path params: replace {{placeholder}} directly in the URL string (not in "params").

FAIL SAFE:
- If NO rule matches perfectly, return "endpoint": "".
- Never guess or hallucinate an endpoint.

OUTPUT FORMAT (return ONLY this JSON, no markdown):
{{
    "endpoint": "<chosen_endpoint_path_with_real_ids>",
    "method": "GET",
    "params": {{"<query_key>": "<value>"}}
}}
"""

_SUMMARY_PROMPT = """\
You are a helpful university AI assistant. An API call was just made to fetch data to answer the user's request.

USER MESSAGE: "{user_message}"
API ENDPOINT CALLED: {method} {endpoint}
USER ROLE: {role}
USER ACADEMIC CONTEXT: {academic_context}

RAW JSON FROM BACKEND:
```json
{raw_response}
```

INSTRUCTIONS:
1. Summarize the answer completely naturally and concisely. DO NOT expose raw technical details or JSON.
2. If the user is a student, speak naturally using their name if present in the context. If they are an admin, be precise and direct.
3. If the JSON implies an error or empty data, inform the user clearly that the data could not be found.
4. Smart Suggestions: Provide 3 short (max 6 words), actionable follow-up questions the user might ask based on this data.
5. Explainability Layer: Provide a short, human-friendly 1-liner explaining where you got this data (e.g. "جبتلك البيانات دي من نظام الطلبة في السيستم"). Avoid raw endpoints.

OUTPUT FORMAT:
Return a JSON object strictly following this structure:
{{
    "narrative": "<your natural spoken response>",
    "suggestions": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"],
    "explain_text": "<explain layer message>"
}}
"""


class DynamicApiModule:
    """
    Executes a dynamic endpoint selection against the allowed Swagger API,
    fetches the data, and summarizes it.
    """

    def __init__(self, model_router: Any, backend_client: Any) -> None:
        self.model_router = model_router
        self.backend_client = backend_client

    async def run(
        self, input_context: AgentInput, plan: ExecutionPlan
    ) -> AgentOutput:
        
        import time
        start_time = time.time()
        
        ctx          = input_context.context or {}
        role         = ctx.get("role", "student")
        selected_model = ctx.get("selected_model", "openai/gpt-4o-mini")
        explain_mode = ctx.get("explain", False)
        debug_mode   = ctx.get("debug", False)
        academic_ctx = json.dumps(ctx.get("academic_context", {}), ensure_ascii=False)
        message      = input_context.message
        intent       = plan.intent or "backend_api_query"

        # 1. Fetch available endpoints
        schema_text = get_allowed_endpoints_schema()
        
        # 2. Ask model to route it
        routing_messages = [
            {
                "role": "system",
                "content": _ROUTING_PROMPT.format(
                    schema=schema_text,
                    message=message,
                    role=role,
                    academic_context=academic_ctx
                )
            }
        ]
        
        # We use a JSON Mode request for routing
        logger.info("DynamicApiModule: Requesting API routing choice from model...")
        routing_response = await self.model_router.generate_with_messages(
            messages=routing_messages,
            model_id=selected_model,
            response_format={"type": "json_object"}
        )
        
        try:
            route_data = json.loads(routing_response)
            endpoint = route_data.get("endpoint", "")
            method   = route_data.get("method", "").upper()
            params   = route_data.get("params", {})
        except Exception as exc:
            duration = round(time.time() - start_time, 4)
            logger.error(
                "[AI] Intent: %s\n[AI] Endpoint: LLM Fallback\n[AI] Method: N/A\n[AI] Status: Failed - LLM Parsing Error\n[AI] Duration: %s",
                intent, duration
            )
            return AgentOutput(
                status="failed",
                response="أنا واجهت مشكلة في تحديد البيانات المطلوبة. لو سمحت وضح طلبك تاني."
            )
            
        if not endpoint:
            duration = round(time.time() - start_time, 4)
            logger.warning(
                "[AI] Intent: %s\n[AI] Endpoint: N/A\n[AI] Method: N/A\n[AI] Status: Blocked - Empty Route\n[AI] Duration: %s",
                intent, duration
            )
            return AgentOutput(
                status="failed",
                response="مش قادر ألاقي جزء النظام الخاص بطلبك دة. ممكن توضح أكتر إنت محتاج إيه؟"
            )

        # 3. Execution Validation Layer (CRITICAL CHECK)
        if not validate_endpoint(method, endpoint):
            duration = round(time.time() - start_time, 4)
            logger.warning(
                "[AI] Intent: %s\n[AI] Endpoint: %s\n[AI] Method: %s\n[AI] Status: Blocked - Not Allowed\n[AI] Duration: %s",
                intent, endpoint, method, duration
            )
            return AgentOutput(
                status="forbidden",
                response="Requested operation is not allowed or endpoint does not exist."
            )

        # 4. Clean and Safely Process Parameters
        clean_params = {}
        for k, v in params.items():
            # Skip empty parameters to prevent .NET 400 Bad Request
            if v == "" or v is None:
                continue
            clean_params[k] = v

        # Default Pagination Injection: If it's a GET request and missing pagination, safely inject
        if method == "GET":
            if "page" not in clean_params:
                clean_params["page"] = 1
            if "size" not in clean_params:
                clean_params["size"] = 10

        # 5. Execute Backend Request
        auth_header = ctx.get("auth_header")
        logger.info("DynamicApiModule: Executing %s %s with params %s", method, endpoint, clean_params)
        
        try:
            if method == "GET":
                raw_data = await self.backend_client.fetch(
                    route=endpoint, auth_header=auth_header, params=clean_params
                )
            else:
                # Safe POSTs
                raw_data = await self.backend_client.post(
                    route=endpoint, payload=clean_params, auth_header=auth_header
                )
        except Exception as exc:
            duration = round(time.time() - start_time, 4)
            logger.error(
                "[AI] Intent: %s\n[AI] Endpoint: %s\n[AI] Method: %s\n[AI] Status: Failed - Backend Error (%s)\n[AI] Duration: %s",
                intent, endpoint, method, str(exc), duration
            )
            return AgentOutput(
                status="failed",
                response=f"مش قادر أوصل للبيانات دلوقتي، حاول تاني (Backend Error on {method} {endpoint})",
                data={"exec_route": endpoint, "error": str(exc)}
            )

        if not raw_data:
            duration = round(time.time() - start_time, 4)
            logger.info(
                "[AI] Intent: %s\n[AI] Endpoint: %s\n[AI] Method: %s\n[AI] Status: Success - Empty Data\n[AI] Duration: %s",
                intent, endpoint, method, duration
            )
            return AgentOutput(
                status="success",
                response="مش لاقي أي بيانات مطابقة لطلبك في السيستم حالياً.",
                data={"exec_route": endpoint}
            )

        # 5. Summarize Data 
        summary_messages = [
            {
                "role": "system",
                "content": _SUMMARY_PROMPT.format(
                    user_message=message,
                    method=method,
                    endpoint=endpoint,
                    role=role,
                    academic_context=academic_ctx,
                    raw_response=json.dumps(raw_data, ensure_ascii=False)[:3000] # Cap size
                )
            }
        ]
        
        logger.info("DynamicApiModule: Summarizing backend data...")
        summary_payload = await self.model_router.generate_with_messages(
            messages=summary_messages,
            model_id=selected_model,
            response_format={"type": "json_object"}
        )
        
        try:
            out_data = json.loads(summary_payload)
            narrative = out_data.get("narrative", "تم جلب البيانات بنجاح.")
            suggestions = out_data.get("suggestions", [])
            explain_text = out_data.get("explain_text", "")
        except Exception:
            narrative = "تمت العملية بنجاح."
            suggestions = []
            explain_text = ""
            
        if explain_mode and explain_text:
            narrative += f"\n\nℹ️ *{explain_text}*"
            
        duration = round(time.time() - start_time, 4)
        logger.info(
            "[AI] Intent: %s\n[AI] Endpoint: %s\n[AI] Method: %s\n[AI] Status: Success\n[AI] Duration: %ss",
            intent, endpoint, method, duration
        )
        
        # 8. Debug Mode support
        metadata = {}
        if debug_mode:
            metadata = {
                "endpoint": endpoint,
                "method": method,
                "execution_time_seconds": duration,
                "intent_detected": intent
            }
        
        return AgentOutput(
            status="success",
            response=narrative,
            data={
                "endpoint_called": endpoint,
                "method_called": method,
                "raw_backend_data": raw_data,
                "suggestions": suggestions,
                "actions_available": suggestions,
                "debug_info": metadata
            }
        )
