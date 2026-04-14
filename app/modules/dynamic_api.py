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
You are an intelligent API Router mapping a user's natural language request to a backend API endpoint.

AVAILABLE ENDPOINTS:
{schema}

USER REQUEST: "{message}"
USER ROLE: {role}
USER ACADEMIC CONTEXT:
{academic_context}

INSTRUCTIONS:
1. Match the user's request to the closest matching endpoint in the AVAILABLE ENDPOINTS list.
2. Extract any required query parameters or URL path variables from the user's message or context.
3. NEVER hallucinate endpoints not in the list.
4. If no endpoint fits, or if the user asks for something destructive like DELETE, return an empty endpoint string.

OUTPUT FORMAT:
Return a JSON object:
{{
    "endpoint": "/api/Students/count",
    "method": "GET",
    "params": {{"any_query_keys": "values"}}
}}
For path parameters (e.g. /api/Admins/{{id}}), replace {{id}} with the actual value directly in the endpoint string (e.g. "/api/Admins/123"), and do NOT put it in "params".
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
            model=selected_model,
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

        # 4. Execute Backend Request
        auth_header = ctx.get("auth_header")
        logger.info("DynamicApiModule: Executing %s %s", method, endpoint)
        
        try:
            if method == "GET":
                raw_data = await self.backend_client.fetch(
                    route=endpoint, auth_header=auth_header, params=params
                )
            else:
                # Safe POSTs
                raw_data = await self.backend_client.post(
                    route=endpoint, payload=params, auth_header=auth_header
                )
        except Exception as exc:
            duration = round(time.time() - start_time, 4)
            logger.error(
                "[AI] Intent: %s\n[AI] Endpoint: %s\n[AI] Method: %s\n[AI] Status: Failed - Backend Error (%s)\n[AI] Duration: %s",
                intent, endpoint, method, str(exc), duration
            )
            return AgentOutput(
                status="failed",
                response="مش قادر أوصل للبيانات دلوقتي، حاول تاني"
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
            model=selected_model,
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
