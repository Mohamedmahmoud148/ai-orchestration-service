from fastapi import APIRouter, HTTPException, Request
from app.models.chat import ChatRequest, ChatResponse
from app.services.ai_service import ai_service
from app.services.tool_registry import tool_registry
from app.services.backend_client import backend_client
from app.core.logging import logger

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, tags=["AI Chat"])
async def chat_endpoint(request: ChatRequest, fastapi_request: Request):
    """
    1. Receives chat message.
    2. Uses AI Service to determine intent.
    3. If a tool maps to this intent, calls the .NET backend.
    4. Returns a structured JSON response.
    """
    auth_header = fastapi_request.headers.get("Authorization")
    if not auth_header:
        logger.warning(f"Unauthorized chat attempt. Missing Authorization header for user: {request.user_id}")
        raise HTTPException(status_code=401, detail="Authorization header missing")

    logger.info(f"Received chat request from user: {request.user_id}")
    
    # 1. Determine Intent using Gemini
    intent = await ai_service.determine_intent(request.message)
    logger.info(f"Identified intent: {intent.intent_name} with parameters: {intent.parameters}")
    
    if intent.intent_name == "Error":
        raise HTTPException(status_code=500, detail=intent.parameters.get("error", "Unknown AI error"))

    # 2. Check Tool Registry for matching route
    backend_route = tool_registry.get_route_for_intent(intent.intent_name)
    
    backend_data = None
    response_text = ""
    
    # 3. Call .NET Backend if tool is registered
    if backend_route:
        logger.info(f"Routing intent '{intent.intent_name}' to backend route '{backend_route}'")
        backend_data = await backend_client.execute_tool(
            route=backend_route,
            parameters=intent.parameters,
            auth_header=auth_header,
            user_id=request.user_id
        )
        
        if backend_data and "error" in backend_data:
            response_text = f"I tried to execute the tool for {intent.intent_name} but encountered an error: {backend_data['error']}"
        else:
            response_text = f"Successfully executed backend tool for intent '{intent.intent_name}'."
    else:
        # No specific tool mapped, treat as simple chat or unhandled intent
        if intent.intent_name == "DefaultChat":
            response_text = "I received your message, but it didn't match any specific action I perform. How can I assist you further?"
        else:
            response_text = f"I understood you want to perform '{intent.intent_name}', but I don't have a backend tool configured for that yet."
    
    # 4. Return structured response
    return ChatResponse(
        response=response_text,
        intent_executed=intent.intent_name if backend_route else None,
        backend_data=backend_data
    )
