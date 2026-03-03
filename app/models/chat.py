from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's raw input message")
    user_id: Optional[str] = Field(None, description="Caller's user identifier")
    role: str = Field(default="student", description="Caller's role: 'student', 'doctor', 'admin'")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID for multi-turn sessions")
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prior turns: [{\"role\": \"user\"|\"assistant\", \"content\": \"...\"}]",
    )
    academic_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific context (syllabus, grades, …)",
    )


# Kept for backward compatibility with any existing callers.
class AiIntent(BaseModel):
    intent_name: str = Field(..., description="The identified intent mapping to a backend tool")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters")


class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI's natural language response to the user")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for follow-up turns")
    intent_executed: Optional[str] = Field(None, description="Intent resolved by the pipeline")
    tool_used: Optional[str] = Field(None, description="Tool selected by the pipeline (route or 'model_only')")
    model_used: Optional[str] = Field(None, description="Model identifier selected by the pipeline")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Per-stage timing and debug info")
