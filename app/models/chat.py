from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input message")
    user_id: Optional[str] = Field(None, description="Optional user identifier for context")

class AiIntent(BaseModel):
    intent_name: str = Field(..., description="The identified intent mapping to a backend tool")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters for the backend tool")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI's natural language response to the user")
    intent_executed: Optional[str] = Field(None, description="The name of the intent actually executed, if any")
    backend_data: Optional[Dict[str, Any]] = Field(None, description="Any raw data returned from the backend tool")
