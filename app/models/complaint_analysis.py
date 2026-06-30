from pydantic import BaseModel
from typing import Optional

class AiAnalyzeComplaintRequest(BaseModel):
    complaint_id: str
    message: str
    target_type: str
    target_id: str

class AiComplaintAnalysisResponse(BaseModel):
    sentiment_score: float
    category: str
    severity: str
    summary: str
    duplicate_group_id: Optional[str] = None
    recommended_action: str
