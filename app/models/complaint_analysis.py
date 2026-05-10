from pydantic import BaseModel
from typing import Optional

class AiAnalyzeComplaintRequest(BaseModel):
    complaintId: str
    message: str
    targetType: str
    targetId: str

class AiComplaintAnalysisResponse(BaseModel):
    sentimentScore: float
    category: str
    severity: str
    summary: str
    duplicateGroupId: Optional[str] = None
    recommendedAction: str
