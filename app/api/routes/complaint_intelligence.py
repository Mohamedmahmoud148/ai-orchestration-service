from fastapi import APIRouter, Depends, Request
from app.models.complaint_analysis import AiAnalyzeComplaintRequest, AiComplaintAnalysisResponse
from app.services.complaint_analyzer_service import ComplaintAnalyzerService
from openai import AsyncOpenAI
from app.core.config import settings

router = APIRouter()

def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "University AI System",
            "X-Title": "University AI Agent",
        },
    )

def get_complaint_analyzer(client: AsyncOpenAI = Depends(get_openai_client)) -> ComplaintAnalyzerService:
    return ComplaintAnalyzerService(client)

@router.post("/ai/analyze-complaint", response_model=AiComplaintAnalysisResponse, tags=["AI Complaint Intelligence"])
async def analyze_complaint(
    request: AiAnalyzeComplaintRequest,
    service: ComplaintAnalyzerService = Depends(get_complaint_analyzer)
):
    """
    Analyzes a student complaint to determine category, severity, sentiment,
    and groups it into a duplicate cluster if necessary.
    """
    return await service.analyze_complaint(request)
