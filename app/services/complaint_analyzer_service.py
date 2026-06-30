import json
import httpx
from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.complaint_analysis import AiAnalyzeComplaintRequest, AiComplaintAnalysisResponse

COMPLAINT_ANALYSIS_SYSTEM_PROMPT = """
You are an advanced AI Complaint Intelligence Agent for a University.
Your task is to analyze a student's complaint and output a STRICT JSON structure.

Guidelines:
- category: Determine if it's "teaching_quality", "exam_difficulty", "technical_issue", "harassment", "grading_issue", "attendance_issue", or "general".
- severity: Must be "low", "medium", "high", or "critical". Abuse, harassment, or severe system outages are critical.
- sentimentScore: Float between -1.0 (extremely negative) and 1.0 (extremely positive). Usually complaints are negative (< 0).
- summary: A concise 1-sentence summary of the core issue. MUST NOT include the student's name or identity.
- recommendedAction: A short sentence on what the administration or target should do. e.g., "Review exam questions", "Contact doctor", "Investigate server logs".

You must only output the raw JSON object, no markdown blocks, no other text.
"""

class ComplaintAnalyzerService:
    def __init__(self, openai_client):
        self.openai_client = openai_client

    async def analyze_complaint(self, request: AiAnalyzeComplaintRequest) -> AiComplaintAnalysisResponse:
        user_message = f"""
        Complaint ID: {request.complaint_id}
        Target Type: {request.target_type}
        Target ID: {request.target_id}
        Complaint Message: {request.message}
        """

        try:
            # We use the provided openrouter/openai client
            response = await self.openai_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": COMPLAINT_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            
            # Check for duplicates / clusters via semantic similarity (Mocked here)
            duplicate_group_id = await self._find_duplicate_cluster(request.target_type, request.target_id, request.message)

            return AiComplaintAnalysisResponse(
                sentiment_score=float(result_json.get("sentimentScore", -0.5)),
                category=result_json.get("category", "general"),
                severity=result_json.get("severity", "medium"),
                summary=result_json.get("summary", "User submitted a complaint."),
                duplicate_group_id=duplicate_group_id,
                recommended_action=result_json.get("recommendedAction", "Manual review required.")
            )

        except Exception as e:
            logger.error("Error analyzing complaint: %s", str(e))
            return AiComplaintAnalysisResponse(
                sentiment_score=-0.5,
                category="general",
                severity="medium",
                summary="Fallback summary due to AI service error.",
                duplicate_group_id=None,
                recommended_action="Manual review required."
            )

    async def _find_duplicate_cluster(self, target_type: str, target_id: str, message: str) -> Optional[str]:
        # Mock logic for clustering
        if "slow" in message.lower() or "bad" in message.lower():
            return "01HXXXXXDUPLICATECLUSTERID"
        return None
