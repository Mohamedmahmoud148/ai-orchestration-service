from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint suitable for container platforms like Railway."""
    return {"status": "ok", "service": "fastapi-ai-service"}
