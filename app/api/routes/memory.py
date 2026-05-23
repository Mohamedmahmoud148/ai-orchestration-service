"""
app/api/routes/memory.py

Academic profile memory management endpoints.

GET    /api/memory/profile/{user_id}           — retrieve academic profile
POST   /api/memory/profile/{user_id}/update    — update profile (called by .NET risk job)
DELETE /api/memory/profile/{user_id}           — clear profile
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.services.memory_store import MemoryStore

router = APIRouter(prefix="/api/memory", tags=["Academic Memory"])

# ── Singleton memory store (same pool as Agent) ───────────────────────────────

def _get_store(request: Request) -> MemoryStore:
    """Retrieve the shared MemoryStore from app state (set in main.py lifespan)."""
    store: MemoryStore | None = getattr(request.app.state, "memory_store", None)
    if store is None:
        # Fallback: create a standalone instance (will be no-op if Redis not configured)
        return MemoryStore()
    return store


# ── Request / Response schemas ────────────────────────────────────────────────

class AcademicProfileUpdateRequest(BaseModel):
    """
    Payload sent by the .NET backend (risk job, grade calculation job) to
    update a student's persistent academic profile.
    """
    gpa: Optional[float] = Field(None, description="Current GPA (0.0 – 4.0 scale)")
    weak_subjects: Optional[List[str]] = Field(
        None, description="Subjects where student is underperforming"
    )
    strong_subjects: Optional[List[str]] = Field(
        None, description="Subjects where student excels"
    )
    attendance_warnings: Optional[List[str]] = Field(
        None, description="Subjects with attendance below threshold"
    )
    last_recommendations: Optional[List[str]] = Field(
        None, description="Recent AI-generated recommendations"
    )


class AcademicProfileResponse(BaseModel):
    user_id: str
    profile: Dict[str, Any]
    personalized_context: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/profile/{user_id}", response_model=AcademicProfileResponse)
async def get_profile(user_id: str, request: Request):
    """
    Retrieve the stored academic profile for a user.

    Called by the .NET backend or front-end to show the student's
    persistent AI-tracked academic context.
    """
    store = _get_store(request)
    profile = await store.get_academic_profile(user_id)
    context_str = await store.get_personalized_context(user_id)
    return AcademicProfileResponse(
        user_id=user_id,
        profile=profile,
        personalized_context=context_str,
    )


@router.post("/profile/{user_id}/update", response_model=AcademicProfileResponse)
async def update_profile(
    user_id: str,
    payload: AcademicProfileUpdateRequest,
    request: Request,
):
    """
    Merge-update the student's academic profile.

    Designed to be called by the .NET risk-analysis job after grade finalization
    or attendance threshold checks.  Only fields present in the payload are updated;
    omitted fields retain their existing values.
    """
    store = _get_store(request)

    existing = await store.get_academic_profile(user_id)

    # Merge: only overwrite fields that were explicitly provided
    if payload.gpa is not None:
        existing["gpa"] = payload.gpa
    if payload.weak_subjects is not None:
        existing["weak_subjects"] = payload.weak_subjects
    if payload.strong_subjects is not None:
        existing["strong_subjects"] = payload.strong_subjects
    if payload.attendance_warnings is not None:
        existing["attendance_warnings"] = payload.attendance_warnings
    if payload.last_recommendations is not None:
        existing["last_recommendations"] = payload.last_recommendations

    await store.save_academic_profile(user_id, existing)

    context_str = await store.get_personalized_context(user_id)
    logger.info(
        "[MemoryRoute] Updated academic profile for user_id=%s keys=%s",
        user_id, [k for k, v in payload.model_dump().items() if v is not None],
    )

    return AcademicProfileResponse(
        user_id=user_id,
        profile=existing,
        personalized_context=context_str,
    )


@router.delete("/profile/{user_id}")
async def delete_profile(user_id: str, request: Request):
    """
    Clear the stored academic profile for a user.

    Useful when a student resets their account or for GDPR data deletion.
    """
    store = _get_store(request)
    await store._delete(f"user:{user_id}:academic_profile")
    logger.info("[MemoryRoute] Deleted academic profile for user_id=%s", user_id)
    return {"message": f"Academic profile cleared for user {user_id}."}
