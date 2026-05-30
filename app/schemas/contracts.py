"""
app/schemas/contracts.py

Typed contracts for cross-module payloads — replaces the loose `dict`
that's currently passed around as `academic_context`.

Design choices:
    - extra="allow"  →  callers can still attach fields we haven't named yet.
                        This is critical for backward-compat with the .NET
                        backend that occasionally adds new context fields.
    - All fields Optional  →  the planner sometimes runs before the backend
                        responds; we never want a validation error to crash
                        the AI pipeline because of a missing key.
    - Coercion validators →  string/int normalisation done once at the edge
                        so modules never have to defensive-cast.

Usage:
    from app.schemas import AcademicContext

    # From a raw dict (validation + coercion):
    ctx = AcademicContext.model_validate(raw_dict)

    # Read with autocomplete:
    if ctx.gpa is not None and ctx.gpa < 2.0:
        ...

    # Or keep using as dict (back-compat):
    raw_dict.get("departmentName")  # still works on the original dict
"""
from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AcademicContext(BaseModel):
    """
    Profile + academic snapshot injected by the .NET backend for every
    authenticated chat request. Drives parameter auto-fill in tool calls
    and personalisation in module responses.
    """

    # Allow unknown fields so the backend can extend without breaking us
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # ── Identity ──────────────────────────────────────────────────────────
    userId:    Optional[str] = Field(None, description="System user ID")
    profileId: Optional[str] = Field(None, description="Role-specific profile ID (admin/doctor)")
    studentId: Optional[str] = Field(None, description="Student-specific profile ID")
    doctorId:  Optional[str] = Field(None, description="Doctor-specific profile ID")

    # Display names — what the LLM uses to address the user
    studentName: Optional[str] = None
    doctorName:  Optional[str] = None
    fullName:    Optional[str] = None
    name:        Optional[str] = None

    # Short university-issued codes (for path params expecting code, not ULID)
    studentCode:        Optional[str] = None
    doctorCode:         Optional[str] = None
    userCode:           Optional[str] = None
    universityStudentId: Optional[str] = None
    universityId:       Optional[str] = None

    # ── Academic structure ───────────────────────────────────────────────
    collegeId:      Optional[str] = None
    collegeName:    Optional[str] = None
    departmentId:   Optional[str] = None
    departmentName: Optional[str] = None
    batchId:        Optional[str] = None
    batchName:      Optional[str] = None
    groupId:        Optional[str] = None

    # ── Current academic state ───────────────────────────────────────────
    semester:        Optional[Union[str, int]] = None
    currentSemester: Optional[Union[str, int]] = None
    gpa:             Optional[float] = None
    GPA:             Optional[float] = None
    cgpa:            Optional[float] = None
    CGPA:            Optional[float] = None
    creditHoursCompleted: Optional[int] = None
    creditHoursRequired:  Optional[int] = None

    # ── Subject context (when user is asking about a specific subject) ──
    subjectId:        Optional[str] = None
    subjectName:      Optional[str] = None
    subjectOfferingId: Optional[str] = None
    courseId:         Optional[str] = None

    # ── Course lists (kept loose — backend shape varies) ─────────────────
    enrolledCourses: Optional[List[Any]] = None
    courses:         Optional[List[Any]] = None
    failedCourses:   Optional[List[Any]] = None
    failedSubjects:  Optional[List[Any]] = None
    passedCourses:   Optional[List[Any]] = None

    # ── File-flow continuity ─────────────────────────────────────────────
    file_url:  Optional[str] = None
    file_name: Optional[str] = None

    # ── Coercion validators ──────────────────────────────────────────────

    @field_validator("gpa", "GPA", "cgpa", "CGPA", mode="before")
    @classmethod
    def _coerce_gpa(cls, v: Any) -> Optional[float]:
        """Backend sometimes sends GPA as string ('3.45') or int — normalise."""
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator(
        "userId", "profileId", "studentId", "doctorId",
        "collegeId", "departmentId", "batchId", "groupId",
        "subjectId", "subjectOfferingId", "courseId",
        mode="before",
    )
    @classmethod
    def _coerce_id(cls, v: Any) -> Optional[str]:
        """ULIDs / GUIDs may arrive as objects — stringify defensively."""
        if v is None or v == "":
            return None
        return str(v)

    # ── Convenience helpers ──────────────────────────────────────────────

    def effective_user_id(self) -> Optional[str]:
        """Pick the best ID for path-param substitution."""
        return self.userId or self.studentId or self.doctorId or self.profileId

    def display_name(self) -> Optional[str]:
        """Best human name to address the user with."""
        return self.studentName or self.doctorName or self.fullName or self.name

    def effective_gpa(self) -> Optional[float]:
        """Pick the first GPA value present across the alias variants."""
        for v in (self.gpa, self.GPA, self.cgpa, self.CGPA):
            if v is not None:
                return v
        return None
