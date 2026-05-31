"""
app/schemas/intents.py

Strongly typed enums for AI intents and user roles.

Why str-Enum:
    Inheriting from (str, Enum) means `Intent.GENERAL_CHAT == "general_chat"`
    returns True. This lets us migrate gradually — every existing site that
    checks against a string literal keeps working, AND new code can use the
    enum members for autocomplete + IDE rename-safety.

Migration plan:
    1. New code → use Intent.X / Role.X
    2. Existing string literals → leave alone; they still match the enum
    3. When refactoring a file, replace its string literals with enum refs
    4. After all callers migrated, enable Pydantic strict validation on the
       few fields that currently accept Optional[str]

Authority:
    This list MUST stay in sync with:
      - app/core/rbac.py     (ROLE_PERMISSIONS, get_denial_message)
      - app/agents/executor.py (_MODULE_CLASS_MAP)
      - app/agents/planner.py  (VALID_INTENTS)
    When you add a new intent, update all three OR the planner will reject
    the new intent, the executor will fail to route it, and RBAC will deny it.
"""
from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    """Every intent the planner is allowed to produce."""

    GENERAL_CHAT         = "general_chat"
    SUMMARIZATION        = "summarization"
    GENERATE_EXAM        = "generate_exam"
    RESULT_QUERY         = "result_query"
    FILE_EXTRACTION      = "file_extraction"
    COMPLAINT_SUBMIT     = "complaint_submit"
    COMPLAINT_SUMMARY    = "complaint_summary"
    FILE_PROCESSING      = "file_processing"
    CV_ANALYSIS          = "cv_analysis"
    ACADEMIC_ADVICE      = "academic_advice"
    MATERIAL_EXPLANATION = "material_explanation"
    MATERIAL_QA          = "material_qa"
    BACKEND_API_QUERY    = "backend_api_query"
    REGULATION           = "regulation"
    ACTION_EXECUTE       = "action_execute"
    ASSIGNMENT_QUERY     = "assignment_query"
    STUDY_PLAN           = "study_plan"

    @classmethod
    def values(cls) -> set[str]:
        """Set of all valid intent string values — used for validation."""
        return {member.value for member in cls}

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.values()


class Role(str, Enum):
    """User roles. Matches what .NET sends after `.ToLower()`."""

    STUDENT    = "student"
    DOCTOR     = "doctor"
    ADMIN      = "admin"
    SUPERADMIN = "superadmin"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}

    @classmethod
    def safe(cls, value: str | None) -> "Role":
        """Coerce unknown/missing role to STUDENT (most restrictive)."""
        if value:
            try:
                return cls(value)
            except ValueError:
                pass
        return cls.STUDENT
