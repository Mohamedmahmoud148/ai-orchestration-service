"""
app/schemas/

Typed contracts shared across the AI service.

Why separated from `app/agents/schemas.py`:
    `app/agents/schemas.py` holds Planner/Executor internal DTOs that the
    AI pipeline produces and consumes. The schemas here are the OUTER
    contracts — Intent/Role enums and the AcademicContext payload — that
    we want to expose to other modules without dragging in agent internals.

Backwards compatibility:
    - `Intent` and `Role` are str-Enums. `Intent.GENERAL_CHAT == "general_chat"`
      is True. Every existing `if intent == "general_chat":` keeps working.
    - `AcademicContext` accepts arbitrary extra keys (extra="allow"), so the
      existing call sites that pass dicts with extra fields keep working.

Import surface:
    from app.schemas import Intent, Role, AcademicContext
"""
from app.schemas.contracts import AcademicContext
from app.schemas.intents import Intent, Role

__all__ = ["Intent", "Role", "AcademicContext"]
