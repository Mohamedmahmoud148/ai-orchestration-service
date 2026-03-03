"""
execution_context.py

Immutable-ish carrier object that flows through every stage of the
OrchestrationPipeline.  All stages READ from it and WRITE back to it
via the provided helper methods — no stage reaches into another stage's
internals directly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionContext:
    """
    Single source of truth for one orchestration request.

    Populated incrementally as the request moves through:
        Planner → Tool Selection → Model Routing → Executor
    """

    # ------------------------------------------------------------------ #
    #  Identity & Request                                                   #
    # ------------------------------------------------------------------ #
    user_id: str
    role: str                              # e.g. "student", "doctor", "admin"
    message: str                           # raw user message
    conversation_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    # ------------------------------------------------------------------ #
    #  Conversation history & domain context                               #
    # ------------------------------------------------------------------ #
    history: List[Dict[str, Any]] = field(default_factory=list)
    """Ordered list of past turns: [{"role": "user"|"assistant", "content": "..."}]"""

    academic_context: Dict[str, Any] = field(default_factory=dict)
    """Domain-specific context injected by the caller (syllabus, grades, …)."""

    # ------------------------------------------------------------------ #
    #  Stage outputs — written by each pipeline stage                      #
    # ------------------------------------------------------------------ #
    intent: Optional[str] = None
    """High-level intent label resolved by the Planner (e.g. 'ScheduleMeeting')."""

    selected_tool: Optional[str] = None
    """Tool name chosen by the Tool Selection stage."""

    selected_model: Optional[str] = None
    """Model identifier chosen by the Model Routing stage (e.g. 'gemini-2.5-flash')."""

    result: Optional[Any] = None
    """Final output produced by the Executor stage."""

    # ------------------------------------------------------------------ #
    #  Observability                                                        #
    # ------------------------------------------------------------------ #
    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Catch-all bag for cross-cutting concerns:
        - timing per stage
        - token counts
        - error details
        - feature flags / A-B test variants
    """

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                  #
    # ------------------------------------------------------------------ #
    def set_intent(self, intent: str) -> None:
        self.intent = intent

    def set_tool(self, tool: str) -> None:
        self.selected_tool = tool

    def set_model(self, model: str) -> None:
        self.selected_model = model

    def set_result(self, result: Any) -> None:
        self.result = result

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the context for logging or API responses."""
        return {
            "user_id": self.user_id,
            "role": self.role,
            "conversation_id": self.conversation_id,
            "message": self.message,
            "intent": self.intent,
            "selected_tool": self.selected_tool,
            "selected_model": self.selected_model,
            "result": self.result,
            "metadata": self.metadata,
        }
