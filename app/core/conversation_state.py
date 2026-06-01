"""
app/core/conversation_state.py  —  Layer 5: Structured Conversation State

Replaces the flat "last_intent + last_entities" string with a typed
object that enables systematic pronoun/coreference resolution.

Stored in Redis: user:{id}:conv_state  TTL: 2 hours

The entity_stack is the core mechanism.  Every time the user discusses
a concrete entity (a regulation, an exam, a course, a complaint), it is
pushed onto the stack.  Short follow-up messages ("اشرحها", "continue",
"use the previous one") are resolved by reading the stack top rather
than asking the LLM to guess.

Design:
  - Pure dataclass, JSON-serializable via to_dict() / from_dict()
  - No LLM calls, no I/O
  - Backward compatible: MemoryStore.get_conv_state() returns None
    (not an empty ConversationState) when nothing is stored, so callers
    can tell "first message" from "resumed session"
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ── Entity type labels ─────────────────────────────────────────────────────────
# These are the types stored on the entity stack so pronoun resolution
# can pick the right intent when the user says "it" / "ها".
ENTITY_INTENT_MAP: dict[str, str] = {
    "regulation":  "regulation",
    "exam":        "generate_exam",
    "material":    "material_explanation",
    "complaint":   "complaint_submit",
    "course":      "material_explanation",
    "result":      "result_query",
    "assignment":  "assignment_query",
    "study_plan":  "study_plan",
    "action":      "action_execute",
}


@dataclass
class EntityFrame:
    """
    A single entity on the conversation stack.

    type        — one of the ENTITY_INTENT_MAP keys
    name        — human-readable label ("fbn", "Data Structures", "Complaint #3")
    intent      — the intent this entity maps to when referenced pronominally
    params      — any parameters associated with the entity (offeringId, etc.)
    """
    type: str
    name: str
    intent: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EntityFrame":
        return cls(
            type=d.get("type", ""),
            name=d.get("name", ""),
            intent=d.get("intent", "general_chat"),
            params=d.get("params", {}),
        )


@dataclass
class ConversationState:
    """
    Full conversation state for one user session.

    entity_stack
        LIFO stack of EntityFrames.  Pushed when a module produces a
        concrete artifact.  Pronoun resolution reads stack[-1].

    current_topic
        Human-readable label of what we're discussing now
        ("regulation:fbn", "exam:os_midterm", etc.)

    current_intent
        Last resolved intent label.

    awaiting_clarification
        True when the planner has sent a clarification question and is
        waiting for the user to answer before proceeding.

    clarification_for_intent
        Which intent we will resume with once the user answers.

    awaiting_confirmation
        True when the action guard has sent a confirmation prompt.

    pending_action_intent / pending_action_params
        What we're about to execute once the user confirms.

    turn_count
        Monotonically increasing per-session counter.
    """
    current_topic: Optional[str] = None
    current_intent: Optional[str] = None
    entity_stack: list[dict] = field(default_factory=list)   # serialized EntityFrames
    awaiting_clarification: bool = False
    clarification_for_intent: Optional[str] = None
    awaiting_confirmation: bool = False
    pending_action_intent: Optional[str] = None
    pending_action_params: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0

    # ── Stack helpers ─────────────────────────────────────────────────────────

    def push_entity(self, frame: EntityFrame) -> None:
        """Push a new entity to the top of the stack (max depth 5)."""
        self.entity_stack.append(frame.to_dict())
        if len(self.entity_stack) > 5:
            self.entity_stack.pop(0)  # drop oldest

    def top_entity(self) -> Optional[EntityFrame]:
        """Return the most recent entity, or None if stack is empty."""
        if self.entity_stack:
            return EntityFrame.from_dict(self.entity_stack[-1])
        return None

    def resolve_pronoun(self) -> Optional[str]:
        """
        Return the intent the most recent entity maps to.
        Used when a message is detected as a pronoun reference.
        """
        top = self.top_entity()
        return top.intent if top else None

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "current_topic": self.current_topic,
            "current_intent": self.current_intent,
            "entity_stack": self.entity_stack,
            "awaiting_clarification": self.awaiting_clarification,
            "clarification_for_intent": self.clarification_for_intent,
            "awaiting_confirmation": self.awaiting_confirmation,
            "pending_action_intent": self.pending_action_intent,
            "pending_action_params": self.pending_action_params,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationState":
        return cls(
            current_topic=d.get("current_topic"),
            current_intent=d.get("current_intent"),
            entity_stack=d.get("entity_stack", []),
            awaiting_clarification=d.get("awaiting_clarification", False),
            clarification_for_intent=d.get("clarification_for_intent"),
            awaiting_confirmation=d.get("awaiting_confirmation", False),
            pending_action_intent=d.get("pending_action_intent"),
            pending_action_params=d.get("pending_action_params", {}),
            turn_count=d.get("turn_count", 0),
        )


# ── Pronoun / coreference detection ──────────────────────────────────────────
# These patterns catch short messages that are clearly referring back
# to something discussed in a previous turn.

import re

_PRONOUN_PATTERNS_AR: list[re.Pattern] = [
    re.compile(r"^(اشرحها|اشرحه|شرحها|شرحه|لخصها|لخصه|لخصيها)$", re.UNICODE),
    re.compile(r"^(ابعتها|ابعته|ابعت|بعتها|بعته)$", re.UNICODE),
    re.compile(r"^(اقراها|اقراه|اقرا|قراها)$", re.UNICODE),
    re.compile(r"^(استمر|كمّل|كمل|تكمل)$", re.UNICODE),
    re.compile(r"^(اعمل زي اللي فوق|زي اللي فوق|نفس الحاجة|نفس الكلام)$", re.UNICODE),
    re.compile(r"^(ايه اللي فيها|ايه اللي فيه|ايه فيها|ايه فيه)$", re.UNICODE),
    re.compile(r"^(هاتها|هاته|جيبها|جيبه)$", re.UNICODE),
]

_PRONOUN_PATTERNS_EN: list[re.Pattern] = [
    re.compile(r"^(explain it|explain that|explain this)$", re.IGNORECASE),
    re.compile(r"^(send it|send that|send this)$", re.IGNORECASE),
    re.compile(r"^(continue|keep going|go on|proceed)$", re.IGNORECASE),
    re.compile(r"^(use the previous (one|exam|plan|result))$", re.IGNORECASE),
    re.compile(r"^(same (one|thing|exam|complaint|request) (again|please|again please)?)$", re.IGNORECASE),
    re.compile(r"^(do (it|that|the same))$", re.IGNORECASE),
    re.compile(r"^(what('s| is) in it|what does it say)$", re.IGNORECASE),
    re.compile(r"^(summarize it|summarise it|summary please)$", re.IGNORECASE),
]


def is_pronoun_reference(message: str) -> bool:
    """
    Return True when the message is almost certainly a pronoun reference
    to something discussed in a prior turn.

    Heuristic: message is very short (≤ 6 words) AND matches one of the
    pronoun patterns above.
    """
    stripped = message.strip()
    word_count = len(stripped.split())
    if word_count > 8:
        return False

    for pat in _PRONOUN_PATTERNS_AR:
        if pat.match(stripped):
            return True
    for pat in _PRONOUN_PATTERNS_EN:
        if pat.match(stripped):
            return True
    return False


def build_entity_context_note(state: ConversationState) -> str:
    """
    Build a compact plain-text note injected into the classification prompt.

    Example:
        [Active context]: The user is discussing a regulation called 'fbn'.
        Pronouns like 'it', 'ها', 'اللي فوق' refer to this entity.
        If the user refers to it, use intent = regulation.
    """
    top = state.top_entity()
    if top is None:
        return ""

    return (
        f"\n[Active context]: The user is currently discussing a "
        f"{top.type} called '{top.name}'. "
        f"Pronouns like 'it', 'ها', 'اشرحها', 'اللي فوق', 'continue', 'same one' "
        f"refer to this entity. "
        f"If the current message is a pronoun reference, use intent = {top.intent}."
    )
