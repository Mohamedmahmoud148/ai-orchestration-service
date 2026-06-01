"""
app/services/companion_memory.py  —  AI Companion Memory Service

Extends the existing MemoryStore with companion-specific features:
  - Rich academic profile (weak subjects, learning style, goals)
  - Study session aggregates (for progress reports)
  - Engagement scoring
  - Proactive trigger state (prevent duplicate notifications)

All data is stored in Redis with appropriate TTLs.
The .NET backend calls the companion APIs to persist profiles to PostgreSQL
for long-term storage; Redis is the fast cache for the AI pipeline.

Redis key schema:
  user:{id}:companion_profile    7 days   — full companion profile
  user:{id}:study_sessions       24h      — recent session aggregates
  user:{id}:engagement_metrics   24h      — computed engagement scores
  user:{id}:followup_triggers    7 days   — deduplication for follow-up engine
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.logging import logger
from app.services.memory_store import MemoryStore, get_memory_store


class CompanionMemoryService:
    """
    High-level memory API for the AI companion pipeline.

    Wraps the existing MemoryStore to add companion-specific keys
    without touching the existing Redis structure.
    """

    _TTL_COMPANION_PROFILE  = 604_800   # 7 days
    _TTL_STUDY_SESSIONS     = 86_400    # 24 hours
    _TTL_ENGAGEMENT         = 86_400    # 24 hours
    _TTL_FOLLOWUP_TRIGGERS  = 604_800   # 7 days

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self._store = memory_store or get_memory_store()

    # ── Companion Profile ─────────────────────────────────────────────────

    async def get_companion_profile(self, user_id: str) -> dict:
        """
        Returns the cached companion profile.  Empty dict means "not yet built."

        Schema:
          {
            learning_style: str,
            current_goal: str,
            weak_subjects: [str],
            strong_subjects: [str],
            preferred_study_time: str,
            streak_days: int,
            engagement_score: float,
            last_interaction: ISO str,
          }
        """
        data = await self._store._get(f"user:{user_id}:companion_profile")
        return data if isinstance(data, dict) else {}

    async def save_companion_profile(self, user_id: str, profile: dict) -> None:
        """Persist companion profile with 7-day TTL."""
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self._store._set(
            f"user:{user_id}:companion_profile",
            profile,
            self._TTL_COMPANION_PROFILE,
        )

    async def update_companion_profile(
        self, user_id: str, updates: dict
    ) -> dict:
        """Merge updates into existing profile."""
        existing = await self.get_companion_profile(user_id)
        merged = {**existing, **updates}
        await self.save_companion_profile(user_id, merged)
        return merged

    async def add_weak_subject(self, user_id: str, subject: str) -> None:
        """Add a subject to the weak subjects list (deduped)."""
        profile = await self.get_companion_profile(user_id)
        weak = profile.get("weak_subjects", [])
        if subject and subject not in weak:
            weak.append(subject)
            profile["weak_subjects"] = weak[-10:]  # keep last 10
            await self.save_companion_profile(user_id, profile)

    async def remove_weak_subject(self, user_id: str, subject: str) -> None:
        """Remove a subject from weak list (student improved)."""
        profile = await self.get_companion_profile(user_id)
        weak = profile.get("weak_subjects", [])
        if subject in weak:
            weak.remove(subject)
            profile["weak_subjects"] = weak
            profile.setdefault("strong_subjects", [])
            if subject not in profile["strong_subjects"]:
                profile["strong_subjects"].append(subject)
            await self.save_companion_profile(user_id, profile)

    # ── Study Session Aggregates ──────────────────────────────────────────

    async def record_study_session(
        self,
        user_id: str,
        topic: str,
        session_type: str,
        duration_minutes: int,
        accuracy_percent: float,
    ) -> None:
        """
        Record a completed study session into the rolling 24-hour aggregate.
        Used by the progress intelligence module.
        """
        key = f"user:{user_id}:study_sessions"
        existing: dict = await self._store._get(key) or {
            "sessions_count": 0,
            "total_minutes": 0,
            "accuracy_sum": 0.0,
            "accuracy_count": 0,
            "topics": [],
            "streak_days": 0,
            "last_session": None,
            "flashcards_reviewed": 0,
        }

        existing["sessions_count"] = existing.get("sessions_count", 0) + 1
        existing["total_minutes"] = existing.get("total_minutes", 0) + duration_minutes

        if accuracy_percent > 0:
            existing["accuracy_sum"] = existing.get("accuracy_sum", 0.0) + accuracy_percent
            existing["accuracy_count"] = existing.get("accuracy_count", 0) + 1

        topics = existing.get("topics", [])
        if topic and topic not in topics:
            topics.append(topic)
            existing["topics"] = topics[-20:]

        existing["last_session"] = datetime.now(timezone.utc).isoformat()

        # Compute derived metrics
        if existing["accuracy_count"] > 0:
            existing["avg_accuracy"] = round(
                existing["accuracy_sum"] / existing["accuracy_count"], 1
            )

        await self._store._set(key, existing, self._TTL_STUDY_SESSIONS)
        logger.debug(
            "CompanionMemory.record_study_session: user=%s topic=%s acc=%.1f",
            user_id, topic, accuracy_percent,
        )

    async def get_study_summary(self, user_id: str) -> dict:
        """Return the rolling study session summary for the last 24 hours."""
        data = await self._store._get(f"user:{user_id}:study_sessions")
        return data if isinstance(data, dict) else {}

    # ── Engagement Scoring ────────────────────────────────────────────────

    async def compute_and_cache_engagement_score(
        self, user_id: str
    ) -> float:
        """
        Compute a 0–100 engagement score from study activity and profile data.

        Components:
          - Streak bonus (up to 30 pts): consistency is key
          - Session frequency (up to 30 pts): active usage
          - Accuracy trend (up to 20 pts): improving performance
          - Activity recency (up to 20 pts): was active this week
        """
        profile  = await self.get_companion_profile(user_id)
        sessions = await self.get_study_summary(user_id)

        streak     = min(profile.get("streak_days", 0), 15)
        streak_pts = streak * 2.0  # 0–30

        session_count = min(sessions.get("sessions_count", 0), 10)
        session_pts   = session_count * 3.0  # 0–30

        avg_acc = sessions.get("avg_accuracy", 0) or 0
        acc_pts = avg_acc * 0.2   # 0–20

        last_str = profile.get("last_interaction") or sessions.get("last_session")
        recency_pts = 0.0
        if last_str:
            try:
                last = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
                days_since = (datetime.now(timezone.utc) - last).days
                recency_pts = max(0, 20 - days_since * 3)
            except Exception:
                pass

        score = min(100.0, streak_pts + session_pts + acc_pts + recency_pts)
        score = round(score, 1)

        # Cache
        await self._store._set(
            f"user:{user_id}:engagement_metrics",
            {"score": score, "computed_at": datetime.now(timezone.utc).isoformat()},
            self._TTL_ENGAGEMENT,
        )
        return score

    # ── Follow-Up Trigger Deduplication ──────────────────────────────────

    async def has_recent_followup(
        self, user_id: str, trigger_type: str, window_days: int = 7
    ) -> bool:
        """
        Returns True if a follow-up of this type was already sent
        within the last `window_days` days.
        """
        key = f"user:{user_id}:followup_triggers"
        triggers: dict = await self._store._get(key) or {}
        if trigger_type not in triggers:
            return False
        try:
            last = datetime.fromisoformat(triggers[trigger_type])
            return (datetime.now(timezone.utc) - last).days < window_days
        except Exception:
            return False

    async def mark_followup_sent(
        self, user_id: str, trigger_type: str
    ) -> None:
        """Mark a follow-up as sent to prevent duplicates."""
        key = f"user:{user_id}:followup_triggers"
        triggers: dict = await self._store._get(key) or {}
        triggers[trigger_type] = datetime.now(timezone.utc).isoformat()
        await self._store._set(key, triggers, self._TTL_FOLLOWUP_TRIGGERS)

    # ── Learning Style Inference ──────────────────────────────────────────

    async def infer_and_update_learning_style(
        self,
        user_id: str,
        session_type: str,
        completed: bool,
        duration_minutes: int,
    ) -> None:
        """
        Simple heuristic: if the student consistently prefers certain session
        types, update their learning style in the profile.
        """
        profile = await self.get_companion_profile(user_id)
        style_counts = profile.get("_style_counts", {})
        style_counts[session_type] = style_counts.get(session_type, 0) + (1 if completed else 0)

        # Infer dominant style
        if style_counts:
            dominant = max(style_counts, key=style_counts.get)
            inferred_style = {
                "quiz":           "practical",
                "flashcards":     "reading",
                "concept_review": "visual",
                "exam_prep":      "practical",
            }.get(dominant, "mixed")

            profile["_style_counts"] = style_counts
            profile["learning_style"] = inferred_style
            await self.save_companion_profile(user_id, profile)


# Module-level singleton
_companion_memory: Optional[CompanionMemoryService] = None


def get_companion_memory() -> CompanionMemoryService:
    global _companion_memory
    if _companion_memory is None:
        _companion_memory = CompanionMemoryService()
    return _companion_memory
