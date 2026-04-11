import json
from typing import Dict, Any, Optional

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger


class MemoryStore:
    """
    Redis-based memory store for the AI Agent.

    Keys and TTLs
    -------------
    user:{id}:memory          1 hour   — last intent, result, entities
    user:{id}:clarification   5 min    — pending disambiguation options
    user:{id}:preferences     7 days   — language, interests, UI prefs
    user:{id}:summary         24 hours — compressed conversation summary
    """

    # TTLs (seconds)
    _TTL_MEMORY        = 3_600        # 1 hour
    _TTL_CLARIFICATION = 300          # 5 minutes
    _TTL_PREFERENCES   = 604_800      # 7 days
    _TTL_SUMMARY       = 86_400       # 24 hours

    def __init__(self):
        url = settings.REDIS_URL.strip() if settings.REDIS_URL else None
        if url:
            url = url.strip('"').strip("'")

        self.redis_url = url

        if self.redis_url and self.redis_url.startswith(("redis://", "rediss://", "unix://")):
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url, decode_responses=True
            )
            self.redis_client = redis.Redis(connection_pool=self.pool)
        else:
            self.redis_client = None
            logger.warning(
                "REDIS_URL not configured or missing scheme ('%s'). "
                "MemoryStore will act as a no-op.",
                self.redis_url,
            )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        try:
            raw = await self.redis_client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.error("MemoryStore._get key=%s error=%s", key, exc)
            return None

    async def _set(self, key: str, value: Any, ttl: int) -> None:
        if not self.redis_client:
            return
        try:
            await self.redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        except Exception as exc:
            logger.error("MemoryStore._set key=%s error=%s", key, exc)

    async def _delete(self, key: str) -> None:
        if not self.redis_client:
            return
        try:
            await self.redis_client.delete(key)
        except Exception as exc:
            logger.error("MemoryStore._delete key=%s error=%s", key, exc)

    # ── Conversation memory ───────────────────────────────────────────────

    async def get_conversation(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load the user's conversation memory (intent, result, entities)."""
        if not user_id:
            return None
        return await self._get(f"user:{user_id}:memory")

    async def save_conversation(self, user_id: str, data: Dict[str, Any]) -> None:
        """Save conversation memory with a 1-hour TTL."""
        if not user_id:
            return
        await self._set(f"user:{user_id}:memory", data, self._TTL_MEMORY)

    # ── Clarification (disambiguation) state ─────────────────────────────

    async def save_clarification(self, user_id: str, data: Dict[str, Any]) -> None:
        """
        Persist a pending clarification with a 5-minute TTL.

        Structure:
          {"options": [...], "original_intent": "...", "step_context": {...}}
        """
        if not user_id:
            return
        await self._set(f"user:{user_id}:clarification", data, self._TTL_CLARIFICATION)
        logger.info(
            "MemoryStore: saved clarification for user_id=%s (%d options)",
            user_id, len(data.get("options", [])),
        )

    async def get_clarification(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load a pending clarification. Returns None if none exists."""
        if not user_id:
            return None
        return await self._get(f"user:{user_id}:clarification")

    async def delete_clarification(self, user_id: str) -> None:
        """Delete a clarification key after the user resolves the selection."""
        if not user_id:
            return
        await self._delete(f"user:{user_id}:clarification")
        logger.info("MemoryStore: deleted clarification for user_id=%s", user_id)

    # ── User preferences ──────────────────────────────────────────────────

    async def get_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load stored user preferences.

        Expected structure:
          {
            "language":  "ar" | "en",
            "interests": ["databases", "networks"],
            "timezone":  "Africa/Cairo",
          }
        Returns None when no preferences have been saved.
        """
        if not user_id:
            return None
        return await self._get(f"user:{user_id}:preferences")

    async def save_preferences(
        self, user_id: str, prefs: Dict[str, Any]
    ) -> None:
        """
        Persist user preferences with a 7-day (rolling) TTL.

        Merges with existing preferences so callers can update individual fields:
          await store.save_preferences(uid, {"language": "ar"})
        does not wipe out previously stored "interests".
        """
        if not user_id:
            return
        existing = await self.get_preferences(user_id) or {}
        merged = {**existing, **prefs}
        await self._set(f"user:{user_id}:preferences", merged, self._TTL_PREFERENCES)
        logger.info("MemoryStore: saved preferences for user_id=%s keys=%s", user_id, list(prefs.keys()))

    # ── Conversation summary (compressed long-term memory) ────────────────

    async def get_summary(self, user_id: str) -> Optional[str]:
        """
        Retrieve the compressed conversation summary for the user.

        Returns the summary string, or None if not yet generated.
        The summary is created by a background task when the conversation
        history exceeds the configured threshold.
        """
        if not user_id:
            return None
        data = await self._get(f"user:{user_id}:summary")
        if isinstance(data, dict):
            return data.get("summary")
        return data  # str or None

    async def save_summary(self, user_id: str, summary: str) -> None:
        """
        Persist a compressed conversation summary with a 24-hour TTL.

        Called by the background summarisation task in Agent after the
        conversation history exceeds the threshold.
        """
        if not user_id or not summary:
            return
        await self._set(
            f"user:{user_id}:summary",
            {"summary": summary},
            self._TTL_SUMMARY,
        )
        logger.info(
            "MemoryStore: saved conversation summary for user_id=%s (%d chars)",
            user_id, len(summary),
        )
