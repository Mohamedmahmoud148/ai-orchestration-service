import json
from typing import Dict, Any, Optional

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger


# ── Module-level singleton ───────────────────────────────────────────────────
# Multiple callers (Agent, DynamicApiModule, etc.) used to instantiate
# MemoryStore independently. Each instance opened a fresh Redis connection
# pool and ran its own ping(), producing duplicate auth logs and wasting
# sockets. `get_memory_store()` returns a single shared instance.
_singleton: Optional["MemoryStore"] = None


def get_memory_store() -> "MemoryStore":
    """
    Return the process-wide MemoryStore singleton.

    First call constructs it. Note: callers SHOULD `await store.ping()` once
    at startup before relying on Redis (main.py does this). Subsequent
    callers from agents/modules can just import and use it.
    """
    global _singleton
    if _singleton is None:
        _singleton = MemoryStore()
    return _singleton


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
        self.redis_client = None
        self._disabled = False
        self.redis_url = self._resolve_url()

        if self.redis_url:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url, decode_responses=True
            )
            self.redis_client = redis.Redis(connection_pool=self.pool)
        else:
            self._disabled = True
            logger.warning("MemoryStore: no valid Redis URL found — running as no-op.")

    def _resolve_url(self) -> str | None:
        """
        Build Redis URL from whatever Railway provides.
        Priority:
          1. REDIS_URL (internal private network — fastest)
          2. REDIS_PUBLIC_URL (external proxy — works across projects)
          3. Build from REDISHOST + REDISPORT + REDISUSER + REDISPASSWORD
        """
        def clean(u: str) -> str | None:
            u = u.strip().strip('"').strip("'")
            return u if u.startswith(("redis://", "rediss://")) else None

        if settings.REDIS_URL:
            url = clean(settings.REDIS_URL)
            if url:
                logger.info("MemoryStore: using REDIS_URL (internal)")
                return url

        if settings.REDIS_PUBLIC_URL:
            url = clean(settings.REDIS_PUBLIC_URL)
            if url:
                logger.info("MemoryStore: using REDIS_PUBLIC_URL (public proxy)")
                return url

        if settings.REDISHOST and settings.REDISPASSWORD:
            user = settings.REDISUSER or "default"
            url = f"redis://{user}:{settings.REDISPASSWORD}@{settings.REDISHOST}:{settings.REDISPORT}"
            logger.info("MemoryStore: built URL from REDISHOST/REDISPASSWORD")
            return url

        return None

    async def ping(self) -> bool:
        """
        Test Redis at startup. If the primary URL fails with an auth error,
        automatically retry with REDIS_PUBLIC_URL before giving up.
        Disables permanently on final failure so no per-request error logs.
        """
        if not self.redis_client or self._disabled:
            return False
        try:
            await self.redis_client.ping()
            logger.info("MemoryStore: Redis OK — %s", (self.redis_url or "").split("@")[-1])
            return True
        except Exception as exc:
            logger.warning("MemoryStore: primary Redis failed (%s)", exc)

            # Try REDIS_PUBLIC_URL as fallback before giving up
            if settings.REDIS_PUBLIC_URL and settings.REDIS_PUBLIC_URL != self.redis_url:
                try:
                    fallback_url = settings.REDIS_PUBLIC_URL.strip().strip('"').strip("'")
                    if fallback_url.startswith(("redis://", "rediss://")):
                        pool = redis.ConnectionPool.from_url(fallback_url, decode_responses=True)
                        client = redis.Redis(connection_pool=pool)
                        await client.ping()
                        self.redis_client = client
                        self.redis_url = fallback_url
                        logger.info("MemoryStore: switched to REDIS_PUBLIC_URL — %s", fallback_url.split("@")[-1])
                        return True
                except Exception as exc2:
                    logger.warning("MemoryStore: public Redis also failed (%s)", exc2)

            logger.warning("MemoryStore: all Redis URLs failed — disabling, running as no-op.")
            self.redis_client = None
            self._disabled = True
            return False

    # ── Internal helpers ──────────────────────────────────────────────────

    # Auth-error strings that indicate the Redis credentials are permanently wrong.
    _AUTH_ERRORS = ("invalid username-password", "WRONGPASS", "NOAUTH", "user is disabled")

    def _is_auth_error(self, exc: Exception) -> bool:
        return any(s in str(exc) for s in self._AUTH_ERRORS)

    async def _get(self, key: str) -> Optional[Any]:
        if not self.redis_client or self._disabled:
            return None
        try:
            raw = await self.redis_client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            if self._is_auth_error(exc):
                logger.warning("MemoryStore: auth error — disabling permanently (%s)", exc)
                self.redis_client = None
                self._disabled = True
            else:
                logger.error("MemoryStore._get key=%s error=%s", key, exc)
            return None

    async def _set(self, key: str, value: Any, ttl: int) -> None:
        if not self.redis_client or self._disabled:
            return
        try:
            await self.redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        except Exception as exc:
            if self._is_auth_error(exc):
                logger.warning("MemoryStore: auth error — disabling permanently (%s)", exc)
                self.redis_client = None
                self._disabled = True
            else:
                logger.error("MemoryStore._set key=%s error=%s", key, exc)

    async def _delete(self, key: str) -> None:
        if not self.redis_client or self._disabled:
            return
        try:
            await self.redis_client.delete(key)
        except Exception as exc:
            if self._is_auth_error(exc):
                logger.warning("MemoryStore: auth error — disabling permanently (%s)", exc)
                self.redis_client = None
                self._disabled = True
            else:
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

    # ── Pending action (write-action confirmation) ────────────────────────

    async def save_pending_action(self, user_id: str, data: Dict[str, Any]) -> None:
        """
        Persist a pending write action awaiting user confirmation. TTL = 5 min.

        Structure:
          {"endpoint": "/api/...", "method": "POST", "params": {...},
           "summary": "human-readable preview"}
        """
        if not user_id:
            return
        await self._set(f"user:{user_id}:pending_action", data, self._TTL_CLARIFICATION)
        logger.info(
            "MemoryStore: saved pending_action for user_id=%s endpoint=%s",
            user_id, data.get("endpoint"),
        )

    async def get_pending_action(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        return await self._get(f"user:{user_id}:pending_action")

    async def delete_pending_action(self, user_id: str) -> None:
        if not user_id:
            return
        await self._delete(f"user:{user_id}:pending_action")
        logger.info("MemoryStore: deleted pending_action for user_id=%s", user_id)

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

    async def get_context(self, user_id: str | None) -> str:
        """
        Synthesize a plain-text context string for the Planner.
        Combines summary + last conversation memory + preferences.
        Returns empty string (never raises) so the Planner degrades gracefully.
        """
        if not user_id:
            return ""
        try:
            parts: list[str] = []

            summary = await self.get_summary(user_id)
            if summary:
                parts.append(f"Conversation summary: {summary}")

            memory = await self.get_conversation(user_id)
            if memory:
                if memory.get("last_intent"):
                    parts.append(f"Last intent: {memory['last_intent']}")
                if memory.get("last_entities"):
                    parts.append(f"Last entities: {memory['last_entities']}")
                if memory.get("last_result"):
                    parts.append(f"Last result summary: {str(memory['last_result'])[:300]}")

            prefs = await self.get_preferences(user_id)
            if prefs:
                if prefs.get("language"):
                    parts.append(f"User language preference: {prefs['language']}")
                if prefs.get("interests"):
                    parts.append(f"User interests: {', '.join(prefs['interests'][:5])}")

            return "\n".join(parts)
        except Exception as exc:
            logger.warning("MemoryStore.get_context failed for user_id=%s: %s", user_id, exc)
            return ""

    async def save_context(self, user_id: str, intent: str, entities: Any, result_summary: str = "") -> None:
        """Save the latest intent + entities after a successful agent run."""
        if not user_id:
            return
        await self.save_conversation(user_id, {
            "last_intent": intent,
            "last_entities": entities,
            "last_result": result_summary,
        })

    async def detect_and_save_language(self, user_id: str, message: str) -> None:
        """
        Detect the user's language from a message and persist it in preferences.

        Only fires if language has not been saved before for this user, or if
        the detected language differs from the stored one — cheap heuristic,
        no LLM required.
        """
        if not user_id or not message:
            return
        try:
            # Simple heuristic: if message contains Arabic Unicode block chars → Arabic
            arabic_chars = sum(1 for c in message if "؀" <= c <= "ۿ")
            detected = "ar" if arabic_chars / max(len(message), 1) > 0.2 else "en"

            prefs = await self.get_preferences(user_id) or {}
            if prefs.get("language") != detected:
                await self.save_preferences(user_id, {"language": detected})
                logger.debug(
                    "MemoryStore: language detected=%s saved for user_id=%s",
                    detected, user_id,
                )
        except Exception as exc:
            logger.warning("MemoryStore.detect_and_save_language: %s", exc)

    async def update_context(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Merge updates into existing conversation memory."""
        if not user_id:
            return
        existing = await self.get_conversation(user_id) or {}
        merged = {**existing, **updates}
        await self.save_conversation(user_id, merged)

    # ── Last seen file URL (persists across turns) ────────────────────────
    _TTL_FILE_CTX = 1_800  # 30 minutes

    async def save_file_context(self, user_id: str, file_url: str, file_name: str = "") -> None:
        """Store the last file URL shown to the user so next turn can read it."""
        if not user_id or not file_url:
            return
        await self._set(f"user:{user_id}:last_file", {
            "file_url": file_url,
            "file_name": file_name,
        }, self._TTL_FILE_CTX)

    async def get_file_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve last file URL if user asks to read it."""
        if not user_id:
            return None
        return await self._get(f"user:{user_id}:last_file")

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

    # ── Academic Profile Memory (Phase 5) ────────────────────────────────

    # TTL: 30 days
    _TTL_ACADEMIC_PROFILE = 2_592_000

    async def save_academic_profile(self, user_id: str, profile: dict) -> None:
        """
        Persist student's academic profile for cross-session context.

        Profile structure:
          {
            "weak_subjects":         [],   # list of subject names
            "strong_subjects":       [],
            "gpa":                   float | None,
            "attendance_warnings":   [],   # subjects with low attendance
            "last_recommendations":  [],   # recent advice strings
          }
        Stored with a 30-day TTL so it survives across sessions.
        """
        if not user_id:
            return
        await self._set(f"user:{user_id}:academic_profile", profile, self._TTL_ACADEMIC_PROFILE)
        logger.info(
            "MemoryStore: saved academic profile for user_id=%s gpa=%s",
            user_id, profile.get("gpa"),
        )

    async def get_academic_profile(self, user_id: str) -> dict:
        """
        Retrieve the persistent academic profile for a user.

        Returns an empty dict (never None) so callers never need a null check.
        """
        if not user_id:
            return {}
        data = await self._get(f"user:{user_id}:academic_profile")
        return data if isinstance(data, dict) else {}

    async def update_weak_subjects(self, user_id: str, subject: str) -> None:
        """
        Add a subject to the user's weak_subjects list (de-duplicated).
        Resets the 30-day TTL on the whole profile.
        """
        if not user_id or not subject:
            return
        profile = await self.get_academic_profile(user_id)
        weak = profile.get("weak_subjects", [])
        if subject not in weak:
            weak.append(subject)
            profile["weak_subjects"] = weak
            await self.save_academic_profile(user_id, profile)
            logger.info(
                "MemoryStore: added weak subject '%s' for user_id=%s",
                subject, user_id,
            )

    async def get_personalized_context(self, user_id: str) -> str:
        """
        Build a plain-text personalized context string for LLM prompt injection.

        Example output:
            "Student profile: GPA 2.8, weak in Algorithms and OS,
             attendance warning in Data Structures,
             previously recommended: review chapter 3"

        Returns empty string (never raises) so the Planner degrades gracefully.
        """
        if not user_id:
            return ""
        try:
            profile = await self.get_academic_profile(user_id)
            if not profile:
                return ""

            parts: list[str] = []

            gpa = profile.get("gpa")
            if gpa is not None:
                parts.append(f"GPA {gpa:.2f}")

            weak = profile.get("weak_subjects", [])
            if weak:
                parts.append(f"weak in {', '.join(weak[:5])}")

            strong = profile.get("strong_subjects", [])
            if strong:
                parts.append(f"strong in {', '.join(strong[:5])}")

            warnings = profile.get("attendance_warnings", [])
            if warnings:
                parts.append(f"attendance warning in {', '.join(warnings[:5])}")

            recs = profile.get("last_recommendations", [])
            if recs:
                parts.append(f"previously recommended: {'; '.join(recs[:3])}")

            return "Student profile: " + ", ".join(parts) if parts else ""
        except Exception as exc:
            logger.warning(
                "MemoryStore.get_personalized_context failed for user_id=%s: %s",
                user_id, exc,
            )
            return ""
