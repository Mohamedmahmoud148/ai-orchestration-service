import json
from typing import Dict, Any, Optional

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger

class MemoryStore:
    """
    Redis-based memory store for the AI Agent.
    Stores contextual information (intent, result, entities) with a 1 hour TTL.
    """

    def __init__(self):
        url = settings.REDIS_URL.strip() if settings.REDIS_URL else None
        if url:
            url = url.strip('"').strip("'")
            
        self.redis_url = url
        
        if self.redis_url and self.redis_url.startswith(("redis://", "rediss://", "unix://")):
            # Reusable connection pool
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True
            )
            self.redis_client = redis.Redis(connection_pool=self.pool)
        else:
            self.redis_client = None
            logger.warning(f"REDIS_URL not configured or missing scheme ('{self.redis_url}'). MemoryStore will act as a no-op.")

        self.ttl = 3600  # 1 hour

    async def get_conversation(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the user's conversation memory from Redis.
        Fails safely and returns None on error or missing memory.
        """
        if not self.redis_client or not user_id:
            return None

        key = f"user:{user_id}:memory"
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("Failed to load memory for user_id=%s: %s", user_id, e)
            return None

    async def save_conversation(self, user_id: str, data: Dict[str, Any]) -> None:
        """
        Save the conversation memory (intent, result, entities) to Redis.
        """
        if not self.redis_client or not user_id:
            return

        key = f"user:{user_id}:memory"
        try:
            payload = json.dumps(data)
            await self.redis_client.setex(key, self.ttl, payload)
        except Exception as e:
            logger.error("Failed to save memory for user_id=%s: %s", user_id, e)

    # ── Clarification (disambiguation) state ─────────────────────────────

    async def save_clarification(self, user_id: str, data: Dict[str, Any]) -> None:
        """
        Persist a pending clarification for the user with a 5-minute TTL.

        Structure:
          {
            "options": [...],
            "original_intent": "...",
            "step_context": {...}
          }
        """
        if not self.redis_client or not user_id:
            return

        key = f"user:{user_id}:clarification"
        try:
            payload = json.dumps(data)
            await self.redis_client.setex(key, 300, payload)  # 5 minutes
            logger.info("Saved clarification for user_id=%s (%d options)", user_id, len(data.get("options", [])))
        except Exception as e:
            logger.error("Failed to save clarification for user_id=%s: %s", user_id, e)

    async def get_clarification(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a pending clarification from Redis. Returns None if none exists.
        """
        if not self.redis_client or not user_id:
            return None

        key = f"user:{user_id}:clarification"
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("Failed to get clarification for user_id=%s: %s", user_id, e)
            return None

    async def delete_clarification(self, user_id: str) -> None:
        """
        Delete a clarification key after the user resolves the selection.
        """
        if not self.redis_client or not user_id:
            return

        key = f"user:{user_id}:clarification"
        try:
            await self.redis_client.delete(key)
            logger.info("Deleted clarification key for user_id=%s", user_id)
        except Exception as e:
            logger.error("Failed to delete clarification for user_id=%s: %s", user_id, e)

