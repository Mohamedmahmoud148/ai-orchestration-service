"""
app/core/rate_limiter.py

Sliding-window rate limiter: Redis-backed when available, in-memory LRU fallback.

Algorithm
---------
Redis  → ZSET per user_id:  score = timestamp, member = uuid.
         ZREMRANGEBYSCORE removes entries older than the window.
         ZCARD checks if the new count would exceed the limit.
         Atomic via a Lua script — no race conditions.

Fallback → thread-safe deque per user_id trimmed on every check.
           Not distributed (each worker has its own counter),
           but prevents runaway usage on a single instance.
"""

import time
import uuid
from collections import OrderedDict
from threading import Lock

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger

# ── Lua script — atomic sliding window in Redis ────────────────────────────────
_SLIDING_WINDOW_LUA = """
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
local member = ARGV[4]

-- remove entries outside the window
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- count remaining entries
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, window)
    return 1   -- allowed
end
return 0       -- rate limited
"""

# ── In-memory fallback (LRU dict of deques) ────────────────────────────────────
_MAX_USERS_IN_MEMORY = 10_000


class _InMemoryRateLimiter:
    """Thread-safe sliding window per user backed by an OrderedDict of lists."""

    def __init__(self):
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    def is_allowed(self, user_id: str, window_seconds: int, limit: int) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = self._store.get(user_id, [])
            # Slide the window
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= limit:
                self._store[user_id] = timestamps
                return False
            timestamps.append(now)
            self._store[user_id] = timestamps
            self._store.move_to_end(user_id)
            # Evict oldest entry if map is too large
            if len(self._store) > _MAX_USERS_IN_MEMORY:
                self._store.popitem(last=False)
            return True


_in_memory = _InMemoryRateLimiter()


class RateLimiter:
    """
    Sliding-window rate limiter.

    Usage
    -----
        limiter = RateLimiter()
        allowed = await limiter.is_allowed(user_id)
        if not allowed:
            raise HTTPException(429, "Too many requests")
    """

    def __init__(self):
        url = settings.REDIS_URL.strip() if settings.REDIS_URL else None
        if url:
            url = url.strip('"').strip("'")

        if url and url.startswith(("redis://", "rediss://", "unix://")):
            pool = redis.ConnectionPool.from_url(url, decode_responses=True)
            self._redis = redis.Redis(connection_pool=pool)
            self._script: redis.client.Script | None = None   # loaded lazily
            self._use_redis = True
            logger.info("RateLimiter: Redis backend configured (%s)", url)
        else:
            self._redis = None
            self._use_redis = False
            logger.warning(
                "RateLimiter: Redis not configured — using in-process fallback "
                "(not suitable for multi-worker deployments)."
            )

    async def _load_script(self) -> redis.client.Script:
        """Register the Lua script once and cache the SHA."""
        if self._script is None:
            self._script = self._redis.register_script(_SLIDING_WINDOW_LUA)
        return self._script

    async def is_allowed(
        self,
        user_id: str,
        *,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> bool:
        """
        Return True if the request is within the rate limit, False otherwise.

        Defaults come from settings (RATE_LIMIT_RPM / 60 seconds window).
        """
        if not user_id:
            return True  # anonymous requests are checked by IP at the infra level

        rpm   = limit          if limit          is not None else settings.RATE_LIMIT_RPM
        win   = window_seconds if window_seconds is not None else 60

        if self._use_redis:
            try:
                script = await self._load_script()
                key    = f"rl:user:{user_id}"
                now_ms = int(time.time() * 1000)
                result = await script(
                    keys=[key],
                    args=[now_ms, win * 1000, rpm, str(uuid.uuid4())],
                )
                return bool(result)
            except Exception as exc:
                logger.error("RateLimiter Redis error — falling back to in-memory: %s", exc)
                # Graceful degradation: fall through to in-memory
                self._use_redis = False

        return _in_memory.is_allowed(user_id, win, rpm)
