"""
app/services/circuit_breaker.py

Lightweight, dependency-free async circuit breaker.

States:
    CLOSED    — calls pass through, failures are counted
    OPEN      — every call rejected immediately with CircuitOpenError
    HALF_OPEN — single trial call allowed; success → CLOSED, failure → OPEN

Why in-house (no `pybreaker`):
- Zero new dependencies for a project under defense.
- The semantics we need are small; full breaker libraries bring threading
  primitives we don't need in an asyncio world.

Usage:
    breaker = CircuitBreaker(
        name="backend",
        fail_threshold=5,
        reset_seconds=30,
    )
    try:
        result = await breaker.call(lambda: client.get(url))
    except CircuitOpenError:
        # short-circuited — fall back fast instead of waiting on a dead service
        ...
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from app.core.logging import logger


class CircuitOpenError(Exception):
    """Raised when the circuit is open and a call is short-circuited."""


_CLOSED   = "closed"
_OPEN     = "open"
_HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Async-safe circuit breaker. All state mutations are guarded by a single
    asyncio lock so concurrent calls see consistent state transitions.
    """

    def __init__(
        self,
        name: str,
        fail_threshold: int = 5,
        reset_seconds: float = 30.0,
    ) -> None:
        self._name           = name
        self._fail_threshold = max(1, int(fail_threshold))
        self._reset_seconds  = max(1.0, float(reset_seconds))

        self._state           = _CLOSED
        self._consecutive_fail = 0
        self._opened_at        = 0.0
        self._lock             = asyncio.Lock()

    # ── Public state helpers ─────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == _OPEN

    async def call(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        """
        Run `fn()` under breaker protection.

        Raises CircuitOpenError immediately if circuit is open AND reset window
        has not elapsed yet.
        """
        # Fast pre-check (no lock) — common case (CLOSED) hits this every call
        if self._state == _OPEN:
            if (time.monotonic() - self._opened_at) < self._reset_seconds:
                raise CircuitOpenError(
                    f"Circuit '{self._name}' is OPEN — calls rejected for "
                    f"{self._reset_seconds - (time.monotonic() - self._opened_at):.1f}s more"
                )
            # Reset window elapsed — try half-open
            async with self._lock:
                if self._state == _OPEN:
                    self._state = _HALF_OPEN
                    logger.info(
                        "CircuitBreaker[%s]: OPEN → HALF_OPEN (trial call)",
                        self._name,
                    )

        try:
            result = await fn()
        except Exception:
            await self._record_failure()
            raise
        else:
            await self._record_success()
            return result

    # ── Internal state transitions ───────────────────────────────────────

    async def _record_success(self) -> None:
        async with self._lock:
            if self._state == _HALF_OPEN:
                logger.info(
                    "CircuitBreaker[%s]: HALF_OPEN → CLOSED (trial succeeded)",
                    self._name,
                )
            if self._consecutive_fail > 0:
                logger.debug(
                    "CircuitBreaker[%s]: failure streak reset (was %d)",
                    self._name, self._consecutive_fail,
                )
            self._state = _CLOSED
            self._consecutive_fail = 0

    async def _record_failure(self) -> None:
        async with self._lock:
            self._consecutive_fail += 1
            if self._state == _HALF_OPEN:
                self._state    = _OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "CircuitBreaker[%s]: HALF_OPEN → OPEN (trial failed)",
                    self._name,
                )
                return

            if self._state == _CLOSED and self._consecutive_fail >= self._fail_threshold:
                self._state    = _OPEN
                self._opened_at = time.monotonic()
                logger.error(
                    "CircuitBreaker[%s]: CLOSED → OPEN after %d consecutive failures. "
                    "Rejecting calls for %.0fs.",
                    self._name, self._consecutive_fail, self._reset_seconds,
                )

    def status(self) -> dict:
        """Snapshot for /health endpoints."""
        return {
            "name":               self._name,
            "state":              self._state,
            "consecutive_fail":   self._consecutive_fail,
            "fail_threshold":     self._fail_threshold,
            "reset_seconds":      self._reset_seconds,
            "opened_at_monotonic": self._opened_at if self._state != _CLOSED else None,
        }
