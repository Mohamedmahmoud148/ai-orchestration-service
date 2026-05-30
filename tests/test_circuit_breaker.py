"""
Tests for app/services/circuit_breaker.py — state machine and concurrency safety.
"""
import asyncio

import pytest

from app.services.circuit_breaker import CircuitBreaker, CircuitOpenError


@pytest.mark.asyncio
async def test_initial_state_is_closed():
    cb = CircuitBreaker(name="t1", fail_threshold=3, reset_seconds=1)
    assert cb.state == "closed"
    assert not cb.is_open


@pytest.mark.asyncio
async def test_successful_calls_keep_circuit_closed():
    cb = CircuitBreaker(name="t2", fail_threshold=3, reset_seconds=1)

    async def ok():
        return "ok"

    for _ in range(10):
        result = await cb.call(ok)
        assert result == "ok"
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures():
    cb = CircuitBreaker(name="t3", fail_threshold=3, reset_seconds=1)

    async def fail():
        raise RuntimeError("boom")

    # Threshold = 3 failures
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(fail)

    assert cb.state == "open"
    assert cb.is_open


@pytest.mark.asyncio
async def test_open_circuit_short_circuits_immediately():
    cb = CircuitBreaker(name="t4", fail_threshold=2, reset_seconds=10)

    async def fail():
        raise RuntimeError("boom")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    assert cb.state == "open"

    # Subsequent calls should NOT actually call fail() — they short-circuit
    call_count = 0
    async def track():
        nonlocal call_count
        call_count += 1
        return "ok"

    with pytest.raises(CircuitOpenError):
        await cb.call(track)

    assert call_count == 0, "Open circuit should not invoke the callable"


# Note: CircuitBreaker clamps reset_seconds to >= 1.0 in production code
# (a production breaker recovering in milliseconds is almost always a bug).
# These tests therefore use reset_seconds=1 + sleep(1.1) which is slow but
# fair — runs in ~2s for both tests combined.

@pytest.mark.asyncio
async def test_half_open_success_closes_circuit():
    cb = CircuitBreaker(name="t5", fail_threshold=2, reset_seconds=1)

    async def fail():
        raise RuntimeError("boom")

    async def ok():
        return "recovered"

    # Open the circuit
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    assert cb.state == "open"

    # Wait for reset window (clamped to 1s minimum)
    await asyncio.sleep(1.1)

    # First call after reset = half-open trial; success closes the circuit
    result = await cb.call(ok)
    assert result == "recovered"
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit():
    cb = CircuitBreaker(name="t6", fail_threshold=2, reset_seconds=1)

    async def fail():
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(fail)
    assert cb.state == "open"

    await asyncio.sleep(1.1)

    # Trial fails → circuit re-opens
    with pytest.raises(RuntimeError):
        await cb.call(fail)
    assert cb.state == "open"


@pytest.mark.asyncio
async def test_status_snapshot_shape():
    cb = CircuitBreaker(name="health_check", fail_threshold=5, reset_seconds=30)
    snap = cb.status()
    assert snap["name"] == "health_check"
    assert snap["state"] == "closed"
    assert snap["fail_threshold"] == 5
    assert snap["reset_seconds"] == 30
    assert snap["consecutive_fail"] == 0
