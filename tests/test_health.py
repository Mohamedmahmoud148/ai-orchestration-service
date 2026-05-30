"""
Tests for health + admin endpoints.
"""
import pytest

from app.api.routes.health import (
    admin_reindex_regulations,
    health_check,
    health_detailed,
    refresh_prompts,
    refresh_schema,
)


@pytest.mark.asyncio
async def test_basic_health_check():
    result = await health_check()
    assert result == {"status": "ok", "service": "fastapi-ai-service"}


@pytest.mark.asyncio
async def test_detailed_health_returns_all_subsystems():
    """All subsystem keys must be present even when subsystems are unavailable."""
    snapshot = await health_detailed()
    expected_keys = {
        "status", "service", "environment",
        "backend", "rag", "regulation_indexed", "memory_store",
    }
    assert expected_keys.issubset(snapshot.keys()), (
        f"Missing keys: {expected_keys - set(snapshot.keys())}"
    )


@pytest.mark.asyncio
async def test_detailed_health_reports_breaker_state():
    """Breaker section must include the canonical fields downstream dashboards expect."""
    snapshot = await health_detailed()
    breaker = snapshot.get("backend", {}).get("circuit_breaker", {})
    assert "state" in breaker
    assert breaker["state"] in ("closed", "open", "half_open")
    assert "fail_threshold" in breaker
    assert "reset_seconds" in breaker


@pytest.mark.asyncio
async def test_detailed_health_never_raises_on_subsystem_failure():
    """Even if everything is broken, /health/detailed returns a JSON dict."""
    # Subsystems already mostly unavailable in tests (no Chroma, no Redis).
    # Just confirm we get back a dict, not an exception.
    snapshot = await health_detailed()
    assert isinstance(snapshot, dict)


@pytest.mark.asyncio
async def test_refresh_prompts_clears_cache_and_returns_ok():
    from app.prompts import load_prompt

    # Warm the cache
    load_prompt("role_student")

    result = await refresh_prompts()
    assert result["status"] == "ok"
    assert "cleared" in result["message"].lower()

    # Cache cleared — loading again should still work (re-reads file)
    body = load_prompt("role_student")
    assert len(body) > 0


@pytest.mark.asyncio
async def test_refresh_schema_does_not_raise():
    """
    /admin/refresh-schema must return a structured dict even when the
    backend is unreachable (which it is in unit tests).
    """
    result = await refresh_schema()
    assert "status" in result
    # In unit tests the .NET backend isn't running — this is expected.
    # The endpoint must STILL return a clean response, not raise.
    assert result["status"] in ("ok", "error")


@pytest.mark.asyncio
async def test_admin_reindex_returns_clean_response_even_without_backend():
    """Same robustness contract as refresh_schema."""
    result = await admin_reindex_regulations()
    assert "status" in result
    assert result["status"] in ("ok", "error")
