"""
Tests for the MemoryStore singleton accessor (H8 fix).

Why: prior code created a new MemoryStore() in agent.py and dynamic_api.py,
each opening its own Redis connection pool and emitting duplicate auth logs.
"""
from app.services.memory_store import MemoryStore, get_memory_store


def test_get_memory_store_returns_same_instance():
    """Repeated calls must return the same object — not new instances."""
    s1 = get_memory_store()
    s2 = get_memory_store()
    s3 = get_memory_store()
    assert s1 is s2
    assert s2 is s3


def test_get_memory_store_returns_memory_store_type():
    store = get_memory_store()
    assert isinstance(store, MemoryStore)


def test_singleton_consistent_across_imports():
    """Importing from a different path should still resolve to the same module-level singleton."""
    from app.services import memory_store as mod
    s1 = mod.get_memory_store()
    s2 = get_memory_store()
    assert s1 is s2
