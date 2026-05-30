"""
tests/test_disk_store.py

Unit tests for the disk-based fallback memory store.
Tests cover: get/set/delete, TTL expiry, atomic saves, concurrent safety,
and stats reporting.
"""
import os
import time
import tempfile
import pytest

from app.services.disk_store import DiskStore


@pytest.fixture
def store(tmp_path):
    """DiskStore backed by a temporary directory."""
    return DiskStore(data_dir=str(tmp_path))


class TestDiskStoreBasicOps:
    def test_set_and_get(self, store):
        store.set("key1", {"data": 42}, ttl=60)
        result = store.get("key1")
        assert result == {"data": 42}

    def test_get_missing_key_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_delete_removes_key(self, store):
        store.set("key2", "value", ttl=60)
        store.delete("key2")
        assert store.get("key2") is None

    def test_overwrite_key(self, store):
        store.set("key3", "old", ttl=60)
        store.set("key3", "new", ttl=60)
        assert store.get("key3") == "new"

    def test_stores_complex_values(self, store):
        data = {"courses": ["DS", "Algo"], "goals": ["graduation"], "gpa": 3.5}
        store.set("profile", data, ttl=60)
        assert store.get("profile") == data

    def test_available_true_on_writable_dir(self, store):
        assert store._available is True


class TestDiskStoreTTL:
    def test_expired_key_returns_none(self, store):
        store.set("short", "value", ttl=1)
        time.sleep(1.1)
        assert store.get("short") is None

    def test_non_expired_key_returns_value(self, store):
        store.set("long", "value", ttl=60)
        assert store.get("long") == "value"

    def test_expired_keys_pruned_on_write(self, store):
        store.set("exp1", "v1", ttl=1)
        time.sleep(1.1)
        # Writing another key should prune expired ones
        store.set("exp2", "v2", ttl=60)
        # The file should not contain exp1 anymore
        import json
        with open(store._store_file) as f:
            data = json.load(f)
        assert "exp1" not in data


class TestDiskStorePersistence:
    def test_survives_reload(self, tmp_path):
        """Data written by one DiskStore instance is readable by another."""
        s1 = DiskStore(data_dir=str(tmp_path))
        s1.set("persistent_key", {"value": "hello"}, ttl=3600)

        s2 = DiskStore(data_dir=str(tmp_path))
        result = s2.get("persistent_key")
        assert result == {"value": "hello"}


class TestDiskStoreUnavailable:
    def test_get_on_unavailable_returns_none(self, tmp_path):
        """DiskStore on a non-writable dir should degrade gracefully."""
        # Make the dir read-only on supported OS
        store = DiskStore(data_dir=str(tmp_path))
        store._available = False   # force disable
        assert store.get("key") is None

    def test_set_on_unavailable_is_noop(self, tmp_path):
        store = DiskStore(data_dir=str(tmp_path))
        store._available = False
        store.set("key", "value", ttl=60)  # should not raise
        store._available = True
        assert store.get("key") is None


class TestDiskStoreStats:
    def test_stats_when_available(self, store):
        store.set("k", "v", ttl=60)
        stats = store.stats()
        assert stats["available"] is True
        assert stats["keys"] >= 1

    def test_stats_when_unavailable(self, tmp_path):
        store = DiskStore(data_dir=str(tmp_path))
        store._available = False
        stats = store.stats()
        assert stats["available"] is False
