"""
app/services/disk_store.py

Disk-based key-value store — used as a fallback when Redis is unavailable.

Provides the same interface as the Redis _get/_set/_delete primitives in
MemoryStore so it can be swapped in transparently.

Storage:
  A single JSON file per "namespace" (e.g. "memory", "preferences") inside
  the configured data directory.  All keys within a namespace share one file,
  which is read/written atomically via a temp-file rename.

TTL:
  Expiry timestamps are stored alongside values. Expired entries are pruned
  lazily on each read/write.

Concurrency:
  Safe for single-worker deployments (one uvicorn process). For multi-worker
  deployments, Redis is the correct solution — this is a development/no-Redis
  convenience only.

Data directory:
  Defaults to ./data/memory or $MEMORY_DATA_DIR env var.
  Falls back gracefully if the directory cannot be created.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional

from app.core.logging import logger

_DATA_DIR = os.environ.get("MEMORY_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "memory",
)

_LOCK = threading.Lock()   # guards file I/O across async tasks in the same process


class DiskStore:
    """
    Disk-backed key-value store with TTL.

    All operations are synchronous internally but wrapped with asyncio.to_thread
    at call sites for async compatibility.
    """

    def __init__(self, data_dir: str = _DATA_DIR) -> None:
        self._dir = data_dir
        self._available = False
        self._store_file = os.path.join(data_dir, "kv_store.json")
        self._init()

    def _init(self) -> None:
        try:
            os.makedirs(self._dir, exist_ok=True)
            # Verify writable
            test = os.path.join(self._dir, ".write_test")
            with open(test, "w") as f:
                f.write("ok")
            os.remove(test)
            self._available = True
            logger.info("DiskStore: initialised at %s", self._dir)
        except Exception as exc:
            logger.warning(
                "DiskStore: could not initialise at %s — disk fallback unavailable: %s",
                self._dir, exc,
            )

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        """Load the entire store from disk. Returns {} on error."""
        try:
            if not os.path.exists(self._store_file):
                return {}
            with open(self._store_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        """Atomically save the store to disk via temp-file rename."""
        tmp = self._store_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            os.replace(tmp, self._store_file)
        except Exception as exc:
            logger.warning("DiskStore._save failed: %s", exc)
            try:
                os.remove(tmp)
            except Exception:
                pass

    def _prune_expired(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove entries past their expiry. Returns cleaned dict."""
        now = time.time()
        return {
            k: v for k, v in data.items()
            if isinstance(v, dict) and v.get("exp", float("inf")) > now
        }

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            return None
        with _LOCK:
            data = self._prune_expired(self._load())
            entry = data.get(key)
            if entry is None:
                return None
            return entry.get("val")

    def set(self, key: str, value: Any, ttl: int) -> None:
        if not self._available:
            return
        with _LOCK:
            data = self._prune_expired(self._load())
            data[key] = {"val": value, "exp": time.time() + ttl}
            self._save(data)

    def delete(self, key: str) -> None:
        if not self._available:
            return
        with _LOCK:
            data = self._load()
            data.pop(key, None)
            self._save(data)

    def stats(self) -> Dict[str, Any]:
        """Return store stats for health checks."""
        if not self._available:
            return {"available": False, "store_file": self._store_file}
        try:
            with _LOCK:
                data = self._prune_expired(self._load())
            return {
                "available": True,
                "store_file": self._store_file,
                "keys": len(data),
                "data_dir": self._dir,
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}
