"""
app/services/vector_store.py

In-memory / persistent vector store for the RAG pipeline.

Primary backend: ChromaDB (chromadb package) with persistent storage in
./chroma_data so embeddings survive restarts.

Graceful degradation: if chromadb is not installed or fails to initialise,
the store logs a warning and disables itself — all public methods return
empty/no-op responses so the rest of the pipeline keeps running.

Collection: "university_materials"

Public API:
  upsert_chunks(material_id, chunks)   → None
  search(query_embedding, ...)         → list[dict]
  delete_material(material_id)         → None
  get_collection_stats()               → dict
"""
from __future__ import annotations

import asyncio
import math
import os
import tempfile
from typing import Any, Dict, List, Optional

from app.core.logging import logger

_COLLECTION_NAME = "university_materials"

# Default to a named directory next to the service root so data survives
# container restarts on Railway (Railway mounts /app persistently when a
# volume is attached).  Override via CHROMA_DATA_DIR env var.
_DEFAULT_CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "chroma_data",
)
_ENV_CHROMA_DATA_DIR = os.environ.get("CHROMA_DATA_DIR")
_FALLBACK_CHROMA_DIR = os.path.join(tempfile.gettempdir(), "university_chroma_data")
_CHROMA_DATA_DIR = _ENV_CHROMA_DATA_DIR or _DEFAULT_CHROMA_DIR

# Minimum cosine similarity to include a chunk in search results.
# Range: 0.0–1.0. Lower = more permissive.
# Default 0.0 = no filtering (safe for keyword_fallback and first-time setups).
# Set RAG_MIN_SCORE=0.25 in production when using real semantic embeddings
# (OpenAI or sentence-transformers) to filter out irrelevant chunks.
_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.0"))


class VectorStore:
    """
    ChromaDB-backed persistent vector store.

    Thread-safe for async callers because all Chroma calls are synchronous
    (Chroma's Python client is sync-only) but are cheap enough for I/O
    workloads.  If needed, wrap with asyncio.to_thread() in the future.
    """

    def __init__(self) -> None:
        self._collection: Optional[Any] = None
        self._backend = "unavailable"
        self._available = False
        self._init_error: Optional[str] = None
        self._memory_chunks: Dict[str, Dict[str, Any]] = {}
        self._init_chroma()

    # ──────────────────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore

            candidates = [_CHROMA_DATA_DIR]
            if not _ENV_CHROMA_DATA_DIR and _FALLBACK_CHROMA_DIR not in candidates:
                candidates.append(_FALLBACK_CHROMA_DIR)

            errors: List[str] = []
            for data_dir in candidates:
                try:
                    self._init_chroma_at_path(chromadb, data_dir)
                    return
                except Exception as exc:
                    errors.append(f"{data_dir}: {exc}")
                    logger.warning(
                        "VectorStore: ChromaDB init failed at %s - %s",
                        data_dir,
                        exc,
                    )

            raise RuntimeError("; ".join(errors))

        except ImportError as exc:
            self._enable_memory_fallback(
                "'chromadb' package not installed. Install with: pip install chromadb",
                exc,
            )
        except Exception as exc:
            self._enable_memory_fallback("ChromaDB init failed", exc)

    def _init_chroma_at_path(self, chromadb: Any, data_dir: str) -> None:
        global _CHROMA_DATA_DIR

        os.makedirs(data_dir, exist_ok=True)
        test_file = os.path.join(data_dir, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)

        client = chromadb.PersistentClient(path=data_dir)
        self._collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        _CHROMA_DATA_DIR = data_dir
        self._available = True
        self._backend = "chroma"
        self._init_error = None
        logger.info(
            "VectorStore: ChromaDB initialised - path=%s collection=%s "
            "existing_chunks=%d writable=True",
            _CHROMA_DATA_DIR, _COLLECTION_NAME, count,
        )

    def _enable_memory_fallback(self, reason: str, exc: Exception) -> None:
        self._collection = None
        self._backend = "memory"
        self._available = True
        self._init_error = f"{reason}: {exc}"
        logger.warning(
            "VectorStore: %s. Falling back to in-memory vector store; "
            "RAG works until process restart. Error: %s",
            reason,
            exc,
        )

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _matches_where(meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(VectorStore._matches_where(meta, clause) for clause in where["$and"])
        for key, expected in where.items():
            actual = meta.get(key)
            if isinstance(expected, dict) and "$eq" in expected:
                expected = expected["$eq"]
            if actual != expected:
                return False
        return True

    def _memory_query(
        self,
        query_embedding: List[float],
        where: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        rows = []
        for chunk_id, chunk in self._memory_chunks.items():
            meta = chunk.get("metadata", {})
            if not self._matches_where(meta, where):
                continue
            similarity = self._cosine_similarity(query_embedding, chunk.get("embedding", []))
            rows.append((chunk_id, chunk, 1.0 - similarity))

        rows.sort(key=lambda row: row[2])
        rows = rows[: max(0, top_k)]
        return {
            "ids": [[row[0] for row in rows]],
            "documents": [[row[1].get("content", "") for row in rows]],
            "metadatas": [[row[1].get("metadata", {}) for row in rows]],
            "distances": [[row[2] for row in rows]],
        }

    def _memory_get(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        rows = []
        for chunk_id, chunk in self._memory_chunks.items():
            if self._matches_where(chunk.get("metadata", {}), where):
                rows.append((chunk_id, chunk))
            if limit is not None and len(rows) >= limit:
                break
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1].get("content", "") for row in rows],
            "metadatas": [row[1].get("metadata", {}) for row in rows],
        }

    @staticmethod
    def _hits_from_result(result: Dict[str, Any], min_score: float) -> List[Dict[str, Any]]:
        hits = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - dist)
            if score < min_score:
                continue
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "content": doc,
                    "score": round(score, 4),
                    "metadata": meta or {},
                }
            )
        return hits

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        material_id: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert a list of chunk dicts into the collection.

        Each chunk must contain:
          {
            chunk_id:  str,
            content:   str,
            embedding: list[float],
            metadata:  {materialId, chunkIndex, materialTitle, offeringId, ...}
          }
        """
        if not self._available:
            logger.warning("VectorStore.upsert_chunks: store unavailable — skipping.")
            return
        if not chunks:
            return

        if self._backend == "memory":
            for chunk in chunks:
                self._memory_chunks[chunk["chunk_id"]] = {
                    "content": chunk["content"],
                    "embedding": chunk["embedding"],
                    "metadata": _sanitise_metadata(chunk.get("metadata", {})),
                }
            logger.info(
                "VectorStore(memory): upserted %d chunks for material_id=%s.",
                len(chunks), material_id,
            )
            return

        if self._collection is None:
            logger.warning("VectorStore.upsert_chunks: Chroma collection missing - skipping.")
            return

        ids        = [c["chunk_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents  = [c["content"] for c in chunks]
        metadatas  = [c.get("metadata", {}) for c in chunks]

        # Chroma requires all metadata values to be str/int/float/bool
        metadatas = [_sanitise_metadata(m) for m in metadatas]

        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "VectorStore: upserted %d chunks for material_id=%s.",
                len(chunks), material_id,
            )
        except Exception as exc:
            logger.error("VectorStore.upsert_chunks error: %s", exc)

    async def search(
        self,
        query_embedding: List[float],
        filter_material_id: Optional[str] = None,
        filter_offering_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = _MIN_SCORE,
    ) -> List[Dict[str, Any]]:
        """
        Search for the top-k chunks closest to query_embedding.

        Optional filters:
          filter_material_id  — restrict to a single material
          filter_offering_id  — restrict to an offering (subject)

        Returns list of:
          {chunk_id, content, score, metadata}
        """
        if not self._available:
            logger.warning("VectorStore.search: store unavailable — returning empty.")
            return []

        # Build where clause
        where: Optional[Dict[str, Any]] = _build_where(
            filter_material_id, filter_offering_id
        )

        if self._backend == "memory":
            result = self._memory_query(query_embedding, where=where, top_k=top_k)
            return self._hits_from_result(result, min_score)

        try:
            # count() and query() are both sync — run together in a thread
            # so the event loop never blocks on either.
            def _run_query() -> Dict[str, Any]:
                count = self._collection.count()
                if count == 0:
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                kwargs: Dict[str, Any] = {
                    "query_embeddings": [query_embedding],
                    "n_results": min(top_k, count),
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    kwargs["where"] = where
                return self._collection.query(**kwargs)

            result = await asyncio.to_thread(_run_query)

            hits = []
            ids        = result.get("ids", [[]])[0]
            documents  = result.get("documents", [[]])[0]
            metadatas  = result.get("metadatas", [[]])[0]
            distances  = result.get("distances", [[]])[0]

            filtered_out = 0
            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                # ChromaDB cosine distance → similarity score [0, 1]
                score = max(0.0, 1.0 - dist)
                if score < min_score:
                    filtered_out += 1
                    continue
                hits.append(
                    {
                        "chunk_id": chunk_id,
                        "content":  doc,
                        "score":    round(score, 4),
                        "metadata": meta or {},
                    }
                )

            if filtered_out:
                logger.debug(
                    "VectorStore.search: filtered %d low-score chunks (min_score=%.2f)",
                    filtered_out, min_score,
                )
            return hits

        except Exception as exc:
            logger.error("VectorStore.search error: %s", exc)
            return []

    async def delete_material(self, material_id: str) -> None:
        """Delete all chunks belonging to material_id."""
        if not self._available:
            logger.warning("VectorStore.delete_material: store unavailable — skipping.")
            return
        if self._backend == "memory":
            before = len(self._memory_chunks)
            self._memory_chunks = {
                chunk_id: chunk
                for chunk_id, chunk in self._memory_chunks.items()
                if chunk.get("metadata", {}).get("materialId") != material_id
            }
            logger.info(
                "VectorStore(memory): deleted %d chunks for material_id=%s.",
                before - len(self._memory_chunks),
                material_id,
            )
            return
        if self._collection is None:
            logger.warning("VectorStore.delete_material: Chroma collection missing - skipping.")
            return
        try:
            await asyncio.to_thread(
                self._collection.delete,
                where={"materialId": material_id},
            )
            logger.info("VectorStore: deleted chunks for material_id=%s.", material_id)
        except Exception as exc:
            logger.error("VectorStore.delete_material error: %s", exc)

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic stats about the collection."""
        if not self._available:
            return {
                "available": False,
                "backend": self._backend,
                "collection": _COLLECTION_NAME,
                "total_chunks": 0,
                "data_dir": _CHROMA_DATA_DIR,
                "message": "ChromaDB is unavailable.",
                "init_error": self._init_error,
            }
        if self._backend == "memory":
            return {
                "available": True,
                "backend": "memory",
                "collection": _COLLECTION_NAME,
                "total_chunks": len(self._memory_chunks),
                "data_dir": _CHROMA_DATA_DIR,
                "data_dir_writable": os.access(_CHROMA_DATA_DIR, os.W_OK),
                "min_score_threshold": _MIN_SCORE,
                "persistent": False,
                "message": "Using in-memory vector store because ChromaDB is unavailable.",
                "init_error": self._init_error,
            }
        if self._collection is None:
            return {
                "available": False,
                "backend": self._backend,
                "collection": _COLLECTION_NAME,
                "total_chunks": 0,
                "data_dir": _CHROMA_DATA_DIR,
                "message": "ChromaDB collection is unavailable.",
                "init_error": self._init_error,
            }
        try:
            count = await asyncio.to_thread(self._collection.count)
            writable = os.access(_CHROMA_DATA_DIR, os.W_OK)
            return {
                "available": True,
                "backend": "chroma",
                "collection": _COLLECTION_NAME,
                "total_chunks": count,
                "data_dir": _CHROMA_DATA_DIR,
                "data_dir_writable": writable,
                "min_score_threshold": _MIN_SCORE,
            }
        except Exception as exc:
            logger.error("VectorStore.get_collection_stats error: %s", exc)
            return {"available": False, "error": str(exc)}

    async def probe(self) -> Dict[str, Any]:
        """
        Lightweight ChromaDB probe — verifies the data directory is writable
        and the collection object is accessible.

        Does NOT insert test vectors — inserting would lock the collection to
        a specific embedding dimension and break subsequent indexing if the
        embedding provider changes.
        """
        if not self._available:
            return {"ok": False, "reason": "ChromaDB not initialised"}
        if self._backend == "memory":
            return {
                "ok": True,
                "backend": "memory",
                "total_chunks": len(self._memory_chunks),
                "persistent": False,
                "reason": "ChromaDB unavailable; memory fallback active",
                "init_error": self._init_error,
            }
        if self._collection is None:
            return {"ok": False, "reason": "ChromaDB collection missing"}
        try:
            def _run_probe():
                # Just verify we can call count() and check writability
                count = self._collection.count()
                writable = os.access(_CHROMA_DATA_DIR, os.W_OK)
                return {"count": count, "writable": writable}

            info = await asyncio.to_thread(_run_probe)
            return {
                "ok": info["writable"],
                "data_dir": _CHROMA_DATA_DIR,
                "total_chunks": info["count"],
                "data_dir_writable": info["writable"],
            }
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    # ──────────────────────────────────────────────────────────────────────
    #  Async-safe escape hatches for callers that need raw collection ops
    #  (regulation_indexer uses these for `where`-filter `get()` and
    #  filtered queries). Always wrap sync chroma calls — never call
    #  self._collection.{get,query,count,...} directly from async code.
    # ──────────────────────────────────────────────────────────────────────

    async def collection_get(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async-safe wrapper for `_collection.get(where=..., limit=...)`."""
        if not self._available:
            return {}
        if self._backend == "memory":
            return self._memory_get(where=where, limit=limit)
        if self._collection is None:
            return {}
        try:
            return await asyncio.to_thread(
                self._collection.get,
                where=where or {},
                limit=limit,
            )
        except Exception as exc:
            logger.warning("VectorStore.collection_get: failed — %s", exc)
            return {}

    async def collection_query(
        self,
        query_embedding: List[float],
        where: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Async-safe wrapper for `_collection.query(...)` with optional where filter."""
        if not self._available:
            return {}
        if self._backend == "memory":
            return self._memory_query(query_embedding, where=where, top_k=top_k)
        if self._collection is None:
            return {}
        try:
            def _run() -> Dict[str, Any]:
                kwargs: Dict[str, Any] = {
                    "query_embeddings": [query_embedding],
                    "n_results": min(top_k, max(1, self._collection.count())),
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    kwargs["where"] = where
                return self._collection.query(**kwargs)
            return await asyncio.to_thread(_run)
        except Exception as exc:
            logger.warning("VectorStore.collection_query: failed — %s", exc)
            return {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma only accepts str/int/float/bool metadata values.
    Convert everything else to str.
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


def _build_where(
    material_id: Optional[str],
    offering_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB where-clause dict from the optional filter args."""
    conditions = []
    if material_id:
        conditions.append({"materialId": {"$eq": material_id}})
    if offering_id:
        conditions.append({"offeringId": {"$eq": offering_id}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Singleton ─────────────────────────────────────────────────────────────────
vector_store = VectorStore()
