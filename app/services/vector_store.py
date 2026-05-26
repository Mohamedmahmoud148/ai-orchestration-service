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

import os
from typing import Any, Dict, List, Optional

from app.core.logging import logger

_COLLECTION_NAME = "university_materials"
_CHROMA_DATA_DIR = os.environ.get("CHROMA_DATA_DIR") or "/tmp/chroma_data"


class VectorStore:
    """
    ChromaDB-backed persistent vector store.

    Thread-safe for async callers because all Chroma calls are synchronous
    (Chroma's Python client is sync-only) but are cheap enough for I/O
    workloads.  If needed, wrap with asyncio.to_thread() in the future.
    """

    def __init__(self) -> None:
        self._collection: Optional[Any] = None
        self._available = False
        self._init_chroma()

    # ──────────────────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore

            os.makedirs(_CHROMA_DATA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=_CHROMA_DATA_DIR)
            self._collection = client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info(
                "VectorStore: ChromaDB initialised (path=%s, collection=%s).",
                _CHROMA_DATA_DIR, _COLLECTION_NAME,
            )
        except ImportError:
            logger.warning(
                "VectorStore: 'chromadb' package not installed — "
                "RAG indexing is unavailable. Install with: pip install chromadb"
            )
        except Exception as exc:
            logger.warning(
                "VectorStore: ChromaDB init failed — RAG indexing is unavailable. "
                "Error: %s", exc,
            )

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
        if not self._available or self._collection is None:
            logger.warning("VectorStore.upsert_chunks: store unavailable — skipping.")
            return
        if not chunks:
            return

        ids        = [c["chunk_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents  = [c["content"] for c in chunks]
        metadatas  = [c.get("metadata", {}) for c in chunks]

        # Chroma requires all metadata values to be str/int/float/bool
        metadatas = [_sanitise_metadata(m) for m in metadatas]

        try:
            self._collection.upsert(
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
    ) -> List[Dict[str, Any]]:
        """
        Search for the top-k chunks closest to query_embedding.

        Optional filters:
          filter_material_id  — restrict to a single material
          filter_offering_id  — restrict to an offering (subject)

        Returns list of:
          {chunk_id, content, score, metadata}
        """
        if not self._available or self._collection is None:
            logger.warning("VectorStore.search: store unavailable — returning empty.")
            return []

        # Build where clause
        where: Optional[Dict[str, Any]] = _build_where(
            filter_material_id, filter_offering_id
        )

        try:
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, max(1, self._collection.count())),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            result = self._collection.query(**kwargs)

            hits = []
            ids        = result.get("ids", [[]])[0]
            documents  = result.get("documents", [[]])[0]
            metadatas  = result.get("metadatas", [[]])[0]
            distances  = result.get("distances", [[]])[0]

            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                # ChromaDB cosine distance → similarity score [0, 1]
                score = max(0.0, 1.0 - dist)
                hits.append(
                    {
                        "chunk_id": chunk_id,
                        "content":  doc,
                        "score":    round(score, 4),
                        "metadata": meta or {},
                    }
                )
            return hits

        except Exception as exc:
            logger.error("VectorStore.search error: %s", exc)
            return []

    async def delete_material(self, material_id: str) -> None:
        """Delete all chunks belonging to material_id."""
        if not self._available or self._collection is None:
            logger.warning("VectorStore.delete_material: store unavailable — skipping.")
            return
        try:
            self._collection.delete(
                where={"materialId": material_id}
            )
            logger.info("VectorStore: deleted chunks for material_id=%s.", material_id)
        except Exception as exc:
            logger.error("VectorStore.delete_material error: %s", exc)

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic stats about the collection."""
        if not self._available or self._collection is None:
            return {
                "available": False,
                "collection": _COLLECTION_NAME,
                "total_chunks": 0,
                "message": "ChromaDB is unavailable.",
            }
        try:
            count = self._collection.count()
            return {
                "available": True,
                "collection": _COLLECTION_NAME,
                "total_chunks": count,
                "chroma_data_dir": _CHROMA_DATA_DIR,
            }
        except Exception as exc:
            logger.error("VectorStore.get_collection_stats error: %s", exc)
            return {"available": False, "error": str(exc)}


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
