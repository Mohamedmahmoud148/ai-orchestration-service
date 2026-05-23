"""
app/api/routes/rag.py

RAG (Retrieval-Augmented Generation) REST endpoints.

POST  /api/rag/index              — chunk, embed, and store a material
POST  /api/rag/search             — semantic search over indexed material
DELETE /api/rag/material/{id}     — remove all chunks for a material
GET   /api/rag/stats              — collection diagnostics
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.services.chunker import chunk_text
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# ── Request / Response schemas ────────────────────────────────────────────────

class IndexRequest(BaseModel):
    material_id: str = Field(..., description="Unique identifier for the material")
    content: str = Field(..., description="Full text content of the material")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Material metadata (materialTitle, offeringId, subjectName, …)",
    )


class IndexResponse(BaseModel):
    material_id: str
    chunk_count: int
    status: str = "indexed"


class SearchRequest(BaseModel):
    query: str = Field(..., description="User's natural-language query")
    material_id: Optional[str] = Field(None, description="Restrict search to this material")
    offering_id: Optional[str] = Field(None, description="Restrict search to this offering")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of chunks to return")


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    chunks: List[ChunkResult]
    context: str  # pre-formatted context string for LLM injection


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/index", response_model=IndexResponse, status_code=200)
async def index_material(req: IndexRequest) -> IndexResponse:
    """
    Chunk a material, embed each chunk, and upsert into ChromaDB.

    Steps:
      1. chunk_text() — split into overlapping windows
      2. embed_batch() — one API call for all chunks
      3. vector_store.upsert_chunks() — persist with metadata
    """
    logger.info(
        "RAG /index: material_id=%s content_len=%d",
        req.material_id, len(req.content),
    )

    if not vector_store._available:
        raise HTTPException(
            status_code=503,
            detail="Indexing unavailable: ChromaDB is not initialised. "
                   "Install chromadb and restart the service.",
        )

    # 1. Chunk
    raw_chunks = chunk_text(req.content, chunk_size=500, overlap=100)
    if not raw_chunks:
        raise HTTPException(status_code=422, detail="Content produced no chunks after splitting.")

    # 2. Embed (batch for efficiency)
    texts = [c["content"] for c in raw_chunks]
    try:
        embeddings = await embedding_service.embed_batch(texts)
    except Exception as exc:
        logger.error("RAG /index: embedding failed — %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    # 3. Build chunk dicts for the vector store
    chunks_to_store = []
    for raw, embedding in zip(raw_chunks, embeddings):
        chunk_id = f"{req.material_id}__chunk_{raw['chunk_index']}"
        chunk_meta: Dict[str, Any] = {
            "materialId":    req.material_id,
            "chunkIndex":    raw["chunk_index"],
            "materialTitle": req.metadata.get("materialTitle", ""),
            "offeringId":    req.metadata.get("offeringId", ""),
            "subjectName":   req.metadata.get("subjectName", ""),
        }
        # Merge any extra metadata the caller provided
        for k, v in req.metadata.items():
            if k not in chunk_meta:
                chunk_meta[k] = v

        chunks_to_store.append(
            {
                "chunk_id":  chunk_id,
                "content":   raw["content"],
                "embedding": embedding,
                "metadata":  chunk_meta,
            }
        )

    await vector_store.upsert_chunks(req.material_id, chunks_to_store)

    logger.info(
        "RAG /index: indexed %d chunks for material_id=%s.",
        len(chunks_to_store), req.material_id,
    )
    return IndexResponse(
        material_id=req.material_id,
        chunk_count=len(chunks_to_store),
        status="indexed",
    )


@router.post("/search", response_model=SearchResponse, status_code=200)
async def search_material(req: SearchRequest) -> SearchResponse:
    """
    Embed the query and perform a semantic search over indexed materials.

    Returns the top-k chunks plus a pre-formatted *context* string ready
    to inject into an LLM prompt.
    """
    logger.info(
        "RAG /search: query=%r material_id=%s offering_id=%s top_k=%d",
        req.query[:80], req.material_id, req.offering_id, req.top_k,
    )

    try:
        query_embedding = await embedding_service.embed_text(req.query)
    except Exception as exc:
        logger.error("RAG /search: embedding failed — %s", exc)
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    hits = await vector_store.search(
        query_embedding=query_embedding,
        filter_material_id=req.material_id,
        filter_offering_id=req.offering_id,
        top_k=req.top_k,
    )

    chunk_results = [
        ChunkResult(
            chunk_id=h["chunk_id"],
            content=h["content"],
            score=h["score"],
            metadata=h["metadata"],
        )
        for h in hits
    ]

    # Build a readable context block for downstream LLM usage
    context = _build_context(chunk_results)

    return SearchResponse(chunks=chunk_results, context=context)


@router.delete("/material/{material_id}", status_code=200)
async def delete_material(material_id: str) -> Dict[str, str]:
    """Delete all indexed chunks for the given material."""
    logger.info("RAG /delete: material_id=%s", material_id)
    await vector_store.delete_material(material_id)
    return {"status": "deleted", "material_id": material_id}


@router.get("/stats", status_code=200)
async def get_stats() -> Dict[str, Any]:
    """Return ChromaDB collection statistics."""
    return await vector_store.get_collection_stats()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(chunks: List[ChunkResult], max_tokens: int = 3000) -> str:
    """
    Assemble a readable context block from search results.

    Chunks are ordered by score (descending) and truncated to *max_tokens*
    approximate tokens (whitespace-split words).
    """
    if not chunks:
        return ""

    ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
    parts: List[str] = []
    total_tokens = 0

    for i, chunk in enumerate(ordered, 1):
        title = chunk.metadata.get("materialTitle", "Course Material")
        excerpt = f"[{i}] {title}:\n{chunk.content}"
        tok_count = len(excerpt.split())
        if total_tokens + tok_count > max_tokens:
            break
        parts.append(excerpt)
        total_tokens += tok_count

    return "\n\n".join(parts)
