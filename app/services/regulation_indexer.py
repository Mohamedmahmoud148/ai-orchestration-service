"""
app/services/regulation_indexer.py

Regulation Indexer — pulls academic regulation PDFs from the .NET backend,
extracts text, chunks it, embeds chunks, and persists them in ChromaDB.

This is the foundation for the AcademicAdvisor v2 feature: instead of
re-downloading + re-parsing the regulation PDF on every user question
(slow + 10K char cap), we index once and let semantic search retrieve
the relevant passages instantly.

Material-id convention:
    regulation::{regulationId}
This namespace prefix lets RAG searches filter regulation chunks separately
from course material chunks.

Public entry-points:
    await index_all_active_regulations(auth_header=None) -> dict
    await reindex_regulation(regulation_id, file_url, title, auth_header=None) -> dict
    await is_regulation_indexed(regulation_id) -> bool
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import httpx

from app.core.logging import logger
from app.services.backend_client import tool_execution_client
from app.services.chunker import chunk_text
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store


REGULATION_MATERIAL_PREFIX = "regulation::"
_REG_CHUNK_SIZE = 500
_REG_CHUNK_OVERLAP = 100


def _material_id_for(regulation_id: str) -> str:
    return f"{REGULATION_MATERIAL_PREFIX}{regulation_id}"


async def _download_pdf_text(file_url: str, auth_header: Optional[str] = None) -> str:
    """Download a PDF (or text file) and return its extracted text."""
    if not file_url or not file_url.startswith("http"):
        return ""

    headers: Dict[str, str] = {}
    if auth_header:
        headers["Authorization"] = auth_header

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(file_url, headers=headers)
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:
        logger.warning("regulation_indexer: download failed for %s — %s", file_url[:120], exc)
        return ""

    url_low = file_url.lower()
    if "pdf" in url_low or url_low.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("regulation_indexer: pypdf parse failed - %s", exc)

        try:
            from pdfminer.high_level import extract_text
            return (extract_text(io.BytesIO(data)) or "").strip()
        except Exception:
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(data))
                return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
            except Exception as exc:
                logger.warning("regulation_indexer: PDF parse failed — %s", exc)
                return ""

    if url_low.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as exc:
            logger.warning("regulation_indexer: DOCX parse failed — %s", exc)
            return ""

    return data.decode("utf-8", errors="ignore").strip()


async def _index_one(
    regulation_id: str,
    title: str,
    text: str,
) -> int:
    """Chunk + embed + upsert. Returns chunk count (0 on failure / empty text)."""
    if not text or not text.strip():
        logger.warning("regulation_indexer: empty text for regulation '%s' — skipping", title)
        return 0

    if not vector_store._available:
        logger.warning("regulation_indexer: vector store unavailable — skipping '%s'", title)
        return 0

    material_id = _material_id_for(regulation_id)

    raw_chunks = chunk_text(text, chunk_size=_REG_CHUNK_SIZE, overlap=_REG_CHUNK_OVERLAP)
    if not raw_chunks:
        return 0

    texts = [c["content"] for c in raw_chunks]
    try:
        embeddings = await embedding_service.embed_batch(texts)
    except Exception as exc:
        logger.error("regulation_indexer: embedding failed for '%s' — %s", title, exc)
        return 0

    chunks_to_store: List[Dict[str, Any]] = []
    for raw, embedding in zip(raw_chunks, embeddings):
        chunks_to_store.append({
            "chunk_id":  f"{material_id}__chunk_{raw['chunk_index']}",
            "content":   raw["content"],
            "embedding": embedding,
            "metadata": {
                "materialId":    material_id,
                "chunkIndex":    raw["chunk_index"],
                "materialTitle": title,
                "type":          "regulation",
                "regulationId":  regulation_id,
            },
        })

    # Replace any previous chunks for this regulation before upserting fresh ones
    try:
        await vector_store.delete_material(material_id)
    except Exception:
        pass

    await vector_store.upsert_chunks(material_id, chunks_to_store)
    logger.info(
        "regulation_indexer: indexed %d chunks for regulation '%s' (id=%s)",
        len(chunks_to_store), title, regulation_id,
    )
    return len(chunks_to_store)


async def reindex_regulation(
    regulation_id: str,
    file_url: str,
    title: str = "Regulation",
    auth_header: Optional[str] = None,
    inline_content: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-index a single regulation (re-downloads the PDF and replaces chunks)."""
    text = ""
    if file_url:
        text = await _download_pdf_text(file_url, auth_header)
    if not text and inline_content:
        text = inline_content

    chunk_count = await _index_one(regulation_id, title, text)
    return {
        "regulation_id": regulation_id,
        "title":         title,
        "chunk_count":   chunk_count,
        "status":        "indexed" if chunk_count else "skipped",
    }


_REINDEX_LOCK_KEY    = "regulation_indexer:reindex_lock"
_REINDEX_LOCK_TTL_S  = 300   # 5 min — generous enough for a 50-PDF reindex


async def _try_acquire_lock() -> bool:
    """
    Best-effort cross-process lock via Redis SET NX EX. Returns True if we
    own the lock. If Redis is unavailable we return True (locking is purely
    an optimisation; the underlying upsert is idempotent).
    """
    try:
        from app.services.memory_store import get_memory_store
        store = get_memory_store()
        client = getattr(store, "redis_client", None)
        if client is None:
            return True  # No Redis → fall through, accept potential overlap
        acquired = await client.set(
            _REINDEX_LOCK_KEY, "1", nx=True, ex=_REINDEX_LOCK_TTL_S,
        )
        if acquired:
            logger.info("regulation_indexer: acquired reindex lock (ttl=%ds)", _REINDEX_LOCK_TTL_S)
        return bool(acquired)
    except Exception as exc:
        logger.warning("regulation_indexer: lock check failed (proceeding) — %s", exc)
        return True


async def _release_lock() -> None:
    """Best-effort release. The TTL will expire it anyway if this fails."""
    try:
        from app.services.memory_store import get_memory_store
        store = get_memory_store()
        client = getattr(store, "redis_client", None)
        if client is not None:
            await client.delete(_REINDEX_LOCK_KEY)
    except Exception:
        pass


async def index_all_active_regulations(
    auth_header: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch /api/Regulations from the .NET backend and index every regulation
    that has a fileUrl (or inline content). Safe to call repeatedly — it
    upserts (replaces existing chunks of the same material_id).

    Uses a Redis lock (TTL 5 min) so two concurrent workers — e.g. during a
    rolling restart — don't both run the same expensive reindex.
    """
    if not await _try_acquire_lock():
        logger.info(
            "regulation_indexer: another worker is reindexing — skipping this run"
        )
        return {"status": "skipped", "reason": "another_worker_holds_lock"}

    logger.info("regulation_indexer: starting full reindex")

    # try/finally guarantees the lock is released even if an unexpected
    # exception escapes the loop. The TTL would expire it anyway in 5 min.
    try:
        try:
            regs_raw = await tool_execution_client.fetch(
                route="/api/Regulations",
                auth_header=auth_header,
                params={"page": 1, "size": 50},
            )
        except Exception as exc:
            logger.error("regulation_indexer: backend fetch failed — %s", exc)
            return {"status": "failed", "reason": "backend_unavailable", "regulations": []}

        # Unwrap envelope shapes
        if isinstance(regs_raw, dict):
            regs_list: List[Any] = (
                regs_raw.get("data")
                or regs_raw.get("items")
                or (regs_raw.get("value") if isinstance(regs_raw.get("value"), list) else None)
                or []
            )
        elif isinstance(regs_raw, list):
            regs_list = regs_raw
        else:
            regs_list = []

        if not regs_list:
            logger.warning("regulation_indexer: no regulations returned from backend")
            return {"status": "empty", "regulations": []}

        indexed: List[Dict[str, Any]] = []
        for reg in regs_list:
            if not isinstance(reg, dict):
                continue
            reg_id = str(reg.get("id") or reg.get("regulationId") or "").strip()
            if not reg_id:
                continue
            title   = str(reg.get("title") or "Regulation")
            file_url = reg.get("fileUrl") or reg.get("filePath") or ""
            content = reg.get("content") or ""
            is_active = reg.get("isActive", True)
            if is_active is False:
                continue

            outcome = await reindex_regulation(
                regulation_id=reg_id,
                file_url=file_url,
                title=title,
                auth_header=auth_header,
                inline_content=content if isinstance(content, str) else None,
            )
            indexed.append(outcome)

        total_chunks = sum(o.get("chunk_count", 0) for o in indexed)
        logger.info(
            "regulation_indexer: full reindex done — %d regulations, %d total chunks",
            len(indexed), total_chunks,
        )
        return {
            "status":       "ok" if total_chunks > 0 else "empty",
            "regulations":  indexed,
            "total_chunks": total_chunks,
        }
    finally:
        await _release_lock()


async def is_any_regulation_indexed() -> bool:
    """Cheap check used at startup to decide whether to auto-index."""
    if not vector_store._available:
        return False
    result = await vector_store.collection_get(
        where={"type": "regulation"},
        limit=1,
    )
    ids = result.get("ids", []) if isinstance(result, dict) else []
    return bool(ids)


async def search_regulation(
    query: str,
    top_k: int = 6,
    regulation_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search restricted to regulation chunks.

    If regulation_id is supplied, restricts to that single regulation.
    Otherwise searches across ALL indexed regulations.
    """
    if not vector_store._available:
        return []

    try:
        query_embedding = await embedding_service.embed_text(query)
    except Exception as exc:
        logger.error("regulation_indexer.search_regulation: embedding failed — %s", exc)
        return []

    where: Dict[str, Any]
    if regulation_id:
        where = {"materialId": _material_id_for(regulation_id)}
    else:
        where = {"type": "regulation"}

    result = await vector_store.collection_query(
        query_embedding=query_embedding,
        where=where,
        top_k=top_k,
    )

    ids        = (result.get("ids") or [[]])[0]
    documents  = (result.get("documents") or [[]])[0]
    metadatas  = (result.get("metadatas") or [[]])[0]
    distances  = (result.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        score = max(0.0, 1.0 - dist)
        hits.append({
            "chunk_id": cid,
            "content":  doc,
            "score":    round(score, 4),
            "metadata": meta or {},
        })
    return hits
