"""
app/modules/material_qa.py

Material Q&A Module — answers student questions grounded ONLY in indexed
course material retrieved from ChromaDB via the RAG pipeline.

Intent: material_qa

Flow:
  1. Extract the student's query from the agent_input message.
  2. Extract offering_id / material_id from context (academic_context).
  3. Embed the query via EmbeddingService.
  4. Search VectorStore for the top-5 most relevant chunks.
  5. If no chunks found → bilingual "not indexed" message.
  6. Build context string (max 3 000 approximate tokens).
  7. Call LLM with a strict grounding prompt.
  8. Return answer + source references as an AgentOutput.

Module is registered in:
  - executor._MODULE_CLASS_MAP    as "material_qa"
  - tool_registry._registry       as "material_qa" → "material_qa"
  - rbac._STUDENT_ALLOWED etc.    as "material_qa"
  - planner.VALID_INTENTS         as "material_qa"
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

_NO_MATERIAL_MSG = (
    "لم يتم فهرسة مواد لهذا الكورس بعد.\n"
    "No indexed course materials were found for this subject. "
    "Please ask your instructor to upload and index the course materials first."
)

_QA_SYSTEM_PROMPT = """\
You are a university course assistant. Answer the student's question ONLY using the provided course material excerpts below.

Rules:
- Answer ONLY from the provided context. Never use outside knowledge.
- If the answer is not in the context, say: "This information is not available in the indexed course materials."
- Always cite which material/chunk you used (e.g., "According to [Material Title], ...")
- Be educational and clear
- Respond in the same language as the question (Arabic or English)
"""

_QA_USER_TEMPLATE = """\
Course Material Context:
{context}

Student Question: {query}

Answer (grounded only in the above context):
"""


class MaterialQAModule:
    """
    Answers questions grounded strictly in indexed course material.

    Constructor parameters:
      model_router — shared ModelRouter instance (injected by PlanExecutor)
    """

    def __init__(self, model_router) -> None:
        self.model_router = model_router

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("MaterialQAModule: starting for user_id=%s", agent_input.user_id)

        ctx: Dict[str, Any] = agent_input.context or {}
        academic_ctx: Dict[str, Any] = ctx.get("academic_context", {}) or {}
        role = ctx.get("role", "student")
        model_id = ctx.get("selected_model", "openai/gpt-4o-mini")

        query = agent_input.message.strip()
        if not query:
            return AgentOutput(
                status="error",
                response="Please provide a question to search the course materials.",
            )

        # ── Extract filters from context ───────────────────────────────────────
        offering_id: Optional[str] = (
            academic_ctx.get("subjectOfferingId")
            or ctx.get("subjectOfferingId")
            or getattr(plan, "offering_id", None)
        )
        material_id: Optional[str] = (
            academic_ctx.get("materialId")
            or ctx.get("materialId")
            or getattr(plan, "material_id", None)
        )

        logger.info(
            "MaterialQAModule: query=%r offering_id=%s material_id=%s",
            query[:80], offering_id, material_id,
        )

        # ── 1. Embed the query ────────────────────────────────────────────────
        try:
            query_embedding = await embedding_service.embed_text(query)
        except Exception as exc:
            logger.error("MaterialQAModule: embedding failed — %s", exc)
            return AgentOutput(
                status="error",
                response=(
                    "حدث خطأ أثناء معالجة سؤالك. الرجاء المحاولة مرة أخرى.\n"
                    "An error occurred while processing your question. Please try again."
                ),
            )

        # ── 2. Search ChromaDB ────────────────────────────────────────────────
        hits = await vector_store.search(
            query_embedding=query_embedding,
            filter_material_id=material_id,
            filter_offering_id=offering_id,
            top_k=5,
        )

        if not hits:
            logger.info("MaterialQAModule: no chunks found for query=%r", query[:80])
            return AgentOutput(
                status="success",
                response=_NO_MATERIAL_MSG,
                data={
                    "module": "MaterialQAModule",
                    "chunks_found": 0,
                    "sources": [],
                },
            )

        # ── 3. Build context (max 3 000 tokens) ───────────────────────────────
        context_str = _build_context(hits, max_tokens=3000)

        # ── 4. Call LLM with grounding prompt ────────────────────────────────
        user_prompt = _QA_USER_TEMPLATE.format(context=context_str, query=query)

        try:
            messages = [
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]
            answer = await self.model_router.generate_with_messages(
                messages=messages, model_id=model_id
            )
        except Exception as exc:
            logger.error("MaterialQAModule: LLM call failed — %s", exc)
            return AgentOutput(
                status="error",
                response=(
                    "تعذّر الحصول على إجابة من النموذج. الرجاء المحاولة مرة أخرى.\n"
                    "Could not get an answer from the AI model. Please try again."
                ),
            )

        if not answer:
            answer = _NO_MATERIAL_MSG

        # ── 5. Build source references ────────────────────────────────────────
        sources = _extract_sources(hits)

        return AgentOutput(
            status="success",
            response=answer,
            data={
                "module":       "MaterialQAModule",
                "chunks_found": len(hits),
                "sources":      sources,
            },
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(hits: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
    """Build a numbered context block from vector-store search hits."""
    ordered = sorted(hits, key=lambda h: h.get("score", 0), reverse=True)
    parts: List[str] = []
    total = 0

    for i, hit in enumerate(ordered, 1):
        title = hit.get("metadata", {}).get("materialTitle", "Course Material")
        content = hit.get("content", "")
        excerpt = f"[{i}] {title}:\n{content}"
        tok = len(excerpt.split())
        if total + tok > max_tokens:
            break
        parts.append(excerpt)
        total += tok

    return "\n\n".join(parts)


def _extract_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a deduplicated list of source references from search hits."""
    seen: set = set()
    sources: List[Dict[str, Any]] = []
    for hit in hits:
        meta = hit.get("metadata", {})
        key = (meta.get("materialId", ""), meta.get("chunkIndex", ""))
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "material_id":    meta.get("materialId", ""),
                    "material_title": meta.get("materialTitle", ""),
                    "chunk_index":    meta.get("chunkIndex", ""),
                    "score":          hit.get("score", 0),
                }
            )
    return sources
