"""
app/modules/summarization.py

Summarization Module.

Flow:
  1. Resolve the subject offering ID from plan / context.
  2. Fetch course material from backend  GET /api/Materials/by-offering/{id}
  3. Extract text from the response (JSON text field, or raw PDF bytes).
  4. Summarize with ModelService (BART-large-CNN).
  5. Return a bullet-point summary.
"""
from __future__ import annotations

import io

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from PDF bytes using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(io.BytesIO(data)) or ""
    except Exception as exc:
        logger.warning("pdfminer extraction failed: %s", exc)
        return ""


def _extract_text_from_bytes(raw: bytes, content_type: str) -> str:
    if "pdf" in content_type.lower():
        return _extract_text_from_pdf(raw)
    # Assume plain text
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


class SummarizationModule:
    """
    Fetches course material from the .NET backend and summarises it
    using the BART-large-CNN HuggingFace model.
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        logger.info("SummarizationModule: starting.")

        # -- 1. Resolve offering ID -------------------------------------------
        context      = agent_input.context or {}
        offering_id  = (
            context.get("offering_id")
            or context.get("subjectOfferingId")
            or (
                plan.exam_params.subjectOfferingId
                if plan and getattr(plan, "exam_params", None)
                else None
            )
        )

        material_text = ""

        # -- 2. Fetch material from backend ------------------------------------
        if offering_id:
            route  = f"/api/Materials/by-offering/{offering_id}"
            result = await self.backend_client.fetch(
                route=route,
                auth_header=agent_input.auth_header,
            )

            if isinstance(result, dict) and "error" in result:
                logger.warning("SummarizationModule: backend error — %s", result["error"])
            elif isinstance(result, dict) and "_raw_bytes" in result:
                material_text = _extract_text_from_bytes(
                    result["_raw_bytes"], result.get("content_type", "")
                )
            else:
                # Unwrap paginated envelope: { "items": [...], "totalCount": N }
                items = result.get("items", []) if isinstance(result, dict) else (result if isinstance(result, list) else [])

                collected: list[str] = []
                for item in items[:5]:
                    if not isinstance(item, dict):
                        continue
                    # 1. Inline text field
                    text = item.get("content") or item.get("text") or item.get("description") or ""
                    if text:
                        collected.append(str(text))
                        continue
                    # 2. Download via fileUrl (public CDN URL built by .NET backend)
                    file_url = item.get("fileUrl") or item.get("signedUrl") or item.get("url") or ""
                    if file_url:
                        try:
                            import httpx
                            async with httpx.AsyncClient() as client:
                                resp = await client.get(file_url, timeout=20.0)
                                resp.raise_for_status()
                            fetched = _extract_text_from_bytes(
                                resp.content, resp.headers.get("content-type", "")
                            )
                            if fetched:
                                title = item.get("fileName") or ""
                                collected.append(f"[{title}]\n{fetched}" if title else fetched)
                                logger.info(
                                    "SummarizationModule: extracted %d chars from '%s'",
                                    len(fetched), title,
                                )
                        except Exception as exc:
                            logger.warning("SummarizationModule: fileUrl fetch failed — %s", exc)

                material_text = "\n\n".join(collected)[:5_000]

        # Fall back to the user's message when no material was fetched
        if not material_text:
            material_text = agent_input.message

        if not material_text.strip():
            return AgentOutput(
                status="failed",
                response="SummarizationModule: no content to summarise.",
            )

        # -- 3. Summarise via ModelRouter --------------------------------------
        selected_model = context.get("selected_model")
        
        # Guard: Prefer task-optimized BART for summarization unless explicitly requested otherwise
        target_model = selected_model
        if not selected_model or not selected_model.startswith("hf/"):
             target_model = "hf/facebook/bart-large-cnn"
             logger.info(
                 "SummarizationModule: overriding pipeline model '%s' with task-optimized 'hf/facebook/bart-large-cnn'.",
                 selected_model or "default"
             )

        logger.info("SummarizationModule: summarising %d chars.", len(material_text))
        summary = await self.model_router.summarize(material_text, model_id=target_model)

        if not summary:
            return AgentOutput(status="failed", response="Summarization failed.")

        # Format as bullet points
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        bullets   = "\n".join(f"- {s}." for s in sentences)

        return AgentOutput(
            status="success",
            response=bullets or summary,
            data={
                "module": "SummarizationModule", 
                "char_count": len(material_text),
                "model_used": target_model
            },
        )

