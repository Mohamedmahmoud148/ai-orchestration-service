"""
app/services/chunker.py

Text chunking utility for the RAG pipeline.

Splits text into overlapping chunks using a paragraph → sentence → word
fallback strategy. Each chunk tracks its index and approximate token count.
"""
from __future__ import annotations

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[dict]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* tokens.

    Strategy (in priority order):
      1. Split on paragraph boundaries (blank lines).
      2. If a paragraph is still too large, split on sentence boundaries.
      3. If a sentence is still too large, split on word boundaries.

    Overlap:
      The last *overlap* tokens of every chunk are prepended to the next chunk
      so that context is not lost at chunk boundaries.

    Returns:
      List of dicts:
        {
          "chunk_index": int,
          "content": str,
          "token_count": int,    # approximate (whitespace-split word count)
        }
    """
    if not text or not text.strip():
        return []

    # ── Rough tokenisation: split on whitespace ───────────────────────────────
    # We treat "token ≈ word" throughout for simplicity.
    def _tokenise(s: str) -> List[str]:
        return s.split()

    def _token_count(s: str) -> int:
        return len(_tokenise(s))

    # ── Paragraph splitting ────────────────────────────────────────────────────
    paragraphs: List[str] = re.split(r"\n\s*\n", text.strip())

    # ── Sentence splitting (fallback inside a large paragraph) ────────────────
    _sentence_re = re.compile(r"(?<=[.!?؟])\s+")

    def _split_paragraph(para: str) -> List[str]:
        """Split a paragraph into sentences, or words if sentences are huge."""
        sentences = _sentence_re.split(para.strip())
        parts: List[str] = []
        buf: List[str] = []
        buf_tok = 0
        for sent in sentences:
            sent_tok = _token_count(sent)
            if sent_tok > chunk_size:
                # Flush buffer first
                if buf:
                    parts.append(" ".join(buf))
                    buf, buf_tok = [], 0
                # Word-level split for giant sentences
                words = _tokenise(sent)
                i = 0
                while i < len(words):
                    parts.append(" ".join(words[i : i + chunk_size]))
                    i += chunk_size
            elif buf_tok + sent_tok > chunk_size:
                parts.append(" ".join(buf))
                buf, buf_tok = [sent], sent_tok
            else:
                buf.append(sent)
                buf_tok += sent_tok
        if buf:
            parts.append(" ".join(buf))
        return parts

    # ── Assemble a flat list of segments ──────────────────────────────────────
    segments: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if _token_count(para) <= chunk_size:
            segments.append(para)
        else:
            segments.extend(_split_paragraph(para))

    # ── Build overlapping chunks ───────────────────────────────────────────────
    chunks: List[dict] = []
    overlap_tokens: List[str] = []   # tail of the previous chunk (for overlap)
    chunk_index = 0

    for seg in segments:
        seg_tokens = _tokenise(seg)
        combined = overlap_tokens + seg_tokens

        # Slide through the combined token window
        start = 0
        while start < len(combined):
            end = min(start + chunk_size, len(combined))
            window = combined[start:end]
            content = " ".join(window).strip()
            if content:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "content": content,
                        "token_count": len(window),
                    }
                )
                chunk_index += 1
            if end >= len(combined):
                break
            start += chunk_size - overlap  # slide with overlap

        # Carry-forward overlap = last *overlap* tokens of this segment
        overlap_tokens = seg_tokens[-overlap:] if len(seg_tokens) >= overlap else seg_tokens

    return chunks
