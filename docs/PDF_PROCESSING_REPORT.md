# PDF Processing Report

> **Files:** `modules/file_extraction.py`, `modules/deep_pdf_study.py`, `api/routes/companion.py` | **Date:** 2026-06-22

---

## 1. Extraction Pipeline

```
PDF bytes
  → pdfminer.six (best quality)
  → pypdf (fallback)
  → vision OCR (gemini-flash-1.5) if text < threshold (scanned/slide PDFs)
```

`extract_text_from_bytes(data, filename)` dispatches by extension: PDF, DOCX, XLSX/XLS, CSV, TXT. Images and scanned PDFs route to the vision model.

## 2. Vision OCR Fix (2026-06-21)

**Bug:** the previous vision fallback sent raw PDF bytes as base64 to the vision model — invalid input → no text → hallucinated answers.

**Fix:** PDFs are now converted page-by-page to PNG images via `_pdf_pages_to_images()` before being sent to the vision model. A `< 150 chars` guard returns an honest error instead of inventing content.

## 3. Large PDF Support ([deep_pdf_study.py](../app/modules/deep_pdf_study.py))

For PDFs > 8,000 chars, `FileExtractionModule` switches to **deep PDF mode**:
- `extract_pdf_pages()` extracts **all** pages (not just the first few).
- `summarize_large_pdf()` processes the full document, returning page-mapped output.

This already supports multi-hundred-page documents. The single-call path caps at 30K chars; the deep path handles the rest page-by-page.

## 4. Page-Aware Answering (Phase 7 — roadmap)

To answer "explain page 72" / "how many pages" precisely, store page metadata on each chunk:

```json
{ "document_id": "...", "filename": "DL.pdf", "page_number": 72, "chunk_index": 5 }
```

Then a query that mentions a page/section filters chunks by `page_number` before generation. The extraction layer already exposes `page_count`; wiring page numbers into chunk metadata is the remaining step.

## 5. Supported Formats

| Format | Extractor | Notes |
|---|---|---|
| PDF (text) | pdfminer → pypdf | best quality |
| PDF (scanned) | vision OCR | page→image→gemini |
| DOCX | python-docx | paragraphs + tables |
| XLSX/XLS | openpyxl / xlrd | headers + rows, capped |
| CSV | csv | capped rows |
| TXT | utf-8 decode | — |
| PNG/JPG | vision OCR | direct |

## 6. Endpoints

- `POST /api/companion/explain-file` — base64 file → explanation + flashcards (with anti-hallucination guard).
- `FileExtractionModule` — used by the chat agent for "explain the file" intents (deep PDF mode for large docs).

## 7. Educational Study Mode (Phase 8 — roadmap)

Designed capabilities to layer on top of extraction: full summary, chapter summary, page-by-page explanation, key concepts, definitions, examples, flashcards, exam questions, mind maps, study plan. Flashcards + exam questions already exist in `companion.py`; the remaining modes reuse the same extracted text + tiered LLM.
