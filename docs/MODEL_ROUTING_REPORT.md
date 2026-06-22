# Model Routing Report

> **File:** `app/agents/model_router.py` | **Date:** 2026-06-22

---

## 1. Tiering (Phase 13)

`ModelRouter.pick_model(task)` resolves a task name to a configured model tier:

| Tier | Setting | Default | Tasks |
|---|---|---|---|
| Simple | `MODEL_SIMPLE` | `openai/gpt-4o-mini` | classification, titles, short prompts |
| Complex | `MODEL_COMPLEX` | `openai/gpt-4o-mini`* | academic_advisor, pdf_explanation, material_explanation, exam_generation, study_plan, regulation_analysis, doctor_intelligence, deep_pdf |
| Vision | `MODEL_VISION` | `google/gemini-flash-1.5` | vision, ocr, scanned_pdf, image |

\* **Recommended:** set `MODEL_COMPLEX=openai/gpt-4o` on Railway for smarter advisor/exam/PDF reasoning.

```python
from app.agents.model_router import ModelRouter
model = ModelRouter.pick_model("academic_advisor")   # → MODEL_COMPLEX
```

## 2. Fallback Chain (Phase 14)

**Before** (invalid — same model twice):
```
gpt-4o-mini → gpt-4o-mini → (nothing)
```

**After** (real cross-provider failover):
```
primary → google/gemini-flash-1.5 → mistralai/mistral-7b-instruct
```

Configured via `OPENROUTER_FALLBACK_MODEL_1` / `OPENROUTER_FALLBACK_MODEL_2`. `_build_fallback_chain()` dedupes so the primary is never retried as a fallback. Applies to JSON, text, and streaming calls.

## 3. Providers

OpenRouter is the default gateway (any `provider/model` slug). Direct Gemini, Anthropic, and HuggingFace-local paths remain supported for `gemini`, `claude`, and `hf/` model IDs respectively.

## 4. Call Surface

| Method | Use |
|---|---|
| `generate(prompt, system_instruction, model_id)` | single-turn text |
| `generate_with_messages(messages, model_id, response_format?)` | multi-turn |
| `generate_structured_json(prompt, system, model_id)` | strict JSON |
| `stream_with_messages(messages, model_id)` | SSE token stream (with fallback) |

## 5. Adoption

`pick_model()` is **opt-in** — existing callers pass explicit model IDs and keep working unchanged. To adopt tiering, replace hardcoded `"openai/gpt-4o-mini"` in complex modules (academic_advisor, exam_generation, deep_pdf_study, material_explanation) with `ModelRouter.pick_model("<task>")`.
