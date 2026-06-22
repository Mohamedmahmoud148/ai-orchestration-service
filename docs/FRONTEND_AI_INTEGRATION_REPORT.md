# Frontend AI Integration Report

> **Audience:** Frontend team | **Date:** 2026-06-22
>
> The frontend calls the **.NET backend**, which proxies to this FastAPI service.
> All endpoints below are the .NET-facing URLs. Auth: `Authorization: Bearer <JWT>`.

---

## 1. Streaming Chat (primary UX)

**`POST /api/chat/stream`** → `text/event-stream` (SSE)

Request:
```json
{ "conversationId": "01HN...", "message": "اشرحلي محتوى الملف" }
```

SSE frames:
```
data: {"type":"token","content":"Your "}
data: {"type":"token","content":"GPA "}
data: {"type":"done","suggestions":["..."],"conversationTitle":null}
```

Frontend:
```js
const res = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ conversationId, message }),
});
const reader = res.body.getReader();
const dec = new TextDecoder();
let full = '';
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of dec.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const f = JSON.parse(line.slice(6));
    if (f.type === 'token') { full += f.content; setText(full); }   // typing effect
    if (f.type === 'done')  { setSuggestions(f.suggestions); }
    if (f.type === 'error') { showError(f.message); }
  }
}
```

**Typing indicator:** show a pulsing cursor while `type:"token"` frames arrive; stop on `type:"done"`.

## 2. Non-streaming Chat

**`POST /api/chat/messages`** → full JSON `{ id, content, sender, isFallback, suggestions, conversationTitle }`. Use when you don't want streaming. If `isFallback:true`, show a subtle "limited response" indicator.

## 3. PDF / File Upload + Explanation

**`POST /api/companion/explain-file`** — `multipart/form-data`, field `file` (PDF/DOCX/XLSX/CSV/TXT/PNG/JPG, ≤100MB).

Response:
```json
{ "explanation": "## ...", "flashcards": [{ "front": "...", "back": "..." }], "charsExtracted": 12500 }
```

Flow + states:
1. Select file → show file chip.
2. Upload → progress bar (`Loading: "جارٍ رفع الملف..."`).
3. Processing → `"جارٍ تحليل الملف..."` (extraction + LLM can take a few seconds).
4. Render `explanation` as markdown; render `flashcards` as a deck.
5. **Honest empty state:** if `charsExtracted < 150`, the API returns a clear "couldn't extract" message — show it, don't pretend.

## 4. Study Mode (Companion)

| Action | Endpoint |
|---|---|
| Flashcards (topic) | `POST /api/companion/flashcards/generate` |
| Quiz/session questions | `POST /api/companion/sessions/{id}/generate-questions` |
| Submit answer (AI-graded) | `POST /api/companion/sessions/{id}/submit-answer` |
| Session report | `GET /api/companion/sessions/{id}/report` |
| Ask about a recording/material | `POST /api/companion/recordings/{id}/ask` |

**Quiz UI:** questions arrive **without** `correctAnswer` (hidden server-side). After `submit-answer`, the response includes `isCorrect`, `correctAnswer`, `explanation` — reveal then.

**Flashcards UI:** flip card (front → back); track review quality 0–5 via `POST /api/companion/flashcards/cards/{id}/review` (spaced repetition).

## 5. Health Indicator

Poll **`GET /health/rag`** (FastAPI) or surface via a .NET passthrough:
```json
{ "status": "healthy", "real_embeddings": true, "vector_store": {"total_chunks": 284} }
```
- 🟢 healthy · 🟡 degraded (tooltip = `status_reason`) · 🔴 unavailable

## 6. Loading / Empty / Error States

| State | Text |
|---|---|
| Streaming | typing cursor |
| File processing | "جارٍ تحليل الملف..." |
| Generating flashcards | "جارٍ إنشاء البطاقات..." |
| Empty file content | API's honest message (charsExtracted < 150) |
| AI service down | "خدمة الذكاء غير متاحة مؤقتاً. حاول مرة أخرى." |

## 7. Error Handling & Retry

| Status | Meaning | UI |
|---|---|---|
| 400 | bad input | show validation message |
| 401 | token expired | refresh token / re-login |
| 429 | rate limited | "حاول بعد لحظات" + backoff |
| 503 | AI temporarily unavailable | retry button |
| SSE `type:"error"` | mid-stream failure | show partial text + retry |

**Retry pattern:** exponential backoff (1s, 2s, 4s) for 429/503; preserve the user's message so retry is one click.

## 8. Progress Indicators

- **Upload:** real byte progress from the `XMLHttpRequest`/`fetch` upload.
- **Processing:** indeterminate spinner (server work, no progress events).
- **Streaming:** the token stream itself is the progress.

## 9. Key Payload Reference

| Endpoint | Method | Body |
|---|---|---|
| `/api/chat/stream` | POST | `{conversationId, message}` → SSE |
| `/api/chat/messages` | POST | `{conversationId, message}` → JSON |
| `/api/companion/explain-file` | POST | multipart `file` |
| `/api/companion/flashcards/generate` | POST | `{topic, subjectOfferingId, cardCount}` |
| `/api/companion/sessions/{id}/submit-answer` | POST | `{questionId, answer}` |
| `/health/rag` | GET | — |
