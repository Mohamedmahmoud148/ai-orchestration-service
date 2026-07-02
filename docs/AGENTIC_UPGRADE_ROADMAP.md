# Agentic Architecture Upgrade — Roadmap

> **Status:** Phases 1-4 implemented and shipped (2026-07-03). Phases 5-12 remain a backlog.
> Checked in with the user twice: once before Phase 5 (deferred — see its section), and again
> after auditing Phases 6-12 individually (see "Phases 6-12 audit" below) — work stopped here by
> explicit choice, not by default. Re-read that audit before resuming any of 6-12; several of them
> looked buildable in the original phase descriptions but turned out to have real blockers once
> checked against the actual code.
> **Constraint that governs every phase:** zero breaking changes to `/api/chat`, `/api/chat/stream`,
> auth, existing modules, existing backend APIs, or the database. Every phase reuses existing
> services rather than rewriting them, and every phase is independently toggle-able or revertible.

## Why this roadmap looks different from the original request

The original ask assumed a pipeline shape — `Intent Classification → RBAC → Planning → Tool
Selection → Tool Execution` as discrete stages — that matched an earlier version of this system.
Verified against the current code: `ReactAgent` already does reason+act as one internal
function-calling loop, and it is the *only* path that runs in production today. The old staged
`PlannerAgent → PlanExecutor` pipeline (`app/agents/agent.py`'s `_plan`/`_route_model`/
`_select_module`/`_execute` helpers) now only fires for one rare edge case — resuming a pending
clarification question.

That changes the honest order of operations: you can't put LangGraph nodes around stages that
don't exist as separate steps yet. Phase 1 wraps what's actually running (the whole `Agent.run()`
call) as a single node. Phases 2+ *create* the separable stages first, then give each one a real
graph node — in that order, not the reverse.

It's also worth noting that some of the original ask already exists and doesn't need to be built:

| Original ask | Already exists as |
|---|---|
| Prompt Templates (Phase 8) | `app/prompts/*.md` with YAML frontmatter + `load_prompt()`/`render_prompt()` loader (`app/prompts/__init__.py`) |
| Human-in-the-loop (Phase 11) | `app/agents/action_guard.py` — plan → ask → approve/cancel → execute, today scoped to `CRITICAL_INTENTS = {action_execute, complaint_submit, file_processing}` |
| Structured intents/roles (part of Phase 9) | `app/schemas/intents.py` (`Intent`, `Role` str-Enums), `app/schemas/contracts.py` (`AcademicContext` Pydantic model) |
| Reliability basics (part of Phase 10) | Circuit breaker (`services/circuit_breaker.py`), LLM fallback chain (`model_router._build_fallback_chain`), per-call timeouts |

Those phases below are framed as *extending* these, not rebuilding them.

---

## Phase 1 — Thin LangGraph Wrapper ✅ Implemented

**What:** `app/agents/graph.py` defines a 3-node `StateGraph` (`entry → agent_core → exit`) where
only `agent_core` does real work — it calls the existing `Agent.run(context)` exactly as `chat.py`
always has. State holds a single field, a reference to the live `ExecutionContext` object (not
field-mapped), preserving the mutate-in-place semantics every other module already depends on.

**Toggle:** `USE_LANGGRAPH_ORCHESTRATION` in `app/core/config.py`, default `False`. When off,
`app/agents/graph.py` (and therefore `langgraph`) is never imported — see `app/main.py`'s lifespan.
`app/api/routes/chat.py::_run_orchestration()` picks direct-`Agent.run()` vs graph based on
whether `app.state.agent_graph` was built.

**Reuses:** `Agent`, `ExecutionContext`, `_PipelineStageError` — unchanged, unmodified.

**Not touched:** `ReactAgent`, `PlannerAgent`, `PlanExecutor`, `core/rbac.py`, `/api/chat/stream`
(the streaming endpoint already bypasses `Agent.run()` for its token-streaming fast path; only its
rare non-streaming fallback calls `agent.run()`, and that's deliberately left as a direct call for
now to keep this phase's surface area minimal).

**Tests:** `tests/test_langgraph_wrapper.py` — construction is inert, flag-on/flag-off produce
identical `ExecutionContext` output, `_PipelineStageError` propagates unchanged, object identity
is preserved, and a subprocess check proves `langgraph` isn't imported when the flag is off.

**Rollback:** flip `USE_LANGGRAPH_ORCHESTRATION=false` in Railway env vars. No code revert needed.

---

## Phase 2 — Extract Stage 0 / Stage 5 into real helpers ✅ Implemented

**What:** `agent.py::run()` had Stage 0 (memory/preferences/entities/academic-profile load) and
Stage 5 (memory save, file-URL extraction, background summarization) as inline blocks inside one
~300-line method. Extracted verbatim (no logic changes) into `Agent._load_memory(context) ->
(memory_key, plan, module_name, should_return_early)` and `Agent._save_memory(context, memory_key,
plan, module_name) -> None`. `run()` now reads as: `_load_memory` → early-return check → the
ReactAgent/clarification-execute branch (unchanged, stays inline — that's Phase 3's job to extract)
→ `_save_memory`.

The one subtlety: the original code has an early `return context` when an in-progress
clarification is answered with an invalid choice, which skips *both* the background
language/entity-extraction tasks *and* Stage 5 entirely. `_load_memory` preserves this exactly via
a `should_return_early` flag that `run()` checks before doing anything else — verified by a
dedicated test (`test_invalid_choice_short_circuits_before_background_tasks`).

**Reuses:** `MemoryStore` (`services/memory_store.py`) — no changes to Redis keys, TTLs, or method
signatures, just where the calling code lives.

**Tests:** `tests/test_agent_memory_stages.py` (13 new characterization tests) lock in: memory-key
composition, all metadata/academic_context population from loaded memory, the file-URL-already-set
guard, background task firing, both clarification-resolution paths (numeric choice, name match,
exam-params population), the invalid-choice short-circuit, both `_save_memory` branches
(clarification-needed vs normal save), file-URL/active-document extraction, and an end-to-end
`run()` orchestration check. Full suite verified before/after: same 7 pre-existing failures
(unrelated to this change — ActionGuard event-loop issue, stale prompt-text assertions, an
out-of-date `Intent` enum count), zero new regressions, 314 passed (up from 301 pre-Phase-2).

**Risk realized:** low, as expected — mechanical extract-method, all characterization tests passed
on the first run.

## Phase 3 — Real 3-node graph ✅ Implemented

**What:** Replaced Phase 1's placeholder `entry`/`agent_core`/`exit` nodes (which just wrapped the
whole `Agent.run()` call) with three real nodes — `load_memory`, `agent_core`, `save_memory` —
each calling one of Phase 2's extracted methods directly. Since each node now does one real thing
instead of one node running the entire pipeline, per-node timing/tracing is now meaningful, not
cosmetic.

One design gap in the original phase description surfaced during implementation: `agent.py`'s
middle branch (ReactAgent smart path vs. legacy-executor clarification path) wasn't extracted by
Phase 2 — it stayed inline in `run()`. Calling `react_agent.run(context)` directly from
`agent_core` (as originally sketched) would have silently dropped the clarification-resolution
branch, which uses the legacy `PlanExecutor` instead. Fixed by extracting that branch too, as
`Agent._execute_core(context, plan, module_name)`, and having `agent_core` call that — preserving
both branches exactly.

The other subtlety: `Agent.run()`'s early `return context` (invalid clarification choice) must
skip *both* `agent_core` and `save_memory`. This is now a genuine LangGraph conditional edge —
`load_memory` routes straight to `END` when `should_return_early` is true, and to `agent_core`
otherwise — rather than an `if` statement, which is the first place this phase actually uses
LangGraph's own branching instead of just sequencing.

**Reuses:** `Agent._load_memory` / `_execute_core` / `_save_memory` (Phase 2's extraction, plus the
one additional extraction above) — no new business logic, purely wiring.

**Graph shape:**
```
START → load_memory ─┬─(should_return_early)─→ END
                      └─(else)──────────────→ agent_core → save_memory → END
```

**Tests:** `tests/test_langgraph_wrapper.py` (8 tests, up from 6) — added a dedicated test for the
conditional early-return edge, and a `TestRealAgentThroughRealGraph` case that runs the *real*
`Agent` class (not the hand-written fake) through the *real* graph, to prove the two modules
compose correctly against Agent's true method signatures. Also added `_load_memory`/`_execute_core`
call-count assertions to the equivalence tests. Full suite verified: same 7 pre-existing failures,
zero new regressions, 316 passed (up from 314 pre-Phase-3).

**Risk realized:** low. The one real design decision (extracting `_execute_core`) was caught and
handled during implementation rather than in a separate review pass, since it was necessary for
correctness, not optional.

## Phase 4 — Streaming via the graph ✅ Implemented

**What:** `/api/chat/stream`'s fallback path (fires when no `ReactAgent` is available) now calls
`_run_orchestration()` — the same helper `/api/chat` uses — instead of calling `agent.run()`
directly, so it routes through the graph when `USE_LANGGRAPH_ORCHESTRATION` is on.

The original plan called for LangGraph's `astream_events` for real token-level streaming through
the graph. That turned out to be unnecessary: this fallback path already computes the full answer
up front and fakes streaming by replaying it in 3-word chunks with a small sleep between each — it
was never real token streaming to begin with. Routing the existing full-answer call through
`_run_orchestration` gives complete parity with `/api/chat` (graph-or-direct dispatch, identical
either way) without adding `astream_events` complexity that wouldn't have changed any observable
behavior. The primary streaming path (`react_agent.stream_run()` called directly, true token-level
SSE) is untouched, as planned — it already bypasses `Agent.run()`/the graph entirely.

**Bug caught by the test suite before it shipped:** the first version of this change reassigned
`context = await _run_orchestration(...)` inside `event_generator()`, a nested closure that reads
`context` from the enclosing function earlier in its body. In Python, assigning to a name anywhere
in a function makes it local to that function throughout — so that single reassignment would have
made every earlier read of `context` in the generator raise `UnboundLocalError` the moment the code
path executed. Fixed by not reassigning at all: `ExecutionContext` is mutated in place and returned
as the same object by both the direct and graph paths (verified by the object-identity tests), so
the return value doesn't need to be captured — matching how the original code called
`await agent.run(context)` without reassignment either.

**Reuses:** `_run_orchestration()` (Phase 1) — zero new dispatch logic, just a second call site.

**Tests:** `tests/test_langgraph_wrapper.py` — two new tests exercise `chat_stream_endpoint`'s
fallback branch directly (constructing the SSE response and consuming its `body_iterator`),
covering both `agent_graph` set and `None`. These are what caught the closure bug above. Full suite
verified: same 7 pre-existing failures, zero new regressions, 318 passed (up from 316 pre-Phase-4).

**Risk realized:** the intended change was low-risk (reusing an already-tested helper), but the
closure bug shows why the "write the test, run it" discipline matters even for changes that look
trivial — a manual code read alone would likely have missed the scoping issue.

## Phase 5 — Decompose ReactAgent's tool loop into graph nodes

**What:** This is the part of the original request that actually rewrites the live reasoning loop
— breaking `ReactAgent`'s hand-rolled `think → tool-call → observe` loop (max 4 iterations,
`_dispatch_tool` parallel execution via `asyncio.gather`) into explicit LangGraph nodes, likely
using LangGraph's own tool-calling primitives. This is the **highest-risk phase** in the whole
roadmap, because it's the one place where "wrap the existing code" stops being possible — the loop
itself has to be re-expressed as graph structure.

**Mitigation:** golden-response regression suite — capture real `(context, expected_tool_calls,
expected_response_shape)` fixtures from the current system first, replay them against the new
graph-based loop, diff before touching production traffic. Should ship behind its own separate
flag, independent of `USE_LANGGRAPH_ORCHESTRATION`, so it can be rolled back independently of
Phase 1-4's wrapper.

**Effort:** ~1-2 weeks.

---

## Phases 6-12 audit (2026-07-03) — why work stopped after Phase 4

Before touching any of Phases 6-12, each was checked against the actual code as it stands after
Phases 1-4, not just against its original description below. Several turned out to have real
blockers that weren't visible when the roadmap was first drafted:

- **Phase 6 (RBAC as a graph edge) — genuinely unsafe as described.** RBAC is deliberately
  enforced in exactly one place today: `PlanExecutor.execute()`'s Step 0. `agent.py` has a comment
  explaining that a previous *duplicate* RBAC gate caused silent permission-mismatch bugs and was
  deliberately removed. Moving the check into the graph would either duplicate it again (the exact
  anti-pattern that comment warns against) or remove it from `PlanExecutor` — which would silently
  disable RBAC on the flag-off path, since that path (`USE_LANGGRAPH_ORCHESTRATION=false`) doesn't
  go through the graph at all. Not safely doable until the graph is the *only* path (see Phase 12's
  original "full cutover" step) or a different mechanism is designed. **Blocked, not just deferred.**

- **Phase 8 (specialized agents as subgraphs)** and the deeper half of **Phase 7** (routing by
  tool/intent before execution) **depend on Phase 5.** Today, tool/intent selection happens
  *inside* `ReactAgent`'s own LLM-driven function-calling loop — there's no pre-computed
  intent to route specialized agents by until that loop is itself decomposed into graph structure.

- **Phase 9 (tool registry enrichment) — nothing to enrich yet.** Checked: `services/tool_registry.py`
  is only read by `Agent._select_module()`, which isn't called anywhere in the live flow (true
  before this session's changes too — it's dead code, not something Phase 2-4 orphaned). Adding
  metadata to a registry nothing consumes is speculative scaffolding with no payoff until Phase 5
  gives the planner something to select tools *from*.

- **Phase 12's "generalize action_guard"** — expanding which intents require confirmation (e.g.
  adding `generate_exam`) is a real UX change: a doctor generating an exam would suddenly see an
  "are you sure?" prompt that doesn't exist today. That's a product decision, not a refactor, and
  outside what "zero breaking changes" authorizes deciding unilaterally.

- **Phase 10 (layered memory)** and **Phase 11 (RAG hybrid retrieval + reranking)** are the two
  that are *technically* buildable standalone right now. Not built anyway: Phase 10 would be a
  facade nothing calls yet (same low-value-scaffolding problem as Phase 9), and Phase 11
  (reranking specifically) changes actual RAG answer ranking/quality behavior — a real feature
  with its own tradeoffs, better scoped as its own dedicated task than folded into a "continue the
  backlog" pass.

**User confirmed stopping here** rather than building any of the above speculatively. Resume
individual phases only when there's a concrete reason to (e.g. Phase 5 gets picked up, unblocking
6/8/9; or a specific RAG-quality or memory-hygiene need makes 10/11 worth building for real
consumers).

---

## Phase 6 — RBAC as an explicit conditional edge

**What:** Relocate the RBAC gate (`core/rbac.py::is_allowed()`, currently invoked from
`executor.py`'s Step 0, plus inline checks inside specific ReactAgent tools like `generate_exam`)
to a single explicit conditional edge in the graph. `core/rbac.py` itself does not change — this
is purely about consolidating *where* it's called from, continuing the "single RBAC gate" design
principle the system already documents.

**Effort:** ~3-4 days.

## Phase 7 — Planner / model-routing as graph nodes

**What:** Fold the legacy clarification-resume path (today's Stage 0.5 in `agent.py`) into the
graph as a conditional branch, using `PlannerAgent`/`ModelRouter` unchanged as node bodies. This
is what finally makes the old staged pipeline's remaining live use case (clarification resumption)
graph-native instead of a special-cased branch in `agent.py`.

**Effort:** ~1 week.

## Phase 8 — Specialized agents as subgraphs

**What:** Introduce the 7 specialized agents from the original request (Student, Doctor, Academic
Advisor, Regulation, Complaint, File, Dynamic API) as LangGraph subgraphs or conditional routing
targets. Each one calls the *existing* modules internally (`academic_advisor.py`, `regulation.py`,
`dynamic_api.py`, `complaint.py`, etc.) — this phase is about routing/organization, not rewriting
module business logic. Natural point to also formalize `Intent → Agent → Tool → Execution` as the
planner's new decision shape (vs. today's `Intent → Module`), per the original request's Phase 4.

**Effort:** ~1-2 weeks, mostly integration/testing time since the underlying modules don't change.

## Phase 9 — Tool Registry enrichment

**What:** `services/tool_registry.py` today is a bare `Dict[str, str]` (intent → module name). Add
`name`, `description`, `permissions`, `parameters`, `timeout`, `retry_policy` metadata per tool,
and have the planner (or Phase 8's specialized agents) select tools dynamically from the enriched
registry instead of via the current `_MODULE_CLASS_MAP` static dict in `executor.py`.

**Effort:** ~4-5 days.

## Phase 10 — Layered memory reorganization

**What:** `services/memory_store.py` already has a well-defined Redis key namespace
(`user:{id}:memory/preferences/entities/pending_action/conv_state/academic_profile/...`). Group
these under named layers (Conversation / User Profile / Academic / Working / Entity / Document
memory) at the *access-pattern* level — i.e. a thin façade organizing which keys belong to which
layer — without changing the underlying Redis store, TTLs, or key formats.

**Effort:** ~3-4 days.

## Phase 11 — RAG upgrades

**What:** Additive improvements on top of the existing `vector_store.py`/`embedding_service.py`
pipeline (already documented in `docs/RAG_PIPELINE_REPORT.md`'s own roadmap section): hybrid
(semantic + keyword) retrieval, query rewriting, a reranking step, citation metadata on returned
chunks. No reindexing rebuild — chunking/embedding/storage stay as-is; only the retrieval step
gains new stages.

**Effort:** ~1-2 weeks (reranking is the most involved piece — needs a cross-encoder or LLM-rerank
call added to the search path).

## Phase 12 — Generalized HITL, observability, prompts, output parsers

Bundles four smaller, independent extensions once the graph structure from Phases 1-8 exists to
hang them on:

- **Generalize `action_guard.py`**: expand `CRITICAL_INTENTS` beyond the current 3
  (`action_execute`, `complaint_submit`, `file_processing`) to cover exam publishing, schedule
  changes, registration — reusing the existing plan→ask→approve→execute pattern and
  `MemoryStore.save_pending_action`/`get_pending_action`, not rebuilding it.
- **Observability/tracing**: now that Phases 3+ give real per-node structure, add
  node/agent/tool/reasoning timing and tracing (LangSmith or a custom exporter) — this was hard to
  make meaningful before nodes did one thing each.
- **Prompt system additions**: add `planner_prompt.md`, `rag_prompt.md`, `tool_prompt.md`,
  `reasoning_prompt.md`, `safety_prompt.md` to the *already-existing* `app/prompts/` loader
  (currently just `role_{student,doctor,admin,superadmin}.md`) — no new prompt infrastructure
  needed, just more files.
- **Output parsers**: formalize `IntentResult`, `ToolCall`, `PlannerDecision`,
  `AcademicAnalysis`, `ExamGeneration`, `ComplaintAnalysis` as Pydantic models building on the
  `Intent`/`Role`/`AcademicContext` schemas that already exist in `app/schemas/`, replacing
  ad-hoc manual JSON parsing where it still occurs.

**Effort:** ~1-2 weeks total across the four pieces, each independently shippable.

---

## Sequencing notes

- Phases 2→3→4 are a clean sequential chain (each depends on the previous).
- Phase 5 (ReactAgent decomposition) can happen any time after Phase 3, independently of Phases
  6-12 — it's the standalone highest-risk item and should get its own dedicated review cycle.
- Phases 9, 10, 11 are independent of each other and of Phase 5 — any can be picked up in
  parallel once Phase 3 exists, since none of them require the tool-loop decomposition.
- Phase 12's four pieces are each independent of one another.
- Total rough sequential effort if done one at a time: ~8-10 weeks. Meaningfully shorter with
  Phases 9/10/11/12 parallelized once Phase 3 lands.
