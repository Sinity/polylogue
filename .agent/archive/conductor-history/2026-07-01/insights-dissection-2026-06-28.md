---
created: "2026-06-28T00:00:00+00:00"
purpose: "Critical dissection of the polylogue 'insights' layer — what each materialized insight computes, who reads it, cost, and whether 'insights' is a coherent category"
status: complete
project: polylogue
---

# Insights Layer Dissection

Read-only analysis. No code changed. Goal: answer the operator's skepticism that
"insights" is even a coherent category — what is computed, why, how they relate,
what is missing / misleading / useless / redundant.

## TL;DR

"Insights" is not one thing. It is **three structurally different products glued
under one deferred materialization stage and one `materializer_version`**:

1. **Cheap deterministic per-session stats** (`session_profiles` evidence half,
   `session_latency_profiles`, `session_tag_rollups`, `threads`, `thread_sessions`)
   — counts, sums, percentiles, time-gap segmentation. These are the *legitimate*
   insights and they are fully consumed.
2. **Heuristic guesses** (`session_work_events`, the `session_profiles` inference
   half: `workflow_shape` / `terminal_state`) — keyword/action-count classifiers
   with hard-coded confidence constants. Consumed, but presented next to facts
   with weak provenance signalling.
3. **Expensive regex text-mining whose output is never read**
   (`session_runs`, `session_observed_events`, `session_context_snapshots`) —
   built by running the full `compile_recovery_digest()` (incl. `_events_from_text`)
   over every session, and **no CLI / MCP / API / daemon surface reads the three
   tables it populates.** This is the 76%/292K-regex-call cost the profile found.

The single most actionable finding: **the profiled cost is paid to populate three
dead tables.** See "The smoking gun" below.

---

## Per-insight table

| Insight (table) | What it computes | From | How | Consumed by | Cost | Verdict |
|---|---|---|---|---|---|---|
| **session_profiles** (evidence half) | counts (messages, words, tool_use, thinking, attachments), durations (engaged/tool_active/wall/think/output ms), token lanes, cost rollup, latency percentiles | messages, blocks, session_events, cost rows | **deterministic** aggregation | CLI `analyze insights profiles`; MCP `session_profile`/list; API `list_session_profile_insights`; daemon `/api/insights/.../?include=profile` | cheap | **keep, inline** |
| **session_profiles** (inference half) | `workflow_shape` (+method/confidence/features), `terminal_state` (+method/confidence/evidence) | derived analysis | **heuristic** classifier, hard-coded confidence | same as above | cheap-ish | **keep but flag as guess** |
| **session_work_events** | per message-range weak label (planning/impl/debug/test/review/research/config/docs/refactor/data_analysis/session), confidence, file_paths, tools_used, duration | phase-segmented message ranges; action-category counts + regex over user text | **heuristic** (`extraction.py:_classify_range`, confidences 0.4–0.75 literals) | CLI `work-events`; MCP list; API `list_session_work_event_insights`; daemon `?include=timeline` | medium (regex over user text only) | **keep but isolate; honesty fix** |
| **session_phases** | time-gap-segmented intervals: range, duration_ms, tool_counts, word_count | message timestamps (fallback: session_events) | **deterministic** (5-min idle gap, `phase/extraction.py`) | CLI `phases`; MCP list; API `list_session_phase_insights`; daemon `?include=phases` | cheap | **keep, inline; drop dead cols** |
| **session_latency_profiles** | median/p90/max tool-call ms, stuck_tool_count, median agent/user response ms, per-category counts | profile + tool/message timing | **deterministic** percentiles | MCP `session_latency_profile`, `find_stuck_sessions`; API; **not** CLI/daemon | cheap | **keep** |
| **threads / thread_sessions** | logical-session lineage spine: session_ids, count, depth, branch_count, totals, repo, work-event breakdown | topology/parent chain + profiles | **structural projection** (lineage) | CLI `threads`; MCP list; API `list_thread_insights`; daemon `?include=threads` | cheap-ish | **keep; this is structural, not "insight"** |
| **session_tag_rollups** | per (tag,day,origin): session_count, logical_session_count, explicit/auto counts, repo breakdown | profiles' tags | **deterministic** aggregation | CLI `tags`; MCP list; API; **not** daemon | cheap | **keep** |
| **session_runs** | run projection: run_ref, role(main/subagent), harness, status, confidence(raw/inferred), cwd, branch, lineage/evidence refs | `compile_recovery_digest().run_projection` | **regex-mined + structural** | **NOTHING** outside storage (`run_projection_reads.py`) | **EXPENSIVE** | **CUT or justify** |
| **session_observed_events** | observed events: kind, summary, delivery_state, subject/object refs | `run_projection.events` ← `_extract_events`→`_events_from_text` | **regex text-mining** | **NOTHING** | **EXPENSIVE (this is the 76%)** | **CUT or justify** |
| **session_context_snapshots** | context boundary/inheritance per run, segment refs | run_projection | **structural** | **NOTHING** | tied to above | **CUT or justify** |

(`session_profiles`, `latency_profiles`, `tag_rollups`, `threads` carry the same
`materializer_version`/`materialized_at`/`input_high_water_mark` bookkeeping —
one version number spanning all three product classes.)

---

## The smoking gun (cost vs. consumers)

`storage/insights/session/rebuild.py:615-617`:

```python
from polylogue.insights.transforms import compile_recovery_digest
run_projection = compile_recovery_digest(session, session_links=()).run_projection
```

`compile_recovery_digest()` (`insights/transforms.py:454`) eagerly runs the FULL
recovery pipeline on every session during materialization:
`_extract_tool_summaries`, `_extract_subagent_reports`, `_extract_run_state`,
`_extract_events` (→ `_events_from_text`, **transforms.py:1810**),
`_extract_decision_candidates`, plus `render_resume_bundle` and
`_build_forensic_index`.

`_events_from_text` (1810-1893) iterates **every line of every message's text and
every block's text** for every session, running ~12 compiled regexes
(`_MERGED_RE`, `_CREATED_PR_RE`, `_REVIEW_PR_RE`, `_PR_RE`, `_ISSUE_RE`,
`_TEST_PASS_RE`, `_TEST_FAIL_RE`, `_CHECK_PASS_RE`, `_CHECK_FAIL_RE`, …) per line.
That is the cProfile finding: 292K `re.Pattern.search`, 76% of deferred-insight
cost = ~9× the archive write.

**Only `.run_projection` is kept** (rebuild.py:617). The rest of the digest
(events, tool_summaries, decisions, resume_markdown, forensic_index) is discarded.
And `.run_projection` feeds exactly three tables — `session_runs`,
`session_observed_events`, `session_context_snapshots` — which the consumer sweep
found **have no reader anywhere** (CLI, MCP, API, daemon all absent; only
`storage/repository/insight/run_projection_reads.py` touches them internally).

So: the single most expensive thing the insight stage does is mine PR/issue/test
events out of free-text to populate tables nothing queries. Cutting the
`run_projection` materialization (or making it lazy / on-demand from the recovery
MCP path that already calls `compile_recovery_digest` directly) reclaims the bulk
of the 9× overhead with zero loss of any *read* surface.

---

## Critique by the operator's dimensions

### Misleading (heuristics presented as fact)

- **`session_work_events` labels are guesses with fabricated confidence.**
  `archive/session/extraction.py:_classify_range` returns hard-coded confidences:
  IMPLEMENTATION 0.75, PLANNING/RESEARCH/TESTING/REFACTORING 0.7, REVIEW 0.6,
  shell-default 0.5, weak_signal 0.4. These are not calibrated probabilities;
  they are author priors. The DDL stores them in a `confidence REAL CHECK BETWEEN
  0 AND 1` column that *looks* like a model score. A consumer reading
  `confidence=0.75` will reasonably mistake a keyword match for a measurement.
  The label set itself (planning/debugging/review/…) is multi-provider-blind: it
  keys off action categories and English keyword regexes over *user* text; on
  Codex/Gemini transcripts or non-English prompts it degrades to
  IMPLEMENTATION/SESSION defaults silently.
- **`workflow_shape` / `terminal_state`** on `session_profiles` are inference, and
  to the project's credit they carry `_method`/`_confidence`/`_evidence` siblings.
  But the CLI profiles view (`insights/registry.py:389-407`) renders `shape` and
  `state` inline in the same field group as deterministic counts, with no visual
  separation of evidence-tier vs inference-tier unless the operator passes
  `--tier`. The honest signal exists in the schema; the default surface flattens it.
- **Naming.** `session_work_events` and `session_observed_events` both read as
  "events that happened." Phases used to be intent-classified; the `kind` field
  was removed (`phase/extraction.py:26-32` docstring) — yet the DDL still has a
  `phase_type` column and the builder writes the constant string `"phase"`
  (`timeline_rows.py:306`). A reader of the column would expect a taxonomy; it is
  a dead constant.

### Useless / unread / redundant

- **`session_runs`, `session_observed_events`, `session_context_snapshots`:** no
  external reader (see smoking gun). Pure cost. Candidates for deletion or lazy
  recomputation. The recovery/postmortem MCP tools call `compile_recovery_digest`
  *directly* at query time anyway, so the materialized copies are not even the
  serving path for the recovery surfaces — they are a redundant, unread cache.
- **`session_phases.phase_type`** — always `"phase"`. Dead column.
- **`session_phases.confidence`** — always `0.0` (the `SessionPhase` default is
  never set for the timestamp path; `timeline_rows.py` copies `phase.confidence`).
  A confidence column that is structurally always zero is noise.
- **Redundant projection:** `session_observed_events` re-derives PR/issue/test
  "events" that are *also* recoverable on demand from `messages` (the regex runs
  against message text either way). It is a derived projection of data already in
  the source table, materialized but unread — the textbook redundant projection.

### Missing / approximated badly

- **Tool calls are already structured; the regex ignores that.**
  `_events_from_text` scrapes `"N passed"`, `"gh ..."`, `"created PR #..."` out of
  free text, but the parsers already produce typed `tool_use`/`tool_result`
  blocks with `tool_name`, `tool_input`, `tool_id` (used correctly by
  `_extract_tool_summaries`, transforms.py:1551). PR/issue/test/check events
  should be derived from the structured `Bash`/`gh` tool calls and their typed
  results, not by regexing rendered output lines. Regex over prose is both the
  expensive path and the unsound one (it matches `"5 passed"` anywhere, attributes
  `closed #123` to whoever quoted it, treats any `name ... ok` as a check).
- **No calibrated confidence anywhere.** Every "confidence" in the layer is a
  literal. There is no provenance distinguishing "provider reported this" from
  "we guessed from a keyword."
- **Phase semantics are thin.** Phases are pure 5-min idle-gap intervals
  (`_PHASE_GAP`). That is a defensible deterministic signal, but it is labelled and
  schema'd as if richer; the intent classification that would make it an "insight"
  was removed and the column scaffolding left behind.

### Conflation (the central question)

`_PER_SESSION_INSIGHT_TABLES` (rebuild.py:864-872) deletes/rebuilds all seven
per-session tables as one unit, and `build_session_insight_records`
(rebuild.py:552-644) computes all of them in one function with one
`materialized_at` and one materializer version. This unifies, under one deferred
stage:

- O(messages) **deterministic stats** that are trivially cheap and always wanted,
- O(messages) **heuristic classification** (work_events, shape, state) that is
  medium cost and should advertise its uncertainty,
- O(messages × lines × regexes) **text mining** (run projection) that is
  expensive and feeds nothing.

Because they share a stage and a version, the cheap-and-wanted stats pay the
latency of the expensive-and-unread mining on every rebuild, and you cannot
bump/retire one product class without versioning the others.

**Proposed re-categorization:**

1. **Inline deterministic stats (Ring 1, compute at write):**
   `session_profiles` (evidence half), `session_phases` (drop `phase_type` +
   `confidence`), `session_latency_profiles`, `session_tag_rollups`. Counts/sums/
   percentiles/gap-segmentation are cheap enough to compute on the ingest write
   path, not a separate deferred materialization.

2. **Structural lineage projection (its own stage):** `threads`,
   `thread_sessions`, and the run/lineage spine. This is topology, not "insight";
   it is already half-owned by topology (rebuild.py:763 `_materialize_thread_spine`).
   Keep it, name it structurally.

3. **Heuristic semantics (explicitly "guessed", separately versioned):**
   `session_work_events` and the `session_profiles` inference half. Give this its
   own `inference_version` (it already has `inference_family`) and make every read
   surface render the inference tier distinctly from evidence.

4. **Expensive recovery mining (on-demand, not materialized):** the
   `compile_recovery_digest` run-projection. Delete `session_runs` /
   `session_observed_events` / `session_context_snapshots` materialization, or gate
   it behind an opt-in, and serve the recovery/postmortem surfaces from the
   already-existing direct `compile_recovery_digest` call. Replace the
   `_events_from_text` prose-regex with derivation from structured tool calls.

### Soundness of the heuristics

- `_events_from_text` patterns are ad hoc and prose-coupled: `_TEST_PASS_RE =
  r"\b(\d+)\s+passed\b"` fires on any sentence containing "3 passed";
  `_CHECK_PASS_RE = r"([A-Za-z0-9_.() -]+)\s+\.\.\.\s+ok"` matches arbitrary
  "... ok" lines; `review_kind` is decided by substring vote
  (`"addressed"/"acted on"/"fixed"` → acted_on) — a brittle keyword sentiment
  classifier. On multi-provider data (Codex/Gemini render tool output
  differently, non-English sessions) these produce garbage or nothing. Regex over
  rendered text is the wrong tool when typed tool-call structure exists.
- `_classify_range` (work events) is more principled — it leans on
  action-category *counts* first (edit/read/search/shell/git/agent), text regex
  only as tiebreaker — but the confidences are invented and the label space is
  English/Claude-Code-shaped.
- Phase segmentation is sound and deterministic; it just isn't an "insight," and
  its schema carries two dead columns.

---

## Bottom line

Split "insights" into the four classes above. The immediate win, independent of
any redesign, is to stop materializing the run-projection trio: it is the entire
profiled regex cost and it populates tables no surface reads. After that, move the
cheap deterministic stats onto the write path and quarantine the heuristic labels
behind an inference tier the read surfaces actually distinguish.
