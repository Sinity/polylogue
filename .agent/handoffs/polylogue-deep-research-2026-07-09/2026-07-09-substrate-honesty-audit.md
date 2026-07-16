# Substrate data-honesty and efficacy audit (2026-07-09)

Read-only audit cluster over the live production archive at
`/home/sinity/.local/share/polylogue` (index.db ~25GB, ops.db ~516MB,
source.db ~23MB, user.db ~32KB, embeddings.db ~3.8GB). Covers Beads
`polylogue-9e5.1`, `polylogue-9e5.3`, `polylogue-9e5.9`, `polylogue-9e5.12`,
`polylogue-9e5.10`. No product code was modified; all SQLite connections were
opened `mode=ro`.

## Method notes

- **Exact, full-population counts** (not samples): every `user.db` number
  (table is empty, trivially exact); every `sessions`, `session_links`,
  `session_context_snapshots`, `session_runs`, `session_profiles` number
  (tables are in the 3.6K–17K row range — full scans are cheap); every
  `messages`/`blocks` group-by (4.4M rows each) — verified via
  `EXPLAIN QUERY PLAN` *before* execution that each ran as `SCAN ... USING
  COVERING INDEX` (index-only, never touches the large TEXT payload columns),
  not an unindexed heap scan. Slowest single query was ~20s. No query in this
  audit showed a non-covering `SCAN` on `messages`/`blocks`/`attachments` and
  none was run to completion without that check.
- **`sqlite_stat1` estimates were wrong by ~6x** and were discarded in favor of
  real `COUNT(*)` queries once EXPLAIN confirmed the queries were index-covered
  and safe to run (e.g. stat1 estimated ~735K messages; the real count is
  ~4.45M). Don't trust `sqlite_stat1` for planning row-count risk on this
  archive — it's stale.
- **Live-ingestion drift**: `polylogued` was actively ingesting during this
  audit (confirmed via `daemon.pid` + a live daemon). Row counts drifted
  slightly between queries run minutes apart (e.g. `sessions` 17076→17079,
  `messages` 4,445,852→4,446,238, `blocks` 4,488,268→4,488,630,
  `session_links` 8,302→8,304 over the audit window). All figures below are
  point-in-time snapshots, not from one consistent transaction; reconciliation
  sums are reported against the count taken in the same breath, with the
  residual drift noted rather than hidden.
- **Beads 9e5.9 and 9e5.12** were investigated primarily via source-code and
  `git log` census (dispatched to two background research subagents, whose
  findings I spot-checked against source directly — e.g. `extraction.py:305-359`,
  `dispatch.py` detector list) rather than live-DB queries; those are exact
  citations, not samples, except the "provider-format-driven vs mechanical"
  commit classification in 9e5.12, which is a judgment call over commit
  subjects, not a mechanical count.
- **All 5 beads were completed** — nothing was time-boxed out or left
  incomplete. The one area that turned up a hard "this data doesn't exist"
  result (9e5.10) is reported as that, not papered over.

---

## polylogue-9e5.1 — Assertion-layer adoption audit

**Result: `user.db` is completely empty.** `SELECT COUNT(*) FROM assertions` = **0**.
`SELECT COUNT(*) FROM user_settings` = **0**. The file is 32KB (schema only,
no data pages of consequence) and its mtime (2026-07-04) predates this audit
by 5 days with no growth.

- All 21 registered `AssertionKind` values (`polylogue/core/enums.py:408-428`:
  `mark, highlight, annotation, correction, suppression, tag, metadata,
  saved_query, recall_pack, workspace_note, note, decision, caveat, lesson,
  blocker, handoff, judgment, run_state, prompt_eval, transform_candidate,
  pathology`) have **zero rows** — 21/21 empty, not a subset.
- `context_policy_json` inject:true count = 0 (no rows to inject).
- `evidence_refs_json` non-empty-array count = 0.
- Reconciliation: `SUM(GROUP BY kind,status,author_kind)` = 0 = total. Trivially
  reconciles because there is nothing to group.
- Impossible statuses / integrity check: none observable (no rows exist to be
  malformed).
- **The mechanism itself is real and wired**, not vaporware: the inject-policy
  read path exists at `polylogue/storage/sqlite/archive_tiers/archive.py:5696-5725`
  (reads `context_policy_json` into a typed `context_policy`), the write path
  defaults new candidate rows to `{"inject": False, "promotion_required": True}`
  (`polylogue/storage/sqlite/archive_tiers/user_write.py:54,1113,1175`), and a
  `context_inject` boolean filter is applied to claims at
  `user_write.py:1646`. `polylogue/core/assertions.py:66` exposes an
  `is_injected` property. So the plumbing for "judge a candidate → flip
  inject:true → it flows into agent context" is fully built and would work the
  moment a row existed — it has simply never been exercised on this archive.

**Verdict:** The assertion/judgment flywheel is not merely under-used, it is
**completely unused** — zero candidates have ever been written, let alone
judged. Per the bead's own three-way framing, this is unambiguously the
**adoption/onboarding** case, not the "mechanism is active, optimize it" case,
and not a data-integrity bug (there's no data to be corrupt). The next
product slice for the ctx/context-loop epic should be making *something*
write a first assertion row (even a low-stakes one, e.g. auto-marking a
session on `mark`/`tag` CLI use, or auto-promoting a `judgment` from an
existing insight) rather than building more assertion-consuming mechanism on
top of a table nobody writes to.

---

## polylogue-9e5.3 — Column honesty audit

Schema source: `polylogue/storage/sqlite/archive_tiers/index.py`. All columns
below are `CHECK`-constrained as described in the bead. The prework packet at
`.agent/scratch/corpus-gpt-pro-2026-07-07/prework-v2/task_packets/task_packets/027_polylogue_9e5_3.md`
mostly restates the bead body and design note verbatim; its "source anchors"
section (`insights/audit.py`, `insights/rigor.py`, `insights/temporal_source.py`)
appears to be boilerplate carried over from a different bead's template — it
is not specific to this column-honesty census and was not used as a basis for
the numbers below (verified de novo against live schema/data instead).

### material_origin (messages)

Total messages (at query time): **4,445,852**. `unknown` sentinel: **1,766
(0.040%)**. Populated (one of the 8 non-unknown values): 4,444,086 (99.96%).

| material_origin | n |
|---|---|
| assistant_authored | 2,533,781 |
| tool_result | 1,690,999 |
| human_authored | 173,800 |
| runtime_context | 14,307 |
| runtime_protocol | 12,932 |
| operator_command | 12,143 |
| generated_context_pack | 5,721 |
| unknown | 1,766 |
| generated_analysis_pack | 395 |

Per-origin `unknown` concentration (origin extracted from `session_id`
prefix): claude-code-session 57/2,052,411 (0.003%), codex-session 0/2,232,363
(0%), chatgpt-export 343/114,274 (0.30%), **hermes-session 1,268/26,117
(4.9%)**, **gemini-cli-session 98/678 (14.5%)**. The two smaller/newer
origins carry almost all the residual unknown mass.

Cross-tab against `role` confirms the column is doing real semantic work, not
defaulting: of 199,408 `role='user'` messages, only 173,801 (87.2%) are
`human_authored` — the remaining 25,607 (12.8%) are `runtime_protocol`
(11,688), `runtime_context` (8,939), `operator_command` (3,285), `unknown`
(1,239), or `generated_context_pack`/`generated_analysis_pack` (456) rows that
share `role='user'` but are structurally distinguished from human authorship
— exactly the CLAUDE.md-documented Claude Code protocol-row exclusion
mechanism, and directly evidenced here.

**Verdict: structural-ready** overall (99.96% populated); **structural-with-caveat**
specifically for `gemini-cli-session` and `hermes-session` origins, which
should carry an explicit coverage caveat in any per-origin analytics that key
off `material_origin`.

### tool_result_is_error / tool_result_exit_code (blocks, eligible = `block_type='tool_result'` only)

Total `tool_result` blocks (at query time): **1,685,155**.

| field | null | populated | populated_pct |
|---|---|---|---|
| tool_result_is_error | 1,233,019 | 452,136 | 26.84% |
| tool_result_exit_code | 1,549,091 | 136,064 | 8.07% |

Per-origin breakdown (this is the load-bearing finding for `polylogue-b0b`,
the heuristic→structural sweep bead):

| origin | tool_result blocks | is_error populated | exit_code populated |
|---|---|---|---|
| codex-session | 955,254 | 136,064 (14.24%) | 136,064 (14.24%) — same rows |
| claude-code-session | ~700,952 | 313,847 (44.77%) | **0 (0%)** |
| claude-ai-export | 2,315 | 2,315 (**100%**) | 0 (0%) |
| chatgpt-export | 13,428 | 0 (0%) | 0 (0%) |
| hermes-session | 13,314 | 0 (0%) | 0 (0%) |
| aistudio-drive | 2 | 0 (0%) | 0 (0%) |
| gemini-cli-session / antigravity-session / grok-export | 0 | n/a | n/a |

`tool_result_exit_code` is **only ever populated for codex-session**, and even
there only 14.24% of the time — it is effectively absent as a general-purpose
signal. `tool_result_is_error` is well-populated for claude-code-session
(44.8%) and fully populated for the small claude-ai-export volume (100%), but
is 0% for chatgpt-export/hermes-session/aistudio-drive and the three origins
with zero tool_result blocks at all.

**Verdict: structural-with-caveat, bordering on keep-heuristic for
exit_code specifically.** Any consumer replacing a prose "did this fail"
heuristic with `tool_result_is_error` must gate on origin
(claude-code-session, codex-session, claude-ai-export only) and must not lean
on `tool_result_exit_code` as a general signal at all — it is a
codex-session-only, partially-populated field.

### message_type / block_type

Both are `NOT NULL` with a closed `CHECK` enum that has **no NULL/unknown
member at all** — the schema makes 100% population a tautology, not an
honesty finding.

| message_type | n | | block_type | n |
|---|---|---|---|---|
| tool_use | 1,669,946 | | tool_result | 1,685,155 |
| tool_result | 1,669,227 | | tool_use | 1,674,353 |
| message | 1,061,396 | | text | 943,889 |
| protocol | 25,075 | | thinking | 151,121 |
| context | 14,572 | | code | 29,114 |
| summary | 5,660 | | document | 3,277 |
| | | | image | 1,350 |

**Verdict: structural-ready** (trivially, by schema construction).

### session_kind

`NOT NULL DEFAULT 'standard'`, CHECK IN (`standard`, `temporary`). **100%
`standard`** (17,076/17,076 sessions at query time); `temporary` has **0**
rows ever.

**Verdict: structural-ready but vestigial.** The column is fully populated
but currently a constant — `temporary` is dead/unreachable in this archive, so
any consumer branching on `session_kind` today gets no discriminating signal.

### branch_type (sessions)

CHECK allows NULL + `continuation`/`sidechain`/`fork`/`subagent`.

| branch_type | n | pct |
|---|---|---|
| NULL | 8,629 | 50.54% |
| subagent | 8,040 | 47.09% |
| continuation | 262 | 1.53% |
| sidechain | 145 | 0.85% |
| fork | 0 | 0% |

NULL here legitimately means "not a branch" (root/standalone session), not
"unknown" — so the 50.5% NULL rate is not itself an honesty problem. But
**`fork` has never been observed** despite being a valid CHECK member, worth
flagging as untested/dead for any consumer or test fixture that assumes it's
reachable.

**Verdict: structural-with-caveat** (NULL is semantically valid, but `fork`
should not be assumed live without further evidence).

### session_links.link_type / inheritance / status

Total `session_links` (at query time): **8,302**.

| link_type | n | pct |
|---|---|---|
| subagent | 8,040 | 96.87% |
| continuation | 262 | 3.16% |
| sidechain / fork / branch / resume / repaired | 0 | 0% |

`inheritance`: `spawned-fresh` 7,737 (93.2%), `prefix-sharing` 345 (4.16%),
NULL 220 (2.65%). `status` (repaired/quarantined): **NULL for 100% of rows**
(8,302/8,302) — the quarantine/repair path described in CLAUDE.md
(`TopologyEdgeStatus` = unresolved/resolved/repaired/**quarantined**) has
never fired a `status` value on this archive.

**Verdict: structural-with-caveat.** Only 2 of 7 possible `link_type` values
have ever been observed. Most consequential for the next section: **`resume`
has never once been recorded as a `link_type`**, despite 262 `continuation`-type
sessions existing.

### Bonus finding directly feeding 9e5.10: `session_context_snapshots.boundary`

Not one of the bead's named columns, but load-bearing for the resume-efficacy
question below, so reported here. Total: **14,422**.

| boundary | n | pct | | inheritance_mode | n |
|---|---|---|---|---|---|
| session_start | 14,377 | 99.69% | | unknown | 14,377 |
| subagent_start | 45 | 0.31% | | summary | 45 |
| resume | 0 | 0% | | | |
| unknown | 0 | 0% | | | |

`boundary='resume'` has **never once been recorded**, and `inheritance_mode`
is `'unknown'` for every `session_start` row (the only mode value ever seen
besides `subagent_start`'s constant `summary`). The context-snapshot table
exists and is populated at real volume, but its two fields most relevant to
"was this a context-informed resume" are functionally empty.

**Verdict: structural-with-caveat**, and this is the concrete evidence behind
9e5.10's verdict below.

### Reconciliation

`COUNT(*)` on each source table matched the grouped-sum totals taken in the
same query pass, within the live-ingestion drift documented in Method notes
(e.g. `messages` GROUP BY material_origin summed to 4,445,844 vs a `COUNT(*)`
of 4,445,852 taken ~1 minute later — an 8-row, sub-0.001% drift consistent
with the daemon actively appending during the audit, not a reconciliation
failure).

### Per-column go/no-go table (feeds `polylogue-b0b`)

| column | verdict |
|---|---|
| material_origin | structural-ready overall; caveat gemini-cli-session, hermes-session |
| tool_result_is_error | structural-with-caveat — gate on origin (claude-code-session, codex-session, claude-ai-export only) |
| tool_result_exit_code | keep-heuristic / do-not-generalize — codex-session only, 14% of even that |
| message_type / block_type | structural-ready (schema-guaranteed) |
| session_kind | structural-ready but vestigial (constant) |
| branch_type | structural-with-caveat (NULL is valid; `fork` untested) |
| session_links.link_type/inheritance/status | structural-with-caveat (narrow coverage; `status` never populated; `resume` never recorded) |

**Verdict:** overall the semantic-classification columns are honestly
implemented (near-100% for the columns that should always populate), but the
**outcome columns (tool_result_is_error/exit_code) are structural-with-caveat
at best and origin-gated** — exactly the caveat `polylogue-b0b` needs before
converting any prose-outcome heuristic to consume them wholesale.

---

## polylogue-9e5.9 — Heuristic accuracy benchmark

This is scored as a **scoping/verdict bead**, per its own framing, not
executed as a full hand-labeling exercise.

**Classifier code located:**
- Work-event-type classifier: `polylogue/archive/session/extraction.py` (556
  lines). Hardcoded confidences blended with structural action counts:
  `_classify_range()` (`extraction.py:305-359`) — 0.75 for ≥2 file edits
  (implementation), 0.7 for refactor/testing/planning/research branches, 0.6
  for git-only ("review") or no-tools ("session"), 0.65/0.5/0.4 for shell-only
  branches down to a `weak_signal` catch-all. Keyword tables
  `_DEBUGGING_PATTERNS`/`_TESTING_PATTERNS`/etc. (`extraction.py:206-233`) feed
  a `_text_signal_from_lowered_text` helper that only breaks ties within the
  structural-count branches — it's a hybrid, not pure prose matching.
- Terminal-state classifier: `polylogue/archive/session/runtime.py:_terminal_state()`
  (lines 239-295). Structural tool-pairing dominates (`tool_left`, confidence
  0.9), with a keyword fallback tier (`_ERROR_MARKERS = ("error", "failed",
  "failure", "traceback", "exception", "panic")`, `runtime.py:199`) for
  `error_left` (0.78/0.7), plus role-based `question_left`/`clean_finish`/`unknown`.
- Both write to `session_work_events` and `session_profiles.terminal_state`
  via `build_session_analysis` (`runtime.py:318,388`) →
  `polylogue/storage/insights/session/storage.py:85-87,246-247` /
  `polylogue/storage/sqlite/queries/session_insight_timeline_writes.py:40`.

**Tooling check:** `devtools bench` exists (`devtools/click_dispatch.py:26`)
with a genuine run/compare skeleton in `devtools/benchmark_campaign.py`
(`run` at line 369, `compare` at line 377, diffing two JSON artifacts via
`_compare_results()`), but there is **no `heuristics` campaign type and no
precision/recall scoring code anywhere** in `devtools/` or `.agent/tools/` — a
`devtools bench heuristics` command as scoped in the bead's design note would
be new work, not an extension of an existing lane.

**No committed labeled fixture exists** for work-event-type or terminal-state
ground truth. `demo/constructs.py:222-231` has synthetic-corpus assertions
keyed on `terminal_state IN ('question_left', 'tool_left')` / `= 'error_left'`,
implying the synthetic session generator (`devtools/large_archive_generator.py`,
`polylogue/schemas/synthetic/models.py`) constructs sessions with a
known-by-construction *terminal state* — but not, as far as located, a
known-by-construction *work-event type* label. So a synthetic-corpus-based
fixture could reduce (not eliminate) the terminal-state labeling burden, but
work-event-type would still need genuine hand-labeling or a generator
extension.

**New evidence this audit produced (cheap, zero hand-labeling):** the archive
already has a structural, non-heuristic proxy for the same coarse
success/failure question — `session_runs.status` (`completed`/`failed`/`unknown`,
derived purely from `tool_result_is_error`/`exit_code` via `_tool_status()`,
`polylogue/insights/transforms.py:2301`, `run_projection.py:425-431`).
Cross-tabbing it against the heuristic `session_profiles.terminal_state` for
`role='main'` runs (14,377 rows, one query, <2s):

| structural status | heuristic terminal_state | n |
|---|---|---|
| completed (n=5,804) | error_left | 2,860 (49.3%) |
| completed | clean_finish | 2,614 (45.0%) |
| completed | tool_left/question_left/unknown | 330 (5.7%) |
| failed (n=3,704) | error_left | 1,861 (50.2%) |
| failed | clean_finish | 1,590 (42.9%) |
| failed | tool_left/question_left/unknown | 253 (6.8%) |
| unknown (n=4,869) | clean_finish | 2,553 (52.4%) |
| unknown | question_left/unknown/error_left | 2,316 (47.6%) |

Treating `error_left` as the heuristic's "error" prediction and everything
else as "no error": binary agreement with structural status on the decisive
(`completed`/`failed`) subset is **(2,944 + 1,861) / 9,508 = 50.5%** —
statistically indistinguishable from a coin flip on this construct. This
doesn't necessarily mean the heuristic is "wrong" (terminal_state and run
status may be measuring genuinely different things — how the *conversation*
ended vs whether *tool calls* succeeded — and the classifier's own low
confidence constants, 0.4–0.5 in ambiguous branches, already hedge this), but
it is a real, free, quantifiable disagreement rate directly answering the
bead's motivating question for the binary error/no-error axis, at zero
labeling cost.

**Verdict:** the bead **as scoped (hand-label ~100 sessions + build a full
`devtools bench heuristics` precision/recall campaign) is not yet ready to
execute** — that's genuine net-new tooling on top of an unquantified,
unfixtured labeling effort, correctly reflected by its own `D-horizon-ready`
delivery-lane note. However, this audit surfaces a **much cheaper re-scoped
first slice that's ready today**: extend the free `session_runs.status` ×
`session_profiles.terminal_state` cross-tab above into a small, committed
`devtools` script (no hand-labeling, no new fixture) to get a standing,
re-runnable accuracy signal for the terminal_state/error axis specifically.
Work-event-*type* accuracy has no structural proxy at all and genuinely needs
either hand-labeling or a synthetic-generator extension — that part remains
legitimately horizon-tier.

---

## polylogue-9e5.12 — Schema-inference ROI

**Structural correction to the bead's framing:** `polylogue/schemas/`
(13,654 lines of `.py` — confirms the bead's "~13.5k lines" — plus 12,190
lines of `.json`/`.json.gz` catalog artifacts, 25,844 lines total) is a
**schema-inference/generation framework** (`schemas/inference/`,
`schemas/generation/`, `schemas/operator/`, `schemas/audit/`,
`schemas/synthetic/`, `schemas/field_stats/`, `schemas/code_detection/`), not
Pydantic provider-record models. The actual Pydantic models that gate
`detect_provider()` live in a separate, much smaller package:
`polylogue/sources/providers/` — **1,343 lines** across `codex.py` (272),
`gemini_message.py` (276), `claude_code_record.py` (197),
`chatgpt_message_models.py` (156), `claude_ai.py` (142),
`claude_code_models.py` (114), `chatgpt_session_models.py` (94), plus small
shims. `schemas/providers/*/catalog.json` is referenced only by
`sources/provider_completeness.py:69-200` (a completeness-check feature, not
detection). Neither `polylogue/pipeline/` nor `polylogue/storage/` imports
either package — provider models are consumed **only at parse time**, never
re-validated at materialize time.

### Per-provider verdict (`polylogue/sources/dispatch.py:124-153`)

| Origin | Detector | Validated? | Verdict |
|---|---|---|---|
| codex-session | `codex.looks_like → _validate_record` (`codex.py:541-563,63`) | **Pydantic**: `CodexRecord.model_validate` | **load-bearing** — the only genuinely schema-gated detector |
| claude-code-session | `claude.looks_like_code → code_detection.looks_like_code` (`code_detection.py:21-33`) | Loose dict-key check only (`parentUuid`/`sessionId`/type membership) | **missing** — `dispatch.py`'s own comment claims Pydantic validation here; it's false. `ClaudeCodeRecord` (`sources/providers/claude_code_record.py:75`) is imported nowhere in the live parse path (`code_parser.py` never references it) — it's exercised only by unit tests, a **dead-in-production, ceremonial model** |
| claude-ai-export | `claude.looks_like_ai → ai_parser.looks_like_ai` (`ai_parser.py:34-46`) | Loose dict-key check at detection; `ClaudeAISession.model_validate` used later, only inside `parse_ai` (line 165) | ceremonial-at-detection / load-bearing-at-parse (split) |
| chatgpt-export | `chatgpt.looks_like` (`chatgpt.py:516-519`) | Loose (`"mapping" in payload`) | **missing** — externally versioned OpenAI export format, single-key detection, high drift risk, currently ungated |
| gemini-cli-session | `local_agent.looks_like_gemini_cli` (`local_agent.py:15-20`) | Loose dict-key | ceremonial-tier (small footprint, low observed churn) |
| hermes-session | `hermes_state.looks_like_state_db_payload` / `local_agent.looks_like_hermes` | Loose dict-key | ceremonial-tier |
| antigravity-session | `antigravity.looks_like_markdown_export`/`looks_like_brain_metadata` (`antigravity.py:198-212`) | Loose dict-key | ceremonial-tier |
| aistudio-drive | none — resolved from source config, not content (`sources/live/batch_support.py:461-466`) | n/a | **missing detection entirely** (config-driven, not content-sniffed) |
| grok-export | **no detector at all** | n/a | **missing** — `Origin.GROK_EXPORT`/`Provider.GROK` exist as vocabulary only (`core/enums.py:50,78`); `detect_provider()` never returns it |

### Telemetry check (ops.db)

`ingest_attempts.error_message` (`polylogue/storage/sqlite/archive_tiers/ops.py:34-46`)
is freeform text — **nothing distinguishes a Pydantic `ValidationError`
rejection from any other parse failure**. `ingest_cursor.failure_count` has no
error taxonomy either. There is no way, today, to query "how often did schema
validation reject a record the loose path would've accepted" — the telemetry
to answer the bead's own measurement (b) doesn't exist as a queryable field.

### Maintenance-cost proxy (git history)

`git log --oneline -- polylogue/schemas` = 107 commits;
`-- polylogue/sources` = 215 commits. Of the last 30 `schemas/` commits,
genuinely provider-format-driven (a real parse/validation bug or format
change): 4/30 (~13%) — e.g. normalizing empty arrays before validation,
accepting Codex object-shaped tool arguments, surfacing silent parse-time
record loss. The remaining ~26/30 are mechanical (renames/refactors,
vocabulary/lint, docs, infra wiring, unrelated test/deploy fixes).

### Test coverage split

Only 8 test files exercise the real Pydantic provider models directly, vs. 12
that exercise `detect_provider()`, vs. 0 that reference the
`schemas/providers/*.json` catalogs at all — consistent with the "13.5k line
schema-inference framework" being tooling/audit machinery around the
much-smaller, load-bearing detection surface, not itself gating detection.

**Verdict:** the bead's own question ("load-bearing or gold-plated") has a
split answer, not a single one — **Codex's Pydantic gate is load-bearing and
earns its keep; the `schemas/` inference/generation framework's ~13.5k lines
are largely orthogonal audit/tooling machinery, not detection logic, so their
ROI question needs to be asked separately from the provider-detector
question the bead's acceptance criteria actually measures.** Concrete,
un-implemented follow-ups this audit surfaces (recommend as separate beads,
not actioned here per audit-only scope):
1. **Fix the `dispatch.py` comment** claiming Claude Code detection is
   Pydantic-validated — it isn't; either wire `ClaudeCodeRecord` into
   `code_detection.py`/`code_parser.py` for real, or correct the comment and
   mark the model as detection-dead.
2. **Add a `grok-export` detector** or, if genuinely out of scope, remove/gate
   the `Origin.GROK_EXPORT` vocabulary so `detect_provider()`'s silence isn't
   mistaken for coverage.
3. **Tighten chatgpt-export detection** (currently single dict-key,
   externally-versioned format, no schema gate) given it's the highest
   format-drift-risk loose detector still ungated.
4. **Tag `ingest_attempts.error_message`** (or add a column) distinguishing
   validation-rejection from other parse failures, so measurement (b) in this
   bead's own design becomes queryable in the future instead of requiring a
   source census every time.

---

## polylogue-9e5.10 — Resume/context efficacy eval (observational)

**This data does not exist yet in a usable form.** Checked every plausible
location:

- `ops.db` has **no MCP-call/hook-event log table at all** — only ingest
  telemetry (`ingest_attempts`, `ingest_cursor`, `convergence_debt`,
  `cursor_lag_samples`, `daemon_events`, `embedding_catchup_runs`,
  `otlp_spans`, `otlp_telemetry`). `daemon_events.kind` values present are
  `ingestion_batch`, `message.appended`, `session.appended`,
  `daemon.lifecycle`, `ingest` — none record MCP tool invocations.
- `otlp_spans` and `otlp_telemetry` (the two tables that could plausibly carry
  MCP-call tracing) are **both empty — 0 rows** in each.
- Source-level confirmation: the MCP call wrapper `_async_safe_call`
  (`polylogue/mcp/server_support.py:210-216`) only catches exceptions and logs
  them via Python's `logger` on failure — it does **not** write any durable
  call-log row on success or failure. There is no code path that would ever
  populate a "was `get_resume_brief`/`compose_context_preamble` invoked for
  session X" fact anywhere in the archive today.
- `/home/sinity/.local/share/polylogue/hooks/` (the on-disk hooks directory)
  is **empty — 0 files**.
- The one candidate durable trace, `session_context_snapshots` (index.db,
  14,422 rows, described fully in the 9e5.3 section above), records a
  `boundary` field with a valid `'resume'` enum member — but **that value has
  never once been written** (100% of rows are `session_start` or
  `subagent_start`); `inheritance_mode` is `'unknown'` for 99.69% of rows.
  This table is populated at real volume but not for the resume-boundary case
  this bead needs.
- `session_links` (8,302 rows) records a `link_type='resume'` CHECK member
  that has likewise **never been observed** — only `subagent` and
  `continuation` link types exist in this archive (see 9e5.3 above). The
  closest proxy for "a session was a continuation" is `sessions.branch_type=
  'continuation'` (262 rows) / `session_links.link_type='continuation'` (262
  rows), but nothing distinguishes, among those 262, which ones actually had
  `get_resume_brief`/`compose_context_preamble` output injected vs. not — the
  n-per-arm split the bead needs (resumed-with-context vs. resumed-bare)
  cannot be constructed from anything in the archive today.

**Verdict:** per the bead's own instruction to say so plainly rather than
fabricate a null result — **this bead cannot be executed as an observational
analysis today because the archive contains zero durable evidence of whether
resume/context-composition tools were ever invoked.** What would need to be
instrumented first, in priority order:
1. A durable call-log for MCP tool invocations (at minimum: tool name,
   session_id argument, timestamp, success/failure) — the missing piece
   underlying both `otlp_spans`/`otlp_telemetry` (present as tables, unused)
   and the `_async_safe_call` wrapper (currently log-only, not persisted).
2. Actually writing `boundary='resume'` `session_context_snapshots` rows when
   a continuation/resume session starts (the schema already supports it; the
   materializer never emits it).
3. Populating `session_links.link_type='resume'` distinctly from
   `'continuation'` so "was this specifically a context-informed resume" is a
   first-class, queryable fact rather than conflated with the broader
   continuation classification.

Only after (1)–(3) exist would a rerun of this bead have n>0 in either arm.
Filing this as a genuine "instrumentation must land before analysis is
possible" finding, consistent with the bead's own `D-horizon-ready` note
(and the immediately-preceding `cfk` controlled-experiment bead this one is
supposed to cheaply pre-size — that pre-sizing cannot happen either, for the
same reason).

---

## Summary verdict table

| Bead | One-line verdict |
|---|---|
| 9e5.1 | Assertion table is 100% empty (0/0 across every kind) — ship adoption, not mechanism. |
| 9e5.3 | Semantic-classification columns (material_origin, message_type, block_type) are honestly populated; outcome columns (tool_result_is_error/exit_code) are origin-gated and must carry per-origin caveats before any heuristic→structural conversion. |
| 9e5.9 | Full scope (hand-label 100 + build bench tooling) not ready; a free structural cross-tab (session_runs.status × terminal_state) already shows ~50% (chance-level) agreement on the error axis and is ready to commit as a cheap first slice today. |
| 9e5.12 | Split verdict — Codex's Pydantic gate is load-bearing; the ~13.5k-line schemas/ framework is mostly orthogonal tooling, not detection; claude-code-session's "Pydantic-validated" claim in dispatch.py is false; grok-export/aistudio-drive have no content-based detector at all. |
| 9e5.10 | Cannot be executed — no MCP-call telemetry exists anywhere in the archive (ops.db tables and disk hooks/ dir are both empty); needs instrumentation before analysis, not analysis of what's there. |
