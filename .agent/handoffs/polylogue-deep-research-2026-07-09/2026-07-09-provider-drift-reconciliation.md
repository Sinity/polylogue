---
created: "2026-07-09T00:00:00+00:00"
purpose: "Read-only reconciliation of 3 overlapping audit-lane beads (9e5.8, ivsc, 38x) — Provider->Origin retirement census, Codex state_5 token-drift classifier, archived-audit residue vs current source"
status: complete
project: polylogue
method: >
  Source-grounded only. Every classification below is backed by a file:line/function
  citation, a git-log/test cross-check where the finding claims "fixed", and (for ivsc)
  a live read-only query against ~/.codex/state_5.sqlite plus the seed probe JSON.
  No product code, tests, or bd state were modified.
---

# Provider-drift reconciliation (2026-07-09)

## 1. polylogue-9e5.8 — Provider->Origin sequenced retirement plan

### Method
`rg -c "Provider\."` / `rg -l` across `polylogue/` (excl. tests), plus explicit census of
`project_origin_payload(` / `provider_from_origin(` call sites (the transitional-shim pair).

### Census

- **321** non-test `Provider.` usages; **`Provider.`-or-shim-touching files by top package**:
  sources 31, storage 20, archive 16, schemas 15, mcp 4, cli 4, pipeline 3, insights 3,
  browser_capture 2, operations 1, core 1, api 1.
- `provider_from_origin(` / `project_origin_payload(` call sites (the shim): 39 non-test
  sites across `mcp/server_insight_tools.py`, `mcp/server_tools.py`,
  `mcp/server_maintenance_tools.py`, `mcp/insight_tool_contracts.py`, `cli/read_views/neighbors.py`,
  `cli/shared/insight_command_contracts.py`, `cli/commands/{insights,maintenance}.py`,
  `api/archive.py`, `insights/{resume.py,registry.py}`, `sources/live/watcher.py`,
  `operations/archive_debt.py`, `archive/coverage.py`, `archive/query/{support.py,fields.py,miss_diagnostics.py}`,
  `storage/{repair.py,hydrators.py}`, `storage/insights/session/latency_profiles.py`,
  `storage/sqlite/queries/{raw_reads.py,mappers_archive.py}`, `schemas/sampling_db.py`.
  `core/sources.py:299` is the shim's single definition (`provider_from_origin`);
  `insights/registry.py:114` is `project_origin_payload`'s single definition.

### Three-tier classification (per the bead's own scheme)

**Tier A — wire-boundary-legitimate (do not touch; `Provider` is the correct vocabulary here).**
`sources/**` (31 files: parsers, dispatch, decoder_zip, drive, browser_capture identity),
`schemas/**` (15 files: providers/, validation/, sampling*, generation/provider_bundle*,
runtime_registry, observation_*), `pipeline/ids.py`, `pipeline/stage_models.py`. These read/write
raw provider-shaped export payloads (`sources/dispatch.py:27-29` `BUNDLE_PROVIDERS`/`GROUP_PROVIDERS`/
`STREAM_RECORD_PROVIDERS`; `sources/providers/*.py:*` — literal `provider=Provider.CODEX` etc. at
parse time) or provider-keyed schema/validator registries (`schemas/validator.py:71`
`_RECORD_VALIDATION_PROVIDERS`). Retiring `Provider` here would require re-deriving parse-time
identity from `Origin`, which is lossy exactly where the non-injective collapse bites
(GEMINI vs DRIVE — see below) — **never flip this tier**.

**Tier B — transitional-shim consumers (safe to leave; already isolated behind the shim).**
The 39 `provider_from_origin`/`project_origin_payload` call sites above. Every one of these
takes a public `Origin` value in and produces a provider-vocabulary value only for an internal
computation (cost-catalog lookup by provider name, a `source_name` field on a typed row, a legacy
`"provider"` key in a payload about to be re-projected). `project_origin_payload` (registry.py:114)
is itself the shim that walks a payload tree and rewrites provider-token keys to origin at the
public boundary — every MCP/CLI insight response passes through it (`server_insight_tools.py:94/164/187/266`,
`cli/commands/insights.py:309/368`). **This tier is correctly shaped already** — no flip needed;
it *is* the byte-compat seam the sequenced plan should preserve.

**Tier C — residual leak candidates (need per-site check before flip).**
`storage/**` (20 files) and `archive/**` (16 files) outside the Tier-B shim list are the
ambiguous middle: some are legitimate internal plumbing (e.g. `storage/usage.py:65/76/87`
literal `Provider.CLAUDE_CODE.value`/`Provider.CODEX.value`/`Provider.CHATGPT.value` as dict keys
for a hardcoded provider-usage-event catalog — Tier A-like, stays), others are candidates for a
`Provider`->`Origin` flip once the non-injective collapse is resolved (see below). Concretely:
`archive/message/models.py:59`, `archive/viewport/models.py:34/54/158`,
`archive/session/{events.py:42,neighbor_candidates.py:126,domain_models.py:28}`,
`archive/semantic/support.py:98`, `archive/query/archive_execution.py:46-48` (a literal
Origin-string -> `Provider` dict — this one is backwards from the intended direction and is
a genuine **leak candidate**, not a shim). `insights/tag_rollups.py:49` also converts a public
`provider` filter param into `Origin` via `Provider.from_string` then `origin_from_provider` —
this is the *inverse* direction of the shim (accepting legacy `provider:` filter input) and
should be checked against `jnj.7` (CLI help leakage) scope before any flip.

### The non-injective blocker (unchanged, still governs sequencing)
`GEMINI` and `DRIVE` both collapse to `Origin.AISTUDIO_DRIVE` (`core/sources.py` — confirmed
still the mapping direction feeding `provider_from_origin`/`origin_from_provider`). Any surface
in Tier C that needs to go **the reverse direction** (`Origin` -> `Provider`, to recover which of
GEMINI/DRIVE a session actually came from) cannot be flipped to origin-only until a disambiguating
field is added (`Source.family`/`runtime_root` per `core/sources.py`'s richer `Source` type is the
existing candidate carrier — not yet wired into any of the Tier C reverse-direction sites).
`archive/query/archive_execution.py:46-48`'s literal `Origin`->`Provider` dict is exactly the
place this bites today.

### Sequenced plan
1. **No-op / already correct**: Tier A (sources/schemas/pipeline) — leave `Provider` in place permanently.
2. **No-op / already correct**: Tier B (39 shim call sites) — the `project_origin_payload`/
   `provider_from_origin` boundary is doing its job; do not unwind it.
3. **Step 1 (safe now)**: flip the Tier-C sites that only go `Origin -> Provider` for an
   *internal* lookup key and never re-expose the provider token publicly (e.g.
   `storage/insights/session/latency_profiles.py:57/73`, `storage/hydrators.py:213`,
   `storage/sqlite/queries/{raw_reads.py:230,mappers_archive.py:130/157}`) to call
   `origin_from_provider`/native `Origin` handling directly where a `Session.origin` is already
   in scope, dropping the round-trip through `Provider`. Byte-compat: none of these are public
   payload fields (verify against a golden diff per PR).
4. **Step 2 (needs the disambiguator first)**: any site needing `Origin -> Provider` reverse
   lookup for GEMINI vs DRIVE distinction (`archive/query/archive_execution.py:46-48`,
   `insights/tag_rollups.py:49`'s legacy-filter-input path) — **blocked** until a `Source`-family
   disambiguating column/param exists; flipping earlier would silently merge GEMINI-origin and
   DRIVE-origin queries.
5. **Step 3 (final gate)**: layering lint (`docs/plans/layering.yaml`) updated to restrict
   `Provider` importability to `sources/` + `schemas/` + `pipeline/ids.py` once steps 1-2 land,
   catching any future regression.

### Follow-up bead proposals
- **"Flip Tier-C origin-scoped internal lookups off Provider round-trip"** — the 5-6 sites in
  step 3 above (latency_profiles, hydrators, raw_reads, mappers_archive) that already have
  `Origin` in scope and only detour through `Provider` for an internal dict key.
- **"Add Source-family disambiguator for GEMINI/DRIVE reverse lookups"** — precondition for
  step 4; scope = `archive/query/archive_execution.py` origin->provider dict +
  `insights/tag_rollups.py:49` legacy filter path.
- **"Layering lint: gate Provider importability to sources/schemas/pipeline.ids"** — the
  step-5 enforcement bead, filed against `docs/plans/layering.yaml`, opened only after 1-2 land.

---

## 2. polylogue-ivsc — Classify Codex state_5 token drift outside lineage replay

### Method
Read the seed artifact
`/realm/tmp/polylogue-cost-reconciliation/codex-logical-probe-current-max100.json` (present,
189KB). Cross-checked its `details.copied_path` (`codex-state-rilh1qk8.sqlite`, one of the
snapshot copies under the same directory) and the **live** `~/.codex/state_5.sqlite`
(read-only `sqlite3 -readonly`, `.tables`/`.schema threads`/aggregate `SELECT`s only — no writes).

### Finding 1 (correction to the bead's own diagnostic premise)
The bead text says the residual class is "external state_5.sqlite thread rows with
`archived=0` and `has_user_event=0`, while archive sessions contain real user/assistant
messages" — implying this is a *distinguishing* flag for the 78/182 residual threads.

**Empirically, on both the probe's own snapshot and the live DB, `archived=0` and
`has_user_event=0` hold for 100% of rows, not just the residuals**:

```
sqlite3 ~/.codex/state_5.sqlite "SELECT archived, has_user_event, COUNT(*) FROM threads GROUP BY archived, has_user_event;"
-- 0|0|2463   (all 2463 rows)
```

Same result against the probe's actual comparison snapshot
(`codex-state-rilh1qk8.sqlite`, 2459 rows, same 0|0 uniformity). **`has_user_event` is either
dead/never-set in this Codex CLI version on this install, or tracks something unrelated to
"has a real first message"** — confirmed separately: many `tokens_used`-repeated rows carry a
genuine, content-specific `first_user_message` (e.g. `"in /realm/inbox/download there should be
samsung health data"`), so message content is present even though `has_user_event=0`. This flag
is **not usable as a discriminator** and the reconciliation probe/status semantics should stop
citing it as one.

### Finding 2 (the actual discriminating signal): `tokens_used` value shape
Querying the live DB's `tokens_used` distribution reveals three distinct non-organic subclasses,
none of which represent trustworthy live per-thread cumulative usage:

1. **Exact-zero sentinel** — 180/2463 threads (7.3%) have `tokens_used=0` outright (thread row
   created, never updated by any usage-reporting path).
2. **Repeated-identical-constant class** — the specific values driving the outside-tolerance
   population cluster on a small set of exact-duplicate numbers shared across threads with
   unrelated content/dates/`cwd`: `272000` appears on 17 distinct threads, `258400` on 8,
   `58742206` on 3, `3445249` on 2 (`repeated_outside_external_token_values` in the seed JSON;
   confirmed live: `SELECT tokens_used, COUNT(*) FROM threads WHERE archived=0 AND
   has_user_event=0 GROUP BY tokens_used ORDER BY COUNT(*) DESC` reproduces the same top values).
   `272000`/`258400` are round numbers in the same order of magnitude as a model context-window
   size, not an organically-accumulated sum — evidence of a **stale/default value stamped once
   and never updated** for these specific threads (all have `model=''` in the live DB — i.e. no
   model-usage event was ever recorded against them either).
3. **Implausible billion-scale outliers** — a handful of rows (e.g. `5246862413`, `4908442169`,
   `1287786830`) are physically impossible as a single thread's token count. Notably, subagent
   worker threads sharing one `parent_thread_id` (`019cbcef-7ec3-70b2-a556-8456f0ec2741`, agents
   "Sartre"/"Aquinas"/"Ptolemy") carry near-identical billion-scale values
   (`1287786830`/`1234079262`/`1235152917`/`1236327061`) — evidence `tokens_used` on these rows
   is a **parent/account-level cumulative counter snapshotted onto the child row**, i.e. a
   *different accounting grain* (cross-thread cumulative), not the child's own usage.

### Classification (answers the bead's exact question)
`state_5.sqlite.tokens_used` is **not a single semantic field** for the archived=0/has_user_event=0
population: it is variously (a) an unset zero sentinel, (b) a stale context-window-sized default
never updated because no usage-reporting event fired for that thread, or (c) a cross-thread/account
cumulative counter inherited by subagent-worker children. None of the three is comparable 1:1
against polylogue's per-thread `session_model_usage` sum, which *is* computed from the real parsed
rollout content (confirmed real message text exists for these threads). **This is not the
fork/resume lineage-replay double-count class** (correctly ruled out by the bead) **and it is not
an archive-side accounting bug either** — it is an artifact of Codex's own local bookkeeping being
unreliable/stale for a subset of threads that the CLI itself never fully instruments.

### Recommended probe adjustment (not implemented — evidence only)
The reconciliation probe should classify a thread as `external-state-unreliable` (not
`outside_tolerance`) when its `state_5` row has `tokens_used=0`, `tokens_used` in the
repeated-across-unrelated-threads set, or `tokens_used` implausibly exceeds any real context
window (e.g. `> 2_000_000`), rather than folding all three into one undifferentiated
tolerance-failure count. This directly satisfies the bead's AC ("distinguishes lineage replay
residuals from external-state/accounting-grain drift").

### Follow-up bead proposals
- **"Reconciliation probe: reclassify state_5 sentinel/stale/cross-thread tokens_used as
  external-state-unreliable, not outside-tolerance"** — implement the 3-subclass filter above in
  `devtools lab probe cost-reconciliation`; drop `archived=0`/`has_user_event=0` as a cited
  discriminator since it is non-discriminating.
- **"Investigate why `has_user_event` is universally 0 in local state_5.sqlite"** — either a
  Codex CLI regression on this install/version or the field tracks a different telemetry event
  than message presence; low priority, informational only (does not gate any polylogue fix).
- No polylogue-side code fix is indicated by this investigation; the bead's own framing
  ("classify... update the reconciliation probe... status semantics") is satisfied by the
  probe-adjustment proposal above, which is a narrow, scoped follow-up (not this read-only task).

---

## 3. polylogue-38x — Reconcile archived audit residue against current source

### Method
Read all 3 seed archives in full (`construct-validity-audit-2026-06-28.md`,
`012-fanout-findings.md`, `insights-dissection-2026-06-28.md`). For each seed finding: located
current source, checked git history for the fixing commit where claimed fixed, checked for
regression-test coverage, and cross-checked `bd` for an owning bead.

### Codex-correctness cluster (combined pass with ivsc, per instructions — no duplication)

| Finding | Classification | Citation |
|---|---|---|
| Codex FORK vs RESUME conflation | **FIXED** | `sources/parsers/codex.py:816-843` no longer assigns `BranchType.FORK` for a bare `forked_from_id` — it now leaves `branch_type=None` (generic `BRANCH` topology link) with an explicit comment ("Leave the type unclassified... rather than fabricate FORK from absent evidence"). Test: `tests/unit/sources/test_parsers_codex.py::test_forked_from_id_sets_unclassified_parent` (l.131) and `::test_forked_from_id_beats_legacy_second_meta_heuristic` (l.193). **Residual**: `LinkType.RESUME` (`core/enums.py:310`) remains a declared-but-never-assigned enum value in production code (`rg RESUME` finds no assignment site) — the enum still advertises a relationship type nothing produces. Minor, worth a one-line follow-up (remove or wire) but not the conflation bug itself. |
| Multi-meta CONTINUATION as proxy | **STILL LIVE (reduced severity)** | `sources/parsers/codex.py:839-841`: `elif len(session_metas_seen) > 1: branch_type = BranchType.CONTINUATION` — the count-of-embedded-metas heuristic is unchanged and still infers a relationship type from a count, not an assertion. Severity is now lower because it is explicitly the *fallback* path only reached when `forked_from_id` is absent (comment at l.822-829 documents it as "the legacy heuristic"), but the proxy-as-truth mechanism itself was not touched. No owning bead found (`bd` search for "continuation" heuristic/proxy turned up nothing specific) — **candidate follow-up**. |
| Scalar paste detection flattening (exact vs fallback) | **FIXED** | `archive/message/paste_detection.py:133-146` `detect_paste()` docstring now explicitly disclaims scalar-fact status ("Do not promote this union boolean into a ground-truth 'a paste occurred' claim"), and the persisted column no longer uses it: `storage/sqlite/archive_tiers/write.py:4159-4166` `_has_paste()` derives `messages.has_paste` from `message.paste_spans` (marker-derived), and `_paste_boundary()` separately persists the exact-vs-`whole_message_fallback` distinction per span. The flattening the audit flagged (union boolean as stored fact) is gone. |
| Timestamp fallback to epoch-zero | **STILL LIVE, unchanged** | `insights/transforms.py:112-118` `_session_transform_timestamp()`: `if timestamp is None: return "1970-01-01T00:00:00+00:00"` — identical to the audit's citation, no change. **Candidate follow-up bead** (none found owning this specifically — `polylogue-z29t`/`polylogue-2seq`/`polylogue-s5mm` fixed *other* epoch-fallback sites in CLI query ordering/work-event windowing/search ranking, but not this insights-transform timestamp construct — this exact site is not yet covered). |
| Codex token-lane-normalizer divergence | **PARTIALLY FIXED — split finding** | The **dominant session-level cost path is fixed**: `storage/sqlite/archive_tiers/write.py:2731-2740` (doc'd "Map Codex token_count totals onto disjoint billing lanes") subtracts cached from raw input before storing `session_model_usage`, landed in `3938bc6c2` ("fix(cost): stop double-billing Codex cached input and re-counted reasoning", verified 7.69x -> 1.08x on the real archive) with regression tests (`tests/unit/storage/test_archive_tiers_write.py`, `tests/unit/storage/test_provider_usage_report.py`). **But the parser-level per-message fields are NOT fixed**: `sources/parsers/codex.py:184-197` `_token_usage()` still maps raw provider `input_tokens` straight to `ParsedMessage.input_tokens` (inclusive of cached, per Codex's own semantics) without subtracting `cache_read_tokens`, and this message-level pair is consumed additively by `archive/semantic/pricing.py:590-621` `estimate_message_cost()` (input lane + cache-read lane both billed, `_cost_components` l.420/430) whenever `estimate_session_cost()`'s preferred session-level-exact path is unavailable (`archive/semantic/pricing.py:643-736` `estimate_session_cost()` falls back to `message_estimates` when `_session_level_estimate()` is not `status="exact"`). This reproduces the original 7.69x-class bug on whatever slice of sessions/messages falls through to the per-message fallback path. **Candidate follow-up bead**: fix `_token_usage()` (codex.py:184) to subtract cached from input at parse time (matching `_codex_token_usage_payload`'s already-correct `uncached_input_tokens` field, which exists but is unused by the message-parsing call site at l.791), closing the fallback-path gap. |

### Non-Codex fanout findings — confirm-and-cite (per instructions, minimal re-investigation)

| Finding | Classification | Citation |
|---|---|---|
| Transcript pagination/batch/stream reads bypass prefix composition | **STILL LIVE, tracked** | Owning bead: `polylogue-20d.5` (open) — "Finish streaming reads: composed transcripts, messages --full writer, origin-filtered pagination SQL"; description confirms "lineage-composed transcript streaming falls back to the eager path" — matches the fanout audit's Lineage-4 finding exactly. No further investigation needed; already an executable bead. |
| Child usage rollups counting inherited prefix | **FIXED** | `storage/sqlite/archive_tiers/write.py:497-516`: `provider_usage_baseline = _provider_usage_cumulative_baseline(conn, parent_session_id, branch_point_message_id) if ... lineage_inheritance == "prefix-sharing" else None`, passed into `_write_session_events()` which subtracts the parent's cumulative totals at the branch point (l.2635-2640) before storing the child's own usage. Commit `4628cd30f` ("fix(storage): slice provider usage on prefix-sharing forks"), test coverage in `tests/unit/storage/test_lineage_normalization.py`. Directly resolves fanout audit's "Lineage-5". |
| MCP scoped aggregates capped by page limit | **FIXED** | `polylogue-1vv` closed (gh-2473) — close reason: "Fixed MCP aggregate tools to use complete scopes for truth-bearing totals, expose truncation metadata for explicit pages, and pin the full registered tool set in tests." Matches fanout's MCP-1/2 exactly; already closed as an executable bead, no residue. |
| ChatGPT image/asset-only nodes dropped | **FIXED** | `polylogue-qda` closed (gh-2474) — close reason: "ChatGPT image/asset-only nodes now survive because structured blocks are built before the empty-text skip." Matches fanout's Parser-1. |
| Antigravity non-UTF-8 drop | **FIXED** | Same bead `polylogue-qda` — close reason covers both fixes ("Antigravity brain metadata artifacts with non-UTF-8 adjacent bodies fall back to the metadata summary instead of dropping the session"). Matches fanout's Parser-2. |
| Dead `phase_type`/confidence fields | **SPLIT: phase_type FIXED, confidence STILL LIVE** | `phase_type`: `rg phase_type polylogue/` (excl. tests) returns only a JS fallback in `daemon/web_shell.py:1734` (`inf.phase_type \|\| ...`) reading a field that no longer exists in the DDL — the `session_phases` CREATE TABLE (`storage/sqlite/archive_tiers/index.py:882-899`) has no `phase_type` column at all; it was removed. `confidence`: **still a dead always-0.0 field** — `archive/phase/extraction.py:28-41` `SessionPhase.confidence: float = 0.0` default, and `_build_phase()` (l.47-72) never sets it explicitly, so every materialized phase carries `confidence=0.0` verbatim, matching the original audit finding exactly. No owning bead found. |
| Heuristic confidence/provenance flattening in insights dissection | **STILL LIVE** | `insights/registry.py:380` still has the operator-facing `--tier` flag (evidence-vs-inference separation exists as an opt-in, matching the dissection doc's own note) but the *default* rendering path still flattens evidence and inference tiers together; no code change found narrowing this since 2026-06-28. No owning bead found — lowest-priority residual of the three "no owning bead" items above; candidate for the same follow-up umbrella as the work-event confidence-literal cleanup already flagged in the dissection doc itself. |

### Summary table for closure

**Confirmed fixed / stale (safe to mark resolved in `bd` by the operator)**:
1. Codex FORK vs RESUME conflation — `sources/parsers/codex.py:816-843` + tests at `test_parsers_codex.py:131,193`.
2. Scalar paste detection flattening — `paste_detection.py:133-146` + `write.py:4159-4166`.
3. MCP scoped aggregates capped by page limit — `polylogue-1vv` (already closed).
4. ChatGPT image/asset-only nodes dropped — `polylogue-qda` (already closed).
5. Antigravity non-UTF-8 drop — `polylogue-qda` (already closed, same bead as #4).
6. Child usage rollups counting inherited prefix — `write.py:497-516` + commit `4628cd30f` + test.
7. `session_phases.phase_type` dead column — removed from DDL entirely.
8. Codex token-lane divergence, **session-level lane** (the historically-cited 7.69x class) — commit `3938bc6c2`.

**Still live (need a new/linked bead)**:
- Multi-meta CONTINUATION-as-proxy heuristic (`codex.py:839-841`).
- Timestamp fallback to epoch-zero in `insights/transforms.py:118` (distinct from the 3 already-fixed epoch-fallback beads elsewhere).
- Codex token-lane divergence, **per-message fallback path** (`codex.py:184-197` + `pricing.py:590-621`) — narrower residual of #8 above.
- `session_phases.confidence` always-0.0 dead field.
- Heuristic confidence/provenance flattening on the default insights render path.
- Transcript pagination/streaming bypassing composition — **already tracked**, no new bead needed (`polylogue-20d.5` open).

**Provider->Origin (9e5.8)** stays open per its own scope; this note supplies the census + sequenced plan it asked for.
