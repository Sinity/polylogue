# Polylogue urgent/correctness/critical-path static prework v2

This file concatenates all packet summaries. Use the individual files in `task_packets/` for handoff.

## 001. polylogue-s7ae.6 — Classify the 74%-aborted full verify from the coordination commit before deploy

**P1 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Release-gate debt rather than one code defect: a full `devtools verify` stopped at 74%, so coordination deployment has unknown blast radius until every failing lane is classified.

Implementation spine:
- Run the exact full verification lane on the current commit and save raw logs under `.agent/reports/` or `docs/audits/`.
- Build a failure-classification table: lane, command, failure signature, first bad commit if known, coordination-caused/pre-existing/flaky, owner bead, fix/defer decision.
- For any coordination-caused failure, land a focused fix before deploying MCP/hook/coordination surfaces.
- For pre-existing failures, create/update a bead and make the deployment verdict explicitly conditional.

Tests:
- The proof is the full verify log plus classification ledger; no code unit test is enough for this packet.

Packet: `task_packets/001_polylogue_s7ae_6.md`

## 002. polylogue-37t.15 — Single agent-write chokepoint in upsert_assertion: non-user authors => CANDIDATE + inject:false, always

**P1 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: `upsert_assertion` defaults missing status to ACTIVE and `upsert_blackboard_note` does not override it, so non-user blackboard/MCP writes can become trusted active assertions by omission.

Implementation spine:
- Patch `upsert_assertion` as the only safety chokepoint. Normalize `author_kind`; only exact `user` may default to active/injected.
- For any non-user author, coerce missing or active status to candidate and set context/injection flags false unless a reviewed user action promotes it.
- Fetch existing assertion status before upsert and forbid agent writes from resurrecting rejected/deleted/superseded terminal states.
- Make all blackboard/MCP/API call paths rely on this same function; remove caller-specific safety guesses.

Tests:
- Unit test agent blackboard post without status: row is candidate, inject false, provenance preserved.
- Unit test user write without status: keeps existing user-default active semantics if intended.
- Regression: rejected candidate cannot be overwritten by an agent into active.
- MCP/API write tests prove non-user authors are demoted even when they request active.

Packet: `task_packets/002_polylogue_37t_15.md`

## 003. polylogue-kwsb.1 — Daemon/capture security hardening: Host/Origin gate, receiver token, spool governor

**P1 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Daemon GET routes dispatch before a Host gate; Origin checks are POST-only and absent-Origin tolerant; browser-capture auth is optional on loopback; query-string tokens are broadly accepted; spool has only request-size bound.

Implementation spine:
- Add a single request-admission function called before GET/POST/DELETE dispatch. Strip port/brackets, allow loopback/configured hosts, reject foreign/absent malformed Host.
- Make Origin/Referer policy explicit for browser-facing state-changing routes; absent Origin is not automatic trust when a browser route can be hit.
- Generate or require a 0600 receiver token for browser capture; compare via `hmac.compare_digest`.
- Restrict `?access_token=` to routes that genuinely need EventSource/SSE compatibility; prefer Authorization header everywhere else.
- Add a spool governor: max queued files, max bytes, max age, and loud degraded status when full.

Tests:
- Negative GET with foreign Host cannot read archive.
- Negative POST with forged Origin/token fails.
- Legitimate shell/bootstrap and extension routes still work.
- Spool quota test proves oversized backlog returns 429/degraded without writing unbounded files.

Packet: `task_packets/003_polylogue_kwsb_1.md`

## 004. polylogue-8jg9.4 — ops doctor cleanup_orphans can delete an in-flight leased blob (the real #818)

**P1 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: The safe `run_blob_gc` path already consults leases/refs/generations, but `BlobStore.cleanup_orphans` deletes hashes supplied by a simpler disk-vs-ID detector without consulting active leases or blob refs.

Implementation spine:
- Convert destructive doctor/orphan cleanup to call the lease-aware GC planner, or make `cleanup_orphans` preview-only unless passed a verified GC plan token/object.
- If keeping `cleanup_orphans`, add lease/ref/generation-age checks inside it so no caller can bypass safety.
- Rename direct disk orphan detection to `preview_orphan_candidates` or similar to remove false safety aura.
- Update ops-doctor output to show protected-by-lease/protected-by-ref/protected-by-generation counts.

Tests:
- Fixture: write a staged blob, acquire operation lease, run ops-doctor cleanup; file survives.
- Fixture: unreferenced old blob with no lease is deleted only when dry_run false and GC generation gate passes.
- Race fixture: cleanup plan computed before lease; lease acquired before delete; delete is skipped.

Packet: `task_packets/004_polylogue_8jg9_4.md`

## 005. polylogue-8jg9.2 — Blob-GC lease/orphan concurrency test (the acquire->commit race)

**P2 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Design direction: internals.md documents the load-bearing lease model (pending_blob_refs + gc_generations bridging the acquire-blob -> write-DB-row commit window) but test-closure-matrix.yaml:179 admits it is not exercised by a dedicated test. #818 has real orphan-detection bugs. Add a test that runs run_blob_gc concurrently with a mid-flight write holding a lease and asserts the leased blob is never reclaimed.

Implementation spine:
- internals.md documents the load-bearing lease model (pending_blob_refs + gc_generations bridging the acquire-blob -> write-DB-row commit window) but test-closure-matrix.yaml:179 admits it is not exercised by a dedicated test.
- #818 has real orphan-detection bugs.
- Add a test that runs run_blob_gc concurrently with a mid-flight write holding a lease and asserts the leased blob is never reclaimed.

Tests:
- Acceptance proof: A test acquires a lease, starts GC, and asserts the leased blob survives
- Acceptance proof: a released-lease orphan is reclaimed
- Acceptance proof: sweep_orphaned_blob_leases clears a SIGKILLed writer's lease past ORPHAN_LEASE_MAX_AGE_S.
- Acceptance proof: Verify: the new pytest under tests/unit/storage.

Packet: `task_packets/005_polylogue_8jg9_2.md`

## 006. polylogue-83u.4 — Classify the 39,586 missing referenced blobs in the production backup

**P1 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Production backup contains many missing blob references. The critical distinction is source-tier lost bytes vs index-tier unfetched attachments vs intentionally omitted/private material.

Implementation spine:
- Build a classifier that enumerates every blob reference by table, column/ref type, origin, hash, source material, and acquisition policy.
- Group into: present, restorable from source, never-acquired metadata-only, intentionally omitted, private/redacted, irrecoverable.
- Restore direct-file-recoverable blobs with SHA-256 verification.
- Write a durable debt report and block public backup/attachment claims until classified.

Tests:
- Synthetic DB with present/missing/restorable/metadata-only refs classifies all buckets.
- Restore path refuses hash mismatch.
- Debt report totals equal raw referenced-hash count; no silent remainder.

Packet: `task_packets/006_polylogue_83u_4.md`

## 007. polylogue-9e5.28 — Rigor audit iterates contracts, not the registry: uncovered number-bearing products vanish from audit

**P1 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: The rigor audit iterates only `list_rigor_contracts()`, while the real product universe is `INSIGHT_REGISTRY`; number-bearing products without registered contracts vanish from the audit.

Implementation spine:
- Make `INSIGHT_REGISTRY` the outer loop. For each product, join optional rigor contract, exemptions, and last materialization state.
- Emit `uncovered` rows for products without a contract, with severity based on whether the product exposes numeric/user-visible claims.
- Add an explicit exemption registry with owner, reason, expiry/review date.
- Update CLI/docs to show covered/uncovered/exempt totals and fail the strict gate on uncovered number-bearing products.

Tests:
- Register a fake number-bearing insight with no contract; audit emits uncovered and strict mode fails.
- Register a fake non-claim product with exemption; audit shows exempt, does not fail until expiry.
- Existing five contracted products remain covered.

Packet: `task_packets/007_polylogue_9e5_28.md`

## 008. polylogue-9e5.29 — Number-over-empty gates: quantitative fields need field-level evidence contracts

**P1 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Aggregate/report paths can render missing backing rows as zero. In this system every numeric field is a claim, so absence, not-applicable, uncovered, and true zero need distinct payload states.

Implementation spine:
- Introduce a small `EvidenceNumber`/field-contract shape or equivalent metadata: value, unit, denominator, evidence_state, evidence_tier, provenance, nullable_when_ungrounded.
- Apply to public insight products first: coverage, usage/cost, action outcomes, archive debt, rollups.
- Change renderers to show unknown/uncovered/not-applicable explicitly; do not coerce null to `0` except where field contract proves true zero.
- Add a registry audit that identifies numeric fields without evidence contracts.

Tests:
- Empty backing table report renders unknown/uncovered, not 0.
- Covered empty sample renders true zero with denominator and evidence_state=covered_zero.
- CLI/web/MCP JSON preserve the distinction.

Packet: `task_packets/008_polylogue_9e5_29.md`

## 009. polylogue-9e5.30 — Prose-mined forensic fields must carry text_derived provenance in the payload model

**P1 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Forensic transforms mine SHAs, decisions, caveats, and counts from prose. These are useful hints but not the same as structured evidence; payloads need text-derived provenance.

Implementation spine:
- Find prose-mined fields in tool summaries, decision candidates, forensic index entries, and report builders.
- Add `evidence_class`/`text_derived_fields` metadata that names which fields were mined from prose.
- Render caveats wherever those fields appear; block promotion to machine-trusted finding unless concrete evidence refs are attached.
- Update export/report schemas so downstream consumers cannot mistake mined text for observed fact.

Tests:
- A prose-only SHA/decision fixture renders with text-derived caveat.
- A structured evidence-ref fixture renders as observed/grounded.
- Finding promotion rejects text-derived-only claims unless policy explicitly allows candidate-only.

Packet: `task_packets/009_polylogue_9e5_30.md`

## 010. polylogue-cpf.5 — Temporal provenance laundering: aggregates collapse to provider_ts; propagate the weakest source

**P1 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: `classify_aggregate_hwm_source` currently returns provider_ts for any non-empty update list, laundering weak timestamps into strong provenance.

Implementation spine:
- Define a TemporalSource lattice: provider_ts > capture_ts > file_mtime > fallback/synthetic > unknown, with explicit ordering.
- Carry source provenance beside every high-water-mark value; aggregates choose the weakest source among contributors, not the strongest.
- Audit callers in archive summaries/rollups/profile sources and update payload/renderers to expose the source.
- Add migration/default handling for older rows with unknown source.

Tests:
- Aggregate(provider_ts, file_mtime) => file_mtime/weakest, not provider_ts.
- All-provider aggregate remains provider_ts.
- Fallback-only aggregate is synthetic/fallback and rendered with caveat.

Packet: `task_packets/010_polylogue_cpf_5.md`

## 011. polylogue-cpf.6 — Temporal correctness: clock seam for relative-date parsing + targeted sort_key_ms audit

**P1 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: `parse_date` uses ambient `datetime.now()` as RELATIVE_BASE, making relative query semantics nondeterministic and hard to test. `sort_key_ms` fallback-to-zero paths need audit for synthetic time leakage.

Implementation spine:
- Add an injected clock parameter or context object to `parse_date`, query parser lowering, and all surfaces that accept relative dates.
- Thread the frozen clock through CLI/daemon/MCP tests; default only at the outer boundary.
- Inventory `COALESCE(sort_key_ms, 0)` and similar ordering fallbacks; replace silent epoch ordering with explicit synthetic-time fields or loud caveats.
- Add operator audit output listing timeless/synthetic sessions and where they affect ordering.

Tests:
- Frozen clock: `since:7d` lowers to identical absolute bound in CLI, daemon, MCP.
- Changing wall clock during test does not change parsed result.
- Timeless rows do not masquerade as 1970/provider time; render confidence as synthetic/unknown.

Packet: `task_packets/011_polylogue_cpf_6.md`

## 012. polylogue-f2qv.2 — Codex disjoint-lane normalizer: decompose cached/uncached and reasoning/completion with a regression guard

**P2 / task / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Mechanism from bead: Codex `input` includes cached tokens and `output` includes reasoning tokens; naive input+output inflated cost by 7.69x in a prior fix. Static source anchor: `polylogue/sources/parsers/codex.py` has token_count/cached/reasoning logic around `:161-193` and `:278+`, but the invariant is not locked.

Implementation spine:
- Implementation shape:
- 1. Create/centralize a provider usage normalizer returning disjoint lanes: `input_uncached`, `cache_read`, `cache_write`, `output_completion`, `output_reasoning`.
- 2. For Codex: derive uncached input as reported input minus cached input, reasoning as a separate output sublane, completion output as reported output minus reasoning. Clamp/report inconsistencies loudly; do not silently negative-clamp without diagnostics.
- 3. For Claude: map cache creation/read lanes and output/reasoning fields into the same disjoint schema.
- 4. Add a helper that asserts lane sum equals provider-reported total where the provider reports a total.

Tests:
- synthetic Codex payload where cached is 96% of input: disjoint lanes sum to reported total; naive input+output would fail the regression guard.
- synthetic output with reasoning: completion+reasoning equals reported output.
- Claude cache_creation/cache_read payload maps to cache_write/cache_read.
- malformed payload with inconsistent totals is reported/classified, not silently accepted.
- optional live scratch cross-check against `state_5.sqlite` if available.

Packet: `task_packets/012_polylogue_f2qv_2.md`

## 013. polylogue-f2qv.1 — Per-model token rollup double-count: session totals partitioned once (#2472)

**P2 / bug / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Mechanism from bead: a session’s token totals are attributed under each model row it touched. Correct behavior: per-model totals are sums of provider usage events whose model equals that model; sum(per-model) == session total. Static anchors: `session_model_usage` materialization in `polylogue/storage/sqlite/archive_tiers/write.py:618`, table/rollup logic around `:2696+`, `:2904+`, `:2953+`.

Implementation spine:
- Implementation shape:
- 1. Locate the builder that groups `session_provider_usage_events` into `session_model_usage`.
- 2. Change grouping to `GROUP BY session_id, model` over event rows; each event contributes only to its own model.
- 3. Preserve session-grain totals separately if needed; do not copy them into every model row.
- 4. Add a rollup invariant helper used by tests: for each session, sum(model rows lanes) equals sum(provider event lanes) within integer exactness/tolerance.

Tests:
- one session with two model events: model A gets only A event tokens, model B gets only B event tokens, sum equals session total.
- mixed cache/reasoning lanes partition independently.
- regression named with GH #2472 or bead id.
- existing single-model session behavior unchanged.

Packet: `task_packets/013_polylogue_f2qv_1.md`

## 014. polylogue-f2qv.4 — Single pricing source of truth: LiteLLM catalog, drop tokencost, last-path-segment match

**P2 / task / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Mechanism from bead: a LiteLLM catalog exists, but tokencost/hardcoded maps may remain. Model ids can be vendor-prefixed, so resolver must match the last path segment rather than failing or resolving a parent path.

Implementation spine:
- Implementation shape:
- 1. `rg 'tokencost|PRICE|pricing|cost_per'` across `polylogue`, `scripts`, `pyproject.toml`.
- 2. Remove `tokencost` dependency/imports.
- 3. Create exactly one resolver module/function. It should normalize model ids by exact id first, then last path segment, then aliases if declared.
- 4. Resolver result should be a structured object: rate fields + source/catalog version + `unknown` reason.

Tests:
- `vendor/family/model-name` resolves via last segment when catalog contains `model-name`.
- observed live-archive model ids either resolve or become labelled unknown.
- no `tokencost` import/dependency remains.
- a fake second map in a fixture/test helper is caught by the no-second-price-table test.

Packet: `task_packets/014_polylogue_f2qv_4.md`

## 015. polylogue-f2qv.3 — Dual cost view: API-list-equivalent and subscription-credit reported separately

**P2 / feature / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Mechanism from bead: cache reads dominate token volume, but subscription plans may meter cache reads differently or not at all. Prior memory also records a 5x output credit-rate bug. Correct reports need two accounting regimes, not one blended number.

Implementation spine:
- Implementation shape:
- 1. Add fields such as `api_equivalent_usd`, `api_price_source`, `subscription_credit_estimate`, `subscription_credit_source`, and `cost_view_caveat` to session/day/origin cost surfaces.
- 2. API-equivalent view uses the LiteLLM resolver from f2qv.4 and disjoint token lanes from f2qv.2.
- 3. Subscription-credit view zeroes/free-prices lanes according to declared plan assumptions; cache-read treatment must be explicit.
- 4. Correct the output credit rate bug and lock it with a fixture.

Tests:
- cache-heavy session has `subscription_credit_estimate < api_equivalent_usd` under Claude Max assumptions.
- output credit-rate fixture catches the old 5x bug.
- unknown model leaves API field unknown/partial rather than false zero.
- JSON schemas/docs show both views.

Packet: `task_packets/015_polylogue_f2qv_3.md`

## 016. polylogue-f2qv.5 — Version-gate provider-usage projection so it self-heals like session_profiles

**P2 / bug / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Root cause from bead and source anchors: provider usage rows are written during ingest (`polylogue/storage/sqlite/archive_tiers/write.py:618`) but are not in the session insight rebuild/convergence path. The status gate covers profiles/logical/work/threads, not provider usage. Archive debt can only recommend manual rebuild (`polylogue/operations/archive_debt.py:854+`, zero-token rows `:899+`).

Implementation spine:
- Implementation shape:
- 1. Add a materializer version for provider-usage/session-model-usage projection.
- 2. Store that version in an existing insight materialization table or a new lightweight table keyed by session id.
- 3. Extend session-insight status/staleness detection to include provider usage.
- 4. Extend the periodic convergence loop to refresh stale provider usage rows for sessions whose source blocks carry usage.

Tests:
- old-version `session_model_usage` row becomes stale.
- daemon/convergence drain refreshes it without `maintenance rebuild-index`.
- zero-token debt over sessions with source usage drains to zero after convergence.
- archive_debt message changes from manual-only to convergence-drainable.

Packet: `task_packets/016_polylogue_f2qv_5.md`

## 017. polylogue-5hf — Provider token accounting: honest cross-provider usage ledger

**P2 / feature / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Coverage, caveats, cached-vs-uncached splits, reasoning tokens, current-window + cumulative session usage. Companions: lineage-tokens (double-count), cost reconciliation probe. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: SCOPE. The honest cross-provider usage ledger surface: given a session, logical session, day, or origin, return coverage, caveats, cached-vs-uncached input split, reasoning-vs-completion output split, and both current-window and cumulative token totals. This is the READ surface that consumes the corrected lane/pricing substrate from the sibling children; it is not the place to fix the underlying decomposition (that …

Implementation spine:
- SCOPE.
- The honest cross-provider usage ledger surface: given a session, logical session, day, or origin, return coverage, caveats, cached-vs-uncached input split, reasoning-vs-completion output split, and both current-window and cumulative token totals.
- This is the READ surface that consumes the corrected lane/pricing substrate from the sibling children
- it is not the place to fix the underlying decomposition (that is the disjoint-lane child) or pricing (the LiteLLM child).
- FILES.

Tests:
- Acceptance proof: Given a session/day/origin, the ledger returns per-lane token totals (cached/uncached input, reasoning/completion output), a coverage class and caveat set drawn from the documented vocabulary, and both API-equivalent and subscription-credit cost figures.
- Acceptance proof: Text-only-estimate and unsupported-origin rows are labelled, never silently zeroed.
- Acceptance proof: A test asserts the ledger consumes decomposed lanes (no raw input+output sum) and that logical-grain totals do not re-count inherited-prefix tokens.
- Acceptance proof: Verify on the live archive: analyze usage over codex-session and claude-session emit labelled lanes and dual cost views without the 7.69x-class inflation.

Packet: `task_packets/017_polylogue_5hf.md`

## 018. polylogue-ivsc — Classify Codex state_5 token drift outside lineage replay

**P2 / task / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: After logical-session high-water token accounting, the live Codex reconciliation probe still shows 78 logical outside-tolerance threads. New residual classification shows 62/78 have zero replay gap and all sampled residuals come from external state_5.sqlite thread rows with archived=0 and has_user_event=0, while archive sessions contain real user/assistant messages. This is no longer the fork/resume replay double-count class; classify whether state_5 tokens_used is stale, sentinel/default, or a different accountin… Design direction: Use /realm/tmp/polylogue-cost-reconciliation/codex-logical-probe-current-max100.json as the seed artifact. Compare sampled thread rows against provider token_count events, session_model_usage, Codex rollout paths where available, and any current Codex state schema docs/source. Produce a bounded classifier in the probe rather than making the whole check fail as undifferentiated token drift. Keep logical-session repla…

Implementation spine:
- Use /realm/tmp/polylogue-cost-reconciliation/codex-logical-probe-current-max100.json as the seed artifact.
- Compare sampled thread rows against provider token_count events, session_model_usage, Codex rollout paths where available, and any current Codex state schema docs/source.
- Produce a bounded classifier in the probe rather than making the whole check fail as undifferentiated token drift.
- Keep logical-session replay-gap diagnostics separate from external-state drift.

Tests:
- Acceptance proof: The Codex reconciliation report distinguishes lineage replay residuals from external-state/accounting-grain drift
- Acceptance proof: live active archive artifact explains the remaining outside-tolerance rows without implying replay double-counting
- Acceptance proof: any adjusted pass/fail status is backed by tests and live evidence.

Packet: `task_packets/018_polylogue_ivsc.md`

## 019. polylogue-xy95 — Speed up provider usage full stale diagnostics

**P2 / bug / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: During polylogue-4ts.2, polylogue analyze usage --origin codex-session --detail full --limit 20 --format json entered D-state and had to be terminated. A targeted SQL audit over the same archive completed in about 30s and showed Codex stale rollups were actually clean after the reasoning-only predicate fix. The full report path likely does avoidable broad Python reconstruction/source sampling work and is too slow for routine devloop evidence. Design direction: Profile provider_usage_report_from_connection(detail='full', origin='codex-session') by stage. Replace the stale-rollup path with bounded SQL/window aggregates or add planner-supporting indexes if needed. Keep raw/source debt and sample collection separate so stale-rollup diagnostics can be requested cheaply. Add a regression/perf smoke that prevents full detail from silently doing unbounded row materialization on l…

Implementation spine:
- Profile provider_usage_report_from_connection(detail='full', origin='codex-session') by stage.
- Replace the stale-rollup path with bounded SQL/window aggregates or add planner-supporting indexes if needed.
- Keep raw/source debt and sample collection separate so stale-rollup diagnostics can be requested cheaply.
- Add a regression/perf smoke that prevents full detail from silently doing unbounded row materialization on large archives.

Tests:
- Acceptance proof: On the active archive, the Codex full usage diagnostic either completes within an agreed interactive budget or exposes separately selectable expensive sections
- Acceptance proof: no D-state wait in the normal stale-rollup path
- Acceptance proof: tests cover reasoning-only rows and the optimized stale-rollup result.

Packet: `task_packets/019_polylogue_xy95.md`

## 020. polylogue-20d.4 — CLI structured-query routing parity with daemon (#1860): no FTS gate for non-FTS queries

**P2 / bug / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Some CLI structured queries still route through FTS readiness gates even when the query can be satisfied structurally. Daemon paths already use `SessionQuerySpec.from_params` in places, creating parity pressure.

Implementation spine:
- Classify query params into structural-only vs FTS-required before readiness checks.
- Route CLI query construction through the same `SessionQuerySpec.from_params` path as daemon/API.
- Move FTS readiness failures into only the branches that need MATCH/text search.
- Add a parity fixture comparing CLI and daemon envelopes for structured-only filters.

Tests:
- `origin=codex status=...` style query works with missing/broken messages_fts.
- A text MATCH query still fails loudly when FTS is unavailable.
- CLI/daemon/MCP row counts match for structural-only queries.

Packet: `task_packets/020_polylogue_20d_4.md`

## 021. polylogue-1xc.12 — FTS drift gauges + metamorphic coherence tests; rowid-reuse requires block_id check

**P2 / bug / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: FTS coherence currently has multiple DDL/repair/readiness paths and risks rowid reuse drift unless checks prove rowid maps to the expected block_id/content.

Implementation spine:
- Add drift gauges that compare blocks↔messages_fts_docsize plus a block_id/content-hash sentinel, not only row counts.
- Create metamorphic tests: insert/update/delete/reuse rowid patterns, then ensure FTS and structural reads converge.
- Unify trigger DDL declarations or generate repair DDL from one source.
- Expose drift counts in daemon status with repair command hints.

Tests:
- Rowid reuse fixture detects stale FTS row with wrong block_id.
- Trigger deletion/recreation is idempotent.
- Global drift repair updates only needed windows or states why full repair is required.

Packet: `task_packets/021_polylogue_1xc_12.md`

## 022. polylogue-a7xr.1 — Sweep remaining sqlite3 connection leaks: 'with sqlite3.connect()' commits but never closes

**P2 / bug / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Design direction: Python's sqlite3 Connection context manager commits/rolls back the TRANSACTION on __exit__ but does NOT close the connection — a well-known trap. insights/otlp_correlation.py:116 already documents it and uses contextlib.closing; but ~9 other sites still leak: coordination/envelope.py:591 (_sqlite_user_version, leaks 3 conns PER envelope build — agent-polled hot path), api/user_state_resolver.py:59/67/91 (per user-st…

Implementation spine:
- Python's sqlite3 Connection context manager commits/rolls back the TRANSACTION on __exit__ but does NOT close the connection — a well-known trap.
- insights/otlp_correlation.py:116 already documents it and uses contextlib.closing
- but ~9 other sites still leak: coordination/envelope.py:591 (_sqlite_user_version, leaks 3 conns PER envelope build — agent-polled hot path), api/user_state_resolver.py:59/67/91 (per user-state read), api/archive.py:2931/4626, archive/raw_payload/decode.py:309, storage/repair.py:112, demo/seed.py:82.
- Connections leak until GC -> ResourceWarnings, fd pressure under sustained load.
- FIX: wrap each in contextlib.closing(sqlite3.connect(...)) (or try/finally: conn.close()), matching the otlp_correlation.py fix and the try/finally pattern already used by _archive_evidence_payloads right below the leaking _sqlite_user_version.

Tests:
- Acceptance proof: Every 'with sqlite3.connect(...)' in non-test polylogue/ either closes via contextlib.closing/try-finally or is justified
- Acceptance proof: a ResourceWarning-as-error test run over the coordination-envelope and user_state_resolver hot paths shows no leaked connections.
- Acceptance proof: Verify: rg 'with sqlite3.connect' polylogue/ --type py -g '!*test*' returns only closing()-wrapped forms
- Acceptance proof: pytest -W error::ResourceWarning on the touched paths passes.

Packet: `task_packets/022_polylogue_a7xr_1.md`

## 023. polylogue-a7xr.2 — Converger and repair disagree on session_profile staleness for NULL-sort-key sessions

**P2 / bug / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: VERIFIED LIVE 2026-07-06 (divergence audit): daemon/convergence_stages.py:829-836 and storage/repair.py:566-584 encode DIFFERENT staleness predicates for the same derived rows. For sessions with sort_key_ms IS NULL the converger compares strftime of source_updated_at vs updated_at_ms/1000 as strings, while repair COALESCEs the NULL to 0.0 and applies the 1e-6 epsilon against source_sort_key — a NULL-sort-key session with non-zero source_sort_key is permanently stale to repair and possibly fresh to the converger. C… Design direction: One session_profile_stale_predicate(sessions_alias, profile_alias) -> str SQL-fragment builder in storage/insights/session/runtime.py (next to SESSION_INSIGHT_MATERIALIZATION_TYPES); both convergence_stages.py and repair.py compose their queries from it; repair's UNION arms for session_latency_profiles reuse the same fragment with the lp alias. Materializer version comes from one accessor. Decide the NULL-sort-key s…

Implementation spine:
- One session_profile_stale_predicate(sessions_alias, profile_alias) -> str SQL-fragment builder in storage/insights/session/runtime.py (next to SESSION_INSIGHT_MATERIALIZATION_TYPES)
- both convergence_stages.py and repair.py compose their queries from it
- repair's UNION arms for session_latency_profiles reuse the same fragment with the lp alias.
- Materializer version comes from one accessor.
- Decide the NULL-sort-key semantics ONCE (the converger's updated_at comparison is the better-considered branch) and encode it in the fragment.

Tests:
- Acceptance proof: rg shows exactly one definition of the staleness predicate
- Acceptance proof: a fixture with sort_key_ms NULL + source_sort_key set is classified identically by a convergence pass and an ops repair pass (regression test asserting agreement)
- Acceptance proof: no repair churn on a converged archive (idempotence test: repair immediately after convergence selects zero rows).
- Acceptance proof: Verify: devtools test -k 'staleness or repair'.

Packet: `task_packets/023_polylogue_a7xr_2.md`

## 024. polylogue-a7xr.3 — message_type_backfill reconstructs prose unordered and unfiltered; message-prose SQL exists 5x

**P2 / bug / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: VERIFIED LIVE 2026-07-06: storage/message_type_backfill.py:54-64 claims (comment) to concatenate block text in position order, but its GROUP_CONCAT has no inner ORDER BY — SQLite GROUP_CONCAT is unordered, so the #839 classifier can receive scrambled prose. It also omits the block_type='text' filter (thinking/tool text leaks into classification) and uses a single-newline separator, while the embeddings/demo family (storage/embeddings/materialization.py:535/754/923, demo/seed.py:607, demo/constructs.py:240) uses do… Design direction: message_prose_sql(alias, *, separator, block_types, min_chars) fragment builder next to archive_embeddable_message_where (the factoring pattern already proven there); backfill gains ordered concatenation via correlated subquery (SELECT GROUP_CONCAT(text, sep) FROM (SELECT text FROM blocks WHERE message_id=m.message_id AND ... ORDER BY position)); all five sites compose the builder; demo/constructs.py imports it (ver…

Implementation spine:
- message_prose_sql(alias, *, separator, block_types, min_chars) fragment builder next to archive_embeddable_message_where (the factoring pattern already proven there)
- backfill gains ordered concatenation via correlated subquery (SELECT GROUP_CONCAT(text, sep) FROM (SELECT text FROM blocks WHERE message_id=m.message_id AND ...
- ORDER BY position))
- all five sites compose the builder
- demo/constructs.py imports it (verification becomes real).

Tests:
- Acceptance proof: One builder
- Acceptance proof: backfill output for a multi-block fixture is position-ordered (regression test with 3+ blocks inserted out of order)
- Acceptance proof: block_type filter applied on the classifier path
- Acceptance proof: embeddings selection output unchanged (golden).
- Acceptance proof: Verify: devtools test -k 'backfill or message_type or embeddable'.

Packet: `task_packets/024_polylogue_a7xr_3.md`

## 025. polylogue-a7xr.5 — FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle repair copies

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Same class as the closed fts_freshness_state double-declaration, three more objects: trigger DDL for messages_fts/session_work_events_fts/threads_fts lives in BOTH storage/sqlite/archive_tiers/index.py (:307-324, :729-767, :449-464) and storage/fts/fts_lifecycle.py (:198-233+ as _BLOCKS/_SESSION_WORK_EVENT/_THREAD trigger DDL constants used by drop-and-recreate repair). Byte-equivalent today; any future edit forks trigger behavior between fresh DBs and repaired DBs. No test couples the two sources. Design direction: Move trigger DDL lists to storage/fts/sql.py (already holds FTS_INDEX_EXISTS_SQL) as the single source; archive_tiers/index.py composes its DDL script from them; fts_lifecycle imports them. Derived-tier regime: pure code move, no schema bump (emitted DDL identical — assert via normalized-text comparison in the PR). Relates 1xc.12 (drift gauges family).

Implementation spine:
- Move trigger DDL lists to storage/fts/sql.py (already holds FTS_INDEX_EXISTS_SQL) as the single source
- archive_tiers/index.py composes its DDL script from them
- fts_lifecycle imports them.
- Derived-tier regime: pure code move, no schema bump (emitted DDL identical — assert via normalized-text comparison in the PR).
- Relates 1xc.12 (drift gauges family).

Tests:
- Acceptance proof: rg finds each trigger body in exactly one module
- Acceptance proof: a drift test asserts fresh-DB and repair-path trigger text are identical (normalized)
- Acceptance proof: rebuild + repair smoke green.
- Acceptance proof: Verify: devtools test -k fts.

Packet: `task_packets/025_polylogue_a7xr_5.md`

## 026. polylogue-a7xr.6 — parse_archive_datetime: 6 copies, one with different tz semantics (naive/aware time bomb)

**P2 / bug / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Divergence audit: identical _parse_archive_datetime copies in context/selection.py:285, mcp/archive_support.py:492, cli/read_views/standard.py:232, api/archive.py:514, archive/query/archive_execution.py:113 (naive stays naive; empty string raises) vs a DIVERGENT copy in storage/insights/session/rebuild.py:763 (empty->None; naive FORCED to UTC). The same stored string parses to offset-naive or offset-aware depending on surface — a latent TypeError (cannot compare naive and aware) across insight vs read paths. Also … Design direction: core/timestamps.py is the designated home (docstring: unified timestamp parsing, all operations UTC): add parse_archive_datetime() with the rebuild copy's UTC-forcing semantics (matches the module contract) + iso_from_epoch_ms(); delete all copies. Audit each call site for naive-datetime comparisons that silently relied on naive semantics (mypy + tests are the net). Part of the cpf temporal doctrine surface.

Implementation spine:
- core/timestamps.py is the designated home (docstring: unified timestamp parsing, all operations UTC): add parse_archive_datetime() with the rebuild copy's UTC-forcing semantics (matches the module contract) + iso_from_epoch_ms()
- delete all copies.
- Audit each call site for naive-datetime comparisons that silently relied on naive semantics (mypy + tests are the net).
- Part of the cpf temporal doctrine surface.

Tests:
- Acceptance proof: One definition each
- Acceptance proof: all six+five sites import core/timestamps
- Acceptance proof: a test asserts the parsed value is ALWAYS tz-aware UTC
- Acceptance proof: no naive-vs-aware comparison remains reachable (grep + focused tests).
- Acceptance proof: Verify: devtools test -k timestamp.

Packet: `task_packets/026_polylogue_a7xr_6.md`

## 027. polylogue-9e5.3 — Column honesty audit: null/unknown density for key semantic columns

**P2 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: This is a read-only audit packet. It supports 9e5.29 by finding numeric/default columns whose meaning is ambiguous, especially across insight products, usage, attachments, and lineage.

Implementation spine:
- Implementation shape:
- 1. Generate a table of public payload fields/DB columns with type, nullable, default, sample null density, sample zero density, and known evidence source.
- 2. Classify each as true-zero-safe, unknown-when-absent, not-applicable, text-derived, or needs-contract.
- 3. Emit JSON + Markdown artifact under docs/audits or `.agent/reports`.
- 4. File follow-up beads only for confirmed high-risk fields.

Tests:
- Acceptance proof: 1.
- Acceptance proof: A committed evidence artifact (CSV/JSON matrix + short markdown, under .agent/scratch/research/ or demo-shelf) reports, per (column, origin, month), the null/unknown/populated counts, populated_pct, and top-5 values, with denominators correct (tool_result_* over tool_result rows, material_origin over authored messages).
- Acceptance proof: 2.
- Acceptance proof: Each column carries a go/no-go verdict — structural-ready vs structural-with-caveat vs keep-heuristic — that the b0b heuristic->structural sweep bead consumes, plus a per-origin coverage-caveat sentence for outcome analytics.
- Acceptance proof: 3.

Packet: `task_packets/027_polylogue_9e5_3.md`

## 028. polylogue-9e5.4 — Get->modify->put race audit across daemon/CLI/MCP writers

**P2 / task / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Bead notes give the method: static sweep for SELECT-then-UPDATE/INSERT, manual upsert emulation, status transitions, shared writer APIs, and connection boundaries. Named candidates include write effects, blob GC, daemon cursors, embeddings, FTS readiness, MCP mutation handlers, and CLI ops writers.

Implementation spine:
- Implementation shape:
- 1. `rg` for read-then-write patterns and connection open boundaries.
- 2. For each candidate, record file:function, invariant, connection/transaction boundary, interleaving, expected consequence, and verdict: safe-by-single-transaction, safe-by-unique/upsert, needs-harness, or confirmed-bug.
- 3. Only create implementation bug beads for confirmed windows with concrete two-connection repro sketches.
- 4. Document refuted windows as safe with reason so agents do not re-triage them.

Tests:
- No broad product tests unless a confirmed race is found. For top 1-2 confirmed windows, add a minimal two-connection repro test in the follow-up bug, not this audit packet.

Packet: `task_packets/028_polylogue_9e5_4.md`

## 029. polylogue-9e5.19 — Storage-layer correctness scenario family

**P2 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Bead design: create scenario family covering split-tier writes, content-hash idempotency, FTS trigger integrity, blob-lease GC, and lineage composition against seeded archives.

Implementation spine:
- Implementation shape:
- 1. Add `storage-correctness` to scenario-coverage configuration, referencing this bead.
- 2. Build seeded archive fixtures for: idempotent re-ingest, split-tier write/read, FTS mutation drift, leased blob GC, lineage composition snapshot.
- 3. Wire into devtools lab/projections lanes.
- 4. The scenario should aggregate existing focused tests where possible rather than duplicate all logic.

Tests:
- Acceptance proof: A storage-correctness scenario family exists and runs via devtools lab lanes
- Acceptance proof: it covers idempotent re-ingest, FTS trigger drift, and lineage composition
- Acceptance proof: scenario-coverage.yaml references this bead, not gh#590.
- Acceptance proof: Verify: devtools lab projections + lab lanes.

Packet: `task_packets/029_polylogue_9e5_19.md`

## 030. polylogue-cpf.4 — Enforce degrade-loudly: sweep silent soft-failure paths to carry a signal

**P2 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Design direction: The cpf degraded-mode doctrine says "degrade loudly once", but a deep read (2026-07-05) found the codebase systematically degrades SILENTLY on derived/fallback/probe paths — robust (never crashes) but serves incomplete/stale data with no signal, which for a system-of-record is a construct-validity hole. Concrete instances found: convergence freshness probes fail-closed to converged with no log (1xc.11); lineage comp…

Implementation spine:
- The cpf degraded-mode doctrine says "degrade loudly once", but a deep read (2026-07-05) found the codebase systematically degrades SILENTLY on derived/fallback/probe paths — robust (never crashes) but serves incomplete/stale data with no signal, which for a system-of-record is a construct-validity hole.
- Concrete instances found: convergence freshness probes fail-closed to converged with no log (1xc.11)
- lineage composition truncates on depth>64 or dangling branch point with no completeness signal (4ts.6)
- coordination archive-evidence returns empty tuples on a 0.2s SQLite timeout (envelope.py:610/616/639) — indistinguishable from "no evidence"
- generic-messages parser drops timestamps silently (tf0e).

Tests:
- Acceptance proof: Each identified silent soft-fail path (probe fail-closed, lineage truncation, timeout-to-empty, fallback data-drop) either emits a typed degradation signal consumers can read, or logs-loudly-once
- Acceptance proof: a reader/agent can distinguish "no data" from "degraded/timed-out/truncated".
- Acceptance proof: A review-gate or lint flags new bare soft-fails.
- Acceptance proof: Verify: the instance beads (1xc.11, 4ts.6, tf0e) close against this, and a test asserts the timeout/truncation/probe-fail paths surface a reason.

Packet: `task_packets/030_polylogue_cpf_4.md`

## 031. polylogue-b0b — Replace remaining keyword outcome/pathology heuristics with structural evidence

**P2 / task / 10-analytics-experiments / blocked-hard**

Mechanism: Bead design says keyword outcome/pathology heuristics should be replaced by structural evidence. Static source likely contains regex/keyword classifiers in insight transforms/forensics/report scripts. These are useful as fallback, not as primary evidence.

Implementation spine:
- Implementation shape:
- 1. Inventory outcome/pathology fields and their current evidence source: tool result exit code, test runner structured status, command result, PR/commit state, assistant prose, keyword heuristic.
- 2. Define evidence precedence: structured tool/test status > persisted action outcome > explicit user/agent annotation > text-derived prose > keyword fallback.
- 3. Add evidence_class/source fields to outcome/pathology records.
- 4. Convert the highest-impact classifiers first: failed tool call, test failure, silent proceed, retry loop, abandonment/resume.

Tests:
- structured failed test beats optimistic prose.
- success prose does not override nonzero exit status.
- keyword-only case still produces fallback but labelled heuristic/text-derived.
- aggregate report footnotes counts by evidence tier.

Packet: `task_packets/031_polylogue_b0b.md`

## 032. polylogue-b0b.1 — Fix substring false-positives in work-event keyword classifier + inventory activity-type label as heuristic-tier

**P2 / task / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Design direction: polylogue/archive/session/extraction.py:_text_signal_from_lowered_text (241-262) matches keyword tables with naive `pattern in lowered_text`, so tokens embed into unrelated words: 'fix'->prefix/suffix, 'test'->latest/contest, 'plan'->explanation/airplane, 'data'->update/validate/metadata, 'spec'->respect/inspect, 'move'->remove, 'config'->reconfigured. This silently mislabels work-event heuristic_label and the noise…

Implementation spine:
- polylogue/archive/session/extraction.py:_text_signal_from_lowered_text (241-262) matches keyword tables with naive `pattern in lowered_text`, so tokens embed into unrelated words: 'fix'->prefix/suffix, 'test'->latest/contest, 'plan'->explanation/airplane, 'data'->update/validate/metadata, 'spec'->respect/inspect, 'move'->remove, 'config'->reconfigured.
- This silently mislabels work-event heuristic_label and the noise is invisible in the hardcoded confidence float.
- Two changes: (1) match on word boundaries (compile the pattern tables to `\b(?:...)\b` regexes or tokenize+set-membership) so substring collisions stop
- keep multiword phrases ('stack trace','should we') as phrase matches.
- (2) b0b's inventory is scoped to 'outcome/pathology heuristics' — the work-event activity-TYPE classifier (planning/debugging/testing/...) is neither, so explicitly record it in the b0b heuristic-tier inventory with a per-origin coverage caveat, since unlike outcomes there is no structural ground truth to convert it to (it stays heuristic-tier by nature).

Tests:
- Acceptance proof: 1.
- Acceptance proof: _TEXT_SIGNAL_TABLE matching uses word boundaries
- Acceptance proof: a regression test asserts 'prefix'/'latest'/'explanation'/'metadata'/'remove' do NOT trigger fix/test/plan/data/move signals while genuine 'fix the bug'/'run pytest'/'let us plan' do.
- Acceptance proof: 2.
- Acceptance proof: The work-event activity-type classifier appears in the b0b heuristic-tier inventory with an explicit 'stays heuristic (no structural ground truth)' note and a coverage caveat.

Packet: `task_packets/032_polylogue_b0b_1.md`

## 033. polylogue-4be — Restore drill: prove the backups restore, quarterly

**P2 / task / 01-blob-attachment-integrity / implementation-ready-after-local-inspection**

Mechanism: The bead states there are multiple backup layers but no restore test. The restore drill should prove latest backup set can produce an archive that passes integrity checks and basic user queries within expected lag.

Implementation spine:
- Implementation shape:
- 1. Add `polylogue ops restore-drill` or `devtools restore-drill`.
- 2. Locate latest configured backup set; restore to a scratch root, never over live archive.
- 3. Run `PRAGMA integrity_check` for each SQLite tier.
- 4. Run a 10-query battery: session count, message count, blob-reference debt, one `find`, one `read`, one insight/profile read, one attachment lookup, one usage/cost summary, one tag/user-state read, one health/status command.

Tests:
- synthetic backup restored to scratch and query battery passes.
- corrupted DB/file fails with clear failure.
- drill refuses to write into live archive path.
- count-lag tolerance behaves as declared.

Packet: `task_packets/033_polylogue_4be.md`

## 034. polylogue-peo — Daemon death leaves no trace: crash forensics + heartbeat sentinel + restart policy

**P2 / bug / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Bead mechanism: under read-only serving flags the daemon terminated twice; logs stopped without traceback; exit code 144 was ambiguous. Current running checks may rely on pid/process rather than fresh heartbeat.

Implementation spine:
- Implementation shape:
- 1. Add `polylogue/daemon/lifecycle.py` to own run id, heartbeat row, signal logging, and stack dumps.
- 2. At daemon start: enable `faulthandler`, create `ops.db daemon_lifecycle` table if absent, insert `started` row.
- 3. Periodic loop writes heartbeat timestamp each tick.
- 4. SIGTERM/SIGINT handlers log signal, active thread stack dump, heartbeat age, and lifecycle row before exiting/chain-calling.

Tests:
- direct lifecycle unit tests for start/heartbeat/clean stop rows.
- subprocess integration: start daemon fixture, send SIGTERM, assert signal row + stack log.
- stale heartbeat makes status/healthz report stale/down.
- clean shutdown distinct from vanish.

Packet: `task_packets/034_polylogue_peo.md`

## 035. polylogue-83u.3 — Preserve uploaded attachment bytes in live browser capture

**P1 / feature / 01-blob-attachment-integrity / implementation-ready-after-local-inspection**

Mechanism: Browser capture payload models can carry attachment bytes, but live capture currently treats many attachments as metadata or extracted text only. Future evidence needs bytes or honest unavailable state.

Implementation spine:
- Define acquisition policy for live browser attachments: inline bytes, fetch URL, defer with tokenized acquisition job, or mark unavailable.
- Persist acquired bytes via blob store with content hash and attachment row link.
- Do not store raw bytes for private/unsupported items without explicit policy; store classified missing/unavailable state.
- Expose attachment acquisition state in capture status and archive read payloads.

Tests:
- Live capture fixture with inline_base64 persists blob and resolves hash.
- Fixture with only provider URL marks deferred/unfetched, not missing-lost.
- Quota/security tests prove attachment bytes obey receiver token and spool governor.

Packet: `task_packets/035_polylogue_83u_3.md`

## 036. polylogue-83u.2 — Attachment byte acquisition for non-inline sources (Drive/zip/local)

**P2 / feature / 01-blob-attachment-integrity / implementation-ready-after-local-inspection**

Mechanism: Bead design identifies three live-handle boundaries: Drive downloads inside iterator scope, export-zip members while zipfile is open, and local paths guarded by a source-root allowlist. Existing parser model already supports `ParsedAttachment.inline_bytes`.

Implementation spine:
- Implementation shape:
- 1. Drive: restore/download asset bytes with `DriveSourceClient.download_bytes` inside the iterator/lifetime of the source handle.
- 2. Export ZIP: resolve attachment member and read it while the `ZipFile` is still open; attach as inline bytes.
- 3. Local path: accept a transport-only `local_source_path`, canonicalize with `realpath`, require it to stay under declared source-root allowlist, reject symlink/`..` escapes before opening.
- 4. Non-live handles remain `unfetched` with `source_url/source_path`; no synthetic hash.

Tests:
- Drive fixture downloads inside iterator and stores true blob.
- ZIP fixture stores member bytes before close.
- local allowlisted file stores true blob.
- local path escape is rejected and no read happens.
- closed/non-live handle stays unfetched.

Packet: `task_packets/036_polylogue_83u_2.md`

## 037. polylogue-83u.6 — Attachment acquisition census by origin and byte volume

**P2 / task / 01-blob-attachment-integrity / implementation-ready-after-local-inspection**

Mechanism: The bead already gives a strong design: read source.db/index.db/blob store read-only; group by origin and acquisition_status; reconcile against blob-reference-debt diagnostics. Baseline numbers in the bead: 7,226 attachment rows, 958 acquired with non-null hash and 0 missing acquired blobs, 6,268 unfetched with NULL hash.

Implementation spine:
- Implementation shape:
- 1. Add a devtools or ops diagnostic command that opens archive DBs using SQLite `mode=ro`.
- 2. Resolve blob paths through `BlobStore` without mutating.
- 3. Group rows by `(origin, acquisition_status)` and emit counts, declared byte sum, acquired blob count, bytes on disk, unfetched, unavailable, missing acquired blob refs, top source_ref classes, and bounded samples.
- 4. Write JSON + Markdown under `.agent/scratch/research/` or a demo shelf.

Tests:
- command refuses or is proven not to open write connections against a fixture/live path.
- fixture totals reconcile with blob-reference-debt primitive.
- missing acquired blob and unfetched NULL hash are separated.
- sample list is bounded.

Packet: `task_packets/037_polylogue_83u_6.md`

## 038. polylogue-ptx — Browser-capture posting channel: un-gate, with attachments

**P2 / feature / 01-blob-attachment-integrity / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Operator decision 2026-07-03: UN-GATE. Agents may drive web chats through the posting channel (agent-private Chrome profile posture per the ambient control model). Scope for this bead: bring the parked worktree branch to production quality, enable the channel, and add attachment support (files/images posted alongside text). Trajectory beyond this bead (separate beads): user-drivable posting from the webui, and harness remote-control lanes (Claude Code remote control, Codex analogue). Residual risk accepted: same-a… Design direction: The channel exists on the worktree branch (worktree-agent-aa5375b510cb4aa5d era work): extension->receiver POSTING path, previously operator-gated OFF. To ship: rebase/land the branch, flip the gate default for the agent-private profile, add attachment upload (multipart through the receiver -> provider web upload flows; store posted attachments as acquired blobs so the archive keeps what was sent), and cover with th…

Implementation spine:
- The channel exists on the worktree branch (worktree-agent-aa5375b510cb4aa5d era work): extension->receiver POSTING path, previously operator-gated OFF.
- To ship: rebase/land the branch, flip the gate default for the agent-private profile, add attachment upload (multipart through the receiver -> provider web upload flows
- store posted attachments as acquired blobs so the archive keeps what was sent), and cover with the deterministic capture smoke pattern.
- Verify the 0mu freshness fix (newest-wins) is in place first so posted-then-captured sessions do not get clobbered by DOM fallback.

Tests:
- Acceptance proof: 1.
- Acceptance proof: The parked worktree posting branch (extension->receiver POSTING path, worktree-agent-aa5375b510cb4aa5d era) is landed at production quality.
- Acceptance proof: 2.
- Acceptance proof: The posting gate default is flipped ON for the agent-private Chrome profile.
- Acceptance proof: 3.

Packet: `task_packets/038_polylogue_ptx.md`

## 039. polylogue-4p1 — Decision: one read algebra — Query x Projection x Render as the only read contract

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The read side has N surfaces multiplying independently: CLI verbs with per-view flags, analyze boolean modes, MCP tools (~61 read tools, many being named parameterizations of the same underlying read), web routes, read-view profiles, read-package layouts. jnj collapses CLI flags into ProjectionSpec; fnm.6 wires DSL terminal stages to projections; t46 deletes parallel dispatch. This bead records the target those programs converge on, so future surface work has a stated invariant instead of rediscovering the directi… Design direction: Doctrine bead, deliverable is the recorded contract + a conformance inventory, not a rewrite: (1) Write the algebra down in docs/architecture-spine.md — the three spec types, their composition law, and the rule that MCP tools / analyze modes / CLI views / web routes are NAMED PRESETS over the algebra (a preset = a (Q,P,R) triple with defaults, registered via the declare-once machinery). (2) Conformance inventory: cl…

Implementation spine:
- Doctrine bead, deliverable is the recorded contract + a conformance inventory, not a rewrite: (1) Write the algebra down in docs/architecture-spine.md — the three spec types, their composition law, and the rule that MCP tools / analyze modes / CLI views / web routes are NAMED PRESETS over the algebra (a preset = a (Q,P,R) triple with defaults, registered via the declare-once machinery).
- (2) Conformance inventory: classify every existing read surface as conformant / preset-expressible / algebra-hole (needs a new spec capability) / genuinely-other
- the holes become the priority list for fnm/jnj slices.
- Include the current projection/render duplication explicitly: ContentProjectionSpec vs ProjectionSpec body policy
- RenderFormat vs session output formats

Tests:
- Acceptance proof: docs/architecture-spine.md gains a 'One read algebra' entry under Major Decisions naming SelectionSpec x ProjectionSpec x RenderSpec (QueryProjectionSpec) as the sole read contract and presets as named (S,P,R) triples, with projection_spec.py cited as the existing realization.
- Acceptance proof: A conformance inventory (table in the doc or a linked docs/plans/*.yaml manifest checked by devtools verify manifests) lists every read surface with its classification
- Acceptance proof: every algebra-hole row links to the fnm or jnj bead that closes it.
- Acceptance proof: The expansion policy is stated as doctrine: a new read affordance (new MCP tool, analyze mode, web panel, static HTML/variant/translation view) must be expressible as a preset/spec value, and if it needs a bespoke read path the algebra gains or reuses the capability FIRST.
- Acceptance proof: The writes/ops/maintenance scope guard is recorded.

Packet: `task_packets/039_polylogue_4p1.md`

## 040. polylogue-4p1.1 — Route daemon split-archive fast path through SessionQuerySpec.from_params

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Design direction: polylogue/daemon/http.py:1970 documents that the split-archive fast path intentionally does not construct a SessionQuerySpec and hand-mirrors every public structured filter (has_paste, has_tool_use, has_thinking, repo, has_type, tool/exclude_tool, action/exclude_action/action_sequence/action_text, referenced_path, cwd_prefix, title, min/max_messages, min/max_words, plus the shared _filter_kw block). This is a parall…

Implementation spine:
- polylogue/daemon/http.py:1970 documents that the split-archive fast path intentionally does not construct a SessionQuerySpec and hand-mirrors every public structured filter (has_paste, has_tool_use, has_thinking, repo, has_type, tool/exclude_tool, action/exclude_action/action_sequence/action_text, referenced_path, cwd_prefix, title, min/max_messages, min/max_words, plus the shared _filter_kw block).
- This is a parallel implementation of build_query_spec_from_params (polylogue/archive/query/spec.py:498).
- A filter field added to the spec builder is silently absent from the daemon fast path until someone edits both sites.
- Collapse it: have the fast path build a SessionQuerySpec via from_params and read its lowered filter fields, keeping only the genuinely count/summary-specific plumbing (session_id passed separately) outside the spec.
- Prove parity with a test that enumerates SessionQuerySpec filter fields and asserts each is honored by the fast path.

Tests:
- Acceptance proof: The daemon split-archive list/search/count path derives all structured filters from a SessionQuerySpec built via from_params (no per-field re-read of HTTP params for filters the spec already models)
- Acceptance proof: a test enumerates SessionQuerySpec filter attributes and fails if the fast path drops any
- Acceptance proof: the in-code 'must mirror those public params here' comment and its manual mirroring block are removed
- Acceptance proof: render surfaces (openapi/cli-output-schemas) still verify.

Packet: `task_packets/040_polylogue_4p1_1.md`

## 041. polylogue-t46.3 — Unify list/search query-spec->ArchiveStore execution across CLI, MCP, and daemon web

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Design direction: Three surfaces re-map the query DSL/params to ArchiveStore filter args and each own pagination/total/cursor: daemon http.py:1902 _do_archive_list_sessions (+ the _do_archive_* fast-path family), mcp/archive_support.py:254-379 archive_session_list_payload/archive_search_payload, and cli/archive_query.py:674/787/815 _query_hits/_paginate_rows/_build_cursor. The http.py:1970 comment admits it 'must mirror those public …

Implementation spine:
- Three surfaces re-map the query DSL/params to ArchiveStore filter args and each own pagination/total/cursor: daemon http.py:1902 _do_archive_list_sessions (+ the _do_archive_* fast-path family), mcp/archive_support.py:254-379 archive_session_list_payload/archive_search_payload, and cli/archive_query.py:674/787/815 _query_hits/_paginate_rows/_build_cursor.
- The http.py:1970 comment admits it 'must mirror those public params here' and it re-fixed bugs #1873/#1860 in the parallel path
- MCP has two internal list surfaces with different total semantics (archive_support estimate vs server_tools.py:363 poly.archive_count_sessions).
- Fix: route every surface through SessionQuerySpec.from_params + a single archive execution helper in archive/query/archive_execution.py that returns (rows, total, cursor)
- surfaces differ only in payload projection (build_search_envelope is already shared).

Tests:
- Acceptance proof: CLI find, MCP archive_list_sessions/archive_search_sessions, and daemon /api/sessions return the same total and page boundaries for identical filters (parity test across the three surfaces)
- Acceptance proof: the per-surface spec->filter mapping and total/cursor logic is deleted in favor of one execution helper (grep shows no second query_terms/contains merge)
- Acceptance proof: the two MCP list surfaces converge to one total semantic
- Acceptance proof: devtools verify green.

Packet: `task_packets/041_polylogue_t46_3.md`

## 042. polylogue-t46.4 — Delegate daemon session-similarity KNN to SqliteVecProvider.query_by_session

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Design direction: daemon/similarity.py re-implements session-seeded vector ranking (raw MATCH/k SQL over message_embeddings, per-session best-distance aggregation, matched-message count, L2->cosine in _l2_to_cosine_similarity:217) that storage/search_providers/sqlite_vec_queries.py:143 SqliteVecProvider.query_by_session already does -- the substrate file even comments that the daemon's _PER_MESSAGE_K mirrors it. Fix: build_similar_pa…

Implementation spine:
- daemon/similarity.py re-implements session-seeded vector ranking (raw MATCH/k SQL over message_embeddings, per-session best-distance aggregation, matched-message count, L2->cosine in _l2_to_cosine_similarity:217) that storage/search_providers/sqlite_vec_queries.py:143 SqliteVecProvider.query_by_session already does -- the substrate file even comments that the daemon's _PER_MESSAGE_K mirrors it.
- Fix: build_similar_payload (http.py:3158) should call the facade/vec-provider session-similarity method and only project the payload
- delete the daemon KNN/aggregation/L2->cosine copy.
- If the daemon needs a per-session rollup the provider does not expose, add it to the provider (substrate), not the surface.

Tests:
- Acceptance proof: daemon _knn_for_embedding/_aggregate_hits/_l2_to_cosine_similarity are deleted
- Acceptance proof: /api/similar ranking equals SqliteVecProvider.query_by_session ordering for a seed session (parity test)
- Acceptance proof: the sqlite_vec_queries comment about mirroring _PER_MESSAGE_K is removed because there is no longer a mirror
- Acceptance proof: devtools verify green.

Packet: `task_packets/042_polylogue_t46_4.md`

## 043. polylogue-t46.5 — Route CLI transcript/dialogue file export through substrate read+render; delete streaming_markdown SQL path

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Design direction: cli/read_views/streaming_markdown.py forks the whole read path for `read --view transcript/dialogue --to file` markdown exports: its own read-only index.db connection, ref resolution (_resolve_session_id), prefix-sharing lineage gating in raw SQL (_has_prefix_sharing_edge), _table_exists, and the message+block keyset join + block filtering -- duplicating api get_session/get_messages_paginated/read_archive_session_en…

Implementation spine:
- cli/read_views/streaming_markdown.py forks the whole read path for `read --view transcript/dialogue --to file` markdown exports: its own read-only index.db connection, ref resolution (_resolve_session_id), prefix-sharing lineage gating in raw SQL (_has_prefix_sharing_edge), _table_exists, and the message+block keyset join + block filtering -- duplicating api get_session/get_messages_paginated/read_archive_session_…
- It deliberately bails (returns False) on prefix-sharing sessions, so forked/resumed session file exports silently diverge.
- Fix: expose a streaming/iterator markdown render over the substrate read (add an iter/stream method on the facade or reuse get_messages_paginated) so standard.py:85/:119 use the same composition+block-filtering as the non-streaming path
- delete streaming_markdown.py's SQL.
- Keep the no-buffering benefit by streaming from the paginated substrate read.

Tests:
- Acceptance proof: streaming_markdown.py raw-SQL read helpers are deleted
- Acceptance proof: transcript/dialogue --to file markdown for a prefix-sharing (forked/resumed) session composes the full lineage identically to stdout output (test compares file export bytes vs the substrate transcript for a forked session)
- Acceptance proof: block filtering (reasoning/prose) matches the substrate projection
- Acceptance proof: devtools verify green.

Packet: `task_packets/043_polylogue_t46_5.md`

## 044. polylogue-t46.6 — Fix referenced_path OR-vs-AND filter divergence and delete dead CLI stats aggregators

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Design direction: cli/query_semantic.py:63 referenced_path_matches_slice uses any(term...) (OR-of-terms) for multi-term referenced_path while the substrate archive/query/runtime_matching.py:35 uses all(term...) (AND-of-terms), so the semantic-stats surface selects different sessions than the actual query filter -- a live correctness divergence. Separately, cli/query_stats.py (origin/date grouping :63-78/:353/:399, semantic grouping :…

Implementation spine:
- cli/query_semantic.py:63 referenced_path_matches_slice uses any(term...) (OR-of-terms) for multi-term referenced_path while the substrate archive/query/runtime_matching.py:35 uses all(term...) (AND-of-terms), so the semantic-stats surface selects different sessions than the actual query filter -- a live correctness divergence.
- Separately, cli/query_stats.py (origin/date grouping :63-78/:353/:399, semantic grouping :446/:514, profile work-kind grouping :628 via auto_tags 'kind:' scan) and query_semantic.py:151 re-derive aggregation that ArchiveStore.stats_by (SQL, api get_stats_by) already owns via workflow_shape/sort_key_ms, and they have no live CLI dispatch caller (only re-exports + tests).
- Fix: route query_semantic path/action matching through the shared predicate/SQL params (delete the CLI copies), and delete the dead query_stats/query_semantic in-memory aggregators in favor of stats_by, removing the tests that pin the dead shape.

Tests:
- Acceptance proof: A two-term referenced_path query returns the same session set from the semantic-stats surface and from the query filter (regression test)
- Acceptance proof: referenced_path_matches_slice/action_matches_slice and the dead query_stats aggregators are deleted (grep confirms callers gone)
- Acceptance proof: origin/date/tool/work-kind grouping goes through stats_by
- Acceptance proof: devtools verify green.

Packet: `task_packets/044_polylogue_t46_6.md`

## 045. polylogue-t46.8 — MCP surface collapse: ~96 tools -> verb algebra (query/get/explain/context/assert/maintenance...)

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The MCP surface (96 tools live) is a discovery burden and a maintenance trap (every tool = contract + names + regen). Collapse to a small verb algebra: one query(expression) over the DSL absorbs ~40 read tools; get/explain/context/correlate/coordinate/assert/retract/maintenance cover the rest; sessions/messages/blocks/evidence-packs become MCP RESOURCES (URI-addressed, subscriptions/list_changed) and recall packs/saved views become MCP PROMPTS — the protocol-native primitive split instead of tools-for-everything. …

Implementation spine:
- Identify the currently duplicated surface paths for this behavior.
- Create/extend the shared contract object and route one surface at a time through it.
- Add parity tests across CLI, daemon/API, MCP, and Python facade.
- Delete dead surface-side code after parity is green.

Tests:
- Acceptance proof: Verb set + resources + prompts cover every retired tool proven by goldens
- Acceptance proof: EXPECTED_TOOL_NAMES shrinks with equivalence evidence per deletion
- Acceptance proof: discovery tests + contracts regenerated
- Acceptance proof: no capability regression reported by the golden suite.
- Acceptance proof: SHADOW TELEMETRY GATE (added 2026-07-06): before any tool deletion, a shadow-mode window records per-tool called-count by client/harness, mapped replacement verb/resource/prompt, golden parity status, and last-seen timestamp — deletion order follows observed compatibility, not design purity alone (MCP clients may have prompts/learned behavior keyed to old tool names).

Packet: `task_packets/045_polylogue_t46_8.md`

## 046. polylogue-jnj.1 — Collapse read per-view flags into ProjectionSpec/RenderSpec algebra

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: read exposes compact algebra (--projection/--render/--spec) alongside per-view flag clusters (--window-hours, --repo-path, --since-hours, --related-limit...). Extend ProjectionSpec/RenderSpec to cover neighbor/correlation/context options FIRST, then remove the aliases. Broad public CLI change, not deletion-only. Design direction: Collapse read per-view flags into the existing Query x Projection x Render algebra, and use this bead to converge the duplication that would otherwise make export/variant work accrete another surface. Current known overlap to resolve or explicitly boundary-test: ProjectionSpec.body_policy/exclude_block_kinds vs ContentProjectionSpec; RenderFormat vs SESSION_OUTPUT_FORMATS/format_session; RenderDestination vs ReadVie…

Implementation spine:
- Collapse read per-view flags into the existing Query x Projection x Render algebra, and use this bead to converge the duplication that would otherwise make export/variant work accrete another surface.
- Current known overlap to resolve or explicitly boundary-test: ProjectionSpec.body_policy/exclude_block_kinds vs ContentProjectionSpec
- RenderFormat vs SESSION_OUTPUT_FORMATS/format_session
- RenderDestination vs ReadViewInvocation.destination/deliver_content
- RenderSpec.layout free strings vs read-view/profile metadata

Tests:
- Acceptance proof: read --spec remains the visible contract for composed selection/projection/render state.
- Acceptance proof: Existing per-view options for neighbor/correlation/context are represented in ProjectionSpec/RenderSpec or an explicitly named profile contract.
- Acceptance proof: At least one duplication pair is removed or converted into a single source of truth with tests
- Acceptance proof: any remaining pair has a documented boundary and drift check.
- Acceptance proof: HTML/static export work can select a reader render profile over QueryProjectionSpec without a bespoke export command family.

Packet: `task_packets/046_polylogue_jnj_1.md`

## 047. polylogue-jnj.5 — Route ops reset --session/--source through the mutation contract

**P2 / bug / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Identity resets tombstone directly before the preview/confirmation branch — a typo mutates suppression state without dry-run or JSON evidence. Require dry-run preview + --yes + stable JSON like other destructive ops. Design direction: Audit-confirmed: ops reset --session/--source tombstones BEFORE the preview/confirmation branch in the reset command implementation (cli/commands/ reset path). Fix: route identity resets through the same mutation contract as other destructive ops — dry-run prints exact target rows (origin/native_id, counts), mutation requires --yes, stable JSON envelope for both. Test: typo'd session ref produces zero-target dry-run…

Implementation spine:
- Audit-confirmed: ops reset --session/--source tombstones BEFORE the preview/confirmation branch in the reset command implementation (cli/commands/ reset path).
- Fix: route identity resets through the same mutation contract as other destructive ops — dry-run prints exact target rows (origin/native_id, counts), mutation requires --yes, stable JSON envelope for both.
- Test: typo'd session ref produces zero-target dry-run and no mutation
- real ref mutates only with --yes.

Tests:
- Acceptance proof: `polylogue ops reset --session <ref>` and `--source <ref>` print a dry-run of the exact target rows (origin/native_id + counts) BEFORE any tombstone write
- Acceptance proof: no mutation occurs without `--yes` (code path confirmed: tombstone no longer runs before the preview/confirmation branch — grep the reset command implementation).
- Acceptance proof: Test: a typo'd/nonexistent session ref produces a zero-target dry-run and zero rows mutated (suppression state asserted unchanged).
- Acceptance proof: Test: a real ref with `--yes` mutates only the named targets
- Acceptance proof: a stable JSON envelope is emitted for both dry-run and mutation (same shape as other destructive ops).

Packet: `task_packets/047_polylogue_jnj_5.md`

## 048. polylogue-x7d — Unify root query row rendering contracts

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The bounded find fix had to patch three projection/rendering paths: archive_query root rows, query_output deterministic rows, and select rows. This duplication let --limit bound row count while multiline titles/snippets still exploded output in the live archive. Collapse list/search/select row rendering onto one projection contract for title normalization, snippet bounds, machine payload shape, and plain text rendering, then keep archive_query/query_output/select as thin adapters. Design direction: Target shape: define a small shared row projection helper or value object for session list rows and search-hit rows, with explicit budgets (title 96 or table budget, snippet 320), single-line normalization, and separate full-read expansion. archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output format_summary_list/format_search_hit_list, and cli.select select_row_from_result should cal…

Implementation spine:
- Target shape: define a small shared row projection helper or value object for session list rows and search-hit rows, with explicit budgets (title 96 or table budget, snippet 320), single-line normalization, and separate full-read expansion.
- archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output format_summary_list/format_search_hit_list, and cli.select select_row_from_result should call that shared contract rather than each carrying its own truncation rules.
- Preserve existing JSON schemas
- change only overlong values.
- Add parity tests proving the three surfaces produce bounded titles/snippets for the same giant title/search hit.

Tests:
- Acceptance proof: A shared row-projection helper/value object exists for session-list rows and search-hit rows with explicit budgets (title 96 / table budget, snippet 320), single-line normalization, and separate full-read expansion.
- Acceptance proof: archive_query._summary_payload/_hit_payload/_summary_line/_hit_line, cli.query_output.format_summary_list/format_search_hit_list, and cli.select.select_row_from_result all call the shared contract (grep shows no per-surface truncation rules remaining).
- Acceptance proof: Existing JSON schemas are preserved
- Acceptance proof: only overlong values change.
- Acceptance proof: Parity tests prove the three surfaces produce bounded titles/snippets for the same giant title/search hit (`devtools test <parity test>` green).

Packet: `task_packets/048_polylogue_x7d.md`

## 049. polylogue-fnm.11 — Pipeline/clause parity across units + generated support matrix

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Live evidence: `sessions where origin:claude-code-session | count` fails with 'pipeline terminal stage must be an executable <unit>s where ... query' while `observed-events where kind:tool_finished | group by handler | count` works — the sessions unit does not support the count pipeline the docs/memory present as canonical. `after:2026-07-01` parses in bare find mode but is 'invalid query expression near column 43' inside `sessions where ...` — the compact-clause vocabulary differs between find-mode and unit-where… Design direction: (1) Build the support matrix FROM the registries (query_units + stage lowerers + clause grammar), not by hand: a generated docs/query-support-matrix.md (devtools render, drift-checked) showing units x pipeline stages x compact clauses. The generator doubles as the gap list. (2) Close the two gaps the evidence hit: `| count` (and group-by) on the sessions unit; date clauses (after:/before:) inside unit-where expressi…

Implementation spine:
- (1) Build the support matrix FROM the registries (query_units + stage lowerers + clause grammar), not by hand: a generated docs/query-support-matrix.md (devtools render, drift-checked) showing units x pipeline stages x compact clauses.
- The generator doubles as the gap list.
- (2) Close the two gaps the evidence hit: `| count` (and group-by) on the sessions unit
- date clauses (after:/before:) inside unit-where expressions.
- Both lower onto existing SQL (sessions has created_at

Tests:
- Acceptance proof: docs/query-support-matrix.md is generated from registries and drift-checked by render all --check.
- Acceptance proof: 'sessions where origin:X | count' and group-by on sessions work.
- Acceptance proof: after:/before: clauses parse inside unit-where expressions.
- Acceptance proof: Every unsupported unit/stage/clause combination errors with the unit, the construct, and the nearest supported alternative
- Acceptance proof: parse errors render a caret line.

Packet: `task_packets/049_polylogue_fnm_11.md`

## 050. polylogue-fnm.1 — Aggregates beyond count (sum/avg/min/max/percentiles)

**P2 / feature / 04-read-contract-query-render / blocked-hard**

Mechanism: Issue description localizes the mechanism: `group by X | count` is the only aggregate; cost/duration/token questions need sum/avg/percentiles to compose instead of spawning bespoke analyze modes. Design direction: Full target shape (fables ladder item 4): `| group by tool, session.origin | agg count, avg:duration_ms, p90:duration_ms, sum:tokens` — multi-field group by AND named aggregate list AND time bucketing `group by bucket:day(time)` (temporal-bucket machinery already exists in the temporal read view; reuse its bucket functions in the lowering). SQLite computes sum/avg/min/max natively; percentiles via nearest-rank in Py…

Implementation spine:
- Full target shape (fables ladder item 4): `| group by tool, session.origin | agg count, avg:duration_ms, p90:duration_ms, sum:tokens` — multi-field group by AND named aggregate list AND time bucketing `group by bucket:day(time)` (temporal-bucket machinery already exists in the temporal read view
- reuse its bucket functions in the lowering).
- SQLite computes sum/avg/min/max natively
- percentiles via nearest-rank in Python over grouped rows (pattern insights/portfolio.py:107-128).
- Pipeline stages are hand-parsed OUTSIDE the Lark grammar (~expression.py:1574/:2777) — no grammar change for the stage itself.

Tests:
- Acceptance proof: On the live archive `messages where ...
- Acceptance proof: | group by tool | agg count, avg:duration_ms, p90:duration_ms` returns per-group rows with each named metric column
- Acceptance proof: sum/avg/min/max lower to native SQLite aggregates and percentiles compute via nearest-rank in Python over grouped rows.
- Acceptance proof: Verify: pytest over a seeded corpus asserts column presence and computed values.
- Acceptance proof: Multi-field group-by (`group by tool, session.origin`) and time bucketing (`group by bucket:day(time)`) reuse the temporal read-view bucket functions.

Packet: `task_packets/050_polylogue_fnm_1.md`

## 051. polylogue-fnm.10 — fields/select stage with parent-field projection (first real Transform)

**P2 / feature / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The upward-access ceiling: session.* fields work for FILTERING on every unit (~25 scoped fields, metadata.py:620-658) and for whitelisted group-by, but output shapes are frozen Pydantic payloads that hardcode exactly two parent fields (MessageQueryRowPayload carries origin+title, payloads.py:~1265-1280). `messages where session.repo:polylogue AND text:timeout | fields text, occurred_at, session.title, session.repo` — the sessions join is already paid at filter time; projection means emitting columns the lowering a… Design direction: Land it as the first real Transform, fulfilling the QueryUnitTransformStage reservation (expression.py:376-386, 'never produced by the current parser'). Chain: hand-parsed stage keyword 'fields'/'select' -> Transform(name='select', args=[field list validated against the unit's field registry + session.* scoped family] ) -> lowering appends the requested columns to the SELECT list (parent columns via the existing ses…

Implementation spine:
- Land it as the first real Transform, fulfilling the QueryUnitTransformStage reservation (expression.py:376-386, 'never produced by the current parser').
- Chain: hand-parsed stage keyword 'fields'/'select' -> Transform(name='select', args=[field list validated against the unit's field registry + session.* scoped family] ) -> lowering appends the requested columns to the SELECT list (parent columns via the existing sessions join) -> output becomes a generic row payload (dict-shaped, field-name keyed) emitted ALONGSIDE the typed default payloads, not replacing them — …
- Registry: mark projectable fields per unit in metadata.py so completions + validation share one source.
- Note partial overlap: field selection for ATTACHED units landed (867b1d094 era)
- this bead is projection on the PRIMARY unit rows.

Tests:
- Acceptance proof: `messages where session.repo:polylogue AND text:timeout | fields text, occurred_at, session.title, session.repo` returns generic dict-shaped rows keyed by requested field name, emitted alongside (not replacing) the typed MessageQueryRowPayload.
- Acceptance proof: Verify: pytest asserts row shape and that the frozen typed default payload is unchanged.
- Acceptance proof: Requested parent fields resolve through the already-paid sessions join
- Acceptance proof: field names are validated against the unit's field registry + the session.* scoped family (metadata.py)
- Acceptance proof: an unknown field errors listing supported fields.

Packet: `task_packets/051_polylogue_fnm_10.md`

## 052. polylogue-fnm.13 — Set-algebra over query results: union/intersect/except between queries

**P2 / feature / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Design direction: BRAINSTORM (2026-07-05, operator asked to explore syntax incl. changing the pipeline operator; grain = message/unit AND session). CORE REFRAME: this is relational algebra. Every construct is relation->relation: - query (and/or/not predicates) = base relation (row-set; grain = session OR message/unit) - set-ops union/intersect/except = binary relation->relation - pipeline stages (group by, fields, read, context-image…

Implementation spine:
- BRAINSTORM (2026-07-05, operator asked to explore syntax incl.
- changing the pipeline operator
- grain = message/unit AND session).
- CORE REFRAME: this is relational algebra.
- Every construct is relation->relation:

Tests:
- Acceptance proof: `polylogue find 'auth intersect week:2026-W01'` returns exactly the session_ids in both operand sets
- Acceptance proof: `except` subtracts
- Acceptance proof: `union` dedups.
- Acceptance proof: Cross-lane composition works: `semantic:"X" except ~keyword` combines a vector set and an FTS set.
- Acceptance proof: Macros compose: `@a intersect @b` (with fnm.12).

Packet: `task_packets/052_polylogue_fnm_13.md`

## 053. polylogue-svfj — Block content-hash citation anchors: blocks.content_hash + resolver with typed drift states

**P2 / task / 05-analysis-provenance-citations / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: THE anchor atom multiple programs stand on (webui cockpit citations, finding evidence refs rxdo.4, drift detection 37t.14, compaction loss anchors gjg.3, export citations). Today block identity is message_id:position — position shifts on re-ingest and fork replay, so any stored block citation can silently point at different content. Verified live: blocks table has NO content_hash (sessions and messages do). Add blocks.content_hash (32B, over canonical block EVIDENCE: type, text, tool_name, canonical tool_input, se… Design direction: Derived index-tier change: canonical DDL edit + version bump — batch with the next index window (gjg.1, ma2, 4ts.5, wohv all queue for the same bump). Writer computes at block write (BOTH storage twins). Boilerplate-duplicate ambiguity is expected (same prompt text N times) — the ambiguous state is the honest answer; position_hint + message hint disambiguate the common case. Empirical dup-rate check on the live arch…

Implementation spine:
- Derived index-tier change: canonical DDL edit + version bump — batch with the next index window (gjg.1, ma2, 4ts.5, wohv all queue for the same bump).
- Writer computes at block write (BOTH storage twins).
- Boilerplate-duplicate ambiguity is expected (same prompt text N times) — the ambiguous state is the honest answer
- position_hint + message hint disambiguate the common case.
- Empirical dup-rate check on the live archive is part of this bead (policy depends on it).

Tests:
- Acceptance proof: Anchor created pre-re-ingest resolves post-re-ingest as ok or drifted_position (verified content)
- Acceptance proof: a fork replay resolves relocated_lineage with the inheritance edge cited
- Acceptance proof: ambiguous returns candidates
- Acceptance proof: hash_mismatch never auto-rewrites.
- Acceptance proof: Verify: re-ingest fixture round-trip tests.

Packet: `task_packets/053_polylogue_svfj.md`

## 054. polylogue-rxdo.1 — ObjectRef expansion: query, query-run, result-set, finding, cohort, analysis, annotation-batch kinds

**P2 / task / 05-analysis-provenance-citations / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Verified live 2026-07-06: ObjectRefKind in core/refs.py is a closed Literal of 29 kinds with none of the analysis-object kinds; normalize_object_ref_text rejects unknown kinds, so nothing can target a query or result set today. This is the narrow prerequisite for the whole analysis-provenance epic: refs first, resolvers stubbed (typed unresolved payload until tables land), tables second. Design direction: Add kinds: query, query-run, result-set, finding, cohort, analysis, annotation-batch to ObjectRefKind + the kind map + normalize paths in core/refs.py. finding:<hash> resolves to the assertion row with kind=finding (assertion:<id> stays valid; finding is the public alias). resolve_ref dispatch gains stub branches returning typed unresolved payloads with reason=substrate-pending until the storage beads land. Register…

Implementation spine:
- Add kinds: query, query-run, result-set, finding, cohort, analysis, annotation-batch to ObjectRefKind + the kind map + normalize paths in core/refs.py.
- finding:<hash> resolves to the assertion row with kind=finding (assertion:<id> stays valid
- finding is the public alias).
- resolve_ref dispatch gains stub branches returning typed unresolved payloads with reason=substrate-pending until the storage beads land.
- Registered-kind hygiene: each new kind needs a user_audit surface entry and regenerated render openapi + cli-output-schemas or the every-kind audit invariant fails (known registration trap).

Tests:
- Acceptance proof: normalize_object_ref_text accepts the new kinds
- Acceptance proof: resolve_ref returns typed pending payloads for them
- Acceptance proof: user_audit + rendered schemas regenerated
- Acceptance proof: existing ref tests extended.
- Acceptance proof: Verify: devtools verify (testmon) + rg for the kind literals across surface schemas.

Packet: `task_packets/054_polylogue_rxdo_1.md`

## 055. polylogue-rxdo.2 — Content-addressed query identity + durable user.db queries/query_names/result_sets/query_edges

**P2 / task / 05-analysis-provenance-citations / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: query:<hash> keyed on the canonical planned AST AFTER macro expansion (mirrors content-hash idempotency: equivalent queries collapse). Mutable human names are a separate git-branch-style pointer table (name mutable, hash immutable). Durable result_sets rows are MANIFESTS (grain, corpus_epoch, member_count, membership merkle root, ordered_rank_hash, exactness, persistence class); exact members durable only for watch/pinned/finding/cohort persistence. query_edges (operand-of/refines/supersedes/derived-from/same-as) … Design direction: Canonicalization: expand macros -> typed AST -> NFC strings -> sort commutative AND/OR children -> preserve non-commutative pipeline/seq/except/sort/limit order -> canonical field aliases -> include grain+lane+rank policy -> relative-time queries hash the DYNAMIC ast while runs store resolved absolute bounds -> sha256 over sorted-key compact JSON. Two hashes on result sets because membership equality is not rank equ…

Implementation spine:
- Canonicalization: expand macros -> typed AST -> NFC strings -> sort commutative AND/OR children -> preserve non-commutative pipeline/seq/except/sort/limit order -> canonical field aliases -> include grain+lane+rank policy -> relative-time queries hash the DYNAMIC ast while runs store resolved absolute bounds -> sha256 over sorted-key compact JSON.
- Two hashes on result sets because membership equality is not rank equality (set algebra needs the first, UX drift the second).
- user.db v4->v5 additive migration: queries, query_names, result_sets, result_set_members, query_edges tables — MUST batch with the other pending user-v5 candidates (see the durable-batch coordination bead) behind a verified backup manifest.
- Migrate existing SAVED_QUERY assertions: compile+hash each into queries, repoint the assertion at query:<hash>.
- Guards to encode from the corpus review: macro identity instability (hash expanded AST, names carry supersedes), supersedes/derived-from DAG acyclicity check at insert.

Tests:
- Acceptance proof: Same query text with reordered AND operands yields one query hash
- Acceptance proof: @macro repoint does not change the hash of past runs
- Acceptance proof: user-tier migration preserves all existing assertions (parity test)
- Acceptance proof: set-algebra grain is part of result-set identity so cross-grain member keys cannot collide.
- Acceptance proof: Verify: focused tests on canonicalization + migration test + devtools verify.

Packet: `task_packets/055_polylogue_rxdo_2.md`

## 056. polylogue-rxdo.3 — Query-run + result-relation telemetry in ops.db; refs on every query envelope

**P2 / task / 05-analysis-provenance-citations / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Every COMMITTED query execution (CLI, MCP, daemon web, API) records an ops.db query_runs row (actor, surface, verb, request+lowered spec, archive epoch, timing, status, degraded state) and a result fingerprint + bounded sample refs; the query response envelope gains query_run_ref + result_set_ref + grain + count_precision. Previews/keystrokes are NEVER persisted (ephemeral preview id only, unless a debug flag). This is what lets Polylogue analyze its own use (query-runs where actor_kind:agent and status:failed) an… Design direction: ops.db is disposable so long-lived citations must not point here without promotion; expired query-run refs resolve to a typed expired-operational-ref payload, never silently vanish. Promotion path (pin/promote to user.db manifest) is the bridge to the durable bead. Envelope change is additive to SearchEnvelope for byte-compat. Wire at the shared execution chokepoint, not per-surface (t46 direction: contracts own sur…

Implementation spine:
- ops.db is disposable so long-lived citations must not point here without promotion
- expired query-run refs resolve to a typed expired-operational-ref payload, never silently vanish.
- Promotion path (pin/promote to user.db manifest) is the bridge to the durable bead.
- Envelope change is additive to SearchEnvelope for byte-compat.
- Wire at the shared execution chokepoint, not per-surface (t46 direction: contracts own surfaces).

Tests:
- Acceptance proof: CLI --json and MCP query responses carry the three refs for the same committed query (parity test)
- Acceptance proof: routine preview typing produces zero rows
- Acceptance proof: a promoted run survives ops.db reset.
- Acceptance proof: Verify: focused envelope tests + parity test.

Packet: `task_packets/056_polylogue_rxdo_3.md`

## 057. polylogue-rxdo.4 — AssertionKind.FINDING: claims with evidence, reusing the candidate->judge lifecycle verbatim

**P2 / task / 05-analysis-provenance-citations / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: A finding is a durable claim (n, statistic, query_ref, result_set_ref, expected) produced by a detector/agent/analysis, stored as an assertion row so the ENTIRE existing lifecycle (candidate default, judge_assertion_candidate accept/reject/defer/supersede, judgment recorded as assertion, promotion flips inject gate) is reused with zero new lifecycle code — exactly the pathology pattern in user_write.py. CORRECTION to the corpus design it derives from: adding the enum member needs NO user-tier migration (AssertionK… Design direction: value_json schema polylogue.finding.v1: finding_kind (query-delta | finding-drift | measure | pathology | claim-vs-evidence), statistic {op,value,unit}, n, query_ref, result_set_ref, baseline/current refs, optional expected {measure,op,value}. Defaults: status=candidate, visibility=private, context_policy={"inject":false,"promotion_required":true} — machine findings NEVER auto-inject (recursive-safety spine). Determ…

Implementation spine:
- value_json schema polylogue.finding.v1: finding_kind (query-delta | finding-drift | measure | pathology | claim-vs-evidence), statistic {op,value,unit}, n, query_ref, result_set_ref, baseline/current refs, optional expected {measure,op,value}.
- Defaults: status=candidate, visibility=private, context_policy={"inject":false,"promotion_required":true} — machine findings NEVER auto-inject (recursive-safety spine).
- Deterministic finding id = hash(claim key + target + value + sorted evidence refs + detector ref) so re-materialization cannot duplicate.
- Evidence laundering guard: findings carry source query/result refs so a report renderer can warn on circular evidence ancestry.

Tests:
- Acceptance proof: upsert_findings_as_assertions mirrors the pathology writer
- Acceptance proof: findings appear in the judgment queue
- Acceptance proof: finding refs resolve
- Acceptance proof: regenerated schemas + user_audit pass
- Acceptance proof: a re-run with identical inputs produces zero new rows.

Packet: `task_packets/057_polylogue_rxdo_4.md`

## 058. polylogue-rxdo.7 — Annotation substrate: schema registry, annotation batches, JSONL import surface, typed value predicates

**P2 / task / 05-analysis-provenance-citations / blocked-hard**

Mechanism: Issue description localizes the mechanism: The missing loop for external-agent analysis: export evidence pack -> agent labels rows under a declared schema -> import as candidate assertions -> query them back -> judge -> report. Storage is ~75% ready (assertions table + upsert + judge lifecycle all real, verified); what is missing: (1) a general import surface (act kind / MCP tool / CLI assertions import) accepting JSONL rows with full assertion shape, defaulting status=candidate + inject:false for agent authors; (2) an annotation_schemas registry declaring… Design direction: Schemas connect to the 9l5.7 measure-registry discipline: a label is an operationalization with construct-validity metadata, not just a JSON key. Trusted-schema auto-active is explicitly rejected for v1: ALL external-agent rows enter candidate (recursive-safety chokepoint in upsert_assertion — author_kind != user => CANDIDATE + inject:false — is a related but separate load-bearing bead in the safety program). Import…

Implementation spine:
- Schemas connect to the 9l5.7 measure-registry discipline: a label is an operationalization with construct-validity metadata, not just a JSON key.
- Trusted-schema auto-active is explicitly rejected for v1: ALL external-agent rows enter candidate (recursive-safety chokepoint in upsert_assertion — author_kind != user => CANDIDATE + inject:false — is a related but separate load-bearing bead in the safety program).
- Import validates refs against the archive, reports per-row failures, refuses rows without evidence refs when the schema demands them.

Tests:
- Acceptance proof: Roundtrip demo: export a bounded evidence pack, import 5 labeled rows as candidates, query them via assertions where with a typed value predicate, judge one active, render.
- Acceptance proof: Batch metadata queryable.
- Acceptance proof: Verify: integration-flavored focused test + MCP tool contract test (EXPECTED_TOOL_NAMES + contract + regen).

Packet: `task_packets/058_polylogue_rxdo_7.md`

## 059. polylogue-bby.15 — Evidence basket -> citable report -> verified export (cockpit core loop)

**P2 / task / 05-analysis-provenance-citations / blocked-hard**

Mechanism: Issue description localizes the mechanism: The missing "report" end of the web workbench: select blocks/spans in the reader -> basket (content-hash anchors + quote + note + provenance of the query that surfaced it) -> live Markdown report draft with footnotes -> EXPORT GATE re-resolves every citation and blocks/flags by state (ok + drifted_position export with verified note; drifted_message/relocated need explicit promotion; ambiguous/missing block by default; quarantined blocks unless the report is explicitly forensic; hash_mismatch hard-fails). Storage v… Design direction: Three-pane cockpit flow (results | reader+graph | basket+draft); daemon API basket/report/verify routes collapse into service verbs when the t46/B8 contract lands. Depends on the block content-hash anchor substrate. Batch overlay endpoint (assertions/marks for a set of refs) serves the reader badges.

Implementation spine:
- Three-pane cockpit flow (results | reader+graph | basket+draft)
- daemon API basket/report/verify routes collapse into service verbs when the t46/B8 contract lands.
- Depends on the block content-hash anchor substrate.
- Batch overlay endpoint (assertions/marks for a set of refs) serves the reader badges.

Tests:
- Acceptance proof: Full loop on the seeded demo corpus: query -> basket 5 items -> draft renders footnotes -> re-ingest the corpus -> verify flags the drifted item and export annotates it
- Acceptance proof: a deleted block blocks export with a typed reason.
- Acceptance proof: Verify: integration-flavored test over the loop.

Packet: `task_packets/059_polylogue_bby_15.md`

## 060. polylogue-3tl.16 — Public claims ledger: every README/launch claim carries a status and an evidence ref

**P2 / task / 05-analysis-provenance-citations / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Turn radical honesty into a product surface: every public claim (README, docs site, launch post, category one-liner) must be exactly one of proven (backed by a finding/proof artifact), capability (code exists, no measured-result claim), aspirational (roadmap only), or retired (no longer true). This is the discipline that keeps the flight-recorder positioning from becoming marketing fog — the product whose pitch is 'every metric resolves to bytes' cannot itself ship unresolvable claims. Complements 3tl.12 (README d… Design direction: A docs/claims.yml ledger (or user-tier finding-backed equivalent once rxdo.4 FINDING lands): claim id, text, status, evidence ref (finding id / artifact path / measurement), last-verified date. README/docs quantitative claims link to a ledger entry by id. CI lint: a quantitative or comparative public claim without a ledger ref fails; a ledger entry with status=proven whose evidence ref does not resolve fails. Upgrad…

Implementation spine:
- A docs/claims.yml ledger (or user-tier finding-backed equivalent once rxdo.4 FINDING lands): claim id, text, status, evidence ref (finding id / artifact path / measurement), last-verified date.
- README/docs quantitative claims link to a ledger entry by id.
- CI lint: a quantitative or comparative public claim without a ledger ref fails
- a ledger entry with status=proven whose evidence ref does not resolve fails.
- Upgrade path: ledger entries become user.db findings once analysis provenance (rxdo) exists, so public claims share the same lifecycle as internal findings.

Tests:
- Acceptance proof: claims.yml exists and covers every quantitative/comparative claim in README + docs site
- Acceptance proof: CI gate rejects unreferenced claims
- Acceptance proof: each status has at least one real entry or an explicit none
- Acceptance proof: the flight-recorder category claim itself is ledgered (initially capability, not proven).
- Acceptance proof: Verify: the CI lint run + a grep sweep of README claims against ledger ids.

Packet: `task_packets/060_polylogue_3tl_16.md`

## 061. polylogue-37t.12 — Judgment queue: operator bulk review/accept/reject of candidate assertions

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Design direction: WHY: 37t's loop names four stages (claims -> JUDGMENT -> preamble -> reboot) but only JUDGMENT has no owning bead. 37t.10 design already assumes 'the existing judgment queue lists setup candidates alongside memory candidates'; 37t.11's blocking security note states 'Candidate->judged promotion IS the QUOTED->OPERATOR transition' and that a source without a judgment gate cannot emit OPERATOR-class items. So the judgm…

Implementation spine:
- WHY: 37t's loop names four stages (claims -> JUDGMENT -> preamble -> reboot) but only JUDGMENT has no owning bead.
- 37t.10 design already assumes 'the existing judgment queue lists setup candidates alongside memory candidates'
- 37t.11's blocking security note states 'Candidate->judged promotion IS the QUOTED->OPERATOR transition' and that a source without a judgment gate cannot emit OPERATOR-class items.
- So the judgment surface is a hard prerequisite for those beads yet unbuilt.
- SUBSTRATE THAT ALREADY EXISTS (verify, do not rebuild): polylogue/storage/sqlite/archive_tiers/user_write.py -- judge_assertion_candidate(candidate_ref, decision in {accept,reject,defer,supersede}, reason, replacement_*) at :1245 records a JUDGMENT-kind assertion (enums.py AssertionKind.JUDGMENT) and, for accept/supersede, promotes via _promote_candidate_assertion at :1338 (candidate->active transition IS speced -…

Tests:
- Acceptance proof: MCP: 'judge_assertion_candidate' and 'list_assertion_candidates' tools exist on the operator/agent-write MCP role, listed in EXPECTED_TOOL_NAMES with TOOL_CONTRACT entries
- Acceptance proof: a candidate written by an agent (author_kind='agent', status='candidate') is listable and can be accepted/rejected via the MCP tool, and an accepted candidate produces a new ACTIVE assertion (verify via list_assertion_claims statuses=active).
- Acceptance proof: CLI: 'polylogue judge queue' lists pending candidates with kind/evidence-ref preview
- Acceptance proof: 'polylogue judge accept <ref1> <ref2> ...' judges MULTIPLE candidates in one invocation and reports per-ref outcome (accepted / skipped-not-candidate)
- Acceptance proof: a bulk accept over a mix of candidate and already-accepted refs succeeds partially without aborting (idempotent).

Packet: `task_packets/061_polylogue_37t_12.md`

## 062. polylogue-37t.11 — Context scheduler: one arbiter for everything that enters an agent's context

**P1 / feature / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: The missing 40% of the OS-vision design, and the coherence fix for a real fragmentation risk: as of today SEVEN independent mechanisms want to write into agent context, each with its own budget rules — repo brief + resume delta (37t.4), semantic recall (mhx.4), SRS-due lessons (rvh), blackboard messages (1hj), PreToolUse/prompt advisories (bfv), compaction re-grounding (gjg), and the affordance-index pointer (pj8). Built separately they will fight for the same tokens, double-inject, and be untunable as a whole. Th… Design direction: (1) SOURCE PROTOCOL: each injector registers (declare-once) as a ContextSource with: moment (session-start | pre-compact-resume | mid-session-advisory | on-demand), priority class (correctness > directives > recall > ambient), a propose() returning candidate items (each: content-or-ref, token cost, relevance score, source ref, expiry), and a degrade order (full -> ref-only -> drop). Existing beads become sources, no…

Implementation spine:
- (1) SOURCE PROTOCOL: each injector registers (declare-once) as a ContextSource with: moment (session-start | pre-compact-resume | mid-session-advisory | on-demand), priority class (correctness > directives > recall > ambient), a propose() returning candidate items (each: content-or-ref, token cost, relevance score, source ref, expiry), and a degrade order (full -> ref-only -> drop).
- Existing beads become sources, not owners of budgets: 37t.4's sections, mhx.4's recall hits, rvh's due lessons, 1hj's messages, gjg's re-grounding items.
- (2) SCHEDULER: per moment, allocate the scoped budget (y4c per-repo/per-moment budgets) across classes by fixed proportions with borrowing (unused directive budget flows to recall), then within class by score
- produce the final assembly deterministically (same inputs -> same context, testable).
- Hard invariants: never exceed moment budget

Tests:
- Acceptance proof: ContextSource protocol + scheduler in context/compiler.py with deterministic assembly (property test: same inputs -> byte-identical context)
- Acceptance proof: 37t.4's sections migrated as the first two sources
- Acceptance proof: budget invariants enforced (property test: never exceeds moment budget at any source combination)
- Acceptance proof: ledger rows written per injection and readable via CLI + MCP
- Acceptance proof: cross-source dedup demonstrated (advisory suppresses same-ref blackboard item in a seeded scenario).

Packet: `task_packets/062_polylogue_37t_11.md`

## 063. polylogue-d1y — polylogue hooks install: one-command harness wiring + hook liveness monitoring

**P1 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Hooks are the highest-fidelity capture channel (event-granularity, 100% coverage vs ~79% post-hoc per docs/hooks.md) and the enabling substrate for context injection — yet wiring them is manual settings.json surgery per harness, per machine, per event type (16 Claude Code events, 6 Codex), and NOTHING notices when they stop firing (harness update, moved script, broken PATH): capture silently degrades to post-hoc JSONL discovery. On this very machine only a recall hook + two agent-event hooks are wired — not even t… Design direction: (1) INSTALL: 'polylogue hooks install [--harness claude-code|codex] [--events recommended|all|<list>]' — idempotently merges the polylogue-hook entries into the harness settings (respects existing hooks, writes the minimal diff, --dry-run shows it); 'polylogue hooks status' shows wired-vs-recommended per harness with the exact missing entries; uninstall symmetric. Settings-file formats are harness-version-dependent …

Implementation spine:
- (1) INSTALL: 'polylogue hooks install [--harness claude-code|codex] [--events recommended|all|<list>]' — idempotently merges the polylogue-hook entries into the harness settings (respects existing hooks, writes the minimal diff, --dry-run shows it)
- 'polylogue hooks status' shows wired-vs-recommended per harness with the exact missing entries
- uninstall symmetric.
- Settings-file formats are harness-version-dependent — VERIFY current schemas at build time and encode per-harness adapters, not one template.
- (2) LIVENESS: the daemon knows which harnesses are active (it ingests their JSONL)

Tests:
- Acceptance proof: On a clean settings.json, hooks install --harness claude-code --events recommended wires the starter set
- Acceptance proof: a second run produces zero diff.
- Acceptance proof: hooks status shows wired vs observed-last-7d per event type.
- Acceptance proof: With hooks wired and the script broken, the daemon raises a hook-flow health alert within one session.

Packet: `task_packets/063_polylogue_d1y.md`

## 064. polylogue-pj8 — Agent query cookbook: MCP prompts + skill recipes as the discoverability layer

**P1 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Agents use what is in their face and skip what requires invention (jgp doctrine). The MCP server exposes ~61 read tools; nothing teaches an agent WHICH five matter for the common intents: 'what was I doing in this repo', 'postmortem the last failed session', 'what did we decide about X', 'what failed recently and was never acknowledged', 'find the session where we touched file Y'. server_prompts.py exists but the prompt surface is thin, and there is no harness-side skill teaching Polylogue idioms the way the beads… Design direction: Three thin layers over existing capability, no new query machinery: (1) MCP prompts: register ~6 intent-named prompts (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the right tool-call sequences with cwd/repo prefilled — prompts are the MCP-native discoverability channel. (2) A 'polylogue' harness skill (dots/claude/skills + codex overlay, …

Implementation spine:
- Three thin layers over existing capability, no new query machinery: (1) MCP prompts: register ~6 intent-named prompts (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the right tool-call sequences with cwd/repo prefilled — prompts are the MCP-native discoverability channel.
- (2) A 'polylogue' harness skill (dots/claude/skills + codex overlay, sinnix-side) with the same recipes in agent-readable form plus the two rules agents get wrong (archive root env var
- refs over dumps).
- (3) The SessionStart preamble (37t.4) ends with a one-line affordance index pointing at those prompts — injection makes the surface ambient.
- Acceptance: affordance-usage report shows tool diversity rising in agent sessions (baseline: today's usage is dominated by search/get_session).

Tests:
- Acceptance proof: ~6 intent-named MCP prompts are registered (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the correct tool-call sequences with cwd/repo prefilled
- Acceptance proof: total prompt count stays small (curation, not another catalog).
- Acceptance proof: The prompt set includes the coordination intents over the shared envelope (agent_status, agent_self, work_item/current packet, coordination_hazards, addressed_messages, handoff) per the s7ae coordination update.
- Acceptance proof: A `polylogue` harness skill (dots/claude/skills + codex overlay, sinnix-side) carries the same recipes plus the two rules agents get wrong (archive-root env var
- Acceptance proof: refs over dumps).

Packet: `task_packets/064_polylogue_pj8.md`

## 065. polylogue-s7ae.2 — Pre-deployment MCP and hook coordination batch

**P1 / task / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: Why: the coordination program will require MCP prompt/tool updates and harness/hook rollout. Deployment should not happen piecemeal after every small MCP change. Before asking for a Sinnix/Home Manager switch or other deployment, batch all MCP-related code/config/test work that can be completed locally, including Beads hook health integration and subtle Polylogue hook affordances. If deployment is the only remaining step, record that state and move on to other work. Design direction: Audit and implement the deployment-sensitive pieces in one pass: Polylogue MCP tools/prompts for coordination views, server tool contract registration, CLI/openapi/generated schema updates, Sinnix MCP registry implications if needed, Beads git hook health detection/reporting, and hook-mediated coordination source points. Keep hooks subtle: mostly silent evidence capture/liveness updates; visible advisories only thro…

Implementation spine:
- Audit and implement the deployment-sensitive pieces in one pass: Polylogue MCP tools/prompts for coordination views, server tool contract registration, CLI/openapi/generated schema updates, Sinnix MCP registry implications if needed, Beads git hook health detection/reporting, and hook-mediated coordination source points.
- Keep hooks subtle: mostly silent evidence capture/liveness updates
- visible advisories only through the context scheduler and only for material events such as direct messages, same-resource activity, stale roots, or merge/integration state.
- Install/verify Beads git hooks as part of the actual implementation lane, but treat hook installation as environment setup plus proof, not as the coordination ontology.
- Do not deploy until all predeploy MCP/hook code paths and tests are done

Tests:
- Acceptance proof: MCP prompt/tool surface for coordination is implemented or explicitly delegated to the envelope bead with no remaining predeploy MCP code gaps.
- Acceptance proof: Generated MCP/OpenAPI/CLI schemas are refreshed where required.
- Acceptance proof: Beads hook health is visible in devloop review/status and the coordination envelope when Beads is present.
- Acceptance proof: Beads git hooks are installed/verified in the Polylogue checkout or a precise blocker is recorded.
- Acceptance proof: Hook-based coordination capture/advisory paths are designed and tested without noisy hardcoded workflow policing.

Packet: `task_packets/065_polylogue_s7ae_2.md`

## 066. polylogue-s7ae.3 — Coordination messages and subtle scheduler-mediated advisories

**P1 / feature / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: Why: multi-agent cooperation needs lightweight communication and awareness, but not a noisy chatroom or hardcoded workflow police. Agents should be able to leave scoped messages, receive direct or relevant notices, and see overlap/resource awareness. Hooks should mostly capture facts silently; visible output should be rare, bounded, and mediated by the context scheduler. Design direction: Realize the coordination-specific parts of existing beads 1hj (blackboard as agent comms), bfv (advisory hooks), d1y (hook install/liveness), and 37t.11 (ContextSource scheduler/ledger). Prefer reusing blackboard/user rows for CoordinationMessage. Addressing scopes: repo, work-item, session-tree, direct session/agent, path/surface, resource scope, broadcast only when explicitly requested. Delivery: SessionStart and …

Implementation spine:
- Realize the coordination-specific parts of existing beads 1hj (blackboard as agent comms), bfv (advisory hooks), d1y (hook install/liveness), and 37t.11 (ContextSource scheduler/ledger).
- Prefer reusing blackboard/user rows for CoordinationMessage.
- Addressing scopes: repo, work-item, session-tree, direct session/agent, path/surface, resource scope, broadcast only when explicitly requested.
- Delivery: SessionStart and on-demand context snapshots first
- mid-session advisories only for direct messages or high-value material changes.

Tests:
- Acceptance proof: Agents can post and receive scoped coordination messages with refs/provenance, using existing blackboard/user-state machinery where viable.
- Acceptance proof: The coordination envelope exposes unread/addressed messages and recent advisories.
- Acceptance proof: Hook-triggered visible advisories are bounded, rare, and emitted only through the scheduler/ledger path.
- Acceptance proof: Tests cover direct message delivery, repo/work-item scoped delivery, TTL/expiry or equivalent boundedness, same-surface overlap as non-blocking awareness, generic resource episode warning, and no noisy injection when there is no material signal.
- Acceptance proof: MCP/CLI expose message/advisory read paths without requiring Beads

Packet: `task_packets/066_polylogue_s7ae_3.md`

## 067. polylogue-s7ae.5 — Live proof: two agents, separate worktrees, one repo — overlap, message, context, handoff

**P1 / task / 06-agent-context-coordination / blocked-hard**

Mechanism: Design direction: Realize the epic s7ae HEADLINE acceptance line ('A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet') — currently owned by no child. Build a reproducible proof (run script + captured JSON artifacts) demonstrating two agents (e.g. Claude + Codex) on ONE repo in SEPARATE git worktr…

Implementation spine:
- Realize the epic s7ae HEADLINE acceptance line ('A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet') — currently owned by no child.
- Build a reproducible proof (run script + captured JSON artifacts) demonstrating two agents (e.g.
- Claude + Codex) on ONE repo in SEPARATE git worktrees, showing via the coordination envelope: (a) mutual peer + same-repo-agent + resource-episode awareness (process-table overlaps already shipped in s7ae.1)
- (b) at least one SCOPED coordination message posted by one agent and observed as delivered/addressed in the other's envelope (s7ae.3)
- (c) context injection into the second agent recorded via the 37t.11 scheduler/ledger

Tests:
- Acceptance proof: A committed, reproducible proof exists (run script + captured before/after JSON envelope artifacts under a devtools workspace path) demonstrating: two agents on one repo in separate worktrees
- Acceptance proof: each envelope shows the other as a same-repo peer with overlap + resource-episode awareness
- Acceptance proof: exactly one scoped coordination message posted and observed as delivered/addressed in the recipient's envelope
- Acceptance proof: context injection recorded via the 37t.11 ledger
- Acceptance proof: a handoff packet produced and referenced by both agents.

Packet: `task_packets/067_polylogue_s7ae_5.md`

## 068. polylogue-ahqd — Observe MCP write adoption after role rollout

**P1 / task / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: Why: polylogue-27p made full/evidence/browser agent profiles write-capable and mutation rows author-attributed, but the current Codex process predates the Home Manager activation. The adoption proof should be collected from a freshly launched agent session using the write-role MCP server so the archive's affordance-usage report contains real MCP write calls rather than unit-test or shell simulations. What: launch or wait for a fresh full-profile agent, perform benign record_correction/add_tag/blackboard_post write…

Implementation spine:
- Confirm all write paths go through the candidate/non-injected assertion policy.
- Make scheduler/context-ledger the only injection route.
- Add a minimal operator-visible envelope/payload and one proof fixture.
- Verify two-agent or simulated-agent flow before broad rollout.

Tests:
- Acceptance proof: A freshly launched full/evidence/browser agent session performs benign record_correction, add_tag, and blackboard_post MCP calls
- Acceptance proof: the resulting archive rows carry the authoring session ref
- Acceptance proof: an affordance-usage artifact/report shows those write calls
- Acceptance proof: lean profile remains read-only.

Packet: `task_packets/068_polylogue_ahqd.md`

## 069. polylogue-3gd — Activation layer: the agent-side setup that makes the substrate get used at all

**P1 / feature / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: Operator directive (2026-07-03), verbatim intent: everything built here — assertions, annotation protocol, blackboard, judge/note verbs, MCP tools, remote control — is WASTED WORK if agents do not actually use it, and adoption is behavioral: models use what their context reminds them of. The fate to avoid is inert substrate. The fix is an unapologetically LARGE agent-side activation layer: the operator explicitly authorizes 10K+ tokens (possibly 30-50K including injected state) of instructions + state in global CL… Design direction: (1) GLOBAL CLAUDE.md SECTION (sinnix-side, dots/claude/ — renders to all agents via render-agents): a substantial 'Polylogue substrate' chapter (~3-6K tokens of standing instruction) structured as: WHAT EXISTS (archive, memory, blackboard, protocol — one paragraph each with the mental model); WHEN-TRIGGERS (a trigger->action table: 'starting work in a repo -> read your preamble, it is evidence', 'about to re-derive …

Implementation spine:
- (1) GLOBAL CLAUDE.md SECTION (sinnix-side, dots/claude/ — renders to all agents via render-agents): a substantial 'Polylogue substrate' chapter (~3-6K tokens of standing instruction) structured as: WHAT EXISTS (archive, memory, blackboard, protocol — one paragraph each with the mental model)
- WHEN-TRIGGERS (a trigger->action table: 'starting work in a repo -> read your preamble, it is evidence', 'about to re-derive how X works -> polylogue find/prior-art flow', 'discovered something durable -> ::lesson marker or polylogue note', 'finished a slice -> markers become candidates automatically
- check bd + blackboard', 'stuck/failed -> postmortem flow')
- HOW (the exact seven t8t flows with copy-paste invocations)
- PROTOCOL SPEC (the 37t.2 kinds with 5 worked examples inline — models imitate examples, not descriptions)

Tests:
- Acceptance proof: The substrate chapter exists in dots/claude and renders to Claude+Codex
- Acceptance proof: trigger table, seven flows, and five protocol examples present
- Acceptance proof: preamble cross-reference live
- Acceptance proof: the weekly adoption report renders from affordance-usage with baseline captured BEFORE the chapter ships (so the delta is measurable)
- Acceptance proof: one month later the report shows material adoption movement or the chapter is revised (decay-watch is part of acceptance, not an afterthought).

Packet: `task_packets/069_polylogue_3gd.md`

## 070. polylogue-t8t — Agent workflow catalog: walk the seven core flows end-to-end, fix what breaks

**P2 / task / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Affordances exist in pieces; nobody has walked the actual workflows end-to-end: (1) RESUME — arrive in repo, recover context, continue; (2) FORENSIC DEBUG — did a past session touch this; (3) PRIOR ART — has anything explored this approach; (4) DECISION LOOKUP — what did we decide and why; (5) POSTMORTEM WRITE — close the loop after failure (37t.7); (6) COST CHECK — what has this repo/task cost; (7) SELF-INSPECTION — agent reads its own live session mid-flight (raw-log 05-28; needs hook-fresh ingest + get_session … Design direction: (1) Each flow = a workflow registry entry (product/workflows.py REQUIRED_WORKFLOW_IDS) with intent, tool sequence, envelope shapes, round-trip token cost, latency budget (20d.14). (2) Execute from a real Claude Code session via polylogue MCP on this machine; archive the walk transcript as evidence. (3) Verify-or-refute known suspects: self-inspection freshness (searchable within seconds?), search-within-session ergo…

Implementation spine:
- (1) Each flow = a workflow registry entry (product/workflows.py REQUIRED_WORKFLOW_IDS) with intent, tool sequence, envelope shapes, round-trip token cost, latency budget (20d.14).
- (2) Execute from a real Claude Code session via polylogue MCP on this machine
- archive the walk transcript as evidence.
- (3) Verify-or-refute known suspects: self-inspection freshness (searchable within seconds?), search-within-session ergonomics, resume-brief token cost vs preamble budget, cost-check cold latency.
- (4) Output: per flow PASS+timing or a filed bead

Tests:
- Acceptance proof: Seven registry entries
- Acceptance proof: seven archived walk transcripts
- Acceptance proof: every gap filed as a linked bead
- Acceptance proof: rendered catalog lists measured tokens+latency per flow
- Acceptance proof: self-inspection demonstrates an agent reading its own in-progress session.

Packet: `task_packets/070_polylogue_t8t.md`

## 071. polylogue-rii.1 — Agent work-event write-leg -> session_events -> materialized read-models

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: record_work_event/emit_decision write surface routed through the existing idempotent ingest seam (no parallel writer); flows into the run-projection read models. Today agents can only record_correction/blackboard_post/tag — there is no 'I ran this tool / spawned this subagent / decided X' write. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) — no parallel writer (gh#2459 body is code-grounded here). Surface: MCP tools record_work_event/emit_decision (mutation role) accepting typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs; land in session_events; run-projection read models pick them up through the normal materia…

Implementation spine:
- Route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) — no parallel writer (gh#2459 body is code-grounded here).
- Surface: MCP tools record_work_event/emit_decision (mutation role) accepting typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs
- land in session_events
- run-projection read models pick them up through the normal materializer.
- MCP registration trap: EXPECTED_TOOL_NAMES + TOOL_CONTRACT + role gating + render openapi/cli-output-schemas regen (see bd memories).

Tests:
- Acceptance proof: MCP tools record_work_event / emit_decision are registered with the mutation role: EXPECTED_TOOL_NAMES + TOOL_CONTRACT updated, role gating enforced, and `devtools render openapi && devtools render cli-output-schemas` regenerated with `devtools render all --check` clean.
- Acceptance proof: Typed events (tool run, subagent spawn, decision, artifact change) with evidence/session refs route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) into session_events — no parallel writer (grep confirms reuse).
- Acceptance proof: Behavior test: an agent posts a work event mid-session and it is queryable via observed-events (session_work_events / DSL) within one convergence cycle
- Acceptance proof: re-posting the same event is idempotent (no duplicate row).
- Acceptance proof: `devtools test <mcp work-event test>` green.

Packet: `task_packets/071_polylogue_rii_1.md`

## 072. polylogue-rii.2 — Materialize hook events + OTLP spans into queryable evidence

**P2 / feature / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: Hook events are captured as raw blobs but never materialized (~95% of hook-only signal invisible: tool annotations, pre-MCP output, permission decisions, cwd changes, subagent lifecycle); OTLP spans likewise. Both converge on the write-leg contract. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Code-confirmed gaps (gh#2461, re-locate lines): archive/artifact_taxonomy/runtime.py:~209 classifies HOOK_EVENT with parse_as_session=False (stored as raw blobs, used only for paste enrichment); artifact_taxonomy/support.py:~82 looks_like_hook_event hardcodes provider in ('claude-code','codex'). Fix: materialize hook events through the write-leg contract into session_events/ObservedEvents keyed to the owning session…

Implementation spine:
- Code-confirmed gaps (gh#2461, re-locate lines): archive/artifact_taxonomy/runtime.py:~209 classifies HOOK_EVENT with parse_as_session=False (stored as raw blobs, used only for paste enrichment)
- artifact_taxonomy/support.py:~82 looks_like_hook_event hardcodes provider in ('claude-code','codex').
- Fix: materialize hook events through the write-leg contract into session_events/ObservedEvents keyed to the owning session (session id is in the hook payload)
- un-hardcode the provider check via the taxonomy.
- OTLP: spans already land in ops.db via the receiver — project them into queryable evidence the same way rather than a second reader.

Tests:
- Candidate-write safety test.
- Scheduler context-ledger determinism test.
- Simulated two-agent coordination envelope test.

Packet: `task_packets/072_polylogue_rii_2.md`

## 073. polylogue-x4s — Express devloop state in Polylogue substrate (dogfood target)

**P2 / task / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Raw-log 2026-07-03: 'perhaps devloops themselves could be expressed in sinex and/or polylogue and/or beads?'. Beads now owns task state. The remaining half: focus transitions, handoffs, proof claims, and velocity notes should eventually be archive/assertion data rather than markdown sidecars only. Candidate first slice: devloop-log dual-writes an assertion (kind=NOTE, author=devloop) with evidence refs to the producing session. Route through the substrate write-leg rather than a parallel writer. Design direction: The full argument (fables devloop reading): the conductor's own memory is the silo it is fighting — ACTIVE-LOOP.md, OPERATING-LOG.md, HANDOFF-LATEST.md, EVENTS.jsonl live outside the archive while the product has the native home sitting unwired: handoff and run_state assertion kinds exist in the enum with NO writer (user_write.py has no helper for them), blackboard has an unresolved filter, and get_resume_brief/comp…

Implementation spine:
- The full argument (fables devloop reading): the conductor's own memory is the silo it is fighting — ACTIVE-LOOP.md, OPERATING-LOG.md, HANDOFF-LATEST.md, EVENTS.jsonl live outside the archive while the product has the native home sitting unwired: handoff and run_state assertion kinds exist in the enum with NO writer (user_write.py has no helper for them), blackboard has an unresolved filter, and get_resume_brief/co…
- Target: conductor-on-assertions — active-loop state, handoffs, and focus transitions written as assertions in user.db, recovered at session start through the product's own context compilation instead of 'read these 11 files in order'.
- The post-compaction discipline exists because agent context is lossy — that is the product's founding problem, currently solved with markdown instead of the product.
- Sequence: (1) first writers for handoff/run_state kinds (dual-write from devloop-handoff/devloop-focus, markdown stays authoritative), (2) a conductor context profile in compile_context, (3) flip authority once recovery quality is proven, keeping markdown as a rendered VIEW of the assertions rather than the source.
- Coordinate with the beads split: beads owns task state

Tests:
- Acceptance proof: First writers for the handoff/run_state assertion kinds are added to user_write.py (the kinds already exist in the enum with no writer — grep confirms the new helpers)
- Acceptance proof: devloop-handoff / devloop-focus dual-write into user.db while markdown stays authoritative.
- Acceptance proof: A conductor context profile is added to compile_context that recovers active-loop state, handoffs, and focus transitions at session start (via get_resume_brief / compose_context_preamble, with provenance).
- Acceptance proof: Authority flips to assertions once recovery quality is proven
- Acceptance proof: markdown becomes a rendered VIEW of the assertions, not the source.

Packet: `task_packets/073_polylogue_x4s.md`

## 074. polylogue-4c0 — Beads-native work loop: session<->bead cross-links and archive-rendered work history

**P2 / task / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: Beads and the archive already observe the same work from two sides but never join: a bead's history (claims, closes, reasons) names no sessions; a session's transcript contains bd commands the archive does not structurally extract. Joining them makes both better: bd show could point at the sessions that did the work (with the postmortem one hop away); polylogue could render a bead's full work history (every session that touched it, what changed — yrx — what it cost, what failed); close reasons become claims checka… Design direction: (1) EXTRACTION: bd invocations are shell tool calls with structured output — a block enricher recognizes bd claim/close/create/update and materializes session<->bead edge rows (bead id, verb, timestamp, session ref); zero heuristics, the commands are structural. (2) READ SURFACES: 'polylogue bead <id>' (or a DSL unit: beads where id:X | sessions) renders the work history envelope: sessions, durations, cost, changes …

Implementation spine:
- (1) EXTRACTION: bd invocations are shell tool calls with structured output — a block enricher recognizes bd claim/close/create/update and materializes session<->bead edge rows (bead id, verb, timestamp, session ref)
- zero heuristics, the commands are structural.
- (2) READ SURFACES: 'polylogue bead <id>' (or a DSL unit: beads where id:X | sessions) renders the work history envelope: sessions, durations, cost, changes summary, close reason vs evidence
- MCP twin for agents.
- (3) BEADS SIDE: a bd-side pointer needs no bd fork — the devloop convention writes the session ref into close reasons/notes automatically via a Stop-hook helper (the hook knows the session id and the claimed bead).

Tests:
- Acceptance proof: On the live archive: session<->bead edges materialize for the recent devloop sessions
- Acceptance proof: the bead work-history envelope renders for a real closed bead with sessions, cost, and changes
- Acceptance proof: a close-reason cross-check runs for one campaign bead and reports agreement
- Acceptance proof: Stop-hook writes the session ref into bd notes on claim/close.

Packet: `task_packets/074_polylogue_4c0.md`

## 075. polylogue-0v9p — Language detection and preference facts for variant selection

**P1 / feature / 07-content-variants / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Why: agents should translate when useful, but the archive first needs honest language facts. Language detection is distinct from translation: it annotates source blocks/messages/sessions and informs projection defaults, filters, and agent prompts without creating transformed content. Design direction: Add a language fact layer at block grain where practical, with message/session rollups derived from children. Automatic detections are rebuildable derived facts with detector/version/confidence; user corrections/preferences live in user.db/user_settings or assertion-backed corrections where appropriate. Support mixed-language messages by preserving block/span facts instead of forcing one session language. Expose que…

Implementation spine:
- Add a language fact layer at block grain where practical, with message/session rollups derived from children.
- Automatic detections are rebuildable derived facts with detector/version/confidence
- user corrections/preferences live in user.db/user_settings or assertion-backed corrections where appropriate.
- Support mixed-language messages by preserving block/span facts instead of forcing one session language.
- Expose query predicates and projection defaults such as preferred target language, translate-if-source-not-preferred, and confidence thresholds.

Tests:
- Acceptance proof: Block/message/session language facts exist with confidence and provenance.
- Acceptance proof: Mixed-language messages are represented without collapsing to one false language.
- Acceptance proof: User preference/correction state overrides derived detection without altering source content.
- Acceptance proof: Query surfaces can filter by source language, and variant projection can choose candidate translation targets from language facts.
- Acceptance proof: Tests cover mixed-language blocks, low-confidence/unknown detection, user override, and no translation created merely by detection.

Packet: `task_packets/075_polylogue_0v9p.md`

## 076. polylogue-arso — Content variant substrate: refs, nodes, alignment, storage

**P1 / feature / 07-content-variants / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Why: translations and other transformed content need a first-class substrate over existing public refs. A target_ref=session must mean the whole declared session composition; target_ref=message must mean the whole message; target_ref=block means exactly that block. The system must not encode these as loose notes or assertion blobs, because variants are transformed content artifacts with provenance and alignment, not epistemic claims. Design direction: Implement typed models and storage for ContentVariant, VariantNode, and VariantAlignment. Extend public refs to include variant:<id> and variant-node:<id>; preserve existing assertion:<id> refs and allow variants to target assertion refs. Use closed vocabularies: kind translation/transliteration/simplification/summary; status candidate/active/rejected/superseded/stale; coverage complete/partial/sparse; relation tran…

Implementation spine:
- Implement typed models and storage for ContentVariant, VariantNode, and VariantAlignment.
- Extend public refs to include variant:<id> and variant-node:<id>
- preserve existing assertion:<id> refs and allow variants to target assertion refs.
- Use closed vocabularies: kind translation/transliteration/simplification/summary
- status candidate/active/rejected/superseded/stale

Tests:
- Acceptance proof: Canonical types, storage DDL, repository/API read/write methods, and public ref resolution exist for variant and variant-node refs.
- Acceptance proof: Variants can target session/message/block/assertion refs.
- Acceptance proof: Alignment supports one-to-one, one-to-many, many-to-one, omitted, and partial mappings.
- Acceptance proof: Tests prove a session-level variant with complete coverage covers all declared child messages/blocks, a partial variant is labeled partial, a summary maps many source nodes to one variant node without positional hacks, and a translated assertion remains a variant of assertion:<id> rather than a projected original assertion.
- Acceptance proof: Generated schemas/docs are refreshed where required.

Packet: `task_packets/076_polylogue_arso.md`

## 077. polylogue-rlsb — Variant-aware projection, query, and reader render profiles

**P1 / feature / 07-content-variants / blocked-hard**

Mechanism: Issue description localizes the mechanism: Why: translated or simplified content should be selectable, readable, exported, and queried through the existing read algebra, not through bespoke translation flags or export modes. Query results and renderers must label source-vs-variant text so downstream analysis does not lie about the underlying archive. Design direction: Extend the existing Query x Projection x Render algebra without adding a new overlay abstraction. Variant inclusion is semantic projection: add a ProjectionSpec variant_policy (include none/exact/inherited/composed, kinds, target_language, status_policy, coverage_policy, alignment_policy) and an EvidenceFamily/terminal unit surface for variants where needed. Do NOT put source-vs-variant inclusion decisions in Render…

Implementation spine:
- Extend the existing Query x Projection x Render algebra without adding a new overlay abstraction.
- Variant inclusion is semantic projection: add a ProjectionSpec variant_policy (include none/exact/inherited/composed, kinds, target_language, status_policy, coverage_policy, alignment_policy) and an EvidenceFamily/terminal unit surface for variants where needed.
- Do NOT put source-vs-variant inclusion decisions in RenderSpec.
- RenderSpec remains delivery/encoding/profile only: format, destination, timestamp policy, out path, and a renderer/profile/layout id that chooses visual arrangement such as original-only, variant-only, dual, interleaved, or hover-source when the projected payload already contains variant lanes.
- Before or while implementing variants, audit and converge the existing selection/projection/render duplication: ProjectionSpec.body_policy and exclude_block_kinds overlap with ContentProjectionSpec

Tests:
- Acceptance proof: CLI/API/MCP/daemon query/read paths can request variants through ProjectionSpec and existing query/projection stages.
- Acceptance proof: JSON payloads label original source text and variant text distinctly, including exact/inherited/composed coverage and aligned refs.
- Acceptance proof: Queries can find variants by target, kind, language, status, and alignment, and source rows can report variant coverage without treating translated text as original evidence.
- Acceptance proof: Markdown/HTML renderers support original, variant, and dual/interleaved visual profiles by consuming projected variant lanes, not by deciding semantic inclusion themselves.
- Acceptance proof: The implementation includes a concrete convergence audit/fix for the current projection/render overlap: no duplicated new variant-layout abstraction, and any retained RenderSpec/read-view/profile/format/content-projection split has a documented boundary and tests.

Packet: `task_packets/077_polylogue_rlsb.md`

## 078. polylogue-d4zk — User and agent UX for creating, reviewing, and messaging about variants

**P1 / feature / 07-content-variants / blocked-hard**

Mechanism: Issue description localizes the mechanism: Why: the operator wants agents to translate at will and wants to view/interact with those translations. The human user should also participate in the same object-ref messaging substrate as agents: point at a block/message/assertion/session, ask an agent to create a variant, review the result, and send decisions back with refs. Design direction: Build UX over existing addressing and coordination messages. Web/reader/in-page surfaces let the user select a session/message/block/span/assertion/variant-node and request create_variant/translate/simplify/summarize from an existing or new agent participant. Agents can send messages to user:local with attached refs such as variant-node low-confidence alignment, missing assertion translation, or review-needed candid…

Implementation spine:
- Build UX over existing addressing and coordination messages.
- Web/reader/in-page surfaces let the user select a session/message/block/span/assertion/variant-node and request create_variant/translate/simplify/summarize from an existing or new agent participant.
- Agents can send messages to user:local with attached refs such as variant-node low-confidence alignment, missing assertion translation, or review-needed candidate.
- MCP prompts/tools expose create_content_variant and translate_target as convenience over the generic variant write path.
- Review UX shows coverage, alignment, status, source language, target language, author/provenance, and missing translated assertions.

Tests:
- Acceptance proof: A user can address an object ref and request a variant-producing action through CLI/MCP and at least one web/in-page UX path.
- Acceptance proof: Agents can create candidate variants with alignment metadata and send a user-addressed coordination message containing clickable refs.
- Acceptance proof: The user can accept/reject/supersede or otherwise mark variant status without changing original source content.
- Acceptance proof: UI distinguishes original assertions from translated assertion variants and handles missing assertion translations honestly.
- Acceptance proof: Tests or demo fixtures cover translate heavily annotated session, review low-confidence alignment, and agent-to-user message with attached variant refs.

Packet: `task_packets/078_polylogue_d4zk.md`

## 079. polylogue-4smp — Content variants: language-aware transformed archive objects with alignment

**P1 / epic / 07-content-variants / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Why: agents should be able to translate source content, annotations, and other addressable Polylogue objects for the operator, and the reader/export/query surfaces should let the operator view and interact with those translations without confusing transformed text with original evidence. The operator's "alternates" sketch is not the requirement; the requirement is a general algebraic substrate for transformed content. Translation is the motivating case, but the same primitive should support transliteration, simpli… Design direction: Core model: ContentVariant(target_ref, kind, source_language, target_language, status, coverage, composition_policy, author_ref, evidence_refs, staleness/supersession, metadata). VariantNode represents structured variant content at session/message/block/span/assertion-body grain. VariantAlignment maps source_ref -> variant_node_ref with relation vocabulary such as translates, transliterates, simplifies, summarizes, …

Implementation spine:
- Core model: ContentVariant(target_ref, kind, source_language, target_language, status, coverage, composition_policy, author_ref, evidence_refs, staleness/supersession, metadata).
- VariantNode represents structured variant content at session/message/block/span/assertion-body grain.
- VariantAlignment maps source_ref -> variant_node_ref with relation vocabulary such as translates, transliterates, simplifies, summarizes, omits, expands, reorders.
- Do not rely on positional convention such as "summary in first block"
- agents may provide partial alignment when exact mapping is unavailable.

Tests:
- Acceptance proof: A typed content-variant model exists over public refs without treating variants as assertions.
- Acceptance proof: Variants support at least translation, transliteration, simplification, and summary with closed relation/status/coverage vocabularies.
- Acceptance proof: Variant nodes and alignment edges allow session/message/block/assertion variants to map source child elements honestly, including many-to-one summary relations and partial alignment.
- Acceptance proof: Query/read/export/web/MCP surfaces label source vs variant text and never present translations as original evidence.
- Acceptance proof: A demo or fixture shows a heavily annotated session translated with transcript variants plus variants of selected assertion annotations, with clickable alignment back to original source and original assertions.

Packet: `task_packets/079_polylogue_4smp.md`

## 080. polylogue-4ts.3 — Distinguish subagent auto-compaction from main-session acompact

**P2 / bug / 03-lineage-compaction-truth / implementation-ready-after-local-inspection**

Mechanism: Bead hints point to Claude parser behavior around `agent-acompact-*` parent assignment. Static next step is to inspect parser code with `rg 'agent-acompact|acompact|compaction|parent' polylogue/sources/parsers` and find where parent/lineage kind is assigned.

Implementation spine:
- Implementation shape:
- 1. Locate parser branch that classifies Claude Code auto-compaction/session ids.
- 2. Add an explicit lineage event kind or flag: `main_compaction`, `subagent_auto_compaction`, `subagent_spawn`, etc.
- 3. Parent assignment: a subagent auto-compact event should attach to the subagent/session tree, not become a main-session compaction boundary.
- 4. Downstream lineage/composed-session renderers should show subagent compaction separately and not truncate/restart main session effective context.

Tests:
- fixture with main session + subagent `agent-acompact-*`: main compaction count remains zero or unchanged; subagent compaction count increments.
- composed lineage tree renders subagent compaction under subagent.
- existing main compaction fixture still works.
- aggregate counts distinguish both classes.

Packet: `task_packets/080_polylogue_4ts_3.md`

## 081. polylogue-4ts.4 — Wrap lineage composition reads in a single read transaction

**P2 / bug / 03-lineage-compaction-truth / implementation-ready-after-local-inspection**

Mechanism: Likely mechanism from bead title: lineage composition performs multiple separate reads/connection steps, so concurrent ingest/refresh can produce an impossible graph. This is especially dangerous for branch/shared-prefix accounting and compaction context.

Implementation spine:
- Implementation shape:
- 1. Locate lineage composition entrypoint and list every DB read it performs.
- 2. Ensure the public composition call opens one read connection and begins a read transaction/snapshot before the first query.
- 3. Pass that connection through lower helpers instead of helpers reopening connections.
- 4. If both source and index tiers are needed, document consistency model; prefer a single tier/projection for composition or record cross-tier snapshot caveat.

Tests:
- two-connection fixture: begin composition read, mutate lineage/messages on another connection, finish composition; result reflects one consistent before/after state, not mixed.
- helper tests assert no lower function opens a new connection when a connection is supplied.
- existing lineage composition tests still pass.

Packet: `task_packets/081_polylogue_4ts_4.md`

## 082. polylogue-4ts.5 — Compaction boundary-range columns + effective-context derivation

**P2 / feature / 03-lineage-compaction-truth / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: session_events boundary_start/end_position + boundary_message_id; get_effective_context(session, at_position) = what the model actually saw vs the full composed prefix. Schema bump + re-ingest. Surfaces: view=effective_context; precise replaced-range signal for stale_context pathology. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Design from gh#2478 (code-grounded): add session_events.boundary_start_position/boundary_end_position (message range a compaction replaces, in the session's own position coordinate) + boundary_message_id (the materialized summary). Parser computes the range while walking records: start = prev boundary end + 1, end = message_position - 1; writer applies position_offset. Read helper get_effective_context(session, at_p…

Implementation spine:
- Design from gh#2478 (code-grounded): add session_events.boundary_start_position/boundary_end_position (message range a compaction replaces, in the session's own position coordinate) + boundary_message_id (the materialized summary).
- Parser computes the range while walking records: start = prev boundary end + 1, end = message_position - 1
- writer applies position_offset.
- Read helper get_effective_context(session, at_position) returns [summary] + post-boundary messages (what the model actually saw) vs the full composed prefix used for forks.
- Index-tier schema bump + re-ingest plan in the PR body (fresh-first doctrine

Tests:
- Acceptance proof: 1.
- Acceptance proof: session_events gains boundary_start_position, boundary_end_position, and boundary_message_id (index-tier schema bump, with the rebuild/re-ingest plan stated in the PR body per fresh-first doctrine and batched with other pending index bumps where possible).
- Acceptance proof: 2.
- Acceptance proof: The parser populates the range while walking records — start = prev boundary end + 1, end = message_position - 1, with position_offset applied — for Codex `compacted` records and Claude inline / agent-acompact-* boundaries.
- Acceptance proof: 3.

Packet: `task_packets/082_polylogue_4ts_5.md`

## 083. polylogue-4ts.6 — Lineage composition silently truncates transcripts; surface a completeness signal

**P2 / bug / 03-lineage-compaction-truth / implementation-ready-after-local-inspection**

Mechanism: Bead title says silently truncated transcripts need a completeness signal. Likely sources: provider export truncation, partial file capture, continuation branches, max-message reads, or parser fallback. The exact current code path needs `rg 'truncated|complete|partial|max_messages|limit' polylogue/sources polylogue/storage polylogue/read` before patching.

Implementation spine:
- Implementation shape:
- 1. Identify all parser/read paths that can return partial sessions: provider export flags, capture payload bounds, read limits, branch-local raw logs, and daemon read pagination.
- 2. Add a `transcript_completeness` enum or structured field: `complete`, `partial_export`, `parser_partial`, `read_limited`, `unknown`, with reason/source.
- 3. Store completeness in session/profile/read payloads.
- 4. Context packs and reports must show the signal and avoid strong claims over partial sessions.

Tests:
- parser fixture with known truncated export stores partial reason.
- daemon/CLI read with `limit` says read-limited while source remains complete.
- context pack includes completeness caveat.
- complete sessions remain uncluttered/complete.

Packet: `task_packets/083_polylogue_4ts_6.md`

## 084. polylogue-gjg.1 — compaction_events + compaction_loss_items derived tables; identity survives rebuild + re-ingest

**P2 / task / 03-lineage-compaction-truth / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Promote compaction from a session_events count + lineage edge to an archived object. New index.db tables: compaction_events (boundary message pointers, lineage link fields, trigger/pre_tokens/preserved_segment from the harness event, snapshot_ref + snapshot_source + snapshot_confidence, degraded_reasons) and compaction_loss_items (tier, canonical item key, retained/lost/transformed/unknown classification, pre/post/later-reference anchors, decomposed scores). Keep the session_events row as the compat index row. Ide… Design direction: Derived tier: edit canonical DDL + bump INDEX_SCHEMA_VERSION, batch with the next index bump window (ma2/4ts.5 rule). Extractor already exists (detect_context_compaction handles legacy summary + modern compact_boundary with trigger/pre_tokens/preserved_segment); this materializes it. Blocked conceptually by 4ts.5 (boundary-range columns) which gjg already depends on — coordinate the two in one index bump.

Implementation spine:
- Derived tier: edit canonical DDL + bump INDEX_SCHEMA_VERSION, batch with the next index bump window (ma2/4ts.5 rule).
- Extractor already exists (detect_context_compaction handles legacy summary + modern compact_boundary with trigger/pre_tokens/preserved_segment)
- this materializes it.
- Blocked conceptually by 4ts.5 (boundary-range columns) which gjg already depends on — coordinate the two in one index bump.

Tests:
- Acceptance proof: Rebuild from source produces identical compaction_ids
- Acceptance proof: a re-ingested session keeps its compaction rows
- Acceptance proof: changed interpretation surfaces as event_content_hash delta not silent overwrite.
- Acceptance proof: Verify: fixture tests over legacy+modern Claude compactions + Codex compacted records.

Packet: `task_packets/084_polylogue_gjg_1.md`

## 085. polylogue-gjg.2 — Pre-compaction snapshot capture: hook payload when available, manifest-of-refs otherwise, honesty ladder always

**P2 / task / 03-lineage-compaction-truth / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Two-level snapshotting with snapshot_source as a FIRST-CLASS honesty axis, not a footnote: precompact-hook (strongest — the actual assembled context payload, blob-stored content-addressed, claim: this WAS model context) > jsonl-boundary (manifest of composed-transcript message refs up to the boundary; claim limited to archive-composed transcript, NOT model context) > reconstructed-composed-context (weakest) > none (epidemiology only). Do not store duplicated text when a manifest of message/block/blob refs suffices… Design direction: PreCompact hook wiring rides d1y (hooks install — existing gjg dependency). VERIFY the current Claude Code PreCompact payload actually carries assembled context before promising the strongest rung (known open question; the hook catalog moves). Codex equivalent via app-server events is ox0 territory. Blob dedup makes repeated compactions near-free; source-tier hook event row (raw_hook_events exists) links the blob.

Implementation spine:
- PreCompact hook wiring rides d1y (hooks install — existing gjg dependency).
- VERIFY the current Claude Code PreCompact payload actually carries assembled context before promising the strongest rung (known open question
- the hook catalog moves).
- Codex equivalent via app-server events is ox0 territory.
- Blob dedup makes repeated compactions near-free

Tests:
- Acceptance proof: A live compaction on the operator machine lands either a hook snapshot or a labeled jsonl-boundary manifest
- Acceptance proof: every snapshot row carries source+confidence
- Acceptance proof: no unlabeled reconstruction.
- Acceptance proof: Verify: live dogfood compaction + fixture for the fallback.

Packet: `task_packets/085_polylogue_gjg_2.md`

## 086. polylogue-gjg.3 — Deterministic loss-forensics: 4-tier structural diff + lost-but-later-needed ranking

**P2 / task / 03-lineage-compaction-truth / blocked-hard**

Mechanism: Issue description localizes the mechanism: The base retained/lost/transformed classifier is deterministic and structural — NO LLM in the base pass (LLM annotation may layer later as separate judgment rows). Four item tiers with canonical keys: file-path (normalized against repo/cwd), tool-outcome (from the actions view; failed outcomes weighted high — losing a failure record is how agents repeat mistakes), marked-decision (assertions kind decision/lesson/caveat/blocker/handoff + 37t.2 inline markers; ranks highest — losing these is how settled debates reop… Design direction: Registered as a 9l5.7 measure (compaction-loss) with tier=structural and coverage gates; epidemiology = plain relation algebra over the two tables (rate by provider/trigger/session-length bucket, marked-decision loss rate, failed-tool-outcome loss rate, snapshot-coverage rate). Eager event materialization, lazy loss-item computation on first read, then cached — compaction forensics on a 38GB archive must not run at …

Implementation spine:
- Registered as a 9l5.7 measure (compaction-loss) with tier=structural and coverage gates
- epidemiology = plain relation algebra over the two tables (rate by provider/trigger/session-length bucket, marked-decision loss rate, failed-tool-outcome loss rate, snapshot-coverage rate).
- Eager event materialization, lazy loss-item computation on first read, then cached — compaction forensics on a 38GB archive must not run at ingest.
- Honest degradation: every item carries degraded_reasons
- unknown never folded into denominators.

Tests:
- Acceptance proof: Classifier is pure + property-tested (same inputs => same items)
- Acceptance proof: ranking exposes per-component scores
- Acceptance proof: epidemiology query renders with n + coverage footnotes.
- Acceptance proof: Verify: fixtures with known-lost items + measure-registry gate test.

Packet: `task_packets/086_polylogue_gjg_3.md`

## 087. polylogue-h6r — Agent identity: a stable who-did-this tuple for every session

**P2 / task / 03-lineage-compaction-truth / blocked-hard**

Mechanism: Issue description localizes the mechanism: Half the analytics tower assumes 'per agent' partitions the schema cannot express: a model name is not an agent — the same model under different CLAUDE.md versions, skills, or MCP profiles is behaviorally a different worker. Calibration (h10), setup A/B, advisory tuning, and the evaluation instrument all need a stable agent-identity key or they average across regime changes and mislead exactly when the setup changes (which is when you look). Design direction: agent_identity = (model_name, harness+version, config_state_ref, role) materialized per session as a derived column set: model/harness from session metadata (exists), config_state_ref from the 7aw config-artifact joins (the hash of the config set in force at session start), role from subagent dispatch context where present. Store as columns on session_profiles (the hot-row home) + a registry of observed identities w…

Implementation spine:
- agent_identity = (model_name, harness+version, config_state_ref, role) materialized per session as a derived column set: model/harness from session metadata (exists), config_state_ref from the 7aw config-artifact joins (the hash of the config set in force at session start), role from subagent dispatch context where present.
- Store as columns on session_profiles (the hot-row home) + a registry of observed identities with first/last seen.
- Measures gain identity as a grouping unit ('measure X by agent').
- Degrades honestly: sessions predating config capture get config_state_ref=unknown, and identity-partitioned measures state the unknown fraction.
- Small bead, foundational — blocks h10's per-agent calibration claims.

Tests:
- Acceptance proof: Identity tuple materialized for new sessions on the live machine (config ref resolving via 7aw)
- Acceptance proof: an identity-partitioned measure runs with the unknown fraction stated
- Acceptance proof: h10's calibration curves key on identity, not bare model name.

Packet: `task_packets/087_polylogue_h6r.md`

## 088. polylogue-b5l — Blue-green index rebuilds: fresh-first without downtime

**P1 / task / 08-scale-performance-live / blocked-hard**

Mechanism: Current derived-tier reset/rebuild doctrine can produce long degraded windows. Blue-green means building a fresh generation beside the served one and swapping only after convergence proof.

Implementation spine:
- Define index generation metadata: generation id, schema version, source snapshot, build status, replay cursor, ready flag, active pointer.
- Build new `index.db` generation out-of-place while serving old generation.
- Replay writes or pause briefly at swap; document exact consistency model.
- Make daemon/web/CLI readiness report old generation served/new generation building, not archive-ready over partial corpus.

Tests:
- Schema bump fixture builds new generation and serves old until ready.
- Crash mid-build leaves old generation active.
- Swap is atomic; post-swap row counts/materialization checks pass.

Packet: `task_packets/088_polylogue_b5l.md`

## 089. polylogue-1xc.8 — Schema rebuild-safety scenario

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Design direction: scenario-coverage.yaml gap 'schema-rebuild-safety' orphaned on gh#590. A scenario proving derived-tier rebuild (index/embeddings) from durable source/user evidence is lossless and idempotent, and durable-tier additive migration preserves user.db assertions. Ties 1xc.7 scale-regression lane + z7rv migration framework.

Implementation spine:
- scenario-coverage.yaml gap 'schema-rebuild-safety' orphaned on gh#590.
- A scenario proving derived-tier rebuild (index/embeddings) from durable source/user evidence is lossless and idempotent, and durable-tier additive migration preserves user.db assertions.
- Ties 1xc.7 scale-regression lane + z7rv migration framework.

Tests:
- Acceptance proof: A rebuild-safety scenario resets a derived tier and rebuilds from source, asserting byte/row parity + no user.db loss
- Acceptance proof: a durable additive migration round-trips behind the backup gate.
- Acceptance proof: Verify: the scenario under devtools lab lanes.

Packet: `task_packets/089_polylogue_1xc_8.md`

## 090. polylogue-20d.15 — Bulk ingest throughput + resource envelope: parallel parse, batched writes, bounded RSS/IO

**P2 / task / 08-scale-performance-live / blocked-hard**

Mechanism: Issue description localizes the mechanism: Live evidence 2026-07-03: the full index rebuild replayed 16,725 raw rows at 12-15 rows/s whole-run (5/s when it hit big sessions) — 20-40 minutes of archive downtime for an operation the fresh-first doctrine treats as routine. Nobody has stated the machine impact budget either: daemon RSS during bulk ingest, write amplification per tier, page-cache pressure, and IO contention with the live desktop are unmeasured-in-anger even though the instruments exist (live_ingest_attempt RSS fields, bench ingest-amplification… Design direction: (1) MEASURE first on a live-archive copy: where do the 12-15 rows/s go (parse vs store vs FTS vs insights — the attempt rows record stage timings); bench ingest-throughput gives the synthetic baseline. (2) PARALLEL PARSE: parsing is CPU-bound JSON; pipeline/services/process_pool.py already provides the safe pool (spawn-context) — fan out parse across N workers, keep the store single-writer (SQLite reality); the para…

Implementation spine:
- (1) MEASURE first on a live-archive copy: where do the 12-15 rows/s go (parse vs store vs FTS vs insights — the attempt rows record stage timings)
- bench ingest-throughput gives the synthetic baseline.
- (2) PARALLEL PARSE: parsing is CPU-bound JSON
- pipeline/services/process_pool.py already provides the safe pool (spawn-context) — fan out parse across N workers, keep the store single-writer (SQLite reality)
- the parallel-parse dogfood branch from 2026-06-29 is prior art to consult.

Tests:
- Acceptance proof: Full replay of a live-archive copy sustains >=100 raw rows/s whole-run on the operator machine and finishes <5 min
- Acceptance proof: rebuild prints live rows/s and ETA.
- Acceptance proof: Ingest RSS stays under the stated cap
- Acceptance proof: bench ingest-amplification shows no per-tier regression
- Acceptance proof: desktop remains responsive during a rebuild (idle IO class verified).

Packet: `task_packets/090_polylogue_20d_15.md`

## 091. polylogue-20d.14 — Interactive SLO tier: named latency budgets, continuously measured, regression-gated

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: 'Snappy' needs numbers or it regresses silently. docs/plans/slo-catalog.yaml + devtools bench slo exist but there is no interactive tier: no stated budget for daemon-served query round trip, completion latency, webui first-paint, cold CLI floor, or ingest-to-searchable lag — and no continuous measurement, so today's evidence (1.6-1.9s CLI floor for EVERYTHING including error paths, 5-9s helps, minutes-stuck facets) had to be discovered by ad-hoc probing rather than read off a dashboard. Design direction: (1) BUDGETS (starting points, tune with evidence): daemon-served query p50<100ms/p95<400ms; completion round trip <50ms (it is keystroke-path); status/facets from cache <30ms; webui first meaningful paint <300ms warm daemon; cold CLI (no daemon) <700ms after 20d.2 import deferral; ingest-to-searchable <10s from JSONL write (measures the whole hook->ingest->FTS->cache-invalidate chain). Add as an 'interactive' tier i…

Implementation spine:
- (1) BUDGETS (starting points, tune with evidence): daemon-served query p50<100ms/p95<400ms
- completion round trip <50ms (it is keystroke-path)
- status/facets from cache <30ms
- webui first meaningful paint <300ms warm daemon
- cold CLI (no daemon) <700ms after 20d.2 import deferral

Tests:
- Acceptance proof: slo-catalog.yaml contains the interactive tier with the stated budgets.
- Acceptance proof: devtools bench slo runs the tier green against the seeded corpus with a live daemon in CI.
- Acceptance proof: /metrics exposes per-route latency histograms.
- Acceptance proof: CLI invocation spans are queryable in ops.db and surfaced by a polylogue analyze latency projection.

Packet: `task_packets/091_polylogue_20d_14.md`

## 092. polylogue-20d.12 — Daemon result cache + post-ingest warming: precomputed answers, cursor-keyed invalidation

**P2 / feature / 08-scale-performance-live / blocked-hard**

Mechanism: Issue description localizes the mechanism: The fast path (20d.1) makes the daemon reachable in milliseconds; this bead makes the daemon WORTH reaching: today every facets/status/aggregate request recomputes from SQLite (live evidence: /api/facets defers repos+action_types by default, was stuck 'loading... stale' for minutes during convergence; bare status re-probes the DB per invocation). A hot daemon should answer the common 80% from memory: facets, status snapshot, recent-session lists, saved-view results, common aggregates — computed once per archive ch… Design direction: (1) CACHE KEY: the archive ingest cursor (ops.db already tracks it) + query fingerprint. A cached entry is valid until the cursor moves — no TTL guessing, no staleness lies; the convergence snapshot (4bu) rides the same key. (2) WRITE-TRIGGERED RECOMPUTE: after each ingest batch commits, the daemon refreshes the hot set in its idle loop (facets complete families INCLUDING the deferred ones, status snapshot, newest-s…

Implementation spine:
- (1) CACHE KEY: the archive ingest cursor (ops.db already tracks it) + query fingerprint.
- A cached entry is valid until the cursor moves — no TTL guessing, no staleness lies
- the convergence snapshot (4bu) rides the same key.
- (2) WRITE-TRIGGERED RECOMPUTE: after each ingest batch commits, the daemon refreshes the hot set in its idle loop (facets complete families INCLUDING the deferred ones, status snapshot, newest-sessions page, saved views marked hot) — the webui then never waits on facets
- it reads the precomputed payload.

Tests:
- Acceptance proof: bench slo (interactive tier): cached facets/status p50 <30ms on the seeded corpus with warm daemon.
- Acceptance proof: Cache entries invalidate within one ingest batch of a cursor move (test: ingest a session, facets reflect it next request).
- Acceptance proof: /metrics exposes cache hit/miss/size
- Acceptance proof: memory stays under the configured cap under a 10k-query soak.

Packet: `task_packets/092_polylogue_20d_12.md`

## 093. polylogue-20d.13 — Daemon push channel: SSE events for live UIs instead of polling

**P2 / feature / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Everything live currently polls: the webui polls for facets/status, live tailing (bby.4) would poll, CLI watch modes would poll. The daemon already HAS the event stream internally (ingest events, convergence stage events, daemon_events.db) — it just has no push transport. One SSE endpoint turns the daemon from a request-answering server into a live substrate: session-ingested, session-updated, convergence-state-changed, cache-invalidated events pushed to subscribers. Design direction: (1) TRANSPORT: Server-Sent Events over the existing HTTP server — chunked responses work on BaseHTTPRequestHandler (one long-lived thread per subscriber; cap subscribers, loopback-only), so this is NOT blocked on the ASGI decision (dx1) — though if dx1 migrates, SSE gets cheaper; note the thread cost in the dx1 evidence. (2) EVENT VOCABULARY (small, versioned): archive.cursor_moved {cursor, sessions_delta}, session.…

Implementation spine:
- (1) TRANSPORT: Server-Sent Events over the existing HTTP server — chunked responses work on BaseHTTPRequestHandler (one long-lived thread per subscriber
- cap subscribers, loopback-only), so this is NOT blocked on the ASGI decision (dx1) — though if dx1 migrates, SSE gets cheaper
- note the thread cost in the dx1 evidence.
- (2) EVENT VOCABULARY (small, versioned): archive.cursor_moved {cursor, sessions_delta}, session.ingested {ref, origin}, session.updated {ref}, convergence.state {snapshot payload from 4bu}, cache.invalidated {scope}.
- Source from the existing daemon event plumbing — no new bus, expose the one that exists.

Tests:
- Acceptance proof: A subscribed browser receives session.ingested within 2s of ingest commit on the seeded corpus.
- Acceptance proof: Reconnect with Last-Event-ID replays missed events from the ring buffer.
- Acceptance proof: The workbench converging banner updates without page reload.
- Acceptance proof: Subscriber cap enforced
- Acceptance proof: endpoint loopback-only.

Packet: `task_packets/093_polylogue_20d_13.md`

## 094. polylogue-20d.6 — Live full-ingest catch-up latency + WAL shape

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: 0.2 files/s full-ingest chunks; parse_s ~274s for 50 small files. Recent daemon backoff commits (no-op retry/catch-up chunks, filtered retry paths) address parts — re-measure before working. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Live evidence (gh#2391): full-ingest chunks ~0.2 files/s; 50 small files -> parse_s ~274s while convergence <2s; WAL ballooned during a 50-file chunk. Recent daemon backoff commits changed the shape — RE-MEASURE first (bounded catch-up + stage timings + ops diagnostics workload before/after). Related invariant to keep verified: full-replace re-ingest rewrites all messages in one transaction (correct for idempotency)…

Implementation spine:
- Live evidence (gh#2391): full-ingest chunks ~0.2 files/s
- 50 small files -> parse_s ~274s while convergence <2s
- WAL ballooned during a 50-file chunk.
- Recent daemon backoff commits changed the shape — RE-MEASURE first (bounded catch-up + stage timings + ops diagnostics workload before/after).
- Related invariant to keep verified: full-replace re-ingest rewrites all messages in one transaction (correct for idempotency) — for live-tailed long sessions the append path (sources/live/append_ingest.py) must stay the hot route

Tests:
- Acceptance proof: RE-MEASURE first (recent daemon backoff commits changed the shape): bounded catch-up run + stage timings + `polylogue ops diagnostics workload` before/after are captured and the baseline recorded.
- Acceptance proof: The idempotency invariant is kept verified — full-replace re-ingest rewrites all messages in one transaction — while for live-tailed long sessions the append path (sources/live/append_ingest.py) stays the hot route
- Acceptance proof: `devtools bench ingest-amplification` on real tails is wired as a scheduled check to catch append-vs-full-replace regressions.
- Acceptance proof: End-to-end ingest-to-searchable latency is measured with a synthetic session write on the seeded corpus (chain: hook/watcher debounce -> parse -> store -> FTS -> cache invalidation (20d.12) -> SSE announce (20d.13))
- Acceptance proof: a session appears in find/webui within the ~10s interactive SLO budget (20d.14).

Packet: `task_packets/094_polylogue_20d_6.md`

## 095. polylogue-20d.5 — Finish streaming reads: composed transcripts, messages --full writer, origin-filtered pagination SQL

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Residue of the streaming-export slice: lineage-composed transcript streaming falls back to the eager path; read --view messages --full --to file lacks a true writer/iterator renderer; material-origin-filtered pagination is eager until SQL owns the predicate. Design direction: Three eager fallbacks to close (prior-audit evidence, re-locate): (1) lineage-composed transcript streaming falls back to the eager path — extend the streaming writer landed in a9dc3f274 to composed (parent-prefix + tail) reads; (2) read --view messages --full --to file lacks a true writer/iterator renderer — same pattern; (3) material-origin-filtered message pagination hydrates eagerly until SQL owns the predicate …

Implementation spine:
- Three eager fallbacks to close (prior-audit evidence, re-locate): (1) lineage-composed transcript streaming falls back to the eager path — extend the streaming writer landed in a9dc3f274 to composed (parent-prefix + tail) reads
- (2) read --view messages --full --to file lacks a true writer/iterator renderer — same pattern
- (3) material-origin-filtered message pagination hydrates eagerly until SQL owns the predicate — push material_origin into the repository pagination SQL (pattern: a17e3af95 routed ordinary paginated reads through repository pagination).
- Verify each with a live-archive file export timing + RSS bound, plus focused unit tests on the streaming/pagination modules.

Tests:
- Acceptance proof: Lineage-composed transcript streaming uses the streaming writer (extend the a9dc3f274 pattern) for composed (parent-prefix + tail) reads — no eager full-materialization fallback remains (grep the composed read path).
- Acceptance proof: `read --view messages --full --to <file>` uses a true iterator/writer renderer rather than eager buffering.
- Acceptance proof: Material-origin-filtered message pagination pushes `material_origin` into the repository pagination SQL (pattern a17e3af95)
- Acceptance proof: hydration no longer filters in Python.
- Acceptance proof: Each of the three is verified with a live-archive file export showing bounded peak RSS (flat vs message count) with export timing recorded, plus focused unit tests on the streaming/pagination modules (`devtools test <streaming/pagination modules>` green).

Packet: `task_packets/095_polylogue_20d_5.md`

## 096. polylogue-20d.2 — Defer heavy imports off the CLI startup path

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: ~2s import tax per invocation; also the residual cold cost when the daemon path is absent. Candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive. Measure first: python -X importtime -c 'from polylogue.cli.click_app import main'. Covers the old help-latency and find-select-cold items; add the help-latency devtools budget check as the regression gate. Design direction: Measure first: python -X importtime -c 'from polylogue.cli.click_app import main' 2>&1 | sort -t'|' -k2 -rn | head -30. Known heavy candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive, storage imports pulled at command-module import time. Mechanics: the repo already uses lazy Click commands (see bd memory: lazy cmds hide flags — use cmd.get_params(ctx) in tests); push heavy impor…

Implementation spine:
- Measure first: python -X importtime -c 'from polylogue.cli.click_app import main' 2>&1 | sort -t'|' -k2 -rn | head -30.
- Known heavy candidates: surfaces/payloads (~2,915 lines of Pydantic model construction), api/archive, storage imports pulled at command-module import time.
- Mechanics: the repo already uses lazy Click commands (see bd memory: lazy cmds hide flags — use cmd.get_params(ctx) in tests)
- push heavy imports inside command bodies / module __getattr__
- keep a leaf path-resolution module import-light for the daemon fast-path handshake.

Tests:
- Acceptance proof: `python -X importtime -c 'from polylogue.cli.click_app import main'` shows surfaces/payloads and api/archive no longer imported on the `polylogue <cmd> --help` path.
- Acceptance proof: Verify: importtime diff before/after.
- Acceptance proof: A new devtools help-latency budget check runs targeted `polylogue <cmd> --help` invocations under a fixed budget (e.g.
- Acceptance proof: <700ms cold, citing the 20d.14 cold-CLI budget) and fails loudly on drift.
- Acceptance proof: Nested helps (import / reset / maintenance archive-read / analyze tools) drop from the observed 5-9s to under the budget.

Packet: `task_packets/096_polylogue_20d_2.md`

## 097. polylogue-20d.1 — CLI->daemon fast path over UDS (persistent hot process)

**P2 / feature / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Route CLI queries through the already-hot daemon when available: skips import cost, warm SQLite page cache, shared readiness state. Silent in-process fallback. Design direction: Precedent: fast-status path (commands/status.py:950, click_app.py:214-221) already prefers the daemon — extend the pattern to the whole read surface. Transport: UDS at $XDG_RUNTIME_DIR/polylogue/daemon.sock (TCP stays for the browser); AF_UNIX HTTPServer subclass ~20 lines; instant-fail when down. Probe: socket exists -> connect (fails in microseconds) -> GET /api/health with 100ms budget; health payload carries {ar…

Implementation spine:
- Precedent: fast-status path (commands/status.py:950, click_app.py:214-221) already prefers the daemon — extend the pattern to the whole read surface.
- Transport: UDS at $XDG_RUNTIME_DIR/polylogue/daemon.sock (TCP stays for the browser)
- AF_UNIX HTTPServer subclass ~20 lines
- instant-fail when down.
- Probe: socket exists -> connect (fails in microseconds) -> GET /api/health with 100ms budget

Tests:
- Acceptance proof: Fast-path read surface: `--verbose` prints `served-by: daemon (uds, <ms>)` and a warm daemon serves find/read/messages/facets within the 20d.14 interactive-tier budget (target 3.6-17s -> 0.3-0.5s wall).
- Acceptance proof: Verify: timed CLI run against a warm daemon
- Acceptance proof: `devtools bench slo` interactive tier green.
- Acceptance proof: Golden parity: `--format json` output is byte-identical between direct and daemon-proxied execution for every read surface on the demo corpus.
- Acceptance proof: Verify: pytest golden-parity test.

Packet: `task_packets/097_polylogue_20d_1.md`

## 098. polylogue-20d.10 — Runtime post-filter efficiency: memoize semantic facts; lower matchers onto the actions view

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: matches_action_sequence, matches_referenced_path, and category matching each call _actions_for(session) -> build_session_semantic_facts (runtime_matching.py:20-25) — full semantic-fact construction over a hydrated session, no memoization across the three matchers, applied as list-comprehension post-filter (runtime_filters.py:188-189). A broad query with SEQ or referenced_path hydrates every SQL-surviving candidate and builds facts up to 3x. Design direction: Minimal fix: memoize facts per session within a filter pass (functools cache keyed per pass, or attach _semantic_facts to the Session object). Real fix: all three matchers' predicates (action category, affected path, sequence) are answerable from actions-view rows — fetch once per candidate set with a single WHERE session_id IN (...) query, group in Python, drop hydration entirely for candidates failing cheap predic…

Implementation spine:
- Minimal fix: memoize facts per session within a filter pass (functools cache keyed per pass, or attach _semantic_facts to the Session object).
- Real fix: all three matchers' predicates (action category, affected path, sequence) are answerable from actions-view rows — fetch once per candidate set with a single WHERE session_id IN (...) query, group in Python, drop hydration entirely for candidates failing cheap predicates.
- The keystone columns (v16) and idx_blocks_type_tool (v20) exist for exactly this shape.
- Also push cheap structured clauses into SQL before hydration.
- SEQ span capture (DSL bead) builds on the same relation — coordinate.

Tests:
- Acceptance proof: 1.
- Acceptance proof: Minimal fix: semantic facts are memoized per session within a single filter pass (no more than one build_session_semantic_facts per session per pass), eliminating the up-to-3x construction across matches_action_sequence / matches_referenced_path / category matching (runtime_matching.py, runtime_filters.py).
- Acceptance proof: 2.
- Acceptance proof: Real fix: the three matchers' predicates (action category, affected path, sequence) are answered from actions-view rows fetched once per candidate set with a single `WHERE session_id IN (...)` query, grouped in Python
- Acceptance proof: candidates failing cheap predicates are dropped before hydration, and cheap structured clauses are pushed into SQL before hydration.

Packet: `task_packets/098_polylogue_20d_10.md`

## 099. polylogue-2qx — OriginSpec: one package per origin, dispatch order derived from declared strictness

**P2 / feature / 11-interoperability-origin / blocked-hard**

Mechanism: Origin dispatch is spread across detectors, parser bases, provider completeness, and preflight. The system needs one explicit OriginSpec per origin to avoid ambiguous importer behavior.

Implementation spine:
- Create OriginSpec with id, detector, strictness, parser, fixture set, normalized schema mapping, source material policy, fidelity declaration, docs link.
- Generate dispatch order from strictness/specificity instead of hand-coded order.
- Make preflight explain which specs matched, which lost, and why.
- Backfill existing ChatGPT/Claude/Codex/Gemini/Antigravity/Hermes origins into specs.

Tests:
- Ambiguous fixture matches deterministic most-specific origin.
- Each OriginSpec has raw and normalized fixtures.
- Preflight output is actionable for unknown/format-drift files.

Packet: `task_packets/099_polylogue_2qx.md`

## 100. polylogue-o21 — Extension-point ergonomics: declare-once registries, scaffolds, actionable completeness errors

**P2 / feature / 11-interoperability-origin / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Every extension today is a scavenger hunt across parallel registration sites, each failing opaquely when missed — the accumulated tribal knowledge lives in bd memories: a new MCP tool needs EXPECTED_TOOL_NAMES + TOOL_CONTRACT + role gating + render openapi + render cli-output-schemas (four separate opaque failures); a new golden-path workflow must be in REQUIRED_WORKFLOW_IDS or CLI startup crashes with an unrelated error; a new AssertionKind breaks two renders plus the every-kind-has-a-surface test; a new module f… Design direction: Three legs, applied per extension point (MCP tool, CLI verb/command, DSL unit/stage, insight, origin, assertion kind, devtools command, workflow): (1) DECLARE-ONCE: each point gets a single declaration object carrying ALL metadata the parallel sites currently hold (name, contract, role gating, schema, docs blurb, owning surface) — the parallel lists become derivations: EXPECTED_TOOL_NAMES is generated FROM tool decl…

Implementation spine:
- Three legs, applied per extension point (MCP tool, CLI verb/command, DSL unit/stage, insight, origin, assertion kind, devtools command, workflow): (1) DECLARE-ONCE: each point gets a single declaration object carrying ALL metadata the parallel sites currently hold (name, contract, role gating, schema, docs blurb, owning surface) — the parallel lists become derivations: EXPECTED_TOOL_NAMES is generated FROM tool de…
- Where a hard second site must remain (generated OpenAPI), the deriver owns it.
- (2) ACTIONABLE ERRORS: every registration validator, when it fails, names the missing step and the command that fixes it ('assertion kind X has no surface entry: add to user_audit surface map at <path>
- then run devtools render openapi') — turn the four opaque failures into one checklist error.
- The registration-traps bd memory becomes obsolete BY CONSTRUCTION, which is the acceptance test: a new agent adds a tool end-to-end without the memory.

Tests:
- Acceptance proof: Slice 1 (size:S, unblocks dependents): a DeclarationSpec dataclass + registry protocol are defined and one pilot extension point (MCP tools) is migrated to declare-once, with a published pattern doc
- Acceptance proof: dependents can build against the protocol immediately.
- Acceptance proof: DECLARE-ONCE (pilot): EXPECTED_TOOL_NAMES is derived FROM the tool declarations (grep shows the parallel list is a derivation, not hand-maintained)
- Acceptance proof: where a hard second site remains (generated OpenAPI) the deriver owns it.
- Acceptance proof: ACTIONABLE ERRORS: the MCP-tool registration validator, on failure, names the missing step and the exact fixing command (e.g.

Packet: `task_packets/100_polylogue_o21.md`

## 101. polylogue-da1 — Provider format-drift sentinel: detect upstream export-shape changes from live ingest

**P2 / feature / 11-interoperability-origin / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Claude Code, Codex, ChatGPT, and Gemini change their export/JSONL shapes without notice. Today drift surfaces as silent parse degradation — dropped fields or nodes discovered manually weeks later (e.g. the ChatGPT asset-only-node and Antigravity non-UTF-8 drops in polylogue-qda). Nothing watches for new-unseen-shape rates in live ingest. A sentinel turns format drift from a forensic discovery into a daemon health signal. Design direction: Reuse the existing schema-inference machinery (schemas/ shape signatures) rather than building a new detector: at ingest, count records whose shape does not match the committed provider schema package, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry (the ops tier explicitly allows additive columns — no schema bump). Daemon health check + 'polylogue ops status' line: 'origin X: N% of reco…

Implementation spine:
- Reuse the existing schema-inference machinery (schemas/ shape signatures) rather than building a new detector: at ingest, count records whose shape does not match the committed provider schema package, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry (the ops tier explicitly allows additive columns — no schema bump).
- Daemon health check + 'polylogue ops status' line: 'origin X: N% of records since <date> carry unseen shapes', with bounded example native_ids.
- The follow-up action stays 'devtools lab schema generate/promote' — the sentinel only detects and points.
- Pitfalls: (1) never fail or skip ingest on drift — raw payloads are stored, so parsing is always redoable after a parser update
- that is the payoff of the fresh-first doctrine.

Tests:
- Acceptance proof: At ingest, records whose shape does not match the committed provider schema package are counted, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry via additive columns only (no schema bump — the ops tier explicitly allows additive columns).
- Acceptance proof: Ingest never fails or skips on drift: a drift-shaped record still ingests and its raw payload is stored (parsing is redoable after a parser update) — verified by a test feeding an unseen-shape record.
- Acceptance proof: The detector distinguishes benign 'new optional payload field' from risky 'known field disappeared / type changed', with different alert thresholds.
- Acceptance proof: The rate is windowed since a date (not lifetime) so old archives do not dilute the signal.
- Acceptance proof: A daemon health check + `polylogue ops status` line reads 'origin X: N% of records since <date> carry unseen shapes' with bounded example native_ids

Packet: `task_packets/101_polylogue_da1.md`

## 102. polylogue-7aw — Ingest agent configuration as a source family (skills, CLAUDE.md, hooks)

**P2 / feature / 11-interoperability-origin / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Treat agent configuration as a corpus polylogue versions, queries, and correlates with session outcomes: CLAUDE.md/AGENTS.md revisions, skills, hook configs — config-over-time x outcome-over-time is the continual-learning dataset (composes with the self-experimentation rail). New source family with real design work (identity, content-hash versioning, correlation keys to sessions via repo+time). Git history already carries much of it — the parser may be a git-log walker rather than a file watcher. Design direction: FULL SCOPE (upgraded from notes 2026-07-03; operator: we should have full control/understanding of CLAUDE.md regardless of replacement ambitions): (1) CAPTURE: CLAUDE.md files (global dots/claude tree incl. @-included world-model/operational files, per-project CLAUDE.md, rendered AGENTS.md), skills (~/.claude/skills + repo skills), hooks configs, and settings.json families ingest as a config-artifact source family —…

Implementation spine:
- FULL SCOPE (upgraded from notes 2026-07-03
- operator: we should have full control/understanding of CLAUDE.md regardless of replacement ambitions): (1) CAPTURE: CLAUDE.md files (global dots/claude tree incl.
- @-included world-model/operational files, per-project CLAUDE.md, rendered AGENTS.md), skills (~/.claude/skills + repo skills), hooks configs, and settings.json families ingest as a config-artifact source family — versioned by content hash, timestamped, watcher-covered (they change via sinnix commits AND live edits).
- Each session then joins to the CONFIG STATE it ran under (session_agent_policies table exists — verify shape, extend to reference config-artifact hashes): 'which rules were in force when this session ran' becomes queryable, which is the precondition for every claim about instruction efficacy.
- (2) UNDERSTAND: skill-invocation tracking (Skill tool calls are structural blocks) -> which skills fire, how often, with what outcomes

Tests:
- Acceptance proof: Config artifacts ingest with content-hash versioning and watcher coverage on the live machine
- Acceptance proof: a session from last week resolves to the exact CLAUDE.md/skill versions it ran under
- Acceptance proof: skill-invocation report renders (which skills, frequency, outcome mix)
- Acceptance proof: one state-like global-CLAUDE.md section migrated to injected-equivalent with drill-verified parity and the static text retired
- Acceptance proof: the classification of the operator's current global CLAUDE.md into the four content classes is committed as the migration map.

Packet: `task_packets/102_polylogue_7aw.md`

## 103. polylogue-ox0 — Codex deep integration: state DBs as authoritative source + AppServer live lane

**P2 / task / 11-interoperability-origin / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Codex writes far more than rollout JSONL: state_5.sqlite (authoritative session/thread state — already used read-only by cost reconciliation lpl), goals_1.sqlite (goal/plan state — unexplored), history.jsonl, hooks.json, prompts/, rules/, skills/ (agent-config, 7aw), shell_snapshots/. And Codex ships an APP-SERVER: a JSON-RPC interface (used by IDE integrations) exposing thread lifecycle, streamed turn events, approval requests, and programmatic session control — which makes it BOTH a richer capture channel (live … Design direction: Three sub-lanes, sequenced: (1) STATE-DB IMPORTER (mirror of fs1.1): state_5.sqlite as authoritative Codex source — threads, turns, token counters with the disjoint-lane semantics already encoded in memory, parent/fork relations for lineage (4ts cross-check); goals_1.sqlite explored and mapped (VERIFY schema — undocumented); rollout JSONL demotes to fallback. Read via the copy-to-scratch discipline (live-locked DB).…

Implementation spine:
- Three sub-lanes, sequenced: (1) STATE-DB IMPORTER (mirror of fs1.1): state_5.sqlite as authoritative Codex source — threads, turns, token counters with the disjoint-lane semantics already encoded in memory, parent/fork relations for lineage (4ts cross-check)
- goals_1.sqlite explored and mapped (VERIFY schema — undocumented)
- rollout JSONL demotes to fallback.
- Read via the copy-to-scratch discipline (live-locked DB).
- (2) APP-SERVER CAPTURE: a daemon-side client subscribing to the event stream during live Codex sessions -> event-granularity ingest (approvals, exec begin/end, plan updates — the Codex analogue of hook events)

Tests:
- Acceptance proof: state_5.sqlite importer materializes threads/turns/lineage on the live machine and reconciles with existing rollout-derived sessions (dedup by content, no double-count — token totals match lpl reconciliation)
- Acceptance proof: goals_1.sqlite mapped with a written schema note
- Acceptance proof: app-server protocol spike artifact committed (capabilities, version, event vocabulary) with go/no-go for lanes 2 and 3.

Packet: `task_packets/103_polylogue_ox0.md`

## 104. polylogue-t0p — Capture the rest of Claude Code: todos, file-history, prompt history, debug artifacts

**P2 / task / 11-interoperability-origin / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Session JSONL is one artifact among many that Claude Code writes, and the others answer questions transcripts cannot: ~/.claude/todos/*.json = the agent's live PLAN state per session (task lists with status — plan-vs-execution comparison becomes structural); file-history/ = pre-edit file snapshots (ground truth for the yrx changes view, catches what tool-log reconstruction misses); history.jsonl = the operator's prompt history across sessions (paste-detection and prompt-reuse analytics); history-summaries/, debug/… Design direction: Priority order by evidence value: (1) TODOS: small JSON, session-keyed, trivially parsed -> new artifact kind + a session-linked read model (plan items with status transitions when multiple snapshots exist); analytics: plan-completion rate by session outcome (a construct-valid 'did it do what it planned' measure — feeds claim-vs-evidence). (2) FILE-HISTORY: content-addressed capture into the blob store keyed to sess…

Implementation spine:
- Priority order by evidence value: (1) TODOS: small JSON, session-keyed, trivially parsed -> new artifact kind + a session-linked read model (plan items with status transitions when multiple snapshots exist)
- analytics: plan-completion rate by session outcome (a construct-valid 'did it do what it planned' measure — feeds claim-vs-evidence).
- (2) FILE-HISTORY: content-addressed capture into the blob store keyed to session+path (dedup makes this cheap)
- yrx gains a ground-truth lane (diff reconstruction cross-checked against actual snapshots — discrepancy is itself a finding).
- (3) PROMPT HISTORY: ingest as operator-authored evidence rows (privacy note: already local, same tier as transcripts).

Tests:
- Acceptance proof: Todos and file-history ingest end-to-end from the live machine into artifact kinds with provenance
- Acceptance proof: plan-vs-outcome measure registered (9l5.7) with tier=structural
- Acceptance proof: yrx cross-check lane reports agreement rate between reconstructed and snapshot diffs
- Acceptance proof: watcher covers the new roots
- Acceptance proof: fidelity declared per artifact kind.

Packet: `task_packets/104_polylogue_t0p.md`

## 105. polylogue-fs1.4 — Report: polylogue forensics for Hermes sessions

**P2 / feature / 11-interoperability-origin / needs-acceptance-criteria**

Mechanism: Issue description localizes the mechanism: Five-section per-session/per-corpus report, computed from the canonical archive (composition over existing primitives where possible): 1) session topology — parents, resumes, compactions, subagents, branches, long turns; 2) LLM/request economy — token lanes, cost, retry/fallback causes, model/provider shifts, cache-read amplification; 3) tool execution profile — durations, failures, approvals, repeated calls, parallel groups; 4) failure patterns — loops, stalls, empty-response retries, repeated shell failures, tru… Design direction: Composition first: sections 1-3 and most of 4 should lower onto existing primitives — get_session_topology/logical session (topology), session_provider_usage_events + cost rollups (economy), actions/tool timing (tool profile), pathology detectors + structural outcomes (failure patterns), session_commits/git correlation (footprint). Only add new detectors where Hermes-specific (loop detection over repeated identical …

Implementation spine:
- Composition first: sections 1-3 and most of 4 should lower onto existing primitives — get_session_topology/logical session (topology), session_provider_usage_events + cost rollups (economy), actions/tool timing (tool profile), pathology detectors + structural outcomes (failure patterns), session_commits/git correlation (footprint).
- Only add new detectors where Hermes-specific (loop detection over repeated identical tool calls
- stall = long gap between spans
- reasoning burn = reasoning-token share per turn).
- Surface: a named read view/report profile (`polylogue forensics hermes --session <id>` or read --view forensics), rendered markdown + JSON.

Tests:
- OriginSpec fixture coverage: match, ambiguous, unknown, drift.
- Idempotent import/export or parse re-run test.

Packet: `task_packets/105_polylogue_fs1_4.md`

## 106. polylogue-mhx.1 — Provider abstraction: one OpenAI-compatible embedding client, local and cloud, model registry in meta

**P2 / feature / 09-embeddings-retrieval / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Voyage is hardcoded (VOYAGE_API_URL/DEFAULT_MODEL/DEFAULT_DIMENSION constants). Generality is one abstraction away: an OpenAI-compatible /v1/embeddings client pointed at a configurable base_url covers OpenAI, Voyage (has an OpenAI-compatible surface), and every local server (Ollama, llama.cpp, Infinity, LM Studio) — and the operator already runs a LiteLLM gateway at 127.0.0.1:4000 that bridges all of them, so local models need zero new protocol code. Design direction: (1) Config: [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute + batch size; keep the native Voyage path as one provider preset, default unchanged. (2) Identity: message_embeddings_meta already records model — extend recorded identity to (base_url_host, model, dimension, model_revision?) so a mixed-provenance table is detectable and refused at query time (vectors from different m…

Implementation spine:
- (1) Config: [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute + batch size
- keep the native Voyage path as one provider preset, default unchanged.
- (2) Identity: message_embeddings_meta already records model — extend recorded identity to (base_url_host, model, dimension, model_revision?) so a mixed-provenance table is detectable and refused at query time (vectors from different models must never RRF together silently).
- (3) Dimension handling: vec0 tables are fixed-dim — dimension change = embeddings tier reset + re-embed with `ops embed preflight` cost estimate shown BEFORE any spend
- wire the reset into ops reset --embeddings if not present.

Tests:
- Acceptance proof: 1.
- Acceptance proof: A single OpenAI-compatible /v1/embeddings client replaces the hardcoded Voyage path (sqlite_vec_support.py VOYAGE_API_URL / DEFAULT_MODEL / DEFAULT_DIMENSION constants), driven by [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute / batch_size
- Acceptance proof: the native Voyage preset stays the unchanged default.
- Acceptance proof: 2.
- Acceptance proof: message_embeddings_meta records (base_url_host, model, dimension) and query time REFUSES to RRF vectors of differing model identity rather than mixing silently (test-asserted).

Packet: `task_packets/106_polylogue_mhx_1.md`

## 107. polylogue-mhx.2 — Embedding target policy: what gets a vector, at what granularity, at what cost

**P2 / feature / 09-embeddings-retrieval / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Today exactly one class is embedded: authored prose messages (user/assistant, human/assistant-authored material origin, positive word count — the v21 partial index). That is the right floor but the wrong ceiling: session-level retrieval runs on message vectors (expensive, noisy), and assertions/memory — the content whose retrieval matters most for the context loop — have no vectors at all. Nobody has written down what SHOULD be embedded and why; this bead is that decision plus its implementation. Design direction: Target classes, each with an explicit purpose, source text, and marginal cost: (1) authored prose messages [exists] — purpose: fine-grained hybrid search. (2) SESSION vectors [new]: one vector per session from derived text that already exists (title + profile summary + top-K salient authored lines); purpose: find_similar_sessions and neighbor candidates at 16k-session scale without scanning message vectors; near-zer…

Implementation spine:
- Target classes, each with an explicit purpose, source text, and marginal cost: (1) authored prose messages [exists] — purpose: fine-grained hybrid search.
- (2) SESSION vectors [new]: one vector per session from derived text that already exists (title + profile summary + top-K salient authored lines)
- purpose: find_similar_sessions and neighbor candidates at 16k-session scale without scanning message vectors
- near-zero marginal token cost because the text is already materialized in profiles.
- Storage: session_embeddings vec0 table in embeddings.db keyed by session_id + model identity.

Tests:
- Acceptance proof: 1.
- Acceptance proof: SESSION vectors implemented: a `session_embeddings` vec0 table in embeddings.db keyed by (session_id, model identity), populated from already-materialized derived text (title + profile summary + top-K salient authored lines) at near-zero new token spend
- Acceptance proof: find_similar_sessions / neighbor candidates can run off it.
- Acceptance proof: 2.
- Acceptance proof: ASSERTION vectors implemented: judged/candidate assertion bodies embedded on-write into their own vec0 table.

Packet: `task_packets/107_polylogue_mhx_2.md`

## 108. polylogue-mhx.3 — Retrieval quality eval lane: measure FTS vs vector vs hybrid before believing any of them

**P2 / task / 09-embeddings-retrieval / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Every retrieval default (auto lane resolves lexical; hybrid RRF constants; --semantic promotion) was chosen by taste, not measurement. Before the substrate grows recall legs (context compilation) and storage optimizations (quantization), build the eval that says which lane actually finds the right sessions for realistic operator queries — the same evidence-first discipline as the heuristics benchmark (9e5.9). Design direction: devtools bench retrieval, campaign run/compare pattern: (1) Labeled set: ~50-100 (query -> expected sessions/messages) pairs; bootstrap from real usage — queries the operator/agents ran followed by which session they actually opened (the archive records its own MCP/CLI usage; affordance-usage machinery can mine query->open chains), then hand-verify. Seeded-corpus subset for the public/CI-runnable variant, live-archi…

Implementation spine:
- devtools bench retrieval, campaign run/compare pattern: (1) Labeled set: ~50-100 (query -> expected sessions/messages) pairs
- bootstrap from real usage — queries the operator/agents ran followed by which session they actually opened (the archive records its own MCP/CLI usage
- affordance-usage machinery can mine query->open chains), then hand-verify.
- Seeded-corpus subset for the public/CI-runnable variant, live-archive set for the real decision.
- (2) Contestants: FTS only, vector only, hybrid RRF (current constants), hybrid with 2-3 alternative K constants, and optionally rerank (cross-encoder through the LiteLLM gateway) as a stretch arm.

Tests:
- Acceptance proof: 1.
- Acceptance proof: `devtools bench retrieval` exists and runs FTS-only, vector-only, hybrid-RRF (current constants), and 2-3 alternative-K hybrid arms (rerank arm optional/stretch) over a committed labeled set of >=50 (query -> expected session/message id) pairs
- Acceptance proof: a public seeded-corpus subset is CI-runnable and a live-archive variant drives the real decision.
- Acceptance proof: 2.
- Acceptance proof: It emits recall@5, recall@10, MRR, plus p50/p95 latency and $/1k-queries per arm to a `.local/` campaign artifact

Packet: `task_packets/108_polylogue_mhx_3.md`

## 109. polylogue-mhx.4 — Semantic recall leg in context compilation: the memory actually retrieves

**P2 / feature / 09-embeddings-retrieval / blocked-hard**

Mechanism: Issue description localizes the mechanism: compose_context_preamble and compile_context currently select by explicit refs, recency, and policy — there is no semantic leg, so a judged lesson about 'SQLite WAL contention' never surfaces when the new session starts debugging a WAL issue unless someone remembers it exists. This is the retrieval moment the entire memory thesis needs: relevant judged assertions + similar prior sessions, recalled by meaning, within budget, with refs. Design direction: (1) Query formation: at SessionStart the recall query is cheap context — repo, recent commit subjects, the resumed session's summary (on resume); mid-session (agent-invoked via MCP) it is the agent's stated intent. (2) Retrieval: assertion vectors + session vectors (emb-targets) under a similarity floor + top-K cap; judged/active assertions rank above candidates; recency and repo-match as tiebreakers, all weights vi…

Implementation spine:
- (1) Query formation: at SessionStart the recall query is cheap context — repo, recent commit subjects, the resumed session's summary (on resume)
- mid-session (agent-invoked via MCP) it is the agent's stated intent.
- (2) Retrieval: assertion vectors + session vectors (emb-targets) under a similarity floor + top-K cap
- judged/active assertions rank above candidates
- recency and repo-match as tiebreakers, all weights visible in the payload (no opaque scores — every recalled item carries why: similarity, kind, judgment state, refs).

Tests:
- Acceptance proof: SessionStart recall proposes items through the ContextSource protocol (37t.11) with visible why-fields (similarity, kind, judgment state, refs) — no opaque scores
- Acceptance proof: judged assertions outrank candidates at equal similarity
- Acceptance proof: recall stays within the preamble segment budget with refs-over-bodies
- Acceptance proof: a seeded lesson about a distinctive topic surfaces when a session starts on that topic and does NOT surface on an unrelated repo (both directions tested)
- Acceptance proof: degrades to silent no-op when embeddings are absent/stale.

Packet: `task_packets/109_polylogue_mhx_4.md`

## 110. polylogue-0k6 — Embedding changed-text full-replace regression vs split embeddings.db metadata

**P2 / task / 09-embeddings-retrieval / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Changed-text reindexing for the same message_id needs an explicit full-replace regression against split embeddings.db metadata (index-tier rows cleared, embeddings tier not). Design direction: Step 1 — QUANTIFY on the live archive (fables analysis 9): count sessions whose updated_at_ms postdates embedding_status.last_embedded_at_ms with unchanged message counts — the concrete stale-vector population the original bug produced; record the number in the bead on completion (it doubles as the fix's impact statement). Step 2 — regression: ingest fixture; re-ingest FULL-REPLACE variant with one message body chan…

Implementation spine:
- Step 1 — QUANTIFY on the live archive (fables analysis 9): count sessions whose updated_at_ms postdates embedding_status.last_embedded_at_ms with unchanged message counts — the concrete stale-vector population the original bug produced
- record the number in the bead on completion (it doubles as the fix's impact statement).
- Step 2 — regression: ingest fixture
- re-ingest FULL-REPLACE variant with one message body changed at same position/count
- assert (a) session selected by select_pending_archive_session_window, (b) after re-embed, message_embeddings_meta.content_hash matches the new hash and the old vector row is replaced not duplicated — the split-tier trap is index-tier rows cleared by full replace while embeddings.db metadata persists.

Tests:
- Acceptance proof: 1.
- Acceptance proof: QUANTIFY step recorded: the count of sessions whose index-tier updated_at_ms postdates embedding_status.last_embedded_at_ms at unchanged message count is measured on the live archive and written into the bead as the fix's impact number.
- Acceptance proof: 2.
- Acceptance proof: Regression test: ingest a fixture, re-ingest a FULL-REPLACE variant with one message body changed at the same position/count, and assert (a) the session is re-selected by select_pending_archive_session_window and (b) after re-embed, message_embeddings_meta.content_hash matches the new hash with the old vector row REPLACED, not duplicated (the split-tier trap: index-tier rows cleared by full replace while embedding…
- Acceptance proof: 3.

Packet: `task_packets/110_polylogue_0k6.md`

## 111. polylogue-0ns — Bound archive embedding work within large sessions

**P2 / task / 09-embeddings-retrieval / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Why: while verifying live daemon convergence on 2026-07-04, a forced embedding debt drain could run longer than the outer daemon session window because _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync can process a very large session internally. What needs to be done: make archive embedding resumable/bounded within a single huge session, or have the daemon select message windows instead of whole-session units so automatic catch-up remains … Design direction: Make archive embedding bounded within a single large session so a forced debt drain cannot exceed the daemon window. Root cause: _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync processes a whole session internally. Fix option (a): check the stop-after deadline inside embed_archive_session_sync at message-window granularity and persist a res…

Implementation spine:
- Make archive embedding bounded within a single large session so a forced debt drain cannot exceed the daemon window.
- Root cause: _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync processes a whole session internally.
- Fix option (a): check the stop-after deadline inside embed_archive_session_sync at message-window granularity and persist a resumable position
- or (b) have the daemon select message windows (via select_pending_archive_session_window) instead of whole-session units.
- Files: the daemon embed loop (_embed_archive_sessions_sync / embed_archive_session_sync) and the pending-window selection helper.

Tests:
- Acceptance proof: 1.
- Acceptance proof: embed_archive_session_sync honors _DAEMON_EMBED_STOP_AFTER_SECONDS (or an equivalent deadline) at message-window granularity within one session and records a resumable position, so the next daemon tick continues the same session rather than restarting it.
- Acceptance proof: 2.
- Acceptance proof: Regression test: a synthetic session larger than one embedding window, with the stop-after deadline set below the whole-session cost, produces a partial embed that resumes and completes across ticks with no unbounded single-session run.
- Acceptance proof: Verify via `devtools test` selection on the daemon embed path.

Packet: `task_packets/111_polylogue_0ns.md`

## 112. polylogue-37t.5 — Local embedding lane via OpenAI-compatible provider (LiteLLM gateway)

**P2 / feature / 09-embeddings-retrieval / blocked-hard**

Mechanism: Issue description localizes the mechanism: Voyage is the only embedding provider; a local lane makes semantic search $0 and the whole loop air-gapped — and pairs with the Hermes bridge program for a fully local, zero-cloud stack. Design direction: Seam: VectorProvider protocol + Voyage constants in sqlite_vec_support.py. Config: [embedding] provider='openai-compatible', base_url, model, dimension in polylogue.toml; implement the OpenAI /v1/embeddings client shape once (LiteLLM gateway 127.0.0.1:4000 bridges to Ollama). Dimension: vec0 table is fixed float[1024] and EMBEDDING_DIMENSION asserted in meta CHECK — dimension becomes a tier-init parameter in embeddi…

Implementation spine:
- Seam: VectorProvider protocol + Voyage constants in sqlite_vec_support.py.
- Config: [embedding] provider='openai-compatible', base_url, model, dimension in polylogue.toml
- implement the OpenAI /v1/embeddings client shape once (LiteLLM gateway 127.0.0.1:4000 bridges to Ollama).
- Dimension: vec0 table is fixed float[1024] and EMBEDDING_DIMENSION asserted in meta CHECK — dimension becomes a tier-init parameter in embeddings.db meta
- changing model/dimension => ops reset --embeddings + backfill (tier is designed expensive-rebuild

Tests:
- No-provider mode test.
- Changed-text invalidation test.
- Budget dry-run and large-session bound test.

Packet: `task_packets/112_polylogue_37t_5.md`

## 113. polylogue-1vpm.1 — Delegation derived unit: materializer + query unit + delegation-card projection

**P2 / task / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: First-class delegations rows in index.db (derived, recomputable, extractor-versioned): delegation identity prefers (parent_session_id, tool_use_block_id) — never prompt text (identical prompts are different delegations). Row carries parent/child session+run refs, instruction/result block refs, task_id/tool_id, delegation_kind (subagent|background-agent|sidecar-report|async-task|unknown), harness, subagent_type/model/family, status, link_status (resolved|unresolved|inferred|quarantined), confidence, evidence+artifa…

Implementation spine:
- Register the measure/outcome with evidence tier, denominator, and uncertainty.
- Materialize only after source units are stable.
- Add fixture proving empty/uncovered samples do not become zeros.
- Render caveats in CLI/report/web outputs.

Tests:
- Acceptance proof: Fixtures: Claude Task pair, acompact exclusion, Codex spawn, unresolved child, no false subagent from forked_from_id
- Acceptance proof: delegations where parent.repo:X and status:failed works
- Acceptance proof: card renders bounded (full prompts only under explicit opt-in)
- Acceptance proof: index bump batched.
- Acceptance proof: Verify: unit fixtures + query-unit tests.

Packet: `task_packets/113_polylogue_1vpm_1.md`

## 114. polylogue-1vpm.2 — Episode unit: tables, 4-signal scorer with false-merge floor, assertion-calibrated

**P2 / task / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: episodes / episode_members / episode_edges in index.db. EDGES ARE THE UNIT OF EVIDENCE (member-only storage loses why A attached to B); episode = connected component over eligible edges only. member_set_hash = sha256 of sorted member refs => idempotent re-stitch, scorer version as metadata not identity (same member set = same hypothesis, confidence may change). Members beyond sessions: commit/pr/issue/artifact/raw_event (telemetry can join with no matching AI session). Signals persisted per-edge with contributions…

Implementation spine:
- Register the measure/outcome with evidence tier, denominator, and uncertainty.
- Materialize only after source units are stable.
- Add fixture proving empty/uncovered samples do not become zeros.
- Render caveats in CLI/report/web outputs.

Tests:
- Acceptance proof: Deliberately under-stitches on first corpus (polylogue repo work first — strongest evidence density)
- Acceptance proof: zero candidate-only merges in default render
- Acceptance proof: edge evidence auditable
- Acceptance proof: operator decisions survive rebuild
- Acceptance proof: episodes where member.origin:chatgpt and member.origin:claude-code returns cross-tool episodes.

Packet: `task_packets/114_polylogue_1vpm_2.md`

## 115. polylogue-1vpm.4 — Turn-pair unit with prompt-burst semantics (no double-claimed answers)

**P2 / task / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Per-turn latency/cost/correction-rate needs a prompt->answer relation, and the naive pairing law (each prompt -> MIN(next assistant)) is WRONG: two human messages before one answer both claim it. Corrected design: group consecutive human_authored/operator_command prompts into a PROMPT BURST before the next assistant_authored active-path answer; expose prompt_message_ids, burst_size, answer refs, latency (NULL unless both timestamps), token columns, abandoned=true for trailing unanswered bursts. material_origin adj…

Implementation spine:
- Register the measure/outcome with evidence tier, denominator, and uncertainty.
- Materialize only after source units are stable.
- Add fixture proving empty/uncovered samples do not become zeros.
- Render caveats in CLI/report/web outputs.

Tests:
- Acceptance proof: human->human->assistant yields ONE pair with burst_size=2
- Acceptance proof: tool rows skipped
- Acceptance proof: trailing burst abandoned=true
- Acceptance proof: latency NULL-safe
- Acceptance proof: turn-pairs where answer_model:X works cross-surface.

Packet: `task_packets/115_polylogue_1vpm_4.md`

## 116. polylogue-9l5.7 — Statistics substrate + measure registry: uncertainty primitives with construct-validity metadata

**P2 / feature / 10-analytics-experiments / blocked-hard**

Mechanism: Issue description localizes the mechanism: The keystone of the analytics tower. Today every number the archive emits is a point estimate with no uncertainty, and construct validity is a discipline (footnotes written by hand in campaign reports) rather than a mechanism. Two deliverables: (1) honest statistical primitives available wherever aggregates compose — proportions with Wilson intervals, mean/median/percentiles with n and CI, two-sample comparisons with effect size + test, histogram/ECDF buckets; (2) the MEASURE REGISTRY: every analytic registers a d… Design direction: (1) polylogue/analytics/stats.py: pure functions over sequences — wilson_interval(k,n), quantiles_with_ci (bootstrap or order-statistic CIs), two_proportion_test, mann_whitney (rank test avoids normality assumptions on latency/cost distributions), cliffs_delta effect size, histogram_buckets. scipy.stats behind the [analytics] extra with hand-rolled fallbacks for the handful used in core paths (Wilson and bootstrap a…

Implementation spine:
- (1) polylogue/analytics/stats.py: pure functions over sequences — wilson_interval(k,n), quantiles_with_ci (bootstrap or order-statistic CIs), two_proportion_test, mann_whitney (rank test avoids normality assumptions on latency/cost distributions), cliffs_delta effect size, histogram_buckets.
- scipy.stats behind the [analytics] extra with hand-rolled fallbacks for the handful used in core paths (Wilson and bootstrap are 20 lines each — core stays dependency-lean).
- (2) MeasureSpec (declare-once discipline, o21): name, construct, unit, formula ref, evidence_tier, required_coverage (e.g.
- priced-provenance-only), confounds list, output schema.
- Registered like query units

Tests:
- Acceptance proof: polylogue/analytics/stats.py exists with property tests (hypothesis: interval coverage on synthetic distributions).
- Acceptance proof: At least 5 existing analytics re-registered as MeasureSpecs with tiers.
- Acceptance proof: A cross-origin comparison without coverage labels is refused at composition with an actionable error.
- Acceptance proof: One DSL query composes measure+group+compare and renders CIs + tier footnotes on the seeded corpus.

Packet: `task_packets/116_polylogue_9l5_7.md`

## 117. polylogue-9l5.13 — activity_spans materializer: edit/test/build/idle/delegate intervals with evidence tiers

**P2 / task / 10-analytics-experiments / blocked-hard**

Mechanism: Issue description localizes the mechanism: The missing bridge between raw structure and "so what": a derived queryable relation of time-bounded work spans composed OVER existing substrate (actions keystone fields, phases 5-min-gap intervals, weak work-event labels, observed events, run projection) — a normalizer/composer, not a new capture pipeline. Span kinds start coarse and construct-valid: read_search, edit, build_compile, test, debug, review_vcs, delegate, synthesize, idle_gap, tool_wait, llm_wait, unknown. LOAD-BEARING DESIGN CHOICE: span kind is SEP…

Implementation spine:
- Register the measure/outcome with evidence tier, denominator, and uncertainty.
- Materialize only after source units are stable.
- Add fixture proving empty/uncovered samples do not become zeros.
- Render caveats in CLI/report/web outputs.

Tests:
- Acceptance proof: Seeded corpus produces spans with evidence refs
- Acceptance proof: >threshold gaps are idle spans
- Acceptance proof: structural test failure yields kind=test outcome=failed
- Acceptance proof: activity-spans where session.repo:X | group by kind | sum duration_ms works (DSL terminal unit + fields registered as part of this bead)
- Acceptance proof: heuristic-classified spans carry the tier visibly.

Packet: `task_packets/117_polylogue_9l5_13.md`

## 118. polylogue-stc — Experiment hosting: declared arms, preregistered metrics, paired analysis, agent-buildable

**P2 / feature / 10-analytics-experiments / blocked-hard**

Mechanism: Issue description localizes the mechanism: Generalize what cfk/jxe did by hand into substrate: an experiment is a first-class declared object — hypothesis, arms, assignment rule, PREREGISTERED metrics (declared before data collection, timestamped — the construct-validity teeth), sample-size intent, analysis plan — and the archive hosts its lifecycle: assignment, observation collection (sessions tagged to arms), paired/grouped analysis through the measure registry, and a cold-reader-gateable report. Agent affordance is the point (operator ask): agents shoul… Design direction: (1) ExperimentSpec as a user.db artifact (assertion-adjacent, judgment-visible): hypothesis, arms (name + treatment description), assignment (manual | alternating | by-session-property), preregistered metrics (each a registry measure ref + direction + minimum-interesting-effect), planned n, analysis plan (paired vs unpaired, test choice via 9l5.7). Prereg timestamp is the assertion created_at — post-hoc metric addit…

Implementation spine:
- (1) ExperimentSpec as a user.db artifact (assertion-adjacent, judgment-visible): hypothesis, arms (name + treatment description), assignment (manual | alternating | by-session-property), preregistered metrics (each a registry measure ref + direction + minimum-interesting-effect), planned n, analysis plan (paired vs unpaired, test choice via 9l5.7).
- Prereg timestamp is the assertion created_at — post-hoc metric additions are visibly post-hoc (labeled exploratory).
- (2) Lifecycle tools (CLI + MCP): experiment define / assign <session-ref> <arm> / status (n per arm, power-ish progress vs planned n) / analyze (runs the plan: per-metric effect + CI + test, paired where declared
- exploratory section separated) / report (markdown artifact, cold-reader-gate ready, .agent/demos pattern).
- (3) Assignment evidence: arm membership is an assertion row with evidence ref to the session — auditable, revocable.

Tests:
- Acceptance proof: cfk's protocol is expressible as an ExperimentSpec and its analysis reproduces via experiment analyze.
- Acceptance proof: An agent (via MCP) can define a valid two-arm experiment end-to-end against the seeded corpus
- Acceptance proof: malformed specs are refused with the missing field named.
- Acceptance proof: Prereg vs exploratory metrics render separately in the report.

Packet: `task_packets/118_polylogue_stc.md`

## 119. polylogue-cfk — Re-run two-arm uplift with freshness-fixed packs (n>=3 pairs, then n=12-20)

**P1 / task / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Successor to campaign polylogue-jxe, which closed diagnostic-negative: raw-ref arm 8/10 vs handoff-pack 5/10, with the loss attributed to packet staleness (pack generated before later devloop work; raw-ref arm found newer beads/archive evidence). The cause was fixed (polylogue-yps freshness metadata + successor links, polylogue-qt3 fast single-process regeneration) but nothing re-tests the hypothesis. Until a re-run exists, the recorded result of the only uplift experiment in the program is 'packs lose', and the c… Design direction: Protocol: identical paired two-arm design as jxe.2/jxe.3 (same scoring rubric, blinded arms, committed comparison artifact under .agent/demos/uplift-two-arm/, cold-reader gate). The one deliberate change: the pack arm consumes a pack REGENERATED AT CONTINUATION START, not a shelf artifact — qt3 made regeneration seconds-fast and in-process; yps metadata must show generated_at ~= consumption time, freshness state fre…

Implementation spine:
- Protocol: identical paired two-arm design as jxe.2/jxe.3 (same scoring rubric, blinded arms, committed comparison artifact under .agent/demos/uplift-two-arm/, cold-reader gate).
- The one deliberate change: the pack arm consumes a pack REGENERATED AT CONTINUATION START, not a shelf artifact — qt3 made regeneration seconds-fast and in-process
- yps metadata must show generated_at ~= consumption time, freshness state fresh, zero successor warnings.
- Run n>=3 pairs first to de-noise the n=1 pilot
- a publishable uplift claim needs n=12-20 pairs.

Tests:
- Acceptance proof: n>=3 paired runs completed under the recorded protocol (fresh continuation-time pack + live query vs raw-ref + live query)
- Acceptance proof: per-pair scores + paired analysis committed under .agent/demos/uplift-two-arm/
- Acceptance proof: cold-reader gate on the comparison artifact
- Acceptance proof: result recorded in the bead (positive, negative, or ambiguous -> three-arm follow-up decision).

Packet: `task_packets/119_polylogue_cfk.md`

## 120. polylogue-212.7 — Demo Finding Packet contract + prompt runner + registry manifest

**P2 / task / 05-analysis-provenance-citations / blocked-hard**

Mechanism: Issue description localizes the mechanism: Convert 212 from a shelf of named demos into a PORTFOLIO CONTRACT: every demo is an executable PROMPT.md handed to a coding agent, and every prompt emits the identical Demo Finding Packet: PROMPT.md, finding.yaml (five-part provenance stanza per 3tl.4: archive cursor, measure/query version, commit SHA, sample-frame predicate, run date), report.md (fixed section order: claim, corpus, method, findings, specimens, counterexamples, limits, reproduce), evidence.ndjson (one row per cited ref), queries.ndjson (text + low… Design direction: Anchor: .agent/demos/ (existing shelf: agent-forensics, claim-vs-evidence, degraded-archive-proof, CURATED_CATALOG.md as the manifest seed). Contract: every demo directory gains PROMPT.md (executable instructions a coding agent runs cold) and emits an identical Demo Finding Packet: finding.yaml (five-part provenance stanza per 3tl.4), rendered artifact, and the exact reproduction commands. Build a registry manifest …

Implementation spine:
- Anchor: .agent/demos/ (existing shelf: agent-forensics, claim-vs-evidence, degraded-archive-proof, CURATED_CATALOG.md as the manifest seed).
- Contract: every demo directory gains PROMPT.md (executable instructions a coding agent runs cold) and emits an identical Demo Finding Packet: finding.yaml (five-part provenance stanza per 3tl.4), rendered artifact, and the exact reproduction commands.
- Build a registry manifest (extend CURATED_CATALOG.md or a demos.yaml) listing id, claim, packet path, substrate features exercised, last-regenerated.
- A prompt runner (thin script or devtools lab command) executes one demo prompt end-to-end and validates packet shape.
- Pitfall: demos run against the LIVE archive — packet outputs must be private-data-audited before any publication lane (3tl.4 owns publishing).

Tests:
- Acceptance proof: Packet schema documented + validated by the runner
- Acceptance proof: one existing demo (D1 receipts) re-emitted through the runner produces a conforming packet on the seeded corpus
- Acceptance proof: registry manifest lint catches a missing packet.
- Acceptance proof: Verify: runner fixture test + manifest check.

Packet: `task_packets/120_polylogue_212_7.md`

## 121. polylogue-212.8 — The honesty anti-demo: a tempting finding that emits verdict not_supported

**P2 / task / 05-analysis-provenance-citations / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Ship a demo whose SUCCESS is refusal: attempt a tempting claim (e.g. minute-by-minute multi-source operator reconstruction) and emit the standard packet with verdict: not_supported, listing missing modalities, missing refs, and the exact query/evidence gap. Published BESIDE the successful demos, not hidden — this is the brand ("refuses rather than fabricates") made demonstrable, and it directly encodes the situation-brief praise for the honest deferral of the multi-source demo. Framing decision for operator in 212… Design direction: Depends on the 212.7 packet contract — this demo is one more packet whose verdict field is not_supported. Pick the tempting claim: minute-by-minute multi-source operator reconstruction (needs modalities the archive lacks). The packet lists missing modalities, missing refs, and the exact query/evidence that WOULD support it, using the same finding.yaml shape. Anchor: .agent/demos/<new-dir>/ + the insight_rigor_audit …

Implementation spine:
- Depends on the 212.7 packet contract — this demo is one more packet whose verdict field is not_supported.
- Pick the tempting claim: minute-by-minute multi-source operator reconstruction (needs modalities the archive lacks).
- The packet lists missing modalities, missing refs, and the exact query/evidence that WOULD support it, using the same finding.yaml shape.
- Anchor: .agent/demos/<new-dir>/ + the insight_rigor_audit surface to enumerate what evidence exists vs required.
- The success criterion is the refusal being specific, not vague: every missing item names the unit/table/modality that would have to exist.

Tests:
- Acceptance proof: Anti-demo packet passes the packet lint with not_supported verdict
- Acceptance proof: report names each missing capability with the bead ref that would supply it
- Acceptance proof: included in the registry manifest and the public mini-portfolio.
- Acceptance proof: Verify: runner emits + lint passes.

Packet: `task_packets/121_polylogue_212_8.md`

## 122. polylogue-212.1 — Post-hoc forensic Q&A demo: questions a tracer cannot answer

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live — when did the bad assumption first enter; which file churned before the regression; what evidence did the agent cite for a design choice; which prior failed attempts resemble today's failure. Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation); packaging is the work, plus one honest 'we cannot answer X' slide (construct validity). Design direction: A category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live, when the bad assumption first entered; which file churned before the regression; what evidence the agent cited for a design choice; which prior failed attempts resemble today's. Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation); packaging is the work,…

Implementation spine:
- A category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live, when the bad assumption first entered
- which file churned before the regression
- what evidence the agent cited for a design choice
- which prior failed attempts resemble today's.
- Composes existing reads (postmortem bundle, work events, phases, neighbor candidates, git correlation)

Tests:
- Acceptance proof: 1.
- Acceptance proof: Against one completed multi-hour session, the demo answers each forensic question live using existing reads (get_postmortem_bundle, session_work_events, session_phases, neighbor_candidates, git correlation): first-bad-assumption entry, file churned before the regression, cited evidence for a design choice, and resembling prior failed attempts.
- Acceptance proof: 2.
- Acceptance proof: One explicit 'we cannot answer X' slide is included (construct-validity honesty).
- Acceptance proof: Verify: the demo runs end-to-end against a chosen archived session (recorded output/artifact) using only existing reads (no new query machinery).

Packet: `task_packets/122_polylogue_212_1.md`

## 123. polylogue-212.2 — D1 'The receipts': claim-vs-evidence on a real PR

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Pick a merged agent-authored PR; resolve PR -> authoring session via session_commits/session_repos; get_postmortem_bundle; render two columns: claimed (PR-body sentences: 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration — drillable to the raw tool_result block). A PR body audited against ground truth in ~10 seconds. Nearly free: all reads exist. Tell the deleted-prose-miner story as part of the demo (why this exists). Design direction: A demo: pick a merged agent-authored PR, resolve PR->authoring session via session_commits/session_repos, run get_postmortem_bundle, and render two columns, claimed (PR-body sentences like 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration, drillable to the raw tool_result block). Audits a PR body against ground truth in ~10 seconds. All reads exist; tell the deleted-prose-miner stor…

Implementation spine:
- A demo: pick a merged agent-authored PR, resolve PR->authoring session via session_commits/session_repos, run get_postmortem_bundle, and render two columns, claimed (PR-body sentences like 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration, drillable to the raw tool_result block).
- Audits a PR body against ground truth in ~10 seconds.
- All reads exist
- tell the deleted-prose-miner story as motivation.

Tests:
- Acceptance proof: 1.
- Acceptance proof: For a chosen merged agent-authored PR, the demo resolves the authoring session from session_commits/session_repos and produces a two-column claim-vs-evidence view: PR-body claim sentences beside the observed actions rows (invocation, exit_code, duration), drillable to the raw tool_result block.
- Acceptance proof: 2.
- Acceptance proof: The demo composes only existing reads (get_postmortem_bundle) with no new query machinery and includes the deleted-prose-miner motivation.
- Acceptance proof: Verify: run against a real merged PR and its authoring session (recorded artifact)

Packet: `task_packets/123_polylogue_212_2.md`

## 124. polylogue-212.3 — D2 'Where did the money actually go': cost by outcome

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Five-axis cost basis shown honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot nobody else can do: cost by outcome — '$N this month; X% spent in sessions that ended abandoned or with a failing final action; five most expensive failures, click through to the exact turn.' Needs the outcome-conditioned join (action outcome fields bead); instruments otherwise exist (cost_rollups, session_costs, terminal-state profiles, per-origin exact/estimate labels rendered as footnotes). Design direction: A demo: show the five-axis cost basis honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot no chat UI can do, cost by outcome: total monthly spend, the % spent in sessions that ended abandoned or with a failing final action, and the five most expensive failures each drillable to the exact turn. Needs the outcome-conditioned join (action outcome fields bead); cost instruments exis…

Implementation spine:
- A demo: show the five-axis cost basis honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot no chat UI can do, cost by outcome: total monthly spend, the % spent in sessions that ended abandoned or with a failing final action, and the five most expensive failures each drillable to the exact turn.
- Needs the outcome-conditioned join (action outcome fields bead)
- cost instruments exist (cost_rollups, session_costs, terminal-state profiles, per-origin exact/estimate labels rendered as footnotes).

Tests:
- Acceptance proof: 1.
- Acceptance proof: The demo renders a five-axis cost basis with provider-reported-exact vs catalog-priced values clearly labeled and coverage stated (per-origin exact/estimate footnotes).
- Acceptance proof: 2.
- Acceptance proof: Cost-by-outcome pivot: total monthly spend, the fraction spent in abandoned or failing-final-action sessions, and the five most expensive failures, each drillable to the exact turn via the outcome-conditioned join.
- Acceptance proof: Verify: the demo runs via cost_rollups/session_costs against the seeded corpus (recorded output)

Packet: `task_packets/124_polylogue_212_3.md`

## 125. polylogue-212.4 — D4 'Behavioral archaeology': six DSL queries, rapid fire

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Each answers a question an engineering lead would ask, each impossible in any chat UI: SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); near:'race condition' semantic probe across providers; abandoned-in-this-repo-this-quarter; then pipe straight into read. Show explain_query_expression once to prove the query means what it says. Nearly free: all reads exist. Doubles as the DSL reference-card content. Design direction: A demo: six DSL queries, each answering a question an engineering lead would ask and each impossible in a chat UI, SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); a `near:'race condition'` semantic probe across providers; abandoned-in-this-repo-this-quarter; then a query piped straight into `read`. Show `explain_query_expression` once to prove a query means what it s…

Implementation spine:
- A demo: six DSL queries, each answering a question an engineering lead would ask and each impossible in a chat UI, SEQ thrash-loop hunt
- failure-rate by model
- which tools break (observed-event outcomes by tool)
- a `near:'race condition'` semantic probe across providers
- abandoned-in-this-repo-this-quarter

Tests:
- Acceptance proof: 1.
- Acceptance proof: Six DSL queries are authored and run against the demo/seeded corpus, each producing sensible results: SEQ thrash-loop, failure-rate by model, tool-breakage by observed-event outcome, `near:` semantic probe across providers, abandoned-this-repo-this-quarter, and a query piped into `read`.
- Acceptance proof: 2.
- Acceptance proof: `explain_query_expression` is shown once demonstrating a query's parsed meaning.
- Acceptance proof: 3.

Packet: `task_packets/125_polylogue_212_4.md`

## 126. polylogue-3tl.12 — README de-meta / de-persuasion pass with reproducible capability claims

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Design direction: Raw-log 2026-07-04 18:16-18:21 (post-dates the closed 3tl.1 skim-ladder rewrite): strip the meta/persuasion register from the README, define agent-coined terms (judged notes, work phases, logical session) on first use, and make each capability claim reproducible on the operator's own archive (a command the reader can run). Distinct axis from 3tl.1's structure work.

Implementation spine:
- Raw-log 2026-07-04 18:16-18:21 (post-dates the closed 3tl.1 skim-ladder rewrite): strip the meta/persuasion register from the README, define agent-coined terms (judged notes, work phases, logical session) on first use, and make each capability claim reproducible on the operator's own archive (a command the reader can run).
- Distinct axis from 3tl.1's structure work.

Tests:
- Acceptance proof: README first screen names the category and four verbs without persuasion register
- Acceptance proof: every coined term is defined at first use
- Acceptance proof: each capability claim links a runnable `polylogue`/`devtools` command
- Acceptance proof: a fresh no-context reader can reproduce >=2 claims.
- Acceptance proof: Verify: docs-commands lint green + cold-reader pass.

Packet: `task_packets/126_polylogue_3tl_12.md`

## 127. polylogue-3tl.13 — Reconcile schema-versioning docs + retire superseded execution-plan.md

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Design direction: architecture-spine.md:34-37 lists 'in-place upgrade chains' as Rejected with no durable-additive carve-out, contradicting the shipped migrations and internals.md's own two-regime text (internals.md:284-289 is internally inconsistent). docs/execution-plan.md is fully superseded (dropped #1807 umbrella; every issue re-encoded as a bead) yet README.md:14 still calls it 'current sequencing plan'. Fix the spine section, …

Implementation spine:
- architecture-spine.md:34-37 lists 'in-place upgrade chains' as Rejected with no durable-additive carve-out, contradicting the shipped migrations and internals.md's own two-regime text (internals.md:284-289 is internally inconsistent).
- docs/execution-plan.md is fully superseded (dropped #1807 umbrella
- every issue re-encoded as a bead) yet README.md:14 still calls it 'current sequencing plan'.
- Fix the spine section, reconcile internals.md, retire execution-plan.md with a pointer to Beads, and repoint README:14.

Tests:
- Acceptance proof: architecture-spine + internals schema-versioning sections describe the two-regime model consistently
- Acceptance proof: execution-plan.md is archived/removed and no doc calls it current
- Acceptance proof: README points at Beads.
- Acceptance proof: Verify: render docs-surface --check + grep 'execution-plan' docs README.

Packet: `task_packets/127_polylogue_3tl_13.md`

## 128. polylogue-3tl.4 — Findings publishing lane: campaign artifacts on the docs site

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The Pages pipeline already builds and deploys the docs site on master push; give campaign artifacts (claim-vs-evidence finding, forensics report) a publishing lane there — rendered report + reproduction instructions, regenerated from the seeded corpus so nothing private ships. The finding needs a URL before anything external can cite it. Design direction: Publishing lane = a devtools render surface, not an ad-hoc workflow. (1) SOURCE: each finding lives at docs/findings/<slug>/finding.yaml carrying a five-part PROVENANCE STANZA (archive cursor id/position at measurement, measure+query-DSL/code version, git commit SHA, sample-frame predicate = the exact population query, run date) plus its structural body. (2) RENDER: add devtools/render_findings.py + a CommandSpec 'r…

Implementation spine:
- Publishing lane = a devtools render surface, not an ad-hoc workflow.
- (1) SOURCE: each finding lives at docs/findings/<slug>/finding.yaml carrying a five-part PROVENANCE STANZA (archive cursor id/position at measurement, measure+query-DSL/code version, git commit SHA, sample-frame predicate = the exact population query, run date) plus its structural body.
- (2) RENDER: add devtools/render_findings.py + a CommandSpec 'render findings' in devtools/command_catalog.py (model on render_pages.py entry at command_catalog.py:191)
- wire it into devtools/render_all.py so 'devtools render all --check' fails on drift.
- It renders docs/findings/<slug>/index.md (+ a per-finding CHANGELOG section).

Tests:
- Acceptance proof: 1.
- Acceptance proof: devtools render findings exists, registered in devtools/command_catalog.py, wired into devtools render all with a working --check.
- Acceptance proof: 2.
- Acceptance proof: At least one real finding (the base claim-vs-evidence finding) renders to docs/findings/<slug>/index.md and is served by pages.yml at a stable citeable URL /findings/<slug>/ (this is the 'published finding URL' 3tl acceptance clause 3 depends on).
- Acceptance proof: 3.

Packet: `task_packets/128_polylogue_3tl_4.md`

## 129. polylogue-3tl.7 — Release is a decision: proven install matrix across package managers and OSes

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Release machinery exists (release-please, PyPI + Homebrew tap + GHCR + Nix flake wired in CI per the grok evidence) but 'wired' is not 'proven': nobody continuously verifies that a stranger's install actually works on each lane, so the first real user on each path is the test. The target state the operator named: everything prepared so that shipping is ONLY the decision to merge the release PR — no scramble, no 'does brew even work', no unknown-OS surprises. Design direction: (1) INSTALL-MATRIX CI (scheduled weekly + pre-release, not per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, docker run from GHCR, nix run — on ubuntu + macos runners (arm+x86 where available); each job runs the same smoke: install -> polylogue demo seed -> one find -> one read -> version check. The demo corpus makes this stranger-equivalent. (2) WINDOWS: decide and STATE the story (native is unteste…

Implementation spine:
- (1) INSTALL-MATRIX CI (scheduled weekly + pre-release, not per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, docker run from GHCR, nix run — on ubuntu + macos runners (arm+x86 where available)
- each job runs the same smoke: install -> polylogue demo seed -> one find -> one read -> version check.
- The demo corpus makes this stranger-equivalent.
- (2) WINDOWS: decide and STATE the story (native is untested
- document WSL2 as the supported path honestly in README install section) — an honest 'WSL2 only' beats a broken native promise.

Tests:
- Acceptance proof: 1.
- Acceptance proof: Install-matrix CI workflow (scheduled weekly + pre-release, NOT per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, `docker run` from GHCR, and `nix run`, on ubuntu + macos runners (arm+x86 where available)
- Acceptance proof: each job runs the same smoke: install -> `polylogue demo seed` -> one `find` -> one `read` -> `polylogue --version` check.
- Acceptance proof: 2.
- Acceptance proof: Windows story stated honestly in the README install section (native marked untested

Packet: `task_packets/129_polylogue_3tl_7.md`

## 130. polylogue-kwsb — Security & privacy: the archive can forget on purpose and never leaks secrets

**P2 / epic / 00-trust-floor / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: WHY: a personal archive of ALL AI work is the most sensitive database on the machine — it must be able to forget on purpose (excision that provably removes bytes, not just rows) and must never leak (localhost daemon reachable from a hostile page, secrets in captured content). Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but backlog ownership was missing. MEMBER BEADS: polylogue-kwsb.1 (Host/Origin gate + receiver token + spool governor — the live DNS-rebinding hole), polylogue-27m … Design direction: No epic owned the security/privacy surface: excision (27m), reset-mutation safety (jnj.5, real bug at reset.py:260-277 where --session/--source tombstone before the preview:327 and --yes gate:331), and crawl-source permissions (jsy) were orphaned. Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but the BACKLOG ownership was missing. Also owns the security-privacy-coverage.yaml manifest …

Implementation spine:
- No epic owned the security/privacy surface: excision (27m), reset-mutation safety (jnj.5, real bug at reset.py:260-277 where --session/--source tombstone before the preview:327 and --yes gate:331), and crawl-source permissions (jsy) were orphaned.
- Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but the BACKLOG ownership was missing.
- Also owns the security-privacy-coverage.yaml manifest gaps.
- NON-GOAL: do not resurrect the paused sanitize/redaction cluster (chatlog != spec).

Tests:
- Acceptance proof: Excision (right-to-forget + secret redaction + blob excision) is execution-grade and shares one mutation-audit/dry-run/--yes contract with reset (jnj.5)
- Acceptance proof: the security-privacy-coverage.yaml gaps each have an owning bead or test
- Acceptance proof: the MCP write/admin destructive path shares the same audit-row contract.
- Acceptance proof: Verify: devtools verify + the reset/excision dry-run tests.

Packet: `task_packets/130_polylogue_kwsb.md`

## 131. polylogue-27m — Excision and secret hygiene: the archive can forget on purpose

**P2 / task / 00-trust-floor / blocked-hard**

Mechanism: Issue description localizes the mechanism: Keep-everything is doctrine; 'cannot remove that API key I pasted in 2025' is a bug in that doctrine's shadow. Two halves: (1) EXCISION — a supported, auditable operation that provably removes one specific piece of content from source rows, index rows, FTS, embeddings, blobs, and derived models, leaving a tombstone recording that something was excised (not what); hard because content-addressing and idempotency assume immutability. (2) SECRET SCANNING at ingest — sessions demonstrably contain pasted credentials; de… Design direction: Excision mechanics: source.db rows get a redacted_at + reason tombstone with the payload replaced by a hash-of-removed marker (content hash boundary: the session's content_hash is recomputed and the old hash recorded on the tombstone so idempotency does not resurrect it); derived tiers rebuild the affected session (blue-green machinery makes this cheap); blobs: reference-counted delete with GC-lease discipline; embe…

Implementation spine:
- Excision mechanics: source.db rows get a redacted_at + reason tombstone with the payload replaced by a hash-of-removed marker (content hash boundary: the session's content_hash is recomputed and the old hash recorded on the tombstone so idempotency does not resurrect it)
- derived tiers rebuild the affected session (blue-green machinery makes this cheap)
- blobs: reference-counted delete with GC-lease discipline
- embeddings: delete rows by message ref.
- The operation is a single 'polylogue ops excise <ref>' with a mandatory reason, a dry-run diff, and an ops.db audit row.

Tests:
- Acceptance proof: Excising a seeded session's message removes it from every tier (verified by grep across source/index/FTS/embeddings/blob refs) and leaves the tombstone
- Acceptance proof: re-ingesting the original source does not resurrect it
- Acceptance proof: the secret scanner flags a seeded fake credential as a candidate without logging its value
- Acceptance proof: retro scan on the live archive produces a bounded candidate list.

Packet: `task_packets/131_polylogue_27m.md`

## 132. polylogue-38x — Reconcile archived audit residue against current source

**P2 / task / 00-trust-floor / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Older archived audits under .agent/archive/conductor-history/2026-07-01 still contain valuable findings that are not all represented as executable Beads. This task is to re-check the remaining concrete findings against current source and either close them as stale/fixed or split/link them to the owning subsystem bead. Seed findings: construct-validity audit flags Codex FORK vs RESUME conflation, multi-meta CONTINUATION as proxy, scalar paste detection flattening exact vs fallback, timestamp fallback to epoch-zero,… Design direction: Run this as a source-grounded reconciliation pass, not as implementation by memory. For each seed finding: inspect current source and tests; classify fixed, still-live, subsumed-by-existing-bead, or split-needed; cite file paths/functions and the owning bead. Live bugs should be turned into narrow child/linked Beads under the relevant parent (lineage, provider usage, insights-as-declared-views, provider parsers, MCP…

Implementation spine:
- Run this as a source-grounded reconciliation pass, not as implementation by memory.
- For each seed finding: inspect current source and tests
- classify fixed, still-live, subsumed-by-existing-bead, or split-needed
- cite file paths/functions and the owning bead.
- Live bugs should be turned into narrow child/linked Beads under the relevant parent (lineage, provider usage, insights-as-declared-views, provider parsers, MCP/query surface).

Tests:
- Acceptance proof: A current-source reconciliation table exists for every seed finding from the archived construct-validity/fanout/insights audits
- Acceptance proof: every still-live finding is linked to an owning executable Beads issue or split into one
- Acceptance proof: every stale/fixed finding cites current source or tests
- Acceptance proof: no archived audit item in the seed list remains only as untriaged markdown
- Acceptance proof: bd ready no longer depends on reading .agent/archive to discover these issues.

Packet: `task_packets/132_polylogue_38x.md`

## 133. polylogue-f2qv — Provider usage & cost honesty: disjoint token lanes, one pricing source, dual cost view

**P2 / epic / 02-usage-cost-honesty / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: WHY: token/cost accounting is a correctness surface with a track record of silent large errors (7.69x Codex inflation; per-model partition double-count #2472) — and cost numbers are exactly what operators quote publicly, so wrong numbers are reputational. Four invariants define honest accounting (full doctrine in design): disjoint token lanes; one pricing source (vendored LiteLLM catalog, last-path-segment match); dual view (API-list-equivalent vs subscription-credit); stale-row hygiene (376.6B-token class artifac… Design direction: PROBLEM / DOCTRINE. Token and cost accounting is a correctness surface, not a nicety: prior bugs produced a 7.69x Codex cost inflation and a residual per-model partition double-count (#2472). Four invariants define 'honest' and are the spine of this epic: 1. DISJOINT LANES. Provider-reported token fields overlap and must be decomposed before summing. Codex 'input' INCLUDES cached tokens (~96% of input in practice) a…

Implementation spine:
- PROBLEM / DOCTRINE.
- Token and cost accounting is a correctness surface, not a nicety: prior bugs produced a 7.69x Codex cost inflation and a residual per-model partition double-count (#2472).
- Four invariants define 'honest' and are the spine of this epic:
- 1.
- DISJOINT LANES.

Tests:
- Acceptance proof: 1.
- Acceptance proof: A cross-provider usage ledger reports cached/uncached input and reasoning/completion output as separate labelled lanes for Codex, Claude, ChatGPT
- Acceptance proof: a property/invariant test asserts no lane is double-summed and cache lanes are never folded into generic input/output (repro of the 7.69x-class inflation stays green).
- Acceptance proof: 2.
- Acceptance proof: Per-model rollups sum to the session total with no multi-model double-count

Packet: `task_packets/133_polylogue_f2qv.md`

## 134. polylogue-9e5.25 — Review zero-use MCP surfaces from affordance usage artifact

**P2 / task / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The current agent-affordance-usage demo classifies 59 MCP tools as zero captured agent use and non-operator surfaces. This is review input, not automatic deletion: use .agent/demos/agent-affordance-usage/surface-inventory.csv and affordance-usage.report.json to decide which surfaces should collapse into query/surface algebra, which need docs/examples, and which should be removed. Design direction: Batch the review through contracts/surface-algebra rather than deleting isolated tools. Preserve operator-only caveats; verify each proposed removal or merge against the registered MCP tool set and actual consumers.

Implementation spine:
- Batch the review through contracts/surface-algebra rather than deleting isolated tools.
- Preserve operator-only caveats
- verify each proposed removal or merge against the registered MCP tool set and actual consumers.

Tests:
- Acceptance proof: 1.
- Acceptance proof: Every MCP kill-candidate row from .agent/demos/agent-affordance-usage/surface-inventory.csv is classified as remove / merge / keep / needs-demo with rationale.
- Acceptance proof: 2.
- Acceptance proof: Removal or merge work is split into executable beads with exact tool names and surface contracts.
- Acceptance proof: 3.

Packet: `task_packets/134_polylogue_9e5_25.md`

## 135. polylogue-9e5.26 — Review zero-use CLI surfaces from affordance usage artifact

**P2 / task / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The current agent-affordance-usage demo classifies 34 CLI commands as zero captured agent use and non-operator surfaces. This is review input, not automatic deletion: use .agent/demos/agent-affordance-usage/surface-inventory.csv and affordance-usage.report.json to decide which commands should collapse into the query/composition surface, which need docs/examples, and which should be removed. Design direction: Batch the review through CLI surface algebra and command-inventory contracts. Prefer removing bespoke fronts when the query DSL/read-package path can express the same operation cleanly; preserve commands with clear operator workflows even if agents have not used them.

Implementation spine:
- Batch the review through CLI surface algebra and command-inventory contracts.
- Prefer removing bespoke fronts when the query DSL/read-package path can express the same operation cleanly
- preserve commands with clear operator workflows even if agents have not used them.

Tests:
- Acceptance proof: 1.
- Acceptance proof: Every CLI kill-candidate row from .agent/demos/agent-affordance-usage/surface-inventory.csv is classified as remove / merge / keep / needs-demo with rationale.
- Acceptance proof: 2.
- Acceptance proof: Removal or merge work is split into executable beads with exact command paths and docs/rendering impact.
- Acceptance proof: 3.

Packet: `task_packets/135_polylogue_9e5_26.md`

## 136. polylogue-9e5.27 — Speed up live affordance usage surface inventory

**P2 / task / 02-usage-cost-honesty / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: After switching the default family report and inventory counts away from action-row materialization, the live .agent/demos/agent-affordance-usage regeneration still took roughly 88 seconds on the full archive. The artifact is usable, but this is too slow for a polished demo/workspace command. Design direction: Profile devtools workspace affordance-usage on the live archive with query-plan evidence. Likely targets: CLI command/path matching over generic tool-use rows, missing expression indexes for generated command/path fields, or repeated direct scans that should be a reusable product query primitive.

Implementation spine:
- Profile devtools workspace affordance-usage on the live archive with query-plan evidence.
- Likely targets: CLI command/path matching over generic tool-use rows, missing expression indexes for generated command/path fields, or repeated direct scans that should be a reusable product query primitive.

Tests:
- Acceptance proof: 1.
- Acceptance proof: Capture query-plan/timing evidence for each major affordance-usage phase on /home/sinity/.local/share/polylogue.
- Acceptance proof: 2.
- Acceptance proof: Reduce default live regeneration to a materially faster target or document the exact storage/index bead needed.
- Acceptance proof: 3.

Packet: `task_packets/136_polylogue_9e5_27.md`

## 137. polylogue-83u — Attachment & blob evidence integrity: bytes exist, are honest, and stay affordable

**P1 / epic / 01-blob-attachment-integrity / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Attachments are metadata-only by construction: 8,425 rows claim 8.4GB, 0 blobs exist, 56% zero-byte; blob_hash was synthetic until v13 made it honest-nullable with acquisition_status. This program makes attachment/blob evidence real end-to-end: acquire bytes where handles are live, classify what is genuinely unfetchable, keep the backup verifier trustworthy, and compress the store. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

Implementation spine:
- Inventory open child beads and map them to the invariant named by the epic.
- Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
- Close only after child beads are closed or explicitly split out with new blockers.

Tests:
- Acceptance proof: REFRAMED (operator 2026-07-04): the goal is to CAPTURE attachment bytes going forward, not miss-then-account.
- Acceptance proof: (1) Forward capture is default at ingest/browser-capture: uploaded + inline bytes land in the blob store at acquisition time (83u.3, 83u.1).
- Acceptance proof: (2) Non-inline bytes that STILL EXIST at their source are re-acquired (83u.2) — 'we're not getting some that exist' is a bug, not acceptable loss.
- Acceptance proof: (3) A permanent unfetchable floor is NORMAL and expected (source deleted, pre-install history, provider expiry) — the census (83u.6) reports it as honest baseline accounting, never as a failure to fix.
- Acceptance proof: Terminal state: no attachment whose bytes were reachable at capture time is lost

Packet: `task_packets/137_polylogue_83u.md`

## 138. polylogue-8jg9 — Operational resilience: recoverable, restorable, survives daemon death and deploy

**P2 / epic / 01-blob-attachment-integrity / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: WHY: an archive whose pitch is durable evidence must itself survive incidents — daemon death mid-write, bad deploys, disk loss. Durable tiers (source.db/user.db) are irreplaceable; a restore path that has never been drilled is a hope, not a capability. ENABLES: trusting the archive as system-of-record; the backup-manifest gate that durable-tier migrations (60i5) already assume. MEMBER BEADS: polylogue-4be (backup-restore + quarterly restore drill), polylogue-peo (daemon-death recovery), polylogue-s8q (deploy trust… Design direction: Backup-restore (4be, quarterly restore drill), daemon-death recovery (peo), and deploy-trust (s8q, parked P4 while prod polylogued is inactive) had no home. This is the 'does the system survive an incident' capability, distinct from security (forgetting on purpose) and from 1xc (correct at scale).

Implementation spine:
- Backup-restore (4be, quarterly restore drill), daemon-death recovery (peo), and deploy-trust (s8q, parked P4 while prod polylogued is inactive) had no home.
- This is the 'does the system survive an incident' capability, distinct from security (forgetting on purpose) and from 1xc (correct at scale).

Tests:
- Acceptance proof: A quarterly restore drill proves backups restore (4be)
- Acceptance proof: daemon crash mid-convergence recovers without stranding debt (peo, ties 1xc.3/1xc.4)
- Acceptance proof: deployed state is provable via deployment-smoke when prod is re-activated (s8q).
- Acceptance proof: Verify: devtools workspace deployment-smoke --json + a restore-drill artifact.

Packet: `task_packets/138_polylogue_8jg9.md`

## 139. polylogue-8jg9.1 — Standing backlog-hygiene invariant lint (bd devloop gate)

**P2 / task / 01-blob-attachment-integrity / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Backlog structure trails filing unless an invariant lint enforces it (the 2026-07-03 session needed a 41-agent sweep to recover). The backlog equivalent of automagic-invariants: violations fail a gate instead of accumulating until an archaeology session. Design direction: This session needed a 41-agent sweep because structure trails filing. Make it self-maintaining: a bd-level lint in the devloop that fails on (a) any P0/P1 bead with null acceptance (respecting notes sidecars), (b) any decision-type bead open past a 'Status: adopted/decided' line, (c) any active bead with no area:* label, (d) any orphan (no epic parent) older than N days, (e) any blocks-edge pointing at a closed bead…

Implementation spine:
- This session needed a 41-agent sweep because structure trails filing.
- Make it self-maintaining: a bd-level lint in the devloop that fails on (a) any P0/P1 bead with null acceptance (respecting notes sidecars), (b) any decision-type bead open past a 'Status: adopted/decided' line, (c) any active bead with no area:* label, (d) any orphan (no epic parent) older than N days, (e) any blocks-edge pointing at a closed bead (false-block).
- The backlog equivalent of automagic-invariants.

Tests:
- Acceptance proof: The lint runs in the devloop and fails on a seeded violation of each of the 5 classes
- Acceptance proof: a clean backlog passes
- Acceptance proof: wired into devtools verify or a bd hook.
- Acceptance proof: Verify: seed one violation per class, assert non-zero exit.

Packet: `task_packets/139_polylogue_8jg9_1.md`

## 140. polylogue-1xc — Scale-hardening: bugs that only bite on real-scale archives

**P1 / epic / 08-scale-performance-live / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Confirmed-severe set of code correct on small/clean fixtures but wrong at real scale (e.g. full insight rebuild = one transaction -> 6GB WAL + minutes-long write lock). Work the checklist on the issue; tier-1 items were observed live. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Tier-1 confirmed-live items (gh#2465 checklist is authoritative; work it there): full insight rebuild runs as ONE transaction -> 6GB WAL + minutes-long write lock on the live archive — chunk the rebuild into bounded per-batch transactions with progress rows (storage/insights rebuild path); the run_ref global-PK collision class was fixed (#2464) — audit for siblings (any global PK derived from non-unique local coordi…

Implementation spine:
- Tier-1 confirmed-live items (gh#2465 checklist is authoritative
- work it there): full insight rebuild runs as ONE transaction -> 6GB WAL + minutes-long write lock on the live archive — chunk the rebuild into bounded per-batch transactions with progress rows (storage/insights rebuild path)
- the run_ref global-PK collision class was fixed (#2464) — audit for siblings (any global PK derived from non-unique local coordinates).
- General class to hunt: code correct on small/clean/distinct-id fixtures but wrong on real-scale shape (16K+ sessions, 5M+ messages, hash collisions, duplicate native ids, giant single artifacts like the 384MB Codex raw row).
- Add scale-tier tests where cheap (synthetic corpus generator exists).

Tests:
- Acceptance proof: Epic terminal state: every child closed and a scale-regression lane exists (seeded large-archive tier or live-copy probe) that would have caught each shipped bug class, wired into the optional lanes.

Packet: `task_packets/140_polylogue_1xc.md`

## 141. polylogue-a7xr.11 — Prune protocols.py zero-consumer protocols + dead repo kwarg query surface + cursor mapping bug

**P2 / chore / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: VERIFIED 2026-07-06: 6 of 14 protocols in protocols.py have zero consumers anywhere (SessionReader, SearchStore, ArchiveMessageQueryStore, SemanticArchiveQueryStore, SessionSemanticStatsStore, SessionArchiveReadStore) — violating the module's own docstring rule ('only protocols with 2+ implementations earn their existence'). The 18-filter-kwarg signature is spelled out 3x in SessionReader alone. The repo kwarg methods are equally dead: RepositoryArchiveQueryMixin.list (docstring-example-only), .count (zero callers… Design direction: Delete the 6 unconsumed protocols, the repo list/list_summaries/count kwarg wrappers + iter_summary_pages, the two TypedDicts (pass SessionRecordQuery through at query_store_archive.py:70-84), and either the cursor field-spec entry or add the cursor field deliberately (decide with rxdo pagination needs — a real cursor concept may arrive with result-set pagination; if so, wire it properly instead of deleting). KEEP p…

Implementation spine:
- Delete the 6 unconsumed protocols, the repo list/list_summaries/count kwarg wrappers + iter_summary_pages, the two TypedDicts (pass SessionRecordQuery through at query_store_archive.py:70-84), and either the cursor field-spec entry or add the cursor field deliberately (decide with rxdo pagination needs — a real cursor concept may arrive with result-set pagination
- if so, wire it properly instead of deleting).
- KEEP protocols with real consumers: SessionQueryRuntimeStore, SessionOutputStore, SessionArchiveStatsStore, TagStore, RawPersistenceStore, RawValidationStore (genuine test double).
- mypy --strict is the net.

Tests:
- Acceptance proof: protocols.py contains only consumed protocols (each with a named consumer in a comment)
- Acceptance proof: dead kwarg surface gone
- Acceptance proof: cursor mapping resolved (deleted or actually wired)
- Acceptance proof: mypy strict green
- Acceptance proof: ~600 LOC removed.

Packet: `task_packets/141_polylogue_a7xr_11.md`

## 142. polylogue-20d — Interactive performance: the front door answers in interactive time

**P2 / epic / 08-scale-performance-live / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Cold CLI invocations pay ~2s of Python imports; some helps took 5-9s; find-then-select cold spikes; claim-vs-evidence regen 43s; ingest catch-up crawled at 0.2 files/s. WAL checkpoint + ANALYZE done (2026-07-03: index.db WAL=0, sqlite_stat1 present, v23). The CLI->daemon fast path is the structural attack; import deferral is the fallback for daemonless cold starts. Design direction: Front-door interactive-latency spine. Mechanism ordering: 20d.14 states the named budgets first (evidence-tuned starting points); 20d.2 removes the ~2s import tax for the daemonless cold path; 20d.1 routes the hot path through the daemon over UDS; 20d.12 makes the daemon worth reaching (cursor-keyed result cache); 20d.13 replaces polling with SSE push; 20d.6/20d.15 own the live vs bulk ingest lanes; 20d.4/20d.5/20d.…

Implementation spine:
- Front-door interactive-latency spine.
- Mechanism ordering: 20d.14 states the named budgets first (evidence-tuned starting points)
- 20d.2 removes the ~2s import tax for the daemonless cold path
- 20d.1 routes the hot path through the daemon over UDS
- 20d.12 makes the daemon worth reaching (cursor-keyed result cache)

Tests:
- Acceptance proof: The 20d.14 interactive SLO tier is defined in docs/plans/slo-catalog.yaml and runs green in `devtools bench slo` against the seeded corpus with a live daemon.
- Acceptance proof: On the operator machine, live measurement meets the daemon-served query, completion round-trip, cold-CLI, and ingest-to-searchable budgets named in 20d.14.
- Acceptance proof: No interactive read verb pays the old cold-import or FTS-gate penalties: the 20d.2 help-latency budget check and the 20d.4 structured-routing regression gate are in place and green.
- Acceptance proof: The evidence the epic cites (2s imports, 5-9s helps, 43s regen, 0.2 files/s ingest) is retired — each has an owning child whose acceptance names its budget.

Packet: `task_packets/142_polylogue_20d.md`

## 143. polylogue-bby.8 — Web reader perceived performance: virtualized list, streamed search, optimistic navigation

**P2 / feature / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Fluidity in the reader is perceived latency, not just server latency: the session list renders 16k+ rows into the DOM (scroll cost grows with archive size), search waits for full results before painting anything, clicking a session blocks on the full detail fetch, and every panel loads with spinners instead of skeletons. Even with a fast daemon the UI will feel sluggish until the client is engineered for perceived speed. Design direction: Four standard techniques, applied to the existing SPA (sequence after bby.6 extracts the JS to real files — refactoring inline-string JS is not viable): (1) LIST VIRTUALIZATION: render only the viewport window of the session list (hand-rolled windowing is ~100 lines, no framework needed); constant DOM cost at any archive size. (2) SEARCH-AS-YOU-TYPE: debounced (~150ms) incremental search against the daemon (FTS is f…

Implementation spine:
- Four standard techniques, applied to the existing SPA (sequence after bby.6 extracts the JS to real files — refactoring inline-string JS is not viable): (1) LIST VIRTUALIZATION: render only the viewport window of the session list (hand-rolled windowing is ~100 lines, no framework needed)
- constant DOM cost at any archive size.
- (2) SEARCH-AS-YOU-TYPE: debounced (~150ms) incremental search against the daemon (FTS is fast when ready), cancel in-flight requests on new keystrokes (AbortController), paint first page immediately with a 'more loading' tail — never a blank list while typing.
- (3) OPTIMISTIC NAVIGATION: clicking a session paints instantly from the list-row data (title/origin/date skeleton) while messages stream in
- hover-prefetch the detail for the row under the cursor (the cache bead makes this nearly free).

Tests:
- Acceptance proof: Session list scrolls at 60fps with 20k sessions (virtualized DOM stays constant-size).
- Acceptance proof: Search-as-you-type paints first results <300ms with a warm daemon and stale requests are cancelled.
- Acceptance proof: Back-navigation to a visited session renders instantly from client cache and revalidates via cursor.

Packet: `task_packets/143_polylogue_bby_8.md`

## 144. polylogue-opc — Self-tracing: the daemon's own spans land in its own archive

**P2 / feature / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Polylogue has an OTLP receiver and stores spans — and instruments itself with none. Self-tracing closes the loop: daemon HTTP requests, converger stage executions, query compile+execute phases, ingest attempts, cache hits/misses, and embedding drain windows emit spans through the daemon's own OTLP intake into ops.db — making 'why was that slow' a query against the archive instead of a log-reading session. Dogfood value doubles as demo value: the tool debugging itself with its own forensics is the most polylogue-sh… Design direction: (1) A tiny internal tracer (no opentelemetry-sdk dependency — spans are dicts posted to the in-process intake; the OTLP wire format only matters for external emitters): span(name, attrs) context manager instrumenting: HTTP route handlers (route, status, duration), converger stages (per session/batch), archive_query compile vs execute vs render phases, write_effects (per effect once 0aj lands), embed windows, cache l…

Implementation spine:
- (1) A tiny internal tracer (no opentelemetry-sdk dependency — spans are dicts posted to the in-process intake
- the OTLP wire format only matters for external emitters): span(name, attrs) context manager instrumenting: HTTP route handlers (route, status, duration), converger stages (per session/batch), archive_query compile vs execute vs render phases, write_effects (per effect once 0aj lands), embed windows, cache lookups (20d.12).
- Request-id correlation: HTTP handler opens the root span
- downstream spans parent to it.
- (2) Sampling doctrine: routes/stages always-on (cheap, bounded)

Tests:
- Acceptance proof: Spans emitted for routes/stages/query-phases on the seeded corpus daemon
- Acceptance proof: request-id ties a route span to its query-phase children
- Acceptance proof: sampling caps enforced with drop counters visible in /metrics
- Acceptance proof: ops traces --slow renders a span tree for a real slow request
- Acceptance proof: retention pruning works.

Packet: `task_packets/144_polylogue_opc.md`

## 145. polylogue-oxz — Performance instrumentation doctrine: slow-query log, phase timings, logging discipline

**P2 / feature / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Beyond spans, three instrumentation gaps and one doctrine gap: (a) no SLOW-QUERY LOG — SQLite statements over a threshold should be recorded with their text and, on demand, their EXPLAIN QUERY PLAN, or every perf regression starts from scratch (the EQP sweep 20d.7 is a snapshot; this is the continuous version); (b) CLI has no phase breakdown — 'polylogue --debug-timing find X' should print import/config/db-open/compile/execute/render wall per phase (the 1.6s floor was diagnosed by hand; it should be one flag); (c)… Design direction: (1) SLOW-QUERY LOG: sqlite3 set_trace_callback (or the profile hook) on both connection profiles; statements >threshold (default 50ms, y4c-tunable) log normalized SQL + duration + connection profile into ops.db (bounded ring, not unbounded rows); 'ops slow-queries' renders top-N with optional EQP capture on a copy. Overhead check: trace callbacks cost ~nothing when the threshold filter is in C-side profile hook — VE…

Implementation spine:
- (1) SLOW-QUERY LOG: sqlite3 set_trace_callback (or the profile hook) on both connection profiles
- statements >threshold (default 50ms, y4c-tunable) log normalized SQL + duration + connection profile into ops.db (bounded ring, not unbounded rows)
- 'ops slow-queries' renders top-N with optional EQP capture on a copy.
- Overhead check: trace callbacks cost ~nothing when the threshold filter is in C-side profile hook — VERIFY per sqlite3 module semantics
- if Python-side per-statement cost is measurable, gate behind a daemon flag default-on only for write profile.

Tests:
- Acceptance proof: Slow-query log captures a seeded slow statement with duration + normalized SQL and bounded storage
- Acceptance proof: debug-timing prints the phase table and matches span data for daemon-served queries
- Acceptance proof: log-doctrine page committed + print()-diagnostic lint wired into verify quick
- Acceptance proof: webui beacons land in ops.db on the seeded workbench.

Packet: `task_packets/145_polylogue_oxz.md`

## 146. polylogue-6wnh — Bound thread refresh cost for large Codex appends

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Current 3wb closure evidence shows the old 260s append.index.graph_resolve rebuild tail is not active on recent daemon appends, but the worst current graph_resolve sample is still dominated by append.index.graph_resolve.thread_refresh: 3.020976s of a 3.040429s graph_resolve step on a 340.8 MB Codex append. This is not a P1 blocker while raw replay backlog is zero and recent samples are bounded, but it is the concrete next optimization if thread_refresh becomes the next tail. Design direction: Use /realm/tmp/polylogue-workload-graph-tail-499e26363.json as the initial evidence. Inspect the thread_refresh implementation behind append.index.graph_resolve.thread_refresh; determine whether it can refresh only affected thread/session rows instead of rebuilding broader thread projections. Add timing/profiling evidence before editing. Preserve lineage/thread correctness and do not skip real topology updates. If t…

Implementation spine:
- Use /realm/tmp/polylogue-workload-graph-tail-499e26363.json as the initial evidence.
- Inspect the thread_refresh implementation behind append.index.graph_resolve.thread_refresh
- determine whether it can refresh only affected thread/session rows instead of rebuilding broader thread projections.
- Add timing/profiling evidence before editing.
- Preserve lineage/thread correctness and do not skip real topology updates.

Tests:
- Acceptance proof: A focused benchmark or live diagnostic shows thread_refresh cost on giant Codex append/replay rows
- Acceptance proof: either the implementation becomes incremental and the worst recent 340 MiB-class thread_refresh path is materially reduced, or the bead records why the current cost is the correct bounded floor with a guardrail that would catch a regression toward the 260s class.

Packet: `task_packets/146_polylogue_6wnh.md`

## 147. polylogue-ma2 — Add FK-supporting index for web_content_constructs message cleanup

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Live v24 rebuild evidence showed ChatGPT full-replace rows spending seconds in append.index.full_replace.delete_messages. Source review found web_content_constructs.message_id has an ON DELETE CASCADE FK to messages(message_id) but no supporting index; active EXPLAIN planned SELECT 1 FROM web_content_constructs WHERE message_id = ? as SCAN web_content_constructs over about 89k rows. This should be a schema-index slice after v24 convergence, not during the active rebuild. Design direction: After polylogue-6h7 closes, add a canonical index on web_content_constructs(message_id) in the index tier DDL, bump INDEX_SCHEMA_VERSION once as part of a batched schema slice, document the re-ingest plan in docs/internals.md, and add a focused planner/DDL test proving message-id lookups no longer scan the table. Coordinate with any other index-tier changes so one rebuild covers all of them.

Implementation spine:
- After polylogue-6h7 closes, add a canonical index on web_content_constructs(message_id) in the index tier DDL, bump INDEX_SCHEMA_VERSION once as part of a batched schema slice, document the re-ingest plan in docs/internals.md, and add a focused planner/DDL test proving message-id lookups no longer scan the table.
- Coordinate with any other index-tier changes so one rebuild covers all of them.

Tests:
- Acceptance proof: EXPLAIN QUERY PLAN for SELECT 1 FROM web_content_constructs WHERE message_id = ? LIMIT 1 uses the new index on a seeded/current archive
- Acceptance proof: full_replace delete_messages stage timing no longer shows web_content_constructs-driven table scans
- Acceptance proof: schema version docs include the re-ingest plan
- Acceptance proof: no in-place migration helper is added.

Packet: `task_packets/147_polylogue_ma2.md`

## 148. polylogue-th0 — Interactive-surface test harness: pty flows, completions, fuzzy pickers

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The suite (248k lines) is strong on units/properties/snapshots and blind on exactly the surfaces the UX program is now building: nothing drives a real pty, so fzf select flows, the coming judge TUI (p5g), bare-invocation triage (jnj.13), pager behavior, and terminal-width/color rendering are untested by construction; shell completions (fnm.4) have no correctness harness at all (a broken completion script fails silently forever); interactive-ambiguity moments (jnj.11) can regress without any red test. As the CLI gr… Design direction: (1) PTY harness in tests/infra: pexpect (or pty+os primitives — decide by trying pexpect's reliability under pytest-xdist) driving the real CLI binary against the seeded corpus: send keys, assert on screen state with normalized snapshots (strip timing/colors via the existing syrupy terminal-snapshot conventions; explicit width matrix 80/120/200 since fzf layouts shift). Keep the pty lane serial and marked (scale tie…

Implementation spine:
- (1) PTY harness in tests/infra: pexpect (or pty+os primitives — decide by trying pexpect's reliability under pytest-xdist) driving the real CLI binary against the seeded corpus: send keys, assert on screen state with normalized snapshots (strip timing/colors via the existing syrupy terminal-snapshot conventions
- explicit width matrix 80/120/200 since fzf layouts shift).
- Keep the pty lane serial and marked (scale tier) — pty tests are inherently slower
- a dozen golden flows, not hundreds.
- (2) COMPLETION CONTRACTS, no pty needed: invoke the completion entry points directly (Click's shell-complete protocol + the daemon completion endpoint once fnm.4 lands) with a table of (partial-input -> expected candidates) cases generated FROM the grammar registries — the registry is the oracle, so new units/stages get completion tests for free (declare-once payoff).

Tests:
- Acceptance proof: PTY harness runs 5+ golden flows green in CI serial lane
- Acceptance proof: completion contract tests are registry-generated and fail when a unit is added without completion metadata
- Acceptance proof: a deliberate fzf-flow regression (reordered candidates) is caught by the harness in a demonstration commit.

Packet: `task_packets/148_polylogue_th0.md`

## 149. polylogue-yeq — Advanced verification lanes: metamorphic DSL, daemon chaos, API-contract walks

**P2 / task / 08-scale-performance-live / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Beyond coverage-by-example, three method upgrades the suite lacks: (1) METAMORPHIC testing for the query engine — the DSL has algebraic laws (filter commutativity, pipeline-stage composition vs post-hoc filtering, unit-count consistency between find and aggregate paths) that hold for ALL queries, not just the ones we thought to write; hypothesis can generate queries from the grammar and assert the laws, catching lowering bugs example tests never will (the sessions-vs-observed-events pipeline inconsistency from the… Design direction: (1) Metamorphic lane: a hypothesis strategy over the registry grammar (bounded depth, seeded corpus) + ~8 laws as properties: predicate-order invariance, LIMIT monotonicity, group-by-count sums equal ungrouped count, unit-where result parity with equivalent find-mode filters, measure composition associativity once 9l5.7 lands. Laws ARE the spec — failures are either engine bugs or documented non-laws (record which, …

Implementation spine:
- (1) Metamorphic lane: a hypothesis strategy over the registry grammar (bounded depth, seeded corpus) + ~8 laws as properties: predicate-order invariance, LIMIT monotonicity, group-by-count sums equal ungrouped count, unit-where result parity with equivalent find-mode filters, measure composition associativity once 9l5.7 lands.
- Laws ARE the spec — failures are either engine bugs or documented non-laws (record which, in the support matrix fnm.11).
- (2) Chaos lane (marked, serial, Linux-only): spawn polylogued against a scratch archive, inject SIGKILL at staged points (mid-ingest-batch, mid-FTS-rebuild, between lease-acquire and commit — the code has natural hook points via the stage events), restart, assert: no lost committed sessions, no orphan leases after sweep, FTS consistent or honestly non-ready, convergence resumes.
- Reuses the supervisor machinery from devtools verify.
- (3) Ref-walk contract lane: for each daemon list route and MCP list tool, walk every emitted ref/id into its detail routes

Tests:
- Acceptance proof: Metamorphic lane finds-or-proves: run against the current engine and either file real bugs or commit the laws as green properties
- Acceptance proof: chaos lane demonstrates one seeded crash-recovery invariant per staged kill point
- Acceptance proof: ref-walk lane covers 100% of list-emitting routes/tools and fails on a deliberately broken ref in a demonstration commit.

Packet: `task_packets/149_polylogue_yeq.md`

## 150. polylogue-fnm — Query DSL: one grammar owns query semantics; compose instead of multiplying verbs

**P2 / epic / 04-read-contract-query-render / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: The Lark grammar in polylogue/archive/query/expression.py is THE query language; extend in place. Landed since the GH issue: with-projection for all units, field selection for attached units, projection-unit completions. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: The Lark grammar in polylogue/archive/query/expression.py IS the query language — extend it in place, never as a parallel verb/flag path. Baseline already landed: with-projection for all units, field selection for attached units, projection-unit completions. Treat the GH issue thread as input, not authority; this bead's scope statement wins where they conflict. Coordinates with t46 (the DSL becomes the sole owner of…

Implementation spine:
- The Lark grammar in polylogue/archive/query/expression.py IS the query language — extend it in place, never as a parallel verb/flag path.
- Baseline already landed: with-projection for all units, field selection for attached units, projection-unit completions.
- Treat the GH issue thread as input, not authority
- this bead's scope statement wins where they conflict.
- Coordinates with t46 (the DSL becomes the sole owner of query semantics).

Tests:
- Acceptance proof: New query semantics are added to the Lark grammar in polylogue/archive/query/expression.py (grep shows the grammar rule) rather than as a parallel verb or flag.
- Acceptance proof: The landed-since baseline (with-projection all units, field selection for attached units, projection-unit completions) stays green
- Acceptance proof: `explain_query_expression` / `query_units` reflect the grammar.
- Acceptance proof: `devtools verify` is green on DSL tests
- Acceptance proof: `devtools render all --check` is clean for any generated query-surface docs/schemas.

Packet: `task_packets/150_polylogue_fnm.md`

## 151. polylogue-t46 — Contracts own surfaces: delete parallel dispatch and the QA middle layer

**P2 / epic / 04-read-contract-query-render / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Make existing contracts (query DSL, terminal units, refs, read-view profiles, action/route contracts, generated docs/schemas) the actual owners of behavior; delete hand-written parallel surfaces. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: First slices (read-only audit 2026-07-02, Herschel): config aliases (daemon_host/daemon_port, top-level observability), hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle), status JSON compatibility aliases, origin->provider projection bridges in outputs, browser-capture old synthetic-ID recovery. Rule per slice: the contract (DSL/registry/generated schema) becomes the owner, the parall…

Implementation spine:
- First slices (read-only audit 2026-07-02, Herschel): config aliases (daemon_host/daemon_port, top-level observability), hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle), status JSON compatibility aliases, origin->provider projection bridges in outputs, browser-capture old synthetic-ID recovery.
- Rule per slice: the contract (DSL/registry/generated schema) becomes the owner, the parallel surface is deleted in the same PR — replacement-first, no compatibility fronts.
- Regenerate render surfaces after each (openapi, cli-output-schemas, cli-reference).

Tests:
- Acceptance proof: For each listed first slice — config aliases (daemon_host/daemon_port, top-level observability)
- Acceptance proof: hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle)
- Acceptance proof: status JSON compatibility aliases
- Acceptance proof: origin->provider projection bridges in outputs
- Acceptance proof: browser-capture old synthetic-ID recovery — the parallel hand-written surface is DELETED in the same PR that makes the contract (DSL/registry/generated schema) the sole owner (grep confirms the alias/bridge is gone, no compatibility front left).

Packet: `task_packets/151_polylogue_t46.md`

## 152. polylogue-fnm.12 — User-defined query macros: named, composable DSL shorthands in user.db

**P2 / feature / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The highest-leverage runtime configurable found in the preference design pass: operators (and agents) repeat the same filter combinations constantly — 'my real coding sessions' = origin:claude-code-session + repo-scope + exclude-subagents + trailing-90d. Today that is retyped or shell-aliased outside the product. Named macros stored in user.db make the DSL personal: define once, compose anywhere the grammar accepts a predicate, share with agents automatically (they resolve server-side, so MCP/webui/CLI all underst… Design direction: (1) Definition: 'polylogue config macro set mine "origin:claude-code-session exclude:subagents after:-90d"' (and MCP/webui equivalents); stored as typed user.db rows (the y4c settings registry), validated at definition time by compiling against the grammar — a macro that does not parse is refused with the caret error. (2) Reference syntax: @mine inside any query position where a predicate group is valid ('@mine "WAL…

Implementation spine:
- (1) Definition: 'polylogue config macro set mine "origin:claude-code-session exclude:subagents after:-90d"' (and MCP/webui equivalents)
- stored as typed user.db rows (the y4c settings registry), validated at definition time by compiling against the grammar — a macro that does not parse is refused with the caret error.
- (2) Reference syntax: @mine inside any query position where a predicate group is valid ('@mine "WAL contention" | group by model | count').
- Expansion happens in the compiler BEFORE lowering (textual-hygienic: expanded predicates carry their macro provenance for error messages and explain output — 'explain' shows the expansion).
- (3) Composability rules: macros may reference macros (depth-capped, cycle-checked at definition)

Tests:
- Acceptance proof: Define/list/delete macros via CLI+MCP
- Acceptance proof: @macro composes inside find, unit-where, and pipeline queries on the live archive
- Acceptance proof: invalid macro refused at definition with caret
- Acceptance proof: explain shows expansion with provenance
- Acceptance proof: cycle/depth guards tested

Packet: `task_packets/152_polylogue_fnm_12.md`

## 153. polylogue-fnm.14 — find <query> | compact: token-budgeted corpus-compaction projection with drop manifest

**P2 / feature / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The R&D-flywheel enabler: package a queried cohort as a decision-dense, lineage-deduplicated digest for an external LLM, with an honest fidelity manifest. A projection/render preset over the read algebra (CompactProjectionSpec x layout:corpus-compaction-pack) — NOT a context subsystem: compile_context answers "what do I hand an agent to continue"; compact answers "what is the highest-value lowest-spam evidence digest of a COHORT" (cross-session ranking, lineage-family dedup, fairness strata, external manifest — sh…

Implementation spine:
- Identify the currently duplicated surface paths for this behavior.
- Create/extend the shared contract object and route one surface at a time through it.
- Add parity tests across CLI, daemon/API, MCP, and Python facade.
- Delete dead surface-side code after parity is green.

Tests:
- Acceptance proof: Fixture with protocol/tool spam compacts to a digest excluding it with per-material_origin drop counts
- Acceptance proof: failed->fix->verify fixture keeps the pair with refs
- Acceptance proof: fork/resume fixture emits shared prefix once and reports duplicate-prefix omissions
- Acceptance proof: 60k budget test proves the deterministic degradation order
- Acceptance proof: every digest anchor round-trips to a source ref

Packet: `task_packets/153_polylogue_fnm_14.md`

## 154. polylogue-fnm.2 — Projection predicates/windows + render/layout stages on attached units

**P2 / feature / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Declared predicates/windows on attached units (e.g. with messages[role:user, last:20]) and render/layout stages so read packages and demos are declarable in the query rather than per-view flags. Direction: .agent/includes/fables-poly-findings.md. Design direction: Two layers: (1) predicates/windows on attached units — extend the with-stage parse (hand-parsed pipeline region, expression.py ~:1484-1601 where WITH_PROJECTION_SUPPORTED_UNITS is enforced) to accept per-unit bracket args (messages[role:user, last:20]); lower onto the existing exact-session-id fetch in attached_units.py (caps exist: _MAX_ROWS_PER_SESSION=200; field selection landed 867b1d048 — extend that payload, d…

Implementation spine:
- Two layers: (1) predicates/windows on attached units — extend the with-stage parse (hand-parsed pipeline region, expression.py ~:1484-1601 where WITH_PROJECTION_SUPPORTED_UNITS is enforced) to accept per-unit bracket args (messages[role:user, last:20])
- lower onto the existing exact-session-id fetch in attached_units.py (caps exist: _MAX_ROWS_PER_SESSION=200
- field selection landed 867b1d048 — extend that payload, don't fork it).
- (2) render/layout stages — new pipeline stage kind (same touchpoint chain as aggregates: stage parser -> AST/to_payload -> executor -> registry -> completions -> render regen) that binds a read-package/render profile to the query result.
- Keep grammar untouched (stages are hand-parsed)

Tests:
- Acceptance proof: `...
- Acceptance proof: with messages[role:user, last:20]` parses per-unit bracket predicates/windows in the hand-parsed with-stage region and lowers onto the existing exact-session-id fetch in attached_units.py, respecting _MAX_ROWS_PER_SESSION and extending the landed field-selection payload rather than forking it.
- Acceptance proof: Verify: pytest asserts filtered/windowed attached rows and cap enforcement.
- Acceptance proof: A new render/layout pipeline stage binds a read-package/render profile to the result and is picked up by explain via to_payload.
- Acceptance proof: The Lark grammar file is unchanged (stages stay hand-parsed).

Packet: `task_packets/154_polylogue_fnm_2.md`

## 155. polylogue-fnm.4 — Shell completion + fuzzy selection as read-only projections of the grammar registries

**P2 / feature / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Completion/query-builder metadata built on the same grammar+registries used by CLI/MCP/daemon/web — not a second parser. Substantial substrate exists (query_completions tool, projection-unit completions landed 07-03); remaining scope per issue. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Scope per gh#1844 minus what landed: query_completions MCP tool, projection-unit completions, and dynamic shell completions (polylogue config completions --shell) exist. Remaining: completion coverage for pipeline stages/operators/read-view names sourced from the SAME registries (metadata.py descriptors, read_view_registry, action contracts — no second vocabulary); bounded archive-backed value providers (origins, re…

Implementation spine:
- Scope per gh#1844 minus what landed: query_completions MCP tool, projection-unit completions, and dynamic shell completions (polylogue config completions --shell) exist.
- Remaining: completion coverage for pipeline stages/operators/read-view names sourced from the SAME registries (metadata.py descriptors, read_view_registry, action contracts — no second vocabulary)
- bounded archive-backed value providers (origins, repos, tags) with latency caps
- fzf-style fuzzy selection beyond the select verb.
- Acceptance: a completion snapshot test asserts every grammar-reachable field/unit/stage/view appears in the completion payload (registry-diff test, so new DSL work cannot silently miss completions).

Tests:
- Acceptance proof: A registry-diff snapshot test asserts that every grammar-reachable field/unit/pipeline-stage/read-view name (enumerated from metadata.py descriptors, read_view_registry, and operations.action_contracts.ACTION_CONTRACTS) appears in the completion payload, so new DSL work cannot silently miss completions.
- Acceptance proof: Verify: pytest registry-diff test.
- Acceptance proof: Completions for pipeline stages, operators, and read-view names are sourced from those same registries (no second vocabulary — completions.py already imports query_unit_descriptor/terminal_query_pipeline_stage_infos/ACTION_CONTRACTS at completions.py:14-32).
- Acceptance proof: Archive-backed value providers (origins, repos, tags) return under a stated latency cap.
- Acceptance proof: Verify: test measures provider latency against the cap.

Packet: `task_packets/155_polylogue_fnm_4.md`

## 156. polylogue-fnm.6 — Wire the terminal stage to projections: | read / | context-image

**P2 / feature / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: QueryUnitTransformStage is reserved-never-parsed and terminal args are reserved for future actions. Wire the terminal stage so a pipeline can end in a projection: `sessions where ... | read view:dialogue` / `| context-image budget:4000` — queries become complete read/context programs. Same hand-parsed stage chain as aggregates; the read/context compilers already accept the needed specs. Design direction: The terminal args slot is explicitly reserved for this (expression.py:397-407 'future actions (read view, analyze mode, bundle kind)'). Target: `messages where session.repo:x AND text:timeout | limit 40 | context-image budget:4000` and `sessions where ... | read view:temporal` and `| bundle:handoff` — the DSL becomes the single language from selection through composition to rendering. Altitude: terminal keywords in …

Implementation spine:
- The terminal args slot is explicitly reserved for this (expression.py:397-407 'future actions (read view, analyze mode, bundle kind)').
- Target: `messages where session.repo:x AND text:timeout | limit 40 | context-image budget:4000` and `sessions where ...
- | read view:temporal` and `| bundle:handoff` — the DSL becomes the single language from selection through composition to rendering.
- Altitude: terminal keywords in the hand-parsed stage region + an executor registry dispatching to the existing compilers — compile_context already accepts seed queries
- read views resolve via read_view_registry.

Tests:
- Acceptance proof: `sessions where ...
- Acceptance proof: | read view:temporal`, `messages where ...
- Acceptance proof: | limit 40 | context-image budget:4000`, and `...
- Acceptance proof: | bundle:handoff` execute end-to-end: the terminal stage dispatches through an executor registry to the existing compile_context / read_view_registry compilers, and the terminal's output replaces the row payload with a typed projection-artifact envelope per terminal kind.
- Acceptance proof: Verify: pytest asserts each of the three forms returns its envelope type.

Packet: `task_packets/156_polylogue_fnm_6.md`

## 157. polylogue-1lm — Composable transcript views: selector x transform x budget algebra

**P2 / task / 04-read-contract-query-render / blocked-hard**

Mechanism: Issue description localizes the mechanism: 'Prose-only' is one point in a space the operator keeps requesting by example: user messages plus directly-adjacent agent replies (raw-log 07-02 — what the agent intended to report, minus the toil); tool outputs truncated from the middle beyond N lines (raw-log 06-23); decisions-only; tool-skeleton (calls + outcomes, no bodies); failure-slices; reboot-with-refs (37t.3); compact recaps for mass export ('every sinex-related chatlog in compact form for gptpro', 06-18). One algebra: SELECTOR (role, material-origin, bl… Design direction: (1) Extend ProjectionSpec (jnj.1) with typed selector predicates (reuse the DSL block-predicate grammar — no second filter language) and per-class TransformSpec; compile_context and renderers consume the same spec (4p1's Projection axis, deepened). (2) Adjacency selectors are the novel primitive: adjacent-to(role:user, distance<=1, after) via window functions over position. (3) Transforms compose with ap7 semantic r…

Implementation spine:
- (1) Extend ProjectionSpec (jnj.1) with typed selector predicates (reuse the DSL block-predicate grammar — no second filter language) and per-class TransformSpec
- compile_context and renderers consume the same spec (4p1's Projection axis, deepened).
- (2) Adjacency selectors are the novel primitive: adjacent-to(role:user, distance<=1, after) via window functions over position.
- (3) Transforms compose with ap7 semantic renderers
- truncate-middle keeps first/last K lines with an omission marker carrying the block ref (expandable, jgp).

Tests:
- Acceptance proof: The three raw-log examples work as presets/inline specs on the live archive
- Acceptance proof: compile_context and read share the machinery
- Acceptance proof: presets visible to completions
- Acceptance proof: omission markers always carry resolvable refs.

Packet: `task_packets/157_polylogue_1lm.md`

## 158. polylogue-ap7 — Semantic transcript rendering: tool-call-aware, provider-agnostic, shared CLI/web renderer registry

**P2 / task / 04-read-contract-query-render / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Transcripts render today as generic message/block sequences — the same flat treatment for a prose paragraph, a 400-line Bash result, an Edit diff, and a Task dispatch. The archive KNOWS what these are (typed blocks, tool names, structural outcomes, normalized across providers) and renders none of that knowledge. Ambitious target (operator directive): chatlogs that read the way the work actually happened — every standard tool call rendered semantically, across every origin, in both the web reader and the terminal. Design direction: (1) RENDERER REGISTRY keyed by (tool family, provider-normalized): Edit/Write -> syntax-highlighted diff/file cards (the yrx diff reconstruction is the data source — same computation, presentation here); Bash/shell -> terminal-styled block with command, collapsed output (first/last N lines, expand), exit badge from structural outcome; Read/Grep/Glob -> compact file-reference cards (path, range, match count) instead …

Implementation spine:
- (1) RENDERER REGISTRY keyed by (tool family, provider-normalized): Edit/Write -> syntax-highlighted diff/file cards (the yrx diff reconstruction is the data source — same computation, presentation here)
- Bash/shell -> terminal-styled block with command, collapsed output (first/last N lines, expand), exit badge from structural outcome
- Read/Grep/Glob -> compact file-reference cards (path, range, match count) instead of dumped contents
- Task/subagent dispatch -> a subagent card with the dispatch prompt, status, and link into the tree (mission-control bby.9 integration)
- WebFetch/WebSearch -> link cards with domain + title

Tests:
- Acceptance proof: On the seeded corpus: Edit shows a highlighted diff, Bash shows exit-badged folded output, Task shows a linked subagent card — in BOTH web and CLI
- Acceptance proof: unknown tools render as today
- Acceptance proof: structure-parity snapshot tests green across backends
- Acceptance proof: a before/after recording of one real session is committed as the demo asset (3tl.5 machinery).

Packet: `task_packets/158_polylogue_ap7.md`

## 159. polylogue-rxdo — Analysis provenance: queries, result-sets, findings, analyses as first-class objects

**P2 / epic / 05-analysis-provenance-citations / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: THE convergent frontier from the 2026-07-05 R&D program (hit independently by swarm waves 2/4 and multiple GPT-Pro review branches). Today a query is a transient execution: nothing addressable survives it, so analyses cannot be iterative, citable, annotatable, or composable, and Polylogue cannot observe its own use. Target object graph: query:<hash> (content-addressed canonical plan) -> query_run (execution event) -> result_set (relation snapshot with grain+corpus epoch) -> finding (assertion-kind claim, judge lif…

Implementation spine:
- Inventory open child beads and map them to the invariant named by the epic.
- Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
- Close only after child beads are closed or explicitly split out with new blockers.

Tests:
- Acceptance proof: A committed query returns stable query/query-run/result-set refs on every surface
- Acceptance proof: assertions can target them
- Acceptance proof: reset --index cannot destroy promoted/cited result sets
- Acceptance proof: findings live in the existing candidate->judge lifecycle
- Acceptance proof: the twelve recursive-loop failure modes recorded in child beads have guards.

Packet: `task_packets/159_polylogue_rxdo.md`

## 160. polylogue-s7ae — Agent coordination substrate: evidence-backed multi-agent work without tracker lock-in

**P1 / epic / 06-agent-context-coordination / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Why: Polylogue should make concurrent agent work operational, not merely visible. The target is a general coordination evidence layer over sessions, topology, repos/worktrees, work items, activity/resource episodes, context injection, messages, handoffs, and proof evidence. Beads is an important adapter when present, but the system must degrade to GitHub/git/session inference without Beads. The concrete Claude+Codex same-repo workflow should be realizable inside this substrate, not bolted into Polylogue as a speci… Design direction: Core shape: add a reusable coordination envelope, not a web-only mission-control feature. The envelope composes existing Polylogue evidence: sessions, topology_edges, tool/action blocks, session events, context compiler/ledger, blackboard/user rows, daemon/hook liveness, git/worktree state, and optional task-system adapters. WorkItemRef is source-agnostic (beads|github|git|inferred|none) with provenance/confidence. …

Implementation spine:
- Core shape: add a reusable coordination envelope, not a web-only mission-control feature.
- The envelope composes existing Polylogue evidence: sessions, topology_edges, tool/action blocks, session events, context compiler/ledger, blackboard/user rows, daemon/hook liveness, git/worktree state, and optional task-system adapters.
- WorkItemRef is source-agnostic (beads|github|git|inferred|none) with provenance/confidence.
- CoordinationMessage should reuse blackboard/user-state machinery where viable.
- ActivityEpisode should reuse action/tool/session event evidence and add only missing normalization for resource scope/liveness.

Tests:
- Acceptance proof: A typed coordination envelope exists and is queryable without assuming Beads.
- Acceptance proof: It joins active/historical agent session trees with repo/worktree/branch, optional work item refs, activity/resource episodes, coordination messages/advisories, context-flow refs, proof/outcome summaries, and freshness/provenance/confidence.
- Acceptance proof: CLI and MCP expose bounded agent-grade views (status, self, work-item/current, conflicts/overlap, handoff, watch) with JSON-first output.
- Acceptance proof: Web mission control renders the same envelope rather than owning a separate ontology.
- Acceptance proof: Context injection uses the 37t.11 scheduler and ledger.

Packet: `task_packets/160_polylogue_s7ae.md`

## 161. polylogue-rsad — MCP agent ergonomics: oversized responses, boilerplate affordances, metadata-only summaries

**P2 / bug / 06-agent-context-coordination / needs-acceptance-criteria**

Mechanism: MCP responses can flood agent context with large payloads and repetitive boilerplate. The fix is ergonomic and safety-critical: metadata-first summaries with continuation handles.

Implementation spine:
- Add a response-budget policy to MCP read tools: max bytes/items/tokens, with metadata-only fallback.
- Return continuation handles for large results and explicit next-read/open instructions.
- Strip repeated boilerplate; include concise schema/fields only on demand.
- Test prompts and client payloads with realistic large sessions.

Tests:
- Oversized query returns summary + continuation, not full payload.
- Continuation reads exact next page.
- Small responses remain direct.
- No boilerplate repeats across common tool calls.

Packet: `task_packets/161_polylogue_rsad.md`

## 162. polylogue-37t — Agent context/memory loop: declared claims -> judgment -> preamble -> reboot

**P2 / epic / 06-agent-context-coordination / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: The judged-memory loop: agents declare structured claims, the operator judges them, active claims compile into context preambles, and sessions reboot into compact evidence packs. Substrate exists (assertions, compile_context, compose_context_preamble, SessionStart hook); these children wire the loop closed. Raw-log design criteria (2026-06-29): entries timestamped, expiry metadata, navigable origin refs, restrained injection with expandable indices/refs. Design direction: Epic spine: the loop closes when a declared claim travels end-to-end through all four named stages against the live archive. Stage owners: CLAIMS = 37t.2 (author-declared markers -> candidate assertions) + 37t.1 (assertion consumer wiring); JUDGMENT = operator bulk review/accept/reject of candidate assertions (currently unowned by any child — needs a judgment-queue bead); PREAMBLE = 37t.4 (SessionStart rollout) refa…

Implementation spine:
- Epic spine: the loop closes when a declared claim travels end-to-end through all four named stages against the live archive.
- Stage owners: CLAIMS = 37t.2 (author-declared markers -> candidate assertions) + 37t.1 (assertion consumer wiring)
- JUDGMENT = operator bulk review/accept/reject of candidate assertions (currently unowned by any child — needs a judgment-queue bead)
- PREAMBLE = 37t.4 (SessionStart rollout) refactored onto 37t.11 (ContextSource scheduler/arbiter)
- REBOOT = 37t.3 (reboot-with-refs).

Tests:
- Acceptance proof: A seeded end-to-end scenario test demonstrates one claim flowing claims->judgment->preamble->reboot: an agent emits a declared marker (37t.2) that lands as a candidate assertion, the operator accepts it via the judgment queue, it appears as a ref in a compiled SessionStart preamble for the matching repo (37t.4/37t.11), and it survives a reboot-with-refs handoff (37t.3) resolvable via resolve_ref.
- Acceptance proof: Verify: a pytest covering the four-stage path (e.g.
- Acceptance proof: tests/unit/context/test_judged_memory_loop.py) plus MCP resolve_ref on the emitted ref.
- Acceptance proof: Every named stage (claims/judgment/preamble/reboot) has an owning open bead with non-null acceptance_criteria
- Acceptance proof: no stage survives only as prose in a sibling's design field.

Packet: `task_packets/162_polylogue_37t.md`

## 163. polylogue-cpf — Land the six doctrines: time, writers, finding-provenance, degraded-modes, non-goals, injected-context trust

**P2 / epic / 06-agent-context-coordination / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: The six doctrine texts were written in full in the 2026-07-03 design session (session transcript is the source — Claude-Session trailer on this commit); they cover the cheap-to-write, expensive-to-lack gaps: time semantics (three times, UTC epoch-ms canon, skew tolerance, duration honesty), writer classes (four classes, one writer-class per file, cross-tier interruption validity), finding provenance (five-part stanza, re-runs supersede, semantic version bumps flag stale findings), degraded-mode ladder (five rungs,… Design direction: (1) Commit texts under docs/doctrine/ (or internals sections — match the ttu docs-IA tiering), adjusted to repo voice; link from architecture-spine. (2) Enforcement hooks, each small: schema-audit check for TEXT timestamps in new DDL; writer-class docstring convention + layering check; provenance-stanza refusal in the findings lane (3tl.4); degraded-rung declaration in feature review; trust-class typing in the Conte…

Implementation spine:
- (1) Commit texts under docs/doctrine/ (or internals sections — match the ttu docs-IA tiering), adjusted to repo voice
- link from architecture-spine.
- (2) Enforcement hooks, each small: schema-audit check for TEXT timestamps in new DDL
- writer-class docstring convention + layering check
- provenance-stanza refusal in the findings lane (3tl.4)

Tests:
- Acceptance proof: Six doctrine documents committed and indexed
- Acceptance proof: the three cheap lints wired (timestamp DDL check, provenance stanza gate, trust deny-lexicon fixture)
- Acceptance proof: architecture-spine links them
- Acceptance proof: bd memory updated to point at doctrines instead of restating them.

Packet: `task_packets/163_polylogue_cpf.md`

## 164. polylogue-rii — Live substrate intake: agents write work-events; evidence materializes in-loop

**P2 / epic / 06-agent-context-coordination / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Invert the relationship for live agents: work lands in Polylogue as it happens (push), and the agent reads context/evidence back in-loop. OPERATOR GATE: direction confirmed as worth phasing, full program needs explicit green-light before a large build. Hermes-specific ingestion lives in the Hermes bridge program; this program owns the generic write-leg and intake seams. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Invert the relationship for live agents: work lands in Polylogue as it happens (push) and the agent reads context/evidence back in-loop. OPERATOR GATE: the direction is confirmed worth phasing, but the full program needs an explicit green-light before a large build. This epic owns the GENERIC write-leg and intake seams (rii.1 is the first child); Hermes-specific ingestion lives in the Hermes bridge (fs1). Treat the …

Implementation spine:
- Invert the relationship for live agents: work lands in Polylogue as it happens (push) and the agent reads context/evidence back in-loop.
- OPERATOR GATE: the direction is confirmed worth phasing, but the full program needs an explicit green-light before a large build.
- This epic owns the GENERIC write-leg and intake seams (rii.1 is the first child)
- Hermes-specific ingestion lives in the Hermes bridge (fs1).
- Treat the GH issue thread as input, not authority

Tests:
- Acceptance proof: The generic write-leg + intake seam scope is defined and split into child beads (rii.1 = the agent work-event write-leg)
- Acceptance proof: Hermes-specific ingestion is explicitly excluded and pointed at fs1.
- Acceptance proof: The program stays gated: no large build starts until an explicit operator green-light is recorded as a bead comment.
- Acceptance proof: The epic advances when rii.1 lands and an agent's pushed work-event materializes into the run-projection read-models within one convergence cycle (see rii.1 acceptance).

Packet: `task_packets/164_polylogue_rii.md`

## 165. polylogue-37t.1 — Assertions: consumer wiring + lifecycle tightening for unified overlays

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Assertion substrate is the live path; remaining work is consumer wiring + lifecycle (promotion, staleness, expiry). Unwired kinds exist: handoff, prompt_eval, highlight. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Wiring points (verify current state first): unwired AssertionKinds handoff/prompt_eval/highlight need first writers + surface registration (user_audit every-kind-has-a-surface invariant will force the surface entry; scope_ref/author_ref must use a registered ObjectRef kind — see #2383 memory). Lifecycle: add staleness/expiry semantics to claims consumed by the preamble compiler (ASSERTION_CLAIM_KINDS reads ACTIVE on…

Implementation spine:
- Wiring points (verify current state first): unwired AssertionKinds handoff/prompt_eval/highlight need first writers + surface registration (user_audit every-kind-has-a-surface invariant will force the surface entry
- scope_ref/author_ref must use a registered ObjectRef kind — see #2383 memory).
- Lifecycle: add staleness/expiry semantics to claims consumed by the preamble compiler (ASSERTION_CLAIM_KINDS reads ACTIVE only — extend with expiry check rather than a new status)
- judgment surfaces: list/accept/reject exist via MCP assertion tools — the gap is operator-ergonomic review flow (bulk judge candidate batches).
- Raw-log criteria: timestamped entries, expiry metadata, navigable origin refs.

Tests:
- Acceptance proof: First writers exist for the three currently-unwired AssertionKinds (handoff, prompt_eval, highlight — present-but-writerless at core/enums.py:409/423/426) and each passes the user_audit every-kind-has-a-surface invariant with a registered ObjectRef scope_ref/author_ref.
- Acceptance proof: Verify: the user_audit surface reports zero surfaceless kinds
- Acceptance proof: pytest asserts a written row per kind.
- Acceptance proof: Preamble-consumed claims gain expiry: the ASSERTION_CLAIM_KINDS admission read (user_write.py:~1520) extends its ACTIVE-only filter with an expiry check (no new status).
- Acceptance proof: Test: an expired claim is excluded from the preamble compiler input.

Packet: `task_packets/165_polylogue_37t_1.md`

## 166. polylogue-37t.2 — Inline annotation protocol: agent-authored structure in plain prose

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Agents write structured markers in prose; extraction at block enrichment turns them into candidate assertions with evidence refs. Author-declared notation, not heuristic mining — the construct-validity-safe way to get structure out of prose. Design direction: PROTOCOL DESIGN (2026-07-03, generalizing the marker idea into a composable protocol): (1) SYNTAX: line-anchored sigil markers — '::kind(args): body' on its own line, inline '[[kind: body]]' for short spans. Chosen for: harness-agnostic (plain text works in ANY provider incl. web chats), streaming-safe (line-complete before parse), markdown-inert (harmless where uninterpreted), collision-resistant (escape via '\::')…

Implementation spine:
- PROTOCOL DESIGN (2026-07-03, generalizing the marker idea into a composable protocol): (1) SYNTAX: line-anchored sigil markers — '::kind(args): body' on its own line, inline '[[kind: body]]' for short spans.
- Chosen for: harness-agnostic (plain text works in ANY provider incl.
- web chats), streaming-safe (line-complete before parse), markdown-inert (harmless where uninterpreted), collision-resistant (escape via '\::').
- Decide the final sigil after a corpus collision scan — grep the live archive for candidate-prefix false positives
- evidence over taste.

Tests:
- Acceptance proof: Final sigil chosen after a corpus collision scan: grep the live archive for candidate-prefix false positives and record the scan result in the PR.
- Acceptance proof: Line-anchored '::kind(args): body' and inline '[[kind: body]]' parse at block enrichment into typed rows with exact message/block provenance
- Acceptance proof: malformed markers extract as kind=malformed with raw text (never silently dropped) and the malformed rate is a recorded measure.
- Acceptance proof: Verify: pytest over fixtures covering well-formed, malformed, markdown-inert, streaming-split, and '\::' escaped inputs.
- Acceptance proof: Kind registry is declare-once: adding a kind (note/claim/lesson/decision/predict/handoff/anchor/bead/eval) is a registry entry, not a parser change.

Packet: `task_packets/166_polylogue_37t_2.md`

## 167. polylogue-37t.3 — Reboot-with-refs: session self-compaction protocol

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Agent reboots into a fresh session carrying all prose verbatim with every tool exchange collapsed to a one-line expandable ref — better than harness compaction because refs resolve via resolve_ref. Raw-log 06-29: refs over stripping; hierarchical expansion affordances. Design direction: Flow: agent calls MCP compile_context with ContextSpec(seed_refs=[current session], purpose=continue) + a new prose_with_refs segment profile -> markdown with authored prose verbatim, tool_use/result collapsed to '-> [tool:Bash exit:0 4.2s] pytest ... <ref:action:...>'; new harness session; SessionStart hook injects when POLYLOGUE_REBOOT_FROM=<session_id> marker present (source field startup|resume|clear|compact; do…

Implementation spine:
- Flow: agent calls MCP compile_context with ContextSpec(seed_refs=[current session], purpose=continue) + a new prose_with_refs segment profile -> markdown with authored prose verbatim, tool_use/result collapsed to '-> [tool:Bash exit:0 4.2s] pytest ...
- <ref:action:...>'
- new harness session
- SessionStart hook injects when POLYLOGUE_REBOOT_FROM=<session_id> marker present (source field startup|resume|clear|compact
- don't tax ordinary startups).

Tests:
- Acceptance proof: compile_context with a new prose_with_refs segment profile emits authored prose verbatim while every tool_use/result collapses to a one-line '<ref:action:...>' marker.
- Acceptance proof: Verify: pytest asserts each emitted ref resolves via resolve_ref back to the original block.
- Acceptance proof: Budget rule enforced: prose verbatim to 60% of budget, then oldest prose collapses to one-line recaps
- Acceptance proof: first user message + last N turns kept verbatim
- Acceptance proof: overflow recorded as ContextOmission(reason=budget) (ContextOmission at context/compiler.py:48).

Packet: `task_packets/167_polylogue_37t_3.md`

## 168. polylogue-37t.6 — Session-aware devshell entry: surface what the last agent session left behind

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: On cd/direnv entry, print what the last agent session in this cwd left: unresolved blackboard blocker/question notes, the last session's terminal state, resume candidates for this directory. All reads exist (blackboard_list with unresolved filter; find_resume_candidates already scores cwd at 0.15 weight) — this is a status-line/devshell-hook integration away. Keep it one bounded line + an expand command; restrained-injection rule applies. Design direction: On cd/direnv entry, print one bounded line summarizing what the last agent session in this cwd left behind: unresolved blackboard blocker/question notes (blackboard_list unresolved filter), the last session's terminal state, and resume candidates for this directory (find_resume_candidates, which already scores cwd at 0.15 weight). All reads exist; this is a devshell-hook / status-line integration. Keep it one bounde…

Implementation spine:
- On cd/direnv entry, print one bounded line summarizing what the last agent session in this cwd left behind: unresolved blackboard blocker/question notes (blackboard_list unresolved filter), the last session's terminal state, and resume candidates for this directory (find_resume_candidates, which already scores cwd at 0.15 weight).
- All reads exist
- this is a devshell-hook / status-line integration.
- Keep it one bounded line plus an expand command and apply the restrained-injection rule (no noisy dumps
- suppress when there is nothing to report).

Tests:
- Acceptance proof: 1.
- Acceptance proof: A devshell/direnv entry hook prints a single bounded line for the current cwd combining unresolved blackboard-note count, the last session's terminal state, and the top resume candidate(s), using existing reads (blackboard_list unresolved filter, find_resume_candidates) with no new query machinery.
- Acceptance proof: 2.
- Acceptance proof: An expand command shows the full detail
- Acceptance proof: the entry line stays one line and suppresses itself when there is nothing to report (restrained injection).

Packet: `task_packets/168_polylogue_37t_6.md`

## 169. polylogue-37t.7 — Close the failure loop: verify postmortem -> next session's context seed

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: workspace failure-context produces an envelope (testmon graph + git history + fixtures for a failing test); the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session. Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify` entry. The obvious first consumer is the devloop itself after a red verify run. Design direction: `workspace failure-context` produces an envelope (testmon graph + git history + fixtures for a failing test) and the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session. Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify`…

Implementation spine:
- `workspace failure-context` produces an envelope (testmon graph + git history + fixtures for a failing test) and the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session.
- Build the splice: a compile_context seed constructed from the latest verify postmortem + failure-context envelope, injectable via the SessionStart hook or an explicit `polylogue context --from-verify`.
- First consumer: the devloop itself after a red verify run.
- Test discipline: session-cut recovery drills (chaos-lane, yeq) — deliberately kill sessions mid-work and measure whether the next session recovers unprompted from injected context alone.

Tests:
- Acceptance proof: A compile_context seed is constructed from the latest verify postmortem (.cache/verify/) + the `workspace failure-context` envelope (testmon graph + git history + fixtures), injectable via the SessionStart hook or an explicit `polylogue context --from-verify` entry point.
- Acceptance proof: The first consumer is wired: the devloop injects the seed after a red verify run.
- Acceptance proof: Verify: `polylogue context --from-verify` on a real red postmortem emits a seed containing the failing test plus the implicated files
- Acceptance proof: `devtools test <context seed test>` green.
- Acceptance proof: Session-cut recovery drills (chaos-lane, yeq) are run: sessions are killed mid-work and the next session's unprompted recovery from injected context alone is measured

Packet: `task_packets/169_polylogue_37t_7.md`

## 170. polylogue-37t.8 — Resume routing: map a session to the harness invocation that reopens it

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Genuinely-missing item: nothing owns 'reopen this session in its harness' — claude --resume <id> vs the codex equivalent, per origin; plus detecting an already-open interactive session (the kitty/hyprland control plane can answer that on this machine, but keep that integration optional/pluggable). Natural terminal action for the continue verb and the last mile of the resumption loop: find ... then continue should end with the session actually open. Design direction: Add a mapping from an archived session to the harness invocation that reopens it, per origin: `claude --resume <id>` for Claude Code, the Codex equivalent, etc. Optionally detect an already-open interactive session via the kitty/hyprland control plane, kept behind a pluggable/optional interface. This is the last mile of the `continue` verb: `find ... then continue` should end with the session actually open (or the e…

Implementation spine:
- Add a mapping from an archived session to the harness invocation that reopens it, per origin: `claude --resume <id>` for Claude Code, the Codex equivalent, etc.
- Optionally detect an already-open interactive session via the kitty/hyprland control plane, kept behind a pluggable/optional interface.
- This is the last mile of the `continue` verb: `find ...
- then continue` should end with the session actually open (or the exact reopen command emitted).

Tests:
- Acceptance proof: 1.
- Acceptance proof: A resume-routing helper maps (origin, native session id) to the concrete harness reopen command, covering at least Claude Code (`claude --resume <id>`) and Codex, with an explicit unsupported/unknown result for origins that have no reopen path.
- Acceptance proof: 2.
- Acceptance proof: The `continue` action (or `find ...
- Acceptance proof: | continue`) emits or executes the correct reopen invocation for the selected session.

Packet: `task_packets/170_polylogue_37t_8.md`

## 171. polylogue-90y — In-page overlay: Polylogue presence on chat sites — archive state, context, assertion capture

**P2 / feature / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The extension currently only EXTRACTS; it could also PRESENT. A tasteful injected surface on chatgpt.com/claude.ai (shadow-DOM isolated, keyboard-summonable, per-site opt-in) turns every chat page into a Polylogue-aware surface: is this chat archived and through when; what has this conversation cost; what does the archive already know that is relevant (judged assertions matching the current topic); and — the operator-flagged killer feature — CREATE and EDIT assertions directly from the page: select any passage -> … Design direction: TASTE CONSTRAINTS FIRST (operator: 'must feel non-crappy'): shadow-DOM component, zero layout shift on the host page (fixed corner chip + slide-over panel, never inline injection into the chat column); respects prefers-color-scheme; one keyboard chord (e.g. Alt+P) summons/dismisses; a per-site toggle and a global kill in the popup; NO badges on messages, NO buttons sprayed into the page — selection-triggered afforda…

Implementation spine:
- TASTE CONSTRAINTS FIRST (operator: 'must feel non-crappy'): shadow-DOM component, zero layout shift on the host page (fixed corner chip + slide-over panel, never inline injection into the chat column)
- respects prefers-color-scheme
- one keyboard chord (e.g.
- Alt+P) summons/dismisses
- a per-site toggle and a global kill in the popup

Tests:
- Acceptance proof: On chatgpt.com and claude.ai: chip+panel render with zero host-page layout shift in light and dark themes
- Acceptance proof: selection pill appears only on text selection
- Acceptance proof: saving a selection creates a candidate assertion whose evidence ref resolves to the exact archived message
- Acceptance proof: per-site toggle and global kill work
- Acceptance proof: panel shows relevant judged assertions when embeddings are enabled.

Packet: `task_packets/171_polylogue_90y.md`

## 172. polylogue-37t.14 — Recursive-safety substrate: citation anchors, provenance edges, grounding verdicts (closed-loop/cycle/drift)

**P2 / task / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: THE load-bearing safety invariant for a self-ingesting archive (browser capture auto-ingests the operator R&D chats; distilled findings become assertions; assertions can inject into future context — the recovery-digest fabrication class generalizes to: an agent claim laundered through other agent claims until it re-enters context as truth). Substrate: assertion_citation_anchors (evidence ref resolved to typed anchor with grounding_class {human_message, human_judgment, tool_result, source_raw, git_commit, external_… Design direction: Derived verdict tier (fail-closed when absent) + durable policy mutation on quarantine (candidate + inject:false + quarantine_reason in context_policy; NO new AssertionStatus axis — reuse candidate machinery). Cycle detection: recursive CTE with path-string membership, depth backstop; quarantined edges block auto-promotion. Drift: anchors store resolved-against hash; converger pass flags mismatches and forces inject…

Implementation spine:
- Derived verdict tier (fail-closed when absent) + durable policy mutation on quarantine (candidate + inject:false + quarantine_reason in context_policy
- NO new AssertionStatus axis — reuse candidate machinery).
- Cycle detection: recursive CTE with path-string membership, depth backstop
- quarantined edges block auto-promotion.
- Drift: anchors store resolved-against hash

Tests:
- Acceptance proof: The laundering scenario is structurally blocked in a test: agent assertion citing only agent sessions/assertions never appears in compiled context
- Acceptance proof: adding a git/tool-result/human-judgment anchor or judging it releases it
- Acceptance proof: cycle + drift each independently quarantine
- Acceptance proof: a transcript-only anchor cannot release a world-claim (compatibility matrix test).
- Acceptance proof: Verify: focused tests over the CTE + scheduler gate fixture.

Packet: `task_packets/172_polylogue_37t_14.md`

## 173. polylogue-37t.4 — SessionStart preamble opt-in rollout (polylogue + sinnix repos)

**P2 / task / 06-agent-context-coordination / blocked-hard**

Mechanism: Issue description localizes the mechanism: Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first (operator decision). Preamble presence is arm B of the uplift experiment; rollout and experiment reinforce each other. Restrained injection — indices/refs over dumps (raw-log criterion). Design direction: Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first. Session-start mechanics (2026-07-03 research): (1) SOURCE-AWARE — the SessionStart payload carries source (startup|resume|clear|compact): fresh startup gets the repo brief (active beads pointer, last session outcome, judged lessons for this repo); resume gets a delta-since-last-session (new sessions…

Implementation spine:
- Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first.
- Session-start mechanics (2026-07-03 research): (1) SOURCE-AWARE — the SessionStart payload carries source (startup|resume|clear|compact): fresh startup gets the repo brief (active beads pointer, last session outcome, judged lessons for this repo)
- resume gets a delta-since-last-session (new sessions/commits/beads touching this repo since the resumed session's end)
- compact gets NOTHING from polylogue (bd prime already reinjects task state
- double-injection burns budget — jgp restraint).

Tests:
- Acceptance proof: compose_context_preamble is wired into the existing SessionStart hook (upgrading sessionstart-polylogue-recall.sh, not adding a second hook) for polylogue + sinnix.
- Acceptance proof: Verify: hook fires and injects on `polylogue` SessionStart in each repo
- Acceptance proof: pytest asserts the single-hook path.
- Acceptance proof: Source-aware branching verified by test: source=startup injects the repo brief, source=resume injects a since-last-session delta, source=compact injects zero polylogue bytes.
- Acceptance proof: Hard token cap enforced (default ~600): a test that an oversized brief degrades to refs rather than exceeding the cap.

Packet: `task_packets/173_polylogue_37t_4.md`

## 174. polylogue-60i5 — Durable-tier batch coordination: one user v4->v5 and one source v2->v3 migration window

**P2 / task / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Cross-cutting operational constraint (the single biggest insight across the R&D specs): MANY pending designs each want a durable-tier bump — user v5: recursive-safety columns, content-variants tables, s7ae coordination messages, config-engine settings, queries/result-sets/analyses (rxdo.2, rxdo.8); source v3: compaction snapshots, ingest-fidelity fingerprints, secret-redaction tombstones, zstd blob placement. Durable migrations run behind verified backup manifests one user_version step at a time — landing them pie… Design direction: Process: when >=2 labeled beads are execution-ready, declare a window, write migrations NNN as one coherent set per tier, verify backup manifest, apply one step, run parity tests. The z7rv migration framework (closed 2026-07-04) is the runner substrate. Index-tier (v24->v25) batching is separate and cheaper (blue-green b5l removes downtime) but the same batching discipline applies — coordinate via the schema-bumps b…

Implementation spine:
- Process: when >=2 labeled beads are execution-ready, declare a window, write migrations NNN as one coherent set per tier, verify backup manifest, apply one step, run parity tests.
- The z7rv migration framework (closed 2026-07-04) is the runner substrate.
- Index-tier (v24->v25) batching is separate and cheaper (blue-green b5l removes downtime) but the same batching discipline applies — coordinate via the schema-bumps bd memory.

Tests:
- Acceptance proof: First batch window executed with a single user-tier migration covering all ready v5 consumers
- Acceptance proof: no durable migration lands outside a declared window.
- Acceptance proof: Verify: migration chain contiguity test + backup manifest check.

Packet: `task_packets/174_polylogue_60i5.md`

## 175. polylogue-9e5.24 — Sink MCP analysis primitives into insights/ + api facade; delete surface-side math

**P2 / task / 06-agent-context-coordination / implementation-ready-after-local-inspection**

Mechanism: Design direction: server_insight_tools.py implements analysis math directly in the MCP surface, unreachable from CLI/library: correlate_sessions (:826 Pearson r + metric-name->field map), find_similar_sessions metadata lane (:652 weighted heuristic), aggregate_sessions/workflow_shape_distribution/find_abandoned_sessions (:510/:233/:298 GROUP-BY + severity-rank + ISO-week), tool_call_latency_distribution (:131 nearest-rank percentile)…

Implementation spine:
- server_insight_tools.py implements analysis math directly in the MCP surface, unreachable from CLI/library: correlate_sessions (:826 Pearson r + metric-name->field map), find_similar_sessions metadata lane (:652 weighted heuristic), aggregate_sessions/workflow_shape_distribution/find_abandoned_sessions (:510/:233/:298 GROUP-BY + severity-rank + ISO-week), tool_call_latency_distribution (:131 nearest-rank percentil…
- Move each into insights/ (archive_rollups.py owns aggregate reducers
- portfolio.py _distribution/DistributionStat is the canonical percentile
- metadata similarity beside SessionNeighborCandidate) and expose via api/insights.py so MCP, CLI, and the library share one definition.
- This is the read/execution half split out from the 9e5.16 parity AUDIT (which stays read-only per the 9e5 rule).

Tests:
- Acceptance proof: correlate/find_similar-metadata/aggregate/workflow_shape/find_abandoned/tool_call_latency/compare have api facade methods and the MCP tools call them (grep shows no math/GROUP-BY left in server_insight_tools.py)
- Acceptance proof: the severity map, similarity weights, and week-bucketing are defined once in insights/
- Acceptance proof: a CLI or library caller produces byte-identical aggregates to the MCP tool for a fixture archive
- Acceptance proof: devtools verify green.
- Acceptance proof: Cross-refs polylogue-9e5.16.

Packet: `task_packets/175_polylogue_9e5_24.md`

## 176. polylogue-4ts — Session lineage truth: shared content stored once, counted once, composed correctly

**P1 / epic / 03-lineage-compaction-truth / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Fork/resume/compaction share content; storage+aggregates ignored it. v12-v14 landed prefix-dedup + composition; this program owns the residuals. Design doc docs/design/session-lineage-model.md. Operator: 'broken unless modeled correctly' — correctness > demo-ladder. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

Implementation spine:
- Inventory open child beads and map them to the invariant named by the epic.
- Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
- Close only after child beads are closed or explicitly split out with new blockers.

Tests:
- Acceptance proof: Terminal state: shared content stored once (prefix dedup verified on live archive), counted once (4ts.2), composed correctly (read paths serve full logical transcripts across the branch-point matrix)
- Acceptance proof: external citation of archive counts uses logical grain with the physical figure footnoted.

Packet: `task_packets/176_polylogue_4ts.md`

## 177. polylogue-gjg — Compaction lifecycle: pre-compaction snapshot, loss forensics, post-compaction re-grounding

**P2 / epic / 03-lineage-compaction-truth / blocked-hard**

Mechanism: Issue description localizes the mechanism: Compaction is where the OS-like context-management vision meets the harness's own memory management, and today Polylogue only observes its AFTERMATH (acompact lineage edges, v12). Three gaps: nothing snapshots the full pre-compaction context (the harness summarizes-and-discards; what was lost is unknowable after the fact); nothing measures what compaction costs (which facts/decisions/refs present before are absent after — the construct behind every 'the agent forgot' complaint); and re-grounding after compaction i… Design direction: (1) SNAPSHOT: wire the PreCompact hook (VERIFY availability/payload in current Claude Code — the hook catalog moves; Codex equivalent via app-server events ox0) to capture the full context state as an artifact (session-linked, blob-stored, content-addressed — dedup makes repeated compactions cheap). Fallback without the hook: the JSONL up to the compaction boundary IS the pre-state; the snapshot adds what JSONL lack…

Implementation spine:
- (1) SNAPSHOT: wire the PreCompact hook (VERIFY availability/payload in current Claude Code — the hook catalog moves
- Codex equivalent via app-server events ox0) to capture the full context state as an artifact (session-linked, blob-stored, content-addressed — dedup makes repeated compactions cheap).
- Fallback without the hook: the JSONL up to the compaction boundary IS the pre-state
- the snapshot adds what JSONL lacks (the exact assembled context, if the payload provides it).
- (2) FORENSICS: a compaction-loss measure (9l5.7-registered): diff pre-snapshot against the post-compact continuation's early context — structurally extractable items (file paths, refs, tool outcomes, decisions marked via 37t.2 notation) present-before/absent-after

Tests:
- Acceptance proof: PreCompact snapshots land for real compactions on the operator machine (or the JSONL-boundary fallback is implemented and labeled)
- Acceptance proof: the loss measure runs corpus-wide with tier=structural and renders an epidemiology table
- Acceptance proof: re-grounding injects only under the flag and its arm comparison is defined as an ExperimentSpec
- Acceptance proof: 37t epic description carries the handoff-triad map.

Packet: `task_packets/177_polylogue_gjg.md`

## 178. polylogue-1vpm — Work-graph units: delegation, episode, artifact edges — the derived units between lineage and analysis

**P2 / epic / 10-analytics-experiments / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Three convergent derived units that make "what work actually happened" queryable, sitting ABOVE within-provider lineage (session_links stays the leaf truth) and BELOW analysis runs. (1) DELEGATION: provider-neutral rows mined from Claude Task tool_use blocks (excluding agent-acompact-* which is compaction, not delegation), Codex source.subagent.thread_spawn (forked_from_id alone proves parentage, NOT subagent-ness), session_runs.role=subagent, and SubagentReport extraction — parent/child refs, instruction/result b…

Implementation spine:
- Inventory open child beads and map them to the invariant named by the epic.
- Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
- Close only after child beads are closed or explicitly split out with new blockers.

Tests:
- Acceptance proof: delegations where / episodes where work as terminal units with set-algebra participation
- Acceptance proof: fixtures prove no false subagent from bare forked_from_id and no acompact false-delegation
- Acceptance proof: episode default render includes only linked+corroborated tiers
- Acceptance proof: per-edge signal contributions auditable in evidence_json
- Acceptance proof: operator stitch decisions round-trip as assertions and constrain rebuilds.

Packet: `task_packets/178_polylogue_1vpm.md`

## 179. polylogue-9l5 — Outcome-grounded analytics: the archive answers 'so what' questions

**P2 / epic / 10-analytics-experiments / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: The archive answers 'so what' questions. Tower map (2026-07-03 design pass): LAYER 0 substrate (exists) — profiles, work events, phases, threads, cost rollups with five-axis accounting, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes. LAYER 1 descriptive (children .1-.6): outcome-conditioned, cross-provider, epidemiology, token economy, saved views, tool episodes. LAYER 2 statistical honesty (.7): uncertainty primitives + the measure registry with con… Design direction: Epic: the archive answers 'so what' questions, layered over the Layer-0 substrate (profiles, work events, phases, threads, five-axis cost rollups, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes). Layer 1 descriptive (children .1-.6), Layer 2 statistical honesty (.7 uncertainty primitives + measure registry with construct-validity metadata, the keystone…

Implementation spine:
- Epic: the archive answers 'so what' questions, layered over the Layer-0 substrate (profiles, work events, phases, threads, five-axis cost rollups, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes).
- Layer 1 descriptive (children .1-.6), Layer 2 statistical honesty (.7 uncertainty primitives + measure registry with construct-validity metadata, the keystone), Layer 3 temporal (.8), Layer 4 duration/sequence (.9 survival, .10 process mining), Layer 5 causal (experiment hosting), Layer 6 predictive (.11), plus cross-cutting measures (.12) and the semantic layer (mhx.5).
- Composition rule: every layer lands as registered measures over the query algebra (fnm/4p1), measure x grouping x window x comparison x uncertainty, never as bespoke analyze modes
- construct validity is enforced by the registry.

Tests:
- Acceptance proof: 1.
- Acceptance proof: All child beads (9l5.1-.12 and folded-in measures) are closed (`bd show polylogue-9l5 --json` shows no open children).
- Acceptance proof: 2.
- Acceptance proof: Every delivered analytic lands as a registered measure over the query algebra (fnm/4p1), not a bespoke analyze mode
- Acceptance proof: the measure registry (.7) enforces evidence tier + sample frame + confounds per measure and renders tier footnotes in every output.

Packet: `task_packets/179_polylogue_9l5.md`

## 180. polylogue-9l5.1 — Outcome-conditioned analytics: cost/duration/retries/tools by structural success

**P2 / feature / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Group cost, duration, retry chains, and tool usage by structural outcome (exit_code/is_error terminal state), with per-origin coverage caveats. The includes High-Value backlog names this directly. Consumes the action outcome fields; surfaces through analyze projections + DSL aggregates + MCP insight tools — one relation, three surfaces. Design direction: Anchored examples (all one step from existing substrate): cost of failed vs clean sessions; failure-rate by model VERSION; retry cascade depth; 'sessions where >30% of tool calls errored' (needs the child-count DSL predicate or the relation directly). Keystone fields tool_result_is_error/exit_code + the actions view are the ground truth; outcomes are captured today but analytics still mostly counts and sums. This is…

Implementation spine:
- Anchored examples (all one step from existing substrate): cost of failed vs clean sessions
- failure-rate by model VERSION
- retry cascade depth
- 'sessions where >30% of tool calls errored' (needs the child-count DSL predicate or the relation directly).
- Keystone fields tool_result_is_error/exit_code + the actions view are the ground truth

Tests:
- Acceptance proof: 1.
- Acceptance proof: One shared relation groups cost, duration, retry-chain depth, and tool-mix by structural outcome (terminal tool_result_is_error / exit_code from the actions view — never assistant prose), reachable identically through the analyze projection, a DSL aggregate, and an MCP insight tool (one relation, three surfaces returning the same numbers).
- Acceptance proof: 2.
- Acceptance proof: A per-origin coverage caveat (from the 9e5.3 column-honesty audit) renders on every grouped row.
- Acceptance proof: 3.

Packet: `task_packets/180_polylogue_9l5_1.md`

## 181. polylogue-9l5.2 — Cross-provider comparative analytics

**P2 / feature / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The archive is the only place Claude/Codex/ChatGPT/Gemini work traces coexist normalized: same task-shape comparisons — failure rates, retry behavior, cost per completed session, tool-mix, session lengths — by origin/model with explicit coverage tiers per origin so partial provenance cannot masquerade as a finding. This relation is also what the public leaderboard variant reads. Design direction: The killer query shape: 'same repo, same month: Claude Code vs Codex — turns per task, $/session, tool-failure rate, subagent usage' — no lab and no observability vendor can run it; the archive is the only place these providers coexist normalized. Honesty by construction: the coverage matrix (storage/usage.py:51-139) already annotates exact vs estimated accounting per origin — every comparison row carries its covera…

Implementation spine:
- The killer query shape: 'same repo, same month: Claude Code vs Codex — turns per task, $/session, tool-failure rate, subagent usage' — no lab and no observability vendor can run it
- the archive is the only place these providers coexist normalized.
- Honesty by construction: the coverage matrix (storage/usage.py:51-139) already annotates exact vs estimated accounting per origin — every comparison row carries its coverage tier as a footnote, so partial provenance cannot masquerade as a finding.
- This relation is also what the public leaderboard variant reads.
- THE $0 LANE (fables interop analysis): once local-model sessions exist in the archive (Hermes/Ollama behind the LiteLLM gateway), the same comparison gains a free-lane column — local-model vs API harnesses on the same repo and task class: turns, failure rates, wall-clock, and actual cost $0 vs the API-equivalent counterfactual the api_equivalent cost axis already computes.

Tests:
- Acceptance proof: 1.
- Acceptance proof: On the seeded corpus a cross-origin same-task comparison (turns/task, $/session, tool-failure rate, subagent usage) renders WITH a per-origin coverage-tier footnote on EVERY row, sourced from the storage/usage.py coverage matrix (exact vs estimated per origin).
- Acceptance proof: 2.
- Acceptance proof: A comparison where one origin lacks priced provenance is REFUSED as a bare number at composition and returns an actionable error (the 9l5.7 composition/honesty guard), not a silent partial.
- Acceptance proof: 3.

Packet: `task_packets/181_polylogue_9l5_2.md`

## 182. polylogue-9l5.6 — tool-episodes projection: call + result + outcome + context + next action

**P2 / feature / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: Sidecar research (Sartre): affordance-usage and analyze tools stop at aggregate evidence. A first-class tool-episodes projection — tool call, paired result, outcome status, surrounding context, what the agent did next, caveats — supports Serena/codebase-memory utility evaluation and is the natural drill-down unit under every aggregate. Likely reuses the action outcome fields + followup_class machinery from the campaign. Design direction: New derived read model `tool_episodes` (rebuildable; registry pattern under polylogue/storage/insights/session/, registered in insights/registry.py so CLI+MCP inherit it). Each episode joins a tool_use block to its paired tool_result via the existing `actions` view and carries: the keystone structural outcome fields (tool_result_is_error, tool_result_exit_code, index schema v16), a bounded surrounding-context window…

Implementation spine:
- New derived read model `tool_episodes` (rebuildable
- registry pattern under polylogue/storage/insights/session/, registered in insights/registry.py so CLI+MCP inherit it).
- Each episode joins a tool_use block to its paired tool_result via the existing `actions` view and carries: the keystone structural outcome fields (tool_result_is_error, tool_result_exit_code, index schema v16), a bounded surrounding-context window (prev/next K messages), followup_class (from the closed sru.1 keystone), and per-episode caveats (unknown-outcome NULL vs structural).
- Surfaces: (a) an `analyze` drill-down projection, (b) a DSL `tool-episodes` unit that is the natural drill-down under affordance-usage / analyze-tools aggregates, (c) an MCP tool.
- Aggregates OVER episodes register as MeasureSpecs via 9l5.7

Tests:
- Acceptance proof: 1.
- Acceptance proof: On the seeded/demo corpus `tool-episodes` is queryable and each row carries call + paired result + structural outcome (is_error/exit_code) + surrounding-context window + next-action + caveat.
- Acceptance proof: 2.
- Acceptance proof: A drill-down from an affordance-usage (or analyze-tools) aggregate cell returns exactly the underlying episodes for that cell.
- Acceptance proof: 3.

Packet: `task_packets/182_polylogue_9l5_6.md`

## 183. polylogue-9l5.8 — Temporal analytics: trends, rolling baselines, changepoint detection

**P2 / feature / 10-analytics-experiments / blocked-hard**

Mechanism: Issue description localizes the mechanism: The archive spans years of daily work but has no time-axis analytics beyond day/week summaries: no trend ('is my silent-proceed rate improving?'), no baseline ('is today's cost anomalous vs my trailing month?'), no changepoint ('did failure rates shift when I switched models / enabled hooks / upgraded the harness?'). Changepoints are the construct-valid way to talk about interventions without a controlled experiment — locate the shift, then check whether it coincides with a known event (harness release, config com… Design direction: (1) Series builder as a DSL stage: any measure over any unit grouped into time buckets -> a typed series ('measure tool_failure_rate window week over 2026') — one series primitive, every measure gains a time axis for free (composability payoff). (2) Rolling baselines: trailing-window median/MAD bands (robust to heavy-tailed cost/latency); points outside k*MAD flag as anomalies — same machinery serves cost_outlook up…

Implementation spine:
- (1) Series builder as a DSL stage: any measure over any unit grouped into time buckets -> a typed series ('measure tool_failure_rate window week over 2026') — one series primitive, every measure gains a time axis for free (composability payoff).
- (2) Rolling baselines: trailing-window median/MAD bands (robust to heavy-tailed cost/latency)
- points outside k*MAD flag as anomalies — same machinery serves cost_outlook upgrades and daemon health anomaly lines (cursor_lag_baseline already does a bespoke version — converge it onto this).
- (3) Changepoint detection: offline PELT or binary segmentation on series (ruptures library under [analytics]
- fallback: simple binary segmentation is ~60 lines)

Tests:
- Acceptance proof: A series stage composes with any registered measure on the seeded corpus.
- Acceptance proof: Rolling-baseline anomaly flags reproduce a seeded anomaly scenario.
- Acceptance proof: Changepoint output on a synthetic step-series locates the step and renders it as candidate + nearby-events annotation, never as a causal claim.

Packet: `task_packets/183_polylogue_9l5_8.md`

## 184. polylogue-3uw — Capture-completeness: the instrument's coverage error as a standing measure

**P2 / task / 10-analytics-experiments / blocked-hard**

Mechanism: Issue description localizes the mechanism: Convergence legibility answers 'how converged is what we ingested'; nothing answers 'how much of what EXISTS did we ingest'. Sessions known to have happened (hook SessionStart fired, harness wrote a file, extension saw a chat) versus sessions fully archived = the coverage error, per origin, over time. An instrument that does not know its own coverage error cannot honestly caveat its findings — and silent capture regressions (the hibernation threat) currently have no number to trip on. Design direction: Three evidence sources joined against the archive: hook events (SessionStart without a matching archived session after a grace window = a miss), watcher-root file inventories (files seen vs raw rows), extension capture-gap events (3v1). Materialize as a per-origin coverage measure (9l5.7 registry, tier=structural) with a trailing-window trend; surface in ops status + daemon health (alert on regression) + the day pag…

Implementation spine:
- Three evidence sources joined against the archive: hook events (SessionStart without a matching archived session after a grace window = a miss), watcher-root file inventories (files seen vs raw rows), extension capture-gap events (3v1).
- Materialize as a per-origin coverage measure (9l5.7 registry, tier=structural) with a trailing-window trend
- surface in ops status + daemon health (alert on regression) + the day page's open-loops.
- The drift sentinel (da1) alerts on shape drift
- this alerts on VOLUME drift — together they are the hibernation-mode floor instrumentation.

Tests:
- Acceptance proof: Coverage renders per origin on the live archive with the known-miss list drillable to refs
- Acceptance proof: a seeded missed-session scenario trips the health alert
- Acceptance proof: findings' sample-frame stanzas can cite the coverage number for their window.

Packet: `task_packets/184_polylogue_3uw.md`

## 185. polylogue-9l5.15 — Triage frontier: worth_reviewing_score + TRIAGED lifecycle — an inbox that empties

**P2 / task / 10-analytics-experiments / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: A context-free frontier over all ~16K logical sessions (inverts the cwd-coupled find_resume_candidates): time-invariant worth_reviewing_score materialized with a decomposable breakdown (unresolved blockers, open questions, decision density, terminal state), collapsed by logical_session_id; inverted-U staleness applied at READ time (materialized staleness goes stale). TRIAGED assertion kind (resumed / wont_resume / archived / snoozed:<until>) makes it a true inbox that empties via WHERE NOT EXISTS triaged; snooze-w… Design direction: The score is a NAMED-FEATURE LINEAR COMBINATION with per-feature contributions visible in the payload (rigor doctrine: no opaque scores) — every feature is a computable structural signal that already exists or has an owning bead: unacknowledged_failure (tool_result_is_error=1 with no subsequent success of a normalized-same command in-session — the 'failed and moved on' signature), abnormal_termination (session ends …

Implementation spine:
- The score is a NAMED-FEATURE LINEAR COMBINATION with per-feature contributions visible in the payload (rigor doctrine: no opaque scores) — every feature is a computable structural signal that already exists or has an owning bead: unacknowledged_failure (tool_result_is_error=1 with no subsequent success of a normalized-same command in-session — the 'failed and moved on' signature), abnormal_termination (session end…
- Weights start hand-set, tuned only against operator triage decisions once TRIAGED data exists (the lifecycle IS the label source
- no invented ground truth).
- LIFECYCLE: worth_reviewing surfaces sessions into a triage view (CLI + webui inbox)
- operator verdicts (reviewed-useful / reviewed-noise / ignore-kind) are assertions (kind=judgment, scope=session) that (a) empty the inbox and (b) accumulate into the weight-tuning set.

Tests:
- Acceptance proof: Frontier returns logical representatives with score breakdown + confidence
- Acceptance proof: triage/snooze removes rows via runtime method
- Acceptance proof: disposable clean-finish rows zero out while blocker sessions surface
- Acceptance proof: demoted buckets visible.
- Acceptance proof: Verify: fixture corpus + scorer tests.

Packet: `task_packets/185_polylogue_9l5_15.md`

## 186. polylogue-mhx — Embedding substrate: provider-general, honest lifecycle, retrieval that earns its cost

**P2 / epic / 09-embeddings-retrieval / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Current state: one hardcoded cloud provider (Voyage voyage-4, 1024-dim, constants in sqlite_vec_support.py), vec0 fixed-dimension tables, embedding targets limited to authored prose messages (v21 partial index), opt-in daemon catch-up with cost caps, hybrid RRF + --semantic/--similar surfaces, ops embed onboarding group. Gaps this program owns: provider/model generality (local AND cloud through one abstraction), an explicit answer to WHAT gets embedded and why, retrieval quality measured instead of assumed, lifecy… Design direction: Epic scope: provider/model generality (one OpenAI-compatible embedding client covering local and cloud), an explicit embedding-target policy (what gets a vector and why), retrieval quality measured rather than assumed, lifecycle honesty (staleness, model switches, spend), and the advanced uses that justify the lane (semantic recall in context compilation, clustering/topics, novelty detection). Doctrine anchor: embed…

Implementation spine:
- Epic scope: provider/model generality (one OpenAI-compatible embedding client covering local and cloud), an explicit embedding-target policy (what gets a vector and why), retrieval quality measured rather than assumed, lifecycle honesty (staleness, model switches, spend), and the advanced uses that justify the lane (semantic recall in context compilation, clustering/topics, novelty detection).
- Doctrine anchor: embeddings.db is a rebuildable tier, so model/dimension switches are tier resets with cost preflight, never in-place migrations.
- Delivered through child beads: mhx.1 (provider abstraction), mhx.2 (target policy), folded-in 37t.5 (local-lane acceptance demo), 0k6 (changed-text staleness), mhx.5 (semantic layer), 0ns (bounded per-session work).

Tests:
- Acceptance proof: 1.
- Acceptance proof: All child beads (mhx.*, folded-in 37t.5, 0k6, 0ns) are closed (`bd show polylogue-mhx --json` shows no open children).
- Acceptance proof: 2.
- Acceptance proof: Provider generality demonstrated end-to-end: a local (qwen3-class) embedding model through the LiteLLM gateway (127.0.0.1:4000) backfills the seeded corpus and `polylogue find --semantic <q>` returns sane neighbors at $0 (mhx.1 acceptance).
- Acceptance proof: 3.

Packet: `task_packets/186_polylogue_mhx.md`

## 187. polylogue-a7xr.10 — Kill-or-adopt the search-provider lane: production bypasses the abstraction it should use

**P2 / chore / 11-interoperability-origin / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: VERIFIED 2026-07-06: FTS5Provider/HybridSearchProvider/factories have zero production call sites — only their own tests import them. Production FTS is inline SQL (archive_tiers/archive.py:4545/4661/7668) and --retrieval-lane hybrid re-implements fusion inline at cli/archive_query.py:830-852. OPERATOR REFRAME (2026-07-06): non-use may indict the SURFACES, not the abstraction — a CLI module implementing retrieval semantics inline violates the substrate-owns-meaning rule, and mhx.3's four-lane bake-off (FTS / dense /… Design direction: If ADOPT: define RetrievalLane protocol from what production actually needs (query, candidate set, scores, lane metadata for the eval payload); implementations wrap the existing inline SQL (fts lane), SqliteVecProvider (dense lane), reciprocal_rank_fusion (hybrid), reranker (mhx.1's client); cli/archive_query.py:830-852 becomes lane dispatch; the current FTS5Provider/HybridSearchProvider bodies are salvage-or-delete…

Implementation spine:
- If ADOPT: define RetrievalLane protocol from what production actually needs (query, candidate set, scores, lane metadata for the eval payload)
- implementations wrap the existing inline SQL (fts lane), SqliteVecProvider (dense lane), reciprocal_rank_fusion (hybrid), reranker (mhx.1's client)
- cli/archive_query.py:830-852 becomes lane dispatch
- the current FTS5Provider/HybridSearchProvider bodies are salvage-or-delete per method (most likely delete — their tests test invented semantics, keep test_hybrid_laws property shapes if the fusion laws transfer).
- If KILL: delete classes+factories+SearchProvider protocol+dead tests

Tests:
- Acceptance proof: A decision recorded WITH mhx.3 (adopt or kill, one paragraph of why)
- Acceptance proof: if adopt: all production retrieval flows through the lane interface, inline fusion in archive_query.py gone, mhx.3 bake-off consumes the lanes, goldens unchanged
- Acceptance proof: if kill: zero references remain, mhx.3 notes it owns lane construction.
- Acceptance proof: Either way devtools verify green.

Packet: `task_packets/187_polylogue_a7xr_10.md`

## 188. polylogue-fs1 — Hermes bridge: state.db + runtime spans -> canonical evidence -> forensics/eval export

**P2 / epic / 11-interoperability-origin / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Positioning: Hermes acts; Polylogue remembers and explains what the agent did; Sinex knows what was happening on the machine around it. Hermes already HAS observability hooks (observer layer, middleware, Langfuse, NeMo Relay ATOF/ATIF export) — so 'add observability' and 'trajectory export' are not the wedge. The wedge: Hermes traces are product-local operational data; Polylogue turns them into a durable, cross-provider, longitudinal evidence corpus with provenance, cost semantics, git correlation, and machine-con… Design direction: Bridge Hermes's durable state.db (sessions, messages, tool calls, token/cache/reasoning/cost counters, parent sessions, compaction/archive/rewind state, FTS) plus observer/runtime spans into canonical Polylogue evidence — NOT the optional ~/.hermes/sessions/session_*.json snapshots (a compatibility adapter, likely off by default). The wedge is turning product-local operational traces into a durable, cross-provider, …

Implementation spine:
- Bridge Hermes's durable state.db (sessions, messages, tool calls, token/cache/reasoning/cost counters, parent sessions, compaction/archive/rewind state, FTS) plus observer/runtime spans into canonical Polylogue evidence — NOT the optional ~/.hermes/sessions/session_*.json snapshots (a compatibility adapter, likely off by default).
- The wedge is turning product-local operational traces into a durable, cross-provider, longitudinal evidence corpus with provenance, cost semantics, git correlation, and machine-context windows.
- Subsumes the old gh#2460 parse_hermes work-structure scope (delegation/tasks/lineage come from state.db + spans).
- Flagship deliverable: a '2-minute polylogue forensics of Hermes' export artifact.
- VERIFY current Hermes internals first (analysis snapshot is v0.18.0

Tests:
- Acceptance proof: Current Hermes internals are verified against the live build before building — the state.db schema (sessions, messages, tool calls, token/cache/reasoning counters, costs, parent sessions, compaction/archive/rewind, FTS) is confirmed and the verification note is recorded on the bead.
- Acceptance proof: The bridge reads Hermes state.db + observer/runtime spans (not the ~/.hermes/sessions snapshots) into canonical Polylogue evidence rows carrying provenance, cost semantics, git correlation, and machine-context windows.
- Acceptance proof: The flagship deliverable is defined and reproducible from a fixture: a '2-minute polylogue forensics of Hermes' forensics/eval export (e.g.
- Acceptance proof: Atropos JSONL round-trip).
- Acceptance proof: gh#2460 delegation/tasks/lineage scope is covered from state.db + spans

Packet: `task_packets/188_polylogue_fs1.md`

## 189. polylogue-exb — Layering: substrate rings import the api facade (6 sites, 2 private-symbol reaches)

**P2 / task / 11-interoperability-origin / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The architecture says surfaces adapt over substrate, but the dependency arrow runs backwards in at least six places: storage/embeddings/preflight.py imports select_pending_embedding_session_window from polylogue.api; storage/embeddings/materialization.py and insights/correlation_view.py import api.sync.bridge.run_coroutine_sync; storage/repair.py imports the PRIVATE api.archive._rebuild_archive_session_insights; sources/live/batch.py imports the whole Polylogue facade; pipeline/run_stages.py imports the PRIVATE ap… Design direction: Per-site relocation, then close the gate: (1) _active_archive_root -> config/core (it is runtime-root resolution, nothing facade-y about it); (2) run_coroutine_sync -> a core/asyncbridge module (two substrate rings need it; the api.sync home is an accident of history); (3) _rebuild_archive_session_insights: repair orchestration needs the insight-rebuild primitive — move the primitive into insights/ or pipeline/ and …

Implementation spine:
- Per-site relocation, then close the gate: (1) _active_archive_root -> config/core (it is runtime-root resolution, nothing facade-y about it)
- (2) run_coroutine_sync -> a core/asyncbridge module (two substrate rings need it
- the api.sync home is an accident of history)
- (3) _rebuild_archive_session_insights: repair orchestration needs the insight-rebuild primitive — move the primitive into insights/ or pipeline/ and have BOTH api and repair call it downward
- (4) select_pending_embedding_session_window -> storage.embeddings owns pending-window selection already (sql.py) — the api re-export should be the alias, not the source

Tests:
- Acceptance proof: The six inward imports are relocated (grep for 'polylogue.api' under storage/, sources/, insights/, pipeline/ returns nothing, including function-local).
- Acceptance proof: layering.yaml disallows polylogue/api for all four substrate rings and devtools verify layering passes.
- Acceptance proof: No behavior change: testmon-affected suite green.

Packet: `task_packets/189_polylogue_exb.md`

## 190. polylogue-3tl — External legibility: a stranger can understand, run, and cite Polylogue

**P1 / epic / 12-external-legibility-demos / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Every finished artifact proves the substrate is honest; this program makes the project legible to someone with no context. The gap is weeks, not months (fables positioning analysis): the value exists but is illegible from outside. Core diagnosis: the problem is category anchoring, not absence of explanation — name the category ('the system of record for AI work') rather than borrowing chat-viewer/observability/memory/QS buckets that all mis-frame it. Deliverable set: README rewrite around the named category and fo…

Implementation spine:
- Inventory open child beads and map them to the invariant named by the epic.
- Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
- Close only after child beads are closed or explicitly split out with new blockers.

Tests:
- Acceptance proof: Terminal state: a stranger can (1) understand from the README's first screen, (2) run the one-command demo successfully, (3) cite a published finding URL.
- Acceptance proof: All three verified by a cold-reader pass from someone/something with no project context.

Packet: `task_packets/190_polylogue_3tl.md`

## 191. polylogue-212 — Demo portfolio: construct-valid demos (D1/D2/D4/D5/D8 + post-hoc forensic Q&A)

**P2 / epic / 12-external-legibility-demos / epic-needs-child-closure**

Mechanism: Issue description localizes the mechanism: Ground rule for all: every displayed number resolves, on click or --explain, to structural evidence (outcome fields, usage events, provenance refs, raw bytes) — never regex over prose. Each runs on the deterministic demo corpus (seed 1843) for public reproduction + a live-archive operator variant. D3 (resurrect a dead session) is covered by the context-loop preamble bead + uplift campaign; D6 (Wrapped/one-year-four-assistants) is the forensics campaign artifact; D7 (candidates on trial) is the context-loop judgmen… Design direction: Portfolio contract (see 212.7): every demo = executable PROMPT.md emitting the uniform Demo Finding Packet; product primitives only, shell as glue; anti-demo (212.8) ships beside successes. IDEA MENU: a 60-item grounded demo catalog from the 2026-07-06 corpus digestion is preserved at .agent/scratch/corpus-gpt-pro-2026-07-06/D-demos.md — pull from it when extending the portfolio; most items converge on six primitive…

Implementation spine:
- Portfolio contract (see 212.7): every demo = executable PROMPT.md emitting the uniform Demo Finding Packet
- product primitives only, shell as glue
- anti-demo (212.8) ships beside successes.
- IDEA MENU: a 60-item grounded demo catalog from the 2026-07-06 corpus digestion is preserved at .agent/scratch/corpus-gpt-pro-2026-07-06/D-demos.md — pull from it when extending the portfolio
- most items converge on six primitives now tracked elsewhere (query runs rxdo.3, cohorts rxdo.2, annotation batches rxdo.7, artifact edges 1vpm.3, analysis runs rxdo.8, context-compile runs 37t.11/gjg.4).

Tests:
- Acceptance proof: Each demo child (212.1 post-hoc forensic Q&A, 212.2 D1, 212.3 D2, 212.4 D4, 212.5 D5, 212.6 D8) ships in two variants: (a) a public seeded-corpus variant (seed 1843) reproducible with one documented command, and (b) a live-archive operator variant.
- Acceptance proof: GROUND RULE: every displayed number resolves, on click or --explain, to structural evidence (outcome fields, usage events, provenance refs, raw bytes) — never regex over prose.
- Acceptance proof: COMPOSITIONALITY: every demo decomposes into product primitives (DSL queries, saved views, read-package layouts, render profiles, workflow-registry entries)
- Acceptance proof: shell/python is glue only, and any bespoke logic beyond glue is first filed and built as a product primitive.
- Acceptance proof: D3/D6/D7 are explicitly out of scope (covered by the context-loop/uplift/forensics campaigns).

Packet: `task_packets/191_polylogue_212.md`

## 192. polylogue-avg — Fold devloop claim-guard vocabulary upstream into ops status/readiness

**P2 / feature / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The loop scripts guard claims better than the product does: devloop-status treats schema-version match as 'openable, not converged', gates convergence claims on raw-materialization debt being zero/classified, and blocks latency claims behind live_performance_proof_blocked. polylogue ops status should expose the same claim-guard vocabulary to ordinary users: a 'what you may claim' section (archive openable / converged / search-ready / perf-measurable) derived from the same signals, instead of leaving the discipline… Design direction: Add a claim-guard section to `polylogue ops status`/readiness that derives 'what you may claim' (archive openable / converged / search-ready / perf-measurable) from the same signals the devloop scripts already use: schema-version match => openable-not-converged; raw-materialization debt zero/classified => converged; FTS freshness => search-ready; the live_performance_proof_blocked gate => perf-measurable. Then have …

Implementation spine:
- Add a claim-guard section to `polylogue ops status`/readiness that derives 'what you may claim' (archive openable / converged / search-ready / perf-measurable) from the same signals the devloop scripts already use: schema-version match => openable-not-converged
- raw-materialization debt zero/classified => converged
- FTS freshness => search-ready
- the live_performance_proof_blocked gate => perf-measurable.
- Then have devloop-status consume the product surface instead of recomputing its own claim vocabulary (silo collapse).

Tests:
- Acceptance proof: `polylogue ops status --json` exposes a claim-guard block with the four claim states (openable / converged / search-ready / perf-measurable), each derived from its documented signal.
- Acceptance proof: Verify: run the command and assert the block and derivations.
- Acceptance proof: An archive that is openable-but-not-converged reports converged=false with the raw-materialization reason string.
- Acceptance proof: Verify: test seeds unmaterialized raw debt and checks the reason.
- Acceptance proof: devloop-status calls the product surface and stops computing its own claim vocabulary.

Packet: `task_packets/192_polylogue_avg.md`

## 193. polylogue-3tl.9 — Docs-and-visuals ownership: coverage lint + regenerable visuals as a standing devloop gate

**P2 / task / 12-external-legibility-demos / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The operator wants agents to comprehensively OWN external-facing docs and visual material, not touch them opportunistically. The repo already has the machinery pattern (render all --check, doc-commands linter, pages build, visual-tapes) but no coverage contract: nothing fails when a public surface ships undocumented, when a doc references a dead flag (doc-commands covers commands only), or when a screenshot/GIF rots against current UI. Docs drift is currently discovered by humans reading. Design direction: (1) COVERAGE LINT (devtools verify docs-coverage): every public CLI command/verb, MCP tool, config key, and daemon route must be reachable from the docs tree (generated inventories make this a set diff, same pattern as the topology gate); new-surface-without-docs fails the lane with the exact missing entry named (actionable-error discipline from o21). (2) VISUAL FRESHNESS: every committed screenshot/GIF must be a vi…

Implementation spine:
- (1) COVERAGE LINT (devtools verify docs-coverage): every public CLI command/verb, MCP tool, config key, and daemon route must be reachable from the docs tree (generated inventories make this a set diff, same pattern as the topology gate)
- new-surface-without-docs fails the lane with the exact missing entry named (actionable-error discipline from o21).
- (2) VISUAL FRESHNESS: every committed screenshot/GIF must be a visual-tapes artifact with a spec (3tl.5 machinery) — a render pass regenerates them against the seeded corpus
- drift = the regen diff exceeds a perceptual threshold -> flagged for re-record.
- No hand-shot images in docs.

Tests:
- Acceptance proof: 1.
- Acceptance proof: New lane `devtools verify docs-coverage`: builds generated inventories of every public CLI command/verb, MCP tool, config key, and daemon route and fails when any is not reachable from the docs tree, naming the exact missing entry (actionable-error discipline, same set-diff pattern as the topology gate).
- Acceptance proof: Passes on the current tree.
- Acceptance proof: 2.
- Acceptance proof: Visual freshness: every committed screenshot/GIF is a visual-tapes artifact with a spec (3tl.5 machinery)

Packet: `task_packets/193_polylogue_3tl_9.md`

## 194. polylogue-bby.11 — Webui architecture v2: the stack that can carry the ambition

**P1 / feature / 99-horizon-or-general / implementation-ready-after-local-inspection**

Mechanism: Issue description localizes the mechanism: The roadmap now on the reader (mission control, timeline+firehose, replay, pinboard, day page, command palette, semantic renderers, SSE-live everything) cannot be built in JS-in-Python-strings, and shouldn't be built three views deep before the foundation is chosen. This bead decides and scaffolds the stack, sized for CODING AGENTS as the builders: maximum training-data familiarity, typed end-to-end, componentized, testable, self-contained (strict no-CDN/offline posture preserved). Design direction: (1) STACK DECISION with rationale: TypeScript + Preact + Vite. Preact because React idioms are the deepest vein of agent training data at 4KB runtime cost (React itself rejected for size; Svelte/Solid rejected for thinner agent familiarity; no-build HTM rejected because losing TypeScript forfeits the mypy-equivalent net the whole codebase strategy relies on). Vite dev server proxies to the daemon (the dev-loop bead …

Implementation spine:
- (1) STACK DECISION with rationale: TypeScript + Preact + Vite.
- Preact because React idioms are the deepest vein of agent training data at 4KB runtime cost (React itself rejected for size
- Svelte/Solid rejected for thinner agent familiarity
- no-build HTM rejected because losing TypeScript forfeits the mypy-equivalent net the whole codebase strategy relies on).
- Vite dev server proxies to the daemon (the dev-loop bead 5en integrates).

Tests:
- Acceptance proof: Scaffold merged: typed generated API client, SSE/cache module, tokens, palette, routing
- Acceptance proof: list + reader views reach parity with the old SPA on the seeded corpus (including the bby.7 ref walk) and the old SPA's list/reader are retired
- Acceptance proof: devtools render webui reproduces byte-identical committed dist in CI
- Acceptance proof: a coding agent added one new view (the judge queue) purely against the scaffold docs — the agent-buildability proof.

Packet: `task_packets/194_polylogue_bby_11.md`

