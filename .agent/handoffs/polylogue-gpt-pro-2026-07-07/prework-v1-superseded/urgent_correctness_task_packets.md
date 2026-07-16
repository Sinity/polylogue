# Polylogue urgent/correctness bead static prework

Generated from the Beads export and the unpacked Polylogue source snapshot. The goal is to turn urgent and correctness-critical beads into coding-agent task packets where the remaining work is mostly implementation plus verification.

This is not a live patch. It is a static diagnosis/handoff package. Each packet contains the bead, the likely failure mechanism, files/functions to inspect, implementation shape, tests, and a verification lane. Line numbers refer to the unpacked source tree in this analysis environment: `/mnt/data/work2/polylogue/tree/polylogue`.

Execution rule: take packets in the listed order unless a lower packet becomes a blocker for the current one. For each packet, first reproduce the current failure or missing invariant with a focused test, then patch the single choke point, then run the focused verification lane. Avoid broad refactors unless the packet explicitly calls for one.

## Packet index

01. `polylogue-s7ae.6` — Classify the aborted full verification run before coordination deploy (P1, release-gate, ready-now / evidence-work) → [tasks/01_polylogue-s7ae.6.md](tasks/01_polylogue-s7ae.6.md)
02. `polylogue-37t.15` — Force non-user assertion writes through candidate/non-injected policy (P1, agent-write-safety, ready-now / code-local) → [tasks/02_polylogue-37t.15.md](tasks/02_polylogue-37t.15.md)
03. `polylogue-kwsb.1` — Daemon/capture Host, Origin, receiver-token, and spool hardening (P1, security, ready-now / code-local with extension smoke) → [tasks/03_polylogue-kwsb.1.md](tasks/03_polylogue-kwsb.1.md)
04. `polylogue-8jg9.4 + polylogue-8jg9.2` — Make ops-doctor orphan cleanup use the same lease/generation invariants as blob GC (P1/P2, blob-integrity, ready-now / code-local) → [tasks/04_polylogue-8jg9.4_plus_polylogue-8jg9.2.md](tasks/04_polylogue-8jg9.4_plus_polylogue-8jg9.2.md)
05. `polylogue-83u.4` — Separate source-tier referenced-blob debt from index attachment-acquisition debt (P1, blob-integrity, ready-now / diagnostic-code) → [tasks/05_polylogue-83u.4.md](tasks/05_polylogue-83u.4.md)
06. `polylogue-9e5.28` — Make the rigor audit iterate the full insight registry, not only existing contracts (P1, evidence-honesty, ready-now / code-local) → [tasks/06_polylogue-9e5.28.md](tasks/06_polylogue-9e5.28.md)
07. `polylogue-9e5.29` — Distinguish absent evidence from true numeric zero at field level (P1, evidence-honesty, needs-small-spec then code) → [tasks/07_polylogue-9e5.29.md](tasks/07_polylogue-9e5.29.md)
08. `polylogue-9e5.30` — Tag prose-mined forensic fields as text-derived (P1, evidence-honesty, ready-now / model-and-renderer) → [tasks/08_polylogue-9e5.30.md](tasks/08_polylogue-9e5.30.md)
09. `polylogue-cpf.5` — Propagate weakest temporal provenance through aggregates (P1, temporal-honesty, ready-now / code-local with schema propagation) → [tasks/09_polylogue-cpf.5.md](tasks/09_polylogue-cpf.5.md)
10. `polylogue-cpf.6` — Inject clock seam for relative-date parsing and audit sort_key_ms epoch pins (P1, temporal-honesty, ready-now / code-local plus audit artifact) → [tasks/10_polylogue-cpf.6.md](tasks/10_polylogue-cpf.6.md)
11. `polylogue-f2qv.2` — Normalize Codex/Claude token lanes into disjoint uncached/cache/reasoning/completion fields (P2, usage-cost-correctness, ready-now / parser+tests) → [tasks/11_polylogue-f2qv.2.md](tasks/11_polylogue-f2qv.2.md)
12. `polylogue-f2qv.1` — Make per-model rollups partition usage events instead of duplicating session totals (P2, usage-cost-correctness, ready-now / storage-rollup) → [tasks/12_polylogue-f2qv.1.md](tasks/12_polylogue-f2qv.1.md)
13. `polylogue-f2qv.4` — Use one LiteLLM-backed pricing resolver and remove tokencost/second maps (P2, usage-cost-correctness, ready-now / grep-and-contract) → [tasks/13_polylogue-f2qv.4.md](tasks/13_polylogue-f2qv.4.md)
14. `polylogue-f2qv.3` — Report API-equivalent dollars and subscription credits as separate fields (P2, usage-cost-correctness, ready-now after lanes/pricing) → [tasks/14_polylogue-f2qv.3.md](tasks/14_polylogue-f2qv.3.md)
15. `polylogue-f2qv.5` — Version-gate provider-usage projection so stale rollups self-heal (P2, usage-cost-correctness, ready-now / convergence-path) → [tasks/15_polylogue-f2qv.5.md](tasks/15_polylogue-f2qv.5.md)
16. `polylogue-20d.4` — Mirror daemon structured-query routing in CLI so non-FTS filters skip the FTS readiness gate (P2, query-correctness, ready-now / code-local) → [tasks/16_polylogue-20d.4.md](tasks/16_polylogue-20d.4.md)
17. `polylogue-1xc.12` — Add FTS drift gauges and metamorphic trigger-coherence tests with rowid-reuse protection (P2, search-integrity, spec-first then code) → [tasks/17_polylogue-1xc.12.md](tasks/17_polylogue-1xc.12.md)
18. `polylogue-83u.3` — Acquire uploaded attachment bytes in live browser capture (P1, attachment-integrity, needs-architecture-note then code) → [tasks/18_polylogue-83u.3.md](tasks/18_polylogue-83u.3.md)
19. `polylogue-83u.2` — Acquire bytes for non-inline sources while live handles are open (P2, attachment-integrity, ready-now after census/classification) → [tasks/19_polylogue-83u.2.md](tasks/19_polylogue-83u.2.md)
20. `polylogue-83u.6` — Run read-only attachment acquisition census by origin/status/bytes (P2, attachment-integrity, ready-now / read-only artifact) → [tasks/20_polylogue-83u.6.md](tasks/20_polylogue-83u.6.md)
21. `polylogue-peo` — Add daemon crash forensics, heartbeat sentinel, and restart evidence (P2, operational-resilience, ready-now / lifecycle module) → [tasks/21_polylogue-peo.md](tasks/21_polylogue-peo.md)
22. `polylogue-4be` — Create a real restore drill for backup proof (P2, backup-integrity, ready-now / devtools+ops artifact) → [tasks/22_polylogue-4be.md](tasks/22_polylogue-4be.md)
23. `polylogue-4ts.3` — Separate subagent auto-compaction from main-session compaction in lineage (P2, lineage-truth, needs-source-confirmation then parser patch) → [tasks/23_polylogue-4ts.3.md](tasks/23_polylogue-4ts.3.md)
24. `polylogue-4ts.4` — Read lineage composition from one transaction/snapshot (P2, lineage-truth, needs-source-confirmation then storage patch) → [tasks/24_polylogue-4ts.4.md](tasks/24_polylogue-4ts.4.md)
25. `polylogue-4ts.6` — Expose transcript completeness instead of silently reading truncated sessions (P2, lineage-truth, needs-source-confirmation then model patch) → [tasks/25_polylogue-4ts.6.md](tasks/25_polylogue-4ts.6.md)
26. `polylogue-b0b` — Replace keyword-only outcome/pathology heuristics with structural evidence where available (P2, evidence-honesty, spec-first then targeted code) → [tasks/26_polylogue-b0b.md](tasks/26_polylogue-b0b.md)
27. `polylogue-9e5.3` — Column-honesty census for nullable/zero/default public fields (P2, evidence-honesty, ready-now / audit-artifact) → [tasks/27_polylogue-9e5.3.md](tasks/27_polylogue-9e5.3.md)
28. `polylogue-9e5.4` — Static get-modify-put race-window audit of shared SQLite writer paths (P2, storage-correctness, ready-now / audit-artifact) → [tasks/28_polylogue-9e5.4.md](tasks/28_polylogue-9e5.4.md)
29. `polylogue-9e5.19` — Storage-layer correctness scenario family in devtools lab (P2, storage-correctness, ready-now after focused bugs) → [tasks/29_polylogue-9e5.19.md](tasks/29_polylogue-9e5.19.md)

## Suggested first implementation batch

1. `polylogue-s7ae.6` release-gate classification.
2. `polylogue-37t.15`, `polylogue-kwsb.1`, `polylogue-8jg9.4`, `polylogue-83u.4` as the trust/data-loss/security floor.
3. `polylogue-9e5.28`, `.29`, `.30`, `cpf.5`, `cpf.6` as evidence-honesty floor.
4. `f2qv.2`, `f2qv.1`, `f2qv.4`, `f2qv.3`, `f2qv.5` as cost/usage truth.
5. Query/search/storage/lineage operational correctness packets.

## Caveats

This package is static prework. Some line numbers will drift after the first patch. Runtime claims still need focused reproduction. Where a packet says `needs-source-confirmation`, the first coding-agent action should be a short `rg`/read pass to confirm the code path before editing.


# Full task packet text

# 01. polylogue-s7ae.6 — Classify the aborted full verification run before coordination deploy

Priority: **P1**  
Lane: **release-gate**  
Readiness: **ready-now / evidence-work**

## Why this is urgent / critical-path

The coordination commit was merged after quick/focused verification, while a full `devtools verify` run was aborted at 74%. Until each failure is classified as caused-by-coordination, pre-existing, flaky, or fixed, every coordination/deployment packet inherits unknown risk.

## Static diagnosis / likely mechanism

This is not primarily a code bug. It is a release-gate debt packet. The static implication is that any live deployment of coordination, scheduler, hook, or MCP surfaces must wait for a fresh full verify log with a failure-classification table. Do not let later packets cite a green quick lane as deploy-clean.

## Implementation plan

Create `docs/audits/coordination-full-verify-classification.md` or `.agent/reports/coordination-full-verify-classification.md` with: command, git sha, environment, start/end time, full output path, failure table, owner bead for each failure, and final deploy verdict. If a failure is coordination-caused, fix it in the same PR or file a blocker bead with a minimal repro. If it is pre-existing, cite the pre-existing bead/issue and show why coordination did not widen it.

## Test plan

No new product test is required unless full verify exposes a real regression. Add a tiny regression only for any failure fixed during the classification.

## Verification command / proof

Run `devtools verify` full. Preserve the log artifact. The gate only opens when every failure has a table row and every coordination-caused failure is fixed or explicitly blocks deployment.

## Pitfalls

Do not silently downgrade this to `verify --quick`. Do not close it on a partial run. The output of this packet is a release decision, not a vibes report.

## Files/functions to inspect or touch

- `devtools/verify*`
- `docs/audits/ or .agent/reports/`


---

# 02. polylogue-37t.15 — Force non-user assertion writes through candidate/non-injected policy

Priority: **P1**  
Lane: **agent-write-safety**  
Readiness: **ready-now / code-local**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

The archive is about to let agents coordinate, post messages, and contribute memory. Non-user writes must not become trusted active context by default.

## Static diagnosis / likely mechanism

Root cause: `upsert_assertion` defaults `status=None` to `AssertionStatus.ACTIVE`, regardless of `author_kind`. `upsert_blackboard_note` passes `author_kind` through and does not provide a status, so agent-authored blackboard notes become active assertions. On conflict, `upsert_assertion` also overwrites `status` and `context_policy_json`, so a repeated non-user upsert can resurrect or re-inject a row.

Source anchors: `polylogue/storage/sqlite/archive_tiers/user_write.py:31-47` defines the ACTIVE default; `:641-671` blackboard note calls `upsert_assertion`; `:901-971` normalizes status/author/context and updates status/context on conflict; `:1245+` contains candidate judgment paths that should remain terminal.

## Implementation plan

Patch the single choke point: `polylogue/storage/sqlite/archive_tiers/user_write.py::upsert_assertion`.

Implementation shape:
1. Normalize `author_kind` before resolving status/context.
2. Add `_is_user_author(author_kind) -> bool`, initially exact normalized `"user"` only.
3. Fetch existing `status` and `context_policy_json`, not only `created_at_ms`.
4. For non-user authors:
   - if no existing row or existing row is still unjudged, force `status=CANDIDATE`;
   - force `context_policy.inject=false` and `context_policy.promotion_required=true`, overriding caller input;
   - if the existing row is terminal judged (`REJECTED`, `DEFERRED`, `SUPERSEDED`, `DELETED`, and probably `ACCEPTED` unless explicitly user-updated), preserve the existing status/context and do not let agent input revive it.
5. User-authored writes keep current behavior, including explicit active/injected assertions where caller policy allows it.
6. Do not add separate checks in MCP or blackboard handlers; they should remain clients of the invariant.

## Test plan

Add focused storage/API tests:
- agent `upsert_assertion(... author_kind="agent")` with no status lands as `CANDIDATE`, not `ACTIVE`.
- agent call with `status=ACTIVE` and `context_policy={"inject": true}` still stores candidate + non-injected.
- `post_blackboard_note(author_kind="agent")` mirrors to an assertion with candidate/non-injected policy.
- rejected candidate re-upserted by an agent remains rejected and non-injected.
- user-authored assertion remains active by default to avoid breaking current user-state flows.

## Verification command / proof

`devtools test tests/unit/storage/test_user_state_contracts.py tests/unit/storage/test_archive_tiers_assertions.py -k 'assertion or blackboard or candidate'`

## Pitfalls

Do not fix only `blackboard_post`; MCP, CLI, daemon, and future coordination messages must all inherit the policy from one place. Avoid a broad ontology change; this is a safety invariant, not a new assertion product.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/user_write.py:31-47`
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641-671`
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901-971`
- `polylogue/mcp/server_mutation_tools.py`
- `polylogue/api/user_state*`


---

# 03. polylogue-kwsb.1 — Daemon/capture Host, Origin, receiver-token, and spool hardening

Priority: **P1**  
Lane: **security**  
Readiness: **ready-now / code-local with extension smoke**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

The daemon and browser-capture receiver expose private archive data and live capture routes on localhost. Localhost is not a browser security boundary; DNS rebinding and same-host processes can reach it.

## Static diagnosis / likely mechanism

Root causes:
- Daemon GET dispatch has no central Host gate. `DaemonAPIHandler.do_GET` parses and dispatches directly (`polylogue/daemon/http.py:1294-1298`).
- Daemon token fallback accepts `?access_token=` for all paths when a token is configured (`http.py:1037-1055`), even though the comment only justifies EventSource.
- POST Origin checks are not centralized for GET, and missing Origin is accepted (`http.py:1301-1313`).
- Browser capture accepts no token when `auth_token is None`; the default receiver config sets `auth_token=None` (`polylogue/browser_capture/receiver.py:52-65`, `polylogue/browser_capture/server.py:68+`).
- Spool writing is bounded by filesystem success, not by receiver quota.

## Implementation plan

Patch in layers:
1. Add a central daemon request-admission helper called by GET/POST/DELETE before route dispatch. It should strip ports/brackets from `Host`, reject absent/malformed/foreign hosts, and allow only loopback/configured hostnames by default.
2. Keep shell bootstrap unauthenticated only for loopback deployments, but still Host-gated.
3. Restrict `access_token` query fallback to the exact SSE/EventSource route. If there is no current SSE route, remove the fallback and update tests/docs. Use `hmac.compare_digest` in token comparison.
4. Extend GET protection: API JSON routes should require configured bearer token and always pass Host gate. Browser-origin GETs from non-loopback origins should be rejected.
5. Browser capture: mint/load a receiver token at startup if none configured, persist it mode 0600, and require it for `/v1/browser-captures`. The extension should obtain it through the existing dev/install path, not from a public status endpoint.
6. Add receiver spool governor fields: `max_spool_bytes`, `max_spool_files`, and possibly `max_payload_bytes`. Enforce before committing a capture envelope; return a clear 413/507-style error and do not write partial files.

## Test plan

Tests:
- daemon GET `/api/sessions` with `Host: evil.example` is denied.
- daemon GET with allowed Host and valid bearer works.
- `?access_token=` is rejected for ordinary API GETs.
- POST cross-origin remains denied; legitimate extension/web shell flows still pass.
- browser-capture POST without token is denied by default.
- forged token denied; valid token accepted.
- spool quota rejects oversized/too-many captures without writing.
- extension/dev smoke proves the real extension can still capture with token.

## Verification command / proof

`devtools test tests/unit/daemon/test_daemon_http_security.py tests/unit/browser_capture -k 'host or origin or token or spool'` plus the existing browser extension smoke lane if present.

## Pitfalls

Do not rely on Origin alone; simple browser GETs can omit Origin. Do not leave a public endpoint that reveals the receiver token. Avoid breaking CLI/curl workflows; missing Origin is fine for non-browser clients only after Host/auth pass.

## Files/functions to inspect or touch

- `polylogue/daemon/http.py:1037-1055`
- `polylogue/daemon/http.py:1294-1359`
- `polylogue/browser_capture/receiver.py:52-65`
- `polylogue/browser_capture/server.py:54-90`
- `polylogue/browser_capture/server.py:259+`
- `browser-extension/src/background.js`


---

# 04. polylogue-8jg9.4 + polylogue-8jg9.2 — Make ops-doctor orphan cleanup use the same lease/generation invariants as blob GC

Priority: **P1/P2**  
Lane: **blob-integrity**  
Readiness: **ready-now / code-local**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

A cleanup tool that can delete an in-flight blob is a data-loss bug. It must be fixed before any backup/blob claim is trustworthy.

## Static diagnosis / likely mechanism

Root cause: there are two deletion paths. `run_blob_gc_report` checks reference surfaces, pending leases, minimum age, and generations (`polylogue/storage/blob_gc.py:163`, `:355+`, `MIN_AGE_S`). But `BlobStore.detect_orphans` and `cleanup_orphans` only compare disk hashes to a caller-supplied set and then unlink files (`polylogue/storage/blob_store.py:352-409`). `repair_orphaned_blobs_data` uses that unsafe path (`polylogue/storage/blob_repair.py:92+`). A blob written to disk with an active pending lease but not yet a durable reference can be seen as orphaned and deleted.

## Implementation plan

Implementation shape:
1. Treat `BlobStore.detect_orphans` as a preview/helper, not an apply-time safety authority.
2. Change ops-doctor / `repair_orphaned_blobs_data(... dry_run=False)` to delegate deletion to `run_blob_gc_report`, passing the archive DB path and blob root so the GC can see leases and reference surfaces.
3. For dry-run, either call `run_blob_gc_report(... dry_run=True)` or mark the generic orphan preview as unsafe/advisory.
4. If a public `cleanup_orphans` apply method remains, document it as low-level and ensure no ops command calls it without GC invariants.
5. Add a dedicated race fixture from `polylogue-8jg9.2`: disk blob exists, `pending_blob_refs` lease exists, no final reference row yet, doctor cleanup apply runs; file must survive.

## Test plan

Tests:
- pending-lease blob survives ops-doctor cleanup apply.
- old unleased orphan is deleted by safe GC after minimum-age/generation conditions.
- dry-run reports lease-skipped or not-deletable status.
- a stale caller-supplied orphan set cannot delete a blob that gained a reference between detection and apply.
- direct low-level `BlobStore.cleanup_orphans` tests may remain, but ops/doctor coverage must exercise the safe wrapper.

## Verification command / proof

`devtools test tests/unit/storage/test_blob_store.py tests/unit/storage/test_blob_gc.py tests/unit/storage/test_blob_repair.py -k 'orphan or lease or gc or cleanup'`

## Pitfalls

Do not reimplement half of GC inside blob_store. The store cannot know pending leases without DB context. The fix is route consolidation, not local unlink cleverness.

## Files/functions to inspect or touch

- `polylogue/storage/blob_store.py:352-409`
- `polylogue/storage/blob_repair.py:92+`
- `polylogue/storage/blob_gc.py:163`
- `polylogue/storage/blob_gc.py:355+`
- `polylogue/operations/*doctor*`


---

# 05. polylogue-83u.4 — Separate source-tier referenced-blob debt from index attachment-acquisition debt

Priority: **P1**  
Lane: **blob-integrity**  
Readiness: **ready-now / diagnostic-code**

Depends on packet(s): polylogue-8jg9.4 + polylogue-8jg9.2

## Why this is urgent / critical-path

The system previously reported 39,586 missing referenced blobs. The bead notes now say current source-tier referenced blob debt appears clean, while many index attachment rows are simply unfetched with `blob_hash NULL`. Those are different states and must not be collapsed.

## Static diagnosis / likely mechanism

Likely mechanism: blob diagnostics count attachment rows without blob hashes as missing referenced blobs. But an unfetched attachment is acquisition debt, not a broken blob reference. A true missing referenced blob means a durable table points at a non-null hash whose file is absent.

## Implementation plan

Implementation shape:
1. Add/extend a diagnostics model with two top-level sections:
   - `source_reference_debt`: raw/source-tier blob reference rows with non-null hashes, present/missing/recoverable/accepted.
   - `attachment_acquisition_debt`: index attachment rows grouped by acquisition status: acquired+present, acquired+missing-file, unfetched-null-hash, unavailable, recoverable-local/zip/drive.
2. Change backup/workload warnings to say exactly which section is bad.
3. `blob_hash IS NULL` attachments must increment `unfetched`, not `missing_blob_ref`.
4. Add bounded samples by source/origin/reference type. Keep live archive probes read-only (`mode=ro`).
5. Feed the same primitives into the 83u.6 census.

## Test plan

Tests:
- clean source DB + index attachments with `blob_hash NULL` => source missing=0, unfetched=N, no “missing referenced blob” warning.
- acquired attachment with hash whose blob file is missing => attachment acquired_missing=N.
- raw/source reference to missing hash => source_reference_debt missing=N.
- diagnostic JSON has both sections and reconciles totals.

## Verification command / proof

`devtools test tests/unit/storage/test_blob_integrity.py tests/unit/operations/test_archive_debt.py -k 'blob_reference_debt or attachment_acquisition or missing_blob'`

## Pitfalls

Do not “fix” missing source references by inventing synthetic hashes or deleting attachment rows. This packet is classification and truthful diagnostics first; restoration is only for rows whose original bytes can be verified.

## Files/functions to inspect or touch

- `polylogue/storage/blob_integrity.py`
- `polylogue/operations/archive_debt.py`
- `polylogue/storage/sqlite/archive_tiers/index.py:attachments`
- `polylogue/storage/blob_store.py`


---

# 06. polylogue-9e5.28 — Make the rigor audit iterate the full insight registry, not only existing contracts

Priority: **P1**  
Lane: **evidence-honesty**  
Readiness: **ready-now / code-local**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

An audit that only loops declared contracts cannot find missing contracts. Product claims then look audited even when entire insight products are invisible to the audit.

## Static diagnosis / likely mechanism

Root cause: `build_insight_rigor_audit_report` explicitly says products without registered rigor contracts are skipped and loops `for contract in list_rigor_contracts()` (`polylogue/insights/audit.py:173-201`). `_RIGOR_MATRIX` only covers a small subset of registry products (`polylogue/insights/rigor.py:85+`), while `INSIGHT_REGISTRY` is the product universe (`polylogue/insights/registry.py:294+`).

## Implementation plan

Implementation shape:
1. Add `coverage_status` to `InsightRigorAuditEntry`: `covered`, `uncovered`, `exempt`.
2. Add an explicit exemption map with reason strings for products that intentionally have no rigor contract.
3. Change audit iteration to sorted registry names. For each registry product:
   - if contract exists: run current audit, `coverage_status=covered`;
   - if exemption exists: emit a row, `coverage_status=exempt`, with reason;
   - else: emit a row, `coverage_status=uncovered`, with an error like `missing_rigor_contract`.
4. Fail the `insight-honesty` policy/lab lane if any non-exempt product is uncovered.
5. Update docs/renderers so missing-contract rows are loud and countable.

## Test plan

Tests:
- monkeypatched registry product without contract appears as `uncovered`, not omitted.
- exemption appears as `exempt` with reason.
- known contract product still audits normally.
- policy gate fails on uncovered product.
- old “all contract names exist in registry” test remains but no longer pretends coverage completeness.

## Verification command / proof

`devtools test tests/unit/insights/test_rigor_audit.py -k 'registry or uncovered or exemption or rigor'`

## Pitfalls

Do not just add more entries to `_RIGOR_MATRIX`. That helps today but preserves the same blindness for the next product.

## Files/functions to inspect or touch

- `polylogue/insights/audit.py:173-201`
- `polylogue/insights/rigor.py:45-85`
- `polylogue/insights/registry.py:294+`
- `devtools/lab or policy lane for insight-honesty`


---

# 07. polylogue-9e5.29 — Distinguish absent evidence from true numeric zero at field level

Priority: **P1**  
Lane: **evidence-honesty**  
Readiness: **needs-small-spec then code**

Depends on packet(s): polylogue-9e5.28

## Why this is urgent / critical-path

Counts, costs, rates, durations, and scores are user-visible claims. `0` must mean true zero, not “no backing rows were available.”

## Static diagnosis / likely mechanism

Mechanism: aggregate SQL and Pydantic payloads use numeric identity values (`COALESCE(SUM(...), 0)`, default `0.0`, default count fields) without denominator/evidence metadata. Product-level rigor contracts cannot detect a specific field whose denominator is absent. Likely hot spots include usage/cost rollups in `polylogue/storage/usage.py`, archive-tier cost aggregations, and summary/forensics payloads.

## Implementation plan

Implementation shape:
1. Add `RigorFieldContract` with fields such as `field_path`, `value_kind`, `denominator_path`, `unit`, `provenance_class`, `evidence_tier`, `nullable_when_ungrounded`, and `zero_semantics`.
2. Add `field_contracts` to `RigorContract`.
3. Teach the rigor audit to evaluate field contracts: if denominator is absent/zero and `nullable_when_ungrounded=true`, a stored/rendered numeric `0` is a defect unless the field’s zero semantics says true zero.
4. Convert worst public offenders first: provider/cost rollups and any forensics/report fields that can render `0.0` over empty backing rows.
5. SQL pattern: use `CASE WHEN COUNT(backing.id)=0 THEN NULL ELSE SUM(...) END`, not blanket `COALESCE`.
6. Render `None`/unknown as `uncovered`, `unknown`, or `not applicable`, not as zero.

## Test plan

Tests:
- empty provider-usage backing rows produce `None`/uncovered for costs/tokens, not `0.0`.
- a real row with zero cost/tokens still renders zero.
- field-level audit catches a fixture product where denominator=0 but value=0.0.
- renderer/surface test shows `uncovered` or equivalent marker.

## Verification command / proof

`devtools test tests/unit/insights/test_rigor_audit.py tests/unit/storage/test_usage*.py tests/unit/cli/test_*usage* -k 'empty or zero or uncovered or field_contract'`

## Pitfalls

Avoid a huge repo-wide migration in one PR. First add the contract/audit machinery and convert the highest-risk public numbers. Then file follow-ups for remaining numeric products revealed by the audit.

## Files/functions to inspect or touch

- `polylogue/insights/rigor.py`
- `polylogue/insights/audit.py`
- `polylogue/storage/usage.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py`
- `scripts/agent_forensics.py or promoted report surface`


---

# 08. polylogue-9e5.30 — Tag prose-mined forensic fields as text-derived

Priority: **P1**  
Lane: **evidence-honesty**  
Readiness: **ready-now / model-and-renderer**

Depends on packet(s): polylogue-9e5.28

## Why this is urgent / critical-path

The system already prevents fabricating events from prose, but still surfaces prose-mined refs and decisions as ordinary fields. Users and downstream agents need to know when a fact came from text interpretation rather than structured evidence.

## Static diagnosis / likely mechanism

Mechanism: forensic transform models (`ForensicIndexEntry`, `ToolSummary`, `DecisionCandidate`) carry extracted refs/labels/test evidence/decisions without per-field evidence class. Source anchors: `polylogue/insights/transforms.py:140`, `:192`, `:280`, extraction functions `:1573+` and `:1923+`.

## Implementation plan

Implementation shape:
1. Define a small evidence-class vocabulary: `raw_evidence`, `structured_tool_result`, `text_derived`, `synthetic`.
2. Add either explicit fields or `field_evidence: dict[str, EvidenceClass]` to `ToolSummary`, `DecisionCandidate`, and `ForensicIndexEntry`.
3. Mark refs extracted from command/output/message prose (`commit_refs`, `pr_refs`, `issue_refs`, `test_evidence`, prose decisions) as `text_derived` unless the parser has structured provider/tool metadata.
4. Structured tool result status/exit code can be `structured_tool_result`.
5. Update report/render surfaces so text-derived fields get a visible caveat. Do not suppress them; label them.
6. Add a downgrade rule: a finding backed only by text-derived fields cannot render as a hard fact without caveat.

## Test plan

Tests:
- synthetic session with a SHA/PR/test mention in assistant prose yields extracted refs with `text_derived` evidence class.
- structured tool-result exit status remains structured, not text-derived.
- rendered report includes a text-derived caveat.
- serialization remains backward-compatible for old rows where evidence class is absent, defaulting to `text_derived` for prose-mined fields.

## Verification command / proof

`devtools test tests/unit/insights/test_transforms.py -k 'text_derived or prose or forensic or evidence_class'` plus renderer tests for the affected report surface.

## Pitfalls

Do not treat text-derived as useless. The important invariant is label + caveat, not deletion.

## Files/functions to inspect or touch

- `polylogue/insights/transforms.py:140`
- `polylogue/insights/transforms.py:192`
- `polylogue/insights/transforms.py:280`
- `polylogue/insights/transforms.py:1573+`
- `polylogue/insights/transforms.py:1923+`
- `render/report modules that display forensic fields`


---

# 09. polylogue-cpf.5 — Propagate weakest temporal provenance through aggregates

Priority: **P1**  
Lane: **temporal-honesty**  
Readiness: **ready-now / code-local with schema propagation**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

Time-source laundering makes fallback or synthetic dates look like provider timestamps once they enter an aggregate. Timeline claims then look more grounded than they are.

## Static diagnosis / likely mechanism

Root cause: `classify_aggregate_hwm_source(source_updates: list[str])` returns `provider_ts` whenever the list is non-empty, explicitly assuming every input is provider-sourced (`polylogue/insights/temporal_source.py:97-108`). Leaf classifiers also treat any non-null timestamp as `provider_ts` (`:66-81`, `:84-94`). Aggregators pass timestamp strings instead of source classes.

## Implementation plan

Implementation shape:
1. Define a temporal-source rank lattice where weaker evidence wins. Suggested weakest-to-strongest: `fallback_date`, `materialization_ts`, `file_mtime`, `sort_key`, `hook_event_ts`, `provider_ts`.
2. Add `weakest_temporal_source(sources: Iterable[TemporalSource]) -> TemporalSource`.
3. Replace aggregate classifier with a source-aware helper. The old `source_updates: list[str]` signature should be deprecated or made to return the weakest safe value, not provider.
4. Ensure profile/work/thread rows store or expose their `input_high_water_mark_source`; aggregates must collect that field.
5. Add audit output for rows where a timestamp exists but its source class is unknown, so future materializers can be corrected.
6. Update renderers/docs to describe “weakest source wins.”

## Test plan

Tests:
- provider + fallback => aggregate fallback.
- provider + file_mtime => aggregate file_mtime.
- provider-only => provider.
- empty => fallback.
- materialized day/tag rollup fixture with mixed input sources stores the weakest source.
- old timestamp-only path does not silently return provider for mixed/unknown inputs.

## Verification command / proof

`devtools test tests/unit/insights/test_temporal_source_taxonomy.py tests/unit/insights/test_archive_summaries.py -k 'temporal or high_water or weakest'`

## Pitfalls

Do not use lexicographic enum order; define an explicit rank. Do not fix only labels in renderer; the stored aggregate provenance must be correct.

## Files/functions to inspect or touch

- `polylogue/insights/temporal_source.py:66-108`
- `polylogue/insights/archive_summaries.py`
- `polylogue/insights/archive_rollups.py`
- `profile/thread/tag/day summary models`


---

# 10. polylogue-cpf.6 — Inject clock seam for relative-date parsing and audit sort_key_ms epoch pins

Priority: **P1**  
Lane: **temporal-honesty**  
Readiness: **ready-now / code-local plus audit artifact**

Depends on packet(s): polylogue-cpf.5

## Why this is urgent / critical-path

Relative date filters and synthetic timestamp fallbacks affect which sessions appear in queries. They must be deterministic in tests and explicit when timeless rows are included.

## Static diagnosis / likely mechanism

Root causes:
- `parse_date` hardcodes `datetime.now(tz=timezone.utc)` as dateparser `RELATIVE_BASE` (`polylogue/core/dates.py:10-44`), so `since:yesterday` is not injectable.
- Query lowering calls `parse_date(value)` directly (`polylogue/storage/sqlite/archive_tiers/archive.py:6966`).
- Many read paths use `COALESCE(..., s.sort_key_ms, 0)` for ordering/window rows (`archive.py:4914+`, `5163+`, `5247+`, etc.), which can pin unknown-time rows to epoch instead of labeling them as unknown/synthetic.

## Implementation plan

Implementation shape:
1. Add `polylogue/core/clock.py` with a `Clock` protocol, `SystemClock`, and `FixedClock`.
2. Change `parse_date(date_str, *, now=None, clock=None)` and use the provided base for relative parsing.
3. Thread clock through query-spec construction and storage lowerers; CLI/daemon pass default system clock, tests pass fixed clock.
4. Add a small audit artifact listing each `sort_key_ms` fallback to `0`, classified as fixed, safe intentional, or needs follow-up.
5. Replace unsafe order/window fallbacks with explicit NULL handling (`IS NULL`, `NULLS LAST` emulation) or surface `time_confidence=synthetic` where timeless rows must remain visible.
6. For `since/until` filters, exclude truly timeless rows by default unless an explicit include-timeless flag/profile says otherwise.

## Test plan

Tests:
- `parse_date("7 days ago", now=fixed)` returns deterministic UTC.
- CLI/daemon structured query using relative date returns stable results under a fixed clock.
- timeless rows do not match `since` solely due to `0` fallback.
- ordering places timeless rows deterministically without pretending they occurred in 1970.
- grep/lint test prevents direct `datetime.now()` in query date parsing.

## Verification command / proof

`devtools test tests/unit/core/test_dates.py tests/unit/archive/test_query_dates.py -k 'relative or clock or sort_key or timeless'` plus audit artifact review.

## Pitfalls

Do not make tests monkeypatch global datetime. The point is a clean seam. Do not delete timeless rows from reads; classify/exclude/include deliberately.

## Files/functions to inspect or touch

- `polylogue/core/dates.py:10-44`
- `polylogue/storage/sqlite/archive_tiers/archive.py:6966`
- `polylogue/storage/sqlite/archive_tiers/archive.py:4914+`
- `polylogue/storage/sqlite/archive_tiers/archive.py:5163+`
- `polylogue/archive/query/*`
- `polylogue/cli/archive_query.py`
- `polylogue/daemon/http.py`


---

# 11. polylogue-f2qv.2 — Normalize Codex/Claude token lanes into disjoint uncached/cache/reasoning/completion fields

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / parser+tests**

## Why this is urgent / critical-path

Codex/Claude token accounting underpins usage reports, cost reports, and public forensics. Double-counting cached or reasoning tokens creates order-of-magnitude false claims.

## Static diagnosis / likely mechanism

Mechanism from bead: Codex `input` includes cached tokens and `output` includes reasoning tokens; naive input+output inflated cost by 7.69x in a prior fix. Static source anchor: `polylogue/sources/parsers/codex.py` has token_count/cached/reasoning logic around `:161-193` and `:278+`, but the invariant is not locked.

## Implementation plan

Implementation shape:
1. Create/centralize a provider usage normalizer returning disjoint lanes: `input_uncached`, `cache_read`, `cache_write`, `output_completion`, `output_reasoning`.
2. For Codex: derive uncached input as reported input minus cached input, reasoning as a separate output sublane, completion output as reported output minus reasoning. Clamp/report inconsistencies loudly; do not silently negative-clamp without diagnostics.
3. For Claude: map cache creation/read lanes and output/reasoning fields into the same disjoint schema.
4. Add a helper that asserts lane sum equals provider-reported total where the provider reports a total.
5. Ensure `session_provider_usage_events` writer stores lanes separately and downstream rollups consume those lanes, not raw input/output totals.

## Test plan

Tests:
- synthetic Codex payload where cached is 96% of input: disjoint lanes sum to reported total; naive input+output would fail the regression guard.
- synthetic output with reasoning: completion+reasoning equals reported output.
- Claude cache_creation/cache_read payload maps to cache_write/cache_read.
- malformed payload with inconsistent totals is reported/classified, not silently accepted.
- optional live scratch cross-check against `state_5.sqlite` if available.

## Verification command / proof

`devtools test tests/unit/sources/test_codex*.py tests/unit/storage/test_provider_usage*.py -k 'token_count or cache or reasoning or disjoint'`

## Pitfalls

Do not change cost formulas before the lane invariant is locked. Cost packets should depend on this packet.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/codex.py:161-193`
- `polylogue/sources/parsers/codex.py:278+`
- `polylogue/sources/parsers/claude/code_parser.py`
- `polylogue/storage/sqlite/archive_tiers/write.py:provider_usage writer`
- `polylogue/storage/usage.py`


---

# 12. polylogue-f2qv.1 — Make per-model rollups partition usage events instead of duplicating session totals

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / storage-rollup**

Depends on packet(s): polylogue-f2qv.2

## Why this is urgent / critical-path

Per-model charts and reports become false when a multi-model session contributes its whole total to every model row it touched.

## Static diagnosis / likely mechanism

Mechanism from bead: a session’s token totals are attributed under each model row it touched. Correct behavior: per-model totals are sums of provider usage events whose model equals that model; sum(per-model) == session total. Static anchors: `session_model_usage` materialization in `polylogue/storage/sqlite/archive_tiers/write.py:618`, table/rollup logic around `:2696+`, `:2904+`, `:2953+`.

## Implementation plan

Implementation shape:
1. Locate the builder that groups `session_provider_usage_events` into `session_model_usage`.
2. Change grouping to `GROUP BY session_id, model` over event rows; each event contributes only to its own model.
3. Preserve session-grain totals separately if needed; do not copy them into every model row.
4. Add a rollup invariant helper used by tests: for each session, sum(model rows lanes) equals sum(provider event lanes) within integer exactness/tolerance.
5. Add a live-diagnostics query that reports any sessions where per-model > session total.

## Test plan

Tests:
- one session with two model events: model A gets only A event tokens, model B gets only B event tokens, sum equals session total.
- mixed cache/reasoning lanes partition independently.
- regression named with GH #2472 or bead id.
- existing single-model session behavior unchanged.

## Verification command / proof

`devtools test tests/unit/storage/test_provider_usage*.py tests/unit/storage/test_session_model_usage*.py -k 'partition or multi_model or f2qv'`

## Pitfalls

Do this after the disjoint lane normalizer, so the partition invariant is defined over the right token fields.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/write.py:618`
- `polylogue/storage/sqlite/archive_tiers/write.py:2696+`
- `polylogue/storage/sqlite/archive_tiers/write.py:2904+`
- `polylogue/storage/usage.py`


---

# 13. polylogue-f2qv.4 — Use one LiteLLM-backed pricing resolver and remove tokencost/second maps

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / grep-and-contract**

Depends on packet(s): polylogue-f2qv.2

## Why this is urgent / critical-path

Two pricing catalogs drift. Cost outputs must say one authoritative API-equivalent price source or labelled unknown.

## Static diagnosis / likely mechanism

Mechanism from bead: a LiteLLM catalog exists, but tokencost/hardcoded maps may remain. Model ids can be vendor-prefixed, so resolver must match the last path segment rather than failing or resolving a parent path.

## Implementation plan

Implementation shape:
1. `rg 'tokencost|PRICE|pricing|cost_per'` across `polylogue`, `scripts`, `pyproject.toml`.
2. Remove `tokencost` dependency/imports.
3. Create exactly one resolver module/function. It should normalize model ids by exact id first, then last path segment, then aliases if declared.
4. Resolver result should be a structured object: rate fields + source/catalog version + `unknown` reason.
5. Update all cost surfaces to call the resolver. Do not leave emergency hardcoded maps in report scripts.
6. Add a test that greps or introspects for forbidden second-table symbols/imports.

## Test plan

Tests:
- `vendor/family/model-name` resolves via last segment when catalog contains `model-name`.
- observed live-archive model ids either resolve or become labelled unknown.
- no `tokencost` import/dependency remains.
- a fake second map in a fixture/test helper is caught by the no-second-price-table test.

## Verification command / proof

`devtools test tests/unit/usage tests/unit/storage -k 'pricing or litellm or cost'` plus `rg tokencost pyproject.toml polylogue scripts` returns no production hits.

## Pitfalls

Do not mix this with subscription-credit math. This packet owns API-list-equivalent price lookup only.

## Files/functions to inspect or touch

- `pyproject.toml`
- `polylogue/**/pricing*.py`
- `polylogue/storage/usage.py`
- `scripts/agent_forensics.py or promoted report`
- `LiteLLM catalog module`


---

# 14. polylogue-f2qv.3 — Report API-equivalent dollars and subscription credits as separate fields

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now after lanes/pricing**

Depends on packet(s): polylogue-f2qv.2, polylogue-f2qv.4

## Why this is urgent / critical-path

A single `cost_usd` field conflates API list-equivalent value with actual subscription metering. That produces persuasive but misleading financial claims.

## Static diagnosis / likely mechanism

Mechanism from bead: cache reads dominate token volume, but subscription plans may meter cache reads differently or not at all. Prior memory also records a 5x output credit-rate bug. Correct reports need two accounting regimes, not one blended number.

## Implementation plan

Implementation shape:
1. Add fields such as `api_equivalent_usd`, `api_price_source`, `subscription_credit_estimate`, `subscription_credit_source`, and `cost_view_caveat` to session/day/origin cost surfaces.
2. API-equivalent view uses the LiteLLM resolver from f2qv.4 and disjoint token lanes from f2qv.2.
3. Subscription-credit view zeroes/free-prices lanes according to declared plan assumptions; cache-read treatment must be explicit.
4. Correct the output credit rate bug and lock it with a fixture.
5. Render both fields side by side. Never label subscription credits as dollars unless converted by a separate plan-capacity model.
6. Keep unknown models/unknown plan assumptions as labelled unknown/partial, not zero.

## Test plan

Tests:
- cache-heavy session has `subscription_credit_estimate < api_equivalent_usd` under Claude Max assumptions.
- output credit-rate fixture catches the old 5x bug.
- unknown model leaves API field unknown/partial rather than false zero.
- JSON schemas/docs show both views.

## Verification command / proof

`devtools test tests/unit/storage/test_usage*.py tests/unit/cli/test_usage*.py -k 'subscription or api_equivalent or credit or cache'`

## Pitfalls

Do not remove the old cost field abruptly without compatibility plan; either deprecate it or make it a wrapper with a caveat. Do not pair all-provider token totals with priced-only dollars.

## Files/functions to inspect or touch

- `polylogue/storage/usage.py`
- `polylogue/storage/sqlite/archive_tiers/write.py`
- `polylogue/cli usage/analyze surfaces`
- `MCP cost tools`
- `scripts/agent_forensics.py or report surface`


---

# 15. polylogue-f2qv.5 — Version-gate provider-usage projection so stale rollups self-heal

Priority: **P2**  
Lane: **usage-cost-correctness**  
Readiness: **ready-now / convergence-path**

Depends on packet(s): polylogue-f2qv.1, polylogue-f2qv.2

## Why this is urgent / critical-path

Usage/cost fixes do not help existing archives if stale `session_model_usage` rows never rederive. Manual full index rebuilds are the wrong maintenance model for derived read-model staleness.

## Static diagnosis / likely mechanism

Root cause from bead and source anchors: provider usage rows are written during ingest (`polylogue/storage/sqlite/archive_tiers/write.py:618`) but are not in the session insight rebuild/convergence path. The status gate covers profiles/logical/work/threads, not provider usage. Archive debt can only recommend manual rebuild (`polylogue/operations/archive_debt.py:854+`, zero-token rows `:899+`).

## Implementation plan

Implementation shape:
1. Add a materializer version for provider-usage/session-model-usage projection.
2. Store that version in an existing insight materialization table or a new lightweight table keyed by session id.
3. Extend session-insight status/staleness detection to include provider usage.
4. Extend the periodic convergence loop to refresh stale provider usage rows for sessions whose source blocks carry usage.
5. Archive-debt should report stale provider usage as drainable by convergence, not only full rebuild.
6. Add a version bump fixture: seed old-version rows, run convergence/drain, assert new rows derived.

## Test plan

Tests:
- old-version `session_model_usage` row becomes stale.
- daemon/convergence drain refreshes it without `maintenance rebuild-index`.
- zero-token debt over sessions with source usage drains to zero after convergence.
- archive_debt message changes from manual-only to convergence-drainable.

## Verification command / proof

`devtools test tests/unit/storage/insights tests/unit/operations/test_archive_debt.py -k 'provider_usage or materializer_version or stale'`

## Pitfalls

Do not put this in a one-off repair command only. The acceptance condition is automagic convergence on daemon run.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/write.py:618`
- `polylogue/storage/insights/session/status.py`
- `polylogue/storage/insights/session/rebuild.py`
- `polylogue/insights/registry.py`
- `polylogue/operations/archive_debt.py:854+`


---

# 16. polylogue-20d.4 — Mirror daemon structured-query routing in CLI so non-FTS filters skip the FTS readiness gate

Priority: **P2**  
Lane: **query-correctness**  
Readiness: **ready-now / code-local**

## Why this is urgent / critical-path

Structured-only queries should work even when full-text-search is stale or rebuilding. Paying the FTS gate for origin/date/path filters is both slow and incorrect.

## Static diagnosis / likely mechanism

Mechanism from bead: daemon route discriminates structured-only vs FTS queries, while CLI search calls the search path unconditionally. Static source confirms daemon has a route split in `polylogue/daemon/http.py`; archive query/lowering uses `SessionQuerySpec` and query terms.

## Implementation plan

Implementation shape:
1. Inspect the single CLI archive search/list execution site, likely in `polylogue/cli/archive_query.py`.
2. Construct `SessionQuerySpec` exactly once.
3. Branch like daemon: if `spec.query_terms` or `spec.contains_terms` or semantic/similar text is present, use search/FTS/vector path; otherwise use list/facade structured path.
4. Keep output rendering identical so users do not see a behavior split except the missing FTS gate.
5. Add a helper shared with daemon if duplication is small enough; otherwise add a comment tying both discriminators together.

## Test plan

Tests:
- archive with deliberately absent/stale FTS still answers `--origin X --since DATE` or equivalent structured-only CLI query.
- query with text terms still uses FTS readiness and fails/repairs as before.
- daemon and CLI return same session ids for the same structured filter fixture.
- regression name references `#1860`/bead id.

## Verification command / proof

`devtools test tests/unit/cli/test_archive_query*.py tests/unit/daemon/test_daemon_http*.py -k 'structured or fts or query_routing'`

## Pitfalls

Verify current v23 shape first; recent freshness work may have moved the discriminator. Do not make structured queries accidentally bypass semantic/vector queries.

## Files/functions to inspect or touch

- `polylogue/cli/archive_query.py`
- `polylogue/archive/query/*`
- `polylogue/daemon/http.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py`


---

# 17. polylogue-1xc.12 — Add FTS drift gauges and metamorphic trigger-coherence tests with rowid-reuse protection

Priority: **P2**  
Lane: **search-integrity**  
Readiness: **spec-first then code**

## Why this is urgent / critical-path

FTS readiness as a boolean hides whether drift is 1 row or catastrophic. Count agreement is not enough when rowids can be reused.

## Static diagnosis / likely mechanism

Mechanism from bead: `messages_fts.rowid == blocks.rowid == docsize.id` is the keystone identity, but SQLite rowid reuse can make a ghost FTS row bind to a different block. Existing readiness checks are count/boolean-oriented; exact reconciliation must compare rowid plus block identity, and a ledger can itself drift.

## Implementation plan

Implementation shape:
1. Inspect `polylogue/storage/sqlite/archive_tiers/index.py` FTS trigger DDL and the current readiness checks in `polylogue/storage/archive_readiness.py` and `polylogue/daemon/fts_startup.py`.
2. Define drift classes: missing FTS row, ghost/excess row, mismatched identity, empty-text transition drift, ledger-vs-exact disagreement.
3. Add O(1) gauges from an `fts_freshness_state`/similar ledger where available; do not count full FTS tables on metrics scrape.
4. Add periodic exact reconciliation that samples or runs bounded full checks outside hot scrape paths.
5. Add `ops.db` drift sample history with retention.
6. Add Hypothesis/stateful tests that apply arbitrary insert/update/delete sequences through real triggers and assert exact convergence.
7. Add a rowid-reuse regression that fails unless identity is checked beyond count equality.

## Test plan

Tests:
- rowid reuse scenario creates equal counts but mismatched identity and the check catches it.
- Hypothesis op sequences over blocks/search text converge to zero drift.
- metrics endpoint emits gauges without table scans.
- exact reconciliation can repair or at least classify ledger drift.

## Verification command / proof

`devtools test tests/unit/storage/test_fts*.py tests/unit/daemon/test_*metrics*.py -k 'fts or rowid or drift or metamorphic'`

## Pitfalls

Contentless FTS cannot be checked by selecting text back. Use docsize/rowid plus block identity or a ledger. Keep heavy reconciliation out of the scrape path.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/daemon/fts_startup.py`
- `polylogue/storage/archive_readiness.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py:8112+`
- `ops.db metrics/telemetry modules`


---

# 18. polylogue-83u.3 — Acquire uploaded attachment bytes in live browser capture

Priority: **P1**  
Lane: **attachment-integrity**  
Readiness: **needs-architecture-note then code**

Depends on packet(s): polylogue-kwsb.1, polylogue-83u.4

## Why this is urgent / critical-path

The DOM adapter currently captures attachment chips but not bytes. For live browser capture, this is the last moment when bytes or authenticated provider handles may still be available.

## Static diagnosis / likely mechanism

Static mechanism:
- Browser capture schema already has byte-carrying fields: `inline_base64`, `content_base64`, `data` (`polylogue/browser_capture/models.py:22-35`).
- Parser already converts those fields to `ParsedAttachment.inline_bytes` (`polylogue/sources/parsers/browser_capture.py:83-132`).
- ChatGPT DOM adapter currently records only name/url/provider_meta (`browser-extension/src/content/chatgpt.js:98-108`), no inline bytes.
So the storage path exists; the missing piece is extension-side acquisition.

## Implementation plan

Implementation shape:
1. First document the MV3 architecture choice in the PR: service-worker lifecycle, content-script/main-world access, permissions, receiver contract, payload size limits.
2. Prefer an extension-side acquisition path. Receiver-side refetch lacks page cookies/session unless the extension supplies bytes or an authenticated fetch result.
3. For uploaded local files, prototype a content/main-world hook on file input/change and/or fetch/FormData submission that captures File bytes into an in-memory bounded cache keyed by name/size/lastModified/provider id.
4. When a turn attachment chip is captured, attach `inline_base64`, `size_bytes`, `mime_type`, and `sha256`/provider_meta acquisition details if a cached file matches.
5. Bump receiver schema version and validate max payload/attachment sizes.
6. Reuse the existing parser `inline_bytes -> blob` path; do not widen content hashing or invent synthetic hashes.
7. If upload-body interception is impossible for a provider, classify as `unfetched` with reason and file a provider-specific follow-up.

## Test plan

Tests:
- JS/unit or browser smoke: synthetic File upload leads to capture payload with inline_base64 and size.
- receiver/model accepts versioned payload and rejects oversized inline content.
- parser turns payload into `ParsedAttachment.inline_bytes` and stored blob has true SHA-256 and nonzero byte_count.
- legacy payload without bytes still parses as unfetched/metadata-only.

## Verification command / proof

`devtools test tests/unit/sources/test_browser_capture*.py tests/unit/browser_capture -k 'attachment or inline_base64 or blob'` plus browser-extension smoke for capture-with-attachment.

## Pitfalls

Do not choose receiver-side refetch without proving cookies/auth can flow safely. Do not put raw bytes into durable transcript JSON beyond the existing transport-only inline path.

## Files/functions to inspect or touch

- `browser-extension/src/content/chatgpt.js:98-108`
- `browser-extension/src/background.js`
- `polylogue/browser_capture/models.py:22-35`
- `polylogue/sources/parsers/browser_capture.py:83-132`
- `polylogue/browser_capture/server.py`
- `polylogue/storage/blob_store.py`


---

# 19. polylogue-83u.2 — Acquire bytes for non-inline sources while live handles are open

Priority: **P2**  
Lane: **attachment-integrity**  
Readiness: **ready-now after census/classification**

Depends on packet(s): polylogue-83u.4

## Why this is urgent / critical-path

Some attachment bytes still exist at source time but are currently bypassed. Those are capture/parser bugs, not irrecoverable gaps.

## Static diagnosis / likely mechanism

Bead design identifies three live-handle boundaries: Drive downloads inside iterator scope, export-zip members while zipfile is open, and local paths guarded by a source-root allowlist. Existing parser model already supports `ParsedAttachment.inline_bytes`.

## Implementation plan

Implementation shape:
1. Drive: restore/download asset bytes with `DriveSourceClient.download_bytes` inside the iterator/lifetime of the source handle.
2. Export ZIP: resolve attachment member and read it while the `ZipFile` is still open; attach as inline bytes.
3. Local path: accept a transport-only `local_source_path`, canonicalize with `realpath`, require it to stay under declared source-root allowlist, reject symlink/`..` escapes before opening.
4. Non-live handles remain `unfetched` with `source_url/source_path`; no synthetic hash.
5. Assert acquired attachment blob hash equals true SHA-256 of bytes and session content hash remains stable for otherwise-identical content.

## Test plan

Tests:
- Drive fixture downloads inside iterator and stores true blob.
- ZIP fixture stores member bytes before close.
- local allowlisted file stores true blob.
- local path escape is rejected and no read happens.
- closed/non-live handle stays unfetched.
- ingest idempotency/content hash unchanged except attachment acquisition state.

## Verification command / proof

`devtools test tests/unit/sources/ -k 'attachment and (drive or zip or local or inline_bytes)'`

## Pitfalls

Do not let source-local paths become arbitrary file read. The allowlist/realpath check is part of the feature, not a hardening add-on.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/drive*.py`
- `polylogue/sources/parsers/drive_support_attachments.py`
- `polylogue/sources/parsers/base_models.py:179-212`
- `source client / zip importer modules`
- `blob write path `_acquire_attachment_blob``


---

# 20. polylogue-83u.6 — Run read-only attachment acquisition census by origin/status/bytes

Priority: **P2**  
Lane: **attachment-integrity**  
Readiness: **ready-now / read-only artifact**

Depends on packet(s): polylogue-83u.4

## Why this is urgent / critical-path

Before claiming attachments are preserved or deciding which acquisition work matters, the project needs a byte-backed census grouped by origin and acquisition class.

## Static diagnosis / likely mechanism

The bead already gives a strong design: read source.db/index.db/blob store read-only; group by origin and acquisition_status; reconcile against blob-reference-debt diagnostics. Baseline numbers in the bead: 7,226 attachment rows, 958 acquired with non-null hash and 0 missing acquired blobs, 6,268 unfetched with NULL hash.

## Implementation plan

Implementation shape:
1. Add a devtools or ops diagnostic command that opens archive DBs using SQLite `mode=ro`.
2. Resolve blob paths through `BlobStore` without mutating.
3. Group rows by `(origin, acquisition_status)` and emit counts, declared byte sum, acquired blob count, bytes on disk, unfetched, unavailable, missing acquired blob refs, top source_ref classes, and bounded samples.
4. Write JSON + Markdown under `.agent/scratch/research/` or a demo shelf.
5. Store baseline now, then rerun after 83u.2/83u.3 to measure delta.

## Test plan

Tests:
- command refuses or is proven not to open write connections against a fixture/live path.
- fixture totals reconcile with blob-reference-debt primitive.
- missing acquired blob and unfetched NULL hash are separated.
- sample list is bounded.

## Verification command / proof

`devtools test tests/unit/operations/test_attachment_census*.py -k 'read_only or acquisition or census'` and one read-only run against a copy/live archive as operator evidence.

## Pitfalls

This packet should not mutate the archive. It is evidence acquisition about evidence acquisition.

## Files/functions to inspect or touch

- `new diagnostics command`
- `polylogue/storage/blob_store.py`
- `polylogue/storage/blob_integrity.py`
- `polylogue/operations/archive_debt.py`
- `index.db attachments read path`


---

# 21. polylogue-peo — Add daemon crash forensics, heartbeat sentinel, and restart evidence

Priority: **P2**  
Lane: **operational-resilience**  
Readiness: **ready-now / lifecycle module**

## Why this is urgent / critical-path

A daemon that silently dies leaves the web UI and operator believing stale states. Crash/death evidence is an operational correctness requirement.

## Static diagnosis / likely mechanism

Bead mechanism: under read-only serving flags the daemon terminated twice; logs stopped without traceback; exit code 144 was ambiguous. Current running checks may rely on pid/process rather than fresh heartbeat.

## Implementation plan

Implementation shape:
1. Add `polylogue/daemon/lifecycle.py` to own run id, heartbeat row, signal logging, and stack dumps.
2. At daemon start: enable `faulthandler`, create `ops.db daemon_lifecycle` table if absent, insert `started` row.
3. Periodic loop writes heartbeat timestamp each tick.
4. SIGTERM/SIGINT handlers log signal, active thread stack dump, heartbeat age, and lifecycle row before exiting/chain-calling.
5. `atexit` writes clean stop; missing clean stop plus stale heartbeat means vanish.
6. `/healthz` and bare `polylogue` status should report heartbeat age and not claim running from pid alone.
7. Check/patch Sinnix systemd unit for `Restart=on-failure` with backoff; if outside repo, record exact sinnix patch prompt.
8. Web banner can be a follow-up/linked bby.1 if not in same PR, but API status must expose daemon-unreachable/stale-heartbeat data.

## Test plan

Tests:
- direct lifecycle unit tests for start/heartbeat/clean stop rows.
- subprocess integration: start daemon fixture, send SIGTERM, assert signal row + stack log.
- stale heartbeat makes status/healthz report stale/down.
- clean shutdown distinct from vanish.

## Verification command / proof

`devtools test tests/unit/daemon/test_daemon_lifecycle*.py tests/unit/daemon/test_health*.py -k 'heartbeat or lifecycle or signal'` plus a manual/subprocess SIGTERM proof.

## Pitfalls

Do not do heavy SQLite work inside an unsafe low-level handler outside Python’s normal signal-dispatch context. Keep stack dumps bounded so a death does not create huge logs.

## Files/functions to inspect or touch

- `polylogue/daemon/cli.py`
- `polylogue/daemon/http.py health routes`
- `polylogue/daemon/status.py`
- `ops.db helpers`
- `sinnix module for polylogued if available`


---

# 22. polylogue-4be — Create a real restore drill for backup proof

Priority: **P2**  
Lane: **backup-integrity**  
Readiness: **ready-now / devtools+ops artifact**

## Why this is urgent / critical-path

A backup that has not been restored is not evidence. Polylogue’s archive is irreplaceable enough that restore proof is critical path.

## Static diagnosis / likely mechanism

The bead states there are multiple backup layers but no restore test. The restore drill should prove latest backup set can produce an archive that passes integrity checks and basic user queries within expected lag.

## Implementation plan

Implementation shape:
1. Add `polylogue ops restore-drill` or `devtools restore-drill`.
2. Locate latest configured backup set; restore to a scratch root, never over live archive.
3. Run `PRAGMA integrity_check` for each SQLite tier.
4. Run a 10-query battery: session count, message count, blob-reference debt, one `find`, one `read`, one insight/profile read, one attachment lookup, one usage/cost summary, one tag/user-state read, one health/status command.
5. Compare counts to live archive with an expected-lag tolerance and record differences.
6. Emit JSON + Markdown ops artifact with timing, backup source, restored root, counts, query results, and failure reasons.
7. Add a corruption fixture that fails loudly.
8. Wire Sinnix quarterly timer or emit exact downstream patch instructions.

## Test plan

Tests:
- synthetic backup restored to scratch and query battery passes.
- corrupted DB/file fails with clear failure.
- drill refuses to write into live archive path.
- count-lag tolerance behaves as declared.

## Verification command / proof

`devtools test tests/unit/operations/test_restore_drill*.py -k 'restore or backup or corruption'` plus one real restore drill artifact.

## Pitfalls

Do not close on “backup command ran.” The acceptance condition is a restored scratch archive that answers queries.

## Files/functions to inspect or touch

- `polylogue/operations/backup*`
- `polylogue/daemon/backup*`
- `devtools/*`
- `archive tier path helpers`
- `Sinnix timer module`


---

# 23. polylogue-4ts.3 — Separate subagent auto-compaction from main-session compaction in lineage

Priority: **P2**  
Lane: **lineage-truth**  
Readiness: **needs-source-confirmation then parser patch**

## Why this is urgent / critical-path

Compaction events affect what context was available. Subagent auto-compaction should not be represented as a main-session compaction boundary.

## Static diagnosis / likely mechanism

Bead hints point to Claude parser behavior around `agent-acompact-*` parent assignment. Static next step is to inspect parser code with `rg 'agent-acompact|acompact|compaction|parent' polylogue/sources/parsers` and find where parent/lineage kind is assigned.

## Implementation plan

Implementation shape:
1. Locate parser branch that classifies Claude Code auto-compaction/session ids.
2. Add an explicit lineage event kind or flag: `main_compaction`, `subagent_auto_compaction`, `subagent_spawn`, etc.
3. Parent assignment: a subagent auto-compact event should attach to the subagent/session tree, not become a main-session compaction boundary.
4. Downstream lineage/composed-session renderers should show subagent compaction separately and not truncate/restart main session effective context.
5. Migration/backfill: old parsed rows may need a derived-lineage rebuild, not source mutation.

## Test plan

Tests:
- fixture with main session + subagent `agent-acompact-*`: main compaction count remains zero or unchanged; subagent compaction count increments.
- composed lineage tree renders subagent compaction under subagent.
- existing main compaction fixture still works.
- aggregate counts distinguish both classes.

## Verification command / proof

`devtools test tests/unit/sources/test_claude* tests/unit/lineage -k 'acompact or compaction or subagent'`

## Pitfalls

Do not key solely on id string if structured provider metadata exists; prefer provider event shape first and id prefix as fallback.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/claude/*`
- `polylogue/lineage or session lineage modules`
- `lineage composition/render tests`


---

# 24. polylogue-4ts.4 — Read lineage composition from one transaction/snapshot

Priority: **P2**  
Lane: **lineage-truth**  
Readiness: **needs-source-confirmation then storage patch**

## Why this is urgent / critical-path

Composed session reads that stitch sessions, edges, messages, compactions, and shared prefixes must not mix rows from different write moments.

## Static diagnosis / likely mechanism

Likely mechanism from bead title: lineage composition performs multiple separate reads/connection steps, so concurrent ingest/refresh can produce an impossible graph. This is especially dangerous for branch/shared-prefix accounting and compaction context.

## Implementation plan

Implementation shape:
1. Locate lineage composition entrypoint and list every DB read it performs.
2. Ensure the public composition call opens one read connection and begins a read transaction/snapshot before the first query.
3. Pass that connection through lower helpers instead of helpers reopening connections.
4. If both source and index tiers are needed, document consistency model; prefer a single tier/projection for composition or record cross-tier snapshot caveat.
5. Add optional `composition_read_id` or `snapshot_started_at_ms` in debug output for diagnostics.

## Test plan

Tests:
- two-connection fixture: begin composition read, mutate lineage/messages on another connection, finish composition; result reflects one consistent before/after state, not mixed.
- helper tests assert no lower function opens a new connection when a connection is supplied.
- existing lineage composition tests still pass.

## Verification command / proof

`devtools test tests/unit/lineage tests/unit/storage -k 'composition or transaction or snapshot or lineage'`

## Pitfalls

Do not serialize writers globally to hide this. The read path needs a coherent snapshot; normal ingest concurrency should continue.

## Files/functions to inspect or touch

- `polylogue/lineage*`
- `polylogue/storage/sqlite/archive_tiers/* lineage/read helpers`
- `composition renderers`


---

# 25. polylogue-4ts.6 — Expose transcript completeness instead of silently reading truncated sessions

Priority: **P2**  
Lane: **lineage-truth**  
Readiness: **needs-source-confirmation then model patch**

## Why this is urgent / critical-path

A transcript reader that returns a partial session without saying so creates false context packs and misleading evidence.

## Static diagnosis / likely mechanism

Bead title says silently truncated transcripts need a completeness signal. Likely sources: provider export truncation, partial file capture, continuation branches, max-message reads, or parser fallback. The exact current code path needs `rg 'truncated|complete|partial|max_messages|limit' polylogue/sources polylogue/storage polylogue/read` before patching.

## Implementation plan

Implementation shape:
1. Identify all parser/read paths that can return partial sessions: provider export flags, capture payload bounds, read limits, branch-local raw logs, and daemon read pagination.
2. Add a `transcript_completeness` enum or structured field: `complete`, `partial_export`, `parser_partial`, `read_limited`, `unknown`, with reason/source.
3. Store completeness in session/profile/read payloads.
4. Context packs and reports must show the signal and avoid strong claims over partial sessions.
5. Parser fixtures should set completeness from provider metadata where available; read-limit completeness is render-time, not stored source truth.

## Test plan

Tests:
- parser fixture with known truncated export stores partial reason.
- daemon/CLI read with `limit` says read-limited while source remains complete.
- context pack includes completeness caveat.
- complete sessions remain uncluttered/complete.

## Verification command / proof

`devtools test tests/unit/sources tests/unit/read tests/unit/daemon -k 'complete or completeness or truncated or partial'`

## Pitfalls

Keep source completeness separate from projection/read-limit completeness. A user selecting first 100 messages did not make the source incomplete.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/*`
- `polylogue/archive/read*`
- `polylogue/daemon/http.py read routes`
- `profile/session models`
- `context pack renderers`


---

# 26. polylogue-b0b — Replace keyword-only outcome/pathology heuristics with structural evidence where available

Priority: **P2**  
Lane: **evidence-honesty**  
Readiness: **spec-first then targeted code**

Depends on packet(s): polylogue-9e5.30

## Why this is urgent / critical-path

Behavioral analytics over agent work should not infer outcomes purely from words when structured tool/test/exit data exists.

## Static diagnosis / likely mechanism

Bead design says keyword outcome/pathology heuristics should be replaced by structural evidence. Static source likely contains regex/keyword classifiers in insight transforms/forensics/report scripts. These are useful as fallback, not as primary evidence.

## Implementation plan

Implementation shape:
1. Inventory outcome/pathology fields and their current evidence source: tool result exit code, test runner structured status, command result, PR/commit state, assistant prose, keyword heuristic.
2. Define evidence precedence: structured tool/test status > persisted action outcome > explicit user/agent annotation > text-derived prose > keyword fallback.
3. Add evidence_class/source fields to outcome/pathology records.
4. Convert the highest-impact classifiers first: failed tool call, test failure, silent proceed, retry loop, abandonment/resume.
5. Keep keyword fallback but mark `text_derived`/`heuristic` and exclude it from strong numeric claims unless caveated.

## Test plan

Tests:
- structured failed test beats optimistic prose.
- success prose does not override nonzero exit status.
- keyword-only case still produces fallback but labelled heuristic/text-derived.
- aggregate report footnotes counts by evidence tier.

## Verification command / proof

`devtools test tests/unit/insights tests/unit/archive -k 'outcome or pathology or tool_result or evidence_class'`

## Pitfalls

Coordinate with 9e5.30 so prose-derived fields share the same evidence-class vocabulary.

## Files/functions to inspect or touch

- `polylogue/insights/transforms.py`
- `scripts/agent_forensics.py or promoted forensics surface`
- `work-event/outcome classifiers`
- `report renderers`


---

# 27. polylogue-9e5.3 — Column-honesty census for nullable/zero/default public fields

Priority: **P2**  
Lane: **evidence-honesty**  
Readiness: **ready-now / audit-artifact**

Depends on packet(s): polylogue-9e5.29

## Why this is urgent / critical-path

Before fixing every dishonest field, produce a field census that shows where null/zero/default semantics are unclear.

## Static diagnosis / likely mechanism

This is a read-only audit packet. It supports 9e5.29 by finding numeric/default columns whose meaning is ambiguous, especially across insight products, usage, attachments, and lineage.

## Implementation plan

Implementation shape:
1. Generate a table of public payload fields/DB columns with type, nullable, default, sample null density, sample zero density, and known evidence source.
2. Classify each as true-zero-safe, unknown-when-absent, not-applicable, text-derived, or needs-contract.
3. Emit JSON + Markdown artifact under docs/audits or `.agent/reports`.
4. File follow-up beads only for confirmed high-risk fields.

## Test plan

Tests are optional unless adding tooling. If adding a command, test on a small fixture schema/payload set and assert classifications render.

## Verification command / proof

Run the census command/artifact generation on the active fixture/live copy read-only. Review artifact against 9e5.29 field-contract work.

## Pitfalls

Do not mutate product code beyond a small audit tool. The goal is a map for follow-up patches.

## Files/functions to inspect or touch

- `polylogue/insights/* models`
- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/storage/usage.py`
- `report/model schema emitters`


---

# 28. polylogue-9e5.4 — Static get-modify-put race-window audit of shared SQLite writer paths

Priority: **P2**  
Lane: **storage-correctness**  
Readiness: **ready-now / audit-artifact**

## Why this is urgent / critical-path

Some correctness failures require two actors. Before filing random race bugs, classify which read-modify-write paths are actually split across transactions and which are safe.

## Static diagnosis / likely mechanism

Bead notes give the method: static sweep for SELECT-then-UPDATE/INSERT, manual upsert emulation, status transitions, shared writer APIs, and connection boundaries. Named candidates include write effects, blob GC, daemon cursors, embeddings, FTS readiness, MCP mutation handlers, and CLI ops writers.

## Implementation plan

Implementation shape:
1. `rg` for read-then-write patterns and connection open boundaries.
2. For each candidate, record file:function, invariant, connection/transaction boundary, interleaving, expected consequence, and verdict: safe-by-single-transaction, safe-by-unique/upsert, needs-harness, or confirmed-bug.
3. Only create implementation bug beads for confirmed windows with concrete two-connection repro sketches.
4. Document refuted windows as safe with reason so agents do not re-triage them.

## Test plan

No broad product tests unless a confirmed race is found. For top 1-2 confirmed windows, add a minimal two-connection repro test in the follow-up bug, not this audit packet.

## Verification command / proof

Review committed `race-window-table.md/json`. Optional: `devtools test -k <new_race_test>` only for confirmed follow-up.

## Pitfalls

Do not file “maybe race” beads. A race bug needs a concrete interleaving and observable lost/stale effect.

## Files/functions to inspect or touch

- `polylogue/archive/write_effects.py`
- `polylogue/storage/blob_gc.py`
- `polylogue/daemon/cursor*`
- `polylogue/storage/embeddings/*`
- `polylogue/storage/archive_readiness.py`
- `polylogue/mcp/server_mutation_tools.py`
- `CLI ops writers`


---

# 29. polylogue-9e5.19 — Storage-layer correctness scenario family in devtools lab

Priority: **P2**  
Lane: **storage-correctness**  
Readiness: **ready-now after focused bugs**

Depends on packet(s): polylogue-8jg9.4 + polylogue-8jg9.2, polylogue-1xc.12, polylogue-4ts.4

## Why this is urgent / critical-path

Several storage correctness invariants are spread across tests. A scenario lane gives future agents one place to prove split-tier/idempotency/FTS/blob/lineage basics.

## Static diagnosis / likely mechanism

Bead design: create scenario family covering split-tier writes, content-hash idempotency, FTS trigger integrity, blob-lease GC, and lineage composition against seeded archives.

## Implementation plan

Implementation shape:
1. Add `storage-correctness` to scenario-coverage configuration, referencing this bead.
2. Build seeded archive fixtures for: idempotent re-ingest, split-tier write/read, FTS mutation drift, leased blob GC, lineage composition snapshot.
3. Wire into devtools lab/projections lanes.
4. The scenario should aggregate existing focused tests where possible rather than duplicate all logic.
5. Emit a compact scenario report with pass/fail and fixture paths.

## Test plan

Tests are the scenario: it must fail if one invariant is intentionally broken in fixture/code. Unit test the scenario registration if devtools has registry tests.

## Verification command / proof

`devtools lab projections --scenario storage-correctness` or the project’s current lab command; exact command should be documented in the PR.

## Pitfalls

Do this after the highest-risk concrete fixes so the scenario lane locks corrected behavior instead of snapshotting known-bad behavior.

## Files/functions to inspect or touch

- `devtools/lab*`
- `scenario-coverage.yaml`
- `tests/fixtures archive builders`
- `FTS/blob/lineage tests`


---

