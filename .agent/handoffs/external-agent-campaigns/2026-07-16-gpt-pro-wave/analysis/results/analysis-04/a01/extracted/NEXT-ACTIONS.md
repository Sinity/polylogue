# Next actions

The order below is designed to remove active semantic forks first, then establish one execution boundary, then consolidate adapters. Each action includes acceptance criteria, falsification evidence, and local verification.

## 0. Repair the daemon split-archive query path

**Owner:** `polylogue-4p1.1` under `polylogue-4p1`  
**Primary code:** `polylogue/daemon/http.py`, query request lowering, `polylogue/api/search_envelope_builder.py`, ArchiveStore query adapter  
**Priority:** immediate, contained

Implementation:

1. Build `query_params` once in `_handle_list_sessions()` and pass the normalized mapping/spec into both execution modes.
2. Compile the expression into a base `SessionQuerySpec` exactly once.
3. Replace the manual `_filter_kw` mirror with a spec/plan-to-ArchiveStore adapter owned below the route, or route the fast path through the shared query executor when available.
4. Build ranked pages through the canonical envelope/page builder. Preserve degraded FTS route state as an additive projection.
5. Delete the manual mirror comment/block once no route depends on it.

Acceptance:

- A generated test enumerates every `dataclasses.fields(SessionQuerySpec)` member and classifies it as filter, ordering, page, similarity, boolean, or projection. No field can be added without an explicit route/executor decision.
- For a seeded archive with more than two pages, canonical and split modes return identical ids, order, complete total/exactness, page boundaries, next cursor/offset, lane/ranking metadata, target refs, diagnostics, and route state for every supported field and representative combinations.
- Explicit `min_messages=0`, `max_words=0`, and equivalent zero bounds remain bounded rather than becoming `None`.
- Exclusion text, project, typed-only, message type, boolean expressions, `since_session_id`, similarity, latest/sample/sort/reverse, and cursor are either implemented identically or rejected with the same typed valid-values/unsupported response before execution. Silent ignore is forbidden.

Falsification:

- Monkeypatch the canonical builder/spec adapter to raise and prove that both daemon modes fail; any surviving response proves a bypass.
- Add a new dummy spec field in a mutation test; the inventory test must fail until classified.
- Seed `limit=2` with five matches; any `total==2` or absent continuation falsifies parity.

Local verification:

- Run focused daemon route tests in both `_web_reader_archive_root()` states.
- Inspect generated SQL/plan descriptors for identical pushed predicates.
- Run quick contract/property gates and the new field-inventory differential suite.

## 1. Land the shared bounded query transaction

**Owners:** `polylogue-z9gh.1`, `polylogue-z9gh.9`, `polylogue-z9gh.9.1`; semantic algebra `polylogue-4p1`  
**Primary code:** archive query execution layer, read-only connection lifecycle, cursor/result page models, surface adapters  
**Priority:** P0

Implementation:

- Make one request/execution/page contract own canonical plan digest, archive snapshot/epoch, actor/disclosure context, admission, deadline/cancellation, exactness/count policy, selected order, physical page, continuation/result ref, and query receipt.
- Run SQLite on an owned read-only connection outside server event loops; cancellation/deadline/disconnect must interrupt that exact connection and clean temp/spool state.
- Fetch/serialize a bounded page rather than materializing the full logical result first.
- Make current `query_units()` execution-control use the same transaction as session list/search/detail/messages/tree/topology.
- Feed existing shared payload builders from the page result; do not make the executor render CLI text, HTTP, or MCP.

Acceptance:

- Identical normalized requests at the same declared snapshot yield identical identities, stable order/frame, total/exactness, page boundaries, cursor state, and result refs across CLI direct/daemon, Python, daemon canonical/split, and MCP.
- Cursor state preserves expression, all structural filters, material scope, projection/render budget, sort/order, snapshot/epoch, and query-run identity; iterating pages yields every logical row exactly once and terminates.
- Cancellation of an expensive recursive/aggregate read interrupts within the recorded SLO while health and a cheap read remain responsive.
- RSS/temp/spool work stays within declared bounds on the live-scale cases recorded by Beads; no surface serializes a full result only to discard it.
- Zero results, empty archive, degraded index, timeout, cancellation, busy/admission-deferred, stale snapshot, and invalid cursor are distinct typed outcomes.

Falsification:

- Remove/bypass the shared executor from any surface and confirm a source/architecture test fails.
- Run a page walk while inserting new rows; the declared epoch behavior must prevent duplicate/skip ambiguity or explicitly reject stale continuation.
- Force a response larger than MCP’s budget; at least one useful page plus lossless continuation must remain, not a metadata-only replacement with incomplete arguments.
- Cancel a query and inspect active SQLite statements/worker resources; any continuing work falsifies ownership.

Local verification:

- Run property tests for pagination laws and differential surface tests over seeded and live-scale archives.
- Capture plan descriptors, peak RSS/temp bytes, cancellation latency, event-loop heartbeat, and cleanup receipts.
- Re-run the known Workflow-tool, coordinator-delegation, 129-session, and recursive topology cases named in `z9gh.9/.1`.

## 2. Build the cross-surface differential parity suite

**Owner:** `polylogue-yeq.3`; generated documentation/matrix consumer `polylogue-s1kr`  
**Primary code:** tests/property, tests/integration or contract harness, generated parity manifest  
**Priority:** P1, parallel with Action 1 once page contract exists

Test design:

- Generate requests from `SessionQuerySpec` field metadata plus a curated interaction matrix, not from manually copied option names.
- Seed at least three pages, tied ranks, missing optional fields, multiple origins/tags/repos/projects, typed/untyped messages, actions/tools, path refs, zero bounds, timestamps, lineage, and degraded search state.
- Execute through CLI direct, CLI daemon, Python facade, daemon canonical, daemon split, MCP canonical, and any retained compatibility tools.
- Normalize only declared presentation differences, then compare logical facts.

Acceptance:

- The comparator checks identity/order, total and exactness, page count, cursor/offset, lane/ranking policy, target/result/query refs, absence/degraded/error categories, and required provenance axes.
- A machine-readable committed matrix maps every semantic operation to each surface and records intentional absence with an owning Bead/reason.
- Mutation tests prove the suite fails for page-total substitution, cursor removal, one omitted filter, provider leakage into normalized rows, zero-as-unknown, and unauthorized destructive execution.

Falsification:

- Replace complete total with page length in one adapter; the test must fail.
- Drop one field from daemon/MCP request lowering; generated inventory must fail.
- Change only CLI text formatting; logical parity must continue passing, proving the suite does not overconstrain presentation.

Local verification:

- Keep a small deterministic fixture for quick gates and a separate live-scale profile for performance/resource laws.
- Record normalized request/plan/page digests in failures so implementers can identify the owning layer.

## 3. Converge MCP verbs and response paging

**Owners:** `polylogue-t46.8`, `polylogue-t46.8.2.1`, `polylogue-z9gh.9.1`  
**Primary code:** `polylogue/mcp/server_tools.py`, `archive_support.py`, `server_support.py`  
**Priority:** after shared page transaction is callable

Implementation:

- Route canonical list/search/query units through the shared transaction.
- Enforce the 25,000-byte transport budget by selecting a smaller physical page/projection before full serialization.
- Emit complete opaque continuation/result refs from the transaction.
- Inventory external/internal callers of legacy archive tools. Remove them when zero-use evidence is sufficient; otherwise place them behind explicit compatibility registration and adapt them to the same transaction.

Acceptance:

- No MCP list/search path owns filter mapping, totals, paging, cursor, or query receipt independently.
- Oversized logical results retain useful rows and complete continuation state.
- Canonical and compatibility verbs produce identical logical results for identical requests.
- Role discovery and write-role enforcement remain MCP-specific edge policy.

Falsification:

- Grep/source test finds a legacy tool calling a broad facade list/search method directly.
- Force an oversized response; if the full body is first built or all evidence rows disappear, the action fails.
- Usage scan finds a live caller after removal without a migration path.

## 4. Make mutation declarations executable and repair operation follow-up

**Owners:** `polylogue-t46.9`, `polylogue-kwsb.2`, maintenance authority `polylogue-71ey`  
**Primary code:** operation specs/bindings, mutation transaction, import ack/status registry, maintenance planner/replay, surface adapters  
**Priority:** P1

Implementation:

- Add executable bindings for real operations only: stable operation id/version, handler, capability, target resolver, preview requirement, confirmation strength, idempotency/conflict rule, affected tiers, and receipt projection.
- Reuse `MaintenanceTargetCatalog` for maintenance resolution/replay rather than creating a competing target model.
- Convert CLI/API/MCP/daemon mutation handlers into actor/auth adapters that submit the same request and project the same preview/receipt/outcome.
- Preserve defense-in-depth storage guards.
- Add a generic operation-status route only if one registry truly owns all ids; otherwise make each follow-up URL name the actual owning registry.

Acceptance:

- Every runtime `OperationSpec` with a mutating effect either has a real executable binding or is explicitly declaration-only and cannot appear as executable surface capability.
- Destructive operations require preview evidence and a confirmation token bound to actor, operation/version, archive, target-set digest, and expiry; stale target sets fail closed.
- Same idempotency key/request cannot execute twice; conflicts and partial failures produce typed durable receipts.
- A test schedules ingest, follows the returned `status_endpoint`, resolves the same operation id, and reaches terminal status.
- CLI, daemon, MCP, and Python delete/tag/maintenance/import outcomes agree on target identity, affected count, no-op/not-found, preview, and receipt refs while retaining edge-specific auth UX.

Falsification:

- Patch an operation’s declared capability/preview rule; a surface that still executes directly proves a bypass.
- Change targets after preview; execution must return `preview_stale`.
- Follow the ingest ack URL in the current implementation; it should fail, preserving the regression until repaired.
- Grep/source test finds direct surface calls to registered mutation handlers outside the transaction adapter allowlist.

Local verification:

- Run real split-tier fixtures for user overlay writes, session delete, reset, import scheduling, and maintenance preview/execute/replay.
- Inspect ops/user/source/index tier effects and receipt durability across restart.

## 5. Converge status facts without flattening disclosure

**Owner:** `polylogue-703`  
**Primary code:** `polylogue/daemon/status.py`, `polylogue/cli/commands/status.py`, ops diagnostics, browser/public projections  
**Priority:** P2 after read transaction fact snapshots stabilize

Implementation:

- Define one status snapshot assembly over archive tiers, daemon lifecycle, convergence, cursor lag, ingest attempts, FTS/insight/embedding readiness, browser receiver, storage/disk, and claim guards.
- Mark collector authority, observed time, freshness, degraded/unavailable reason, and disclosure class.
- Make CLI, daemon, MCP, web, and browser status pure projections/compactors/redactors.

Acceptance:

- State-transition fixtures cover empty, ingesting, pending convergence, partially failed, stale, repaired, healthy, and unavailable.
- Every surface agrees on shared fact value/state/authority/time while allowed field sets and path redaction differ.
- A single snapshot prevents mixed-frame claims during rebuild.
- Browser public status never exposes spool/artifact paths; local receiver status may.

Falsification:

- Patch a shared fact collector; all projections must change consistently.
- Remove the daemon status builder and prove CLI direct status cannot silently recompute the same semantic fact.
- Seed known zero versus unavailable; renderers must not collapse them.

## 6. Collapse storage/API twins and generate the governed matrix

**Owners and order:** `polylogue-hiu` → `polylogue-s1kr` → `polylogue-1fp`  
**Priority:** P2/P3 after Actions 1–3

Implementation:

- Execute the ratified sync-core/async-adapter migration mixin by mixin with ingest and interactive-read gates.
- Redirect facade/repository methods to the same core transaction/operations rather than preserving duplicate SQL.
- Generate the committed CLI/MCP/Python/HTTP semantic-operation matrix from declarations and actual bindings.
- Decompose `polylogue/api/archive.py` by capability protocols only after duplicate semantics are removed.

Acceptance:

- Async storage twins and documented divergence comments are deleted as specified by `polylogue-hiu`; facade signatures remain compatible where promised.
- Ingest throughput remains within the ratified noise envelope and read/cancellation SLOs hold.
- Generated matrix drift-checks every operation/surface, including intentional absence and compatibility mode.
- Facade decomposition changes ownership/imports, not evidence semantics.

Falsification:

- Differential tests pass only because both twins share the same bug; mutation tests against canonical laws must still fail.
- A facade file split introduces a new count/filter/page implementation; source guard fails.
- Generated matrix lists a method by name but cannot resolve its actual handler/call path; generation fails.

## 7. Close identity/evidence/browser protocol gaps

**Owners:** `polylogue-2qx/.1`, `polylogue-cuxz.2/.3`, `polylogue-06zm`, `polylogue-ptx`, `polylogue-gnie`, `polylogue-jlme.5`  
**Priority:** continue alongside later consolidation where dependencies permit

Acceptance:

- Generated origin declarations cover public filters/rows and prove Provider→Origin bridges, including non-injective mappings.
- ImportExplain and capture state preserve material source, provider/native id, capture mode, parser binding, normalized origin, and produced archive refs as distinct coordinates.
- Declared EvidenceValue fact families preserve absence, authority, temporal provenance, enumeration/coverage, freshness, and calibrated confidence across every surface.
- Receiver errors use one versioned envelope; capture-job consumers have contract tests.
- Pairing, receiver identity, durable capture recovery, and browser action conduit retain exact-origin/token and disclosure boundaries.

Falsification:

- Replace provider-native id with archive id before normalization or leak provider as the normalized public source filter; identity tests fail.
- Render unknown price/coverage as zero or archive-wide certainty; evidence law tests fail.
- Capture-job error omits receiver/schema identity without an explicit versioned subprotocol; receiver contract fails.

## Expected value of another iteration

A second ordinary iteration has high value after Action 0 or a prototype of Action 1 lands. It can run the generated field inventory against the new code, inspect concrete query-page and mutation-binding types, and resolve the ingest status registry from implementation rather than inference. Before code changes, another static pass would mostly refine inventories and discover additional call sites; useful, but lower value than converting the current findings into executable differential tests.
