# Evidence register

## Authority and snapshot

This analysis used the task brief as the question and applied the authority order requested there: current source, repository instructions, current Beads records, then older plans/history where they still agree.

| Input | SHA-256 | Size |
| --- | --- | ---: |
| `00-polylogue-all.tar(11).gz` | `3d0c5a2e5024632c05438dbc3a08aae1dcd31529472b1185ad5b80c0cd933e30` | 92,457,778 bytes |
| `00-slightly-stale-context-testsuite-diet.tar(11).gz` | `d4fa7fc31c70ff30db076e526836ffc81b8124d3f54c7add5e58d3f57c9dcbff` | 934,006 bytes |
| `04-surface-semantic-matrix.md` | `2d55c9e7b05884f4445881026be348e8f4f6dd395b31e1ddc3aa46bfc40fa530` | 3,278 bytes |

The project-state snapshot reports generation at `2026-07-17T043202Z`, branch `master`, commit `f654480cadb7cc4c194704e24dfd483199547b35`, dirty state true, and 976 Beads. The extracted branch-delta file list, log, and patch are empty. Analysis was performed in a clone at the recorded commit. The only local tracked change created during analysis was an index rewrite of `uv.lock` while enabling dev dependencies; it was reverted before packaging.

`AGENTS.md:17-39` is the central repository doctrine: substrate owns meaning, surfaces are leaf adapters, and new semantics land in storage/insights or the product layer first.

## Source evidence by concept

### Identity, origin, provider, and provenance

- `AGENTS.md:47-53` defines generated identity: session `origin:native_id`, message beneath session using native id or position/variant, and block beneath message.
- `polylogue/core/identity_law.py:33-70` implements those deterministic laws.
- `polylogue/storage/sqlite/archive_tiers/write.py:216-220` maps provider to origin before generating the normalized session id.
- `docs/provider-origin-identity.md:15-43` separates provider-wire family, public origin, material source, capture mode, and archive refs.
- `docs/provider-origin-identity.md:47-63` provides the classification table; `:65-92` records current invariants.
- `polylogue/core/sources.py:214-277` implements `Provider`→`Origin`; Gemini and Drive both map to `Origin.AISTUDIO_DRIVE`, so reverse mapping is not one-to-one.
- `polylogue/browser_capture/models.py:112-181` models provider-native sessions and capture ids before normalization.
- `polylogue/browser_capture/models.py:209-228` preserves provider/native/raw coordinates while separately reporting indexed archive session identity.
- `polylogue/surfaces/payloads.py` ImportExplain payloads keep material source, provider hint, capture mode, origin, and produced refs as separate fields.

**Observed conclusion:** public archive identity/vocabulary is coherent. Browser/raw/provider use is not a public-origin leak when it occurs before normalization or as explicit provenance.

### Canonical query intent and response

- `polylogue/archive/query/spec.py:199-243` lists recognized public parameter names.
- `polylogue/archive/query/spec.py:439-495` defines the full immutable `SessionQuerySpec`, including structured filters, ranking/similarity controls, pagination, boolean predicate, and attached-unit projection.
- `polylogue/api/search_envelope_builder.py:42-98` builds a search envelope from an already normalized spec and centralizes cursor/fetch/count/diagnostics assembly for callers that use it.
- `polylogue/surfaces/payloads.py:1249-1318` declares `SearchEnvelope`. Its documentation says `total` is the full matching-session count and that every read surface computes it through the shared spec count. It also defines offset, next offset, opaque cursor, lane, ranking policy, diagnostics, route state, provenance refs, and exactness.
- `polylogue/surfaces/payloads.py:3009-3065` is the canonical envelope builder; its own comment says all four surfaces should call it.

**Observed conclusion:** query intent and wire shape have canonical models. Execution, counting, and paging are not yet singular.

### Daemon canonical versus split-archive path

Canonical path:

```text
GET /api/sessions
  -> DaemonAPIHandler._handle_list_sessions
  -> _build_query_spec_params
  -> _do_list
  -> SessionQuerySpec.from_params + compile_expression_into
  -> list/count or build_search_envelope_for_spec
```

Source: `polylogue/daemon/http.py:2267-2408`.

Split-archive path:

```text
GET /api/sessions
  -> _handle_list_sessions
  -> _web_reader_archive_root() is present
  -> _do_archive_list_sessions(archive_root, raw params, limit, offset)
  -> compile only query expression + manually merge selected HTTP fields
  -> ArchiveStore.list_summaries/search_summaries/count_sessions
  -> manually assembled JSON
```

Source: `polylogue/daemon/http.py:2282-2285`, `:2410-2734`.

Load-bearing observations:

- `:2476-2479` explicitly says the fast path does not construct `SessionQuerySpec` and must mirror parameters.
- `_filter_kw` at `:2512-2539` contains only a subset of the spec.
- `:2610-2678` hardcodes `retrieval_lane="dialogue"` and ranking metadata, sets `total=len(hits)`, and emits no cursor or next offset.
- List mode computes a filtered count (`:2704-2734`), but unsupported behavioral fields remain ignored.
- Explicit zero bounds can be erased by `or ... or None` at `:2504-2507`, unlike `optional_int()` in `query/spec.py:189-196`, which deliberately preserves zero.

`daemon-query-field-risk.csv` turns this call-path comparison into a field inventory.

### CLI paths

```text
CLI common read
  -> _try_emit_daemon_session_page
  -> support discriminator
  -> daemon /api/sessions
  -> normalize daemon row / rebuild CLI envelope

CLI unsupported read or mutation
  -> local ArchiveStore
  -> local list/search/mutation implementation
  -> local envelope
```

- `polylogue/cli/archive_query.py:894-980` is daemon-first read dispatch.
- `:1071-1101` excludes unsupported query/projection controls from daemon mode.
- `:1335-1354` normalizes the legitimate `word_count`/`words` presentation difference.
- `:1357-1416` derives daemon list continuation but discards daemon search continuation.
- `:1880-1946` directly implements delete preview/confirmation/execute.
- `:1971-2050` sets direct list/search `total` to emitted row count.

**Observed conclusion:** CLI presentation compatibility is intentional; totals, cursor loss, and direct write policy are semantic forks.

### Python API and repository twins

- `polylogue/api/archive.py:3661-3790` exposes overlapping list/search methods with direct ArchiveStore access.
- `polylogue/api/archive.py:3792-3901` builds `search_envelope()` manually with the shared payload builder rather than delegating to `build_search_envelope_for_spec()`.
- `polylogue/api/archive.py:3903-4085` exposes a second broad archive list/count/search family.
- `polylogue/api/archive.py:4541-4570` exposes another spec-based query method.
- `polylogue/api/archive.py:4820-4850` resolves and deletes directly through `ArchiveStore`.
- `polylogue/storage/repository/__init__.py:44-93` exposes the async repository/backend boundary, separate from current split-tier sync ArchiveStore paths.

**Observed conclusion:** names do not determine authority. The storage/domain implementation that actually owns normalized rows is authoritative; repository/facade methods are adapters or legacy twins. `polylogue-hiu` contains the already-ratified consolidation direction.

### MCP canonical and legacy paths

- `polylogue/mcp/server_tools.py:209-266` canonical search lowers a typed request to `SessionQuerySpec`, opens ArchiveStore, attaches response context, and emits `archive_search_payload()`.
- Canonical list uses the corresponding typed request and `archive_session_list_payload()` in the same module.
- `polylogue/mcp/archive_support.py:277-469` separately orchestrates list/search counts, rows, and envelopes. Similar-session totals may be inferred from page state rather than counted exactly.
- `polylogue/mcp/server_tools.py:451-652` retains legacy `archive_list_sessions` and `archive_search_sessions`, which call facade methods and build their own totals/paging.
- `polylogue/mcp/server_tools.py:268-373` shows the desired execution direction: `query_units()` creates `QueryExecutionContext` and calls `execute_archive_read()`.
- `polylogue/mcp/server_support.py:47-48` defines process roles and a 25,000-byte budget.
- `polylogue/mcp/server_support.py:193-277` enforces that budget after serialization and may replace the entire payload with a metadata-only envelope and narrower replay call.

**Observed conclusion:** MCP has typed canonical request/payload code, duplicate legacy tools, and a transport budget that is not yet physical lossless paging.

### Mutations and operation authority

- `polylogue/operations/specs.py:47-101` defines descriptive `OperationSpec`/`OperationCatalog`; `:845-856` builds cached catalogs.
- Production references found by source search are artifact graph, scenario metadata/execution, verification, and tests. No production mutation handler dispatches through `OperationCatalog.resolve()`.
- `polylogue/operations/operation_contract.py:3-12` explicitly records that the generic operation framework was removed because only import landed and nothing dispatches on `.kind`.
- `polylogue/maintenance/targets.py:14-109`, `:158-246` defines the narrower executable maintenance target catalog.
- `polylogue/maintenance/planner.py:423-543` uses that catalog for preview and execute; `polylogue/maintenance/replay.py` uses it for replay.
- `polylogue/daemon/http.py:4445-4499` calls maintenance preview/execute directly.
- CLI maintenance and MCP maintenance modules similarly call concrete planners; CLI/API/MCP/daemon deletes use separate direct implementations/adapters.

**Observed conclusion:** `OperationSpec` describes required policy but is not current mutation authority. Maintenance target resolution is already a real narrow authority and should be composed, not replaced.

### Follow-up route defect

- `polylogue/daemon/http.py:4410-4415` returns an `OperationFollowUp` with `/api/operations/{op_id}`.
- `polylogue/operations/operation_contract.py:65-90` says the emitted status identifier must be accepted by that surface’s lookup tool.
- `polylogue/daemon/route_contracts.py:449-455` registers `/api/maintenance/status/:id`.
- Source search found no `/api/operations/:id` route or dispatch.

**Observed conclusion:** the advertised URL has no matching route. **Unresolved:** which registry owns ingest ids and therefore what endpoint should be canonical.

### Authorization and disclosure

- `polylogue/daemon/route_contracts.py:27-34` defines daemon auth vocabularies.
- `polylogue/daemon/http.py:1709-1733` enforces machine auth, same-origin, and write gating for reset/ingest/maintenance. First-party web cookies are intentionally insufficient for these archive-control operations.
- `polylogue/browser_capture/server.py:197-230` enforces allowed origin and optional token; preflight checks origin only.
- `polylogue/mcp/server_support.py:47-48` defines process-level `read/write/review/admin` roles.
- CLI/Python/repository execution relies on local process/OS authority.

**Observed conclusion:** transport authorization should remain edge-specific. Shared operation policy must own capability, target resolution, preview, confirmation, idempotency/conflict, and receipt semantics without trying to replace CORS, bearer, MCP process role, or OS policy.

### Browser errors and status

- `polylogue/browser_capture/models.py:192-205` defines full local receiver status including spool path.
- `polylogue/daemon/status.py:419-444` strips spool/artifact paths for public daemon status.
- `polylogue/browser_capture/models.py:338-344` defines the stable safe receiver error envelope.
- `polylogue/browser_capture/server.py:193-195` uses it for normal receiver errors.
- `polylogue/browser_capture/server.py:520-525` emits capture-job errors with a different nested object.

**Observed conclusion:** status redaction is legitimate; the error shape is a real divergence unless made an explicit versioned subprotocol.

### Status duplication

- `polylogue/daemon/status.py:349-390` declares `DaemonStatus` as consumed by all surfaces.
- `polylogue/daemon/status.py:1986` and `:2192` build and serialize daemon status.
- `polylogue/cli/commands/status.py:603`, `:755`, `:1140`, `:1319`, `:1528`, `:1924`, and `:1986` independently probe tiers, readiness, workload, compact output, direct component readiness, and claim guards.

**Observed conclusion:** projection may differ; fact computation should not. Bead `polylogue-703` records observed production disagreement and owns convergence.

## Beads adjudication

| Bead | Current relevance to this analysis |
| --- | --- |
| `polylogue-4p1` / `.1` | Existing read algebra and exact daemon fast-path repair; no new surface framework required |
| `polylogue-z9gh.1` | Interruptible, resource-bounded read execution |
| `polylogue-z9gh.9` / `.1` | Sole bounded/resumable query transaction and exact cross-surface totals/order/pages/cursors/refs |
| `polylogue-yeq.3` | Query laws, differential parity, and scale regression ownership |
| `polylogue-t46`, `.8`, `.8.2.1` | Surface contract ownership and MCP duplicate-tool convergence/removal |
| `polylogue-t46.9` | Executable operation declaration/handler inventory |
| `polylogue-kwsb.2` | Mutation transaction semantics |
| `polylogue-71ey` | Maintenance catalog/replay authority |
| `polylogue-703` | One status-fact assembly |
| `polylogue-hiu` | Ratified storage-twin collapse onto sync core plus async adapter |
| `polylogue-s1kr` | Generated, committed surface parity matrix and docs |
| `polylogue-1fp` | Facade decomposition after semantics stabilize |
| `polylogue-2qx/.1` | Origin declaration/drift control |
| `polylogue-cuxz.2/.3` | Independent absence/authority/provenance/coverage/freshness axes |
| `polylogue-06zm`, `ptx`, `gnie`, `jlme.5` | Browser durable recovery, action conduit, secure pairing, receiver identity |

No new top-level architecture Bead is necessary. The ingest follow-up defect can be an acceptance item/subtask under operation contract execution (`t46.9`) unless tracker policy requires a dedicated bug.

## Test evidence

Environment setup and test evidence are separated from product results:

1. Initial test invocation failed before collection with `ModuleNotFoundError: hypothesis`.
2. `uv sync --extra dev --frozen` could not resolve locked public package URLs because public DNS/network access was unavailable in the analysis environment.
3. `uv sync --extra dev` resolved the dev bundle from the available internal package index. This rewrote `uv.lock`; the tracked rewrite was reverted before packaging.
4. The selected contract/property suite passed: **86 passed in 7.37s**. Files covered canonical search-envelope construction, cursor laws, origin vocabulary, query descriptor parity, browser receiver and capture jobs, operation specs/contracts, maintenance endpoints, and CLI/daemon golden parity.
5. A focused daemon/operation run passed: **11 passed in 3.58s**. It covered route query-parameter construction, bounded SQL helper use, list/search envelope shape and target refs, split-tier search, degraded FTS state, shared list payload, privacy-safe browser status, and OperationSpec structural contracts.
6. An earlier broad invocation of `test_web_reader.py` plus `test_daemon_http_contracts.py` did not finish within the 300-second analysis limit after substantial progress. It is recorded as a non-conclusive harness timeout, not as a passing or failing product result.

The 97 passing tests establish that current isolated contracts hold. They do **not** establish cross-path equivalence: the seeded search envelope test has only three hits; the bounded helper test searches source text; OperationSpec tests validate catalog structure, not runtime dispatch; no test follows the ingest ack URL; no generated test enumerates every `SessionQuerySpec` field through canonical and split daemon execution.

## Secondary, explicitly stale evidence

The smaller archive is dated `2026-07-16` and is treated only as secondary corroboration:

- `architecture/06-query-cancellation-and-bounds.md` recommends one query execution context and owned read lifecycle.
- `architecture/07-evidence-provenance-and-public-algebra.md` preserves independent EvidenceValue axes and rejects semantic strengthening by projection.
- `architecture/04-destructive-and-authentication-boundaries.md` recommends executable operation declarations, but its proposed universal `OperationExecutor` is broader than current source evidence. This report narrows the recommendation to bindings for real landed mutations and respects `operation_contract.py`’s anti-speculation decision.
- `architecture/09-capture-delivery-and-deployed-status.md` supports independently refreshed component snapshots and disclosure-aware status.
- `areas/status-and-facades.md` supports a shared state-transition matrix and one semantic fact source with surface projections.

Where these documents conflict with current source or Beads, current source/Beads control.

## Limitations and missing evidence

- No live browser extension, receiver, daemon deployment, MCP process, or production archive was available.
- No live operation registry was queried, so ingest-id ownership remains unresolved.
- No live usage telemetry was available to confirm whether legacy MCP tools are externally used; removal must pass a usage/compatibility gate.
- No live-scale archive was used to reproduce response-budget erasure, query cancellation latency, memory pressure, or status disagreement recorded in Beads.
- The broad daemon test files exceeded the local 300-second invocation limit; focused relevant tests passed.
- Static call-path inspection cannot prove that every indirect/decorated mutation path was found. Source search was broad, but runtime plugin registration or external consumers could add paths outside the snapshot.
- The project snapshot was marked dirty even though its branch-delta artifacts were empty. The report treats the captured working tree/source archive as authority and records the commit for orientation, not as a claim that `git show <commit>` alone reproduces every byte.
