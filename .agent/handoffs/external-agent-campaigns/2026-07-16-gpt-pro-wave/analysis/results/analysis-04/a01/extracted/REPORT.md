# Cross-surface semantic parity matrix

**Job:** `analysis-04`  
**Revision:** `r01`  
**Evidence snapshot:** Polylogue `master` at `f654480cadb7cc4c194704e24dfd483199547b35`, generated `2026-07-17T043202Z`  
**Surfaces covered:** CLI, Python API, repository/operations, daemon HTTP, MCP, browser capture/status

## Executive conclusion

Polylogue already has the right architectural rule: storage and product layers own meaning, while CLI, API, daemon, MCP, and browser code are projections. Current code follows that rule well for normalized identity and source vocabulary, but not yet for read execution, mutation admission, or status assembly.

The strongest shared semantics are identity and public origin. Stored session identity is deterministically computed as `origin:native_id`; message and block identities are derived beneath it. Public archive reads use `origin`; parser, acquisition, and browser-capture boundaries retain `provider` and provider-native ids as provenance. The Gemini/Drive-to-`aistudio-drive` bridge is intentionally non-injective. This is a sound model, not a naming defect.

The highest-risk divergence is execution-path-dependent query meaning. `SessionQuerySpec` is the canonical selection intent and `SearchEnvelope` is the declared cross-surface response, but the daemon split-archive path manually re-lowers filters and manually assembles search responses. It omits or weakens fields including exclusion text, retrieval lane, project, typed-only, message type, `since_session_id`, boolean predicates, similarity, sample/latest/sort/reverse, and cursor state. It reports `total=len(hits)` for a search page and emits no continuation. CLI direct mode also reports page length as `total`; CLI daemon search discards continuation; legacy MCP archive tools have their own paging and total rules; Python `search_envelope()` repeats orchestration rather than calling the existing shared spec builder. The result model is shared more than the result transaction.

Mutation semantics are even less centralized. `OperationSpec` declares names, effects, surfaces, previewability, and safety guards, and its tests validate those declarations. Runtime CLI, API, daemon, and MCP handlers still call concrete write implementations directly. No production dispatcher authorizes or invokes a handler by `OperationSpec`; the source explicitly records that no generic `.kind` dispatch exists. The maintenance target catalog is a real executable authority for maintenance targets, but it does not unify delete, import, tag, reset, and other mutations. This makes `OperationSpec` descriptive metadata, not current mutation authority.

One concrete operation-contract defect is already visible without a live deployment: daemon ingest returns a follow-up URL `/api/operations/{id}`, while the registered status route is `/api/maintenance/status/:id`; no generic `/api/operations/:id` route was found. The follow-up contract says the emitted identifier must be accepted by the same surface. The correct repair depends on which registry actually owns ingest ids; mapping ingest ids into the maintenance registry is not established by the snapshot.

Status and browser differences divide cleanly into legitimate projections and real duplication. Browser receiver status may expose its spool path locally, while daemon public status removes spool and artifact paths; daemon machine mutations require bearer-style machine auth plus same-origin and a write gate; browser capture requires exact allowed origin plus an optional token; MCP uses process role; CLI/Python inherit local process authority. Those are legitimate edge policies. By contrast, daemon status and CLI status independently probe and assemble many of the same archive facts, and browser capture-job errors bypass the receiver’s otherwise stable safe error envelope. Those are parity defects.

The correct remedy is not one giant “surface abstraction.” It is a small set of owning contracts: canonical query intent, one bounded query transaction and page result, shared fact payloads, executable operation bindings plus a mutation transaction, one status-fact assembly, and transport-specific projection/auth adapters. This ordering matches the existing Beads rather than inventing a parallel architecture.

## Semantic ownership map

| Repeated concept | Current substrate/product owner | Surface role | Assessment |
| --- | --- | --- | --- |
| Normalized session/message/block identity | generated identity law and split archive storage | render refs, anchors, native provenance | Coherent |
| Public source family | `Origin`; `Provider`→`Origin` bridge at normalization | accept/render `origin`; retain `provider` before normalization | Coherent with deliberate non-injective bridge |
| Query selection intent | `SessionQuerySpec`, expression compiler, query plan | parse transport arguments into one spec | Correct owner, incompletely used |
| Query execution, snapshot, bounds, totals, page, continuation, receipts | no single production owner yet; partial `QueryExecutionContext`, `build_search_envelope_for_spec`, `build_search_envelope` | invoke and project one page | Systemic gap; Beads `z9gh.9/.1` own it |
| Read wire payloads and target refs | `polylogue/surfaces/payloads.py` | JSON/text/YAML/MCP/browser projection | Mostly coherent; adapters still reassemble facts |
| Mutation target/effect mechanics | concrete ArchiveStore/facade methods; maintenance target catalog/planner/replay | admission, auth, preview, confirmation, projection | Mechanics exist; authority is fragmented |
| Mutation declaration | `OperationSpec` catalog | metadata, verification, discovery | Descriptive only; not a dispatcher/authorizer |
| Authorization/disclosure | transport edge plus local process/OS boundary | enforce edge policy | Legitimate variation; domain guards still need one owner |
| Status facts | daemon and CLI currently assemble separately | redact/render per surface | Duplicate semantic assembly |
| Evidence absence/provenance | typed payloads plus emerging `EvidenceValue` design | preserve unknown/degraded/redacted/coverage axes | Partial; several families still use ad hoc envelopes |

## Actual surface call paths

### CLI

Common non-mutating session pages first attempt `_try_emit_daemon_session_page()` in `polylogue/cli/archive_query.py:894-980`. `_daemon_session_page_supported()` deliberately rejects projections and query features that the HTTP route cannot represent (`:1071-1101`). The adapter then normalizes the daemon web row (`word_count`→legacy CLI `words`) and builds a CLI envelope (`:1335-1416`). This shape conversion is legitimate compatibility work; discarding search cursors is not.

Unsupported reads and all mutations fall back to a local `ArchiveStore`. Direct list/search emitters create their own envelopes and set `total` to the number of emitted rows (`:1971-2050`). Delete performs its own dry-run, interactive/plain confirmation, and direct `delete_sessions()` call (`:1880-1946`). Thus the CLI has two execution paths with different totals and continuation behavior, plus a third semantic path for writes.

### Python API

`Polylogue` is a broad async facade, but many methods open the synchronous split-tier `ArchiveStore` directly. `list_sessions()`, `list_summaries()`, `list_sessions_for_spec()`, `search_session_hits()`, legacy `search()`, `search_envelope()`, `archive_list_sessions()`, `archive_search_sessions()`, `query_sessions()`, counts, and mutations overlap in capability (`polylogue/api/archive.py:3661-4085`, `:4541-4570`, `:4820-4850`). `search_envelope()` uses the canonical payload builder but duplicates spec/cursor/fetch/count assembly instead of delegating to `build_search_envelope_for_spec()`.

The public Python facade’s smaller keyword signature is a legitimate projection difference. The duplicate execution and counting paths are not.

### Repository / operations

`SessionRepository` wraps a separate async SQLite backend and retains legacy `source`-named query concepts, while current API and daemon paths use the split-tier sync `ArchiveStore`. Source evidence therefore does not support “the repository” as the semantic owner merely because of its name. The practical fact owner is the current storage/domain implementation; Bead `polylogue-hiu` already decides to collapse the storage twins onto the sync core behind async adapters.

`OperationSpec` is used by artifact graphs, scenarios, verification, and contract tests. No production surface dispatches a mutation from `OperationCatalog.resolve()` or `.kind`. Concrete maintenance does use `MaintenanceTargetCatalog` in `polylogue/maintenance/planner.py` and `polylogue/maintenance/replay.py`, so that narrower catalog is current executable domain authority for maintenance target resolution.

### Daemon HTTP

Without the web-reader split archive root, `/api/sessions` constructs a `SessionQuerySpec`, compiles the expression into it, and routes ranked search through `build_search_envelope_for_spec()` (`polylogue/daemon/http.py:2267-2408`). With a split archive root, the same route bypasses that code and calls `_do_archive_list_sessions()` (`:2282-2285`, `:2410-2734`). The fast path says it must “mirror” public parameters manually (`:2476-2479`), constructs `_filter_kw`, and hand-assembles search output. It hardcodes the dialogue lane and mixed policy, sets `total=len(hits)`, and emits no `next_offset` or `next_cursor` (`:2610-2678`).

Daemon machine mutation routes have a clear edge policy: machine auth only, cross-origin rejection, and a write gate (`:1709-1733`). That policy should remain at the HTTP boundary. It does not replace the need for one domain mutation request/preview/receipt contract.

### MCP

Canonical `search` builds `MCPSessionQueryRequest`, lowers it to `SessionQuerySpec`, uses `ArchiveStore`, and emits `archive_search_payload()` under a replay context (`polylogue/mcp/server_tools.py:209-266`). Canonical `list_sessions` follows the same pattern. `query_units()` is already the best model for the next step: it creates a shared `QueryExecutionContext` and calls `execute_archive_read()` (`:268-373`).

Legacy `archive_list_sessions` and `archive_search_sessions` remain separately registered and call Python facade methods with separate totals and pagination. The MCP response budget is enforced after full serialization; when exceeded, the body is replaced by metadata and a narrowed continuation (`polylogue/mcp/server_support.py:193-277`). This is transport-safe but not lossless query paging. Beads `polylogue-t46.8.2.1` and `polylogue-z9gh.9.1` already own removal/compatibility and physical paging respectively.

### Browser capture / status

Browser capture is a pre-normalization acquisition surface. Its session model correctly uses `Provider` and `provider_session_id`; default `capture_id` is provider-native (`polylogue/browser_capture/models.py:112-181`). Archive state later carries both raw/native provenance and an optional indexed archive session id (`:209-228`). Forcing public `Origin` into this receiver would erase useful acquisition truth.

Receiver auth is exact-origin plus optional bearer token, with preflight origin-only because CORS preflights cannot carry Authorization (`polylogue/browser_capture/server.py:197-230`). Receiver status includes local spool information; daemon public status removes it (`polylogue/daemon/status.py:419-444`). Those are legitimate disclosure differences. The receiver’s general errors use `BrowserCaptureErrorPayload`, while capture-job errors emit a nested `{error:{code,details}}` object without the shared `ok/receiver/schema_version` wrapper (`polylogue/browser_capture/server.py:193-195`, `:520-525`); no versioned contract exception was found.

## Ranked findings

### F1 — The declared read contract is not the production read transaction

**Observed.** `SessionQuerySpec` owns selection and `SearchEnvelope` documents uniform totals/cursors, but CLI, Python, daemon split mode, MCP canonical helpers, and MCP legacy helpers independently execute, count, page, and assemble results.

**Source-supported inference.** A consumer can receive different `total`, continuation, ranking metadata, and even filter semantics for the same logical request solely because an execution path changed. The daemon fast path and CLI daemon/direct discriminator make this path switch routine rather than theoretical.

**Decision.** Implement the existing `polylogue-z9gh.9/.1` shared query transaction beneath surface adapters. Do not replace the query algebra or payload models; make them inputs/outputs of one bounded execution/page boundary.

### F2 — The daemon split-archive route is a concrete semantic fork

**Observed.** `_do_archive_list_sessions()` manually mirrors a subset of `SessionQuerySpec`. `daemon-query-field-risk.csv` records each field. Search totals are page length, lane is hardcoded, and continuation is absent.

**Source-supported inference.** Tests with only three matches cannot detect a page-total defect. The source-inspection test only checks that the bounded SQL helper appears, not that every spec field reaches the same storage predicates.

**Decision.** `polylogue-4p1.1` is the immediate contained repair: construct the spec once from the route parameters, use canonical execution/envelope code, and delete the manual mirror block. This should land before the broader transaction work because it removes an active semantic fork and provides the first field-inventory differential test.

### F3 — Totals and pagination mean different things across adapters

**Observed.** Canonical `SearchEnvelope.total` is documented as the complete matching-session count. Daemon split search, CLI direct list/search, and legacy MCP search use page length. CLI daemon search discards continuation. MCP post-serialization budget replacement can erase all result rows and substitute a narrower replay call.

**Decision.** One page result must name both logical `total` and emitted `page_count` when both are needed. Exactness must be explicit (`exact`, `estimate`, `capped`, `sampled`). Cursor state must preserve the canonical expression, filters, projection, order, snapshot/epoch, and query identity. A response budget may choose a smaller physical page but must not redefine the logical result.

### F4 — `OperationSpec` is declaration metadata, not current mutation authority

**Observed.** Runtime catalog users resolve metadata for artifact/scenario/verification purposes. CLI, daemon, MCP, and API mutation handlers call concrete implementations directly. `operation_contract.py` explicitly states that generic operation bases were removed because only import had a production consumer and nothing dispatches on `.kind`.

**Decision.** Follow `polylogue-t46.9` and `polylogue-kwsb.2`, but bind only real operations. Introduce executable handler/capability/preview/receipt bindings around existing mutation implementations and the existing maintenance target catalog. Do not resurrect a speculative polymorphic base for operation kinds that still have no handler.

### F5 — Daemon ingest advertises an unregistered status URL

**Observed.** `POST /api/ingest` emits `/api/operations/{id}`. The route registry exposes `/api/maintenance/status/:id`; no `/api/operations/:id` registration or dispatch was found.

**Unresolved.** The snapshot does not establish that ingest operation ids are present in the maintenance registry. Therefore changing the link to `/api/maintenance/status/{id}` without proving registry ownership could replace a broken URL with a semantic mismatch.

**Decision.** Add an endpoint-contract test that schedules a stub ingest operation and follows the returned URL through actual route dispatch and registry lookup. Then either register a generic operation-status route backed by the ingest registry or emit the existing route that truly owns that id.

### F6 — Identity and origin/provider semantics are already substantially aligned

**Observed.** Public archive payloads and filters use `origin`; normalized ids use `origin:native_id`; provider/native ids remain parser, raw, capture, and topology provenance. Daemon archive rows intentionally omit `provider`. Browser capture remains provider-native before normalization.

**Decision.** Preserve this division. Continue `polylogue-2qx/.1` for generated declaration/drift control, not a runtime vocabulary rewrite. Test that public normalized rows never regress to `provider`, and that ImportExplain/browser capture never lose provider/material/capture-mode provenance.

### F7 — Absence/error/provenance are typed in important places but not governed end to end

**Observed.** Query miss, degraded route state, target refs, exactness, and safe receiver error models exist. Different surfaces still construct empty/degraded/error envelopes manually; capture-job errors have a separate shape; several numeric and status facts do not yet carry independent authority, coverage, freshness, and absence axes.

**Decision.** Use `polylogue-cuxz.2/.3` to govern fact families, not to wrap every scalar indiscriminately. Parity tests must distinguish known zero from unknown/unavailable, exact census from estimate/sample, stale from absent, and redacted from missing.

### F8 — Status needs one fact assembly and multiple projections

**Observed.** `DaemonStatus` is intended for all surfaces, but `polylogue/daemon/status.py` and `polylogue/cli/commands/status.py` each perform substantial archive probing and readiness assembly. Browser public status legitimately removes local paths.

**Decision.** Implement `polylogue-703`: one substrate status snapshot/fact assembly, then CLI, daemon, MCP, web, and browser projections with explicit disclosure policies. Do not force all surfaces to expose the same fields.

### F9 — Storage/API twins amplify drift but should be sequenced after semantic tests

**Observed.** The Python facade exposes overlapping old/new query methods; an async repository/backend and the sync split-tier store coexist. Bead `polylogue-hiu` has already selected sync core plus async adapter.

**Decision.** First land differential semantic tests and the shared query transaction; then execute `polylogue-hiu`, generate the operation-level surface matrix under `polylogue-s1kr`, and only then decompose the broad facade under `polylogue-1fp`. File splitting before semantic convergence would redistribute drift rather than remove it.

## Real divergence versus legitimate presentation differences

| Case | Classification | Reason |
| --- | --- | --- |
| `word_count` in daemon web rows versus `words` in legacy CLI rows | Legitimate compatibility projection | Same underlying fact; CLI adapter normalizes the stable wire shape |
| Browser receiver exposes spool path; daemon public status redacts it | Legitimate disclosure projection | Same readiness fact under different privacy boundary |
| CLI text/YAML/JSON, daemon JSON, MCP serialized model | Legitimate render difference | Allowed if logical ids/order/totals/absence/provenance remain equal |
| Browser `provider` before normalization versus archive `origin` | Legitimate lifecycle vocabulary | Different coordinates with an explicit bridge |
| Daemon split route omits filters and hardcodes lane | Real semantic divergence | Selection and ranking change with route implementation |
| Page length used as `total` | Real semantic divergence | Contradicts the canonical complete-match count contract |
| Cursor removed by CLI daemon adapter | Real semantic divergence | A successful logical result becomes non-resumable |
| MCP full serialization followed by metadata-only replacement | Real execution/transport mismatch | Payload size is checked only after full serialization, and rows can be erased |
| OperationSpec safety guards not consumed by runtime mutation dispatch | Real authority gap | Declaration and enforcement can drift independently |
| Ingest follow-up URL has no registered matching route | Real route-contract defect | Client cannot follow the advertised lookup identifier |
| CLI and daemon independently compute status facts | Real duplication | Same archive state can produce conflicting semantic claims |
| Browser capture-job nested error object | Real envelope divergence unless versioned | Same receiver has two incompatible error shapes without documented split |

## Priority and ownership

1. **Remove the active daemon query fork** — `polylogue-4p1.1`.
2. **Land bounded, resumable query execution across surfaces** — `polylogue-z9gh.1`, `polylogue-z9gh.9`, `polylogue-z9gh.9.1`, with parity laws in `polylogue-yeq.3` and read algebra in `polylogue-4p1`.
3. **Converge MCP tools and physical paging** — `polylogue-t46.8`, `polylogue-t46.8.2.1`, under the same query transaction.
4. **Make real mutations executable from declared policy** — `polylogue-t46.9`, `polylogue-kwsb.2`, and `polylogue-71ey`; include the ingest follow-up repair.
5. **Converge status facts** — `polylogue-703`.
6. **Collapse implementation twins and govern the public matrix** — `polylogue-hiu`, `polylogue-s1kr`, then `polylogue-1fp`.
7. **Continue identity/evidence declarations and browser delivery** — `polylogue-2qx/.1`, `polylogue-cuxz.2/.3`, `polylogue-06zm`, `polylogue-ptx`, `polylogue-gnie`, `polylogue-jlme.5`.

Detailed acceptance criteria and falsification checks are in `NEXT-ACTIONS.md`. Source and test evidence are in `EVIDENCE.md`. The machine-readable cross-surface map is `surface-semantic-matrix.csv`; the daemon field inventory is `daemon-query-field-risk.csv`.
