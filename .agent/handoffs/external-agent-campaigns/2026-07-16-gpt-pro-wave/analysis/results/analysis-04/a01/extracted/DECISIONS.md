# Decisions

## D1 — Facts are owned by the current domain/storage substrate, not by whichever adapter is named “repository”

Normalized sessions, messages, blocks, refs, queryable action relations, and derived archive facts are owned by the current split-tier storage/domain code. `SessionRepository`, `Polylogue`, CLI, daemon, and MCP are access/projection boundaries. The async repository/backend is a legacy twin to collapse under `polylogue-hiu`, not a second semantic authority.

**Consequence:** parity work compares adapters against canonical domain/query outputs. It does not copy semantics into a “surface model” or choose the repository API by name.

## D2 — `SessionQuerySpec` owns selection; a bounded query transaction must own execution and paging

`SessionQuerySpec` plus expression lowering remains the canonical selection intent. `Query × Projection × Render` from `polylogue-4p1` remains the product algebra. The missing owner is narrower: canonical plan execution against one declared snapshot/epoch, cancellation/deadline/admission, count/exactness, page boundaries, cursor/result refs, and query receipt.

**Consequence:** implement `polylogue-z9gh.9/.1` beneath existing adapters. Do not create one omniscient surface abstraction or move rendering/auth into the query engine.

## D3 — Logical result facts must be invariant; wire presentation may differ

For identical normalized requests and the same archive snapshot, all read surfaces must agree on selected identities, order, logical total/exactness, page membership, continuation meaning, target/result refs, absence/degraded state, and evidence provenance. CLI text versus JSON, field naming compatibility, MCP wrappers, and browser redaction may differ only through declared projections.

**Consequence:** introduce an explicit `page_count` when consumers need emitted-row count. Do not overload `total`. A budget may shrink a page but may not erase the logical result or drop replay state.

## D4 — Delete the daemon’s manual query mirror

The split-archive fast path must build the same `SessionQuerySpec` as the non-split path and use canonical execution/envelope assembly. ArchiveStore-specific optimization may remain below the spec/plan boundary.

**Consequence:** `polylogue-4p1.1` lands first. A generated field-inventory test compares every spec field and fails whenever a new field is not handled by the fast path.

## D5 — Public normalized source vocabulary is `Origin`; provider/native coordinates remain explicit provenance

Public archive queries and rows use `origin`. Parser/raw acquisition, browser capture, and ImportExplain may use `provider`, material source, capture mode, parser binding, and native ids because they describe pre-normalization/provenance coordinates. Archive ids remain product identity.

**Consequence:** no global rename from provider to origin. Generate declaration/drift tests under `polylogue-2qx/.1`, including the non-injective Gemini/Drive bridge.

## D6 — Absence and provenance are independent semantic axes

Known zero, unknown, unavailable, skipped, redacted, stale, degraded, sampled, capped, and estimated are not interchangeable. Authority/provenance/coverage/freshness must survive all projections for fact families that declare those axes.

**Consequence:** continue `polylogue-cuxz.2/.3` selectively by fact family. Do not wrap every scalar in a universal container; do not let adapters synthesize zero or certainty from absence.

## D7 — Operation declarations become executable only for real landed mutations

`OperationSpec` currently describes effects and guards but does not authorize or dispatch runtime writes. Existing concrete mutation implementations and `MaintenanceTargetCatalog` remain the mechanics. Add stable handler bindings, capability requirements, preview/confirmation rules, idempotency/conflict behavior, and receipt projection for operations that actually exist.

**Consequence:** implement `polylogue-t46.9`, `polylogue-kwsb.2`, and `polylogue-71ey` incrementally. Do not resurrect a generic polymorphic base for unimplemented operation kinds. HTTP CORS/bearer, MCP process role, and OS authority stay at their edges; the mutation transaction consumes the resulting actor/capability context.

## D8 — The ingest follow-up string must be executable, not illustrative

Every accepted/pending operation’s `status_endpoint` must be followed through the same surface and resolve the same operation id. The current `/api/operations/{id}` ingest URL fails this contract because no route is registered.

**Consequence:** add a route-follow test before choosing the final URL. Register a generic route only if there is a generic registry; otherwise emit the actual ingest-status route. Do not assume maintenance registry ownership.

## D9 — One status fact assembly, many disclosure-aware projections

Daemon, CLI, MCP, web, and browser status should consume one snapshot of shared facts. Each projection may omit, redact, compact, or relabel fields according to audience and privacy policy.

**Consequence:** `polylogue-703` owns convergence. Browser spool-path redaction remains intentional; status assembly must explicitly represent unknown/stale/unavailable data rather than recomputing it differently at each edge.

## D10 — Remove or quarantine duplicate MCP verbs after usage evidence

Canonical typed `search`/`list_sessions` and query-unit tools are the forward path. Legacy `archive_list_sessions`/`archive_search_sessions` should be removed or placed behind an explicit compatibility mode after source/telemetry evidence confirms callers and a migration window is defined.

**Consequence:** `polylogue-t46.8.2.1` follows the shared query transaction so all surviving tools inherit physical paging, exactness, cursor, and response-budget behavior.

## D11 — Browser capture remains a separate acquisition/auth boundary, with a stable receiver protocol

The receiver keeps provider-native identity and exact-origin/token policy. Daemon public status keeps its privacy-safe projection. Capture-job errors should use the receiver’s stable error wrapper or declare a separately versioned protocol with explicit consumer tests.

**Consequence:** browser recovery/pairing/action work remains under `polylogue-06zm`, `polylogue-ptx`, `polylogue-gnie`, and `polylogue-jlme.5`; archive query refactors do not absorb the receiver.

## D12 — Sequence consolidation after semantic proof

Land differential parity tests and shared execution first; collapse storage twins second; generate the public operation matrix third; decompose the Python facade last.

**Consequence:** order is `4p1.1` → `z9gh.9.1`/`yeq.3` → `hiu`/`s1kr` → `1fp`. This prevents a structural refactor from preserving or hiding divergent semantics.
