# Beads 03 evidence and contradiction record

## Authority order used

1. Supplied current project-state archive and reconstructed source at `f654480cadb7cc4c194704e24dfd483199547b35`.
2. Repository instructions in `CLAUDE.md`.
3. Complete relevant current Beads records and later notes.
4. Current tests and generated witnesses.
5. Git history for intent and previous migration patterns.
6. Test Suite Diet architecture where it constrains proof shape without overriding current source.

## Snapshot findings

The snapshot manifest records:

```text
generated_at: 2026-07-17T043202Z
source: /realm/project/polylogue
branch: master
commit: f654480cadb7cc4c194704e24dfd483199547b35
dirty: true
```

The same archive records `origin/master` and merge base at that commit. Its branch patch, changed-file list, and branch-only commit log are all empty. The reconstructed tracked tree matched the commit before implementation. The only supportable identity is therefore: named clean tracked commit plus an unexplained dirty marker, not a recoverable dirty patch.

No live operator worktree, daemon, browser, archive, secrets, or deployment was available or claimed.

## Missing prerequisite

The mission names the accepted `beads-02` algebra/equivalence kernel as a dependency. Searches covered:

- `/mnt/data` supplied files;
- extracted project snapshot and all refs;
- source symbols for `DeclarationSpec`/declaration packages;
- available File Library entries.

Only the `beads-02` mission prompt was available. No accepted result ZIP, patch, files, or merged implementation was present. This falsifies any claim that the patch integrates the exact accepted kernel. The implementation is a current-source-compatible reconstruction and is marked as such in `HANDOFF.md`.

## Source findings

### Registration and schema authority before the patch

- `polylogue/mcp/server_tools.py` mixed direct `@mcp.tool()` decorators, explicit `mcp.tool()` calls, a private `_MCPReadToolSpec` table, and a source-freshness helper registration.
- `polylogue/mcp/server_insight_tools.py` used `InsightListToolSpec` for dynamic list-tool signatures but directly decorated the remaining special tools.
- Role gates were distributed through the registration functions and `server_support.role_allows`.
- FastMCP inspected the original Python callable/signature to produce schemas. Replacing handlers with a generic forwarding wrapper would therefore be a schema regression unless every signature were manually recreated.
- `tests/infra/mcp.py` contained a second hand-maintained 104-name set.
- The committed `mcp-tool-schemas.json` witness contained only 76 rows and was stale relative to the actual 104-tool admin surface.

### Lower-layer behavior authority

- Query/list/search handlers already delegate through `MCPSessionQueryRequest`, archive/query support, and ArchiveStore rather than a declaration-local evaluator.
- The recent `fd7b35492` execution-control change gives `query_units` shared cancellation/deadline/admission infrastructure. `polylogue-z9gh.9.1` remains open for the complete sole query transaction across all reads.
- Insight tools increasingly delegate through `polylogue.insights` and `Polylogue`, especially after `881392211`; MCP should remain a leaf adapter.
- `server_support` owns safe-call/error and response-budget behavior. This migration must not add a second error/continuation contract.

### Current live surface

Direct FastMCP discovery at the base commit returned:

- read: 66 tools;
- write: 95 tools;
- review: 97 tools;
- admin: 104 tools.

The patched surface is ordered-identical for all roles. All 104 declaration descriptions match the live FastMCP descriptions; 33 are legitimately empty because their current handlers expose no description.

## Beads findings

### `polylogue-t46.8`

The epic identifies the 103/104-tool choice architecture as the defect and requires one declaration algebra, semantic result classifications, role-scoped discovery, no adapter-local query engine, and evidence before removal.

### `polylogue-t46.8.1`

The prerequisite explicitly names `polylogue/mcp/declarations/models.py` and `registry.py`, an exact one-row-per-tool inventory, semantic result/paging/ref fields, generated expected inventories/contracts, and no legacy deletion in that slice. Its complete target also includes resources, prompts, observed-use/workflow/telemetry/deprecation fields and a 10–15-tool target surface. Those broader accepted-kernel capabilities are not inferable from the missing result package and are not claimed here.

### `polylogue-t46.8.2`

The read migration must remain an adapter to the shared query transaction, preserve selection/order/totals/coverage/continuation/refs/errors, and prove production-route equivalence before retiring aliases. The supplied mission narrows this delivery further by explicitly requiring exact discovery-name preservation.

### `polylogue-t46.8.2.1`

This later child proposes removing `archive_list_sessions` and `archive_search_sessions` as zero-use duplicates. That conflicts with the present mission's exact-name preservation. The patch follows the present mission and keeps them; the child remains future work.

### `polylogue-z9gh.9.1`

The shared bounded query transaction is unfinished and owns canonical selection, physical paging, resumability, cancellation, resource bounds, and deletion of surface-local owners. A declaration registration migration cannot honestly claim to complete it.

### `polylogue-s1kr`

The generated semantic-operation CLI/MCP/Python parity matrix is still an open task. Current source contains only a static legacy `docs/plans/api-parity.yaml`; there is no operation-level renderer/artifact to update. Inventing that program here would be a new authority, not a migration update.

## History findings

- `fd7b35492` establishes the recent shared execution-control seam and explicitly leaves remaining direct MCP reads to `z9gh.9.1`.
- `113d1af97` defines current MCP response-budget, replay, pagination, excerpts, and diagnostics behavior; registration changes must preserve it.
- `56eaa2245` shows the repository's established alias-removal pattern: usage evidence, exact inventory/schema/contract updates, then deletion. It supports preserving aliases here until the named proof exists.
- `881392211` moved analysis math from MCP into insights and the facade, supporting a thin registration adapter rather than a new MCP abstraction for execution.
- `6b416f091` documents the discovery-cost and route-choice problem but also keeps recipe expansion separate from query machinery.

## Test Suite Diet findings

`architecture/06-query-cancellation-and-bounds.md` requires one immutable query execution context and owned read lifecycle. Surface adapters must not add their own parser, timeout, background task, semantic cap, or full-result buffer. Its migration sequence preserves aliases until equivalence proof. The no-wrapper registration adapter conforms to these constraints by changing only public registration authority.

## Contradictions adjudicated

| Contradiction | Evidence | Decision |
|---|---|---|
| Snapshot says dirty, but branch delta is empty | Manifest `dirty: true`; delta patch/files/log all zero; tracked reconstruction matched commit | Name the commit and dirty ambiguity; do not invent a patch. |
| Mission depends on accepted `beads-02`, but package/implementation is absent | `/mnt/data`, snapshot, all refs, and File Library contain no accepted result | Reconstruct the smallest compatible kernel; mark exact dependency compatibility unverified. |
| Future Bead removes two aliases, current mission preserves exact names | `t46.8.2.1` versus mission text | Preserve both aliases in this patch. |
| Long-term epic wants 10–15 default tools, current mission requires exact discovery names | `t46.8`/`.1` versus delivery prompt | Migrate authority without reducing the surface; no false epic closure claim. |
| Mission mentions generated Python parity matrix, current source has none | `s1kr` open; only static legacy YAML, no renderer | Do not invent a parallel matrix. Record as incomplete pending the owning artifact. |
| Existing schema witness says 76 tools, live admin server says 104 | Generated fixture versus actual FastMCP discovery | Regenerate from live server and add a full wire-contract witness. |
| Runtime expected inventory projected from production can be vacuous | `tests/infra/mcp.py` now consumes declarations | Add a fixed committed 104-row declaration witness and compare production to it. |

## Falsification evidence and residual uncertainty

Evidence that would invalidate or materially change this patch:

1. The accepted `beads-02` ZIP defines an incompatible declaration schema/registration generator or already contains a merged implementation.
2. A full CI/Nix run finds import-layer, generated-surface, or platform-specific failures not exercised here.
3. The project treats tool listing order as a separately versioned contract beyond the direct base/current equality already observed.
4. The canonical Python parity renderer exists outside this snapshot and requires declaration rows to bind stable semantic operation IDs.
5. Incident-scale or cancellation tests show that moving registration timing changes lifecycle behavior despite exact handler and wire parity.

No evidence found in the supplied authority supports deleting existing aliases, changing query/pagination semantics, or adding a second execution framework in this patch.
