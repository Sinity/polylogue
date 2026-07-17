# Beads 03 handoff — MCP read-surface algebra migration

## Mission and result

This patch migrates the existing MCP query/read/search/insight/topology/timeline/status registration authority onto one typed declaration inventory while preserving the live public surface. It does not create a second query engine: every handler remains the original callable and continues through the existing ArchiveStore, Polylogue facade, insight registry, query contracts, response-budget, and safe-call paths.

The patch declares all 104 live tools exactly once, enables declaration-driven registration for the 56 read-family tools owned by this mission, removes their superseded registration glue, generates fixed semantic and live-wire witnesses, and proves representative search, ObjectRef, topology, and timeline behavior through the actual FastMCP registration and production operations route.

## Snapshot identity

- Supplied project-state archive SHA-256: `3d0c5a2e5024632c05438dbc3a08aae1dcd31529472b1185ad5b80c0cd933e30`.
- Manifest generation time: `2026-07-17T043202Z`.
- Manifest source: `/realm/project/polylogue`.
- Branch: `master`.
- Commit and merge base: `f654480cadb7cc4c194704e24dfd483199547b35`.
- Remote default ref in the bundle: `origin/master`, at the same commit.
- The manifest says `dirty: true`, but `polylogue-branch-delta.patch`, `polylogue-branch-delta-files.txt`, and `polylogue-branch-delta-log.txt` are all zero bytes. Before implementation, the reconstructed tracked working tree was byte-equivalent to the named commit. The archive therefore proves a dirty marker but supplies no captured tracked patch; ignored/local runtime state is unavailable.
- Supplied Test Suite Diet archive SHA-256: `d4fa7fc31c70ff30db076e526836ffc81b8124d3f54c7add5e58d3f57c9dcbff`.
- Mission prompt SHA-256: `410f7d2493a956c44eee713367deb04a3ffa9d4bcde27b00d6a9551cfba67d09`.

## Dependency status

The accepted `beads-02` result package was not supplied, was not present under `/mnt/data`, was not present in the snapshot, and was not found in the available File Library. Current source contains no `DeclarationSpec` implementation or accepted declaration package to import. The only available `beads-02` material was its mission prompt.

Accordingly, this patch reconstructs the smallest current-source-compatible declaration/registration kernel needed for this migration. It deliberately does not claim byte/API compatibility with an unavailable accepted `beads-02` package. Reconciliation against that package is the highest-priority remaining integration check.

## Evidence inspected

Repository guidance and architecture:

- `CLAUDE.md`, especially substrate-first layering, MCP leaf-adapter rules, tool-contract expectations, and generated topology requirements.
- Test Suite Diet `architecture/06-query-cancellation-and-bounds.md`, plus the query/rewrite/status area notes and execution/proof maps. The relevant decision is one shared query lifecycle, no adapter-local parser/timeout/buffering authority, and alias preservation until equivalence proof.

Production route and dependencies:

- `polylogue/mcp/server.py`, `server_tools.py`, `server_insight_tools.py`, `server_support.py`, `archive_support.py`, `query_contracts.py`, `insight_tool_contracts.py`, and the context/mutation/personal-state/maintenance registration families.
- `polylogue/api/`, the archive/query execution and source-freshness surfaces, `polylogue/insights/`, and `polylogue/storage/sqlite/archive_tiers/archive.py` where they own actual behavior.
- FastMCP's installed registration/listing implementation, including explicit `name`/`description` support and handler-signature inspection.

Tests and generated authorities:

- `tests/infra/mcp.py`.
- `tests/unit/mcp/test_tool_contracts.py`, `test_tool_discovery.py`, `test_envelope_contracts.py`, and `test_server_surfaces.py`.
- Existing `tests/data/witnesses/mcp-tool-schemas.json` and generated-surface/devtools tests.
- `docs/mcp-reference.md`, `docs/devtools.md`, topology projections, and docs-coverage verification.

Beads:

- `polylogue-t46.8`, `polylogue-t46.8.1`, `polylogue-t46.8.2`, `polylogue-t46.8.2.1`, `polylogue-z9gh.9.1`, and `polylogue-s1kr`, including later notes.

History:

- `fd7b35492` — shared interruptible/admission-controlled query execution.
- `113d1af97` — MCP response-budget and pagination/diagnostic behavior.
- `56eaa2245` — prior evidence-gated duplicate MCP alias removal.
- `881392211` — insight math moved beneath MCP into insights/API owners.
- `6b416f091` — discovery/cookbook history and the large-surface usability problem.

## Mechanism

### Typed declaration algebra

`polylogue/mcp/declarations/models.py` introduces frozen typed declarations for:

- semantic family and public verb;
- object kind and role gate;
- result semantics: exhaustive page, point read, top-k, aggregate, bounded context, recursive graph, status, mutation, or maintenance;
- continuation kind;
- implementation and lower-layer operation owner;
- output contract, minimal invocation, ObjectRef/query/absence semantics, compatibility route, migration owner, and declaration-registration state.

`polylogue/mcp/declarations/registry.py` is the single executable inventory. It contains all 104 live tools exactly once and marks 56 tools as migrated here:

- query: 7;
- read: 10;
- insight: 18;
- topology: 7;
- timeline: 6;
- status: 8.

The remaining 48 context/assertion/coordination/personal-state/mutation/review/maintenance tools are inventoried but remain on their current registration paths for the separately owned migration.

### No-wrapper FastMCP adapter

`polylogue/mcp/declarations/registration.py` looks up the declaration, applies the existing `role_allows` policy, sets the exact public name/description, tags the original callable for anti-bypass testing, and passes that callable directly to `FastMCP.tool(...)`.

There is no `*args/**kwargs` wrapper and no new parser, request model, timeout, cursor, result buffer, or error mapper. FastMCP still inspects the original handler annotations and any explicit `__signature__`; the existing operations remain behavior authority.

### Production migration and dominated glue removal

- `server_tools.py` now registers the 13 query-family and 15 read/status tools from ordered declaration mappings.
- The superseded `_MCPReadToolSpec`, `_MCP_READ_TOOL_SPECS`, `_register_mcp_read_tool`, and source-freshness registration helper call are removed from this route.
- `server_insight_tools.py` registers all 28 mission-owned insight/topology/timeline tools through the same adapter. Existing `InsightListToolSpec` remains the dynamic signature/schema owner; the declaration owns public inventory and semantic classification.
- Context/assertion/coordination and privileged families retain their current decorators and role-specific implementations.

### Generated parity witnesses

`devtools render mcp-contract-fixtures` now renders and drift-checks:

- `mcp-tool-declarations.json`: 104 fixed semantic inventory rows;
- `mcp-tool-schemas.json`: refreshed from the actual 104-tool FastMCP surface (the supplied witness had only 76 rows and stale shapes);
- `mcp-tool-contracts.json`: 104 exact live rows containing name, role, description, input schema, output schema, and annotations.

The production declaration inventory now supplies `tests/infra/mcp.py`, docs coverage, the generated MCP tool index, and affordance-usage inventory. Fixed committed witnesses remain independent anti-vacuity oracles.

## Decisions

1. **Preserve every live name.** The mission explicitly requires exact discovery names. Although later child `polylogue-t46.8.2.1` proposes removing `archive_list_sessions` and `archive_search_sessions`, this patch keeps both aliases and their independent behavior.
2. **Migrate registration authority, not read execution semantics.** `polylogue-z9gh.9.1` still owns the unfinished sole bounded query transaction. This patch neither duplicates nor pretends to complete it.
3. **Keep FastMCP as wire-schema authority.** The declaration adapter passes original callables so dynamic signatures and Pydantic constraints cannot drift through a generic wrapper.
4. **Use a fixed witness to prevent production-derived vacuity.** Runtime inventories project from production, while committed declaration and wire fixtures fail if a row, role, description, or schema changes without an intentional regeneration.
5. **Do not delete existing tests/helpers.** Existing tests were retained and updated only where their old private registration-spec authority was superseded.
6. **Do not invent the missing Python operation-parity program.** `polylogue-s1kr` remains open; the snapshot has a static `docs/plans/api-parity.yaml` but no generated operation-level CLI/MCP/Python matrix or renderer. Creating that unrelated program here would violate current authority. This is recorded as incomplete rather than disguised.

## Changed files

Production declaration and migration:

- `polylogue/mcp/declarations/__init__.py`
- `polylogue/mcp/declarations/models.py`
- `polylogue/mcp/declarations/registration.py`
- `polylogue/mcp/declarations/registry.py`
- `polylogue/mcp/server_tools.py`
- `polylogue/mcp/server_insight_tools.py`

Generated-surface and production inventory consumers:

- `devtools/render_mcp_contract_fixtures.py`
- `devtools/generated_surfaces.py`
- `devtools/command_catalog.py`
- `devtools/render_mcp_tool_index.py`
- `devtools/verify_docs_coverage.py`
- `devtools/affordance_usage.py`
- `docs/devtools.md`
- `docs/mcp-reference.md`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

Witnesses and tests:

- `tests/data/witnesses/mcp-tool-declarations.json`
- `tests/data/witnesses/mcp-tool-contracts.json`
- `tests/data/witnesses/mcp-tool-schemas.json`
- `tests/infra/mcp.py`
- `tests/unit/mcp/test_read_tool_declarations.py`
- `tests/unit/mcp/test_tool_contracts.py`
- `tests/unit/mcp/test_tool_discovery.py`
- `tests/unit/mcp/test_envelope_contracts.py`

No complete replacement files are needed; `PATCH.diff` fully disambiguates the change.

## Acceptance matrix

| Requirement | Status | Evidence |
|---|---|---|
| Apply-ready against named snapshot | Pass | Clean detached worktree at `f654480…`; `git apply --check --binary`; exact staged replay byte-equal to `PATCH.diff`. |
| Every live tool inventoried exactly once | Pass | 104 unique declaration rows; fixed fixture and actual admin discovery both contain 104 names. |
| Mission-owned families migrated | Pass | 56 handlers carry the declaration marker and register only through the shared adapter. |
| Exact names and discovery order | Pass | Direct base-commit/current FastMCP comparison is ordered-equal for read/write/review/admin. |
| Exact role gates | Pass | Discovery counts and membership unchanged: 66 / 95 / 97 / 104. |
| Exact descriptions, input/output schemas, annotations | Pass | Pre-change/current comparison equal by name for all roles; full committed wire witness matches live admin surface. |
| Query behavior | Pass for representative route | Seeded `search` traverses actual registered handler, query/archive support, ArchiveStore/FTS, and returns the planted parent only. |
| ObjectRef semantics and absence | Pass for representative route | Registered `resolve_ref` uses real Polylogue/archive resolution, returns canonical `session:<id>`, and preserves typed unresolved payload/caveat. |
| Topology behavior | Pass for representative route | Registered `get_session_topology` returns planted parent/root/thread through the real facade/archive path. |
| Timeline/insight behavior | Pass for representative route | Registered generic `usage_timeline` traverses insight registry and real Polylogue operation. |
| Pagination and error/absence contracts | Pass for focused existing routes | Search cursor/total, message offsets, invalid offset, not-found summary, and envelope suites pass unchanged. |
| Superseded per-tool registration glue removed | Pass for migrated slice | Old read spec/helper and direct decorators/calls are removed; declaration marker test fails on bypass. |
| MCP fixtures/generated docs updated | Pass | Renderer checks, generated-surface tests, tool index, devtools reference, topology, and docs coverage pass. |
| Generated Python semantic-operation parity matrix | Not available in snapshot | `polylogue-s1kr` is open; no renderer/artifact exists to update. No parallel substitute invented. |
| Exact compatibility with accepted `beads-02` package | Unverified | Package/API absent from all supplied authority. |
| Full repository/Nix/live daemon/browser/incident-scale verification | Unverified | Not available or not run in this container. |

## Apply order

1. Check out `f654480cadb7cc4c194704e24dfd483199547b35` with no local changes.
2. Run `git apply --check --binary PATCH.diff`.
3. Run `git apply --binary PATCH.diff`.
4. Sync the repository's normal development dependencies.
5. Run the generated-surface checks listed in `TESTS.md`.
6. Run the focused MCP/devtools lanes listed in `TESTS.md`, then the repository's normal quick/full gates.
7. Before merge, reconcile `polylogue/mcp/declarations/` against the actual accepted `beads-02` package if that package exists outside the supplied authority.

## Risks and remaining work

The principal risk is dependency mismatch: an actual accepted `beads-02` package may define different types, fields, generated artifacts, or registration APIs. If its conceptual shape matches this patch, reconciliation should be a small repair—rename/import adaptation plus fixture regeneration. If it contains the full t46.8.1 resource/prompt/equivalence/manual model or an already-generated Python operation matrix, integrating it would be a substantial second pass and could replace much of this reconstructed kernel.

The patch intentionally does not finish `polylogue-z9gh.9.1`, collapse the default read surface to 10–15 verbs, remove competing aliases, add URI resources/prompts, run cold-model trials, or prove cancellation/RSS/temp cleanup at incident scale. Those are separate architecture and execution contracts. Treating this registration migration as their completion would be unsafe.

The focused suites and clean-apply proof are strong for this patch's registration and representative runtime behavior. Another ordinary iteration without the missing `beads-02` artifact or full CI evidence is likely to add only small repair value. A second pass with that artifact, the canonical Python parity renderer, or failures from the full Nix/CI gate could add substantial value.
