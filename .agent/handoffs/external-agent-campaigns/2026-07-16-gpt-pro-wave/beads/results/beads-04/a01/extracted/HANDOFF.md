# beads-04 MCP context and assertion algebra migration — handoff

## Mission and delivered result

This package is an apply-ready implementation draft for `beads-04`. It migrates the current MCP context, assertion, personal-state/recall, correction, judgment, and maintenance families onto one executable declaration and role-scoped registration seam while preserving each existing production handler and substrate owner. It also closes the source-authorized interim confirmation gap for ten destructive or rebuild tools, keeps candidate agent material distinct from operator judgment, pins immutable annotation schema/batch provenance, and proves a real candidate-write/review/readback lifecycle through the registered MCP surface and an isolated durable `user.db`.

The patch deliberately does **not** introduce a generic mutation or maintenance executor. The declaration registrar validates and registers exact handlers; it never wraps, invokes, dispatches, authorizes, or audits their effects. `polylogue-t46.9` remains the owner of the future cross-surface `OperationExecutor`, preview-token, stale-target, and durable receipt authority.

## Snapshot identity and patch base

| Field | Value |
|---|---|
| Project | Polylogue |
| Snapshot source | `/realm/project/polylogue` as recorded by the supplied project-state manifest |
| Snapshot generated | `2026-07-17T043202Z` |
| Branch | `master` |
| Base commit | `f654480cadb7cc4c194704e24dfd483199547b35` |
| Commit subject | `chore(beads): file archive-insight benchmark findings wofr/vhjs/r7p6/1wtm` |
| Commit date | `2026-07-17T03:45:52+02:00` |
| Bundled `origin/master` delta | `0 / 0`; branch-delta patch and file list are empty |
| Manifest dirty marker | `true` |
| Recoverable tracked dirty patch | None. The exported tracked working tree overlaid on the bundled commit with an empty Git diff. The manifest does not identify the ignored/untracked source of `dirty=true`, so this package does not claim or reproduce it. |

`PATCH.diff` is based directly on the named commit. It was checked and applied in a separate detached worktree at that commit, followed by `git diff --check`, an import/discovery smoke, and the new eleven-test declaration/lifecycle suite.

A clean baseline runtime census found 66 read-role, 95 write-role, 97 review-role, and 104 admin-role tools. This contradicts the stale 103-tool count in `polylogue-t46.8`; current source and `tests/infra/mcp.py` both establish 104, and this patch preserves all 104 names.

## Dependency condition

The mission names the accepted `beads-02` algebra/equivalence kernel as a dependency. No `beads-02` result package, landed declaration package, or accepted kernel was present in the supplied archives, all bundled Git refs, current source, or available uploaded-file evidence. The source Bead `polylogue-t46.8.1` nevertheless explicitly authorizes `polylogue/mcp/declarations/models.py` and `registry.py` and defines their required fields.

Accordingly, this patch includes the smallest source-authorized prerequisite kernel needed for a coherent real migration:

- immutable typed declaration records;
- a registry for this disjoint 41-tool family;
- deterministic equivalence rendering;
- role-scoped exact-handler registration with fail-closed completeness checks.

This is the package's largest material limitation. If the accepted `beads-02` kernel exists outside the supplied authority and differs in model shape or registration API, a substantial reconciliation pass will be required rather than a cosmetic repair.

## Implemented mechanism

### Executable declarations

`polylogue/mcp/declarations/` adds an immutable declaration model, a 41-entry registry, deterministic equivalence JSON, and a registration adapter. Every migrated declaration records:

- public name and exact production module;
- semantic family, verb, object kind, required MCP role, and result semantics;
- canonical substrate owner and canonical plan identifier, both pinned to an exact existing production symbol in this provisional kernel;
- canonical minimal invocation;
- discovery and continuation guidance;
- optional resource/prompt alternatives, with no resource URI claimed until a resolver exists and only the already registered `resume_context` prompt referenced;
- compatibility route, workflow coverage, and per-tool telemetry key;
- injection policy, provenance contract, confirmation policy, and lifecycle.

The registry contains 3 context, 3 assertion, 23 personal-state, 3 correction, 2 judgment, and 7 maintenance tools. Required-role counts are 3 read, 29 write, 2 review, and 7 admin.

### Registration and discovery ownership

`MCPDeclarationRegistrar` replaces the four family-level role gates for the migrated families. Each registration module is now always composed, but each exact handler is registered only when `role_allows(current_role, declaration.required_role)` succeeds. Startup fails if a declaration lacks a handler, a handler lacks a declaration, a handler moves modules, an async handler becomes sync, a minimal invocation drifts from its signature, or a declared confirmation parameter loses its fail-closed default.

The registrar returns the original function unchanged. FastMCP therefore sees the exact existing handler, public name, signature, documentation, error envelope, telemetry callback, and substrate call path. There is no second execution authority.

`tests/infra/mcp.py` now owns only the 63 unmigrated literals and unions the 41 declaration names. This removes duplicate discovery-name authority for the migrated family while retaining the exact 104-tool admin inventory.

### Trust, injection, and provenance contracts

The declarations and real-route test pin these boundaries:

- `capture_assertion_candidate`, `import_annotation_batch`, and `blackboard_post` are write-role candidate material and cannot self-promote or inject.
- Candidate capture preserves `author_ref`, `author_kind=agent`, evidence refs, and scope refs.
- Annotation import preserves immutable `batch_id`, `schema_id`, `schema_version`, `source_result_ref`, `actor_ref`, `model_ref`, and `prompt_ref`; rows remain ordinary candidate assertions while batches remain independent provenance containers.
- `judge_assertion_candidate` and `judge_assertion_candidates` remain review-role operator-judgment paths and preserve the reviewer actor.
- Context delivery/image reads remain non-injecting receipts or bounded context; the preamble declaration records that active guidance is operator-judgment-derived rather than candidate-authored authority.
- No resource URI is claimed because the current server has no matching resolver. The only prompt alternative is the already registered read-only `resume_context` prompt for `compose_context_preamble`; this patch creates no prompt/resource path that can acquire mutation authority.

### Interim destructive and maintenance authorization

The ten tools named by `polylogue-jn40` now fail closed before session resolution, facade access, planner construction, or mutation unless their distinct handler receives the required confirmation:

- `remove_tag`
- `remove_mark`
- `delete_annotation`
- `delete_saved_view`
- `delete_recall_pack`
- `delete_workspace`
- `delete_metadata`
- `maintenance_execute`
- `rebuild_index`
- `rebuild_session_insights`

The existing `delete_session` gate is retained and its error now carries the same typed `confirmation_required` code. `maintenance_execute(dry_run=True)` remains usable without confirmation and calls the real planner with `dry_run=True`; non-dry execution requires `confirm=True`.

These are intentionally separate checks inside the existing handlers. They are an interim compatibility contract, not the preview-bound authorization proof required by `polylogue-t46.9`.

### Real user-tier lifecycle proof

The new production-route test creates an isolated archive session and durable `user.db`, then performs:

1. write-role MCP `capture_assertion_candidate` by an agent;
2. read-role MCP `list_assertion_candidates`, proving candidate/non-injected readback;
3. review-role MCP `judge_assertion_candidate` with an operator actor and `inject=True`;
4. read-role MCP `list_assertion_claims`, proving the accepted active claim and injection policy.

The test uses registered FastMCP functions and the real `Polylogue` facade/storage path. Only the server composition root is patched to point at the isolated real facade; candidate, judgment, persistence, and readback operations are not mocked.

## Changed files

### New production and generated files

- `polylogue/mcp/declarations/__init__.py`
- `polylogue/mcp/declarations/models.py`
- `polylogue/mcp/declarations/registry.py`
- `polylogue/mcp/declarations/registration.py`
- `docs/generated/mcp-context-assertion-equivalence.json`

### Modified production composition and handlers

- `polylogue/mcp/server_tools.py`
- `polylogue/mcp/server_context_tools.py`
- `polylogue/mcp/server_mutation_tools.py`
- `polylogue/mcp/server_personal_state_tools.py`
- `polylogue/mcp/server_maintenance_tools.py`

### Modified discovery, generated topology, and tests

- `tests/infra/mcp.py`
- `tests/unit/mcp/test_context_assertion_declarations.py`
- `tests/unit/mcp/test_per_tool_contracts.py`
- `tests/unit/mcp/test_tag_idempotency.py`
- `tests/unit/mcp/test_tool_contracts.py`
- `tests/unit/mcp/test_user_state_tools.py`
- `tests/unit/maintenance/test_envelope_contracts.py`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

No existing test or helper was deleted. No complete replacement files are required; `PATCH.diff` is unambiguous, so this package intentionally omits `FILES/`.

## Acceptance matrix

| Mission requirement | Result | Evidence |
|---|---|---|
| Migrate context/recall/assertion/correction/maintenance families onto the algebra | Strong coherent subset | 41 executable declarations drive role-scoped registration and discovery inventory. Missing accepted dependency makes the kernel provisional. |
| Preserve materially different write, trust, injection, authorization, and audit contracts | Met for migrated handlers | Exact handlers remain substrate adapters; declaration validation, role discovery tests, existing telemetry path, and lifecycle test cover the distinctions. |
| Keep agent candidates distinct from operator judgment | Met | Candidate-only declarations plus real write/read/review/readback durable test. |
| Preserve immutable annotation schema/batch provenance | Met | Registry validation and tests pin all durable schema/batch provenance fields; importer/storage owner is unchanged. |
| Delete superseded glue and update discovery/contracts | Partially met by safe deletion | Four family role gates and 41 duplicate expected-name literals are removed. Public compatibility tools are retained because deletion telemetry, cold-model equivalence, and the `t46.8.2` prerequisite are absent. |
| Exercise real user-tier writes and readback through MCP | Met | Isolated real `user.db` candidate lifecycle test passes. |
| Do not collapse destructive/maintenance authorization into a generic handler | Met | Eleven independently implemented confirmation gates are tested before backend access; registrar never executes handlers. |
| Preserve maintenance dry-run/apply distinction | Met | Dry-run reaches real planner without confirm; non-dry apply fails closed without confirm. |
| Update generated discovery/equivalence evidence | Met for this family | Deterministic 41-entry JSON artifact, inventory union, topology regeneration, drift test. |
| Remove old tools after equivalence/telemetry | Not attempted by design | Source Beads require telemetry/cold-model proof first; compatibility routes remain explicitly marked `retained_compatibility`. |
| Deliver target 10–15 default read algebra | Outside this disjoint slice | Owned by unfinished `t46.8.2`; this patch supplies only a provisional `t46.8.1`-shaped declaration prerequisite, and current role inventories remain unchanged. |
| Preview-token/receipt authority for destructive actions | Not implemented | Explicitly owned by `polylogue-t46.9`; boolean confirmation is only the authorized interim mitigation. |

## Apply order

1. Start from a clean worktree at `f654480cadb7cc4c194704e24dfd483199547b35`.
2. Run `git apply --check PATCH.diff`.
3. Apply with `git apply PATCH.diff`.
4. Run `git diff --check`.
5. Run the declaration/lifecycle test first: `pytest -q tests/unit/mcp/test_context_assertion_declarations.py`.
6. Run the focused contract partitions listed in `TESTS.md`.
7. Run strict lint/type checks and generated-surface checks listed in `TESTS.md`.
8. Before merge, reconcile this provisional declaration model with any accepted `beads-02` package unavailable here. If that kernel is different, preserve the 41 declaration facts and tests while adapting the model/registrar API.

## Verification performed

The exact command ledger and anti-vacuity mutations are in `TESTS.md`. The final verified highlights are:

- patch applied cleanly in an independent detached base worktree;
- applied-tree `git diff --check` passed;
- applied-tree import/discovery smoke reported 41 declarations and 104 exact admin tools;
- applied-tree new declaration/lifecycle suite: 11 passed;
- every one of the 713 tests under `tests/unit/mcp` passed when partitioned by file into isolated pytest processes;
- focused maintenance envelope/scope suite: 49 passed;
- changed-file Ruff format and lint: passed;
- strict MyPy over all nine changed production modules: passed;
- generated equivalence artifact byte-drift test: passed;
- `render all --check`, topology projection/status checks, layering check, topology check, and test-infra currency check: passed.

One all-in-one `pytest -q tests/unit/mcp` process and one xdist aggregate stalled after partial progress despite every constituent file passing in partitions. No assertion failure was emitted. This is recorded as an unresolved suite-order/resource interaction, not converted into a pass. A full `devtools verify --quick` rerun with the virtual-environment tools on `PATH` passed Ruff format and lint, then did not complete the repository-wide MyPy phase in the available container execution. Targeted strict MyPy and all generated checks passed independently.

## Risks, limitations, and continuation value

### Important limitations

- The accepted `beads-02` dependency was not supplied or landed. This package's declaration kernel may need structural reconciliation.
- The public MCP input schemas of the ten newly guarded tools now include optional `confirm`; clients executing those effects must send `confirm=true`. This is an intentional safety-compatible contract change.
- Boolean confirmation is unbound to actor, archive identity, target digest, expiry, or spec version. It does not close the structural authorization program in `polylogue-t46.9`.
- `clear_corrections` and `update_index` retain their current source contracts because `polylogue-jn40` does not name them. They should be classified by the future operation inventory rather than silently swept into this interim patch.
- No old tool, alias, prompt, or resource was retired. There was no supplied shadow telemetry, observed-use report, cold-model route proof, or completed read-algebra dependency to justify deletion.
- The declaration schema retains future resource/prompt-alternative fields, but this patch claims no resource URI and only points at the already registered `resume_context` prompt. No new resolver, prompt recipe, subscription, or notification route is added.
- No operator live daemon, browser, real archive, secrets, NixOS deployment, or current worktree was available. Live protocol handshake, deployed role credentials, and real-archive migration are unverified.
- The snapshot's manifest says dirty, but the exported tracked tree and branch delta provide no dirty patch to preserve. Unknown ignored/untracked local state is not represented.

### Additional iteration value

A **small repair pass** could improve wording, declaration examples, or add a dedicated render command for the equivalence JSON, but would add little behavioral value while the dependency remains unknown.

A **substantial second pass** is valuable only with one or more missing authorities: the accepted `beads-02` package, completed `t46.8.2` read migration, real MCP usage telemetry/cold-model trials, or the `t46.9` operation executor. With those inputs, the next pass could reconcile the provisional kernel, generate resource/prompt surfaces, retire proven duplicate tools, and replace boolean confirmation with preview-bound authorization and durable receipts. Without them, further broad changes would mostly speculate beyond current source authority.
