# Source, Beads, test, and history evidence

## Authority order used

1. Supplied mission `04-mcp-context-migration.md`.
2. Supplied Polylogue project-state snapshot and exact current source at `f654480cadb7cc4c194704e24dfd483199547b35`.
3. Current repository guidance (`AGENTS.md` symlinked to `CLAUDE.md`, plus `TESTING.md`).
4. Current `.beads/issues.jsonl`, including later notes that supersede older design wording.
5. Bundled Git history and all refs.
6. The supplied “slightly stale” Testsuite Diet as adversarial architecture/test-design evidence, never as code authority.

No web source was needed. No live operator system was available.

## Snapshot evidence

The supplied `polylogue-manifest.json` records:

- project `polylogue`;
- source `/realm/project/polylogue`;
- generated at `2026-07-17T043202Z`;
- branch `master`;
- commit `f654480cadb7cc4c194704e24dfd483199547b35`;
- `dirty=true`.

The bundled Git repository establishes:

- `HEAD` and `origin/master` are the same commit;
- branch divergence is `0 0`;
- the bundled branch-delta file list, log, and patch are empty;
- commit subject is `chore(beads): file archive-insight benchmark findings wofr/vhjs/r7p6/1wtm`;
- commit timestamp is `2026-07-17T03:45:52+02:00`.

The working-tree tar was overlaid onto a clone of the all-refs bundle before implementation. The tracked overlay produced no Git diff. Therefore the manifest's dirty state likely came from ignored or untracked local state that the supplied metadata does not identify as an apply-ready patch. This package names the exact commit as the patch base and does not fabricate a dirty-patch identity.

A separate clean worktree at the base commit proved that the live baseline contains 104 admin tools and that `tests/infra/mcp.py` expects exactly those 104. Role counts at the base are:

- read: 66;
- write: 95;
- review: 97;
- admin: 104.

## Repository instructions inspected

`CLAUDE.md` / `AGENTS.md` established the implementation discipline used here:

- substrate/domain owners define semantics; MCP stays a leaf adapter;
- existing typed interfaces are preferred over parallel frameworks;
- added Python modules require topology projection/status regeneration;
- `devtools verify --quick` consists of format, lint, MyPy, and generated-surface checks;
- tests must exercise production dependencies and generated surfaces must remain synchronized.

`TESTING.md` was inspected for pytest/devtools conventions, isolated scratch behavior, and focused-versus-broad verification expectations.

## Production source inspected

### MCP composition, roles, and discovery

- `polylogue/mcp/server.py`
- `polylogue/mcp/server_support.py`
- `polylogue/mcp/server_tools.py`
- `polylogue/mcp/server_context_tools.py`
- `polylogue/mcp/server_mutation_tools.py`
- `polylogue/mcp/server_personal_state_tools.py`
- `polylogue/mcp/server_maintenance_tools.py`
- `polylogue/mcp/server_insight_tools.py`
- `tests/infra/mcp.py`

Findings:

- roles are monotonic `read < write < review < admin` through `role_allows`;
- existing family-level gates controlled write/review/admin registration;
- exact public names and handlers were already distributed across the modules above;
- the current baseline is 104 tools, not the 103 in older Bead prose;
- call telemetry is emitted by the existing `ServerCallbacks.async_safe_call` route, so retaining exact handlers preserves per-tool call attribution.

### Candidate assertion and judgment ownership

- candidate capture and review handlers in `server_mutation_tools.py`;
- `polylogue/api` assertion facade methods;
- assertion payloads/storage and candidate judgment lifecycle;
- context preamble guidance filtering and candidate safety tests;
- tests for candidate capture, assertion judgment, blackboard, and context preamble.

Findings:

- agent-authored assertion material enters candidate state and is non-injecting;
- reviewer judgment is a distinct review-role route carrying an actor and decision;
- active/injecting state is the result of judgment, not candidate authorship;
- current source already has the typed durable lifecycle needed for a real MCP roundtrip test.

### Annotation schema and batch provenance

- `polylogue/annotations/schema.py`
- `polylogue/annotations/write.py`
- `polylogue/annotations/importer.py`
- durable user-tier annotation schema/batch repository and migrations;
- MCP annotation import handler and its existing tests.

Findings:

- schemas are versioned and immutable by id/version/fingerprint;
- batches are independent provenance containers; assertion rows remain assertions;
- import records schema identity, source result, actor/model/prompt references, counts, and validation outcomes;
- external-agent rows remain candidate/non-injected through the shared assertion write chokepoint.

### Personal state, corrections, and maintenance

- user-state facade methods for tags, marks, annotations, views, recall packs, workspaces, metadata, and corrections;
- maintenance planner, envelope, scope filter, registry, and MCP routes;
- operations declarations/action contracts and the current direct adapter paths.

Findings:

- these families have materially different result/idempotency contracts and should not be collapsed into one generic handler;
- `maintenance_preview` and `maintenance_execute(dry_run=True)` share real scope-filter/planner semantics;
- ten tools named by `polylogue-jn40` lacked the interim confirmation gate;
- `delete_session` already supplied the local fail-closed pattern;
- the existing operation declarations are not yet executable cross-surface authority.

## Beads evidence

### `polylogue-t46.8` — protocol-native MCP algebra

Status: open, P1. It requires one executable semantic inventory, generated role-scoped discovery/contracts, preservation of result semantics and bounded execution, negative authority tests, telemetry before deletion, and a target 10–15 default read transaction surface. It explicitly says MCP must remain a leaf adapter and must not create a parallel parser, query engine, or policy layer.

### `polylogue-t46.8.1` — declaration/equivalence kernel

Status: open, P1. It explicitly names:

- `polylogue/mcp/declarations/models.py`;
- `polylogue/mcp/declarations/registry.py`;
- fields for verb/object, role, result semantics, canonical plan, minimal invocation, discovery, continuation, resource/prompt alternative, compatibility route, workflow coverage, telemetry, and deprecation. In this provisional implementation, canonical plans are exact existing owner symbols; no resource URI is claimed, and the only prompt alternative names the already registered `resume_context` prompt;
- generated registration/contracts/discovery/expected inventory/equivalence evidence;
- no legacy tool deletion in this kernel slice.

This is the direct source authorization for the minimal prerequisite included in the patch.

### `polylogue-t46.8.2` — read migration

Status: open, P1, and a blocker for `t46.8.3`. It owns query/read/get/explain migration, bounded shared query execution, resources, alias retirement, cancellation/cleanup, cold-model route proof, and the default read-surface budget. It has not landed in the supplied authority, so this patch does not pretend to deliver its read collapse or use it as an available dependency.

### `polylogue-t46.8.3` — this family migration

Status: open, P2. It requires thin context/write/judge/run/maintenance adapters over typed owners, resource receipt/status objects, prompts without instruction authority, candidate/judged preservation, actor context, dry-run/authorize/apply/receipt/reconcile lifecycles, idempotency, role isolation, and deletion only after equivalence/authorization/recovery proof.

The patch implements the strongest compatible subset available before `t46.8.2`: declaration-driven exact-handler registration, trust/provenance/role contracts, real candidate lifecycle proof, and interim maintenance/destructive gates. It does not claim final alias deletion, a resource surface, a new prompt recipe, or operation-executor authority.

### `polylogue-jn40` — interim confirmation mitigation

Status: open, P2 bug. It names exactly ten previously unprotected tools and says to apply the same `confirm: bool = False` guard pattern used by `delete_session`. Its 2026-07-16 note is explicit that this is a cheap interim fail-closed mitigation and that `polylogue-t46.9` owns the structural preview-token/receipt solution.

The patch implements all ten named gates and preserves dry-run planning for `maintenance_execute`.

### `polylogue-t46.9` — executable operation authority

Status: open, P1 feature. It requires one `OperationSpec`-to-handler inventory and `OperationExecutor` across CLI/API/MCP/daemon/maintenance, actor/archive/target-digest-bound preview tokens, stale-preview rejection, receipts, and bypass tests. It also says `jn40` booleans are interim compatibility only.

The patch avoids creating a generic mutation or maintenance executor and documents this residual obligation.

### `polylogue-rxdo.7.1` and `polylogue-rxdo.7.2`

Both are closed and establish the current annotation substrate:

- `.7.1` persisted immutable annotation schemas and independent batch provenance;
- `.7.2` landed candidate-only JSONL import through shared production operation plus CLI/MCP, live target/evidence validation, independent batches, typed query, and adjudication.

These records justify preserving rather than reimplementing the current importer/storage path.

## Git history inspected

Relevant commits and what they establish:

- `d66f97f43` — split personal-state MCP registration; current module boundary is intentional.
- `5aa34e6c5` — reviewed candidate judgment flow; candidate versus judgment semantics are current production behavior.
- `73959d471` — context delivery receipts; receipt provenance is a real source owner.
- `246c48d08` — durable annotation schema/batch provenance.
- `f4504cb4d` — provenance-stamped JSONL annotation batch import.
- `4ed0cf2dc` — typed annotation joins to structural targets.
- `54af5477c` and `eff7c2abe` — durable MCP call telemetry and delivery; exact handler telemetry identity matters.
- `113d1af97` — bounded MCP response payloads; declarations must preserve bounded result semantics.
- `56eaa2245` — prior duplicate MCP tool retirement, showing deletions are performed when equivalence is known rather than retained indefinitely.
- `fd7b35492` — interruptible/admission-controlled query execution; read transaction mechanics exist but are owned by the unfinished read migration.
- `c2fd1e902` — excision bypass repair; supports the architecture warning that one direct destructive bypass defeats declaration-only safety.

No bundled Git ref contained the new `polylogue/mcp/declarations/` package or an accepted `beads-02` result.

## Tests inspected

The implementation followed dependencies beyond the new test file. Inspected and executed test families included:

- role/discovery, server surfaces, runtime, error isolation, call telemetry, and contracts;
- candidate capture, assertion judgment, annotation import/join, blackboard, context image/preamble;
- all personal-state and idempotency contracts;
- maintenance envelope and scope parity;
- query/insight/logical-session/tool-timing tests to detect composition regressions;
- generated topology and repository layering/test-infra checks.

Every one of the 713 MCP unit tests passed in isolated file partitions. The complete command ledger and the aggregate-process limitation are in `TESTS.md`.

## Slightly stale Testsuite Diet evidence

The supplied Testsuite Diet was used as adversarial design evidence only.

`architecture/04-destructive-and-authentication-boundaries.md` recommends an executable `OperationExecutor`, distinguishes reversible writes, judgment/replacement, destructive scoped effects, and broad maintenance, and states that a Boolean `confirm=true` is only a compatibility adapter rather than authority proof. This directly supports keeping the current interim guards separate and refusing to invent the future executor in this patch.

`architecture/07-evidence-provenance-and-public-algebra.md` requires surfaces to project one canonical domain fact without strengthening authority, and treats agent-declared versus judged authority as distinct. This supports declaration metadata and the real candidate/judgment lifecycle test, not a generic fact store.

`17-unresolved-architecture-scope.md` says architecture decisions do not make packets executable automatically: current symbols, write sets, historical seeds, and mutation witnesses still must be reconciled. It also ranks destructive/authentication bypass as a high-risk branch and identifies rewrite-native MCP work as still requiring sizing and implementation.

## Contradictions and resolutions

### Accepted `beads-02` dependency is absent

Mission: depends on an accepted kernel. Supplied authority: no result package or landed kernel exists. Resolution: implement only the minimal model/registry/registrar shape explicitly authorized by `t46.8.1`, mark it provisional, and preserve tests/data so it can be adapted rather than discarded.

### Tool count says 103 in Bead prose, 104 in source

Current clean source and expected inventory both produce 104 admin tools. Resolution: current source wins; the patch preserves 104 and migrates 41.

### Mission asks to delete superseded glue; Beads prohibit premature tool deletion

`t46.8.1` says no legacy tool deletion in the kernel slice, and `t46.8.3` requires telemetry/equivalence/authorization/recovery before deletion. Resolution: delete only superseded family role gates and 41 duplicate expected-name literals. Retain public compatibility routes with explicit metadata.

### Interim Boolean confirmation versus final authorization architecture

`jn40` requires Boolean confirmation now; `t46.9` and Testsuite Diet say it is insufficient as structural authority. Resolution: land exact interim gates, test them before backend access, preserve dry-run, and avoid claiming final operation authorization.

### Manifest dirty marker versus empty tracked delta

Manifest says dirty; branch delta and exported tracked overlay are clean. Resolution: name the exact commit as patch base and report the unknown dirty state as unavailable rather than inventing a patch.

### Full aggregate test process versus constituent passes

One-process/aggregate MCP runs stalled; every collected file passed in partitions, 713/713. Resolution: report both facts. Do not convert the aggregate stall into a pass or infer a product failure without evidence.

## Evidence that would falsify this design

The package should be reconsidered if any of the following becomes available:

- an accepted `beads-02` package with incompatible declaration or registration semantics;
- a current Bead note changing roles/injection/provenance for any of the 41 tools;
- source showing a migrated handler has another registration call site not covered by `register_tools`;
- production telemetry proving a compatibility route is safe to delete or must remain;
- a completed `t46.9` executor that makes the interim Boolean gates obsolete;
- a reproduction showing the full-suite aggregate stall is introduced by this patch rather than an environmental/global-test interaction;
- a live deployed MCP contract showing client schema incompatibility that requires a staged compatibility adapter.
