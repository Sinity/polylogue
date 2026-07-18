# HANDOFF — polylogue-1vpm.6.2 observed repository effects

## Mission and delivered slice

This package implements a cohesive provider-neutral reconciliation slice for `polylogue-1vpm.6.2`. It extends the existing work topology/claim graph rather than adding a provider-specific hierarchy. The patch adds independently observed repository effects, candidate/direct work-to-effect associations, assertion-backed evaluations, Beads baseline/change projection, repository-scoped Bead session history, and real SQLite persistence/traversal for the expanded graph.

The central invariant is structural: a provider report remains a `claim`; git/GitHub/Beads/artifact/verification facts are `effect` nodes; evaluated acceptance is an assertion-backed `evaluation` node. Direct identifiers and direct evidence create `observed_effect` edges. File/time overlap creates only `candidate_effect` edges with candidate state. A graph cannot validate if a claim is used as an effect endpoint or if a candidate edge is presented as observed causality.

## Snapshot identity and authority

The supplied project-state archive manifest identifies:

- project: `polylogue`
- generated: `2026-07-18T013442Z`
- branch: `master`
- commit: `bf8191b3f56aa40da8f271df7f3385c712825497`
- working tree: dirty
- supplied archive SHA-256: `47ad17ea5a44d148a3c58baa74da9cff650b51e46ceed38e41468810a0896df4`
- all-refs bundle manifest SHA-256: `2ef6279ca5633633277870745d7c398a58481f282c3f6b6f7441692250f0b546`
- working-tree archive manifest SHA-256: `00e38f085be14a9f825d17886c0239e1ee7109a2d11545f9faef486ecc71d652`

The supplied dirty state changes exactly these unrelated tracked files:

- `polylogue/archive/query/unit_results.py`
- `polylogue/daemon/http.py`
- `polylogue/hooks/__init__.py`

That supplied dirty patch is 2,272 bytes with SHA-256 `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`. It is not copied into this result. For isolation, I committed only those supplied edits in a temporary local baseline `1eaa0433b71178a90b46847135dcc67333e6d600`; `PATCH.diff` is the mission-only delta from that baseline. Because none of the three supplied dirty files are touched by this patch, `PATCH.diff` also applies to clean `bf8191b3...` and to the supplied dirty working tree.

## Evidence inspected

Repository instructions and architecture:

- `CLAUDE.md` / symlinked `AGENTS.md`, especially substrate-first semantics, rebuildable `index.db`, semantic-reparse schema policy, focused-test discipline, synthetic-fixture privacy, and acceptance-criteria honesty.
- `docs/architecture.md`, `docs/internals.md`, and schema lifecycle code where needed to follow the production composition.

Production source and composition:

- `polylogue/core/refs.py`
- `polylogue/insights/work_evidence.py`
- `polylogue/insights/work_reconciliation.py`
- `polylogue/insights/claude_workflow_evidence.py`
- `polylogue/insights/run_projection.py`
- `polylogue/insights/session_commit.py`
- `polylogue/sources/parsers/claude/orchestration.py`
- `polylogue/storage/query_models.py`
- `polylogue/storage/repository/__init__.py`
- `polylogue/storage/repository/insight/work_evidence.py`
- `polylogue/storage/sqlite/query_store.py`
- `polylogue/storage/sqlite/query_store_work_evidence.py`
- `polylogue/storage/sqlite/queries/work_evidence.py`
- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/storage/sqlite/lifecycle.py`

Tests and fixtures:

- existing topology/claim, Claude projection, ObjectRef, repository composition, and index lifecycle tests
- current fixture conventions under `tests/fixtures/`
- new synthetic committed Beads baseline/interaction fixtures and the real repository route test in this patch

Beads and history:

- complete records for `polylogue-1vpm`, `polylogue-1vpm.6.1`, `polylogue-1vpm.6.2`, `polylogue-2qx`, `polylogue-2qx.2`, `polylogue-t8t`, `polylogue-z9gh`, and `polylogue-z9gh.7`
- `.beads/issues.jsonl` and `.beads/interactions.jsonl`
- all-ref git history for the affected source paths
- introducing squash commit `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` (`feat: harden archive continuity closure (#3051)`)

## Implemented mechanism

### Typed graph boundaries

`WorkEvidenceNode` now carries direct source refs, repository and repository-snapshot scope, effect kind/action, Beads subject identity, evaluation status, and JSON-compatible attributes. `WorkEvidenceEdge` carries the same direct-source and scope information. Validation enforces:

- effect type and ObjectRef kind agree;
- effects have direct source refs, an action, repository scope, and repository snapshot;
- Beads changes have a branch/snapshot-scoped Beads issue subject and stable subject key;
- evaluations use assertion refs and retain their assertion source;
- `observed_effect`/`candidate_effect` originate from observed work, not claims/effects/evaluations;
- claim evaluation is `claim -> evaluation -> effect`, never `claim -> effect`;
- repository/corpus snapshots agree across graph nodes, edges, effects, and evaluations;
- candidate edges cannot validate as observed edges.

### Authority-bearing effect adapters

`work_reconciliation.py` now provides admitted-fact adapters for:

- git commits and branch-head observations;
- GitHub issue and PR lifecycle observations, including squash/merge commit identity;
- artifacts and verification receipts;
- explicit unresolved coverage gaps rather than invented missing 2qx.2 content;
- Beads issue baselines and append-only interactions, including created/edited/claimed/closed/reopened/corrected actions, branch-local identities, missing-baseline degradation, and correction/supersession edges;
- existing `SessionCorrelationResult` as a projection whose explicit refs are direct and whose time/file overlap remains candidate-only;
- `ObservedEvent` direct-object refs, with exact effect identity taking precedence over incidental source-ref reuse.

One PR may relate to a merge commit and several Beads issues. Multiple sessions may independently associate to the same effect. Conflicting duplicate identities fail instead of silently overwriting facts.

### Claude artifact degradation

The Claude Workflow adapter no longer borrows coordinator evidence when an admitted sidecar/path has no matching `EvidenceRef`. It retains the admitted file ref, marks the artifact and derived facts unresolved with reduced confidence, and merges identities without allowing a resolved coordinator observation to falsely resolve missing artifact-derived calls/attempts/results. Coordinator events without a provider message id receive a deterministic event identity rather than disappearing.

This does not claim that the missing 2qx.2 incident corpus was present. The current Bead explicitly says that its prior claim was orphaned and that no matching master commit exists; that evidence remains unresolved.

### Persistence and production query route

Index schema v40 persists source refs, repository/snapshot scope, Beads subjects/actions, effect/evaluation fields, and attributes. The delta is declared `SEMANTIC_REPARSE`; no unsafe clone-forward from v39 is offered.

The existing `SessionRepository` composition now exposes:

- bounded multi-hop `traverse_work_evidence(...)`;
- `find_bead_work_sessions(...)`, explicitly scoped by graph, repository, Bead id, optional repository snapshot, action set, and candidate inclusion.

The Bead query reads the real persisted effect and edge rows, derives session identity only from the work-association side (source node plus association edge), returns effect provenance separately, and partitions direct matches from overlap-only candidates. This prevents an effect observer's session from being misattributed to every candidate edge pointing at that effect.

## Decisions

1. Repository snapshots are a new ObjectRef kind distinct from archive corpus `context-snapshot` refs. The former identifies external project state; the latter identifies graph input state.
2. Effects remain ordinary graph nodes so the existing generic traversal and repository composition continue to work. No Workflow-only or Beads-only graph/store was introduced.
3. Evaluation is a node rather than an edge attribute. This makes acceptance authority, rationale, status, source assertion, and supersession independently queryable.
4. Missing provider artifacts are unresolved facts with admitted path refs. They do not inherit unrelated evidence and do not fabricate run/call/attempt membership.
5. Exact direct identity outranks a shared source ref. A commit named by an observed event still maps to the commit effect even when a PR also cites that commit as its squash result.
6. Beads identities include repository, repository snapshot, branch, issue, and event. Two branches at the same commit therefore do not collapse tracker state.
7. `index.db` v40 requires semantic rebuild/reprocess. A derived-tier migration helper was deliberately not added.
8. No existing tests/helpers were deleted. There are no proposed dominated deletions in this revision.

## Changed files

Production:

- `polylogue/core/refs.py`
- `polylogue/insights/claude_workflow_evidence.py`
- `polylogue/insights/work_evidence.py`
- `polylogue/insights/work_reconciliation.py`
- `polylogue/storage/query_models.py`
- `polylogue/storage/repository/insight/work_evidence.py`
- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/storage/sqlite/lifecycle.py`
- `polylogue/storage/sqlite/queries/work_evidence.py`
- `polylogue/storage/sqlite/query_store_work_evidence.py`

Fixtures/tests:

- `tests/fixtures/beads/work-effects-baselines.jsonl`
- `tests/fixtures/beads/work-effects-interactions.jsonl`
- `tests/unit/core/test_refs.py`
- `tests/unit/insights/test_claude_workflow_evidence.py`
- `tests/unit/insights/test_work_effects_route.py`
- `tests/unit/insights/test_work_reconciliation.py`

## Acceptance matrix for polylogue-1vpm.6.2

| AC | Status | Evidence / exact residual |
|---|---|---|
| 1. Same bidirectional graph across work, claims, repository effects, artifacts, receipts | Satisfied for the implemented substrate route | Expanded typed graph, v40 persistence, bounded traversal, prior provider-topology tests, and new end-to-end repository test. No new public MCP/API surface is claimed. |
| 2. Claim, observed effect, evaluated satisfaction distinct | Satisfied | Model/graph validators plus reconciliation tests require three node kinds and `claim -> evaluation -> effect`; direct claim-to-effect association is rejected. |
| 3. Direct Workflow/git/GitHub/Beads/artifact/verification support; overlap candidate-only | Partially satisfied | Generic direct-event and effect adapters, Beads baselines/interactions, artifacts/receipts, and candidate forcing are implemented. The actual missing 2qx.2 incident artifacts remain unresolved; no complete live Workflow corpus is invented. |
| 4. Many-to-many, branch-local state, squash, corrections, contradiction/supersession | Satisfied on synthetic/committed fixtures | Tests cover one PR to commit plus multiple Beads, same-snapshot different-branch identities, later correction, missing baseline, candidate/direct multiplicity, and evaluation states. |
| 5. `wf_54d4fb2e-841` before/after P1 reconciliation | Deferred; still blocking | The snapshot lacks the admitted 2qx.2 artifact family and authorized live archive. No honest incident census or 25-open-P1 before/after proof can be produced from the supplied authority. Exact continuation is under `polylogue-2qx.2` then `polylogue-z9gh.7`. |
| 6. Seeded production query identifies sessions that created/edited/claimed/closed a Bead | Satisfied | `test_work_effects_route.py` persists the graph through `SessionRepository`, then obtains four exact direct sessions, one separate time-overlap candidate, and empty results for wrong repository/snapshot scopes. |
| 7. Existing correlation paths project/retire; mutations reject collapsed invariants | Partially satisfied | `SessionCorrelationResult` now projects to direct/candidate graph associations. No old path was deleted because repository-wide retirement/call-site certification was not available. Mutation tests cover claim=effect, overlap=causality, missing baseline, missing snapshot, wrong endpoint/scope, and ambiguous source refs. |
| 8. Focused adapters/reconciliation/Claude integration/default affected verification | Partially verified | Focused source/model/SQLite route tests pass under the described local compatibility harness. Managed `devtools`, full dependency environment, testmon affected selection, ruff, mypy, and authorized live checks remain unverified. |

## Edge and uncertainty matrix

| Input | Graph representation | State/authority |
|---|---|---|
| exact effect ObjectRef in archived event | `observed_effect` | direct, source authority retained |
| unique admitted source ref naming one effect | `observed_effect` | direct evidence |
| explicit legacy commit ref | `observed_effect` | direct identifier |
| file overlap only | `candidate_effect` | candidate / inferred |
| time overlap only | `candidate_effect` | candidate / inferred |
| claim acceptance judgment | assertion `evaluation` plus `evaluated_as`/`evaluated_against` | operator authority; supported/partial/contradicted/unresolved/superseded |
| missing Beads baseline | unresolved `beads-issue` baseline plus admitted changes | Beads authority, confidence 0 for missing baseline |
| later Beads correction | change effect plus `superseded` relation | source-preserving correction |
| missing Claude sidecar/path evidence | unresolved artifact and artifact-derived facts | provider-admitted path, reduced confidence; no borrowed coordinator evidence |
| unexecuted live verification | verification receipt outcome `unverified` | unresolved, confidence 0 |

## Apply order

1. Start from `bf8191b3f56aa40da8f271df7f3385c712825497` on `master`. The three supplied dirty files may remain dirty because the patch does not touch them.
2. Run `git apply --check PATCH.diff`.
3. Run `git apply PATCH.diff`.
4. Rebuild/reprocess `index.db` for schema v40 rather than attempting an in-place derived migration. In the repository's managed environment this is the documented `polylogue ops reset --index && polylogued run` path, against synthetic/demo data first.
5. Run the focused commands in `TESTS.md`, then managed affected verification, ruff, mypy, and schema/render policy in a complete dependency environment.
6. Do not close `polylogue-1vpm.6.2`, `polylogue-2qx.2`, or `polylogue-z9gh.7` based only on this package. The incident-specific live/admitted evidence gate remains open.

## Exact residual for polylogue-z9gh.7

The terminal gate still requires all of the following, none of which this synthetic substrate package claims to execute:

1. Run all seven `polylogue-t8t` continuity flows as real MCP walks.
2. From sparse repository/time/parallel-agent clues, find coordinator `cf0c6474-da22-44be-af3e-666037aa5ea4` and run `wf_54d4fb2e-841`; distinguish four Workflow invocations from one resumed run; reconstruct 50 call keys, 91 attempt transcripts, 65 result records over 49 completed keys, one unresolved key, and the final structured result; exclude the coordinator's other 38 child sessions.
3. Distinguish model, material, call, attempt, and effect scopes, citing git, PR, and Beads observations with uncertainty.
4. Prove paging is lossless, cancellation stops work, and measured latency/memory satisfy declared SLOs.
5. Prove a cold model succeeds from MCP schemas, errors, and catalog evidence alone.
6. Run mutations for continuation state, selective SQL, orchestration links, source coverage, and provenance classification.
7. Publish the terminal AC matrix with every mandate Bead classified satisfied, deferred to a named successor, or still blocking.

The immediate prerequisite remains `polylogue-2qx.2`: admit and provenance-link coordinator streams, run state, journal revisions, all transcript/meta pairs, and adopt manifests. The current Bead state says its previous claim was orphaned and was reset open on 2026-07-18 because no matching commits existed on master.

## Verification performed

- 13 focused work-evidence/reconciliation/Claude/repository-route tests passed.
- 82 ObjectRef/core reference tests passed.
- 2 targeted index lifecycle declaration-policy tests passed.
- all changed Python modules/tests passed `compileall`.
- staged patch passed `git diff --check`.
- `PATCH.diff` was checked for application against clean `bf8191b3...` during package validation.

See `TESTS.md` for commands, harness constraints, and the two unrelated lifecycle-test failures.

## Risks and remaining work

- The actual `wf_54d4fb2e-841` source family was not supplied as admitted raw evidence. Incident counts, P1 before/after state, live git/GitHub APIs, daemon behavior, MCP walks, browser/archive secrets, and deployment state are unverified.
- The local environment lacked the repository's locked dependency set and network cache. Focused tests used temporary compatibility modules for `aiosqlite`/`ijson` and omitted the unavailable sqlite-vec embeddings tier while exercising the real production schema, repository mixins, SQL, DTOs, and query functions. Those temporary modules are not in the ZIP.
- Full `test_index_fast_forward_lifecycle.py` has two pre-existing internally inconsistent assertions that call the report for version 37 while retaining newer declarations. The authoritative v39 baseline already returns `(38, 39, 37)`; this patch's v40 correctly adds `40`. The two current failures return `(38, 39, 40, 37)`. The target current-version declaration checks pass.
- Existing provider-specific correlation entry points were adapted, not deleted. A repository-wide call-site/deprecation pass could certify retirement separately.
- Generated-surface checks, ruff, mypy, managed testmon selection, full storage bootstrap with sqlite-vec, and default affected verification remain unverified.

A small repair iteration could address lint/type findings or packaging defects discovered in the operator's complete environment. A substantial second pass is worthwhile only when the missing 2qx.2 artifacts or an authorized privacy-safe incident fixture are available; that pass could add the complete Claude materialization route, actual before/after Beads/git reconciliation, and the z9gh.7 MCP replay. Without that evidence, a larger pass would mainly add unsupported scaffolding rather than trustworthy value.
