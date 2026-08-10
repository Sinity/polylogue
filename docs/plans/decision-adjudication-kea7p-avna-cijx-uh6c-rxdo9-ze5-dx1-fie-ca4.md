# Decision adjudication: kea7p, avna, cijx, uh6c, rxdo.9, ze5, dx1, fie, ca4

Status: Sol adjudication packet, 2026-08-03. Scope is decision and implementation slicing only. This packet does not change production code or Beads state.

## Executive decisions

| Bead | Final decision | Policy owner | Readiness |
| --- | --- | --- | --- |
| `polylogue-kea7p` | Build differential reindex as a declared, fingerprint-gated transition state machine. A hash match alone never authorizes a skip. Preserve from-empty rebuild as recovery and equivalence proof. | Implementer, constrained by existing derived-tier doctrine | dependency-blocked |
| `polylogue-avna` | Generalize `seq()` into an actions-only row-pattern model with embedded event-order semantics, captures, measures, explicit overlap, and SQL/Python parity. Keep mixed streams and fuzzy semantic tokens out of v1. | Implementer, following the corrective contract | execution-ready with packet |
| `polylogue-cijx` | Model repository and file trajectories as graded observations, checkpoints, and replay receipts. Repository identity is evidence-ranked, file identity is not path identity, and Polylogue does not become a VCS. | Implementer, following the corrective contract | execution-ready with packet |
| `polylogue-uh6c` | Separate membership, affinity, and confidence in types, storage, queries, and renderers. One operator choice is required for existing unqualified tag migration. | Model is settled; namespace migration is operator-owned | operator-decision-blocked |
| `polylogue-rxdo.9` | Treat rigor as one frame-exact claim protocol over canonical definitions and evidence axes. Reconcile landed primitives into production routes and consume the `stc` experiment definition rather than creating another one. | Operator doctrine is settled; implementation is constrained by dependencies | dependency-blocked |
| `polylogue-ze5` | Keep unified `assertions` storage, derive a four-class lens, add relation and revision tables in the next user-tier migration, and use class-appropriate surface nouns. | Operator-ratified | execution-ready with packet |
| `polylogue-dx1` | Proceed with Starlette and uvicorn through an ASGI-fronted compatibility ramp. Keep browser capture separate and preserve all security and byte contracts. | Operator-ratified | execution-ready with packet |
| `polylogue-fie` | Keep everything permanently. Proceed with blob zstd, measure before FTS sharding, and retain blue-green fresh-first rebuilds. Do not adopt mutation-only index maintenance. | Operator-ratified | evidence-blocked |
| `polylogue-ca4` | Stay SQLite-only now. DuckDB may become an optional read lowerer only after canonical measure lowering exists and a real five-query probe meets the ratified 10x gate. | Operator-ratified | dependency-blocked |

The only current operator question is the namespace assigned to existing unqualified tags. Every other choice is either ratified already or follows directly from current durability, provenance, and single-writer invariants.

## Evidence method and global constraints

The Bead records were read directly from `.beads/issues.jsonl`, including description, design, acceptance criteria, notes, comments, dependencies, and status. `bd` was not invoked because every `bd` call can re-import an aging worktree's JSONL into shared state. Merged history was inspected on `origin/master`. Current source, tests, schemas, and the read-only live archive were inspected from this worktree. `polylogue-qj5x` was not adjudicated. It appears below only where it blocks the fingerprint-bootstrap rebuild.

The following architecture rules apply to every lane:

1. Durable source and user changes use numbered additive migrations and a verified backup manifest. Derived index changes use canonical DDL plus a declared lifecycle delta.
2. Public surfaces consume product, insights, operations, or API contracts. They do not add new direct substrate imports.
3. Content-addressed identities include every semantic field that can change a result. Friendly names are aliases, not competing identities.
4. Unknown, absent, ambiguous, stale, and failed are distinct states. Confidence never substitutes for evidence grade or frame coverage.
5. Tests must traverse the production writer, reader, lowerer, or actuator. A fixture-only replica is not acceptance evidence.
6. The next lane that owns `storage/sqlite/archive_tiers/user.py` owns the entire user-schema version bump. The next lane that owns `storage/sqlite/archive_tiers/index.py` owns the entire index-schema bump. Parallel lanes must not edit those files concurrently.

## `polylogue-kea7p`: differential reindex

### Current state and unresolved decision

The ordinary ingest writer already performs the useful T1 primitive. `_write_session` in `polylogue/pipeline/services/ingest_batch/_core.py` compares `sessions.content_hash` with `SessionWritePayload.content_hash`, refreshes flags and raw links, schedules per-session FTS repair when needed, and returns before row replacement on a match. `session_content_hash` in `polylogue/pipeline/ids.py` covers the normalized parsed payload. It does not prove historical lowering fidelity, contextual lineage composition, derived-at-write columns, identity, or multi-session raw membership.

`polylogue/storage/sqlite/lifecycle.py` already distinguishes clone-safe operations, `SHAPE_FORWARD_TARGETED_REPROCESS`, and `SEMANTIC_REPARSE`. `polylogue/storage/index_generation.py`, `polylogue/maintenance/rebuild_index.py`, and `polylogue/daemon/bulk_rebuild.py` already supply owned generations, resumable transactions, source-snapshot staleness, bulk terminal convergence, promotion, and one retained superseded generation. The unresolved decision is how to reuse these mechanisms without allowing a stale semantic result to pass a cheap skip.

Current sessions have no trustworthy semantics stamps. `polylogue-xselt` is intended to add parser and lowering fingerprints, but it is blocked by `polylogue-slshy`. The production bootstrap `polylogue-818fy` also depends on the deliberately excluded `polylogue-qj5x`. Therefore the current archive cannot safely use differential semantic skips.

### Final adjudication

Adopt a four-input skip oracle and a transition state machine. A session is skippable only when all of these are true:

1. The raw is an accepted current authority head under the current authority fingerprint.
2. The stored parser fingerprint equals the current fingerprint for the origin.
3. The stored lowering fingerprint equals the current shared lowering fingerprint.
4. When parsing is performed, the current parsed content hash equals the stored content hash.

The authority fingerprint covers raw revision selection, membership arbitration, identity derivation policy, and accepted-head semantics. The parser fingerprint covers origin-specific detection, parser, sidecar assembly, and parser-owned classifications. The lowering fingerprint covers parsed-tree identity and hashing, normalized row lowering, search text, material-origin and message-type classification, lineage composition, and write-time projections. Fingerprints are semantic dependency manifests, not git commit ids or whole-package hashes.

The state machine is:

1. `DISCOVER`: acquire the current source authority epoch, active index epoch, schema versions, fingerprints, and declared delta set.
2. `CLASSIFY`: classify each delta as clone-safe DDL, projection backfill, parser semantic, lowering semantic, authority semantic, derived convergence, or full-rebuild-only.
3. `PLAN`: compute affected origins, raw heads, session and lineage closure, reverse-census scope, estimated changed bytes, temporary disk, and the in-place versus blue-green cost.
4. `SNAPSHOT`: for in-place work, create an owned reflink rollback generation. If an exact cheap snapshot is unavailable, select blue-green.
5. `APPLY_SHAPE`: apply only declared clone-safe canonical DDL. A `SEMANTIC_REPARSE` declaration can never be converted into a shape-only operation.
6. `T0_CENSUS`: skip parsing only for accepted raw heads whose materialization receipt and every member session carry current authority, parser, and lowering fingerprints. Missing stamps are a mandatory miss.
7. `T1_PARSE_COMPARE`: parse every remaining raw. Replace the affected session closure when content differs or when a semantic fingerprint differs. A hash match may avoid payload replacement only after fingerprint equality; it cannot avoid a declared projection or convergence obligation.
8. `T2_BACKFILL`: run projection-only backfills whose declaration names source columns, target columns, scope, producer version, and verifier. If values depend on parser semantics, this state routes through reprocess instead.
9. `REVERSE_CENSUS`: tombstone sessions whose expected identity is no longer emitted by an accepted head, verify the expected session-id set per raw, and prove membership conservation for every multi-session raw.
10. `CONVERGE_CORE`: refresh every index-local dependency required for a self-consistent reader snapshot, including lineage closure, FTS, action pairs, delegation facts, and affected index insights. In-place mode performs writes, core convergence, and proof inside one bounded SQLite transaction; if that transaction would violate the writer budget, the planner must select blue-green.
11. `PROVE`: compare the affected closure against a from-empty scratch materialization, validate canonical table manifests, verify stored-row hashes independently of the stored content hash, and run archive invariants before the in-place commit or candidate promotion.
12. `COMMIT` or `ACTIVATE`: commit the bounded in-place transaction after proof, or atomically promote the exact-ready candidate generation. Record the source epoch, delta manifest, fingerprints, counts, timing, and proof roots.
13. `CONVERGE_FOLLOW_ON`: enqueue embeddings and any other cross-tier or intentionally deferred model. `false_means_pending` records convergence debt. The reindex operation remains pending until required follow-on stages reach their declared freshness target, even though readers may use the core-consistent index with explicit freshness/debt metadata.
14. `CLEANUP`: retain one exact rollback generation and prune only generations already covered by retention policy.
15. `STALE`, `ROLLBACK`, or `ESCALATE`: source-authority change before proof makes the plan stale. Replan changed raws until a fixed point. Any proof failure restores the rollback generation and escalates the same declaration to blue-green or a full rebuild.

In-place remains the default only when canonical DDL is compatible, an exact reflink rollback exists, the affected closure and proof fit one bounded transaction, and the measured cost model favors it. Selection uses estimated changed bytes and graph closure, not a hard-coded percentage. Blue-green remains mandatory for incompatible DDL, unavailable rollback, multi-pass core mutation, large contextual closure, uncertain authority, or failed differential proof.

### Rejected alternatives

- Hash-only skip is rejected because historical lowering bugs, derived columns, identity drift, and lineage context can survive a matching parsed-payload hash.
- Parser fingerprint without lowering and authority fingerprints is rejected because it cannot detect shared writer, identity, classifier, or raw-selection changes.
- Whole-archive `SEMANTIC_REPARSE` as the permanent default is rejected because origin and surface declarations can safely bound future work after bootstrap.
- Mutation-only index maintenance is rejected. The from-empty builder remains the recovery oracle, periodic audit path, and proof reference required by the fresh-first doctrine.
- Per-session reverse existence checks are rejected because they miss identity changes and wrong membership splits.

### Contract, schema, compatibility, failure, and rollback

Add `authority_fingerprint`, `parser_fingerprint`, and `lowering_fingerprint` to derived session rows, plus a derived `raw_materialization_receipts` relation containing raw id, fingerprints, expected membership digest and count, materialized session ids digest, source authority epoch, and producer version. `xselt` currently specifies only parser and lowering stamps; its implementation should leave the authority column and receipt shape to `kea7p` or be extended before bootstrap. This is an index-tier semantic-reparse change. Existing rows cannot be backfilled truthfully from stored columns.

Projection declarations extend `IndexDeltaDeclaration` with affected surface, origin scope, dependency fingerprints, backfill plan, convergence stages, and proof plan. They do not weaken the existing rule that any crossed semantic-reparse delta blocks SQL fast-forward.

Compatibility: routine semantic transitions stop instructing operators to reset the index after the fingerprint bootstrap. The full rebuild command remains as recovery and audit. Privacy is unchanged because fingerprints and membership digests contain no content. Performance improves only after the initial bootstrap. Failure is fail-closed: missing stamps, source drift, ambiguous membership, or proof mismatch means reprocess or rebuild. Rollback is atomic pointer restoration to the exact pre-operation generation, not reversal by best-effort SQL.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| K0 | Finish stamp preconditions in `polylogue-slshy`, then `polylogue-xselt` owns `sources/origin_specs.py`, `storage/sqlite/archive_tiers/index.py`, `storage/sqlite/archive_tiers/write.py`, `storage/sqlite/lifecycle.py`, and archive verification. | Differential planning and mode selection | Serial index-schema owner. Blocks every later slice. |
| K1 | Transition contracts in a new `polylogue/maintenance/differential_reindex.py`; extend lifecycle declarations without executing writes. | `user.py`, daemon HTTP, query DSL | Parallel with non-index work after K0's schema lands. |
| K2 | Raw materialization receipts and reverse census in the same maintenance module plus focused helpers in raw authority and index generation. | Provider parsers except declared fingerprint inputs | Serial with any lane editing `write.py` or index DDL. |
| K3 | Planner and cost model integration in `maintenance/rebuild_index.py`, `storage/index_generation.py`, and `daemon/bulk_rebuild.py`. | Public CLI grammar and unrelated convergence stages | Depends on K1 and K2; serial with `b5l` changes in these files. |
| K4 | Convergence and proof integration in `daemon/convergence.py`, `maintenance/archive_verification.py`, and a canonical table-manifest helper. | Embedding implementation internals unless its declared stage changes | Can parallelize proof helper and convergence adapter with separate files. |
| K5 | CLI and daemon operation adapters plus retirement of reset-as-routine guidance. | Full rebuild engine and recovery command | Last, after behavior is proven. |

### Acceptance evidence and anti-vacuity

- Property test every delta class against the planner: any parser, lowering, or authority fingerprint mutation selects every affected raw and cannot reach T0.
- Mutation test that removes each fingerprint conjunct. The bootstrap fixture must then incorrectly skip and the test must fail.
- Real-writer test: build a fixture through `write_parsed_session_to_archive`, rerun unchanged, mutate a parser semantic input, a lowering input, and authority selection, and observe T0, T1, and replacement paths respectively.
- Identity test: one raw changes emitted session id. The old id is tombstoned and the new id is written.
- Membership property: arbitrary multi-session raw revisions preserve exactly the expected member set across reorder, split, merge, duplicate, and supersession cases.
- Lineage test: a parent change expands the affected closure to prefix-sharing descendants and matches a from-empty scratch build.
- Projection test: a projection-only delta backfills without parsing, while relabeling the same delta semantic-reparse makes parsing mandatory.
- Failure tests cover crash after snapshot, after DDL, midway through parse, after reverse census, and during proof. Restart resumes or rolls back without exposing an unproved state.
- Live proof after bootstrap records affected raw count, parsed count, hash skips, fingerprint skips, tombstones, convergence debt, table manifests, source epoch, elapsed time, I/O, and rollback-generation id.

Anti-vacuity requires the production ordinary ingest writer and generation promoter. Removing the content-hash early return must increase writes; removing a fingerprint conjunct must create a detected mismatch; disabling reverse census must leave a stale identity and fail the manifest comparison.

Readiness: **dependency-blocked** on `slshy -> xselt -> 818fy`. `818fy` also depends on excluded `qj5x`; that is the only qj5x relationship recorded here.

## `polylogue-avna`: typed row-pattern matching

### Current state and unresolved decision

`QuerySequencePredicate` in `polylogue/archive/query/predicate.py` models fixed action subsequences with `ordered`, `next`, and `within` edges. The Lark grammar in `expression.py` only accepts action field clauses inside `seq()`. `_action_sequence_steps_clause` in `storage/sqlite/archive_tiers/archive.py` orders by message position, variant index, and block position; `within` additionally requires nondecreasing timestamps. `runtime_matching.py` implements a separate Python matcher. `query_ast_schema.py` persists the strict sequence fragment. Matches currently filter sessions and discard bindings.

The unresolved decision is the smallest honest generalization that supports repetition, absence, captures, measures, and explicit overlap without making timestamp order or future semantic classifiers look structural.

### Final adjudication

Introduce a content-addressed `PatternDefinition` embedded in the canonical query AST. It contains an `EventOrderSpec`; no independent order registry is created. `seq()` becomes syntax sugar for a fixed concatenation pattern and lowers through the same engine.

Actions-only v1 has these types:

- `EventOrderSpec`: partition key, physical-session lineage policy, unit kind `action`, structural coordinate definition, optional event-time field, tie policy, evidence grade, horizon/as-of receipt, and relation-manifest version.
- `PatternExpr`: typed row predicate, concat, alternation, group, repetition with bounded or explicit unbounded maximum, optional, negative interval, start/end anchor, and capture.
- `MatchPolicy`: `first`, `all`, or `leftmost-longest`; `AfterMatchPolicy`: `skip-past-last` by default or declared overlap.
- `MeasureSpec`: capture-derived start/end, duration, count, field, and bounded aggregate measures. It consumes canonical metric identity when the output is a quantitative claim.
- `PatternMatch`: partition ref, structural span, captures, measures, order receipt, evidence grade, and ambiguity state.
- `MatchSet`: its own content identity over pattern ref, corpus epoch, relation manifest, order and overlap policy, ordered match membership, captures, measures, and Merkle roots. It may reference a result set projection, but is not a result-set alias.

Default ordering is the structural action coordinate already used by SQL. Event timestamps may constrain a structural sequence but do not establish order by themselves. Event-time ordering requires `reject-ambiguous` ties by default. A declared structural fallback yields a distinguishable order receipt and cannot claim replay-verified temporal order. Equal timestamps therefore remain ambiguous or visibly downgraded.

The execution plan is SQL prefilter plus a bounded Python Thompson NFA. Fixed, quantifier-free, absence-free action patterns keep a SQL fast path. Both paths consume one shared predicate and order contract and emit the same match protocol. Candidate and per-partition row caps produce a typed boundedness error rather than truncating silently.

### Rejected alternatives

- Strict adjacency by default is rejected because unrelated actions are normal. `next` remains alphabet-relative and explicit.
- Timestamp-only sequence is rejected because missing and equal timestamps cannot prove order or absence.
- A standalone `EventOrderSpec` registry is rejected until the order definition has an independent lifecycle.
- Full SQL lowering for arbitrary patterns is rejected because absence, repetition, captures, and overlap make it complex and backend-specific.
- A Python-only implementation is rejected because current fixed sequences already have an efficient SQL path and parity is an acceptance criterion.
- Fuzzy `test-fail` or `unresolved` tokens are deferred. V1 accepts structural action predicates and explicitly versioned rule classifiers only. Judged outcome and goal-resolution semantics remain separate later work.
- Mixed action/message/session streams are rejected in v1.

### Contract, schema, compatibility, failure, and rollback

PACK-A is storage-free except for the canonical query payload and generated AST schemas. PACK-B adds durable tables only for explicitly retained match sets: `pattern_match_sets` and `pattern_matches` in `user.db`, keyed to canonical query and corpus epoch, with JSON captures/measures/order receipts and membership roots. Routine results stay transient. This is an additive durable migration and must share or follow the exclusive user-schema migration owner.

Existing saved `seq()` queries retain their original query identity and definition protocol version. At load, a versioned adapter normalizes the legacy sequence node into an equivalent `PatternDefinition`; newly authored queries use the pattern node. The old SQL and Python sequence implementations are removed after parity fixtures prove the adapter, leaving one matcher. Rollback disables the new grammar and retained-match writes while legacy `seq()` remains executable through the adapter.

Privacy follows the referenced action fields. Captures containing commands, paths, or output remain private unless an existing projection redacts them. Performance is bounded by SQL prefilter, row caps, and explicit rejection of an unbounded corpus scan. A missing order field, ambiguous tie, candidate overflow, or relation-manifest drift yields an explicit non-match/ambiguous/error state, never a false negative presented as exact.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| A1 | Domain protocol in new `archive/query/patterns.py`; AST additions in `predicate.py` and `query_ast_schema.py`. | SQL storage, user schema, semantic classifier ontology | Parallel-safe except for other query AST work. |
| A2 | Grammar and canonicalization in `expression.py`; legacy `seq()` normalization. | Runtime and SQL matcher bodies | Depends on A1; owns the Lark grammar exclusively. |
| A3 | Python NFA in new `archive/query/pattern_matching.py`, using shared action order keys. | SQL lowering and storage | Parallel with A4 after A1. |
| A4 | SQL fast path in a new SQLite query helper, replacing `_action_sequence_steps_clause` after parity. | Python NFA | Parallel with A3; serial final deletion in `archive.py`. |
| A5 | Match-grain payloads and query pipeline integration in `unit_results.py`, metadata, API/MCP/CLI adapters. | User persistence | Depends on A2 through A4. |
| A6 | Retained match-set migration and repository methods in `user.py`, a numbered user migration, and focused user read/write modules. | Any ze5 or uh6c migration running concurrently | Exclusive user-schema lane; after ze5's version allocation. |

### Acceptance evidence and anti-vacuity

- Metamorphic generation of fixed action patterns proves SQL and Python emit identical ordered captures, measures, overlap, and ambiguity receipts.
- Property tests cover repetition, alternation, optional groups, anchors, absence intervals, empty matches, and overlap policies without catastrophic runtime.
- Equal-timestamp fixtures vary tie policy and evidence grade. No default path establishes event-time order.
- A mixed-stream parse fails with a named deferred-capability error.
- A candidate-overflow fixture returns a boundedness error and no partial exact result.
- Retained match-set tests mutate one capture, overlap policy, order spec, classifier ref, or relation manifest and prove the content identity changes.
- Cross-surface production tests run one expression through CLI, MCP, daemon, and API and compare the same match refs and provenance.

Anti-vacuity requires both production lowerers. Removing `EventOrderSpec` from either lowerer, changing `next` to gap-tolerant, or ignoring overlap must fail the parity corpus.

Readiness: **execution-ready with packet**. PACK-A through A5 can start. A6 waits for the exclusive user migration sequence.

## `polylogue-cijx`: repository and file evidence identity

### Current state and unresolved decision

The repository substrate is farther along than the Bead's last AC wording. `repo_identity.py` normalizes repository paths and projects repo-relative paths. `repo_observations.py` writes `repos`, `repo_checkouts`, and `session_repos`. `write_parsed_session_to_archive` writes parser-reported checkout commits to `session_commits`, and `queries/session_commits.py` plus `correlation_view.py` read them. `file_edits` now stores structured patches and original-file checkpoints from Claude Code wire evidence. Merged commits `8beef5f6a` and `5e23e6abf` supplied the current checkout-commit and file-edit evidence.

The remaining gaps are repository identity without a remote, path identity across renames, explicit observation/checkpoint/replay grades, coverage receipts, safe reproduction, and separation of checkout-head facts from heuristic live-git correlation.

### Final adjudication

Adopt evidence-ranked repository identity and observation-based file identity.

Repository resolution uses the strongest available evidence in this order:

1. Canonical remote identity, normalized across SSH, HTTPS, trailing `.git`, and case rules for the host.
2. Git history-root identity, including object format and the sorted root commit set, as ancestry evidence for a repository with no remote. It is not canonical repository identity when independent forks share that history; the result remains ambiguous/provisional until corroborating evidence exists.
3. Git common-dir identity for an empty or history-unavailable local repository, graded provisional.
4. Checkout root only as checkout identity. A cwd without git evidence remains a directory and never creates a repository.

`RepositoryIdentityReceipt` records the chosen key, authority, aliases, observed remote, common dir, history roots, checkout root, and observation time. Stronger later evidence merges aliases through an explicit `same-repository` edge; it never silently rewrites unrelated repositories.

A file entity is scoped to a repository and is not equal to a path or blob. `FileObservation` records session/action refs, checkout, repository, file entity, repo-relative path, operation, proposed/applied/reverted/unknown state, pre/post hashes when captured, tool outcome, observation time, evidence grade, and coverage gaps. Rename/same-file edges require explicit tool or git-diff evidence. Equal content alone never merges file entities.

Grades are:

- `observed`: action-derived evidence only, with explicit gaps for shell side effects, generators, human changes, concurrency, and external processes.
- `checkpointed`: a pre/post file hash, git tree, or equivalent state anchors the interval.
- `replay-verified`: a bounded disposable-worktree reconstruction matches the checkpoint and verifier receipt.

The existing `session_commits` relation becomes the narrow checkout-head fact. Add `checkout_head` to its detection vocabulary, attach repository identity/evidence grade, and keep heuristic time-window/file-overlap correlation as a visibly separate derived view. No request-time live-git computation may masquerade as the persisted checkout relation.

Reproduction prefers applying a captured patch or checking out a target commit at the recorded base. A disposable worktree is not a host sandbox: `safe_verify` is a command classification, not authorization to execute repository-controlled code. Automatic execution requires an actual filesystem/process/network sandbox with credentials and ambient archive access denied, or explicit operator authorization recorded in the receipt. `mutating_patch` is allowed only inside the disposable worktree and sandbox. `networked`, `secret_sensitive`, `interactive`, and `unknown` remain plan-only without explicit authorization. The receipt cites original evidence, exact base/target, commands, sandbox/authorization identity, environment fingerprint, outputs, and cleanup state.

### Rejected alternatives

- Cwd or absolute root as repository identity is rejected because it breaks across subdirectories, worktrees, and renames.
- Remote-only identity is rejected because local repositories may have no remote.
- Blob hash as file identity is rejected because content changes and identical files are common.
- Git-style line authorship is rejected. The product reports proposer, applier, generator, and observed committer only where evidence supports them.
- Replaying every historical command is rejected as unsafe and less reproducible than applying the resulting patch or target commit.
- A generic VCS or replay executor is rejected.

### Contract, schema, compatibility, failure, and rollback

Add derived index relations for repository identity receipts, repository aliases, file entities, file observations, identity/rename edges, trajectory checkpoints, and replay receipts. Extend `session_commits` and normalize its detection vocabulary. This is a derived semantic-reparse schema change because repository ids and file observations depend on parser, git, and trajectory semantics.

Promoted evaluation definitions and operator judgments remain assertions or `stc` experiment records in `user.db`; the trajectory substrate does not add durable truth tables. Existing `repos` ids become aliases when the new resolver can prove continuity. Public paths are repo-relative. Absolute checkout paths, original file bodies, and verifier output are private evidence and pass existing redaction before public projection.

Failure to resolve a repository leaves a directory observation. Missing pre/post bytes leaves hashes unknown. A deleted checkpoint automatically downgrades dependent trajectory claims. Disposable worktrees are created under the established temporary-work policy, have no credentials injected by default, and are removed after a receipt records cleanup. Rollback is an index-generation rebuild; no durable user data is destructively migrated.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| C1 | Repository identity domain in `archive/session/repo_identity.py` and new identity receipt types. | Index DDL and file observations | Parallel-safe with C2 design tests if symbols do not overlap. |
| C2 | File observation/checkpoint domain in new `insights/file_trajectory.py`, reusing `file_edits`. | Reproduction execution and public surfaces | Parallel with C1. |
| C3 | Exclusive index schema and writer/readers in `archive_tiers/index.py`, `write.py`, `storage/insights/session/repo_observations.py`, and focused query modules. | User schema, query DSL, work-evidence tracker adapters | Depends on C1 and C2; exclusive index-schema lane. |
| C4 | Correlation view conversion so persisted checkout-head and heuristic candidates are distinct. | Session parser metadata writer | Depends on C3. |
| C5 | Safe reproduction planner/executor under `operations` and `insights`, with an actuator classification contract. | Daemon HTTP and arbitrary shell replay | Can parallelize with C4 after C2. |
| C6 | G1/G2/G4/G5/G7 readers and privacy-aware API/MCP/CLI projections. | G3 authorship claims, G6/G8 causal evaluation | Depends on C3; G6/G8 additionally depend on rxdo/stc. |

### Acceptance evidence and anti-vacuity

- Fixtures cover one remote through two worktrees, SSH/HTTPS spellings, repository rename, no-remote history, empty git repository, and plain directory.
- Property tests prove path normalization cannot escape the repo root and equal blobs do not imply same-file identity.
- A tool edit plus an uncaptured external edit renders observed coverage gaps and cannot claim exact tree, authorship, or replay.
- Removing a checkpoint downgrades all dependent claims; changing a captured hash makes replay verification fail.
- The reproduction integration test creates a real disposable git worktree, applies a patch or target commit, runs a declared safe verifier, and compares the checkpoint. Networked and unknown commands never execute.
- A production write/read test persists checkout-head metadata, reads it through repository/API, and proves disabling the writer removes the fact. The live-git heuristic must not recreate it.
- G5 tests a selected repository generation, dead path, renamed path with evidence, and unresolved repository.

Anti-vacuity depends on real `write_parsed_session_to_archive`, git worktree operations, and the public correlation reader. Mock-only git graphs are insufficient.

Readiness: **execution-ready with packet**. Tracker effects that later consume repository observations must follow the eventual qj5x outcome, but file/repository trajectory work does not need qj5x to start.

## `polylogue-uh6c`: tag membership, affinity, and confidence

### Current state and unresolved decision

Current `session_tags` in index.db combines tag text, `user|auto` source, method, scalar confidence, and evidence. `_all_session_tags_sql` unions parser auto rows with active user-tier `AssertionKind.TAG` rows. `upsert_session_tag_assertion` explicitly calls `require_promotion=False`; its comment says tags are categorization rather than epistemic claims. This contradicts the Bead's corrective authority requirement because an agent-authored membership can become active without the canonical judgment transaction. Public filters and mutations use ambiguous free-form `tag` strings.

The three-axis model is settled. The unresolved operator decision is how existing unqualified durable tags map into namespaces.

### Final adjudication

Adopt three independent protocols:

- `TagMembership`: asserted membership of a subject in a `TagRef`, with actor, authority, status, qualifiers, evidence refs, and judgment lineage. Operator-authored membership may be active directly. Parser-structural membership may be active with a structural definition receipt. Agent, model, detector, and heuristic membership is always a non-injected candidate until the canonical `37t.12` transaction accepts it.
- `TagAffinity`: a derived score between a subject and content-addressed prototype under a named embedding/model and evaluation world. It lives in a rebuildable derived relation and never grants membership.
- `TagConfidenceReceipt`: calibrated uncertainty about a classifier or judgment, bound to assertion ref, actor, execution context, definition, calibration ref, and evaluation world. An uncalibrated scalar is labeled uncalibrated and never grants membership or affinity.

`TagRef` is namespace plus name. It is plural and non-hierarchical; the namespace separator carries no parent/child semantics. A prototype is a separate `prototype:<hash>` resource associated with a tag for discovery, not the tag's identity.

Public predicates are axis-specific: `tagged:`, `tag-affinity:`, and `tag-confidence:`. `tag:x>0.7` is rejected. Any axis conversion requires a content-addressed conversion definition and emits a new receipt.

### Operator decision brief: existing unqualified tags

Exact question: **Should existing user-authored tags map to `personal/<name>` and parser-authored tags map to `system/<name>`?**

Option A, recommended: map user rows to `personal`, parser structural rows to `system`, copy-forward durable user assertions with supersession relations, and provide a read-only bare-name compatibility resolver for saved queries. This preserves known authority, gives useful namespaces, changes canonical API values, and requires a verified user.db migration.

Option B: map all existing strings to `legacy/<name>` and keep authority only in membership receipts. This makes no product-semantic guess and is easiest to roll back, but leaves the main corpus in a permanent low-information namespace and makes ordinary user tags less pleasant.

Option C: retain an unqualified default namespace. This maximizes wire compatibility but contradicts the ratified namespaced model and is rejected unless the operator explicitly changes that policy.

Compatibility/cost/risk: A changes canonical refs and therefore saved query and assertion ids; it needs copy-forward, aliases, and backup. B changes refs too but does not split by authority. Neither option loses original rows. Smallest answer needed: `A` or `B`, plus replacement namespace names only if `personal` and `system` are not desired. Blocked work: durable migration, canonical API examples, saved-query rewriting, and final membership fixtures. Affinity and confidence domain work can proceed independently.

### Rejected alternatives

- One scalar over membership, similarity, and confidence is rejected as a category error.
- Thresholding affinity into membership without a conversion definition and judgment is rejected.
- A prototype id as tag identity is rejected because model/prototype changes must not rename membership.
- Tag-specific review queues are rejected. Membership uses `37t.12`.
- The current `require_promotion=False` agent path is rejected.
- Implicit hierarchical semantics are rejected.

### Contract, schema, compatibility, failure, and rollback

Replace derived `session_tags` with a clearly named structural `session_tag_memberships` relation. Durable user and agent membership remains `AssertionKind.TAG`, with an axis-qualified value and normal assertion lifecycle. Add a durable content-addressed `tag_prototypes` definition table only if prototypes need operator retention; vectors and affinity rows stay in embeddings/index derived tiers. `tag_affinities` records subject, prototype, model, evaluation world, score, definition, source epoch, and evidence. Structured confidence receipts live in the assertion value; the existing top-level confidence remains a convenience projection only when its calibration is declared.

The index change is semantic-reparse. Prototype and durable membership copy-forward are additive user migration plus data migration behind backup. Privacy: private membership, prototype text, and affinity are not exposed by default. Performance: membership uses equality indexes; affinity uses bounded vector candidate retrieval and never joins every subject to every prototype. Unknown axis data remains unknown. Failure to resolve a conversion, calibration, or judgment fails closed. Rollback restores the user backup and prior index generation; compatibility aliases are read-only and cannot create new legacy rows.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| U1 | `TagRef`, membership, affinity, confidence, prototype, and conversion contracts in new core/insights modules. | User and index schemas, query grammar | Can start before operator answer. |
| U2 | Fix agent membership lifecycle in `user_write.py` and canonical judgment integration. | Namespace migration and affinity storage | Depends on `37t.12`; serial with ze5/rxdo edits to `user_write.py`. |
| U3 | Derived membership and affinity schema/readers. | Durable user migration | Exclusive index-schema lane; can proceed with namespace parameter abstracted. |
| U4 | Durable prototype and namespace copy-forward migration. | Any concurrent user-schema lane | Operator answer required; exclusive user-schema lane after ze5. |
| U5 | DSL predicates and lowerers in query grammar, metadata, and SQLite/runtime matching. | Pattern grammar changes in avna | Serialize with avna grammar ownership. |
| U6 | API/MCP/CLI/renderers and saved-query compatibility. | Storage internals | Depends on U2 through U5. |

### Acceptance evidence and anti-vacuity

- Seed high affinity without membership, membership with unknown affinity, and low-confidence classifier output without either axis changing.
- Run agent `add_tag` and bulk tag through production actuators. Before judgment the candidate is `inject:false` and invisible to `tagged:`. After canonical acceptance it becomes visible exactly once.
- An existing same-name row tests axis, actor authority, and judgment state; it cannot short-circuit a new candidate into active state.
- Changing prototype/model/world changes affinity receipt identity but not `TagRef`.
- Axis-mixing expressions fail with a named conversion requirement.
- Cross-surface payloads retain axis, actor, definition, calibration, and evidence.
- Migration tests start from a real pre-migration user.db and prove aliases, supersession, idempotency, backup manifest, and rollback.

Anti-vacuity requires the real tag actuators, assertion promotion transaction, query lowerer, and union read path. Removing `require_promotion` enforcement must make the pre-judgment visibility test fail.

Readiness: **operator-decision-blocked** for the durable namespace cutover. U1 can start; U2 additionally waits on `37t.12`.

## `polylogue-rxdo.9`: analysis rigor

### Current state and unresolved decision

The operator adopted the program, but the program is not complete. Current child state is: `.9.1`, `.9.6`, `.9.7`, and `.9.12` in progress; `.9.4`, `.9.8`, and `.9.10` open; the other mechanism children are closed. Landed primitives include `EvidenceValue`, content-addressed metrics and ratios, registration ordering, public-claim validation, negative-control validation, comparative judgments, rankers, blinding, calibration, elicitation, cascades, and experiment projection.

Several contracts are not yet coherent. `docs/design/analysis-rigor.md` still opens with the superseded population claim. `MetricDefinition` has a second measurement-authority vocabulary and uses `required_enumeration="exact"`, while `EvidenceValue` owns the canonical census/sample/inferred-partial axes. `registration_status` compares timestamps, epochs, and metric/query refs but not a frozen analysis-definition digest. `FindingAssertion` can store query/result/frame/evaluation refs but does not require one canonical evidence envelope. `experiments.py` consumes a structural `ExperimentDefinitionLike` because `stc` has not landed. Many mechanism modules have narrow or no production consumers.

The unresolved decision is not whether the mechanisms remain. It is the closure contract that turns primitives into enforceable production claim semantics without parallel identities.

### Final adjudication

Adopt one `AnalysisDefinition` and one `EvidenceValue` envelope across measurements, findings, controls, experiments, and judgments.

`AnalysisDefinition` is content-addressed over query refs, metric refs, frame definition, enumeration requirement, exclusions, stopping rule, analysis plan, controls, claim class, and relevant actor/execution context. Preregistration stores this definition ref before execution. Evaluation must cite the identical definition ref and a later corpus epoch. Any change yields exploratory definition drift.

`EvidenceValue` remains the canonical independent-axis protocol. Exact means census-exact over the named stored frame and definition. Frame coverage, capture coverage, measurement authority, classifier uncertainty, judgment uncertainty, freshness, and temporal quality remain independent. No sampling interval appears for a census value; incomplete capture and model-derived measurement still render.

`MetricDefinition` imports the core authority and enumeration vocabularies. `catalog-estimated` maps to `catalog-derived`; generic `heuristic` is removed in favor of explicit `rule-derived` or `model-derived`. Hash changes are correct because the old definition was less precise. Friendly metric names may point to the new refs, but old refs are never silently reinterpreted.

`FindingAssertion` requires `analysis_definition_ref`, `metric_ref` when quantitative, and a serialized evidence value or a typed non-measurement capability declaration. Public support verdicts require frame, definition, as-of epoch, and evidence ancestry. Circular, stale, expired, private-only, or unresolved evidence blocks current-supported and cold export.

Mechanism J consumes the actual versioned `stc ExperimentDefinition`, assignments, exposures, preregistration, frame, exclusions, stopping, outcomes, and metric definition. Without all of them, the result is observational/exploratory. The structural protocol remains only as a boundary interface if it prevents a layering cycle; it cannot be satisfied solely by tests with no production producer.

Comparative judgment keeps tie, incomparable, abstain, insufficient evidence, partial order, judge actor, and execution-context calibration. Confidence intervals for latent rankings are judgment-process uncertainty, not sampling intervals over archive counts.

### Rejected alternatives

- Unqualified population exactness is rejected. The correct claim is frame-exact under named definitions.
- A second `MeasureSpec`, `JudgeSpec`, universal receipt table, or experiment object is rejected.
- Bootstrap intervals for missing capture/parser bias are rejected.
- Causal wording without preregistration, assignment, exposure, and outcomes is rejected.
- Design adoption or unit-only primitives as program completion is rejected.
- Dashboard-first work that does not change a claim or action gate is rejected.

### Contract, schema, compatibility, failure, and rollback

Most finding and judgment changes fit typed assertion `value_json` without a schema bump. Immutable analysis definitions should reuse the existing durable query/definition substrate or add one narrowly owned table in a coordinated user migration; do not store them in ops. `stc` owns experiment identity and lifecycle. Derived evaluation outputs remain rebuildable unless explicitly retained as findings.

Compatibility uses versioned definition protocols and friendly-name aliases. It does not preserve old hashes as if they meant the new vocabulary. Privacy and blinding are projection policies over retained provenance, not destructive omission. Performance is bounded by ancestry depth, result-set manifests, and declared evaluation budgets. Any missing required ref degrades or blocks the claim; it does not fill with prose. Rollback disables new publication/actuation gates only by restoring the prior user backup or code, while stored definitions and assertions remain readable by version.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| R1 | Correct doctrine and canonical vocabularies in `docs/design/analysis-rigor.md`, `core/evidence_value.py`, and `insights/measurement/metric.py`. | User schema and experiments | Parallel-safe except with other evidence vocabulary work. |
| R2 | Add `AnalysisDefinition` and definition-digest verification to measurement registration and canonicalization. | stc identity and query identity | Depends on R1. |
| R3 | Tighten `FindingAssertion`, writer validation, public claims, and finding provenance around the common evidence envelope. | Comparative judgment storage | Serial with ze5/uh6c in `user_write.py`. |
| R4 | Finish holdout and sampled-only uncertainty children through real query planner and result-set routes. | Generic statistics dependency | Can parallelize by separate modules after R1. |
| R5 | Wire blinding, controls, calibration, elicitation, and cascades through all production judgment surfaces. | New judgment identity | Depends on `37t.12`; split by surface files but keep one contract owner. |
| R6 | Replace test-only experiment fixtures with actual `stc` definitions and receipts. | Any second experiment table/type | Hard dependency on `polylogue-stc` and R2. |
| R7 | Program reconciliation matrix for all `.9.1` through `.9.16`, with falsification receipts. | Beads state in implementation lanes | Last. Coordinator owns Beads reconciliation. |

### Acceptance evidence and anti-vacuity

- A census-exact value with incomplete capture and model-derived measurement renders all three facts and no sampling interval.
- Changing any analysis-definition field after registration produces exploratory definition drift even when metric and query refs are unchanged.
- Circular, stale, expired, ambiguous, or private-only ancestry blocks current-supported and cold export through production renderers.
- A production experiment without any one of preregistration, assignment, exposure, frame, exclusions, stopping, or outcome receipts cannot produce a causal/confirmatory result.
- Judgment aggregation preserves nondirected verdicts and partial-order ambiguity; actor and execution context remain separate calibration strata.
- Every child identifies a production consumer and a mutation/removal that makes its test fail. Zero-caller modules do not satisfy program closure.

Anti-vacuity requires the actual query planner, finding writer, publication exporter, compare/judge surfaces, and eventual stc producer.

Readiness: **dependency-blocked** on `stc`, `9l5.7`, `37t.12`, and unfinished child work. R1 and R2 are immediately executable.

## `polylogue-ze5`: user.db vocabulary and record sufficiency

### Current state and unresolved decision

The operator already ratified the four-class lens and surface nouns. The migration recipe is stale: current `USER_SCHEMA_VERSION` is 10, `assertions.confidence` already exists, `user_settings` is already separate state, and annotation schemas/batches already have dedicated tables. `supersedes_json` exists but there is no normalized relation table. Upserts overwrite mutable fields and preserve no general revision history. `ASSERTION_CLAIM_KINDS` is an informal epistemic subset, not a complete class registry.

The unresolved implementation decision is how to land the ratified model without storing a second class value that can drift or losing durable history.

### Final adjudication

Keep the unified assertions table and derive `AssertionClass = epistemic | curation | workspace | comms` exhaustively from `AssertionKind`. Do not add a stored class column.

Initial mapping:

- Epistemic: annotation, correction, decision, caveat, lesson, blocker, run_state, prompt_eval, ontology_candidate, ontology_governance, transform_candidate, pathology, finding, judgment, comparative_judgment, secret_candidate, and excision_record.
- Curation: mark, highlight, suppression, tag, and metadata.
- Workspace: saved_query, recall_pack, and workspace_note.
- Comms: note, handoff, and excision_request.

`NOTE` remains comms because the current helper uses it for blackboard notes. User-facing epistemic objects use “notes”; API types and operations use “records”; `assertion` remains storage/enum terminology. Domain-specific curation, workspace, and comms surfaces retain their own nouns.

The next user migration, version 11 unless another migration lands first, adds:

- `assertion_relations(src_assertion_id, dst_assertion_id, relation, created_at_ms)` with relation `supersedes | contradicts | refines`, FKs, indexes in both directions, and no self-edge.
- `assertion_revisions(assertion_id, revision_seq, body_json, created_at_ms)` with immutable per-record sequence. `body_json` is a canonical snapshot of every mutable semantic field, not only `body_text`.
- A trigger or the single writer chokepoint inserts the old snapshot only when semantic fields change. Updated-at-only idempotent writes do not create revisions.

Migration backfills `supersedes_json` into normalized relation rows. New writes use normalized relations. The old durable column remains physically present because durable destructive migration is forbidden, but it is no longer authoritative. Older binaries cannot open schema v11, so dual-writing obsolete JSON is unnecessary.

Relations and revisions cascade on an authorized physical deletion so privacy excision removes historical content too. Ordinary retraction remains status plus reason and retains history.

### Rejected alternatives

- Per-class tables are rejected because the unified audit/query/lifecycle substrate is load-bearing.
- A stored class column is rejected because class is a pure exhaustive derivation from kind.
- Reusing `supersedes_json` for contradiction and refinement is rejected because relations need indexed traversal and integrity.
- Capturing only changed prose is rejected because value, status, confidence, visibility, and policy changes are also belief/history changes.
- Adding confidence again is rejected because it already exists. Calibrated confidence semantics belong to the owning record contract.
- Dual-writing normalized relations and JSON indefinitely is rejected.

### Contract, schema, compatibility, failure, and rollback

This is an additive durable user migration with backup manifest and one-version-at-a-time upgrade. Fresh DDL and migration SQL must match. The relation vocabulary has one typed Python owner and a generator-tied DDL check or schema-policy assertion. The exhaustive class map fails if a new `AssertionKind` lacks placement.

Surface compatibility changes labels and payload names only at class-appropriate boundaries. Storage ids, assertion refs, and enum wire values remain. Existing API fields may retain versioned aliases where clients depend on them, but new documentation must not call saved views or tags assertions.

Revision bodies can contain private content and inherit the assertion's privacy and excision policy. Indexes keep relation traversal bounded. A failed migration restores the verified user.db backup. A failed revision insert aborts the assertion update in the same transaction. Binary downgrade requires restoring the pre-v11 database, not ignoring the version.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| Z1 | Class registry and surface vocabulary audit in `core/enums.py`, a focused vocabulary module, glossary, and surface payload names. | User DDL and relation writers | Parallel-safe with Z2 planning, but serialize enum edits with rxdo/uh6c. |
| Z2 | Exclusive migration: `user.py`, next numbered migration, backup/version tests, relation/revision domain types. | Any avna or uh6c user migration | Sole user-schema owner. |
| Z3 | Writer integration and supersedes backfill/read conversion in `user_write.py` and focused relation/revision modules. | Tag authority and finding rigor changes | Depends on Z2; serial hotspot ownership. |
| Z4 | Judge queue contradiction pairing and record history readers through API/MCP/CLI. | Storage terminology | Depends on `37t.12` for final judge transaction integration. |
| Z5 | Generated OpenAPI/CLI schemas and vocabulary docs. | Product code | Last. |

### Acceptance evidence and anti-vacuity

- Upgrade a real v10 fixture through the backup-aware migrator; verify schema 11, backfilled supersedes edges, referential integrity, and restore.
- Exhaustively map every current `AssertionKind`; adding an unmapped kind must fail registry validation.
- Two conflicting lessons can be linked, traversed in both directions, and presented together in the production judge queue.
- Updating semantic fields creates one immutable old snapshot. Repeating an identical upsert creates none. Updating status/confidence creates a revision.
- Physical privacy deletion removes assertion, relations, and revisions.
- Surface contract tests show notes/records for epistemic records, tags/marks for curation, saved views/workspaces for state, and messages/handoffs for comms.

Anti-vacuity requires the production `upsert_assertion`, migration runner, judge queue reader, and public payloads. A test that inserts directly only into a fixture schema is insufficient.

Readiness: **execution-ready with packet**. Z4 waits on `37t.12`; Z1 through Z3 can proceed in the exclusive user-schema lane.

## `polylogue-dx1`: daemon HTTP substrate

### Current state and unresolved decision

The operator ratified ASGI with a presumption to proceed. Current evidence strengthens that decision: `polylogue/daemon/http.py` is about 5,500 lines, `stable_route_contracts()` declares 48 API routes, and host admission, authentication, and origin/CSRF remain separate handler calls. The daemon runs TCP and UDS `ThreadingHTTPServer` instances through `asyncio.to_thread`; shutdown needs a dedicated daemon thread to avoid `serve_forever` deadlocks. Archive reads use a bounded executor because accepted connections otherwise create unbounded threads. `/api/events` already has SSE and polling, but its handler sleeps in a request thread once per second.

Starlette and uvicorn are currently transitive through MCP, not direct runtime dependencies. Browser capture is a separate `BrowserCaptureHTTPServer` and must remain outside this migration. The unresolved work is the precise compatibility ramp, lifecycle ownership, and measurable abort gate.

### Final adjudication

Proceed with one in-process Starlette application and one uvicorn worker for the daemon API. Add Starlette and uvicorn as direct constrained dependencies. Do not migrate the browser-capture receiver under this Bead.

The ASGI app owns the public TCP and UDS listeners, composed middleware, typed parameter decoding, response/error envelopes, SSE cancellation/backpressure, route contracts, and lifecycle hooks. During migration, unmatched routes proxy to a private legacy server over an internal UDS. New routes never land on the legacy handler. One route family moves per slice, and the compatibility proxy and legacy server are deleted when the legacy route count reaches zero.

Security middleware composes, in order, trusted Host admission, credential resolution, exact Origin/CSRF policy, route capability/role, request budget, and sanitized error handling. `route_contracts.py` remains the migration authority until `polylogue-3utv` generates the router, OpenAPI, and client from one typed declaration.

Probe thresholds are implementer-owned guardrails: on a representative status/read/mutation/SSE mix, p95 non-streaming latency must not regress by both more than 20 percent and more than 5 ms; idle RSS must not regress by both more than 15 percent and more than 50 MiB; sustained SSE disconnects must leave no leaked tasks; write coordination, UDS, SPA, and extension-facing behavior must be compatible. A threshold breach pauses migration for diagnosis and can trigger the ratified abort.

### Rejected alternatives

- Staying hand-rolled is rejected by the ratified decision and growing private-framework cost.
- New-routes-only permanent hybrid is rejected because it leaves two security and lifecycle substrates indefinitely.
- A big-bang rewrite is rejected because 48 route contracts, the SPA, UDS clients, and mutations need family-level rollback.
- Migrating browser capture together is rejected because it is a separate receiver and would enlarge the blast radius.
- Multiple uvicorn workers are rejected because the daemon is one process and one SQLite writer.
- Reimplementing auth per endpoint is rejected; it belongs in middleware plus route policy.

### Contract, compatibility, performance, failure, and rollback

There is no database schema change. `/metrics`, `/healthz/live`, and `/healthz/ready` remain byte-stable, including content type and status. API JSON, headers, cookies, errors, pagination, cache validators, SSE ids, heartbeat/coalescing, TCP loopback default, UDS path/mode, bearer behavior, web credentials, and write-coordinator semantics are compatibility contracts.

Request bodies and paths become typed at the router boundary; domain handlers remain transport-neutral. Privacy projections remain downstream of authentication. Uvicorn uses one event loop and bounded thread offload only for blocking SQLite/domain functions. Cancellation must interrupt or abandon reads through the existing execution context without leaking a write.

During the ramp, a configuration switch selects ASGI-fronted or legacy listener ownership. Rollback switches the public listener back to legacy while the route family remains available there. Once a family is removed from legacy, rollback is the prior release, not an in-tree duplicate implementation. The final deletion occurs only after a full dogfood window and zero legacy contracts.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| D1 | Direct dependencies, ASGI app factory, typed request/error helpers, composed security middleware, and contract parity tests in new daemon modules. | Existing handler route bodies, browser capture | Parallel-safe with domain route extraction. |
| D2 | Alternate-port probe for status plus events/SSE, with latency/RSS/task receipts. | Canonical listener ownership | Depends on D1; no deployment change. |
| D3 | Listener lifecycle, TCP/UDS ownership, internal legacy UDS proxy, and shutdown integration in `daemon/cli.py`. | Browser capture lifecycle | Serial lifecycle hotspot. |
| D4 | Migrate read-only route families: health/observability, sessions/query, then insights/reference. | Mutations and maintenance | One family per PR, mostly parallel extraction but serialized router registration. |
| D5 | Migrate user mutations, ingest/reset, and maintenance through the existing write coordinator. | New write executor | Depends on D4 security and error parity. |
| D6 | Migrate SPA/static/bootstrap, delete legacy proxy/server at zero routes, simplify shutdown and bounded-read scaffolding. | Browser capture receiver | Last, after dogfood and client smoke. |

### Acceptance evidence and anti-vacuity

- Byte/golden parity for health, metrics, representative JSON, errors, headers, cookies, cache validators, and SSE frames across legacy and ASGI.
- Security matrix varies Host, Authorization, cookie credential, Origin, method, route policy, TCP/UDS, and auth-disabled local mode. Removing any middleware must expose a failing case.
- Real daemon tests cover concurrent reads, bounded admission, timeouts, cancellation, writer serialization, graceful shutdown, UDS clients, SSE reconnect and coalescing.
- Probe records p50/p95/p99, throughput, idle and loaded RSS, thread/task counts, disconnect cleanup, and code-per-route.
- Required client proof: SPA smoke, extension smoke, concurrent spool/dedup, and capture-gap fixture. The receiver remains unchanged, but integrations must not regress.

Anti-vacuity requires a real uvicorn listener and the production daemon app, not only Starlette's in-process test client.

Readiness: **execution-ready with packet**. This adjudication unblocks `polylogue-3utv` after D1 establishes the target declaration shape.

## `polylogue-fie`: archive scaling doctrine

### Current state and unresolved decision

The keep-everything and lever-order decisions are ratified. Current read-only evidence on 2026-08-03 shows:

- Active index target: 40,554,500,096 bytes, 23,496 sessions, 4,949,871 messages, 5,070,427 blocks, and 5,001,240 FTS rows.
- Durable source.db: 1,891,467,264 bytes and 43,124 raw rows. Logical raw blob bytes total 100,142,342,655; distinct raw blob hashes account for 72,045,958,857 bytes before the blob-directory census.
- embeddings.db: 845,926,400 bytes; ops.db: 86,810,624 bytes; user.db: 425,984 bytes.
- Promoted rebuild transaction wall windows: 4.37 hours for 41,363 raws and 99.0 GB, 34.95 hours for 101,347 raws and 97.4 GB, and 74.01 hours for 36,451 raws and 97.1 GB. These windows include bounded-pass idle time and are recovery-window evidence, not pure CPU benchmarks.

The current code already has blue-green owned generations, one-generation rollback retention, resumable byte progress, cost receipts, bulk terminal FTS/projection rebuild, and optional sharded from-empty builds. `devtools/archive_space_report.py` can produce a dbstat census, and `tests/benchmarks/test_sharded_rebuild.py` proves small-fixture K=1/4/8 equivalence. What is missing is the required immutable-copy object census, growth projection, and current/3x/10x resource benchmark. Therefore the conditional FTS decision remains evidence-blocked, not design-blocked.

### Final adjudication

Keep every session and raw observation permanently unless a separate explicit privacy deletion applies. The lever order is fixed:

1. Implement `polylogue-83u.5` blob zstd unconditionally with content-hash and restore proof.
2. Measure FTS footprint and rebuild cost. Shard hot/cold FTS only if FTS is the measured dominant degradation.
3. Continue blue-green, resumable, fresh-first index rebuilds, including from-empty sharding and kea7p differential convergence where proof permits.
4. Do not adopt mutation-only incremental index maintenance as the source of truth.

Kea7p does not violate this ruling. It replays authoritative source evidence through ordinary materialization, preserves from-empty equivalence, and escalates to blue-green. It is a convergence optimization, not abandonment of rebuildability.

Cold means access-temperature placement or compression, never deletion or loss of queryability. Any cold split has one logical query contract, exact freshness metadata, and an automatic fallback when a shard is unavailable.

### Rejected alternatives

- Retention windows, pruning, sampling away old sessions, and destructive cold storage are rejected by permanent operator policy.
- FTS sharding before a dbstat/consumer/rebuild census is rejected.
- Treating historical rebuild transaction wall time as pure engine performance is rejected.
- Incremental-only index evolution is rejected because it removes the recovery oracle and compounds semantic drift.
- Multiple simultaneous live full walks are rejected. Measurement uses one immutable/reflink copy and one reader.
- DuckDB is not an archive scaling lever here; ca4 owns optional analytical lowering.

### Contract, schema, compatibility, failure, and rollback

Blob compression is a durable source/blob representation change and needs a versioned envelope or manifest, atomic write, hash-over-uncompressed-content semantics, mixed compressed/uncompressed reads during conversion, backup/restore proof, and no recompression on dedup. FTS sharding is derived index architecture and requires a semantic index generation, unified query/continuation ordering, and exact per-shard freshness.

Blue-green temporarily amplifies derived disk usage by roughly one active index plus candidate and rollback overhead. The planner must preflight free space and refuse before work. Compression must not leak private content into shared dictionaries. Corruption or unsupported codec fails closed while preserving the original blob until verified conversion. Rollback reads old envelopes and atomically restores the prior generation.

### Implementation and evidence slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| F1 | Serialized reflink census using `devtools/archive_space_report.py`: every table/index bytes, rows, producer, consumers, rebuild cost, unread classification. | Live write paths and parallel walkers | Independent evidence lane. |
| F2 | Growth model from ops/source telemetry with 12/24 month ranges and assumptions. | Retention policy changes | Parallel with F1. |
| F3 | Current/3x/10x full-index and FTS benchmark harness, recording elapsed, compute versus idle, I/O, RSS, disk amplification, and recovery window. | Production live archive and policy defaults | Depends on representative scenario fixtures; serialized heavy lane. |
| F4 | `83u.5` zstd implementation in blob store, source metadata/migration, readers, GC, and restore tooling. | FTS and index schema | Independent implementation lane; durable blob owner. |
| F5 | Conditional FTS shard design and implementation only if F1/F3 identify FTS as dominant. | Blob format and general query algebra | Evidence gate; exclusive index/query owner. |
| F6 | Doctrine and operator runbook with capacity alerts, preflight, recovery, and rollback. | New policy | Last, summarizes receipts. |

### Acceptance evidence and anti-vacuity

- F1 scans one immutable/reflink copy with one reader and enumerates every object, including expensive unread structures with exact consumer search evidence.
- F3 uses the real rebuild and FTS engines, not a toy loop. Remove bulk-build suppression or sharding and the timing/resource receipts must change.
- Zstd property tests round-trip arbitrary bytes, preserve SHA-256 over original bytes, deduplicate mixed envelopes, survive crash boundaries, and restore from backup.
- Query parity spans hot/cold boundary, global ranking, continuation, counts, and stale/unavailable shard states if FTS sharding is admitted.
- Capacity failure tests refuse insufficient disk before creating a candidate and preserve active availability throughout a failed rebuild.

Readiness: **evidence-blocked** overall. F4 is execution-ready independently; F5 cannot start until F1 and F3 prove FTS is the bottleneck.

## `polylogue-ca4`: optional DuckDB OLAP

### Current state and unresolved decision

There is no DuckDB product dependency or lowering path. Polylogue's repository and `ArchiveStore` abstractions are SQLite-specific, and query-unit aggregate lowering is owned by SQLite. Named `agg` currently fetches at most 50,001 rows and reduces 50,000 in Python, marking larger results inexact. `polylogue-9l5.7` remains open and owns honest measure composition; `polylogue-4p1` remains open and owns the executable read algebra; `polylogue-5dx` owns evaluated optional dependency policy. A separate audit-tool Bead mentions DuckDB as development tooling, which does not authorize a product dependency.

The operator ratified the decision threshold: below 10x, including the 3x to 10x band, remain SQLite-only. The unresolved work is when to trigger the probe and where a successful lowerer would live.

### Final adjudication

Stay SQLite-only now. Do not run or implement a DuckDB product probe until `9l5.7` defines canonical measures and at least five real analytics workloads exceed the SQLite/Python ceiling.

If triggered, the probe runs both engines from the same engine-neutral `AnalysisPlan` or measure-lowering IR. SQLite remains the default and correctness oracle. DuckDB is an optional `analytics` extra and a read-only lowerer selected only for plans explicitly classified heavy.

The adoption gate is:

1. Exact result parity on seeded and live-snapshot data, including NULLs, integer/float/decimal behavior, timestamps, ordering, collation, percentiles, and empty groups.
2. Five real queries spanning transition/process joins, window/survival analysis, percentile/group aggregation, block scans, and one current named-aggregate ceiling.
3. Median warm wall-time speedup at least 10x, at least four of five queries at least 3x, no query more than 20 percent slower, and peak RSS no more than 2x unless the absolute memory budget is separately approved.
4. Safe concurrency through an immutable SQLite snapshot or a proven WAL-consistent scanner path. Direct attachment to an actively written live file is not assumed safe.

Query ownership stays in insights/product analysis definitions and `9l5.7` lowerer interfaces. Backend adapters emit SQLite or DuckDB SQL. DuckDB never owns public query semantics, canonical identity, persistence, writes, migrations, or source truth.

### Rejected alternatives

- Adopting from generic 10x to 100x database reputation is rejected; Polylogue needs real workload evidence.
- A 3x middle-band adoption is rejected by ratified policy because a second engine has ongoing dependency and parity cost.
- Direct DuckDB calls from CLI/MCP/daemon or storage repository mixins are rejected as architecture leaks.
- Attaching the live WAL database without a concurrency proof is rejected.
- Persisting DuckDB copies or results as a second archive is rejected.
- Using DuckDB to avoid implementing canonical measure semantics is rejected.

### Contract, compatibility, privacy, failure, and rollback

No schema change is authorized now. A successful future adoption adds only an optional dependency group, an engine-neutral lowerer protocol, and a DuckDB adapter. Canonical query/metric refs hash engine-neutral semantics, not selected backend. Receipts record engine, versions, snapshot epoch, plan ref, timing, RSS, and parity hash.

The optional path sees the same private data as SQLite and must use local files, private temp permissions, no extension auto-download, and no network. Missing dependency or adapter failure falls back to SQLite only when the plan remains within its budget; otherwise it returns a typed unavailable/over-budget result. It never silently returns a capped Python sample as exact. Rollback removes the optional adapter and leaves every stored identity/result readable through SQLite.

### Implementation slices

| Order | Slice and owner files | Avoid list | Dependencies and parallel safety |
| --- | --- | --- | --- |
| O1 | `9l5.7` canonical measure identity and engine-neutral lowering IR. | DuckDB dependency and adapter | Hard prerequisite; owns measure semantics. |
| O2 | Select and freeze five production workloads plus correctness fixtures. | Engine-specific optimizations | Depends on analytics consumers and O1. |
| O3 | Read-only DuckDB lab probe, using snapshot and resource receipts. | Runtime product dependency | Evidence-only and optional dev environment. |
| O4 | Decision gate evaluation. Stay SQLite-only or authorize the optional adapter. | Product code before receipt | Coordinator/operator record; no implementation if threshold fails. |
| O5 | If authorized, add `[analytics]`, adapter, planner selection, and one heavy production measure. | Writes, source tiers, public surface branching | Depends on `4p1`, `5dx`, and O4. |

### Acceptance evidence and anti-vacuity

- Query parity compares typed values and deterministic ordering, not only row counts.
- Concurrency tests run a real daemon writer while the probe uses the declared snapshot/scanner method and detect locks, stale reads, or WAL omissions.
- Benchmark runs cold and warm repetitions, records versions and cache state, and reports wall time plus RSS.
- The production demonstration, if adopted, executes the same `AnalysisPlan` through both real lowerers. Removing DuckDB selection must change the engine receipt while preserving results; mutating NULL/order semantics must fail parity.
- Absence of DuckDB must leave all default CLI, MCP, API, daemon, and package tests functional.

Readiness: **dependency-blocked** on `9l5.7`, real analytics workloads, and then `4p1`/`5dx` for any adoption. The current executable decision is to remain SQLite-only.

## Cross-Bead dependency graph and Luna dispatch order

```text
slshy -> xselt -> 818fy -> kea7p K1-K5
                    ^
                    +-- qj5x (excluded content/identity lane)

b5l -----------------------> kea7p planner/prove/activate
  +------------------------> fie blue-green recovery doctrine

ze5 Z1-Z3 --exclusive user schema--> avna A6
        +---------------------------> uh6c U4
37t.12 ------------------------------> uh6c U2 and rxdo R5

4p1 query algebra -----> avna integration
                   +---> ca4 future adapter
9l5.7 -------------> rxdo metric/statistical closure
   +----------------> ca4 probe IR
stc ----------------> rxdo experiment projection

cijx trajectories ---> rxdo G6/G8 evaluation evidence
avna match sets ------> rxdo pattern-derived analyses
83u.5 ----------------> fie unconditional compression
dx1 D1 ---------------> 3utv typed route registry
```

Recommended dispatch waves:

1. Wave 0, blockers and exclusive migrations: `slshy`, then `xselt`; ze5 Z1 through Z3 as the sole user-schema lane; dx1 D1/D2 probe; fie F1/F2 evidence. These have disjoint primary files except global enum edits, which the coordinator must serialize.
2. Wave 1, independent domains: avna A1/A3, cijx C1/C2, uh6c U1, rxdo R1/R2, fie F3/F4, and dx1 D3. Do not let avna and uh6c edit query grammar concurrently. Do not let cijx and kea7p edit index DDL/write concurrently.
3. Wave 2, storage and production wiring: cijx C3 as exclusive index owner, then uh6c U3, then kea7p K2/K3 after bootstrap. Run avna A2/A4/A5 when it owns query files. Run ze5 Z4 and rxdo R3/R5 only with explicit `user_write.py` ownership.
4. Wave 3, dependency consumers: avna A6 after ze5; uh6c U4 after the operator namespace answer and ze5; rxdo R6 after stc; ca4 O1/O2 after 9l5.7; dx1 route-family migration; conditional FTS work only after fie evidence.
5. Wave 4, closures: differential proof and activation, ASGI zero-route legacy deletion, rxdo child reconciliation, scaling doctrine report, and optional DuckDB gate. The coordinator, not implementation lanes, updates or closes Beads.

Hotspot exclusion matrix:

| Hotspot | Sole owner at a time | Lanes that must wait |
| --- | --- | --- |
| `storage/sqlite/archive_tiers/user.py` and user migrations | ze5, then avna A6 or uh6c U4 | rxdo durable definition work |
| `storage/sqlite/archive_tiers/index.py` and `write.py` | xselt, then cijx C3, uh6c U3, kea7p K2 | fie FTS schema work |
| `archive/query/expression.py`, predicate/AST, metadata | avna A1/A2/A5 | uh6c U5, ca4 plan syntax |
| `storage/sqlite/archive_tiers/user_write.py` | ze5 Z3, uh6c U2, rxdo R3/R5 | all other assertion lanes |
| `daemon/cli.py` and HTTP lifecycle | dx1 D3/D6 | kea7p daemon adapter changes |
| `maintenance/rebuild_index.py`, `storage/index_generation.py` | kea7p K3 with b5l coordination | fie rebuild implementation changes |

## Evidence commands used for this adjudication

All commands were read-only except creation of this document and its commit.

```text
jq select(...) .beads/issues.jsonl
rg and sed over query AST, runtime matching, SQLite lowerers, repository identity, file edits, tags, assertions, judgment/measurement modules, daemon routing/lifecycle, tier DDL, rebuild code, optional dependencies, and tests
git log origin/master --oneline, git log -S, and git log --grep over the relevant files and concepts
readlink -f /realm/db/polylogue/index.db
stat on the active index target and archive tier files
sqlite3 -readonly against index.db and source.db for row and byte counts
read-only parsing of .index-rebuild-transactions/*.json
```

The live object-level dbstat census and representative current/3x/10x rebuild benchmarks were not completed here. They are deliberately retained as `fie` evidence work rather than inferred from file size or historical transaction wall windows.
