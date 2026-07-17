---
created: 2026-07-16
purpose: Convert Polylogue bug history into execution-ready generalized test laws
status: active-planning-map
project: polylogue
---

# Bead-derived test laws

This is the missing middle layer between the 181-Bead audit trail in
[`12-bead-regression-class-map.md`](12-bead-regression-class-map.md) and an
implementation dossier. It consolidates bugs by causal mechanism. The result
is 33 proposed laws, not 181 one-bug tests and not 17 class labels with no
executable content.

Each law records:

- the production incidents that justify it;
- an implementation-independent invariant;
- dimensions that must vary rather than remain one fixed example;
- the strongest economical proof form;
- exact historical seeds that should remain easy to diagnose;
- a representative production mutation that demonstrates sensitivity;
- its intended Diet packet or prerequisite.

The Bead threads and current source remain authoritative at dossier time.
These designs do not claim implementation, close Beads, or freeze production
symbols that are still being rewritten.

## A. Evidence authority, identity, and lineage

### L01 — Raw-authority reconciliation reaches a terminal fixed point

- **Bead evidence:** `polylogue-hjpx`, `polylogue-lkrc`, `polylogue-yla8`,
  `polylogue-yla8.1`, `polylogue-yla8.2`, `polylogue-yla8.5`,
  `polylogue-yla8.10`, `polylogue-07hj`, `polylogue-vwsv`.
- **Concrete witness:** ordinary replay repeatedly selected already-terminal
  revisions, grew quarantine/debt, or stopped after planning rather than
  executing accepted work; bundle raws and untyped accepted heads could never
  leave the candidate set.
- **Invariant:** for a finite retained evidence set, reconciliation either
  reaches one stable accepted authority or one stable, explained blocked
  state. A second pass performs zero writes and emits the same terminal facts.
- **Vary:** raw arrival order; full, append, and bundle forms; accepted-head
  presence; typed versus legacy envelopes; parse failure; terminal receipt;
  retry count; empty versus populated derived index.
- **Generalized proof:** a rule-based real-SQLite state machine builds evidence,
  executes the production reconciler to quiescence, and compares candidates,
  heads, receipts, materialized sessions, debt, and the second-pass write set.
- **Retained seeds:** the 391→393→395 candidate-growth incident, fully governed
  bundle raw, untyped accepted ChatGPT head, and parse-failed raw group.
- **Sensitivity mutation:** stop after plan construction; omit terminal receipt
  exclusion; or exclude the accepted head from classification. Each must turn
  quiescence into repeat work or an unjustified authority change.
- **Dossier home:** new authority/replay dossier after the upstream reconciler
  has landed; apply the adjudicated lattice in
  [`architecture/01-evidence-authority-and-identity.md`](architecture/01-evidence-authority-and-identity.md),
  then Terra implements the exact current-symbol packet.

### L02 — Equivalent evidence histories choose the same authority

- **Bead evidence:** `polylogue-25vy`, `polylogue-lkrc.1`,
  `polylogue-lkrc.2`, `polylogue-lkrc.3`, `polylogue-lkrc.4`,
  `polylogue-rgh2`, `polylogue-yla8.4`, `polylogue-yla8.9`,
  `polylogue-yla8.6`, `polylogue-57rp`, `polylogue-t0dy`,
  `polylogue-0mu`.
- **Concrete witness:** legacy browser heads, accepted semantic heads, append
  chains, and newer DOM/GDPR snapshots produced different winners depending on
  acquisition path or replacement order; timestamp-only newest-wins could let
  a poorer DOM capture overwrite richer material.
- **Invariant:** histories that contain equivalent evidence and causal order
  converge to the same accepted semantic content, frontier, and explanation,
  regardless of ingestion batching or representation. Less authoritative or
  less complete evidence cannot replace a proven richer head.
- **Vary:** every topological permutation of prefix-compatible revisions;
  full-versus-split bytes; semantic-versus-byte frontier; legacy identity;
  equal timestamps; richer/poorer content; multi-session divergence.
- **Generalized proof:** generate a small evidence DAG, replay several valid
  linearizations into cloned archives, reconcile, and compare accepted content
  facts, frontier reachability, receipts, and rejected-candidate reasons.
- **Retained seeds:** byte-identical full snapshot folding a multi-append chain,
  same-path richer browser reacquisition, and equal-time DOM fallback versus
  native/GDPR content.
- **Sensitivity mutation:** restore last-writer-wins, compare timestamps alone,
  or treat semantic and byte frontiers as always incomparable.
- **Dossier home:** authority/replay; share archive builders with L01 but keep
  the independent permutation oracle.

### L03 — Acquisition identity is deterministic and content-semantic

- **Bead evidence:** `polylogue-sjf6`, `polylogue-fmob`, `polylogue-nkmy`,
  `polylogue-8k91`, `polylogue-tbe5`.
- **Concrete witness:** the same Claude resume/fork bytes acquired twice could
  derive different native IDs; receipt replay keyed the wrong semantic
  identity; split tier roots admitted two writable indexes; stable captured
  refs resolved through one surface but not root find.
- **Invariant:** the same logical source and bytes yield one stable identity
  across acquisition, receipts, storage tiers, coordination envelopes, and
  public reference resolution. Distinct active archives cannot masquerade as
  one identity.
- **Vary:** filename UUID versus embedded session ID; symlinked tier layouts;
  provider aliases; repeated acquisition; process restart; captured URL/native
  ref; receipt serialization round-trip.
- **Generalized proof:** metamorphic identity round-trips through actual
  acquisition and reference APIs, with path/alias transformations that must
  preserve identity and deliberate split-brain layouts that must fail closed.
- **Retained seeds:** Claude filename/content-ID mismatch and the live split
  `/home` versus `/realm` index configuration.
- **Sensitivity mutation:** reintroduce random/filename-first extraction, drop
  one tier from archive identity, or route root find only through lexical FTS.
- **Dossier home:** authority foundation plus status/reference surface checks.

### L04 — Relational projections conserve logical cardinality

- **Bead evidence:** `polylogue-xnkf`, `polylogue-tilk`,
  `polylogue-f2qv.1`, `polylogue-85z0`, `polylogue-4ts.2`,
  `polylogue-y964`.
- **Concrete witness:** duplicate tool IDs created an N×M actions fanout;
  inconsistent upsert identities alternated between append and update; model
  rollups assigned the full session total to every partition; lineage replay
  and run-ref collisions double-counted logical work.
- **Invariant:** each logical relation has an explicit cardinality and identity
  rule. Joins do not multiply facts, partitions sum to the whole exactly, and
  replayed/inherited material is counted once at the declared grain.
- **Vary:** zero/one/many duplicate or missing IDs; orphan and erroneous tool
  results; two or more models; fork/resume chains; repeated upsert; child/main
  run-ref collision; rebuild order.
- **Generalized proof:** plant independent relation facts, derive actions,
  delegations, runs, assertions, and rollups through production queries, then
  assert exact multiset equality and partition conservation.
- **Retained seeds:** duplicate tool-use/tool-result N×M fixture, two-model
  session, and subagent run colliding with the child main run.
- **Sensitivity mutation:** remove a join disambiguator, group by physical
  session, assign total to every group, or switch stable update to content-hash
  append.
- **Dossier home:** query composition with secondary storage/lineage ownership.

### L05 — Lineage histories compose independently of arrival order

- **Bead evidence:** `polylogue-866e`, `polylogue-9p0y`,
  `polylogue-4ts.3`, `polylogue-4ts.6`, `polylogue-5q2u`.
- **Concrete witness:** child-before-parent ingestion, parent replacement, and
  ambiguous compaction could leave dangling branch points, amplify deferred
  tails during rebuild, misclassify subagent compaction, or silently truncate
  composed transcripts.
- **Invariant:** equivalent lineage graphs yield the same divergent tails,
  branch points, logical transcript, topology, completeness signal, and
  accounting regardless of ingest/replacement order. Incomplete composition is
  explicit, never silently shortened.
- **Vary:** parent/child arrival permutations; parent full replacement;
  prefix-sharing versus spawned-fresh; main/subagent compaction; dangling
  parent; depth boundary; cycle quarantine; rebuild replay order.
- **Generalized proof:** Hypothesis rule-based lineage state machine exercising
  production writes and composed reads, with an independent in-memory graph
  fact oracle limited to public transcript/topology semantics.
- **Retained seeds:** dangling prefix-sharing branch point, subagent acompact
  misclassification, and depth-greater-than-64 incomplete transcript.
- **Sensitivity mutation:** restore arrival-order-dependent branch resolution,
  replay raws in arbitrary order, or remove the completeness flag.
- **Dossier home:** dedicated architecture-heavy lineage packet; Sol designs,
  Terra executes. Do not hide it inside storage equivalence.

### L06 — Composed reads observe one lineage snapshot

- **Bead evidence:** `polylogue-4ts.4`, `polylogue-41ow` as the analogous
  stale-read/write witness.
- **Concrete witness:** recursive composition issued multiple autocommit reads,
  so a parent replacement between edge and message reads could produce a torn
  transcript.
- **Invariant:** one public composed read observes one coherent archive
  snapshot; it returns the old or new lineage, never a mixture.
- **Vary:** replacement at each recursion boundary; sync/async path; batch and
  paginated read; lineage depth; WAL versus rollback journal where supported.
- **Generalized proof:** deterministic barrier schedules pause the production
  reader at edge/message boundaries while another connection replaces the
  parent, then compare with valid old/new fact sets.
- **Retained seeds:** parent replacement between branch-edge lookup and
  recursive parent message load, at both shallow and multi-level recursion.
- **Sensitivity mutation:** remove the encompassing deferred read transaction.
- **Dossier home:** lineage packet, sharing deterministic scheduling utilities
  with concurrency laws rather than bespoke sleeps.

## B. Atomicity, recovery, rebuilds, and convergence

### L07 — Read-modify-write operations are linearizable or conflict explicitly

- **Bead evidence:** `polylogue-41ow`, `polylogue-hleq`,
  `polylogue-qug2`, `polylogue-y337`, `polylogue-n2wy`.
- **Concrete witness:** concurrent assertion upserts silently reverted operator
  judgment; cursor failure increments were lost across separate get/set
  transactions; embedding success cleared a newer reindex generation; watcher
  and maintenance writers could overlap.
- **Invariant:** every shared-key mutation has a linearization point. Concurrent
  successful operations compose according to some serial order, or a stale
  writer receives an explicit conflict; monotonic counters/generations never
  move backward.
- **Vary:** all two-operation schedules for insert/update/delete, judgment
  change, counter increment/reset, embed-success/reindex, and watcher/repair;
  existing versus absent row; sync versus async writer.
- **Generalized proof:** reusable deterministic transaction barriers drive a
  small schedule matrix and compare durable state plus receipts to enumerated
  legal serial outcomes.
- **Retained seeds:** operator judgment revert, two simultaneous
  `mark_failed` increments, and config-generation mark racing embed success.
- **Sensitivity mutation:** split atomic SQL into get-then-set, remove the
  generation predicate, or remove the shared writer lock.
- **Dossier home:** new concurrency dossier; Terra high reasoning.

### L08 — Leases, reservations, and checkpoints survive adversarial schedules

- **Bead evidence:** `polylogue-8jg9.4`, `polylogue-v7e0`,
  `polylogue-mpig`, `polylogue-0puw`, `polylogue-qs0a`.
- **Concrete witness:** orphan cleanup could delete an in-flight leased blob;
  the advertised lease payload was never populated; ingest left publication
  reservations behind; stale checkpoint writes could overwrite newer state or
  leave UI state permanently pending.
- **Invariant:** a live lease/reservation protects its resource; every terminal
  operation drains or explains it; expiry is monotonic; stale checkpoints
  cannot supersede newer progress.
- **Vary:** reserve/publish/cleanup/crash order; lease present, missing, expired,
  or dead-coded; retry; stale/new checkpoint sequence; callback failure.
- **Generalized proof:** resource-lifecycle state machine with a reference set
  of protected and collectible blobs plus monotonic checkpoint sequence facts.
- **Retained seeds:** cleanup during the acquire-blob→commit-row window,
  completed batch with surviving reservation, and badge stuck pending after
  `onSave` failure.
- **Sensitivity mutation:** ignore leases during GC, omit reservation drain, or
  accept checkpoint writes without a generation comparison.
- **Dossier home:** storage durability with browser checkpoint seed retained as
  a surface-specific example.

### L09 — Multi-file publication is old-or-new across filesystem boundaries

- **Bead evidence:** `polylogue-7ufv`, `polylogue-b08j`,
  `polylogue-rze2`, `polylogue-8jg9.5`.
- **Concrete witness:** rename-based promotion failed across subvolumes;
  post-promotion WAL evidence failure left new tiers with a prepared receipt;
  rollback and migration activation were not bound to verified backup
  receipts.
- **Invariant:** after any publication failpoint and reopen, all tiers and the
  receipt describe one valid old version or one valid new version. Cross-device
  copying, WAL/SHM cleanup, and backup verification cannot create a mixed
  archive.
- **Vary:** same versus different device; failure after copy, fsync, promote,
  checkpoint, evidence, or receipt transition; durable versus rebuildable tier;
  WAL sidecars; retry and rollback.
- **Generalized proof:** subprocess failpoint matrix using actual archive files
  on two mount identities when available, followed by cold reopen and version,
  content, sidecar, receipt, and rollback assertions.
- **Retained seeds:** v35→v36 final-evidence failure and cross-subvolume reused
  index clone.
- **Sensitivity mutation:** replace copy-forward with rename, finalize receipt
  before checkpoint/evidence, or skip backup digest verification.
- **Dossier home:** storage durability; `verify --all` because the harness uses
  filesystem and migration surfaces.

### L10 — Interrupted work resumes exactly without duplicate or lost effects

- **Bead evidence:** `polylogue-kwlu`, `polylogue-b5l.1`,
  `polylogue-1xc.1`, `polylogue-1xc.4`, `polylogue-r3o3`.
- **Concrete witness:** raw parse state could commit before index durability;
  rebuild ownership/resume was ambiguous; one transaction caused WAL blowup;
  partial insight rebuild could become permanently failed; generated demo
  artifacts ignored repository closure.
- **Invariant:** a crash after any committed unit leaves a durable resume point.
  Restart executes each remaining logical unit once, preserves completed work,
  and produces the same final facts as uninterrupted execution.
- **Vary:** interruption before/after each commit and receipt; batch size;
  repository closure change; already-complete unit; retry count; sync/async
  operation.
- **Generalized proof:** process-level kill/failpoint campaign over a small
  multi-unit workload, compare resumed versus uninterrupted archive and exact
  unit receipts, and assert bounded transaction size.
- **Retained seeds:** raw/index commit inversion and crash after K insight
  chunks.
- **Sensitivity mutation:** advance resume cursor before durable write, clear
  partial progress on restart, or wrap all units in one transaction.
- **Dossier home:** seeded artifact integrity plus storage/convergence packet.

### L11 — Incremental, full, targeted, sync, and async derivations agree

- **Bead evidence:** `polylogue-61zb`, `polylogue-oucx`,
  `polylogue-y964`, `polylogue-1xc.2`, `polylogue-lyv4`,
  `polylogue-a7xr.2`.
- **Concrete witness:** refresh skipped the rebuild heavy-session threshold;
  cache removal dropped run/OTel units; async targeted insight rebuild wiped the
  whole archive; converger and repair disagreed for NULL sort keys; reset could
  lose sessions whose source files rotated away.
- **Invariant:** given the same durable evidence, every supported derivation
  route produces the same scoped public facts. Targeted work changes only the
  requested scope; retained durable source is sufficient for full recovery.
- **Vary:** incremental/full, refresh/rebuild/repair, sync/async, one/many
  targets, NULL/large session, rotated source path, derived cache present or
  absent.
- **Generalized proof:** clone one planted archive, execute alternate production
  routes, compare exact public fact sets and untouched-scope sentinels, then
  reopen both archives.
- **Retained seeds:** targeted async archive-wide wipe, NULL-sort-key staleness
  disagreement, and reset with a rotated-source session.
- **Sensitivity mutation:** remove target predicate, restore the differing
  staleness expression, or omit a query-unit derivation from one route.
- **Dossier home:** prepared incremental-versus-rebuild dossier, expanded to
  cover scope and route variants.

### L12 — Convergence debt forms a live, resumable state machine

- **Bead evidence:** `polylogue-b5l.2`, `polylogue-1xc.3`,
  `polylogue-1xc.11`, `polylogue-5vbs`, `polylogue-x1uh`,
  `polylogue-n846`, `polylogue-egm8`, `polylogue-09rn`.
- **Concrete witness:** readiness could go green while read models were stale;
  orphan raws had no automatic feeder; errors were treated as converged; FTS
  session debt had no live feeder; poisoned embedding rows retried forever;
  backlog-wait tests hung without terminal explanation.
- **Invariant:** every incomplete unit is pending, retryable, or terminal with
  an inspectable reason; every pending class has a production feeder; restart
  preserves debt; bounded passes reach quiescence or stable terminal debt.
- **Vary:** each stage and unit; transient/permanent error; false-means-pending;
  restart; poisoned row; missing feeder; batch boundary; quiet deferral;
  already-current unit.
- **Generalized proof:** rule-based convergence state machine over real ops and
  index databases, including restart, with per-unit receipts and a liveness
  bound on passes rather than wall-clock sleeps.
- **Retained seeds:** FTS debt with no feeder, provider HTTP 400 embedding row,
  orphan raw materialization, and catch-up completion hang.
- **Sensitivity mutation:** return converged on exception, unregister a feeder,
  omit terminal classification, or drop `false_means_pending`.
- **Dossier home:** prepared daemon convergence dossier.

### L13 — Derived freshness is one monotonic content-and-recipe invariant

- **Bead evidence:** `polylogue-wmsc`, `polylogue-1xc.12`,
  `polylogue-f2qv.5`, `polylogue-1dk1`, `polylogue-1ty`,
  `polylogue-w379`.
- **Concrete witness:** embedding success could hide a newer recipe change;
  FTS rowid reuse made counts look coherent while identity drifted; usage
  projections lacked version self-healing; orphan embeddings survived index
  generations; freshness DDL/hash declarations existed without one authority
  or reader.
- **Invariant:** a derived row is current iff both source content identity and
  derivation recipe/generation match. Freshness never moves backward, counts
  alone cannot prove identity, and one canonical definition drives all paths.
- **Vary:** content edit/delete/reinsert with rowid reuse; recipe/version bump;
  generation rebuild; unchanged content; concurrent success; orphan row;
  rollback.
- **Generalized proof:** mutation-driven state machine through actual triggers
  and materializers, checking identity sets, generations, gauges, and public
  freshness after every operation.
- **Retained seeds:** delete/reinsert reusing FTS rowid and embedding success
  racing a recipe-generation change.
- **Sensitivity mutation:** compare row counts only, omit recipe identity, or
  keep one stale duplicate freshness definition.
- **Dossier home:** storage/convergence, reusing the proposed FTS Hypothesis
  state machine rather than adding a parallel catalog.

## C. Query, evidence, and public projections

### L14 — Query algebra conserves membership across equivalent forms

- **Bead evidence:** `polylogue-z9gh.2`, `polylogue-1vv`,
  `polylogue-20d.4`, `polylogue-kixp`, `polylogue-tsk`.
- **Concrete witness:** scoped aggregates silently capped totals; structured
  CLI queries incorrectly required FTS readiness; action/delegation queries
  materialized the archive; canonical origins were rewritten; ranking referred
  to classifier labels that could never be emitted.
- **Invariant:** equivalent expressions and routes select the same multiset;
  page concatenation equals the unpaged result; grouped partitions conserve
  totals; structured-only queries do not depend on unrelated FTS state; every
  filter/ranking value is in the production vocabulary.
- **Vary:** boolean rewrites; structured versus FTS; unit/projection; page size;
  limit above/below result size; origin enum; duplicate relations; irrelevant
  archive growth.
- **Generalized proof:** independent planted fact algebra exercised through the
  real parser/lowering/executor and stable public routes, with metamorphic query
  rewrites and page/partition conservation.
- **Retained seeds:** aggregate over more than 1,000 scoped rows, structured-only
  query with stale FTS, and `grok-export` tool-usage filter.
- **Sensitivity mutation:** clamp before aggregation, route all queries through
  FTS, remove one predicate, or reintroduce a dead vocabulary label.
- **Dossier home:** prepared exact-query C-03 dossier, broadened beyond exact
  session selection.

### L15 — Query work is bounded, interruptible, and input-depth safe

- **Bead evidence:** `polylogue-z9gh.1`, `polylogue-u0dm`,
  `polylogue-rsad`.
- **Concrete witness:** archive queries lacked reusable deadlines/cancellation
  and could materialize huge results; deeply nested DSL input reached a Python
  `RecursionError`; oversized MCP results became unrecoverable budget envelopes.
- **Invariant:** admitted work has explicit depth, row, byte, time, and memory
  bounds; cancellation reaches SQLite and cleanup; oversized successful results
  remain losslessly pageable or addressable rather than becoming dead ends.
- **Vary:** nesting around/beyond limit; cancellation before/during SQL and
  rendering; result cardinality/row width; cursor continuation; concurrent
  admission; client disconnect.
- **Generalized proof:** bounded adversarial expression generator plus a
  cancellable production-query harness that records SQLite progress, peak
  materialization, response pages, and cleanup.
- **Retained seeds:** transformer recursion overflow and successful query whose
  MCP serialization exceeded the response budget.
- **Sensitivity mutation:** remove pre-parse depth guard, disconnect the SQLite
  interrupt hook, or serialize the complete result before paging.
- **Dossier home:** query/work dossier; architecture decisions in `z9gh.1` must
  land before worker dispatch.

### L16 — Performance scales with selected work, not archive size

- **Bead evidence:** `polylogue-cnaj`, `polylogue-w79`,
  `polylogue-rgbj`, `polylogue-ng9m`, `polylogue-dlmv`,
  `polylogue-xy95`, `polylogue-3wb`, `polylogue-qhk`,
  `polylogue-35d`, `polylogue-20d.17`, `polylogue-s7ae.8`,
  `polylogue-zdeo`, `polylogue-1xc.6`, `polylogue-k8k`.
- **Concrete witness:** active append ingestion retained growing files; graph
  resolution and replacement scanned large relations; status and usage reads
  consumed gigabytes or tens of seconds; giant-session insights took minutes;
  wrapper overhead dominated cheap reads.
- **Invariant:** the work metric for a bounded request is a function of selected
  rows/edges/bytes and declared diagnostic mode, not total archive size.
  Streaming paths remain memory bounded and status stays interactive.
- **Vary:** fixed selected set under irrelevant archive growth; active-file
  length; graph fanout; giant row; cold/warm cache; wrapper/native execution;
  compact/full/exact mode; batch size.
- **Generalized proof:** named workload tiers with identical selected facts and
  increasing irrelevant material, recording VM steps or rows visited, bytes
  read, peak RSS, fixture receipts, and result equality.
- **Retained seeds:** 11.1-GB status read, 615-second graph resolution tail,
  90-second full usage scan, and giant-session nine-minute batch.
- **Sensitivity mutation:** remove an index/predicate, restore archive-wide
  Python materialization, or compute compact status by collecting the full
  diagnostic first.
- **Dossier home:** scale/work consolidation using realized workload-profile
  tiers and receipts.

### L17 — Progress and cleanup reflect logical work, not incidental output

- **Bead evidence:** `polylogue-27rb`, `polylogue-88jp.1`,
  `polylogue-a7xr.1`, `polylogue-peo`.
- **Concrete witness:** testmon/xdist workers entered D-state while output bytes
  kept the supervisor apparently alive; SQLite context managers committed but
  leaked connections; daemon exits lacked workload/host attribution.
- **Invariant:** liveness is measured by completed logical units/events;
  terminal success, failure, timeout, cancellation, and crash close owned
  resources and emit attributable receipts.
- **Vary:** master chatter with hung worker; no output with real test progress;
  normal/error/cancel close; owned/borrowed executor; connection warning;
  process exit under host pressure versus workload failure.
- **Generalized proof:** synthetic event-emitting/hanging workers and
  ResourceWarning-as-error lifecycle scenarios, with exact process, lock,
  executor, connection, and attribution receipts.
- **Retained seeds:** master emits while test never finishes and partially
  initialized daemon server close.
- **Sensitivity mutation:** key stall detection to output bytes, remove one
  close/finally path, or omit the last logical-work timestamp.
- **Dossier home:** verification tooling plus daemon service lifecycle.

### L18 — Removing evidence cannot increase certainty or completeness

- **Bead evidence:** `polylogue-7ry`, `polylogue-b2r9`,
  `polylogue-4bu`, `polylogue-z9gh.6`, `polylogue-egm8`,
  `polylogue-4iv`, `polylogue-07hj`.
- **Concrete witness:** partial rebuilds reported ready; unknown embedding state
  collapsed into counts; some surfaces returned zero while others said
  converging; missing sidecars or parse-failed raws disappeared from coverage;
  terminal failures were aggregated without inspectable identities.
- **Invariant:** certainty and completeness are monotone in available evidence.
  Removing or corrupting evidence can move exact→partial→unknown, never toward
  ready/exact/zero. Every non-complete state identifies missing or terminal
  obligations.
- **Vary:** raw/materialized/FTS ratios; absent/corrupt/ignored sidecars;
  unknown/approximate/terminal embedding state; parse failures; surface; empty
  versus genuinely zero data.
- **Generalized proof:** generate an evidence lattice, project each state
  through status/query/diagnostic surfaces, and assert monotonic truth ordering
  plus stable object references for debt.
- **Retained seeds:** mid-rebuild bare “No sessions matched,” missing Workflow
  journal, and terminal provider error with no actionable identity.
- **Sensitivity mutation:** coalesce unknown to zero, calculate readiness from
  the materialized subset alone, or drop one evidence class from diagnostics.
- **Dossier home:** status/facades fact algebra with source coverage inputs.

### L19 — Quantitative claims conserve totals and weakest provenance

- **Bead evidence:** `polylogue-f2qv.6`, `polylogue-9e5.28`,
  `polylogue-9e5.29`, `polylogue-9e5.30`, `polylogue-cpf.5`,
  `polylogue-rvtu`.
- **Concrete witness:** profiles/costs disagreed with provider usage;
  number-bearing products vanished from a contract-only audit; quantitative
  fields lacked field-level evidence; prose-mined fields lost `text_derived`;
  aggregates laundered weak timestamps; timeless usage disappeared.
- **Invariant:** every reported number is a conserved partition of a declared
  population with field-level evidence and the weakest contributing
  provenance. Unknown/timeless inputs remain visible rather than silently
  excluded or upgraded.
- **Vary:** provider-reported versus estimated usage; cached/input/output lanes;
  missing fields; mixed provenance; timeless rows; aggregation grain; newly
  registered number-bearing product.
- **Generalized proof:** plant exact disjoint usage/evidence facts, derive every
  registered quantitative product, assert partition sums, provenance meet,
  unknown buckets, and provider reconciliation.
- **Retained seeds:** Codex cached-token semantics, timeless usage row, and a
  prose-derived forensic field entering an aggregate.
- **Sensitivity mutation:** drop the timeless bucket, choose strongest instead
  of weakest provenance, double-count a partition, or iterate a hand-written
  contract subset rather than the product registry.
- **Dossier home:** evidence-honesty packet before analytics/public claims.

### L20 — Stable surfaces project one fact algebra

- **Bead evidence:** `polylogue-rsad`, `polylogue-s7ae.7`,
  `polylogue-g9j6`, `polylogue-4pm`, `polylogue-bby.7`,
  `polylogue-6o9b`, `polylogue-vh57`, `polylogue-f57q`.
- **Concrete witness:** registered HTTP route lacked a handler; web list refs
  404ed at detail; read could return zero content; DB-backed and archive-backed
  routes flattened messages differently; advertised text format failed;
  preview/apply reported dishonest phases.
- **Invariant:** for a stable public operation, repository, CLI, daemon HTTP,
  and supported adapters expose the same selected object, semantic facts,
  completeness, phase, and typed error. Presentation and budgets may differ
  without changing those facts.
- **Vary:** surface; DB/archive backend; compact/detail/text/JSON; budget below
  and above content size; missing ref; pending/converged state; preview/apply
  success/failure.
- **Generalized proof:** one planted state-transition table projected through
  real stable routes and normalized to an independent fact algebra.
- **Retained seeds:** provider-usage route crash, list-to-detail 404, zero-segment
  read, and divergent flattened `message.text`.
- **Sensitivity mutation:** bypass the shared product route, delete a handler,
  use a surface-local flattener, or mark preview work as applied.
- **Dossier home:** status/facades and query packets. Current MCP/web reader
  cases become rewrite obligations where implementation is being replaced.

## D. Source fidelity, security, configuration, and temporal truth

### L21 — Detector dispatch chooses the tightest valid parser

- **Bead evidence:** `polylogue-yla8.3`, `polylogue-segf`,
  `polylogue-fs1.1`, `polylogue-qda`.
- **Concrete witness:** mutable JSON was incorrectly treated as append JSONL;
  a Hermes detector recognized only a synthetic marker; schema/WAL variation
  threatened reproducibility; image-only nodes and non-UTF-8 bodies were
  silently lost.
- **Invariant:** each artifact is claimed by the tightest parser whose real
  structural contract it satisfies; unsupported or partial material is
  classified explicitly, and every meaningful structural element survives
  normalization even without prose text.
- **Vary:** ambiguous outer shapes; `.json` versus `.jsonl`; drive/bundle
  nesting; WAL/schema version; image-only/asset-only nodes; encoding errors;
  empty text with structural blocks.
- **Generalized proof:** provider-family semantic blueprint with curated real
  wire witnesses, ambiguity negatives, detector-order permutations, and
  independent normalized fact assertions.
- **Retained seeds:** real Hermes ATIF/ATOF sample once available, ChatGPT
  image-only node, Antigravity non-UTF-8 body, and inbox browser JSON.
- **Sensitivity mutation:** loosen an earlier detector, key append behavior to
  source name alone, skip nodes before building structural blocks, or decode
  strictly without classified fallback.
- **Dossier home:** source normalization. Real-wire examples remain even after
  generalized laws because they are compatibility evidence.

### L22 — Normalization preserves authoredness, commands, outcomes, and time

- **Bead evidence:** `polylogue-z9gh.5`, `polylogue-j2zz`,
  `polylogue-ih67`, `polylogue-1frn`, `polylogue-t0p.1`,
  `polylogue-83u.1`, `polylogue-kixp`, `polylogue-g99u`,
  `polylogue-a7xr.3`, `polylogue-tf0e`.
- **Concrete witness:** generated subagent instructions became human-authored;
  nested Codex child calls and `cmd` fields disappeared from actions; title
  enrichment ignored authored history; background outcomes, attachments,
  origins, ordered prose, and available timestamps were dropped or rewritten.
- **Invariant:** normalization preserves provider-independent semantic facts:
  authoredness, action command/result relationships, structural attachments,
  event order, origin, and available temporal fields. Equivalent provider
  encodings yield equivalent facts.
- **Vary:** nested/flat command keys; generated/runtime/human material origins;
  child calls and outcomes; text/tool block insertion order; provider aliases;
  attachments; optional timestamps and title sources.
- **Generalized proof:** schema-driven provider variants mapped to an
  independent fact blueprint, plus round-trip query assertions for commands,
  authored history, outcomes, attachments, origin, and dates.
- **Retained seeds:** Codex `functions.exec` nested `cmd`, Claude background
  completion, out-of-order inserted text blocks, and generic messages with
  created/updated timestamps.
- **Sensitivity mutation:** classify by role only, ignore nested command keys,
  omit block ordering/filtering, or rewrite an already-canonical origin.
- **Dossier home:** source normalization followed by one query projection.

### L23 — Every write route passes the same destructive/excision contract

- **Bead evidence:** `polylogue-layg`, `polylogue-jnj.5`,
  `polylogue-jn40`, `polylogue-tilk`.
- **Concrete witness:** a second write chokepoint bypassed excision; reset paths
  bypassed the mutation contract; destructive MCP tools lacked confirmation;
  assertion upsert semantics were inconsistent.
- **Invariant:** every public and internal route capable of changing protected
  state passes one authorization/excision/confirmation/identity decision and
  emits an auditable mutation receipt. No alternate adapter can bypass it.
- **Vary:** CLI/MCP/API/daemon/direct maintenance; create/update/delete/reset;
  confirmed/unconfirmed; protected/suppressed material; stable/content identity;
  dry-run/apply.
- **Generalized proof:** enumerate production write entry points from dispatch,
  execute an adversarial operation matrix, and assert identical contract
  decisions and durable receipts—not a textual allowlist of function names.
- **Retained seeds:** the exact second-chokepoint excision bypass and an
  unconfirmed destructive MCP call.
- **Sensitivity mutation:** route one adapter directly to storage, bypass the
  mutation contract, or default confirmation to true.
- **Dossier home:** new security-boundary packet; keep destructive behavior out
  of generic surface snapshot tests.

### L24 — Authentication remains fail-closed across transport lifecycles

- **Bead evidence:** `polylogue-jlme.2`, `polylogue-jlme.3`,
  `polylogue-jlme.5`, `polylogue-5k5l.1`, `polylogue-6jjv`,
  `polylogue-gnie`, `polylogue-kwsb.1`.
- **Concrete witness:** browser backfills lost first-party auth; stale receiver
  contracts failed invisibly; pairing lacked stable runtime identity;
  interpreter assets were classified expired without authenticated fetch;
  fetch/SSE credentials diverged; receiver defaulted unauthenticated.
- **Invariant:** every transport phase is bound to one stable receiver identity
  and authenticated session; missing, stale, cross-origin, or mismatched
  credentials fail closed with a visible recoverable state.
- **Vary:** initial pair/restart/reconnect; fetch/SSE/POST/assets; Host/Origin;
  missing/wrong/rotated token; stale contract version; offline queue; first-party
  cookie available/unavailable.
- **Generalized proof:** production-valid receiver/extension lifecycle scenario
  with credential and identity fault injection at every transition.
- **Retained seeds:** 160 state GETs with zero capture POSTs, stale receiver
  contract, unauthenticated interpreter asset, and fetch-auth/SSE mismatch.
- **Sensitivity mutation:** remove one Host/Origin/token check, regenerate
  receiver identity on restart, or classify 401/403 as expired content.
- **Dossier home:** installed capture lifecycle after production-valid service
  profiles exist.

### L25 — Output encoding and generated schemas never expose attacker content

- **Bead evidence:** `polylogue-2n39`, `polylogue-1xc.14.1.2`.
- **Concrete witness:** attachment metadata and inline handler attributes were
  escaped for the wrong context; observed content values could become schema
  property names and therefore leak sensitive material into promotable
  artifacts.
- **Invariant:** untrusted values remain data, never syntax, identifiers, or
  schema keys. Each output context applies its own correct encoding, and
  promotable workload/schema artifacts contain only approved structural
  vocabulary plus anonymous distributions.
- **Vary:** HTML text/attribute/JS-string contexts; quotes, slashes, Unicode,
  control characters, and closing tags; path/repository/model/tool/error/content
  values; high-cardinality and adversarial keys.
- **Generalized proof:** property strategies generate adversarial strings
  through actual renderers and workload-profile serialization; parse the output
  with the target grammar and inspect artifact keys for structural provenance.
- **Retained seeds:** stored attachment metadata closing an inline handler and
  a content-derived field name containing a sensitive value.
- **Sensitivity mutation:** substitute the HTML-text escaper for an attribute or
  JS context, or use observed mapping keys as schema properties.
- **Dossier home:** security/privacy packet; Luna may implement only after exact
  sinks and artifact contract are adjudicated.

### L26 — Layered configuration obeys composition and path-coherence laws

- **Bead evidence:** `polylogue-fd2s`, `polylogue-9itr`,
  `polylogue-cxlk`, `polylogue-nj80`, `polylogue-rzve`,
  `polylogue-nkmy`.
- **Concrete witness:** TOML `archive_root` and Voyage key were inventoried but
  bypassed by real consumers; nested health tables replaced rather than merged;
  split-tier readiness used inconsistent roots; help promised token generation
  that runtime did not implement.
- **Invariant:** configuration layers have one documented precedence and deep
  composition rule; all consumers observe the same resolved typed value and
  archive identity; help/default claims are executable consequences of that
  resolver.
- **Vary:** defaults/TOML/env/CLI; absent/partial nested tables; empty versus
  explicit values; split/symlinked tier paths; secret sources; documented
  auto-generation on/off.
- **Generalized proof:** metamorphic configuration matrix resolves through the
  production loader and representative real consumers, asserting override,
  preservation of unrelated nested keys, path identity, and help/runtime truth.
- **Retained seeds:** TOML-only `archive_root`, TOML-only Voyage key, and partial
  `[health]` override dropping sibling thresholds.
- **Sensitivity mutation:** make one consumer call a bypass resolver, restore
  shallow dictionary replacement, or advertise a default absent at runtime.
- **Dossier home:** new configuration-composition packet.

### L27 — Inventories and catalogs are executable single authorities

- **Bead evidence:** `polylogue-71ey`, `polylogue-ihp0`,
  `polylogue-w379`, `polylogue-iyew`, `polylogue-mhx.7`,
  `polylogue-1ty`, `polylogue-9e5.28`, `polylogue-tsk`,
  `polylogue-l8ee`, `polylogue-vt0m`, `polylogue-gxly`,
  `polylogue-at44`, `polylogue-j9dt`.
- **Concrete witness:** catalogs advertised operations but did not own
  execution; inventory probes named nonexistent tables; duplicate DDL and
  freshness definitions drifted; a stored hash was never read; audits iterated
  hand-written contracts; tests asserted stale table/tool sets; dead settings
  DDL and stale workflow docs survived.
- **Invariant:** one production authority enumerates a capability and dispatches
  it. Fresh and upgraded state agree with that authority; every advertised item
  is executable or explicitly unsupported; no write-only declaration claims
  validation it never performs.
- **Vary:** fresh/upgraded/rebuilt archive; add/remove registered operation,
  tool, table, workflow, ranking label, or quantitative product; direct bypass;
  stale generated docs.
- **Generalized proof:** iterate the production registry/inventory through real
  dispatch and schema introspection, compare fresh versus upgraded behavior,
  and prove that unregistered direct entry points cannot create a second path.
- **Retained seeds:** nonexistent `artifact_observations`, duplicate vec0 DDL,
  three MCP tools missing minimal invocation, and dead `user_settings` table.
- **Sensitivity mutation:** add a registry item without a handler, bypass the
  catalog, reintroduce duplicate DDL, or compute-but-never-compare a hash.
- **Dossier home:** devtools verification subtraction plus storage/config
  packets. Do not create a third mirror catalog to test the first two.

### L28 — Equivalent instants have identical temporal behavior

- **Bead evidence:** `polylogue-cpf.6`, `polylogue-rvtu`,
  `polylogue-z29t`, `polylogue-2kvn`, `polylogue-2seq`,
  `polylogue-s5mm`, `polylogue-a7xr.6`, `polylogue-tf0e`.
- **Concrete witness:** epoch-zero fallbacks changed ordering/windowing;
  timeless usage vanished; six datetime parsers disagreed on naive/aware
  semantics; raw `parsed_at` and parsed writes diverged; generic parser dropped
  available timestamps; relative parsing depended on wall clock.
- **Invariant:** representations of the same instant compare, filter, group,
  and paginate identically. Missing/timeless is a separate explicit state, not
  epoch zero. Relative time uses the injected clock; stored parse timestamps
  obey one contract.
- **Vary:** UTC/offset/naive forms; epoch zero; NULL primary with secondary
  fallback; timeless; DST boundary; relative expressions; created/updated/
  occurred/sort/parsed fields; page boundaries.
- **Generalized proof:** metamorphic instant generator with `frozen_clock`,
  exercised through canonical parsing, SQL predicates, ordering, windows,
  pagination, and reader projections.
- **Retained seeds:** timeless usage row, NULL `occurred_at_ms` with valid
  fallback, and generic messages carrying created/updated timestamps.
- **Sensitivity mutation:** coalesce missing to zero, reintroduce one divergent
  parser, use host wall time, or omit the secondary sort key.
- **Dossier home:** dedicated temporal packet after central predicate/parser
  authority settles.

## E. Installed runtime and verification substrate

### L29 — Capture delivery is a recoverable installed-runtime state machine

- **Bead evidence:** `polylogue-jlme.6`, `polylogue-s2x7`,
  `polylogue-r4no`, `polylogue-7s57.1`, `polylogue-qvgt`,
  `polylogue-enj7`.
- **Concrete witness:** extension popup execution broke; completed Sol/Pro
  deliveries were lost to stale payload/tab close; auto-capture never posted;
  telemetry was not durable/session-complete; service tests used invalid
  construction profiles.
- **Invariant:** each captured payload moves through explicit queued,
  delivering, acknowledged, failed, and retryable states bound to a stable
  session; restart/tab close/reconnect cannot lose an acknowledged or queued
  deliverable, and failures are operator-visible.
- **Vary:** popup/background/auto trigger; tab close at every phase; receiver
  restart; stale/new payload; offline queue; duplicate delivery; telemetry
  failure; valid service profiles.
- **Generalized proof:** production extension/receiver lifecycle harness with
  deterministic phase barriers and durable delivery/telemetry receipts.
- **Retained seeds:** completed deliverable followed by stale payload and tab
  close, plus 160 GETs and zero capture POSTs.
- **Sensitivity mutation:** clear queue on tab close, acknowledge before durable
  receipt, suppress trigger failure, or construct a service missing production
  invariants.
- **Dossier home:** installed-route/capture packet; real browser policy remains
  a runtime canary even after the deterministic core law exists.

### L30 — Deployed status is bound to the running artifact and host evidence

- **Bead evidence:** `polylogue-k2m`, `polylogue-6rvt`,
  `polylogue-s8q`, `polylogue-peo`.
- **Concrete witness:** status failed to recognize live daemon HTTP readiness;
  packaged runtimes exposed incomplete build revision; deployed archive/capture
  freshness was not attested; daemon exits could not be correlated to host or
  workload evidence.
- **Invariant:** a status claim identifies the running build, archive identity,
  service endpoint, freshness horizon, and relevant host/workload receipt.
  Source checkout state cannot impersonate deployed state.
- **Vary:** packaged/dev build; stale/restarted daemon; multiple archive roots;
  healthy process with stale capture; graceful/error/OOM/unknown exit; missing
  host evidence.
- **Generalized proof:** launch a production-valid packaged service, query
  status through public routes, perturb build/archive/freshness/exit evidence,
  and assert conservative typed attribution.
- **Retained seeds:** daemon HTTP healthy but status says absent and unexplained
  process exit without host/workload correlation.
- **Sensitivity mutation:** report checkout git head instead of running build,
  infer readiness from PID only, or default missing exit evidence to success.
- **Dossier home:** daemon lifecycle and status/facades.

### L31 — Affected-test selection is proved by real production mutations

- **Bead evidence:** `polylogue-b054.1.1`,
  `polylogue-b054.1.1.1`, `polylogue-88jp.1`, `polylogue-p5li`.
- **Concrete witness:** fresh-checkout testmon selection was unbounded or
  baseline-dependent; xdist nondeterministically lost demo constructs; clean
  baseline failures lacked explicit ownership.
- **Invariant:** from a fresh checkout, changing a production dependency
  selects at least one test that fails for the behavioral reason, with bounded
  setup and stable failure ownership. Isolated and xdist execution preserve the
  same semantic construct set.
- **Vary:** representative production module/failure class; seeded/unseeded
  testmon; isolated/xdist worker count/order; clean/pre-existing failure;
  collection import; cache presence.
- **Generalized proof:** the realized mutation/testmon campaign plus repeated
  schedule witnesses and exact selection/baseline receipts.
- **Retained seeds:** the actual production mutation from `b054.1.1.4` and the
  nondeterministic demo construct loss.
- **Sensitivity mutation:** delete a dependency edge/seed receipt, accept an
  empty selection, or use worker-local unordered construction.
- **Dossier home:** upstream `b054.1.1` is authoritative; Diet only consumes
  receipts and subtracts dominated verifier declarations.

### L32 — Harness artifacts are isolated, attributable, and closure-aware

- **Bead evidence:** `polylogue-ra3w`, `polylogue-r3o3`,
  `polylogue-ooqh`, `polylogue-gxjh`, `polylogue-nu2h`.
- **Concrete witness:** basetemp escaped a worktree to host `/tmp`; cloud
  bootstrap hid failures and used unsafe worker/temp defaults; Beads imported
  full JSONL on every invocation; demo shelves ignored repository closure;
  a test used `__new__` to create an impossible server object.
- **Invariant:** every harness artifact and object is created through a
  production-valid, checkout-owned lifecycle with an exact identity and
  closure; failures remain visible and unrelated host/global state is neither
  consumed nor mutated.
- **Vary:** worktree/main/cloud; temp roots; repository closure changes;
  bootstrap failure; valid/invalid initialization; repeated metadata command;
  owned/borrowed runtime.
- **Generalized proof:** fresh-checkout matrix records resolved paths, artifact
  identities, repository inputs, subprocess failures, and cleanup; object
  lifecycle tests use real constructors or explicit complete fakes.
- **Retained seeds:** worktree test writing host `/tmp`, hidden cloud render
  failure, and partially initialized daemon server close.
- **Sensitivity mutation:** drop checkout hash from basetemp, swallow bootstrap
  exit status, ignore repository closure, or bypass initialization.
- **Dossier home:** devtools verification subtraction and harness foundation.

### L33 — Verification progress detects hangs under isolated and xdist runs

- **Bead evidence:** `polylogue-27rb`, `polylogue-09rn`,
  `polylogue-b054.1.1.1`, `polylogue-88jp.1`.
- **Concrete witness:** D-state xdist workers hung while supervisor output
  continued; embedding catch-up waited beyond 300 seconds; demo construction
  varied with schedule.
- **Invariant:** a run either advances named logical test/work events or reaches
  a bounded terminal state. Incidental output and process existence do not
  count as progress; repeated isolated/parallel schedules preserve results.
- **Vary:** worker hang with chatty master; quiet but progressing test; D-state;
  process child tree; isolated and several worker counts; retry; catch-up debt
  terminal/transient states.
- **Generalized proof:** synthetic worker protocols plus repeated real focused
  schedules, with progress/event timestamps, process cleanup, and identical
  semantic result receipts.
- **Retained seeds:** master emits bytes while one test never completes and the
  exact periodic embedding backlog timeout.
- **Sensitivity mutation:** update liveness on any stdout byte, omit child
  process tracking, or accept different construct sets across worker counts.
- **Dossier home:** upstream repeated hang-witness program; Diet removes weaker
  output-based and declaration-only checks after receipts exist.

## Dossier conversion and subtraction rule

These laws are portfolio designs, not yet worker prompts. Before dispatch, Sol
must resolve exact production symbols, exact owned/avoided files, prerequisite
Beads, current historical reproduction, permitted focused commands, and named
deletion candidates. A law may become multiple implementation dossiers when
write sets or architecture decisions differ; several laws may share one
fixture or deterministic scheduler.

Deletion is justified only when the implemented law:

1. reaches the authoritative production route;
2. retains the named historical seed;
3. varies the causal dimensions that make the class broader than the seed;
4. fails under the named representative mutation;
5. preserves unique compatibility, security, recovery, and diagnostic cases;
6. records runtime and LOC economics after the result is real.

One-off external behavior remains a canary when deterministic local proof would
only simulate the risk. In particular, real provider wire samples, browser
policy changes, cross-device filesystem behavior, packaged-runtime identity,
and host exit attribution retain dogfood or installed-route witnesses alongside
their generalized deterministic cores.
