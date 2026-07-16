## [user 307]
You are a senior engineer + technical strategist reviewing R&D for Polylogue, a local, single-writer, cross-provider archive for AI coding/chat sessions (split-tier SQLite: durable source/user, rebuildable index/embeddings, disposable ops; query-first CLI + MCP + daemon). The attached file has two parts: Part 1 is a master synthesis (shared context, identical across 6 sibling files — do not base deliverables on it, use it only to understand the whole); Part 2 is this file's unique slice of ~15 subagent design specs. Five sibling files hold the rest; your outputs must stay disjoint from theirs, so anchor everything you produce to Part 2.

Do three things.

1. Vet, hard. Adversarially verify the Part-2 specs — especially the conclusions: bug diagnoses (are the file:line claims and failure modes
real and correctly characterized?), the tier/regime/schsments, and any strategic or prioritization verdicts.Separate what you confirm, refute, and extend/correct, with reasoning. Then research around it: use your own knowledge and web search for
prior art, competing approaches, relevant standards/libure patterns, and anything the specs assume but don'tjustify. Bring in what's missing; challenge what's overconfident. Note where you'd need to read live source to be sure.
2. Write a self-contained analysis. Produce one coherenontains all the value in this file's material plus yourvetting and research — written so a reader who never se Restate and own the substance; never say "the specclaims" or reference the attachment. Organize it well (themes, confirmed problems, design decisions and their tradeoffs, your corrections, external context, open questions, a prioritized take). Be opinionated and concrete; preserve the sharp specifics (numbers, file paths,       mechanisms, risks) rather than abstracting them away.

3. Propose the bead changes that would enact everything your analysis covers — nothing substantive should be analysis-only. Output them as a single fenced ```json block with this shape:

{
  "file": <the N from the file's "file N of 6" header>,
  "proposed_beads": [
    {
      "proposed_id": "p<N>-01",                 // namespaced by file N so 6 sets never collide
      "change_kind": "new",                     // new
      "target_bead": null,                      // for update/close/supersede: the extant bead id, e.g. "polylogue-f2qv"
      "title": "...",                           // ≤72
      "issue_type": "feature|bug|task|epic|chore",
      "priority": 0-3,
      "labels": ["area:storage", ...],
      "description": "problem + evidence (what's true t
      "design": "approach, tier/regime, key mechanisms, alternatives rejected",
      "acceptance_criteria": ["observable, checkable ...", ...],
      "dependencies": ["polylogue-<extant>", "p<N>-0X"],   // extant bead ids and/or sibling proposed ids
      "parent": "polylogue-<epic>|p<N>-0X|null",
      "risks": ["top 1-3"],
      "confidence": "high|med|low",             // after your vetting
      "provenance": "one line: which finding/recommendaacts"
    }
  ]
}

Rules for the bead set: use real extant Polylogue bead es them (for target_bead, dependencies, parent); everyproposed bead must trace to something in your analysis (provenance); prefer update/supersede on an existing bead over a duplicate new one; size them as landable units with genuine acceptance criand keep them scoped to this file's payload so the sixJSON blocks concatenate into one non-overlapping backlog.

## [assistant 326]
bash -lc cd /mnt/data/poly_tree/polylogue && python - <<'PY'
from pathlib import Path
checks = [
 ('core/refs.py', ['ObjectRefKind','_OBJECT_REF_KINDS','query']),
 ('core/enums.py', ['class AssertionKind','PATHOLOGY','FINDING','SAVED_QUERY']),
 ('storage/sqlite/archive_tiers/index.py', ['INDEX_SCHEMA_VERSION','INDEX_DDL','query_defs','blocks']),
 ('daemon/convergence_stages.py', ['make_default_convergence_stages','StandingQuery','ConvergenceStage']),
 ('storage/sqlite/archive_tiers/user_write.py', ['upsert_pathology_findings_as_assertions','judge_assertion_candidate','ASSERTION_CLAIM_KINDS','upsert_assertion']),
 ('daemon/http.py', ['ThreadingHTTPServer','asyncio.run','Polylogue','query-completions','preview']),
 ('cli/archive_query.py', ['ArchiveStore.open_existing','create_vector_provider','reciprocal_rank_fusion','query_unit_rows']),
 ('daemon/cli.py', ['polylogued','daemon.pid','flock','api-port','Restart']),
 ('insights/transforms.py', ['commit','decision','regex','test-pass']),
]
for file,terms in checks:
    p=Path(file)
    print('\n###', file, p.exists(), 'lines', sum(1 for _ in p.open(errors='ignore')) if p.exists() else '')
    if not p.exists(): continue
    txt=p.read_text(errors='ignore')
    for t in terms:
        print(f'{t!r}:', txt.find(t))
PY

## [assistant 333]
bash -lc cd /mnt/data/poly_code/polylogue && python - <<'PY'
from pathlib import Path
checks = [
 ('polylogue/core/refs.py', ['ObjectRefKind','_OBJECT_REF_KINDS','query']),
 ('polylogue/core/enums.py', ['class AssertionKind','PATHOLOGY','FINDING','SAVED_QUERY']),
 ('polylogue/storage/sqlite/archive_tiers/index.py', ['INDEX_SCHEMA_VERSION','INDEX_DDL','query_defs','blocks']),
 ('polylogue/daemon/convergence_stages.py', ['make_default_convergence_stages','StandingQuery','ConvergenceStage']),
 ('polylogue/storage/sqlite/archive_tiers/user_write.py', ['upsert_pathology_findings_as_assertions','judge_assertion_candidate','ASSERTION_CLAIM_KINDS','upsert_assertion']),
 ('polylogue/daemon/http.py', ['ThreadingHTTPServer','asyncio.run','Polylogue','query-completions','preview']),
 ('polylogue/cli/archive_query.py', ['ArchiveStore.open_existing','create_vector_provider','reciprocal_rank_fusion','query_unit_rows']),
 ('polylogue/daemon/cli.py', ['polylogued','daemon.pid','flock','api-port','Restart']),
 ('polylogue/insights/transforms.py', ['commit','decision','regex','test-pass']),
]
for file,terms in checks:
    p=Path(file)
    print('\n###', file, p.exists(), 'lines', sum(1 for _ in p.open(errors='ignore')) if p.exists() else '')
    if not p.exists(): continue
    txt=p.read_text(errors='ignore')
    for t in terms:
        print(f'{t!r}:', txt.find(t))
PY

## [assistant 356]
bash -lc cd /mnt/data/poly_code/polylogue && rg -n "ObjectRefKind|_OBJECT_REF_KINDS|class AssertionKind|PATHOLOGY|SAVED_QUERY|INDEX_SCHEMA_VERSION|CREATE TABLE IF NOT EXISTS assertions|kind\s+TEXT|upsert_pathology_findings_as_assertions|judge_assertion_candidate|ASSERTION_CLAIM_KINDS|make_default_convergence_stages|ConvergenceStage|query_completion|QueryCompletionCandidate|preview_command|_QUERY_GRAMMAR|_split_pipeline_stages|_iter_top_level_with_positions|projection_from_view|QueryProjectionSpec|DaemonAPIHTTPServer|ThreadingHTTPServer|asyncio.run\(|ArchiveStore.open_existing|create_vector_provider|reciprocal_rank_fusion|query_unit_rows|session_runs|observed_events|session_context_snapshots|session_work_events|raw_artifacts|blob_placement|zstandard|vec0|origin_family|source_family|Source" polylogue/core/refs.py polylogue/core/enums.py polylogue/storage/sqlite/archive_tiers/{index,user,source,embeddings}.py polylogue/storage/sqlite/archive_tiers/user_write.py polylogue/daemon/convergence_stages.py polylogue/archive/query/completions.py polylogue/archive/query/expression.py polylogue/surfaces/projection_spec.py polylogue/daemon/http.py polylogue/cli/archive_query.py | sed -n '1,260p'

## [assistant 403]
According to a document from 2026-07-05, I treated `rnd-bundle-3-of-6.md` as **file 3 of 6** and used only its Part 2 payload as the scope anchor; Part 1 was useful only for confidence caveats and global context. The package/code snapshot let me verify many of the “greenfield vs existing seam” claims, but I could not verify live Git tags, current release state, or the original `tasks/*.output` line anchors because those task-output files and live repo metadata were not included. The payload itself also warns that some synthesis-level file:line claims were summary-derived after compaction and should be re-opened before coding. fileciteturn2file1turn3file3

## Executive verdict

File 3 is the “analysis substrate” slice. Its strongest coherent direction is: make Polylogue able to turn archive evidence into **durable query objects, reproducible findings, citable evidence packs, compressed corpus digests, provenance-aware reports, and validity-gated analytics**. This is the right frontier for Polylogue because it reinforces the product identity as a local system of record / flight recorder for AI work rather than a generic chat viewer or memory toy. The strongest parts are query/findings-as-objects, content-hash citation anchors, corpus compaction, source/blob tiering discipline, vector retrieval evaluation, and meta-quality tests. The weaker parts are where the payload overstates “no schema break,” under-specifies canonicalization and staleness, or treats derived result sets as more durable than they really are.

My highest-confidence correction is that **query objects need to be split into at least four concepts**: `query_definition`, `query_run`, `result_relation`, and `render_artifact`. The Part-2 design’s `query_defs/result_sets/result_set_members` is a strong start, but a single `query:<hash>` plus index-tier snapshots is not enough to support citations, findings-as-tests, analysis replay, and agent loops cleanly. `query:<hash>` should identify the canonical definition; executions and result relations need separate refs.

My second-highest-confidence correction is that **provenance is the common abstraction behind almost everything in this file**. Findings, reports, annotations, query runs, context packs, model-drift observations, citation anchors, and corpus compaction all need the same pattern: structured object, evidence refs, source freshness, derived-vs-raw tier, and revalidation behavior. This aligns well with W3C PROV’s entity/activity/agent model and OpenLineage’s run/job/dataset/facet model: both treat provenance as structured event metadata rather than prose attached after the fact. citeturn346437search2turn346437search6turn346437search3turn346437search7

## Hard vetting: confirmed

The query/findings object work is genuinely greenfield on real seams. In the code snapshot, there is no `query_defs`, `result_sets`, `query_edges`, `StandingQueryStage`, `query` ObjectRef kind, or `AssertionKind.FINDING`. `index.db` is still v24; `user.db.assertions.kind` is plain `TEXT`, so adding a new assertion kind is a code/schema-surface change rather than a durable DDL migration. The existing assertion candidate → judge → promote lifecycle is real and can be reused. The Part-2 design’s placement of query registry/result snapshots in derived `index.db`, durable standing-query intent in `user.db`, and standing-run bookkeeping in `ops.db` is directionally correct. fileciteturn1file4turn1file5turn1file11turn1file16

The main correction is canonicalization. “Equivalent queries collapse” is right only where equivalence laws are explicit. Boolean `A and B` can commute. Pipeline stages, ranking, `limit`, `sort`, semantic/hybrid retrieval, sequence, and set-op rank policies usually cannot. The query identity layer needs an operator-law table: commutative, associative, order-sensitive, rank-bearing, stochastic/model-dependent, and macro-expanded. Without that, the exact risk the payload names—canonical-form drift orphaning result sets and findings—will happen. fileciteturn2file2

The archive-economics/tiering design is grounded. The current blob store stores raw sharded blobs addressed by SHA-256 of the uncompressed payload; verification currently hashes raw file bytes; there is no zstd placement/dictionary/tombstone/frozen-segment layer in the snapshot. So the safe beachhead is exactly right: update read/verify to sniff/decompress before compressing a single blob. Source-tier durable placement/dictionary/tombstone metadata is justified because losing a zstd dictionary can make compressed evidence unreadable; RFC 8878 describes a dictionary ID field, but decompression still depends on having the dictionary bytes, so the dictionary registry must be backed up and append-only. fileciteturn1file6turn1file10turn3file16 citeturn346437search4turn346437search16turn346437search20

The “address is SHA(uncompressed)” principle is exactly the right invariant. It preserves deduplication, re-ingest idempotency, and evidence identity while allowing compression, cold movement, frozen pack files, or citable tombstones underneath. The correction is that `last_read_at_ms` is not really source-tier durability; it is access-temperature telemetry. It can be co-located for performance, but the design should either tolerate losing it or mirror it in ops. Physical placement and decompression dependencies are durable; access heat is not.

Corpus compaction is correctly placed in the read algebra. It should not be a separate report generator. The payload frames `find <query> | compact` as `Selection × CompactionProjection × COMPACT render`, distinct from the existing context compiler that builds handoff/context images around seed sessions. That distinction is important: context compiler is for continuation; corpus compaction is for feeding an external model a bounded, decision-dense digest of many sessions, with a drop manifest. fileciteturn3file16

The context scheduler direction is real but less implemented. The assertions schema already supports author, evidence refs, confidence, staleness, and `context_policy_json`, but the scheduler/ContextSource/ledger behavior is not present in the snapshot as a real subsystem. The payload’s ops-ledger design is a good first cut, especially its explicit exclusion of `context_inject` from attention signals so the scheduler cannot reinforce its own injections. The correction is retention: if a context compile or injection ledger becomes evidence in an uplift experiment, report, or finding, it must be snapshotted/exported/promoted, because pure ops retention is disposable by design. fileciteturn3file10turn1file17

The blue-green index rebuild design is well-founded. The current code is still built around `index.db`, so generation pointer files, `index.gN.db`, atomic swap, reader leases, and ops mirrors are new work. The pointer file should be authoritative; an ops table can mirror state but cannot be the thing you need before opening the index. The test strategy is strong: pointer atomicity, held-reader generation stability, zero failed queries during swap, and lease-safe reaping. fileciteturn1file7turn1file8

The provider→origin aggregate honesty issue is real, but one phrasing needs tightening. The code already has some source-family concepts, so the problem is not “there is no source-family anywhere”; the problem is that physical session identity and headline rollups are still keyed in ways that can collapse distinct runtime families. Adding `source_family` and lossy markers can fix aggregate honesty; it cannot fix a physical row collision caused by `session_id = origin || ':' || native_id`. The payload correctly isolates that residual as a separate durable/identity problem. fileciteturn2file13turn2file6

The vector-search plan is plausible and properly skeptical. The snapshot has two live vec0 DDL definitions with incompatible metadata naming (`+origin` versus legacy `+source_name`), no int8/bit lane, and no quantization/eval gate. The payload’s insistence on f32 baseline, int8 rerank, recall@k from lineage-labeled pairs, and “do not default until live eval passes” is exactly the right posture. The correction is prior-art freshness: sqlite-vec has binary quantization support, and SQLite’s official `vec1` extension has recently documented ANN support with IVF/PQ-style compression. Before building a homegrown centroid/IVF layer, Polylogue should run a short decision spike comparing sqlite-vec int8/binary, SQLite `vec1`, and an external ANN option. fileciteturn2file12turn3file9 citeturn346437search1turn346437search9

The evidence-cockpit anchor diagnosis is confirmed. Current block refs are position-based; message content hashes exist, but block content hashes do not. A stable citation system needs block hashes, but block hash alone is not enough because compaction/replay can duplicate identical blocks. Anchors need captured position, parent message/session hash, lineage segment, and ambiguity states. The payload’s exact/drift/deleted/quarantined framing is correct; I would add “ambiguous” as a first-class resolution state rather than treating multi-hit as degraded exactness. fileciteturn2file8turn3file17

The missing-unit and ontology work is valuable only if kept disciplined. Several “units” should be views or runtime methods, not tables. The SQLite rule is real: a persistent view in one attached database cannot reference objects in another attached database, so correction-edge / assertion-over-block joins must be runtime queries, TEMP views, or materialized derived tables built without cross-tier DDL. SQLite’s own forum guidance aligns with this constraint. fileciteturn3file1 citeturn437157search0turn437157search6

The meta-quality suite is one of the highest-leverage parts of this file. Identity round-trip, content-hash idempotency, lineage no-double-count, FTS trigger coherence, daemon-vs-direct differential, convergence idempotency, blob-GC state machine, parser fuzz, and scale-shaped synthetic corpus are exactly the tests that make the “system of record” claim credible. They should outrank a lot of UI polish. fileciteturn3file11turn3file2

## Hard vetting: refuted or corrected

The “no break” phrasing in archive compression is too optimistic. The address layer can remain stable, but the read layer changes semantically: every reader, verifier, backup checker, evidence resolver, and GC path must understand codec sniffing, dictionaries, frozen ranges, tombstones, and logical-vs-stored bytes. That is not an identity break, but it is a wide behavioral migration. The bead must land read/verify decompression first and block compression writes until every path is proven codec-aware. fileciteturn1file12

The query-object design underplays query-run durability. It says ad-hoc snapshots are disposable, which is correct for index-tier caches, but any promoted finding, published report, or analysis recipe cannot cite a disposable result set. Promotion must require a saved/standing query or a durable evidence pack. A finding can cite `query:<hash>` and durable evidence refs; it should not depend on `result_set_id` surviving `ops reset --index`. fileciteturn3file7

`corpus_version = MAX(updated_at_ms)` is not good enough as a general staleness boundary. It is coarse, can thrash every standing query after irrelevant ingest, and may not be stable across rebuilds. The staleness dependency should be a vector: source high-water marks by origin/family, materialization generation, query lanes used, and maybe relevant filter constraints. The Part-2 risk section already names thrash/staleness; I would make dependency vectors part of the design rather than an optimization. fileciteturn2file2

The model-drift observatory is correct to avoid causal claims, but it should not become another analytics “museum.” Drift measures need to be tied to a measure registry and to actual launch/use questions. Existing systems like Evidently and NannyML distinguish data drift, performance estimation, and label-free monitoring, while changepoint libraries like `ruptures` can find structural changes but do not identify causes by themselves. Polylogue should render “candidate + nearby events + confounds,” never “model X regressed because…”. fileciteturn3file0 citeturn231898search0turn231898search1turn231898search2

The release/distribution analysis is only partially verifiable from the package. I confirmed that the tree carries release workflows and docs that still describe packaged installation as unavailable, and that the version appears frozen at `0.1.0` in local files. I could not verify the “zero git tags” claim from the tarball because `.git` metadata is absent. The actionable conclusion remains: release is mostly a decision and smoke-matrix issue, but the exact current public tag/release state needs live GitHub verification before bead closure. The right release baseline should use PyPI trusted publishing/OIDC, Sigstore attestations, CycloneDX SBOMs, and `uvx`/`uv tool` smoke tests rather than manual tokens or clone-only docs. fileciteturn3file5 citeturn566573search0turn566573search1turn566573search2turn566573search3

The “pure Python/no native deps” launch phrasing should be retired or narrowed. Even if Polylogue is written in Python, its packaging path can depend on native/Rust-backed wheels or SQLite extensions. The launch claim should be “installable without a compiler on supported platforms” only after wheel smoke tests prove no sdist fallback.

The “finding as AssertionKind” idea is right, but `AssertionKind.FINDING` is not sufficient as a product model. A published finding also needs a manifest: producing query, query run, corpus data sheet, expected value/tolerance, evidence refs, artifact URI, and revalidation status. This is where PROV/OpenLineage-style run metadata becomes more useful than a single row in `assertions`. fileciteturn3file4turn3file13

## What should be designed better than the current payload

The strongest unifying abstraction missing from file 3 is **analysis objects**. The payload has query objects, findings, reports, context ledgers, model-drift cohorts, compaction outputs, and demos, but it does not quite name the general object:

`analysis_run = recipe/prompt + query_runs + result_relations + evidence_packs + annotation_batches + findings + render_artifacts`

This matters because external agents will not only query Polylogue; they will run iterative analyses. They need stable refs for “what I asked,” “what came back,” “which rows I annotated,” “which claims I promoted,” and “which report I rendered.” This is more general than YAML recipes and more durable than CLI stdout.

The second missing abstraction is **value-level evidence refs**. The honesty audit correctly notes that block/message refs are too coarse for a cost, token count, occurrence count, or provenance-derived timestamp. A number should be able to point to `block + json-path/field + reducer + denominator/frame`, or else the product promise must say “every number resolves to evidence at block granularity,” not “every number resolves to bytes.” fileciteturn2file16turn3file12

The third missing abstraction is **identity-policy on every relation**. Query results, cohorts, result sets, episodes, drift cohorts, set algebra, and saved snapshots all depend on identity: physical session, logical session, message row, composed message, block hash, artifact hash, or source-family-qualified native id. This should not be implicit in each feature. It should be part of `result_relation`.

The fourth missing abstraction is **promotion gates**. Agent-written assertions, findings, prompt templates, drift observations, text-derived facts, and context-injected memories all need one chokepoint for author_kind/trust_class/status/inject policy. The payload’s recursive-safety note is right: if this is per-feature, one path will bypass it. fileciteturn3file8

## Prioritized take

First, land the honesty substrate before the gallery features: query/run/result refs, finding lifecycle, value-level evidence, text-derived provenance contracts, and meta-quality tests.

Second, land blue-green before the v25/v3/v5 pile. File 3 repeatedly needs index v25 and source v3; rebuilding a 38 GB archive one feature at a time is operationally wrong. The payload’s own cross-cutting note says schema bumps must be batched and blue-green removes the downtime cost. fileciteturn3file8

Third, do the compression beachhead early but slowly. Read/verify codec support first; placement/dictionaries second; compression writes third; frozen/drop later. Compression is high leverage, but only if evidence integrity is never put at risk.

Fourth, do not build the model-drift/analytics observatory before its validity gates. Embed coverage, cohort anchors, denominator rules, and uncertainty should block output. Otherwise it becomes exactly the “insight museum” the red-team warned about.

Fifth, preserve the launch lesson: Polylogue’s public story should be flight recorder/system-of-record plus one hard finding, not a giant feature catalog. The demo proof engine should force every public number through query/finding/evidence refs.

## Proposed bead changes

```json
{
  "file": 3,
  "proposed_beads": [
    {
      "proposed_id": "p3-01",
      "change_kind": "new",
      "target_bead": null,
      "title": "Query object refs and canonical identity",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:refs", "tier:index", "theme:evidence"],
      "description": "Polylogue has no query ObjectRef, query_defs table, or durable query identity. Query objects are needed for findings, standing queries, reports, analysis runs, and reusable agent workflows. The identity boundary must be lowered, macro-expanded, and law-aware rather than raw-text-based.",
      "design": "Add ObjectRef kind query and implement query_definition identity over canonical lowered specs. Define an operator-law table: commutative boolean selections may normalize; pipeline order, set-op rank policy, sort, limit, sequence, semantic/hybrid scoring, and model-dependent stages remain order-sensitive. Store canonical bytes and an epoch/version so canonicalization changes are explicit and auditable.",
      "acceptance_criteria": [
        "core/refs.py resolves query:<hash> to a query definition envelope with canonical text, grain, canonicalization epoch, and lowered-spec JSON.",
        "Property tests prove whitespace and safe boolean reordering do not change query ids while rank/order-sensitive changes do.",
        "Golden canonical-byte fixtures fail CI on accidental canonicalization drift.",
        "Generated OpenAPI, CLI output schemas, and topology projection include query refs."
      ],
      "dependencies": ["polylogue-fnm.13", "polylogue-4p1"],
      "parent": "polylogue-fnm",
      "risks": ["Over-normalizing rank-bearing queries", "Canonical-form drift orphaning downstream refs", "Macro expansion cycles"],
      "confidence": "high",
      "provenance": "Part-2 query/finding-object payload plus code vetting confirmed query refs and query_defs are absent."
    },
    {
      "proposed_id": "p3-02",
      "change_kind": "new",
      "target_bead": null,
      "title": "Result relations and query-edge DAG",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:algebra", "tier:index", "theme:analysis"],
      "description": "A query definition is not enough for analysis. Polylogue needs typed result-relation snapshots with grain, identity policy, exactness, member fingerprint, and query-edge provenance so set algebra, cohorts, findings, and reports can reason about results without depending on stdout.",
      "design": "Add derived-tier result_relations/result_relation_members and query_edges. Keep result members rebuildable and disposable unless promoted into an evidence pack. Store grain, identity_policy, rank_policy, member_count, merkle_root, corpus/dependency vector, computed_at, and exactness. Populate query_edges from set-algebra EXPLAIN nodes, not from ad hoc string parsing.",
      "acceptance_criteria": [
        "A set-op query records child query edges matching EXPLAIN exactly for union/intersect/except and subquery operands.",
        "Snapshotting the same query at the same dependency vector is idempotent and Merkle deterministic under member reordering.",
        "Result relations expose grain and identity_policy in CLI/MCP/API JSON.",
        "No promoted finding can cite a disposable result_relation without a durable query/evidence-pack fallback."
      ],
      "dependencies": ["p3-01", "polylogue-fnm.13"],
      "parent": "polylogue-fnm",
      "risks": ["Large result_set_members on broad queries", "Confusion between disposable cache and citable evidence", "Cross-grain set algebra mistakes"],
      "confidence": "high",
      "provenance": "Part-2 result_sets/query_edges design, corrected to split definition/run/relation concepts."
    },
    {
      "proposed_id": "p3-03",
      "change_kind": "new",
      "target_bead": null,
      "title": "FINDING assertions and standing-query tests",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:assertions", "area:query", "area:daemon", "theme:honesty"],
      "description": "Published and standing findings need a first-class lifecycle: candidate, human judgment, promotion, invariant recheck, regression emission, and generated-surface visibility. The assertion substrate can support this with no durable DDL migration, but the promotion gate must prevent disposable result snapshots from becoming citable truth.",
      "design": "Add AssertionKind.FINDING, register it in assertion claim/review surfaces, and implement standing query runs in ops.db. Promoted findings cite query definitions plus durable evidence refs, not index-only result_set ids. StandingQueryStage re-materializes stale query results, checks value.expected invariants, and emits regression findings as candidates.",
      "acceptance_criteria": [
        "FINDING candidates flow through list/judge/promote/reject/supersede using existing assertion review APIs.",
        "Promoting a finding requires a saved/standing query or a durable evidence pack; disposable ad hoc result sets are rejected with an actionable error.",
        "A promoted expected.member_count finding emits a regression candidate when the corpus changes.",
        "ops reset --index followed by polylogued run re-materializes standing query results without downgrading promoted findings."
      ],
      "dependencies": ["p3-01", "p3-02", "polylogue-3tl.4", "polylogue-9e5"],
      "parent": "polylogue-fnm",
      "risks": ["Recursive-safety bypass by agent-authored findings", "Standing queries thrash on coarse staleness", "Overloading assertions without schema registry"],
      "confidence": "high",
      "provenance": "Part-2 query/finding-object and demo/proof-engine payloads; code vetting confirmed assertion lifecycle exists and FINDING does not."
    },
    {
      "proposed_id": "p3-04",
      "change_kind": "update",
      "target_bead": "polylogue-83u.5",
      "title": "Blob codec read/verify beachhead before compression",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:storage", "area:blobs", "tier:source", "theme:evidence"],
      "description": "Compression is safe only if every read, verify, backup, citation, and GC path treats the address as SHA(uncompressed) and decompresses before hashing. Current blob verification hashes stored bytes directly, so enabling compression before reader support would corrupt the evidence contract.",
      "design": "Update the existing zstd bead so phase 1 lands codec sniffing and decompress-then-hash verification while all writes still emit raw blobs. Use magic sniffing as the truth source for zstd/plain-zstd and require dictionary lookup for zstd_dict. Do not enable compression writes until verify_all, backup verifier, evidence resolver, and GC tests pass under mixed raw/zstd fixtures.",
      "acceptance_criteria": [
        "read_blob(store_blob(payload)) == payload for raw, zstd, zstd_dict, cold, frozen, and tombstoned test fixtures.",
        "verify() and verify_all() hash decompressed payload bytes and fail on corrupted compressed frames.",
        "All existing blob readers pass mixed raw/zstd fixtures before any write path emits compressed frames.",
        "BlobStore.stats exposes logical_bytes and stored_bytes without changing blob_size semantics."
      ],
      "dependencies": ["polylogue-83u", "polylogue-83u.4"],
      "parent": "polylogue-83u",
      "risks": ["A missed reader silently breaks backup verification", "Dictionary loss makes zstd_dict blobs unreadable", "Stored/logical byte aliases leak to user surfaces"],
      "confidence": "high",
      "provenance": "Part-2 archive-economics payload plus code vetting of raw blob store and raw-byte verify."
    },
    {
      "proposed_id": "p3-05",
      "change_kind": "new",
      "target_bead": null,
      "title": "Blob placement, tombstones, and frozen packs",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:storage", "area:blobs", "tier:source", "theme:scale"],
      "description": "Long-term archive economics need physical placement, compression dictionaries, citable tombstones, and frozen pack files while preserving uncompressed content-hash identity. The metadata gates evidence integrity and decompression, so it belongs in source-tier durable migration rather than ops-only telemetry.",
      "design": "Add source v3 additive tables for blob_placement, blob_dicts, blob_tombstones, frozen_segments, and frozen_segment_entries. Keep address = SHA-256(uncompressed). Move access-temperature signals to ops or make them explicitly lossy. Implement blob-compact phases: verify, recompress, cold move, freeze, drop-with-tombstone, GC, each resumable and lease-aware.",
      "acceptance_criteria": [
        "Source migration backfills blob_placement for existing raw blobs with logical/stored sizes matching current raw_sessions data.",
        "Frozen segment range reads decompress to the addressed payload and whole-pack checksum verifies.",
        "Dropped reacquirable blobs resolve to a BlobDropped/tombstone payload instead of 500 or silent absence.",
        "GC treats hot, cold, frozen, and dropped storage classes correctly and never collects leased or referenced blobs."
      ],
      "dependencies": ["p3-04"],
      "parent": "polylogue-83u",
      "risks": ["Dictionary registry becomes a backup-critical dependency", "GC misclassifies cold/frozen blobs", "Source-tier migration too large without batching"],
      "confidence": "med",
      "provenance": "Part-2 archive-economics storage-class design, corrected for access-telemetry tiering."
    },
    {
      "proposed_id": "p3-06",
      "change_kind": "update",
      "target_bead": "polylogue-b5l",
      "title": "Generation pointer and reader-lease rebuild core",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:storage", "area:daemon", "tier:index", "theme:scale"],
      "description": "Many Part-2 designs require index v25 or expensive rebuilds. Live rebuild downtime and degraded windows should be removed before batching those schema bumps. The current tree has no generation pointer or index.gN path resolver.",
      "design": "Implement index.pointer.json as the authoritative active-generation record, with gen-0 adoption from legacy index.db, index.gN.db path resolution, ops.db mirror for status only, atomic pointer swap, held-reader generation stability, and lease-safe reaping. Default reset --index should schedule blue-green rebuild; --offline keeps destructive rebuild for recovery.",
      "acceptance_criteria": [
        "Legacy archives adopt index.db to index.g0.db idempotently and all index path consumers resolve through the pointer helper.",
        "Concurrent readers see either generation N or N+1 during swap, never torn state or no such table.",
        "Zero-failed-query harness passes while a schema-bump rebuild runs and pointer swap pause is measured under 100 ms on seeded corpus.",
        "Old generations are not reaped while a read lease is open and are reaped after grace plus lease release."
      ],
      "dependencies": ["polylogue-20d.15"],
      "parent": "polylogue-b5l",
      "risks": ["Delta replay race at swap boundary", "Path consumers bypass pointer helper", "Reaper deletes active generation"],
      "confidence": "high",
      "provenance": "Part-2 blue-green rebuild payload plus code vetting that current index path is legacy index.db."
    },
    {
      "proposed_id": "p3-07",
      "change_kind": "new",
      "target_bead": null,
      "title": "Blue-green delta replay correctness harness",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:storage", "area:tests", "tier:index", "theme:correctness"],
      "description": "Blue-green is only safe if writes during rebuild are replayed exactly once into the new generation. A naive updated_at_ms or max timestamp boundary is too coarse for a multi-origin archive and risks lost or double-applied rows.",
      "design": "Define the rebuild dependency cursor: source raw_id/acquired_at high-water marks, origin-specific ingest cursors, materialization generation, and index schema version. Build a differential test that mutates source/user data during rebuild and proves post-swap generation equals a cold rebuild from durable tiers.",
      "acceptance_criteria": [
        "A synthetic concurrent ingest during rebuild appears in the new generation after swap exactly once.",
        "Post-swap generation diff against cold rebuild is empty for sessions, messages, blocks, FTS, lineage, and selected insights modulo ops cursors.",
        "Tests cover writes before rebuild start, during materialization, during swap, and after swap.",
        "Failure leaves pointer on old generation and exposes actionable status."
      ],
      "dependencies": ["p3-06", "polylogue-1xc"],
      "parent": "polylogue-b5l",
      "risks": ["False confidence from timestamp-only replay", "Long rebuilds make test slow", "User-tier overlay interactions missed"],
      "confidence": "med",
      "provenance": "Vetting correction to Part-2 blue-green delta-replay design."
    },
    {
      "proposed_id": "p3-08",
      "change_kind": "update",
      "target_bead": "polylogue-37t.11",
      "title": "Context scheduler trust gate and injection ledger",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:context", "area:assertions", "tier:ops", "theme:safety"],
      "description": "The scheduler is the single arbiter of what enters agent context, but the code snapshot does not contain a real ContextSource scheduler. Assertions already have context_policy_json, confidence, evidence, and author fields, so the right move is to build one scheduler over existing overlays with a hard trust gate.",
      "design": "Implement ContextSource protocol, schedule_context entrypoint, trust_class derivation, and ops.db context ledger. Non-user or unpromoted agent-authored content cannot emit operator-trust context; tool output/web/runtime context is refs-only or fenced. Ledger records included, degraded, and dropped candidates with scores and budget state. Context injection events are excluded from attention training.",
      "acceptance_criteria": [
        "SessionStart and PreCompact context flows call schedule_context rather than assembling their own memory list.",
        "Seeded injection-string fixture never enters an unfenced preamble.",
        "Ledger rows record candidate score, trust class, inclusion/drop reason, and budget class.",
        "polylogue context ledger <session> and MCP equivalent read the same ledger envelope."
      ],
      "dependencies": ["polylogue-37t", "polylogue-cpf"],
      "parent": "polylogue-37t",
      "risks": ["Trust retrofit leaves a bypass", "Ops ledger not durable enough for experiments", "Ranking becomes self-reinforcing"],
      "confidence": "high",
      "provenance": "Part-2 context-scheduler payload and previous compaction/memory analysis."
    },
    {
      "proposed_id": "p3-09",
      "change_kind": "new",
      "target_bead": null,
      "title": "Corpus compaction read-algebra terminal",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:query", "area:projection", "area:context", "theme:rnd-flywheel"],
      "description": "Large multi-session R&D analyses need a bounded, decision-dense digest for external LLMs. Existing compile_context is a seed-session handoff image, not a corpus-level compaction over a find result.",
      "design": "Add compact as a read-algebra terminal/projection: SelectionSpec over logical sessions, CompactionProjection policy, and RenderFormat.COMPACT. Keep human turns, decisions, error-to-fix chains, outcomes, and citations; drop tool spam by material_origin; emit fidelity drop manifest and recoverable refs. Reuse context token estimation and omission structures but do not conflate with ContextImage.",
      "acceptance_criteria": [
        "find <query> | compact renders a token-budgeted digest and drop manifest on a seeded multi-session corpus.",
        "Lineage prefixes deduplicate at logical-session grain.",
        "Digest labels omitted material classes and gives recoverable refs for dropped evidence.",
        "Golden test proves tool-output spam is dropped while decisions, failures, fixes, and outcomes are retained."
      ],
      "dependencies": ["polylogue-4p1", "polylogue-fnm.13"],
      "parent": "polylogue-fnm",
      "risks": ["Summary becomes ungrounded prose", "Token estimator too inaccurate", "Compaction hides contradictory evidence"],
      "confidence": "high",
      "provenance": "Part-2 corpus-compaction payload, corrected as projection/render rather than separate subsystem."
    },
    {
      "proposed_id": "p3-10",
      "change_kind": "update",
      "target_bead": "polylogue-mhx.6",
      "title": "Quantized vectors behind live recall gate",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:embeddings", "area:search", "tier:embeddings", "theme:performance"],
      "description": "Quantized vector storage can reduce scan cost, but the snapshot has no int8/bit lane and has two vec0 DDL definitions with incompatible metadata columns. Quantization must not silently degrade recall on the real archive.",
      "design": "Unify embeddings DDL first. Add int8-primary plus f32 rerank path, origin partitioning, and optional centroid prefilter only after eval. Build lineage-labeled recall@k from session_links and compare f32 baseline versus int8 and int8+prefilter on seeded and live corpora. Keep centroid prefilter off by default until tolerance passes.",
      "acceptance_criteria": [
        "Only one canonical embeddings DDL/table-creation path remains.",
        "Int8 reranked top-k matches brute-force f32 top-k within declared tolerance on seeded corpus.",
        "Live lineage-labeled recall@5/@10 and MRR are emitted for f32, int8, and int8+prefilter.",
        "Default lane flips only with cited eval artifact and documented metric deltas."
      ],
      "dependencies": ["polylogue-mhx.3"],
      "parent": "polylogue-mhx.6",
      "risks": ["Mixed-model vectors break unit normalization", "Centroid prefilter prunes true neighbors", "Single mean vector loses multi-topic sessions"],
      "confidence": "high",
      "provenance": "Part-2 vec0-ANN payload plus code vetting of duplicate DDL definitions."
    },
    {
      "proposed_id": "p3-11",
      "change_kind": "new",
      "target_bead": null,
      "title": "ANN backend decision spike",
      "issue_type": "task",
      "priority": 3,
      "labels": ["area:embeddings", "area:research", "theme:performance"],
      "description": "Before building a custom centroid/IVF layer on sqlite-vec, Polylogue should compare current sqlite-vec quantization, SQLite vec1 ANN features, and an external ANN option against its live lineage-labeled evaluation set.",
      "design": "Run a short, evidence-backed decision spike: measure build complexity, recall@k, query latency, index size, packaging impact, and schema/rebuild implications for sqlite-vec int8/binary, SQLite vec1, and one external ANN candidate. Choose the minimal backend that passes recall/latency gates and preserves local-first packaging.",
      "acceptance_criteria": [
        "Decision doc records three candidates with recall@10, MRR, p95 query latency, index bytes, and packaging constraints.",
        "Recommendation names a winner and runner-up and explicitly states why others are rejected.",
        "No centroid/prefilter default work begins until the spike lands.",
        "Bead mhx.6 acceptance links to the chosen backend decision."
      ],
      "dependencies": ["p3-10"],
      "parent": "polylogue-mhx",
      "risks": ["Research delays obvious int8 win", "External ANN adds native packaging burden", "Benchmarks not representative of live archive"],
      "confidence": "med",
      "provenance": "External prior-art check found relevant sqlite-vec/vec1 capabilities not fully accounted for in Part 2."
    },
    {
      "proposed_id": "p3-12",
      "change_kind": "update",
      "target_bead": "polylogue-2qx",
      "title": "Source-family aggregate honesty marker",
      "issue_type": "bug",
      "priority": 1,
      "labels": ["area:origins", "area:cost", "theme:honesty"],
      "description": "Provider/origin projection is non-injective. Adding source_family and lossy_grouping markers can fix aggregate honesty, but it must not be presented as fixing physical identity collisions under session_id = origin || ':' || native_id.",
      "design": "Wire source_family/lossy_grouping into cost, usage, summaries, tool usage, CLI/MCP/API payloads, and registry projection boundaries. Emit lossy markers whenever a public grouping merges multiple source families. Keep the identity-collision residual explicit and link to the separate physical identity design.",
      "acceptance_criteria": [
        "Aggregates over origins that merge multiple source families carry row-level and top-level lossy_groupings metadata.",
        "Cost_rollups/provider_usage/session_costs split by source_family where needed and preserve numbers for 1:1 origins.",
        "CLI, MCP, and JSON surfaces expose the same marker via one projection helper.",
        "Docs explicitly state that aggregate markers do not repair pre-existing physical row collisions."
      ],
      "dependencies": ["polylogue-f2qv"],
      "parent": "polylogue-2qx",
      "risks": ["Surfaces drift if marker added per-path", "Users misread aggregate fix as identity fix", "Backfill command mutates durable identity accidentally"],
      "confidence": "high",
      "provenance": "Part-2 provider→origin payload, corrected because code has some source-family concepts but not full identity/rollup wiring."
    },
    {
      "proposed_id": "p3-13",
      "change_kind": "new",
      "target_bead": null,
      "title": "Source-family-safe physical session identity",
      "issue_type": "task",
      "priority": 2,
      "labels": ["area:origins", "area:storage", "tier:source", "theme:identity"],
      "description": "If two runtime families share an origin projection and native_id, current generated session_id can collapse distinct physical sessions into one row. Aggregate markers cannot recover rows already merged below identity.",
      "design": "Design a durable identity migration path that either namespaces native_id by source_family or introduces a stable physical_session_key while preserving public origin vocabulary. Include collision census, compatibility story for existing refs, copy-forward/backfill plan, and refusal criteria if unsplittable collisions exist.",
      "acceptance_criteria": [
        "Census reports whether any live archive rows are suspected family/native collisions and labels confidence.",
        "Design doc specifies old-ref resolution semantics and migration/reparse path.",
        "Synthetic two-family same-native-id fixture stores two physical sessions and projects a lossy public origin without collision.",
        "No code migration lands until backup manifest and compatibility plan are reviewed."
      ],
      "dependencies": ["p3-12", "polylogue-4ts"],
      "parent": "polylogue-4ts",
      "risks": ["Durable identity break invalidates existing refs", "Unsplittable historical rows", "Over-migration for a rare edge case"],
      "confidence": "med",
      "provenance": "Part-2 identity-collision residual under provider→origin analysis."
    },
    {
      "proposed_id": "p3-14",
      "change_kind": "update",
      "target_bead": "polylogue-bby.11",
      "title": "Content-hash citation anchors with ambiguity states",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:webui", "area:citations", "tier:index", "theme:evidence"],
      "description": "Position-based block refs drift on re-ingest and fork replay. Message hashes exist but block hashes do not. Evidence cockpit, reports, marginalia, and exported findings need stable anchors that survive rebuilds and surface ambiguity instead of guessing.",
      "design": "Add block_content_hash in index materialization and a citation anchor resolver over composed lineage transcripts. Anchor captures block hash, captured position, parent message/session hash, lineage segment, and optional surrounding hash window. Resolution states include exact, drifted, deleted, quarantined, and ambiguous.",
      "acceptance_criteria": [
        "Fork-prefix re-ingest resolves prior anchors as exact or drifted with no position-only match.",
        "Duplicate block-content hashes produce ambiguous or degraded resolution unless disambiguation fields select one copy.",
        "Report footnotes and marginalia use the resolver rather than raw block positions.",
        "Index rebuild cost is measured on seeded corpus and batched with the v25 schema bump plan."
      ],
      "dependencies": ["polylogue-4ts", "polylogue-b5l"],
      "parent": "polylogue-bby.11",
      "risks": ["Duplicated blocks mis-anchor silently", "38GB rebuild cost lands unbatched", "SPA renderer bypasses resolver"],
      "confidence": "high",
      "provenance": "Part-2 evidence-cockpit payload plus code vetting of position-based block refs."
    },
    {
      "proposed_id": "p3-15",
      "change_kind": "new",
      "target_bead": null,
      "title": "Evidence report and basket objects",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:webui", "area:reports", "area:assertions", "theme:legibility"],
      "description": "Published findings and demos need citable report objects rather than loose Markdown. Evidence baskets should gather anchored refs, query definitions, result relations, and promoted findings into renderable reports with drift-aware footnotes.",
      "design": "Add assertion-backed evidence_report and report_basket object kinds, PATCH/GET routes, Markdown/HTML renderers, and integrity verifier. Reports cite query definitions, durable evidence refs, and anchor resolution states. Live re-resolve updates drift badges without rewriting prose.",
      "acceptance_criteria": [
        "A basket can add block/session/query/finding refs and render a Markdown report with provenance footnotes.",
        "Deleted/quarantined anchors block export unless override is explicit and recorded.",
        "Golden report renders through pandoc and contains stable evidence refs.",
        "Report drift status updates after re-ingest via the same resolver used by the web cockpit."
      ],
      "dependencies": ["p3-03", "p3-14", "polylogue-3tl.4"],
      "parent": "polylogue-bby",
      "risks": ["Report model becomes a mini CMS", "Footnotes drift from resolver truth", "Basket UX delays core anchor engine"],
      "confidence": "med",
      "provenance": "Part-2 evidence cockpit and demo/proof engine payloads."
    },
    {
      "proposed_id": "p3-16",
      "change_kind": "new",
      "target_bead": null,
      "title": "Model drift observatory with validity gates",
      "issue_type": "feature",
      "priority": 3,
      "labels": ["area:analytics", "area:measures", "theme:construct-validity"],
      "description": "Model-drift analytics are valuable only if they do not conflate workload drift, model upgrade events, embedding coverage gaps, and causal inference. Drift output must be candidate-only and validity-gated.",
      "design": "Implement drift measures as MeasureSpec rows with construct, denominator, evidence tier, confounds, coverage preconditions, and uncertainty. Cohort anchors are explicit assertions requiring workflow-shape and embedding agreement. Changelog/changepoint output uses PELT/binary segmentation but renders candidates with nearby events and causal=false.",
      "acceptance_criteria": [
        "Drift measures refuse output when embed coverage, n_min, or pricing/usage coverage gates fail.",
        "Rendered changepoints include nearby model/harness/repo events and never emit causal wording.",
        "Synthetic step test locates a known changepoint within tolerance and labels causal=false.",
        "Live demo output includes confidence intervals, n, denominator, evidence tier, and top confounds."
      ],
      "dependencies": ["polylogue-mhx.3", "p3-03"],
      "parent": "polylogue-9e5",
      "risks": ["Intent anchors mix workloads", "Goodharting model comparisons", "Dashboard invites causal overclaim"],
      "confidence": "med",
      "provenance": "Part-2 model-drift observatory payload plus external drift/changepoint prior art."
    },
    {
      "proposed_id": "p3-17",
      "change_kind": "new",
      "target_bead": null,
      "title": "Episode and artifact unit pilot",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:units", "area:lineage", "area:artifacts", "tier:index"],
      "description": "File 3 proposes many missing units. The landable slice should prove the ontology with a small set that directly helps work reconstruction: episode, artifact, and verification-run. Artifact must avoid collision with source-tier raw_artifacts.",
      "design": "Add derived episode table/view over sessions with repo/time/embedding/shared-artifact scorer and false-merge floor. Add session_artifact view/table separate from raw_artifacts for produced/consumed/mentioned blobs/files/reports. Add verification-run view from actions using exit_code/is_error. Keep project durable-user design separate.",
      "acceptance_criteria": [
        "Synthetic multi-session task with shared commit/artifact stitches into one episode while thematic-only sessions remain candidate.",
        "session_artifact distinguishes produced, consumed, mentioned, and derived_from edges and does not reuse raw_artifacts naming.",
        "verification-run view derives pass/fail only from structural action fields, not prose.",
        "Query units expose episode, artifact, and verification-run with documented grain and identity keys."
      ],
      "dependencies": ["polylogue-4ts", "p3-14"],
      "parent": "polylogue-4ts",
      "risks": ["False episode merges", "Artifact naming collision", "Prose-mined verification claims"],
      "confidence": "med",
      "provenance": "Part-2 episode and units A/B/C/D payloads, narrowed to a proof slice."
    },
    {
      "proposed_id": "p3-18",
      "change_kind": "new",
      "target_bead": null,
      "title": "Runtime assertion query-back for cross-tier overlays",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:query", "area:assertions", "tier:user", "theme:annotations"],
      "description": "External annotations, corrections, findings, and marginalia need to be queryable alongside messages/blocks, but SQLite persistent views cannot join index.db to attached user.db. Query-back must be a runtime method or TEMP view, not canonical index DDL.",
      "design": "Implement query_assertions/query_overlay runtime path that joins user-tier assertions to index-tier targets per connection. Support filtering by kind, key, status, author_kind, scope_ref, target_ref, evidence_ref, and context_policy. Provide query DSL source assertions where ... without persistent cross-tier VIEW.",
      "acceptance_criteria": [
        "assertions where key:<x> and target:message returns rows across CLI/MCP/API using one runtime implementation.",
        "Devtools policy rejects persistent CREATE VIEW in index DDL that references user_tier.",
        "Correction-edge and annotation query-back tests pass using TEMP/runtime path.",
        "Cross-surface parity test covers assertion query results and candidate review refs."
      ],
      "dependencies": ["p3-03", "polylogue-fnm.13"],
      "parent": "polylogue-fnm",
      "risks": ["Contributor adds illegal persistent cross-tier view", "Runtime join too slow without indexes", "Agent-authored candidates accidentally included as authoritative"],
      "confidence": "high",
      "provenance": "Part-2 units/annotation-recipe payload plus SQLite ATTACH/view constraint vetting."
    },
    {
      "proposed_id": "p3-19",
      "change_kind": "new",
      "target_bead": null,
      "title": "Meta-quality scale and fault lane",
      "issue_type": "epic",
      "priority": 1,
      "labels": ["area:tests", "area:scale", "area:storage", "theme:correctness"],
      "description": "The strongest file-3 quality work is executable: identity round-trip, content-hash idempotency, lineage no-double-count, FTS coherence, daemon-vs-direct differential, convergence fixpoint, fault injection, blob-GC race, parser fuzz, and scale-shaped corpus. These should be a coherent test lane, not scattered ideas.",
      "design": "Create a scale/fault meta-quality lane under scale-hardening. Build a parameterized synthetic corpus generator and a set of property/differential/chaos tests. Keep slow/live tests behind markers but make seeded versions CI-runnable. Every future scale/correctness bead should add one regression to this lane.",
      "acceptance_criteria": [
        "Synthetic corpus generator supports sessions, messages, forks, duplicate native ids, giant artifacts, hash adversaries, and provider export variants.",
        "CI seeded lane covers identity round-trip, content-hash idempotency, FTS coherence, lineage no-double-count, and daemon-vs-direct differential.",
        "Scale marker lane runs convergence idempotency and blob-GC state-machine tests.",
        "A shipped bug class from 1xc has a corresponding regression in the lane."
      ],
      "dependencies": ["polylogue-1xc", "polylogue-83u", "polylogue-b5l"],
      "parent": "polylogue-1xc",
      "risks": ["Slow tests become ignored", "Synthetic corpus misses live failure modes", "Chaos harness flakiness"],
      "confidence": "high",
      "provenance": "Part-2 meta-quality R&D payload; judged highest leverage for credibility."
    },
    {
      "proposed_id": "p3-20",
      "change_kind": "update",
      "target_bead": "polylogue-9e5",
      "title": "Text-derived provenance contracts",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:honesty", "area:insights", "theme:construct-validity"],
      "description": "Polylogue’s structural outcome axis is strong, but other digest fields still mine prose for commits, decisions, caveats, test counts, and handler kinds. Text-derived values are not forbidden; they must be tagged and prevented from masquerading as structural facts.",
      "design": "Extend insight/payload contracts with ProvenanceClass and text_derived/unverified markers. Every number-bearing or claim-bearing insight must declare structural, provider-reported, text-derived, heuristic, fallback, or unknown provenance. Aggregate provenance propagates weakest source. Handler kind must derive from command/tool tokens or carry text-derived status.",
      "acceptance_criteria": [
        "Insight rigor audit iterates the registry and reports uncovered contracts rather than silently skipping.",
        "Prose-derived commit/test/decision/caveat fields carry text_derived or unverified in JSON payloads and render footnotes.",
        "Aggregate HWM/source provenance propagates weakest contributing source.",
        "A regression fixture mentioning a PR merge in assistant prose surfaces as candidate/unverified, never structural event."
      ],
      "dependencies": ["polylogue-f2qv", "polylogue-2qx"],
      "parent": "polylogue-9e5",
      "risks": ["Breaking consumers with stricter payloads", "Too many warnings reduce usability", "Legacy reports need migration notes"],
      "confidence": "high",
      "provenance": "Part-2 honesty audit payload and meta-quality regression ideas."
    },
    {
      "proposed_id": "p3-21",
      "change_kind": "update",
      "target_bead": "polylogue-3tl.7",
      "title": "Release cutover and install smoke matrix",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:release", "area:docs", "theme:legibility"],
      "description": "The local tree contains release workflow machinery and docs that still describe packaged installation as unavailable. Public launch depends less on new code than on cutting a verified release and reconciling install docs. Live tag state must be verified before closure.",
      "design": "Use trusted publishing/OIDC for PyPI, Sigstore attestations, CycloneDX SBOM, and install smoke tests for uvx/uv tool, pipx, Homebrew, Nix, and container image. Claims in README/getting-started must match actual channels. Avoid manual API tokens and prevent sdist fallback surprises.",
      "acceptance_criteria": [
        "Live GitHub tag/release state is checked and recorded before closing.",
        "Cold machine smoke matrix installs and runs polylogue --help plus one seeded demo command for each supported channel.",
        "PyPI publish uses trusted publisher/OIDC and release artifacts have Sigstore provenance and SBOM.",
        "Docs no longer say packaged installs are unavailable once a channel is verified."
      ],
      "dependencies": ["polylogue-3tl", "polylogue-212"],
      "parent": "polylogue-3tl",
      "risks": ["Native wheel/sdist fallback breaks install", "Docs overclaim unsupported channels", "Release distracts from flagship finding"],
      "confidence": "med",
      "provenance": "Part-2 launch/distribution payload plus local workflow/docs vetting and packaging prior art."
    },
    {
      "proposed_id": "p3-22",
      "change_kind": "new",
      "target_bead": null,
      "title": "Cross-machine sync durability boundary",
      "issue_type": "task",
      "priority": 2,
      "labels": ["area:sync", "area:storage", "theme:local-first"],
      "description": "Cross-machine sync should not copy derived or ops tiers. Durable source/user tiers can be merged only with explicit identity, conflict, and provenance rules. Source ids that include machine-local paths can duplicate the same session across peers.",
      "design": "Design durable-tier-only sync: source.db content-hash union, user.db assertion natural-key merge with LWW/terminal-status conflict policy, no derived/ops sync, and rebuild on peer. Add peer manifest with archive fingerprint, schema versions, hash algorithm, source-family/origin map, and conflict report.",
      "acceptance_criteria": [
        "Two-peer fixture merges identical source content idempotently without duplicating sessions due to source_path-local raw ids.",
        "User assertions with same natural key and conflicting values produce explicit conflict rows or deterministic terminal-status policy.",
        "Derived index/embeddings/ops files are never synced and are rebuilt on peer.",
        "Sync manifest records schema versions, content_hash_algo, origin/source_family map, and conflict counts."
      ],
      "dependencies": ["p3-13", "p3-05"],
      "parent": "polylogue-3tl",
      "risks": ["Machine-local ids cause duplicates", "User assertion conflict policy loses intent", "Looks like federation before core is ready"],
      "confidence": "med",
      "provenance": "Part-2 cross-machine sync payload, sharpened around single-writer durable-tier boundary."
    },
    {
      "proposed_id": "p3-23",
      "change_kind": "new",
      "target_bead": null,
      "title": "Claims-ledger .polydemo proof engine",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:demo", "area:findings", "theme:legibility", "theme:honesty"],
      "description": "The launch story should be one hard, reproducible finding rather than a feature catalog. Demos need to be executable product-primitive scripts whose claims resolve to queries, evidence refs, corpus datasheets, and expected values.",
      "design": "Define .polydemo format with frontmatter budget, allowed argv restricted to polylogue/devtools product primitives, finding_id = hash(claim + metric + anchors + refs + corpus datasheet hash), and refusal manifest. Connect published findings to FINDING assertions and evidence reports. Add claims-ledger for README/marketing claims with status proven/capability/aspirational.",
      "acceptance_criteria": [
        "A demo-as-CI test re-runs the flagship finding and fails on finding_id drift without an updated manifest.",
        "Parser rejects arbitrary shell/python steps outside approved polylogue/devtools argv.",
        "Rendered demo includes claim, metric, construct-validity note, evidence refs, corpus datasheet hash, and refusal manifest.",
        "README claims ledger lists every headline claim and its evidence status."
      ],
      "dependencies": ["p3-03", "p3-15", "polylogue-212", "polylogue-3tl.4"],
      "parent": "polylogue-212",
      "risks": ["Demo engine recreates monolithic script", "Methodology fight swamps product story", "Claims ledger becomes stale marketing bureaucracy"],
      "confidence": "high",
      "provenance": "Part-2 demo/proof engine, positioning, and finding-as-object payloads."
    }
  ]
}
```
