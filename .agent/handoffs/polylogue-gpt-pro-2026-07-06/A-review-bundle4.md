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

## [assistant 325]
bash -lc sed -n '466,722p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_ingest.txt; sed -n '723,979p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_delegation.txt; sed -n '980,1200p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_temporal.txt; sed -n '1201,1396p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_attach.txt; sed -n '1397,1584p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_secret.txt; sed -n '1585,1786p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_kg.txt; sed -n '1787,1983p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_notif.txt; sed -n '1984,2177p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_replay.txt; sed -n '2178,2357p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_blobgc.txt; sed -n '2358,2638p' /mnt/data/rnd-bundle-4-of-6.md > /mnt/data/part2_ideas.txt; wc -l /mnt/data/part2_*.txt

## [assistant 415]
File 4 of 6. I treated the master synthesis as orientation only and anchored the deliverable to this file’s unique Part 2 payload: ingest fidelity, delegation, temporal doctrine, attachment honesty, secret/forget, entity graph, notifications, replay, blob-GC, economics, triage/attention, query objects, ambient surfacing, self-referential analysis, non-English/accessibility, and positioning. The uploaded bundle itself says Part 2 is the unique payload after the divider, so the bead proposals below are scoped to file 4 only. fileciteturn1file16

## Executive verdict

This slice is unusually strong. The best specs are not “ideas”; they are close to implementation briefs with correct tier placement, realistic tests, and good construct-validity instincts. I would trust four clusters enough to turn directly into work: attachment referenced-vs-stored honesty, secret/forget + reset dry-run contract, blob-GC state-machine hardening, and ingest fidelity/fingerprint tracking. Those are credibility work: they prevent Polylogue from lying about bytes, mutating the wrong target, losing parse coverage silently, or deleting data during GC.

The largest correction is that several specs are overconfident about “already 90% built.” The substrate is often 60–80% present, but the missing 20% is usually the product boundary: queryability, object refs, import/review ergonomics, generated contracts, and recursive-safety chokepoints. That difference matters. It is the difference between “we have rows that could support this” and “agents can reliably use this as an analysis substrate.”

The highest-leverage strategic improvement is to make **query/result/annotation/analysis objects** first-class sooner. That makes many Part-2 ideas compose: delegation demo, Fable analysis, standing queries, ambient notifications, external annotation batches, context packs, triage, and self-referential Polylogue analysis.

## Hard vetting

### Confirmed

The ingest fidelity design has the right tier split. `raw_fidelity` is derived from blob plus parser and belongs in `index.db`; `parser_fingerprint` is the durable parse-identity switch that belongs with `raw_sessions`. The spec’s `detect_all()` addition is also directionally right because today detection is first-match, and the dispatch blind-spot around sampling only the first 32 JSONL records is a real class of loss signal to record. The file explicitly calls out the same tier rationale: derived ratios should not be durable, and only the decision input needs to survive rebuilds. fileciteturn1file11 The proposed F1–F7 breakdown is landable: `detect_all`, source v3 fingerprint/decode class, `raw_fidelity`, ambiguity/tripwires, drop accounting, parser-drift convergence, and golden/property tests. fileciteturn1file10

The attachment honesty diagnosis is real and important. The archive already has the load-bearing truth: `byte_count`, nullable `blob_hash`, and `acquisition_status`. The bug is projection: read surfaces expose a bare `size_bytes`/declared byte number without saying whether bytes were actually acquired. The spec is correct that this is not a schema migration; it is a projection, payload, accounting, and lint problem. fileciteturn1file0 The two-sum invariant is the right product contract: `referenced_bytes` must be separated from `stored_bytes`, with `unfetched` and `unavailable` kept explicit; a citation to a non-stored attachment must never render as a confident retrievable hit. fileciteturn1file4

The secret/forget spec identifies a real mutation safety failure. The reset path can resolve an unknown token into a suppression target instead of failing closed, and mutation is behind confirmation rather than behind a typed resolved-target dry-run. The content-hash resurrection vector is also real: if excised content is re-ingested from the original source, idempotency needs a tombstone guard or the content can reappear. The spec’s split into dry-run mutation contract, source tombstones, excision engine, idempotency guard, secret candidate queue, read overlay, and reconciliation stage is coherent. fileciteturn2file1 fileciteturn2file5

The delegation unit is well-framed. A delegation should not mean “any child session.” It should mean a parent-dispatched subagent edge with a structural parent-side task/action, returned artifact, and outcome fields; Codex-style sidechains without parent task result data should stay visible as `unknown` and excluded from ROI/yield denominators. That is exactly the construct-validity line the demo needs. fileciteturn2file9 The DSL/unit/card/yield breakdown is also right: the demo should be built over a generic `delegation` relation, not as Fable-specific logic. fileciteturn2file14

The blob-GC hardening spec correctly says the implementation is not absent. The current system already has leases, generation gates, GC history, failure-path cleanup, and nontrivial tests. The missing work is a stronger state-machine model, SIGKILL/fault coverage, and reconciliation between divergent orphan-detection surfaces. That is a high-quality correction: do not rewrite GC; prove its invariants. fileciteturn1file18 Hypothesis’s stateful testing model is a good fit for this exact class of interleaving invariant, because it defines operations and checks invariants across generated state transitions rather than only checking fixed examples. citeturn603773search0

The non-English/accessibility slice surfaces real gaps. The SQLite FTS5 `unicode61` issue around Polish `ł/Ł` is not imaginary; SQLite maintainers have explained that the relevant characters lack decomposition data, so `remove_diacritics` does not fold them the way users expect. citeturn222185search0 The Part-2 text’s proposed split between natural-language detection, query/index normalization symmetry, trigram/Polish expansion fallback, reply-language preference, TTS/STT, and a true screen-reader/plain mode is useful. fileciteturn2file6 fileciteturn2file3

The query-objects/annotation direction is exactly right. The assertions table already has the right bones: scope, target, evidence refs, staleness, supersession, confidence, author kind, and assertion kinds such as saved query, recall pack, workspace note, decision, blocker, lesson, and judgment. The spec correctly identifies the missing frontier: saved queries are currently inert blobs/strings, not referenceable query objects with result-set memory and query→query dependency edges. fileciteturn1file17

### Corrected or narrowed

The temporal bug is real, but one diagnosis should be narrowed. The file family says `RELATIVE_BASE = datetime.now()` behaves as an import/process-start anchor; in the code snapshot I inspected, the visible direct `datetime.now()` base is not a global import-time constant but a wall-clock call inside date parsing. That still needs a `core/clock.py` seam and hermetic `frozen_clock` tests, but the precise failure mode should be described as “uncontrolled wall-clock in relative-date parsing,” not necessarily “process-start-stale relative base.” The temporal design’s acceptance tests remain right: hermetic relative windows, half-open `[since, until)`, explicit timeless handling, comparator properties, and backfill invariants. fileciteturn2file10

The `sort_key_ms=COALESCE(...,0)` issue is real but not uniform across all read paths. Some ordering paths already bucket NULLs explicitly; others do coalesce to zero or otherwise risk synthetic 1970 behavior. The bead should start with a targeted ordering/window audit and then replace bad sites, not assume every sort path is equally broken. The design’s deeper move is still correct: `sort_key_ms` needs provenance/confidence, NULL/timeless rows need explicit semantics, and cursor stability must be protected by a backfill invariant. fileciteturn2file13

Delegation should not be implemented as “just a VIEW” without an audit pass. The proposed view is plausible, but the join `actions.message_id = branch_point_message_id` may miss cases where the parent Task dispatch is “near” the branch point rather than exactly equal, or where the provider emits subagent relationships differently. The right first bead is an audit/backfill over a seed/live sample that computes match rate: subagent links, parent dispatch action found, returned artifact found, `unknown` rate by origin. Only after that should the public yield measure ship. The Part-2 risk section already anticipates `unknown` dominating non-Claude; the backlog should encode that as a gating metric, not a footnote. fileciteturn2file4

The ingest unparsed-key census via a read-tracking proxy is promising, but overconfident. A proxy catches `.get`/`__getitem__` reads, but parsers may copy dicts, hand payloads to Pydantic, iterate keys, destructure, or consume nested objects in ways that the proxy does not faithfully attribute. Treat the census as a best-effort coverage detector until each parser opts into an explicit consumed-key contract or golden corpus comparison. The golden “key-set only shrinks” test is the stronger guard.

The notification specs need a stricter anti-feedback rule. Recording a filtered/unsafe signal into an ops ledger is fine; emitting it as a user-facing content notice or injecting it into a context preamble is not. Alerting prior art strongly favors actionable, user-relevant alerts and warns against alert noise; Polylogue should additionally require citation anchors and recursive-safety provenance checks before surfacing content notices. citeturn216365search1 The Part-2 notification spec already gestures at recursive gates and suppression assertions; I would make them first-class acceptance criteria, not optional polish. fileciteturn1file5

“Verifiable forget” should be named carefully. The design can provide **logical archive forget** and **reingest resurrection prevention**. It should not imply cryptographic or physical erasure from filesystem slack, WAL history, backups, copied exports, external provider logs, or old bundles unless those scopes are explicitly included. NIST media-sanitization guidance frames sanitization as making access to target data infeasible on the relevant media; Polylogue’s default scope is narrower than whole-media sanitization, so the product language must say so. citeturn222185search3

Archive economics is useful but must be measurement-gated. Zstd dictionaries, cold shards, embedding pruning, and content-defined chunking are credible directions, but the first bead should be “measure dedup/rebuild/storage dividend,” not “implement FastCDC.” FastCDC is real prior art for efficient content-defined chunking, but whether Polylogue benefits depends on the distribution of large near-duplicate blobs and rebuild cost. citeturn646079search14

The replay/reproduction design is appropriately cautious. It should stay a two-phase feature: dry reconstruction plan first, sandboxed execution second, and no deterministic replay claims by default. ReproZip, rr, and Nix show the design space: capture execution environments, record/replay deterministic executions, or rebuild in declarative isolated environments. Polylogue’s session logs are weaker than those systems’ full execution capture, so “reconstruct/reproduce plan” is the honest baseline. citeturn646079search0turn646079search1turn646079search20

### Extended recommendations

The file’s most important missing unifier is an explicit **analysis object model**. Query objects, result sets, annotation batches, standing queries, context packs, delegation cards, triage frontiers, ambient notices, and reports are currently described in separate specs. They should share one object vocabulary:

`query_definition`, `query_run`, `result_relation`, `cohort`, `annotation_batch`, `analysis_run`, `evidence_pack`, `report_artifact`.

This avoids building “delegation demo,” “triage inbox,” “ambient notifications,” and “query history” as separate systems. A query run can feed a result relation; a result relation can become a snapshot cohort; an annotation batch can target that cohort; an analysis run can link the query runs, evidence pack, imported judgments, and final report. That is the substrate needed for the Fable/delegation demo, external annotations, and Polylogue tracking its own use.

OpenTelemetry’s GenAI semantic conventions are relevant here: model names, token counts, prompts, completions, tool calls, and tool results are becoming a shared telemetry vocabulary. Polylogue should not simply clone OTel, because it has stronger archive/evidence needs, but the replay/delegation/observability work should map outward to OTel concepts where possible. citeturn603773search1

The entity graph is valuable but should be staged conservatively. Structural entity extraction from URLs, issue refs, commits, repo paths, and action evidence is high value. Candidate prose mining is dangerous in a self-archiving system; the Part-2 risk around recursive fabrication is correct. fileciteturn1file2 Ship structural mentions and query surfaces first. Treat prose-mined mentions as candidate assertions requiring the same recursive-safety gate as external annotations.

The accessibility slice should be split between **machine mode** and **human accessibility**. `--json` is not screen-reader UX. A true `--plain` or `--a11y` mode should avoid Rich tables/box glyphs and render linear labeled text. WCAG’s “color is not the only visual means” principle is a useful baseline for terminal/web cockpit output, even though Polylogue is not a conventional website. citeturn216365search6

## Prioritized take

Tier 0, credibility and safety: attachment two-sum honesty, reset/excision dry-run, tombstone resurrection guard, temporal clock/window fixes, and blob-GC state-machine tests. These prevent data lies and destructive mistakes.

Tier 1, analysis substrate: query objects, result relations/cohorts, annotation batches, assertion query-back, delegation unit, and evidence packs. These turn Polylogue from a search surface into an analysis workbench.

Tier 2, ambient/context: semantic notifications, triage frontier, attention/read-access logs, context preambles, and standing queries. These are powerful only after recursive-safety and object refs are in place.

Tier 3, breadth/frontier: entity graph, replay execution, archive economics, non-English recall, TTS/STT/accessibility, and public positioning. These matter, but several should be measurement-gated or proof-artifact-gated.

The bead set below is deliberately scoped to file 4. I used existing bead ids where they were clearly available from the current bundle/bead context; where a file-4 spec implies new work rather than an existing child, I used `new` and linked it to the nearest extant epic.

```json
{
  "file": 4,
  "proposed_beads": [
    {
      "proposed_id": "p4-01",
      "change_kind": "new",
      "target_bead": null,
      "title": "Ingest detector candidates and parser fingerprints",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:ingest", "area:storage", "tier:source", "construct-validity"],
      "description": "Ingest fidelity cannot currently answer whether acquired bytes were parsed under the current parser/detector regime. The source tier has raw payloads and parse outcome fields, and dispatch is first-match; Part 2 correctly identifies parser_fingerprint and detect_all as the foundation for reprocess-on-parser-improvement and ambiguity accounting.",
      "design": "Add detect_all(payload,path) returning ranked DetectionCandidate rows while keeping detect_provider behavior byte-identical. Add source.db durable migration for raw_sessions.parser_fingerprint and raw_artifacts.decode_failure_class. Fingerprint must be stable across daemon restarts and composed from detector rules plus parser semver, not timestamps. Batch with the next source-tier migration.",
      "acceptance_criteria": [
        "detect_provider golden corpus output is byte-identical before and after detect_all lands",
        "detect_all returns at least one candidate for every non-empty known-origin fixture and preserves tightness ordering",
        "source.db migration is additive, single user_version step, and passes schema-versioning policy with backup manifest",
        "parse path stamps parser_fingerprint and raw_artifacts decode_failure_class is validated by enum/CHECK rendering"
      ],
      "dependencies": ["polylogue-2qx"],
      "parent": "polylogue-2qx",
      "risks": [
        "Fingerprint churn can enqueue the whole archive repeatedly",
        "Detector ambiguity scoring can become cargo-cult confidence if not calibrated",
        "Durable migration should be batched with other source v3 work"
      ],
      "confidence": "high",
      "provenance": "Ingest Fidelity F1/F2; confirmed source-tier DDL and dispatch gaps"
    },
    {
      "proposed_id": "p4-02",
      "change_kind": "new",
      "target_bead": null,
      "title": "Raw fidelity materializer and parser-drift stage",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:ingest", "area:index", "area:daemon", "construct-validity"],
      "description": "Polylogue lacks a rebuildable read model answering how much of a raw payload survived parsing, what keys were unparsed, whether detection was ambiguous, and whether parser improvements require reprocessing.",
      "design": "Add index.db raw_fidelity as a derived table populated during materialization: byte-fidelity band, roundtrip status, unparsed-key census, detection ambiguity, misclassification tripwire, zero-message anomaly, drop accounting, computed parser_fingerprint. Add a bounded parser_drift convergence stage using false_means_pending. Treat parser fingerprint bumps as reviewed batch operations.",
      "acceptance_criteria": [
        "Index rebuild populates raw_fidelity 1:1 for parsed raw_sessions",
        "Golden corpus fails when a previously parsed key regresses to unparsed and allows shrink-only updates when parsers improve",
        "Streaming blind-spot fixture with signature after record 32 records stream_window_blind and does not silently fall to unknown",
        "Bumping parser semver enqueues only stale-fingerprint rows and respects a bounded window per quiet cycle"
      ],
      "dependencies": ["p4-01"],
      "parent": "polylogue-2qx",
      "risks": [
        "Byte-fidelity ratio can be misread as exact preservation",
        "Read-tracking proxy may miss parser consumption through copies/Pydantic",
        "Roundtrip equality is unavailable for some origins"
      ],
      "confidence": "med",
      "provenance": "Ingest Fidelity F3-F8; corrected proxy census to be best-effort unless parser opt-in is explicit"
    },
    {
      "proposed_id": "p4-03",
      "change_kind": "new",
      "target_bead": null,
      "title": "Temporal clock seam, windows, and sort provenance",
      "issue_type": "bug",
      "priority": 1,
      "labels": ["area:query", "area:time", "area:index", "correctness"],
      "description": "Relative-time parsing and sort/window behavior need a single clock seam and explicit timeless semantics. The uploaded spec correctly requires hermetic time tests, half-open windows, and sort provenance, but the diagnosis should be narrowed from import-time RELATIVE_BASE to uncontrolled wall-clock use in the current code snapshot.",
      "design": "Add core/clock.py and route relative-date parsing through it. Change query windows to half-open [since,until). Audit COALESCE-to-zero ordering/window paths and replace with explicit NULL buckets and time_confidence. Add TimeKind, SortKeyProvenance, TimeConfidence, and TzProvenance in generated schemas. Preserve cursor stability with a backfill-never-moves-sort_key property test.",
      "acceptance_criteria": [
        "since:7d under frozen_clock is deterministic and advances only when frozen_clock advances",
        "Adjacent windows [t0,t1) and [t1,t2) contain boundary rows exactly once",
        "Timeless sessions are not assigned synthetic 1970 positions and are surfaced with explicit confidence when include_timeless is requested",
        "Property tests cover comparator totality, antisymmetry, transitivity, stability, and band-idempotence",
        "Audit report identifies every replaced COALESCE-to-zero path and any intentional exceptions"
      ],
      "dependencies": ["polylogue-cpf"],
      "parent": "polylogue-cpf",
      "risks": [
        "Changing sort_key semantics can reorder history and break cursors",
        "Pairwise skew comparison can violate transitivity",
        "A partial clock seam leaves hidden wall-clock calls"
      ],
      "confidence": "high",
      "provenance": "Temporal Doctrine; confirmed need, corrected precise RELATIVE_BASE failure mode"
    },
    {
      "proposed_id": "p4-04",
      "change_kind": "update",
      "target_bead": "polylogue-83u.6",
      "title": "Attachment acquisition evidence census",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:attachments", "area:storage", "honesty", "read-only-smoke"],
      "description": "Attachment rows already distinguish acquired/unfetched/unavailable, but current read paths and diagnostics do not make the acquisition split visible. 83u.6 should become the read-only baseline census over origin x acquisition_status before aggregation surfaces are changed.",
      "design": "Run mode=ro census over attachments joined to attachment_refs. Report referenced byte sums, acquired blob counts, stored bytes on disk, unfetched/unavailable counts, acquired-file-missing floor breaches, and top source_ref classes. Reconcile against existing blob-reference-debt diagnostics. Persist JSON and markdown baseline as a research artifact.",
      "acceptance_criteria": [
        "Census opens index/source read-only and never obtains a write connection",
        "Grouped origin x acquisition_status totals reconcile with direct SQL and blob-reference-debt diagnostics",
        "acquired rows with missing files are counted as acquisition debt, not unfetched debt",
        "Baseline artifact is attached to 83u with live archive command and timestamp"
      ],
      "dependencies": ["polylogue-83u"],
      "parent": "polylogue-83u",
      "risks": [
        "Live archive figures can drift; baseline must record archive/version",
        "Statting blob files can be slow without batching",
        "Confusing source-tier missing blobs with index-tier unfetched attachments"
      ],
      "confidence": "high",
      "provenance": "Attachment honesty B4; confirmed no schema change required"
    },
    {
      "proposed_id": "p4-05",
      "change_kind": "new",
      "target_bead": null,
      "title": "Attachment two-sum payloads and honesty lint",
      "issue_type": "bug",
      "priority": 1,
      "labels": ["area:attachments", "area:surfaces", "honesty", "area:devtools"],
      "description": "Surfaces currently risk projecting declared attachment bytes as retrievable bytes. This is a correctness and trust bug, not an analytics nicety.",
      "design": "Project acquisition_status and blob_hash through attachment read queries; add AttachmentEvidence plus referenced_bytes, stored_bytes, evidence, and blob_hash_hex to internal and public models while retaining size_bytes as referenced-not-retrievable. Replace aggregate single-byte fields with attachment_referenced_bytes and attachment_stored_bytes. Add a devtools policy forbidding bare attachment byte totals.",
      "acceptance_criteria": [
        "Seed with acquired, unfetched, and unavailable attachments returns correct evidence and stored_bytes values",
        "No new surface emits a bare attachment_bytes/size/total field for attachment bytes",
        "Session profile, stats, coverage, cost, and MCP payloads expose referenced and stored byte sums separately",
        "OpenAPI and CLI output schemas regenerate cleanly",
        "Citation resolver renders exists/unfetched/permanently_gone as distinct states"
      ],
      "dependencies": ["p4-04"],
      "parent": "polylogue-83u",
      "risks": [
        "Backward compatibility pressure may keep size_bytes misleading",
        "Per-blob stat on hot paths can be expensive",
        "Schema regeneration drift"
      ],
      "confidence": "high",
      "provenance": "Attachment honesty B1-B3/B6; confirmed projection bug"
    },
    {
      "proposed_id": "p4-06",
      "change_kind": "update",
      "target_bead": "polylogue-27m",
      "title": "Resolved-target dry run for reset and excision",
      "issue_type": "bug",
      "priority": 1,
      "labels": ["area:security", "area:cli", "mutation-safety"],
      "description": "The reset/excision path can mutate based on unverified resolved tokens. A typo can become a suppression target, and --yes bypasses a resolved-target dry run. This is the highest-risk user-facing bug in the file-4 security slice.",
      "design": "Introduce one mutation-audit contract for reset/excise/delete/admin actions: resolve all targets, render exact target count and refs, require --yes only after resolution, and make zero-target/unknown-token a non-mutating result. Route reset --session/--source through the shared contract before suppression or tombstone writes.",
      "acceptance_criteria": [
        "Typo session/source ref produces zero-target dry run and leaves suppression/user state unchanged",
        "Real ref mutates only with --yes after resolved-target preview",
        "MCP/admin mutation paths use the same resolver contract or are explicitly out of scope",
        "Regression tests cover nonexistent ref, ambiguous ref, and valid ref"
      ],
      "dependencies": ["polylogue-kwsb"],
      "parent": "polylogue-kwsb",
      "risks": [
        "Breaking existing scripts that relied on permissive reset",
        "Resolver differences across CLI/MCP",
        "Preview must not leak secret span contents"
      ],
      "confidence": "high",
      "provenance": "Secret-redaction/forget jnj.5 finding; confirmed reset failure mode"
    },
    {
      "proposed_id": "p4-07",
      "change_kind": "new",
      "target_bead": null,
      "title": "Logical forget tombstones and resurrection guard",
      "issue_type": "feature",
      "priority": 1,
      "labels": ["area:security", "area:source-tier", "area:ingest", "privacy"],
      "description": "Excision without a tombstone can be undone by re-ingesting original source bytes. Polylogue needs logical archive forget with an explicit scope: prevent archive resurrection and reconcile derived tiers, without overclaiming whole-media sanitization.",
      "design": "Add source.db excision_tombstones with original content hash, excised content hash, span_text_hash/block hash, reason, and audit metadata. Excision engine removes or masks content across source/index/FTS/embeddings/blob refs, writes tombstone, and reconciles reingest by tombstone lookup before accepting unchanged content. Product language says logical archive forget, not physical sanitization of backups/slack/provider copies.",
      "acceptance_criteria": [
        "Source migration is additive and backup-manifest gated",
        "Excising a seeded block removes it from source projection, index rows, FTS, embeddings refs, and read views",
        "Re-ingesting the original source does not resurrect excised content",
        "Reconciliation stage detects injected resurrection by content hash and span_text_hash",
        "Docs define forget scope and explicitly exclude unmanaged backups/provider copies"
      ],
      "dependencies": ["p4-06"],
      "parent": "polylogue-kwsb",
      "risks": [
        "Parser drift can evade content-hash tombstones",
        "SQLite/WAL/backup remnants can make physical erasure claims false",
        "Over-masking can destroy evidence needed for audits"
      ],
      "confidence": "med",
      "provenance": "Secret-redaction/forget B3-B8; narrowed verifiable-forget claim to logical archive scope"
    },
    {
      "proposed_id": "p4-08",
      "change_kind": "update",
      "target_bead": "polylogue-8jg9.2",
      "title": "Blob-GC state machine and orphan-surface reconciliation",
      "issue_type": "task",
      "priority": 1,
      "labels": ["area:storage", "area:gc", "property-testing", "ops"],
      "description": "Blob GC already has leases, generation gates, startup sweep, and nontrivial tests, but the closure gap remains: prove acquire→commit→GC interleavings as a state machine and reconcile divergent orphan-detection paths.",
      "design": "Add Hypothesis RuleBasedStateMachine over disk set, blob_refs, raw_sessions refs, pending_blob_refs, and gc_generations. Assert never delete reachable or leased blobs and eventual deletion only after generation/age gates. Add SIGKILL/failure-path injection where feasible. Unify or cross-check ops doctor cleanup_orphans against lease/ref/generation-safe run_blob_gc.",
      "acceptance_criteria": [
        "State machine fails against a deliberately unsafe GC implementation and passes current safe path",
        "Tests cover lease acquired before commit, commit failure, lease sweep, generation age gate, and concurrent orphan creation",
        "ops doctor cleanup_orphans cannot delete pending or referenced blobs",
        "Divergent orphan reports produce identical safe-to-delete sets on seeded fixtures"
      ],
      "dependencies": ["polylogue-8jg9"],
      "parent": "polylogue-8jg9",
      "risks": [
        "State model can omit a real reference surface",
        "Fault injection may be flaky if it relies on signals/timing",
        "Doctor path may need product behavior changes"
      ],
      "confidence": "high",
      "provenance": "Blob-GC spec; confirmed implementation exists and remaining gap is proof/reconciliation"
    },
    {
      "proposed_id": "p4-09",
      "change_kind": "new",
      "target_bead": null,
      "title": "Delegation relation, card, yield, and demo",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:query", "area:analytics", "area:demo", "construct-validity"],
      "description": "The Fable/subagent analysis needs a generic delegation unit rather than a one-off demo. A delegation must be a structurally anchored parent dispatch with returned artifact and outcome, with unknowns separated honestly.",
      "design": "Add primary_model_name/family to session_profiles, audit subagent links to parent dispatch actions, then add delegations view/query method, DelegationQueryRowPayload, DSL unit, completion fields, delegation-card projection, delegation-yield measure, and a polydemo seed. Gate ROI/yield on structural action outcome only; unknown status remains visible and denominator-excluded.",
      "acceptance_criteria": [
        "Audit reports match rate for subagent links to parent Task/subagent actions by origin before public yield ships",
        "delegations where ... | group by ... | count works through the query unit infrastructure",
        "result_status derives only from action/tool_result evidence and unknown is never folded into ok/error denominators",
        "delegation-card lines are explain-resolvable to session/action/block refs",
        "Demo fixture includes at least three dispatches, two model families, one error, and one unknown"
      ],
      "dependencies": ["polylogue-fnm", "polylogue-f2qv"],
      "parent": "polylogue-fnm",
      "risks": [
        "Parent dispatch action may be near rather than exactly at branch point",
        "Unknown status may dominate non-Claude providers",
        "Model-family mapping can leak provider vocabulary or collapse non-injectively"
      ],
      "confidence": "med",
      "provenance": "Delegation spec; corrected view-only confidence with required match-rate audit"
    },
    {
      "proposed_id": "p4-10",
      "change_kind": "new",
      "target_bead": null,
      "title": "Query objects, result relations, and cohorts",
      "issue_type": "epic",
      "priority": 2,
      "labels": ["area:query", "area:assertions", "area:analysis", "object-refs"],
      "description": "Saved queries are inert today: they cannot accumulate annotations, expose result-set memory, participate in query-edge DAGs, or serve as stable analysis operands. This blocks scalable external annotation and self-referential Polylogue analysis.",
      "design": "Introduce query_definition, query_run, result_relation, and cohort object refs. Store reusable lowered/canonical query definitions in user tier when promoted; store routine query runs/results in ops tier by default; allow promotion to user tier when cited, named, watched, or used by an analysis. Add assertion targets/evidence refs for query/result/cohort, query dependency edges, and standing-query delta candidates.",
      "acceptance_criteria": [
        "Every JSON query execution can return query_run_ref and result_relation_ref with grain, count, exactness, and degraded state",
        "A named query can be referenced as query:<id> and targeted by an assertion",
        "A snapshot cohort freezes member refs and a dynamic cohort stores the query_definition",
        "Standing query result deltas produce candidate assertions without self-injection",
        "Query edges mark dependent cached results stale when a base query changes"
      ],
      "dependencies": ["polylogue-fnm.13", "polylogue-37t.12"],
      "parent": "polylogue-37t",
      "risks": [
        "Recording every query durably can create noise",
        "Canonical query identity can drift across grammar evolution",
        "Cross-tier joins with user assertions must avoid persistent attached-db views"
      ],
      "confidence": "high",
      "provenance": "Query/finding objects and saved-query slice; extended into split query_definition/query_run/result_relation/cohort model"
    },
    {
      "proposed_id": "p4-11",
      "change_kind": "update",
      "target_bead": "polylogue-37t.12",
      "title": "Batch assertion import and query-back",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:assertions", "area:agents", "recursive-safety", "analysis"],
      "description": "External annotation workflows are technically possible through assertions, but agent ergonomics and safety are incomplete: no batch schema import, evidence-ref validation, query-back surface, or single chokepoint forcing agent-authored rows to candidate + inject:false.",
      "design": "Add assertion schema registry and JSONL batch import. Each row targets a session/message/block/action/query/result/cohort ref and carries evidence_refs, author_kind, model/prompt metadata, confidence, and status. Enforce non-user author safety inside upsert_assertion, never per caller. Add DSL/MCP query-back for assertions by key/status/scope/target.",
      "acceptance_criteria": [
        "assertions import validates schema, target refs, evidence refs, and row counts before writing",
        "Agent-authored rows always land candidate and context_policy.inject=false unless promoted by user judgment",
        "Rejected/superseded terminal rows are never resurrected by a later agent import",
        "find assertions where key:... and status:candidate works across CLI/MCP",
        "Batch metadata records model, prompt/ref, created_at, row_count, accepted/rejected counts"
      ],
      "dependencies": ["p4-10"],
      "parent": "polylogue-37t",
      "risks": [
        "Unsafe agent-authored assertions can launder into context",
        "Too-strict validation can block useful exploratory annotation",
        "Assertion query-back over user tier must respect SQLite attached-db constraints"
      ],
      "confidence": "high",
      "provenance": "External annotation/query-object slices; confirmed assertion substrate exists but ergonomics/safety missing"
    },
    {
      "proposed_id": "p4-12",
      "change_kind": "new",
      "target_bead": null,
      "title": "Structural entity graph with gated candidate mentions",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:graph", "area:index", "recursive-safety", "area:query"],
      "description": "A second-brain graph is useful only if structural mentions and prose-mined candidates are kept separate. Self-archived sessions about Polylogue can otherwise regenerate fabricated refs as graph truth.",
      "design": "Ship structural entity extraction first: GitHub issues/PRs from URLs/actions, commits from structural commit refs, repos/projects/files from existing tables. Add entities/entity_mentions/entity_topics and query surfaces. Prose-mined mentions are candidate assertions only, gated by material_origin/trust and never promoted to structural edges without review.",
      "acceptance_criteria": [
        "Bare #N resolves repo-scoped and never merges across repos without structural URL evidence",
        "Structural entity mentions are queryable by entity/repo/source with explainable evidence refs",
        "Candidate prose mentions carry provenance class and cannot enter structural graph tables",
        "Regression fixture prevents runtime/context-pack prose from creating structural issue/commit truth",
        "Convergence stage is bounded and versioned through insight_materialization"
      ],
      "dependencies": ["polylogue-37t.12"],
      "parent": "polylogue-9l5",
      "risks": [
        "Prose mining can create recursive fabrication loops",
        "Bare refs are ambiguous across repos and time",
        "Entity clustering can become expensive at 38GB scale"
      ],
      "confidence": "med",
      "provenance": "Second-brain graph spec; staged structural-first to reduce self-archive fabrication risk"
    },
    {
      "proposed_id": "p4-13",
      "change_kind": "new",
      "target_bead": null,
      "title": "Semantic notices with fatigue and recursive-safety gates",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:daemon", "area:notifications", "recursive-safety", "attention"],
      "description": "Ambient surfacing is valuable only if it avoids alert fatigue and self-referential loops. Notices must be actionable, evidence-cited, suppressible, and excluded from feedback loops over generated context.",
      "design": "Extend daemon notice pipeline for content notices from standing queries, repeated-mistake similarity, unfinished-thread surfacing, and cost anomalies. Every content notice carries citation anchors, provenance class, confidence, policy id, and suppression key. Add per-family token buckets, durable suppression assertions with TTL, no-self-alert filters, and a filtered-event ledger for audit.",
      "acceptance_criteria": [
        "Standing query notice fires only after baseline and dedups by stable finding identity",
        "Notice payload includes citation refs, provenance class, confidence, and policy id",
        "Generated context/runtime/tool-result material cannot trigger user-facing content notices",
        "Suppression assertion mutes matching notices until TTL or explicit retract",
        "Token-bucket tests prove fatigue control under repeated matching events"
      ],
      "dependencies": ["p4-10", "p4-11"],
      "parent": "polylogue-37t",
      "risks": [
        "Notifications become ignored noise",
        "Self-generated notices can recursively trigger themselves",
        "Similarity-based repeated-mistake nudges can overfit weak embeddings"
      ],
      "confidence": "med",
      "provenance": "Semantic notification and proactive ambient-surfacing slices; hardened with alerting prior art"
    },
    {
      "proposed_id": "p4-14",
      "change_kind": "new",
      "target_bead": null,
      "title": "Triage frontier and read-access attention ledger",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:triage", "area:ops", "attention", "area:query"],
      "description": "Heuristic detectors over immutable sessions will nag forever unless detector output is converted into an inbox/frontier with verdicts, snooze, audit, clustering, and attention state.",
      "design": "Add ops read-access log and triage_frontier projection over detector candidates. State lives as user assertions for durable triage verdicts/snoozes and ops samples for attention. Today's frontier excludes dismissed/snoozed/superseded items, clusters related items, and ranks by worth-reviewing score with staleness and access history. Exclude context_inject reads from attention to avoid self-reinforcement.",
      "acceptance_criteria": [
        "Triage verdicts include keep/dismiss/snooze/cluster-dismiss with actor, reason, and timestamp",
        "today frontier query returns no dismissed items and wakes snoozed items only after wake time",
        "Read-access events are debounced, stored in ops, and exclude context injection reads",
        "Cluster-level dismissal applies to future detector hits sharing the cluster key with an audit trail",
        "Frontier render shows why each item is present and how to suppress it"
      ],
      "dependencies": ["p4-10", "p4-13"],
      "parent": "polylogue-37t",
      "risks": [
        "Attention ledger can become surveillance noise",
        "Bad clustering can hide valuable items",
        "Staleness and access signals can be gamed by the UI itself"
      ],
      "confidence": "med",
      "provenance": "Attention/Triage/Human Frontier plus read-access-log idea"
    },
    {
      "proposed_id": "p4-15",
      "change_kind": "update",
      "target_bead": "polylogue-bby.12",
      "title": "Session replay dry plans before execution",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:replay", "area:ops", "safety", "agent-work"],
      "description": "Polylogue has enough action/session/repo evidence to reconstruct many agent sessions, but not enough to promise deterministic replay by default. The feature should start as a dry replay plan with explicit refusal reasons.",
      "design": "Implement ReplayPlan from session_runs, actions, session_commits, and observed events. Persist replay_runs/replay_steps in ops only for executed/evaluated runs. Phase 1 prints plans and refusal reasons. Phase 2 requires explicit --execute, explicit/origin-reported anchor SHA, sandboxed worktree, network deny, frozen clock, fail-closed unknown step classification, and checkpointed cursor.",
      "acceptance_criteria": [
        "Dry plan for seeded Claude/Codex sessions lists ordered tool steps, cwd, anchor source, classification, and recorded outcomes",
        "Sessions with capture gaps, no cwd, non-agentic origins, network steps, or unknown anchors refuse with explicit skip_reason",
        "Execute mode cannot run without explicit opt-in and an anchor",
        "Sandbox prevents writes outside worktree and denies network by default",
        "Replay result classifies match/regressed/newly_passes/divergent/skipped without producing RL rewards for unknowns"
      ],
      "dependencies": ["polylogue-bby.12", "polylogue-8jg9"],
      "parent": "polylogue-bby",
      "risks": [
        "False determinism claims from incomplete action streams",
        "Side effects escaping sandbox",
        "Environment drift masquerading as model regression"
      ],
      "confidence": "med",
      "provenance": "Session Replay/Reproduction spec; narrowed to dry reconstruction-first"
    },
    {
      "proposed_id": "p4-16",
      "change_kind": "new",
      "target_bead": null,
      "title": "Archive economics measurement before tiering",
      "issue_type": "task",
      "priority": 3,
      "labels": ["area:storage", "area:economics", "measurement", "area:embeddings"],
      "description": "Archive tiering, zstd dictionaries, FastCDC, cold shards, and embedding pruning may matter at 10-year scale, but implementation should be gated by measured storage/rebuild/dedup dividends.",
      "design": "Produce an archive economics report: source/blob/index/embeddings size by origin and heat; lineage/message/blob dedup dividend; derived rebuild wall time and estimated API dollar cost; embedding size and cold-session pruning estimate; large-blob near-duplicate sample to decide whether FastCDC/content-defined chunking is justified.",
      "acceptance_criteria": [
        "Report includes per-tier size, per-origin bytes/session, and hot/warm/cold access-temperature distribution",
        "Derived rebuild cost records wall-clock, CPU, and embedding API cost estimate from a measured or dry-run basis",
        "Dedup dividend separates lineage message dedup, full-file blob dedup, and candidate near-duplicate large blobs",
        "Recommendation explicitly says build or do not build FastCDC based on measured threshold",
        "No storage layout changes land in this bead"
      ],
      "dependencies": ["polylogue-83u", "polylogue-mhx"],
      "parent": "polylogue-83u",
      "risks": [
        "Measurements can be stale if archive changes quickly",
        "Heat tracking can add write overhead",
        "Premature optimization can displace correctness work"
      ],
      "confidence": "high",
      "provenance": "Archive Economics & Longevity; measurement-gated correction to implementation-heavy ideas"
    },
    {
      "proposed_id": "p4-17",
      "change_kind": "update",
      "target_bead": "polylogue-0v9p",
      "title": "Human-language facts and reply preference",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:language", "area:context", "area:variants", "accessibility"],
      "description": "Current language facts focus on programming-language/content variants, while natural human language is invisible as a query/context dimension. This hurts Polish/English workflows and reply-language continuity.",
      "design": "Add derived prose_lang facts over human/assistant authored prose blocks, distinct from programming-language detection. Surface lang: filters, per-language stats, and a reply_language preference stored as assertion/settings and injected through context packs only when policy allows. Keep tool output/code blocks excluded or separately classified.",
      "acceptance_criteria": [
        "Seed corpus with Polish, English, mixed prose, code, and tool output assigns prose_lang only to appropriate authored prose",
        "find lang:pl and language facets work across CLI/MCP",
        "reply_language preference can be set, viewed, and included in context compile output with provenance",
        "Generated context/tool_result material does not train or override reply_language automatically",
        "Generated schemas and docs distinguish prose language from programming language"
      ],
      "dependencies": ["polylogue-37t", "polylogue-4smp"],
      "parent": "polylogue-4smp",
      "risks": [
        "Language detectors can misclassify short/code-mixed text",
        "Reply preference can be over-injected into contexts where it is irrelevant",
        "Confusing prose language with content variant language"
      ],
      "confidence": "med",
      "provenance": "Non-English operator slice; confirmed natural-language axis missing from current design"
    },
    {
      "proposed_id": "p4-18",
      "change_kind": "new",
      "target_bead": null,
      "title": "Polish FTS normalization and accessible plain mode",
      "issue_type": "feature",
      "priority": 2,
      "labels": ["area:search", "area:accessibility", "area:cli", "i18n"],
      "description": "Polish recall and accessibility are currently assumed-English. unicode61 does not fold ł/Ł as users expect; Polish inflection/stopwords fragment BM25; Rich/plain output is not a true screen-reader mode.",
      "design": "Add query/index normalization symmetry for Polish diacritics including ł->l, evaluate unicode61 remove_diacritics 2 plus custom fold and trigram fallback, and add regression corpus for Polish queries. Add --plain/--a11y linear text mode with no color-only semantics or box glyphs, and terminal-width tests for long Polish words.",
      "acceptance_criteria": [
        "kanał/kanal and Łódź/Lodz search fixtures match symmetrically at index and query time",
        "Polish inflection recall probe documents BM25, trigram, and semantic fallback behavior before picking default",
        "Screen-reader/plain mode emits labeled line-oriented text with no Rich box drawing",
        "Color is never the sole means of conveying danger/stale/degraded state in CLI/web evidence surfaces",
        "Narrow terminal fixture with Polish long words and wide chars wraps without corrupting columns"
      ],
      "dependencies": ["p4-17", "polylogue-fnm"],
      "parent": "polylogue-fnm",
      "risks": [
        "Custom normalization can diverge between index and query paths",
        "Trigram fallback can increase index size and false positives",
        "A11y mode can become untested alternate renderer"
      ],
      "confidence": "high",
      "provenance": "Non-English/Accessibility slice; externally confirmed sqlite unicode61 ł gap"
    },
    {
      "proposed_id": "p4-19",
      "change_kind": "new",
      "target_bead": null,
      "title": "Evidence-pack analysis run object",
      "issue_type": "epic",
      "priority": 2,
      "labels": ["area:analysis", "area:reports", "area:assertions", "area:demo"],
      "description": "File-4 specs repeatedly need a generic representation of the analysis itself: query runs, evidence packs, annotation imports, cohorts, reports, and ambient/standing results. Without this, demos and agent analyses remain prompt-specific scripts.",
      "design": "Add analysis_run and analysis_step objects that link recipe/prompt refs, query runs, result relations/cohorts, evidence packs, annotation batches, report artifacts, actor/model, status, and degraded state. Recipes may live in git/YAML, but the actual run graph lives in the archive/ops-user object layer. Start with read-only evidence-pack generation and report provenance, not a workflow engine.",
      "acceptance_criteria": [
        "analysis start/step/finish records a DAG linking query_run_refs, cohort_refs, annotation_batch_refs, artifact_refs, and report_refs",
        "Evidence pack export is bounded, cited, machine-readable, and records completeness/degraded flags",
        "Report artifact can cite the exact query/result/cohort/assertion batch that produced each finding",
        "A Fable/delegation demo can run without bespoke storage tables beyond delegation unit and analysis objects",
        "Routine scratch runs remain ops-scoped unless promoted or cited"
      ],
      "dependencies": ["p4-10", "p4-11", "p4-09"],
      "parent": "polylogue-3tl",
      "risks": [
        "Can turn into an overbuilt workflow engine",
        "Object graph may duplicate query/cohort semantics",
        "Too much self-tracking can create noise"
      ],
      "confidence": "med",
      "provenance": "Cross-cutting extension from query objects, external annotations, delegation demo, and self-referential analysis slices"
    },
    {
      "proposed_id": "p4-20",
      "change_kind": "new",
      "target_bead": null,
      "title": "Public proof wrapper for analysis demos",
      "issue_type": "task",
      "priority": 3,
      "labels": ["area:legibility", "area:demo", "positioning", "docs"],
      "description": "The positioning slice is strongest when it stays tied to a live evidence-backed artifact rather than broad product claims. File 4's demo-oriented specs need one reusable wrapper for publishing a reproducible finding without SaaS/product detours.",
      "design": "Create a proof wrapper template for a static public artifact: problem hook, corpus datasheet, method, query/analysis refs, specimen evidence cards, aggregate tables, counterexamples, limitations, reproduce command, and no unmeasured memory-uplift claims. Use it for delegation/Fable only after the generic delegation and analysis objects exist.",
      "acceptance_criteria": [
        "Template can render from an analysis_run/evidence_pack without bespoke Python",
        "Every numeric claim links to query/result/cohort refs and construct-validity notes",
        "Report includes counterexamples and limitations section by default",
        "README/launch copy avoids SaaS or unmeasured uplift claims and points to the reproducible artifact",
        "One dry-run artifact builds on fixture corpus before any public live-corpus publication"
      ],
      "dependencies": ["p4-19"],
      "parent": "polylogue-3tl",
      "risks": [
        "Premature publishing before substrate correctness lands",
        "Meta-positioning work can displace proof work",
        "Demo may overfit to Fable rather than prove generic analysis substrate"
      ],
      "confidence": "med",
      "provenance": "Whole-product/GTM slice; narrowed to public proof artifact gated on analysis substrate"
    }
  ]
}
```
