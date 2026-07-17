Title: "Seed annotation ontologies and governed per-archive ontology bootstrap (activity, goals, outcomes, knowledge, reusability)"

Result ZIP: `ann-01-ontology-seed-r01.zip`

## Mission

Implement bead `polylogue-dve1` (read its FULL record in the snapshot's
`.beads/issues.jsonl`): versioned seed annotation schemas over the existing
annotation/batch/judgment substrate, plus the governed bootstrap path by
which archive-specific ontology candidates emerge without polluting the
operator's durable vocabulary.

The substrate you extend (do NOT build a parallel one):
`user.db` unified `assertions` table (kind is plain TEXT; `AssertionKind`
is the typed boundary), immutable `annotation_schemas` (versioned construct
definitions), `annotation_batches` (label-run provenance), context policy
via `context_policy_json` (candidates stay inject:false until judged).
Read `polylogue/storage/sqlite/archive_tiers/user.py` DDL,
`user_write.py`, and `docs/data-model.md`.

The five seed families (bead design is authority; summary):

1. **activity** at declared session/segment grain: debugging, design,
   implementation, research, writing, ideation, ops, procurement;
2. **goal events**: opened, blocked, resumed, declared-resolved, superseded,
   explicitly-abandoned ONLY when an actor actually declares abandonment —
   `unresolved_inactive(H)` is DERIVED by the goal graph with a named
   horizon/frame/evaluation receipt and right-censoring (bead `7yk5` owns
   that derivation; never emit abandonment as a timeless annotation);
3. **outcome evidence**: test passed, commit observed, deployment observed,
   user accepted, answer declared, unknown — with structural/rule/judged
   authority preserved as distinct provenance;
4. **knowledge artifacts**: decision, lesson, preference, fact
   candidate/established under named authority, commitment;
5. **reusability** classification.

Deliver:

1. The versioned seed schemas as annotation_schemas rows (registration
   code + migration-free vocabulary growth — confirm no user-tier schema
   bump is needed; if one is, STOP and report, durable tiers are
   additive-migration-only behind backup manifests).
2. The bootstrap path: informal tags/affinity may NOMINATE candidates; a
   governed promotion flow (candidate schema → operator judgment → active
   schema version) with receipts; a high-affinity tag produces at most a
   candidate, never an active construct (bead AC).
3. Label-writing API surface for batch annotators (used by the calibration
   job ann-02 and runbook ann-03 — state the interface they consume) that
   respects assertion authority: agent labels are candidates/scoped rows,
   operator judgments are never silently overwritten (note the `41ow`
   TOCTOU bead: write via the BEGIN IMMEDIATE preserve/write shape).
4. Tests: schema registration/versioning, the distinctness invariants from
   the bead's AC (prospective events vs structural outcomes; no inferred
   abandonment), promotion-flow authority, and render-surface updates
   (new AssertionKind values propagate into `render openapi` +
   `cli-output-schemas` — regenerate; this is a known 4-trap registration).

## Deliverable emphasis

HANDOFF.md: schema catalog (every construct, grain, fields, authority
model), promotion-flow state machine, the annotator-facing API, exact
generated-surface regenerations performed, and what ann-02/ann-03 can
assume as landed interface.
