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


---

## Context and authority

You are a long-running ChatGPT Pro engineering worker. A recent Polylogue
project-state archive will be attached. Retrieve and inspect it broadly; do not
assume attachment bytes consume your active prompt context. The attached
snapshot is the code authority. This prompt defines your mission. Repository
instructions and complete relevant Beads records define constraints and intent;
later Beads notes may supersede older descriptions. Current source wins when a
stale plan names paths or APIs that no longer exist.

Start by reporting the snapshot commit/branch/dirty-patch identity you found and
the source, tests, Beads, and history you inspected. Follow dependencies beyond
the obvious files when they affect the production route. Do not invent an API,
test helper, product contract, or parallel framework to make the task easy.

## Working contract

- Produce the largest internally coherent implementation draft that fits the
  mission. Prefer one real end-to-end behavior over disconnected scaffolding.
- Preserve Polylogue's substrate-first architecture and existing typed
  interfaces. Small production seams are allowed only when real production
  behavior needs observation or control.
- Write concrete production changes and real-route tests. A test must name the
  production dependency it exercises and the representative implementation
  mutation/removal that should make it fail.
- Do not delete existing tests or helpers. Identify proposed dominated
  deletions separately for independent local certification.
- Use your container and run meaningful self-contained checks when possible.
  Never claim access to the operator's live daemon, browser, archive, secrets,
  NixOS deployment, or current worktree. Mark those checks `unverified`.
- If the full scope is unsafe, complete the strongest coherent subset and make
  the remaining decisions and exact continuation steps explicit. Do not return
  placeholders, ellipses, pseudocode presented as code, or a generic plan in
  place of implementation.

## Deliverable

Create the exact `Result ZIP` named near the top of this prompt under
`/mnt/data/`. Do not include the supplied repository/project-state archive or
other copied inputs in the result. The finished ZIP must be attached to the
conversation through a working, user-clickable download link. Work left only
in an internal shell directory, temporary notebook, scattered sandbox files,
or prose is not delivered.

The ZIP must contain:

- `HANDOFF.md`: mission, snapshot identity, inspected evidence, mechanism,
  decisions, changed files, acceptance matrix, apply order, risks, and exact
  verification performed/remaining;
- `PATCH.diff`: one apply-ready unified diff against the named snapshot;
- `TESTS.md`: test design, production dependencies, anti-vacuity mutation,
  commands, and honest execution results;
- `EVIDENCE.md`: relevant source/Bead/history findings and any contradictions;
- `FILES/`: complete replacements only where they materially disambiguate the
  patch; omit it when unnecessary.

Before answering, reopen the ZIP, list and validate its members, compute its
SHA-256 and byte size, and confirm that `PATCH.diff` has no placeholders or
copied source snapshot. Your final chat response must begin with a substantive
operator-readable report of what you did and why. It must also state important
limitations, missing or unverified work, and how much additional value another
iteration could plausibly add—distinguishing a small repair from a substantial
second pass. Then report verification and risks and give a prominent working
link to the exact `/mnt/data/` ZIP. A bare download receipt is not acceptable.

## Continuation protocol

Do not perform a separate adversarial review unless the user explicitly asks
for one. If the user asks to **iterate** or **continue**, preserve valid prior
work, perform the highest-value remaining implementation/research pass, and
publish a new cohesive package revision with the same complete structure—not a
loose supplemental patch. Explain exactly what changed, what improved, what
still remains, and whether another iteration is likely to pay off.

If the user explicitly asks for an **adversarial review**, attack your prior
result against the original mission and current attached authority: search for
unsupported claims, invented or stale APIs, missing call sites, composition
failures, unsafe assumptions, vacuous tests, patch/apply defects, incomplete
acceptance criteria, and evidence that would falsify the design. Preserve work
that survives. Then repair every legitimate finding you can, regenerate the
entire cohesive package as the next revision, and report findings, repairs,
remaining disputes, and the value of another adversarial/implementation pass.
