Title: "Exact FTS identity ledger: rowid↔block_id↔source_hash↔recipe binding, drift magnitude gauges, and trigger-coherence property tests (1xc.12)"

Result ZIP: `misc-03-fts-identity-r01.zip`

## Mission

Implement bead `polylogue-1xc.12` (P1 — read its full record). The FTS5
index (`messages_fts`, contentless, synchronized by insert/delete/update
triggers over `blocks.search_text`) has a keystone identity assumption:
`messages_fts.rowid == blocks.rowid == docsize.id`. SQLite ROWID REUSE
breaks the safety of count-based checks: after delete+insert, a ghost FTS
row can bind to a DIFFERENT block while counts still agree. Search results
then cite the wrong content — a silent correctness hole in the archive's
most-used read path. Current freshness reporting is also too boolean:
operators need drift MAGNITUDE.

Build (bead design is authority):

1. **`messages_fts_identity` ledger** in the index tier: keyed by rowid
   with UNIQUE block_id, plus source_hash and recipe_id; maintained by the
   SAME trigger events as the contentless FTS table; rebuilt atomically by
   the repair/rebuild path. Consume the storage-neutral DerivationKey
   VALUE SHAPE from `polylogue-wmsc` (a parallel job drafts it — mirror
   its subject/source/recipe/output-contract field semantics, keep an
   FTS-owned ledger and lifecycle; state the interface assumption if the
   wmsc result is absent from your inputs).
2. **Exact freshness state**: O(1)-readable state rows covering missing,
   excess, identity mismatch, source mismatch, recipe mismatch, last check
   time, repair generation (bead AC #1-2). Exact checks JOIN on rowid AND
   confirm block_id — never count parity alone.
3. **Drift gauges**: magnitude metrics from the freshness state exposed
   through the existing status/observability path (find the ops/OTLP
   metric emission seam — `daemon/` observability wiring; Prometheus-style
   naming per the bead).
4. **Metamorphic trigger-coherence property tests**: Hypothesis state
   machine over arbitrary block mutations (insert/update/delete/full
   session replace/rowid-reuse-inducing delete+insert sequences) asserting
   FTS ∥ docsize ∥ identity-ledger coherence after every step — the
   rowid-reuse ghost case must be a named regression that FAILS on
   current master's count-only logic (prove the vulnerability first,
   fixture it, then show the ledger catches it).
5. **Schema window discipline**: this is an index-tier addition — edit
   canonical DDL + bump the index schema version (rebuild route, NO
   migration helper; `devtools lab policy schema-versioning` enforces
   this). The bead says implement "in the next batched index-schema
   window": check the snapshot for other pending index-tier bumps and note
   batching opportunities for the integrator rather than assuming you own
   the version bump timing.

## Constraints

- Triggers are hot-path: measure and state the write-amplification cost of
  ledger maintenance (per-block insert/delete adds one ledger row op) and
  keep it O(1) per block event.
- The FTS repair path (`fts` convergence stage + rebuild) must rebuild the
  ledger in the same transaction boundary as the FTS rebuild — find the
  current repair implementation (convergence stages + storage FTS module)
  and integrate, don't parallel-build.
- Do not change the tokenizer or FTS schema shape (`unicode61`, no porter
  — pinned decision).

## Deliverable emphasis

HANDOFF.md: DDL + version-bump diff, trigger design with cost analysis,
the freshness-state row schema, gauge names/semantics, the rowid-reuse
vulnerability demonstration (this is the story: show the ghost, then the
catch), property-test design, and batching notes for the index-window
integrator.


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
