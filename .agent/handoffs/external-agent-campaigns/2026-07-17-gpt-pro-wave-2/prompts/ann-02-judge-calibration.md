Title: "Judge calibration infrastructure: blinded judgment, gold questions, and agreement/calibration measurement for archive annotation"

Result ZIP: `ann-02-judge-calibration-r01.zip`

## Mission

Polylogue's annotation substrate exists (unified `assertions` table in
user.db, immutable `annotation_schemas`, `annotation_batches` provenance —
read `polylogue/storage/sqlite/archive_tiers/user.py` DDL and
`docs/data-model.md`). The plan is to mass-annotate the operator's 13k-session
archive with agent judges (terminal-state labels, claim-verification labels,
session-quality constructs). What does NOT exist is the calibration layer that
makes such labels trustworthy. Beads `polylogue-rxdo.9`, `rxdo.9.6`,
`rxdo.9.12`, and `h10` carry the design intent — read them in the snapshot's
`.beads/issues.jsonl` (search by id).

Build an implementation draft:

1. **Gold-question protocol**: how gold items are authored, stored (assertion
   kind + schema version), stratified, and injected blind into annotation
   batches; per-judge accuracy against gold computed as a queryable product.
2. **Blinded judgment**: mechanism ensuring a judging agent cannot see prior
   labels/judgments for the item (what exact query surface the judge gets);
   enforcement at the batch-composition layer, not judge goodwill.
3. **Agreement + calibration measures**: inter-judge agreement (Krippendorff
   alpha or justified alternative) per construct; calibration curves when
   judges emit confidence; drift detection across batches; all computed into
   typed, queryable rows (annotation_batches-scoped), not ad-hoc reports.
4. **Judge identity**: judges as actors (human or agent+model+version) with
   stable identity refs so calibration attaches to the right judge (rxdo.9.12).
5. Tests with synthetic batches proving: gold-blindness, agreement math
   against hand-computed values, and that a judge's labels cannot silently
   overwrite operator judgments (respect the assertion-authority model; note
   the known TOCTOU bug `polylogue-41ow` — design your write path to the
   BEGIN IMMEDIATE preserve/write shape its bead prescribes).

## Constraints

- Extend the existing assertions/annotation substrate; do NOT invent a
  parallel labels store or a new schema regime (user.db is durable-tier:
  additive numbered migrations only, and prefer no schema change at all —
  the TEXT kind column + schemas registry is designed for vocabulary growth).
- Analysis products must be computable offline from user.db + index.db reads.

## Deliverable emphasis

HANDOFF.md: the exact assertion kinds/schemas introduced, batch-composition
API, the calibration query surface (how an operator asks "which judge is
trustworthy for construct X"), what the first real calibration batch over the
archive should look like (size, strata, gold ratio), and open decisions.


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
