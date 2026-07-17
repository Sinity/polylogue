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
