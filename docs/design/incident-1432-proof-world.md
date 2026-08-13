# Incident 14:32 — the shared deterministic proof world

Owning beads: `polylogue-212.11` (corpus), `polylogue-212.12` (Demo Packet v2
contract). Epic: `polylogue-212` (demo portfolio). This document is the
standing model for the corpus; the beads carry execution state and win on
conflict.

## Why one world

Every public demo today invents its own synthetic fixture, so each demo pays
its own construct-validity bill and no two demos corroborate each other. The
alternative is one public-safe incident world that every flagship demo replays
from a different angle: the same memorable event grounds The Receipts, Count
It Once, compaction honesty, context autopsy, honest refusal, and the joint
world-around-the-claim story — without contaminating any single demo's
primary construct.

## The narrative

A developer asks an agent to repair a flaky clock-sensitive test in a small
repository. At 14:32 the first agent edits the wrong fixture, receives a
nonzero structural test result, and nevertheless reports the issue resolved.
A fork copies the prior transcript. Context compaction omits the failed
experiment. A second agent receives a bounded brief, observes repository
evidence, repairs the correct fixture, and verifies the result structurally.
The world also contains one deliberate capture outage and one parser-semantics
revision of the same material.

## Required constructs (Polylogue side)

The current deterministic demo corpus (`polylogue/scenarios/corpus.py`,
seed 1843) already provides: multiple origins, tool use/result pairs with a
structural failure, a copied-prefix fork, a fresh subagent, a compaction
boundary, attachments with retained bytes, usage lanes, context snapshots,
and user overlays. Incident 14:32 adds what is missing:

1. an assistant **success claim contradicted by a structural result** in the
   same session, followed by a **later verified repair** (second verifier run,
   exit 0) — the Receipts anchor;
2. a **compaction summary that omits the failed attempt** — the compaction
   honesty anchor;
3. a **deliberate source-outage interval** — coverage honesty; pairs with the
   Sinex "Missing Source" demo;
4. **one record parsed under semantics v1 and v2**, both interpretations
   preserved, one promoted — the changes-mind-honestly anchor;
5. an **ambiguous cross-material duplicate** — occurrence identity /
   import-twice anchor;
6. **terminal, Git, and Beads-shaped observed events around 14:32** — hooks
   for joint cross-source demos (bead with acceptance criteria, a premature
   close attempt, a final evidence-backed completion).

## The anti-circularity rule

Fixtures are generated from a declarative scenario file, and a separate,
independently authored **oracle file** declares the expected physical
sessions and ordinals, logical composition, structural outcomes, usage-lane
totals, coverage intervals, parser-semantics diff, content hashes, assertion
states, and context-delivery manifest. The product reads the fixtures; the
verifier reads the oracle. Generating both from one reducer would prove only
that the reducer agrees with itself.

The corollary is the anti-vacuity witness: for every declared construct there
is a test that withholds or deletes the evidence and asserts the dependent
demo or claim goes red / `not_supported`. A verification structure that stays
green when its proof is removed validates shape, not reality.

## Public-safety constraints

Invented code, domains, paths, people, and model names throughout; realistic
provider structure and failure modes preserved. No copied private
conversation text, no real secret, no personal hostname, no absolute user
path. Secret-handling demos use unmistakably synthetic tokens (e.g.
`POLYLOGUE_DEMO_SECRET_DO_NOT_USE_7YQ9`) and assert both suppression in the
declared public view and preservation in the restricted evidence lane when
that is the claimed policy.

## Relationship to Demo Packet v2

Every demo built on this world ships a machine-readable packet
(`polylogue-212.12`): one primary construct, the claim stated before
execution, a declared independent oracle, negative and missing-evidence
controls, a baseline arm, an explicit falsifier, resolvable receipts, and a
non-claims section. The world exists so those packets can share receipts; the
packets exist so the world cannot be mistaken for a feature montage.

## Provenance

Distilled 2026-07-10 from the GPT-5.6 Pro external-legibility kit
(`02b-demo-portfolio-expanded.md`, session escrow under
`.agent/scratch/legibility-kit-2026-07-10/`), adjudicated against the live
corpus and verifier. The kit is inspiration, not authority; this document and
the owning beads are the in-repo authority.
