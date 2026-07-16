# Fork 03 — Polylogue flagship “The Receipts” deterministic demo

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-212.2`, the demo-corpus contracts, and related external-legibility Beads.

## Mission

Build a private-data-free flagship demonstration that answers one memorable question:

> The agent said the work succeeded. What actually happened?

The demo must prove structural evidence, not merely search for failure vocabulary.

## Owned scope

Own:

- deterministic scenario/fixture definitions needed for this story;
- demo seed and semantic verification logic;
- the flagship tour command or packet generator;
- focused tests and public-safe generated artifacts.

Do not own landing-page prose, generic semantic-renderer architecture, daemon behavior, or claims-site infrastructure. Coordinate through stable fixture IDs and a small output contract.

## Story

Construct one compact incident with these stages:

1. user asks an agent to make a change and verify it;
2. agent invokes a test command;
3. the structural result fails with a nonzero exit or typed provider error;
4. the assistant either acknowledges it or makes a deliberately bounded claim;
5. a later action or session supplies the repaired verification outcome;
6. a fork or continuation reuses prior context so physical and logical history differ;
7. the final report identifies the evidence chain and safe resume point.

Use provider-shaped fixtures and the normal parser/storage path. Do not insert normalized rows directly merely to make the demo easy.

## Required proof packet

The generated packet must include:

- question and bounded claim;
- exact fixture manifest;
- source/origin inventory;
- stable session/message/block/action refs;
- rendered claim and structural result adjacency;
- a structured action aggregation that grep cannot reproduce semantically;
- lineage view;
- independent fixture oracle;
- negative controls;
- explicit “does not prove” section;
- machine-readable JSON/YAML result;
- human-readable report;
- complete command transcript;
- regeneration command and artifact digests.

## Negative controls

At least:

1. include prose containing “error” with no failed structural result and prove it is not counted;
2. include a failed result whose output avoids the word “error” and prove it is counted;
3. remove the relevant tool-result block and require `not_supported` or a caveat rather than an inferred failure;
4. copy a lineage prefix and prove logical counting does not charge it as unique work.

## Timing and presentation

The first useful receipt should appear quickly. Do not open with a full construct-audit JSON dump. The full audit belongs in the machine-readable packet.

## Validation

Run focused fixture, seed, verify, CLI, snapshot, and public-artifact leak tests. Verify determinism from two fresh output roots. Record all observed timings without converting the fixture result into a real-scale performance claim.

## Deliverables

Produce the patch, generated demo packet, a concise tour transcript suitable for a GIF, test output, and a one-page construct-validity assessment.
