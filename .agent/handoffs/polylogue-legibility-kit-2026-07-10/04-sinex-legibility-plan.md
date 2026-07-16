# Sinex external-legibility plan

## Objective

Make Sinex legible as an evidence substrate rather than an infrastructure stack or activity logger. A stranger should understand source material, interpretation, replay, temporal quality, coverage gaps, and authority before encountering NATS or PostgreSQL details.

## Public release cut

### 1. Fix the category and vocabulary

**Beads:** `sinex-o6w`, `sinex-83g`.

- Standardize: “the local evidence substrate for digital life and agent work.”
- Lead with the user question and epistemic model, not the deployed stack.
- Publish a plain-language concept bridge for material, interpretation, three clocks, replay, projection, settlement, reflection, and judgment.
- Replace retired GitHub issue navigation with stable docs and Bead IDs.
- Add a public status block: substantial personal deployment, pre-stable general-user product.

The static patch in this kit implements the documentation portion.

### 2. Separate smoke verification from thesis demos

`sinexctl ops verify --demo` remains an operational smoke gate. It should not be the primary public proof.

Create a `sinexctl demo` or equivalent product surface whose artifacts obey the common manifest contract. Do not seed by directly writing canonical tables if the claim concerns admission, settlement, or product-query behavior. A direct-DB fixture path may remain for narrow reducer tests, but the public demo must cross the boundary it asks users to trust.

### 3. Ship two flagship deterministic stories

**Reconstruct Tuesday, Including the Hole** — `sinex-cem.8`, `sinex-jdp`, related moment/refs work.

**The System Changes Its Mind Honestly** — combine the public narrative of `sinex-cem.3` and `.14` while preserving separate implementation Beads if useful.

These explain broad value and replay semantics better than a service diagram.

### 4. Ship one fault story

**It Diagnosed Its Own Blind Spot** — `sinex-cem.2`, with `sinex-jdp` and source-quality prerequisites.

The outage must remain visible after recovery. The demo should distinguish an unavailable source from a healthy source that observed no events.

### 5. Add small rigorous proof cards

- occurrence idempotence (`sinex-cem.13`, `sinex-908`);
- crash/no-loss around receipt barriers (`sinex-cem.15`, durability prerequisites);
- deterministic model-effect reuse (`sinex-cem.4`, after model-effect substrate);
- retrieval hit@k (`sinex-cem.5`, after embedding worker);
- disclosure control versus true forgetting (split `sinex-cem.1`).

### 6. Publish field evidence honestly

Production restore, 80M-scale event counts, recall latency, and coverage audits should be presented as field packets:

- exact deployment and date;
- query boundary used;
- whether counts are exact or estimated;
- intentional data-loss or purge caveats;
- what the packet establishes;
- what it does not generalize to.

### 7. Make Polylogue the flagship domain consumer

**Beads:** `sinex-4j2`, `sinex-a4w.3.9`, eventual `sinex-cem.7`.

Correct the authority doctrine:

- current low-volume session-indexed events are a notification bridge;
- ultimate Sinex-backed mode stores provider-native materials, normalized transcript segments, durable Polylogue-domain events, assertions, context deliveries, and lifecycle;
- Polylogue remains the domain kernel and product;
- SQLite remains the edge/standalone projection;
- raw text is protected by capability and privacy policy, not excluded from Sinex persistence.

The first decisive proof is a complete Polylogue SQLite rebuild from Sinex-held material and history with explained parity differences.

## Sequence

### Wave A — story and stable docs

README, concepts, agents page, demos page, public claims ledger, and maximal backend ADR. These changes are static and can proceed without a working service stack.

### Wave B — demo substrate contract

Define a common packet, fixture loader, product-boundary runner, and evidence-ref resolver. Make the corpus audit a gate (`sinex-cem.11`) rather than another public narrative.

### Wave C — flagship demos

Build Reconstruct Tuesday and Changes Its Mind in separate worktrees against the shared packet contract. The outage demo can run in parallel if it touches separate source/runtime files.

### Wave D — hard proof and release audit

Run import-twice, crash-no-loss, secret scrub, public artifact scrub, and cold-reader review. Publish field packets only after private-data review.

### Wave E — joint backend proof

Full transcript material ingestion, stable identity bridge, revision settlement, PostgreSQL projections, SQLite rebuild, reverse ambient context, and Agent Work Packet.

## Explicit deferrals

- Closed-loop actuation (`sinex-cem.9`) is not part of the first public wedge. It changes the safety and category discussion before the evidence substrate is understood.
- Full semantic-lane adjudication (`sinex-cem.6`) follows the basic retrieval evaluation.
- General autonomous agent sessions follow the authority and budget model.
- Broad public deployment instructions should not imply general production readiness before `sinex-83g` and secret/corpus decisions close.
