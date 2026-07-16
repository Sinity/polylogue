START-HERE-polylogue.md
# Start here: Polylogue first-impression implementation

Use this as the orchestration prompt for a coding agent inside the Polylogue repository.

## Objective

Make the repository’s first 10–60 seconds prove one thing: Polylogue can compare an agent’s claim with structural evidence at the claim boundary, retain a later repair without rewriting history, and reject prose-only false positives.

## Inputs

- hypothesis: `generated/README.polylogue.receipts-first.md`;
- visual layout only: `prototypes/assets/polylogue-receipt-card.png`;
- task contracts `polylogue-01` through `polylogue-03`;
- cold-reader, timing, and claim-integrity experiments;
- structural baselines under `reports/`.

## Execution

1. Run `bd prime`, inspect `AGENTS.md`, and reconcile task packets with current owners and branch state.
2. Freeze the deterministic receipt’s typed output contract and required evidence fields.
3. Close recipe/media/output drift with seeded negative tests.
4. Generate a clean compact visual from the real command; never use the lab PNG as final evidence.
5. Run the clean-machine route benchmark and select the measured winner.
6. Run current-vs-receipts-first-vs-search-first cold-reader tests without identifying the proposed variant.
7. Reconcile winning copy against current docs/status/security surfaces.
8. Run focused tests, `devtools verify --quick`, and the publish boundary required by changed owners.
9. Close the owner with exact commands, timing receipts, reader results, hashes, and artifact paths.

## Stop conditions

- A later success collapses or overwrites the claim-time contradiction.
- Prose containing `error` is counted as structural failure.
- The visual is built from parsed terminal text rather than a typed product contract.
- The chosen no-install command fails the clean supported environment.
- The README implies prevalence, completeness, passive capture, or improved downstream performance not established by the fixture.

Keep the narrower truthful claim when a stop condition fires.

