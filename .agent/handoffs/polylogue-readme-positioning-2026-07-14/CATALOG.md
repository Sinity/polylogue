CATALOG.md
# Component catalog

30 reusable components. Status is part of the component contract; a target/prototype must not be silently promoted by composition.

| Path | Component | Status | Purpose | Used by |
|---|---|---|---|---|
| [`library/polylogue/capabilities.md`](../library/polylogue/capabilities.md) | `polylogue.capabilities` | current | What you can do | `polylogue-receipts-first`, `polylogue-search-first` |
| [`library/polylogue/continuity.md`](../library/polylogue/continuity.md) | `polylogue.continuity` | current capability; no uplift claim | Bounded context, not memory magic | `polylogue-continuity-first` |
| [`library/polylogue/hero-continuity.md`](../library/polylogue/hero-continuity.md) | `polylogue.hero-continuity` | capability-safe; uplift claims forbidden | Give the next agent evidence, not a summary. | `polylogue-continuity-first` |
| [`library/polylogue/hero-receipts.md`](../library/polylogue/hero-receipts.md) | `polylogue.hero-receipts` | current-safe | Know what the agents actually did. | `polylogue-receipts-first` |
| [`library/polylogue/hero-search.md`](../library/polylogue/hero-search.md) | `polylogue.hero-search` | current-safe | Search every AI-work history as one archive. | `polylogue-search-first` |
| [`library/polylogue/install.md`](../library/polylogue/install.md) | `polylogue.install` | current | Install | `polylogue-continuity-first`, `polylogue-receipts-first`, `polylogue-search-first` |
| [`library/polylogue/local-boundary.md`](../library/polylogue/local-boundary.md) | `polylogue.local-boundary` | current | local-boundary | `polylogue-continuity-first`, `polylogue-receipts-first`, `polylogue-search-first` |
| [`library/polylogue/proof-receipts.md`](../library/polylogue/proof-receipts.md) | `polylogue.proof-receipts` | current deterministic proof | A claim is not a receipt | `polylogue-continuity-first`, `polylogue-receipts-first`, `polylogue-search-first` |
| [`library/polylogue/run-proof.md`](../library/polylogue/run-proof.md) | `polylogue.run-proof` | command winner must be measured | Run the proof | `polylogue-receipts-first` |
| [`library/polylogue/search-first.md`](../library/polylogue/search-first.md) | `polylogue.search-first` | current | Search first | `polylogue-continuity-first`, `polylogue-search-first` |
| [`library/polylogue/status.md`](../library/polylogue/status.md) | `polylogue.status` | current | status | `polylogue-continuity-first`, `polylogue-receipts-first`, `polylogue-search-first` |
| [`library/shared/first-action.md`](../library/shared/first-action.md) | `shared.first-action` | guidance | First action | — |
| [`library/shared/identity-contract.md`](../library/shared/identity-contract.md) | `shared.identity-contract` | guidance | Identity contract | — |
| [`library/shared/navigation-strip.md`](../library/shared/navigation-strip.md) | `shared.navigation-strip` | guidance | navigation-strip | — |
| [`library/shared/non-claim.md`](../library/shared/non-claim.md) | `shared.non-claim` | guidance | non-claim | — |
| [`library/shared/proof-contract.md`](../library/shared/proof-contract.md) | `shared.proof-contract` | guidance | Proof contract | — |
| [`library/shared/proof-shelf.md`](../library/shared/proof-shelf.md) | `shared.proof-shelf` | guidance | Proof shelf | — |
| [`library/shared/status-strip.md`](../library/shared/status-strip.md) | `shared.status-strip` | guidance | status-strip | — |
| [`library/shared/trust-strip.md`](../library/shared/trust-strip.md) | `shared.trust-strip` | guidance | Trust strip | — |
| [`library/sinex/capabilities.md`](../library/sinex/capabilities.md) | `sinex.capabilities` | current concepts; cross-source wording coverage-bounded | What Sinex is for | `sinex-rebuildable-first`, `sinex-recall-first`, `sinex-replay-first` |
| [`library/sinex/hero-event-spine.md`](../library/sinex/hero-event-spine.md) | `sinex.hero-event-spine` | current-safe | The event spine for a local machine. | `sinex-event-spine` |
| [`library/sinex/hero-rebuildable.md`](../library/sinex/hero-rebuildable.md) | `sinex.hero-rebuildable` | current-safe conceptual framing | Your data should not be interpreted only once. | `sinex-rebuildable-first` |
| [`library/sinex/hero-recall.md`](../library/sinex/hero-recall.md) | `sinex.hero-recall` | current-safe | Ask your machine what you were doing. | `sinex-recall-first` |
| [`library/sinex/hero-replay.md`](../library/sinex/hero-replay.md) | `sinex.hero-replay` | current-safe | Capture once. Rebuild understanding later. | `sinex-replay-first` |
| [`library/sinex/nix-map.md`](../library/sinex/nix-map.md) | `sinex.nix-map` | current-safe analogy; recorded-effect row explicitly dormant | Nix for personal data | `sinex-rebuildable-first` |
| [`library/sinex/proof-recall-current.md`](../library/sinex/proof-recall-current.md) | `sinex.proof-recall-current` | current product surface; public visual prototype | A work window, not another history file | `sinex-rebuildable-first`, `sinex-recall-first`, `sinex-replay-first` |
| [`library/sinex/proof-replay.md`](../library/sinex/proof-replay.md) | `sinex.proof-replay` | current invariant; visual proof still to generate | The history stays. The interpretation can change. | `sinex-event-spine`, `sinex-rebuildable-first`, `sinex-replay-first` |
| [`library/sinex/run-proof.md`](../library/sinex/run-proof.md) | `sinex.run-proof` | current deterministic system proof | Run the proof | `sinex-event-spine`, `sinex-rebuildable-first`, `sinex-recall-first`, `sinex-replay-first` |
| [`library/sinex/status.md`](../library/sinex/status.md) | `sinex.status` | current | status | `sinex-event-spine`, `sinex-rebuildable-first`, `sinex-recall-first`, `sinex-replay-first` |
| [`library/sinex/target-publication.md`](../library/sinex/target-publication.md) | `sinex.target-publication` | TARGET; never select silently | Target public receipt | — |

## Composition rule

Manifests select small components in order and retain the technical tail of the current README. Generated candidates are reproducible functions of snapshot + manifest + component files.

