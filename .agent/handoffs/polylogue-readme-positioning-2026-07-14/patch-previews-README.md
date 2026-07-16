# README patch previews

Each patch changes only `README.md` and is generated against the supplied snapshot. The images referenced by the recommended candidates are included separately in `overlays/`; copy the appropriate prototype asset only for review, never as final product evidence.

| Patch | Snapshot | Candidate | Intent |
|---|---|---|---|
| [`sinex-recall-first.patch`](sinex-recall-first.patch) | `snapshots/sinex.README.current.md` | `generated/README.sinex.recall-first.md` | Recommended general front door: practical bounded recall. |
| [`sinex-replay-first.patch`](sinex-replay-first.patch) | `snapshots/sinex.README.current.md` | `generated/README.sinex.replay-first.md` | Technical differentiation: reinterpretation without erasure. |
| [`sinex-rebuildable-first.patch`](sinex-rebuildable-first.patch) | `snapshots/sinex.README.current.md` | `generated/README.sinex.rebuildable-first.md` | Concept/launch framing: rebuildable understanding. |
| [`sinex-event-spine.patch`](sinex-event-spine.patch) | `snapshots/sinex.README.current.md` | `generated/README.sinex.event-spine.md` | Conservative contributor/infrastructure framing. |
| [`polylogue-receipts-first.patch`](polylogue-receipts-first.patch) | `snapshots/polylogue.README.current.md` | `generated/README.polylogue.receipts-first.md` | Recommended general front door: claim-boundary receipts. |
| [`polylogue-search-first.patch`](polylogue-search-first.patch) | `snapshots/polylogue.README.current.md` | `generated/README.polylogue.search-first.md` | Fast-action discovery framing. |
| [`polylogue-continuity-first.patch`](polylogue-continuity-first.patch) | `snapshots/polylogue.README.current.md` | `generated/README.polylogue.continuity-first.md` | Agent-handoff capability framing without uplift claims. |

## Apply for local review

From a clean checkout whose `README.md` matches the corresponding snapshot:

```bash
git apply --check /path/to/first-impressions-lab/patches/sinex-recall-first.patch
git apply /path/to/first-impressions-lab/patches/sinex-recall-first.patch
```

The patches intentionally exclude generated proof media. Follow the relevant agent packet and regenerate media through the product-owned route before merging.
