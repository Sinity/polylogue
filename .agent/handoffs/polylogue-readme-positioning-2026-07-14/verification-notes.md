# Verification notes and execution boundary

This document distinguishes what was inspected, what the lab itself validates, what product behavior was executed, and what remains for a repository-native environment.

## Source identity

The supplied archives were generated on 2026-07-13 at `2026-07-13T14:37:03Z` from clean detached snapshots:

| Project | Full commit | Snapshot README SHA-256 |
|---|---|---|
| Sinex | `1e47b9d11be105be60816fa3d83029bc6ab9e8da` | `8062be0fbb1f342bd861341d68f5b90d792f037f5a21b9e143172c240d84f2db` |
| Polylogue | `59bcbe28e66d7ab7b6485a1ea0c8c4d2b95bdecd` | `ca6bcde2f640e87e4bc266c9bf4bc4c11dc191e9c60c4f2b92e1fbfbf7793e98` |

The lab keeps byte-identical README snapshots and the original snapshot manifests under `snapshots/`; `scripts/validate_lab.py` pins both commit identities and README hashes. See [`sinex.snapshot-manifest.json`](../snapshots/sinex.snapshot-manifest.json) and [`polylogue.snapshot-manifest.json`](../snapshots/polylogue.snapshot-manifest.json).

## Repository evidence inspected

For Sinex, the analysis traced the shipped root recall command, the typed context/timeline envelope, explicit source caveat/gap items, replay semantics, the deterministic system demo/verify owner, and the closed internal multi-source work graph. The internal field packet is personal and absent from the supplied snapshot, so it is treated as dogfood history rather than a public artifact.

For Polylogue, the analysis traced the deterministic receipt fixture, CLI owner, typed compact result, failed-at-claim-time/later-repaired semantics, anti-grep control, public claim registry, visual-tape renderer, visual-evidence documentation, and focused tests.

The grounded file/owner mapping and public-claim boundaries are in `audit/sinex.md`, `audit/polylogue.md`, and `audit/claim-ledger.md`.

## Product execution attempted in this build environment

### Polylogue

Two source-checkout attempts were made from the supplied working tree:

```bash
uv run --offline --frozen polylogue demo receipts --compact
```

This exited `2` before product execution because the supplied snapshot has no `uv.lock`:

```text
error: Unable to find lockfile at `uv.lock`, but `--frozen` was provided.
```

A second attempt removed `--frozen` while retaining offline isolation:

```bash
uv run --offline polylogue demo receipts --compact
```

This exited `1` during dependency resolution because `aiosqlite>=0.19.0` was not available in the local cache. The fixture and focused tests were inspected statically, but the product route did not execute in this environment. No conclusion about the fastest or most reliable public install route is drawn from these failures; the clean-machine experiment remains required.

### Sinex

`nix`, `direnv`, and `sinexctl` were unavailable on `PATH`. Consequently, the repository’s Nix/devshell system proof and live recall command were not executed here. The command surface, typed payload, render path, and test/work owners were inspected from the supplied code and repository evidence.

## What the lab validates

`make check` performs artifact-level validation only. It compiles the Python tooling and checks:

- 30 component contracts and seven composition manifests;
- deterministic regeneration of all seven README candidates;
- catalog, report, and patch freshness;
- application of every README patch to its pinned snapshot;
- 20 rendered prototype/contact-sheet dimensions and SHA-256 hashes;
- byte identity between recommended candidates/assets and review overlays;
- selected internal navigation links;
- pinned source README identities.

The route-manifest dry runs validate command definitions without invoking package managers or products. Their captured expansion is in [`route-manifest-dry-runs.txt`](route-manifest-dry-runs.txt).

The final artifact pass executed `make check` successfully in this build environment and reported: `30 components, 7 compositions, 20 rendered images, 9 report variants, 7 patches, overlays, links, and checksums`.

## Prototype boundary

Every Sinex and Polylogue proof card in this lab is a layout prototype. The watermarks are deliberate. They demonstrate hierarchy, copy density, falsifier placement, and candidate visual grammar; they do not demonstrate that the displayed scenario was produced by the current product.

Promotion requires repository-native generation from a typed owner plus a freshness gate:

- Sinex: a deterministic source-shaped recall fixture or disclosure-cleared scrubbed packet, queried through the existing `sinexctl recall`/`ContextSummaryView` path;
- Polylogue: normalized deterministic receipt output rendered through the existing visual-tape owner, with recipe/output/tape/media drift made terminal.

The implementation packets specify the necessary positive and negative tests. A hand-authored lab PNG must never replace that work.

## Claims intentionally not made

This delivery does not claim that `uvx` is Polylogue’s winning first command, that Polylogue improves downstream agent performance, that unsupported claims are common in real archives, that Sinex’s private field packet is publishable, that the Sinex recall prototype was generated by the current demo, or that either candidate README has won a cold-reader test. Those are explicit experiments or publication gates, not copy decisions.
