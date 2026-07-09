# Packet Contract Stub

## Claim

The seeded demo corpus (seed 1843) has a fixed, reproducible session count.

## Corpus

`polylogue demo seed` output, seed 1843 — deterministic, no private data.

## Method

`polylogue --format json find` over the full seeded corpus; count result rows.

## Findings

The seeded corpus reproduces the same session count on every regeneration
(this is a fixture proving the packet contract's shape, not a real analytical
finding — see evidence.ndjson for the (stubbed) citation format).

## Specimens

See `evidence.ndjson` for the cited rows.

## Counterexamples

None — this is a trivial reproducibility check by construction.

## Limits

This packet is a fixture for `devtools lab policy demo-packet-registry`, not
a real 212 demo. Do not cite its "claim" externally.

## Reproduce

```bash
polylogue demo seed --seed 1843
polylogue --format json find
```
