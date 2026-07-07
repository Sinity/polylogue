# Handoff Pack Demo

Current, curated handoff packets for the two large devloop sessions used by the
uplift-campaign bead.

This shelf is not append-only. Regenerate or replace it when the active archive
or read-view contract changes.

## Sessions

- `current/polylogue-devloop-019f12b5-fc19-7110-b069-4f49a78da82d/`
- `current/sinex-devloop-019f12b5-1a85-7b42-858e-44eccf8469dc/`

Each folder contains:

- `handoff/temporal.json` — bounded temporal evidence window.
- `handoff/chronicle.json` — bounded authored-dialogue edge chronicle.
- `handoff/spec.json` — compact selection/projection/render contract.
- `read-package-summary.json` — timings and file sizes for the run.

`current/manifest.json` records the archive root, schema version, archive
counts, and per-session timings.

## Current Caveat

The CLI `devtools workspace read-package` route timed out under live archive I/O
pressure before this packet was generated. The packet was then generated in one
Python process through the same temporal and chronicle read-view builders. The
product fix made exact-id temporal reads avoid generic query enumeration and use
lightweight session-scoped message/action occurrence reads.
