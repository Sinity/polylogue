# Adversarial Review

## Known Failure Modes

- Quoting stale 16K-session counts from pre-dedup backups as current truth.
- Running a daemon against XDG while reasoning about `/realm/tmp/polylogue-dev`.
- Keeping prod and dev archive roots alive at once, then comparing mixed counts.
- Letting obsolete DB directories remain discoverable as plausible current roots.
- Treating recovery/export/read flags as durable product surfaces after the DSL
  and projection substrate should own the capability.
- Preserving compatibility endpoints or DTOs after deciding they are trash.
- Materializing insight read models that are cheap to compute or based on weak
  heuristics, then letting their table names make them sound authoritative.
- Building demos as bespoke reports instead of pressure-testing the shared query,
  projection, and rendering algebra.
- Waiting on long imports/tests without producing scaffold, demo, or source-review
  progress.
- Probing devloop helper scripts with placeholder arguments and accidentally
  mutating `ACTIVE-LOOP.md`, `OPERATING-LOG.md`, `EVENTS.jsonl`, or
  `DEMO-RADAR.md`.

## Mitigations

- Run `.agent/scripts/devloop-review` at startup and before broad status claims.
- Run `.agent/scripts/devloop-sync` after changing current notes.
- Discover helper interfaces with `.agent/scripts/devloop-* --help`; review
  verifies that every help path is present and side-effect-free.
- Include root + schema + sessions + messages in every archive status line.
- Quarantine obsolete databases under `/realm/inbox/polylogue-obsolete-db-quarantine`
  with a manifest instead of leaving them near active paths.
- Before keeping any old public command, endpoint, DTO, or flag, state why it is
  not just a compatibility silo. If there is no reason, remove it.
- Before calling an insight useful, show a real report or demo where the output
  changes a decision. If not, demote/delete/compute-on-read.
