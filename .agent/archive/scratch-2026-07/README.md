# Archived scratch directories (retired 2026-07-16)

Three `.agent/scratch/` working directories moved here after confirming zero
live-bead dependency (no open bead's notes/design/acceptance fields reference
any path inside them — checked via `grep` over `.beads/issues.jsonl`).
`scratch/` is gitignored; these are promoted to tracked evidence because
their source work is finished, not superseded-but-still-referenced.

- `2026-07-04-beads-swarm/` — no bead references it at all (open or closed).
- `gpt-fork-deliveries-2026-07-10/` — no bead references it at all.
- `swarm2/` — its only reference is the closed bead `polylogue-ejm3`
  ("Tech-tree integration: digest 2026-07-05 R&D corpus into vision-tiered
  bead graph"), i.e. this directory's content was already fully digested into
  the bead graph before closure.

Not moved (deliberately left in `.agent/scratch/`): every other dated
directory there — `corpus-gpt-pro-2026-07-06`, `corpus-gpt-pro-2026-07-07`,
`new-gpt-pro`, `new`, `research`, `legibility-kit-2026-07-10`,
`legibility-kit-v2-2026-07-10`, `fanout-prompts`,
`readme-positioning-2026-07-14` — despite being similarly old, each has
active open-bead references (`corpus-gpt-pro-2026-07-07` alone backs the
"[Prework packet]" notes on 122 open beads; `new` backs 181). Moving those
would silently break in-flight work. See `polylogue-60v8` for the full
per-directory open/closed reference counts as of this audit, and for
follow-up: re-check periodically and move each directory here once its
open-bead references clear.
