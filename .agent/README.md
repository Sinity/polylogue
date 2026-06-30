# Polylogue Agent Workspace

This directory stores Polylogue agent scaffold plus local devloop state.
Reusable scaffold files are tracked; live state and generated evidence are
ignored.

If a fresh agent is told only "continue the devloop setup in `.agent`", it
should follow the bootstrap below without needing chat context.

## Bootstrap

1. Run `.agent/scripts/devloop-status`.
2. Run `.agent/scripts/devloop-review`.
3. Read `.agent/conductor-devloop/README.md`.
4. Continue through `.agent/conductor-devloop/RUNBOOK.md`.
5. If local state files exist, read `.agent/conductor-devloop/ACTIVE-LOOP.md`.
6. Start or resume the next slice with `.agent/scripts/devloop-start` or
   `.agent/scripts/devloop-focus`.

## Read First

- `conductor-devloop/README.md` — resume instructions for a contextless agent.
- `conductor-devloop/INDEX.md` — startup order, current goal, and archive
  baseline warning.
- `conductor-devloop/ACTIVE-LOOP.md` — ignored local state, when present:
  current slice, focus, warnings, and next action.
- `conductor-devloop/RUNBOOK.md` — exact loop protocol, focus modes, proof
  ladder, heavy-work rules, git protocol, and inbox routing.
- `conductor-devloop/OPERATING-LOG.md` — timestamped loop entries.
- `conductor-devloop/VELOCITY.md` — cadence and acceleration rules.
- `conductor-devloop/SELF-PROMPTS.md` — long-form self-prompts and process goal.
- `scratch/README.md` — routing index for research, artifacts, and archive
  material behind the conductor packet.

## Stable Areas

- `conductor-devloop/` — tracked process docs plus ignored local resume state
  for the active Polylogue dogfood/demo loop.
- `demos/` — current curated demo shelf. It is the best current set, not an
  append-only archive.
- `scripts/devloop-log` — append timestamped entries with elapsed-time notes.
- `scripts/devloop-sync` — refresh the repo-local `.agent/conductor-devloop`
  packet manifest, event sidecar, and helper-script snapshots.
- `scripts/devloop-review` — adversarial preflight for scaffold, packet drift,
  daemon/root drift, duplicate archives, and active heavy processes.
- `scripts/devloop-status`, `devloop-start`, `devloop-checkpoint`,
  `devloop-handoff`, `devloop-ahead`, and `devloop-velocity` — executable loop
  gates, speed diagnostics, and foreground work prompts.
- `scratch/README.md` — tracked routing rules for ignored scratch material.
- `tools/` — small helper scripts.

## Rule

Do not put new loose files at `.agent/` or `.agent/scratch/` root. Put active
loop state in ignored files under `conductor-devloop/` via the devloop scripts,
durable research in ignored `scratch/research/`, generated proof under the
relevant ignored demo/artifact directory, and completed/stale notes under an
ignored archive directory.
