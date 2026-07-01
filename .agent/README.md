# Polylogue Agent Workspace

This directory stores Polylogue agent scaffold plus local devloop state.
Reusable scaffold files are tracked; live state and generated evidence are
ignored.

If a fresh agent is told only "continue the devloop setup in `.agent`", it
should follow the bootstrap below without needing chat context.

## Bootstrap

1. Read `.agent/DEVLOOP.md`.
2. Run `.agent/scripts/devloop-status`.
3. Run `.agent/scripts/devloop-review`.
4. Read `.agent/conductor-devloop/README.md`.
5. Continue through `.agent/conductor-devloop/RUNBOOK.md`.
6. If local state files exist, read `.agent/conductor-devloop/ACTIVE-LOOP.md`.
7. Start or resume the next slice with `.agent/scripts/devloop-start` or
   `.agent/scripts/devloop-focus`.

## Read First

- `DEVLOOP.md` — first-stop guide for a fresh agent.
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
- `includes/` — tracked durable conventions and compact project memory.
- `scratch/README.md` — routing index for research, artifacts, and archive
  material behind the conductor packet.

## Stable Areas

- `conductor-devloop/` — tracked process docs plus ignored local resume state
  for the active Polylogue dogfood/demo loop.
- `demos/` — current curated demo shelf. It is the best current set, not an
  append-only archive.
- `scripts/devloop-status` — fast live status: git, active loop, archive,
  daemon, packet size, and pressure signals.
- `scripts/devloop-review` — adversarial preflight for scaffold, packet drift,
  daemon/root drift, duplicate archives, and active heavy processes.
- `scripts/devloop-start` — start one named slice and write active/log state.
- `scripts/devloop-checkpoint` — record reassessment without changing slice
  identity.
- `scripts/devloop-log` — append timestamped entries with elapsed-time notes.
- `scripts/devloop-focus` — record focus transitions with trigger and decision.
- `scripts/devloop-demo` — update demo radar and, when applicable, the log.
- `scripts/devloop-baseline` — capture lightweight runtime/resource/git
  baseline evidence.
- `scripts/devloop-wait` — record wait state plus useful foreground work while
  a long command runs.
- `scripts/devloop-ahead` — print useful foreground work prompts.
- `scripts/devloop-meta` — record process failure/self-improvement audits.
- `scripts/devloop-handoff` — write the latest handoff note inside the
  conductor packet.
- `scripts/devloop-sync` — refresh generated local state such as event, demo,
  script-hash, and packet manifests.
- `scripts/devloop-velocity` — summarize cadence, proof gaps, friction, and
  slow commands.
- `scripts/devloop-refresh-demos` and `scripts/devloop-refresh-events` —
  regenerate ignored demo/event indexes from current local artifacts.
- `scripts/lib-devloop` — sourced helper functions shared by the executable
  primitives; it is tracked scaffold, not a user-facing command.
- `scratch/README.md` — tracked routing rules for ignored scratch material.
- `archive/` — ignored archaeology only. It is not a startup surface and should
  not receive current state, generated proofs, or demos.
- `task-history/` — ignored devtools execution ledger and runtime baselines;
  useful for velocity analysis, not a startup surface.
- `tools/` — small helper scripts.

## Rule

Do not put new loose files at `.agent/` or `.agent/scratch/` root. Put active
loop state in ignored files under `conductor-devloop/` via the devloop scripts,
durable research in ignored `scratch/research/`, generated proof under the
relevant ignored demo/artifact directory, and completed/stale notes under an
ignored archive directory.

Ignored support shelves must stay small and non-authoritative. `.agent/archive/`
may hold compact archaeology such as old operating-log windows; it must not grow
into a second conductor packet. `.agent/task-history/` is a local execution and
baseline ledger for analysis and can be rotated if it becomes bulky.
