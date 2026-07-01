# Polylogue Devloop

This is the first stop for a contextless agent asked to continue the
Polylogue devloop. It names the executable process, the active state location,
and the durable memory surfaces.

## Start

From `/realm/project/polylogue`:

```bash
.agent/scripts/devloop-status
.agent/scripts/devloop-review
```

Then read, in order:

1. `.agent/conductor-devloop/README.md`
2. `.agent/conductor-devloop/RUNBOOK.md`
3. `.agent/conductor-devloop/INDEX.md`
4. `.agent/conductor-devloop/ACTIVE-LOOP.md` when present
5. `.agent/conductor-devloop/OPERATING-LOG.md` tail
6. `.agent/conductor-devloop/DEMO-RADAR.md`
7. `.agent/includes/README.md`

If `devloop-review` warns, fix the warning or record the conscious exception in
the active loop state before broad work.

## Shape

- `.agent/conductor-devloop/` is the active loop packet. Tracked files explain
  the process; ignored files hold current local state.
- `.agent/includes/` holds tracked durable project/devloop knowledge that should
  survive checkout and context loss.
- `.agent/scripts/` holds tracked executable primitives. Do not copy these into
  the conductor packet.
- `.agent/demos/` is ignored and current-curated. It is the best current demo
  set, not an append-only archive.
- `.agent/scratch/` is ignored supporting research only. It is not active loop
  state.

## Process

Use the shared focus modes exactly:

```text
Direction -> Evidence -> Construction -> Proof -> Artifact -> Velocity -> Meta
```

Record material transitions with:

```bash
.agent/scripts/devloop-focus <from> <to> "<trigger>" "<decision>"
```

Start one concrete slice with:

```bash
.agent/scripts/devloop-start "<slice title>"
```

Refresh generated local state with:

```bash
.agent/scripts/devloop-sync
```

## Current Goal

Conduct the Polylogue dogfood/demo devloop indefinitely: continuously choose the
highest-value live-archive capability slice, produce inspectable artifacts
proving Polylogue improves agents with real history, collapse silos into shared
acquisition/query/projection/rendering substrate, verify on the active archive
or live capture, maintain logs and handoffs, and reprioritize by evidence.

## Defaults

- Default archive root: `/home/sinity/.local/share/polylogue`.
- Production `polylogued.service` should stay inactive during this devloop.
- The intended daemon is `polylogued-devloop.service`.
- Always state archive root, schema version, and relevant counts when quoting
  live archive facts.
- Treat `devloop-status` git fields as part of the start gate: branch, HEAD,
  tracked-change count, and untracked-change count tell you whether you are
  resuming a clean branch, local process edit, or product slice.
- Treat `devloop-review` ignore-policy checks as load-bearing: tracked scaffold
  must survive checkout, while active loop state and demos stay local/current.
- `/realm/inbox` is staging only. The devloop must not depend on it.
