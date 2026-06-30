# Polylogue Conductor Devloop

This is the tracked contextless resume entrypoint for the active Polylogue
dogfood/demo loop.

The process docs and helper scripts are tracked. Current loop state is local and
ignored: `ACTIVE-LOOP.md`, `OPERATING-LOG.md`, `EVENTS.jsonl`, `DEMO-RADAR.md`,
generated manifests, demos, scratch notes, and task history are not meant to be
committed.

If the current chat history is cleared and the operator says only:

```text
continue the devloop setup in .agent
```

then do this:

1. From `/realm/project/polylogue`, run `.agent/scripts/devloop-status`.
2. Run `.agent/scripts/devloop-review`.
3. Read `RUNBOOK.md`.
4. If local `ACTIVE-LOOP.md` exists, read it and continue that slice.
5. If local state is absent, start in `Direction`: choose a slice from the goal,
   current repo state, and any available archive/daemon evidence.
6. If review reports stale generated scaffold state, run
   `.agent/scripts/devloop-sync` and review again.
7. Use the focus modes in `RUNBOOK.md`: Direction, Evidence, Construction,
   Proof, Artifact, Velocity, and Meta.
8. Record material transitions with `.agent/scripts/devloop-focus`; start new
   slices with `.agent/scripts/devloop-start`.
9. Before ending, refresh/review and leave `ACTIVE-LOOP.md` plus
   `OPERATING-LOG.md` resumable.

The process goal is indefinite: continuously choose the highest-value
live-archive capability slice, produce inspectable artifacts proving Polylogue
improves agents with real history, collapse silos into shared
acquisition/query/projection/rendering substrate, verify on the active archive
or live capture, maintain logs and handoffs, and reprioritize by evidence.

Default state lives here, not in `/realm/inbox`. The default demo shelf is
`.agent/demos`, and it is current-curated rather than append-only.
