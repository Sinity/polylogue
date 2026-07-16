# Launch control plane

The plan is deliberately split into three horizons. The first makes Polylogue presentable. The second makes the presentation materially stronger. The third proves the joint architecture.

The executable source is [control/mission-plan.yaml](control/mission-plan.yaml). It contains 23 missions, real Beads, dependencies, resources, write paths, outputs, verification commands, stop conditions, and merge policy.

## Present state

The weighted scorecard currently assigns 64 of 100 points. The exact evidence and gaps are in [current-legibility-score.csv](control/current-legibility-score.csv). The score is not a maturity metric. It is a launch-control heuristic weighted toward the first public experience.

The strongest completed dimensions are structural evidence, reproducibility, and category clarity. The largest deficits are degraded-state product behavior, semantic rendering, field proof, and the joint backend proof.

## The 72-hour cut

Goal:

> A cold reader can install Polylogue, run one trustworthy proof, understand why it is not grep, and see exactly what remains unproved.

Execution order:

1. `PLG-72-01` — Receipts and public claims gate.
2. `PLG-72-02` — readiness/degradation vocabulary.
3. `PLG-72-03` — claims CI.
4. `PLG-72-04` — README and anti-grep copy.
5. `PLG-72-05` — evidence-first site.
6. `PLG-72-06` — proof media.
7. `QA-72-01` — cold-reader and adversarial-claims attack.
8. `INT-72-01` — integration and release packet.

The current artifact package already implements most of `PLG-72-01`, a substantial part of `PLG-72-03`, the copy/site/media slices, and the deterministic generated-surface fix. The readiness/degradation mission remains the largest 72-hour implementation gap.

### 72-hour stop conditions

Do not launch when:

- the proof verdict depends on prose keywords;
- `uvx --from` cannot run the proof cleanly;
- generated surfaces drift;
- a timeout looks like an empty result;
- public copy implies that deterministic proof satisfies the real-PR Bead;
- the Sinex backend is described as shipped;
- any public artifact contains private paths or corpus names.

## The seven-day cut

Goal:

> The public arc includes semantic receipts, one field-valid reconstruction, truthful lineage accounting, visible degradation, and a clean package proof.

Missions:

- `PLG-7D-01` — narrow shared semantic renderer.
- `PLG-7D-02` — one real merged-PR Receipts packet.
- `PLG-7D-03` — Count It Once with fresh-subagent control.
- `PLG-7D-04` — findings shelf.
- `PLG-7D-05` — clean install and release proof.
- `PLG-7D-06` — degraded-state web rendering.
- `INT-7D-01` — release candidate integration.

The crucial sequencing rule is that findings and copy consume executable packets. They do not lead them.

## The 30-day joint cut

Goal:

> Prove that Sinex can become the complete durable backend of Polylogue without erasing Polylogue semantics, stable refs, local operation, or epistemic caveats.

Missions:

- `SNX-30-01` — Missing Source.
- `SNX-30-02` — Import It Twice and Changes Its Mind Honestly.
- `JNT-30-01` — stable identity across replay.
- `JNT-30-02` — immutable transcript revision and settlement.
- `JNT-30-03` — SQLite rebuild and parity.
- `JNT-30-04` — World Around the Claim / Agent Work Packet.
- `EXP-30-01` — Resume Under Oath.
- `INT-30-01` — joint release integration.

This horizon should not begin with a mass historical import. It begins with one deterministic complete revision, one interrupted admission, one replay, and one rebuild. Scale follows the identity and settlement proof.

## Resource model

The plan encodes these single-machine capacities:

| Resource | Capacity | Reason |
| --- | ---: | --- |
| Polylogue Python-light | 4 | Small docs/tests can coexist. |
| Polylogue Python-heavy | 2 | Archive generation and broad tests contend. |
| Sinex Rust-heavy | 1 | Linking and incremental caches dominate memory/disk. |
| PostgreSQL/NATS stack | 1 | One isolated proof stack avoids port and state contamination. |
| Browser/VHS | 1 | Shared display and recording resources. |
| Full verification | 1 | Broad suite belongs to the integrator. |
| Integration worktree | 1 | One authority publishes product state. |
| Static docs | 6 | Low-cost when file ownership is bounded. |

## Why not start all sixteen prompts at once

Prompt count is not machine capacity. The correct launch is wave-based:

- Wave 0: executable proof, claims gate, and read-only copy audit.
- Wave 1: readiness, site, and media on distinct resources.
- Wave 2: cold-reader attack.
- Wave 3: exclusive integration.

The seven-day and 30-day waves begin only after their dependency cut is integrated. This prevents every agent from building against a different public contract.

## Machine commands

Validate the graph:

```bash
python scripts/legibilityctl.py mission-plan \
  --polylogue-beads /path/polylogue-beads-export.jsonl \
  --sinex-beads /path/sinex-beads-export.jsonl
```

Initialize state:

```bash
python scripts/swarmctl.py init --reset
python scripts/swarmctl.py ready --horizon 72h
```

Claim a mission:

```bash
python scripts/swarmctl.py claim PLG-72-01 \
  --agent receipts-a \
  --worktree /worktrees/plg-72-01
```

Keep a long-running lease alive and record one-line progress:

```bash
python scripts/swarmctl.py heartbeat PLG-72-01 \
  --agent receipts-a \
  --extend-hours 12 \
  --note "focused proof green; assembling handoff"
```

Finish it with a machine-validated handoff:

```bash
cp templates/HANDOFF.md /worktrees/plg-72-01/.agent-handoff/HANDOFF.md
python scripts/swarmctl.py finish PLG-72-01 \
  --agent receipts-a \
  --handoff /worktrees/plg-72-01/.agent-handoff/HANDOFF.md
```

The state file is lock-protected. A mission cannot be claimed before dependencies complete, beyond resource capacity, or while its write roots conflict with an active mission. Expired leases are visible and can be reaped explicitly; they are never silently stolen.
