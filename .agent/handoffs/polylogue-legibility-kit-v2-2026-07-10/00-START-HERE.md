# Polylogue + Sinex external legibility, second edition

This iteration turns the first package from a broad strategy document into an executable launch control plane.

The most useful starting points are:

1. [Iteration audit](01-ITERATION-AUDIT.md) — what the first edition got right, what it still blurred, and what changed.
2. [Public story v2](02-PUBLIC-STORY-V2.md) — exact category, narrative, audience framing, and page structure for both projects.
3. [Proof portfolio v3](03-PROOF-PORTFOLIO-V3.md) — a full reconsideration of the demos, with independent oracles, negative controls, falsifiers, and rejected concepts.
4. [Launch control plane](04-LAUNCH-CONTROL-PLANE.md) — 72-hour, seven-day, and 30-day cuts backed by a 23-mission DAG.
5. [Single-machine swarm runbook](05-SWARM-RUNBOOK-V2.md) — worktree, resource, leasing, handoff, and integration procedure.
6. [Maximal interop contract](06-MAXIMAL-INTEROP-CONTRACT.md) — the transcript-complete Sinex-backed target and its real impedance mismatches.
7. [Beads delta](07-BEADS-DELTA.md) — exact existing Beads to execute, supersede, or re-scope.
8. [Sixteen fork prompts](fork-prompts/README.md) — parallel missions ready to paste into forks of the current conversation.

## The executable anchors

The Polylogue patch now includes a real deterministic command:

```bash
polylogue demo receipts
```

It seeds and traverses normal parser, storage, query, and ref paths to compare an assistant success claim with a structurally failed test result, a later repair, and a prose-only anti-grep control. It is a deterministic contract proof, not a real-project field proof.

The package also includes [Incident 14:32](incident-1432/README.md), a product-independent oracle corpus spanning transcript, terminal, Git, Beads, lineage, source coverage, assertion judgment, context delivery, and parser replay. Its verifier currently confirms all declared material hashes and 24 oracle facts.

## Run the package validators

```bash
python scripts/legibilityctl.py all \
  --polylogue-repo /path/to/polylogue \
  --sinex-repo /path/to/sinex \
  --polylogue-beads /path/to/polylogue-beads-export.jsonl \
  --sinex-beads /path/to/sinex-beads-export.jsonl
```

Initialize and inspect the swarm:

```bash
python scripts/swarmctl.py init --reset
python scripts/swarmctl.py ready --horizon 72h
python scripts/swarmctl.py claim PLG-72-01 --agent receipts-a --worktree /worktrees/plg-72-01
```

## Apply-ready repository changes

The package contains full patches relative to the supplied repository bases:

- `patches/polylogue-legibility-v2-full.patch`
- `patches/sinex-legibility-v2-full.patch`

The Polylogue patch includes executable product behavior, tests, claims linting, generated-site work, deterministic media, and a fix for nondeterministic generated-surface rendering. The Sinex patch remains primarily public-story and architecture-contract work; this environment did not contain a Rust/Sinex runtime stack for execution.

## Current honest boundary

Already proved here:

- Polylogue can produce a structural claim-versus-receipt verdict on its deterministic provider-shaped corpus.
- The anti-grep control prevents prose keywords from acting as the operational oracle.
- The corpus passes 34 of 34 declared construct checks.
- Polylogue’s public claims ledger is machine-checked and part of release readiness.
- The independent Incident 14:32 corpus is internally consistent and publicly safe.
- Sinex–Polylogue authority, identity, and transcript-revision contracts validate statically.

Still planned rather than proved:

- one public-safe real merged-PR Receipts packet;
- the full provider-neutral semantic transcript renderer;
- complete degraded-state UI behavior;
- the Sinex Missing Source and replay demonstrations against a running stack;
- transcript-complete admission into Sinex;
- a full Polylogue SQLite rebuild from Sinex;
- general agent-performance uplift from memory or context.
