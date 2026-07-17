# Single-machine frontier-agent swarm runbook

## Operating principle

Optimize for independent file ownership, fast evidence, and disposable integration history. Do not optimize for a pretty branch graph while the launch cut is moving.

Use checkpoint commits because they are the cheapest transferable artifact between worktrees. Permit ugly WIP commit messages, merge commits, and later squash. The non-negotiable hygiene is semantic: no two agents simultaneously own the same files, no unreported generated-file drift, and no claim without a validation artifact.

## Roles

One long-lived coordinator process owns:

- the integration branch;
- the worktree registry;
- Beads status changes and dependency edits;
- shared contract files;
- full validation and release artifacts;
- conflict arbitration.

Specialized agents own bounded lanes. Agents may inspect the whole repository but only edit their declared paths.

A separate validation captain runs expensive checks. This prevents sixteen workers from each saturating CPU, disk, SQLite, PostgreSQL, or Nix builds.

## Worktree topology

Use one integration worktree and one worktree per lane.

```bash
mkdir -p ../swarm/polylogue ../swarm/sinex ../swarm/out

git -C polylogue worktree add -b launch/integration ../swarm/polylogue-integration master
git -C sinex worktree add -b launch/integration ../swarm/sinex-integration master

# Example lane
git -C polylogue worktree add -b launch/readme ../swarm/polylogue/readme master
git -C polylogue worktree add -b launch/demo-receipts ../swarm/polylogue/demo-receipts master
git -C sinex worktree add -b launch/concepts ../swarm/sinex/concepts master
```

The included `scripts/bootstrap-worktrees.sh` creates the recommended lane set after paths are supplied.

## Shared coordination directory

Keep coordination outside every repository so branch switches and merges cannot destroy it.

```text
../swarm/out/
  registry.csv
  locks/
  status/
  contracts/
  evidence/
  patches/
  validation/
```

Each agent writes `status/<lane>.json` after every meaningful checkpoint:

```json
{
  "lane": "polylogue-demo-receipts",
  "state": "ready_for_integration",
  "branch": "launch/demo-receipts",
  "commit": "<sha>",
  "owned_paths": ["polylogue/demo", "polylogue/scenarios", "tests/unit/cli/test_demo_command.py"],
  "validation": ["targeted test command and result"],
  "generated_files": [],
  "known_failures": [],
  "contract_changes": []
}
```

Use an atomic write (`tmp` then rename). Do not coordinate by editing one shared Markdown checklist from every worker.

## File ownership

A lane has write authority over a path set. Cross-lane contract changes are proposed in `out/contracts/` and merged by the contract owner.

Recommended launch lanes:

| Lane | Repository | Exclusive write scope |
|---|---|---|
| P1 narrative | Polylogue | `README.md`, `pyproject.toml`, landing-page copy |
| P2 docs | Polylogue | new high-level docs, docs registry/config |
| P3 claims/findings | Polylogue | claims ledger, findings templates/pages |
| P4 receipts fixture | Polylogue | demo/scenario fixture and verification |
| P5 tour/report | Polylogue | demo tour runner and demo CLI tests |
| P6 semantic contract | Polylogue | provider-neutral transcript card model/registry |
| P7 terminal renderer | Polylogue | terminal backend and snapshots |
| P8 web renderer | Polylogue | web backend and snapshots |
| P9 web reliability | Polylogue | daemon HTTP/query scheduling and slow-route UX |
| P10 install/media | Polylogue | install matrix harness, tapes, launch artifacts |
| S1 narrative/concepts | Sinex | README and top-level concepts/agent docs |
| S2 demo contract | Sinex | common demo packet/runner/manifest |
| S3 moment demo | Sinex | deterministic multi-source moment fixture/command |
| S4 replay demo | Sinex | interpretation-revision fixture/command |
| S5 coverage outage | Sinex | outage fixture/coverage surface |
| J1 backend ADR | Both, docs only | cross-project authority/identity/protocol documents |
| J2 Agent Work Packet | Both, isolated contract paths | combined fixture and packet schema |

Sixteen workers do not need sixteen simultaneous code-writing lanes. Run static/docs/research lanes immediately; admit CPU/IO-heavy lanes according to capacity.

## Contract-first handoffs

Before implementation agents diverge, freeze small shared contracts:

1. semantic transcript card JSON shape;
2. demo manifest/report shape;
3. stable Polylogue/Sinex ref envelope;
4. public claim states;
5. context/Agent Work Packet artifact shape.

The contract owner lands the schema and fixtures first. Backend agents branch from that commit or cherry-pick it. This is faster than reconciling two attractive but incompatible implementations later.

## Integration strategy

Agents create checkpoint commits. The integration captain chooses the fastest of:

```bash
# Clean lane with isolated files
git merge --no-ff launch/readme

# One useful commit among noise
git cherry-pick <sha>

# History is irrelevant, patch applies cleanly
git diff master...launch/demo-receipts --binary > ../swarm/out/patches/demo-receipts.patch
git apply --3way ../swarm/out/patches/demo-receipts.patch

# Conflict is cheaper to replace than merge
rsync -a --delete ../swarm/polylogue/readme/docs/new-surface/ ./docs/new-surface/
git add -A && git commit -m 'WIP integrate public docs lane'
```

Do not rebase every lane whenever integration advances. Rebase only when a lane cannot validate against the current contract. Otherwise integrate, run focused checks, and repair on the integration branch.

## Resource scheduling on one machine

### Cheap, parallel work

Run concurrently:

- documentation and copy;
- static source inspection;
- small unit tests;
- fixture and manifest design;
- claims audit;
- screenshots from already-built static pages.

### Serialized or capacity-limited work

Use one token for each scarce class:

- **full Python validation token** — one `devtools verify --all` or broad pytest run;
- **site/media token** — one site rebuild or VHS/browser capture at a time;
- **Nix/Rust build token** — one large Sinex build/test lane at a time unless memory headroom is proven;
- **live archive token** — one agent may touch the real archive or daemon;
- **database/infra token** — one fault-injection or schema/replay campaign owns PostgreSQL/NATS;
- **embedding/network token** — one effect-bearing model or embedding run, with a declared budget.

The validation captain maintains token files under `out/locks/`. Agents unable to obtain a token continue with static work rather than starting competing heavy jobs.

## Isolation rules

Polylogue agents use unique roots:

```bash
export XDG_DATA_HOME="$PWD/.sandbox/data"
export XDG_CONFIG_HOME="$PWD/.sandbox/config"
export XDG_CACHE_HOME="$PWD/.sandbox/cache"
export POLYLOGUE_ARCHIVE_ROOT="$PWD/.sandbox/archive"
export POLYLOGUE_FORCE_PLAIN=1
```

Web/daemon lanes allocate distinct ports and record them in lane status. No lane reads the operator's live archive unless its prompt explicitly grants the live-archive token.

Sinex implementation lanes should use the repository's sandbox/xtask facilities and unique environment/database namespaces. Only one integration lane owns shared NATS/PostgreSQL infrastructure. Demo fixtures should be private-data-free and recreatable from source.

## Validation ladder

Every lane runs the cheapest relevant rung before handoff.

1. syntax/format or documentation build;
2. focused unit/contract tests for owned files;
3. nearby package/module tests;
4. generated-surface drift checks;
5. integration demo on fresh fixture;
6. full repository gate, run by the validation captain;
7. cold-reader and secret/path scrub;
8. optional live/private field proof, never required for deterministic public claims.

A lane may hand off with a known broad failure only when the focused oracle passes and the failure is recorded with ownership.

## Merge order

1. Shared claim/demo/ref contracts.
2. Static narrative and docs.
3. Deterministic fixture changes.
4. Semantic renderer contract.
5. Terminal and web renderers.
6. Tour/report integration.
7. Web reliability.
8. Findings and media generated from the integrated product.
9. Install matrix and release audit.
10. Beads reconciliation and final claims ledger.

Media is generated late because it captures the product and copy. Generating it before renderer and wording convergence creates avoidable churn.

## Beads coordination

One agent owns `.beads/issues.jsonl` changes. Other agents cite Bead IDs in commits and status packets but do not independently edit the Beads export. At integration checkpoints, the coordinator:

- marks actual work started or closed;
- records proof commands and artifact paths;
- adds any newly discovered blockers;
- supersedes the metadata-only Polylogue/Sinex authority decision with an explicit maximal-backend decision;
- avoids closing a broad Bead when only a launch slice landed.

## Failure policy

- A broken lane is quarantined, not allowed to block unrelated lanes.
- A contract conflict is decided once by the contract owner, then propagated.
- A flaky or slow test becomes an artifact with command, seed, timing, and owner.
- A live demo failure is not edited out; it becomes a caveat or blocker.
- When merging is slower than reimplementation, copy the validated file or reproduce the small patch on integration and cite the discarded lane.

## Final swarm deliverables

The integration captain should finish with:

- a clean or knowingly dirty release candidate worktree;
- exact commit/patch provenance for every lane;
- deterministic demo packets;
- generated site and media;
- validation report;
- public claims ledger;
- secret/absolute-path scrub;
- Beads update proposal;
- release/no-release decision with blockers.
