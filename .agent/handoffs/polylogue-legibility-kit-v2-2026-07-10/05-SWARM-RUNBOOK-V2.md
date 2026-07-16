# Single-machine frontier-agent swarm runbook v2

The objective is maximal validated throughput on one machine. Git history quality is secondary. Shared state corruption, hidden proof drift, and full-suite contention are not.

## Roles

### Coordinator

Owns the integration worktree, mission state, shared contracts, public claims, and release packet. Only this role runs full verification or publishes generated artifacts.

### Implementation agent

Owns a bounded mission and write-path set. Produces code, focused tests, proof receipts, and a handoff. Does not broaden scope to “clean up” adjacent modules.

### Evidence agent

Builds fixtures, oracles, proof packets, and adversarial controls. It should ideally not write the product code whose conclusion it scores.

### Adversarial reader

Attacks claims and demos without preserving momentum. It can block launch but does not own implementation.

### Build operator

Serializes Rust builds, the live PostgreSQL/NATS stack, browser recording, and broad tests. This can be the coordinator on a modest machine.

## Worktree layout

Use one root outside either repository:

```text
/worktrees/
  integration-polylogue/
  integration-sinex/
  plg-72-01/
  plg-72-02/
  ...
  jnt-30-01-polylogue/
  jnt-30-01-sinex/
```

Joint missions receive one worktree in each repository but one mission lease. Their handoff must declare cross-repository ordering.

Create worktrees from the same base commit for each horizon unless the coordinator intentionally rebases the whole wave. Do not let agents choose arbitrary newer bases.

The supplied bootstrap script creates disposable branches:

```bash
scripts/bootstrap-worktrees-v2.sh \
  --polylogue /repos/polylogue \
  --sinex /repos/sinex \
  --root /worktrees \
  --horizon 72h
```

## Isolation rules

Each worktree gets unique:

- archive roots;
- temporary directories;
- database names;
- NATS streams/subjects where possible;
- HTTP ports;
- browser profiles;
- test basetemp;
- generated-output directory;
- model cache/output path when experiments run.

Never point two implementation lanes at the same mutable demo archive or PostgreSQL schema.

## Mission leasing

The coordinator initializes `swarmctl.py`. Agents do not start from the CSV by hand.

```bash
python scripts/swarmctl.py ready --horizon 72h
python scripts/swarmctl.py claim PLG-72-03 --agent claims-ci --worktree /worktrees/plg-72-03
```

A lease checks:

- dependency completion;
- mission state;
- scarce resource capacity;
- active write-root overlap;
- wildcard integration ownership.

If an agent cannot continue, release the mission with a reason rather than abandoning an active lease.

## File ownership

A write path is a lease, not a suggestion. Read broadly; write narrowly.

The main collision zones are:

- `polylogue/cli/` and command registration;
- renderer descriptors shared by CLI and web;
- `docs/public-claims.yaml`;
- generated site templates;
- demo seed/construct registries;
- Sinex event/identity primitives;
- interop schemas;
- package and lock files.

When two missions need the same file, sequence them or nominate one contract owner. Do not rely on later conflict resolution to reconcile semantic choices.

## Resource scheduling

### Python-heavy lanes

At most two. Examples: archive seeding, broad query changes, rendering, data migrations. Four smaller docs/test lanes can run around them.

### Rust-heavy lane

One. Other Sinex agents can inspect, design, or write docs while Cargo links, but they should not launch independent full builds.

Use shared target caches only when branches have compatible dependency graphs and the cache itself is not being cleaned. Otherwise isolate target dirs to avoid agents invalidating one another.

### Live Sinex stack

One. Allocate named time windows to missions. Snapshot receipts after each mission and reset or use separate database/schema prefixes before the next.

### Full verification

One, in the integration worktree. Implementation agents run focused tests and perhaps package-local checks. Repeating the full suite sixteen times wastes hours and increases flake noise.

### Browser and media

One display/profile lane. Generate from deterministic fixtures after the integration commit, not from each agent branch.

## Handoff contract

Every mission copies [`templates/HANDOFF.md`](templates/HANDOFF.md) to
`.agent-handoff/HANDOFF.md`. `swarmctl finish` rejects a handoff unless it keeps
these exact headings:

```markdown
# Mission
# Base commit
# Changed files
# Verification
# Known failures
# Merge recommendation
```

Put owned paths, contracts, generated artifacts and hashes under **Changed
files**. Put exact commands, exit states, counts, and receipt paths under
**Verification**. Put unresolved defects, timeouts, unsupported environments,
privacy caveats, and expected conflicts under **Known failures**. The merge
recommendation must choose one of: cherry-pick, apply patch, copy bounded files,
manual synthesis, mine the implementation and discard the branch, or do not
merge.

A handoff without exact test results is incomplete even when the code looks
plausible.

### Lease heartbeat and stale work

Claims default to a 12-hour lease. Long-running agents extend it explicitly:

```bash
python scripts/swarmctl.py heartbeat PLG-72-01 \
  --agent receipts-a \
  --extend-hours 12 \
  --note "fixture and focused tests complete; packaging next"
```

The coordinator can inspect expired leases and then release them atomically:

```bash
python scripts/swarmctl.py reap-stale --dry-run
python scripts/swarmctl.py reap-stale --reason "agent stopped without a handoff"
```

An expired lease is visible in `swarmctl status`; it is not silently stolen by
another agent.

## Integration strategy

The coordinator is allowed to be ruthless.

Use:

- cherry-pick when the branch is narrow and clean;
- `git diff <base> -- <owned-paths>` when commits are noisy;
- file copy when generated artifacts are the only useful output;
- manual synthesis when two agents implemented incompatible contracts;
- reject/re-run when the oracle is invalid or the scope is uncontrolled.

Do not spend time preserving a beautiful commit graph if a clean patch can be assembled faster.

Merge order:

```text
independent corpus/oracle
→ core evidence and readiness contracts
→ shared renderer descriptors
→ proof commands
→ claims gate
→ public copy
→ generated site
→ media
→ install proof
→ field proof
→ integration verification
```

## Coordination without conversational memory

Use files as the blackboard:

```text
.agent-handoff/
  HANDOFF.md
  test-results.json
  artifacts.sha256
  conflict-notes.md
```

The mission state records who owns what. Beads record project intent and acceptance. Polylogue/Sinex may eventually record the work itself, but the launch swarm should not depend on the feature it is trying to build.

## Failure triage

Classify failures before fixing them:

- **product defect** — implementation contradicts contract;
- **fixture defect** — corpus does not encode the intended construct;
- **oracle defect** — scorer depends on product output or ambiguous ground truth;
- **environment defect** — unavailable Rust/NATS/browser dependency;
- **performance defect** — bounded command exceeds budget;
- **claim defect** — implementation works but public wording is broader;
- **integration defect** — individually valid branches conflict semantically.

This classification belongs in the handoff and final validation ledger.

## When to stop the swarm

Stop and integrate when the current horizon’s public claim is proved. Do not keep every lane alive because more improvement is possible.

Stop immediately for:

- private-data leakage;
- unresolvable evidence refs;
- an oracle generated from the product answer;
- silent partial or timeout results;
- stable identity tied to replay-specific event IDs;
- two active authorities for reviewed state;
- uncontrolled edits outside leased paths;
- repeated full-suite contention;
- a growing conflict queue with no contract owner.
