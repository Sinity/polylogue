## Turn 71 (role=assistant) — heredoc writing 6 legibility-deliverable template files: WORKER-HANDOFF.md, INTERFACE-DECISION.md, COLD-READER-REPORT.md, DEMO-PROOF-REPORT.md, swarm-board.tsv, resource-plan.sh

bash -lc set -euo pipefail
mkdir -p /mnt/data/legibility-deliverables/templates
cat > /mnt/data/legibility-deliverables/templates/WORKER-HANDOFF.md <<'EOF'
# Worker handoff

Package: `P00 / S00 / C00`
Repository/worktree:
Base commit:
Branch:
Owning Beads:

## Result

State: `ready_for_review | needs_rework | abandoned`

One-paragraph statement of what now works and what does not.

## Commits

- `<sha>` — purpose

## Files owned and changed

- `path` — why

## Interfaces introduced or changed

- Name:
- Contract:
- Consumers:
- Compatibility and migration note:

## Verification actually run

```text
exact command
exact result
```

## Proof artifacts

- Manifest/report/transcript/screenshot path:
- Reproduction identity:
- Negative/refusal control result:
- Private-path or secret scan result:

## Claims affected

- Claim ledger ID:
- Status supported:
- Approved wording:
- Wording that remains unsupported:

## Known caveats and unresolved risks

- Concrete caveat with file/ref or evidence path.

## Integration instructions

- Cherry-pick order:
- Expected conflicts:
- Generated files to refresh after landing:
- Follow-up Beads that should be added or amended by the sole Beads captain:
EOF

cat > /mnt/data/legibility-deliverables/templates/INTERFACE-DECISION.md <<'EOF'
# Interface decision

Decision ID:
Date:
Owner/package:
Consumers:
Related Beads:

## Problem

What cross-worker incompatibility or ambiguity must be resolved?

## Decision

State the exact wire, data, CLI, rendering, identity, or lifecycle contract.

## Invariants

1. 
2. 
3. 

## Current versus target

- Current implementation:
- This campaign lands:
- Explicitly deferred:

## Compatibility

- Existing producers/readers:
- Version marker:
- Fallback behavior:
- Failure/refusal behavior:

## Evidence and verification

- Fixture/ref:
- Test command:
- Artifact:

## Rejected alternatives

- Alternative and the concrete reason it was rejected.
EOF

cat > /mnt/data/legibility-deliverables/templates/COLD-READER-REPORT.md <<'EOF'
# Adversarial cold-reader report

Reader:
Repository revision(s):
Environment:
Started from:
Time/command budget:

## Category comprehension

Without quoting the README, answer in one sentence:

- Polylogue is:
- Sinex is:
- Together they are:

## Public path attempted

Record every documented command in order, including failures and ambiguity.

## Questions the product answered

For each answer, include the exact source or product ref.

1. What consequential question was answered?
2. Which evidence was source material versus interpretation?
3. Which claim was refused or caveated?
4. Could the result be regenerated?

## Unsupported or misleading claims

| Public wording | Why unsupported or ambiguous | Evidence | Required correction |
|---|---|---|---|

## Construct-validity audit

- Ground truth independent of product: `yes/no`
- Credible simpler baseline: `yes/no`
- Positive control: `pass/fail`
- Negative control: `pass/fail`
- Refusal/incompleteness control: `pass/fail`
- Exact refs resolve: `pass/fail`
- Private material absent: `pass/fail`
- Reproduction identity present: `pass/fail`

## Navigation and vocabulary failures

List every place where a term, ref, command, proof class, or current/target distinction required private explanation.

## Verdict

`ship | ship after listed corrections | do not ship`

The verdict must cite blocking evidence rather than preference.
EOF

cat > /mnt/data/legibility-deliverables/templates/DEMO-PROOF-REPORT.md <<'EOF'
# Demo proof report

Demo ID:
Claim ID:
Implementation revision:
Fixture/material digest:
Generated at:
Privacy class:

## Falsifiable claim

One sentence.

## Consequential decision

What could a user or agent decide after seeing this result that they could not safely decide from the baseline?

## Ground truth

How ground truth was planted or independently obtained. It must not be derived from the product output being evaluated.

## Baseline

The strongest plausible simpler method and its exact command/output.

## Product path

Exact command(s), configuration, and output directory.

## Controls

| Control | Expected | Observed | Evidence ref | Result |
|---|---|---|---|---|
| Positive | | | | |
| Negative | | | | |
| Refusal/incomplete | | | | |

## Result

State only what the evidence establishes.

## Artifact chain

- Machine receipt:
- Human report:
- Command transcript:
- Structured manifest:
- Visual recording:
- Exact source refs:

## Threats to validity

- Selection effects:
- Fixture realism:
- Leakage:
- Baseline weakness:
- Generalization limits:
- Performance caveats:

## Claim ledger update

- Previous status:
- New status supported:
- Approved public wording:
- Forbidden stronger wording:
EOF

cat > /mnt/data/legibility-deliverables/templates/swarm-board.tsv <<'EOF'
slot	project	branch	base	bead_ids	state	owned_paths	last_commit	tests	blockers
p01-readme	polylogue	legibility/p01-readme	<poly-base>		queued				
p02-receipts-demo	polylogue	legibility/p02-receipts-demo	<poly-base>		queued				
p03-renderer	polylogue	legibility/p03-renderer	<poly-base>		queued				
p04-findings	polylogue	legibility/p04-findings	<poly-base>		queued				
p05-lineage-cost	polylogue	legibility/p05-lineage-cost	<poly-base>		queued				
p06-resume	polylogue	legibility/p06-resume	<poly-base>		queued				
p07-onboarding	polylogue	legibility/p07-onboarding	<poly-base>		queued				
p08-visual-launch	polylogue	legibility/p08-visual-launch	<poly-base>		queued				
s01-readme-visual	sinex	legibility/s01-readme-visual	<sinex-base>		queued				
s02-golden-demo	sinex	legibility/s02-golden-demo	<sinex-base>		queued				
s03-outage	sinex	legibility/s03-outage	<sinex-base>		queued				
s04-replay	sinex	legibility/s04-replay	<sinex-base>		queued				
s05-producer	sinex	legibility/s05-producer	<sinex-base>		queued				
s06-interop	sinex	legibility/s06-interop	<sinex-base>		queued				
c01-agent-work-packet	both	legibility/c01-agent-work-packet	poly:<poly-base>,sinex:<sinex-base>		queued				
r01-adversarial-review	both	(read-only integration branches)	poly:<poly-base>,sinex:<sinex-base>		queued				
EOF

cat > /mnt/data/legibility-deliverables/scripts/resource-plan.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4)"
mem_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 16777216)"
mem_gib=$(( mem_kib / 1024 / 1024 ))

if (( mem_gib < 24 )); then
  cargo_jobs=2
  pytest_workers=1
  simultaneous_editors=8
elif (( mem_gib < 48 )); then
  cargo_jobs=3
  pytest_workers=2
  simultaneous_editors=12
elif (( mem_gib < 96 )); then
  cargo_jobs=4
  pytest_workers=2
  simultaneous_editors=16
else
  cargo_jobs=6
  pytest_workers=3
  simultaneous_editors=16
fi

# Never let build parallelism consume every logical CPU; integration and the UI need headroom.
if (( cargo_jobs > cores - 2 )); then
  cargo_jobs=$(( cores > 3 ? cores - 2 : 1 ))
fi

cat <<PLAN
Detected: ${cores} logical CPUs, approximately ${mem_gib} GiB RAM

Recommended starting values:
  export CARGO_BUILD_JOBS=${cargo_jobs}
  export POLYLOGUE_PYTEST_WORKERS=${pytest_workers}
  export LEGIBILITY_MAX_ACTIVE_EDITORS=${simultaneous_editors}
  export LEGIBILITY_LOCK_ROOT=/realm/tmp/worktrees/legibility/locks

Scheduling:
  - keep one Sinex compile-heavy process behind with-heavy-lock.sh;
  - allow documentation/static-analysis forks to run concurrently;
  - run only focused Polylogue tests during worker phases;
  - reserve broad checks, media rendering, and full target refreshes for the proof captain;
  - if swap or sustained load becomes visible, pause workers rather than starting another build.

These are conservative starting values, not performance claims about either repository.
PLAN
EOF
chmod +x /mnt/data/legibility-deliverables/scripts/resource-plan.sh
bash -n /mnt/data/legibility-deliverables/scripts/resource-plan.sh
/mnt/data/legibility-deliverables/scripts/resource-plan.sh | sed -n '1,30p'


---

