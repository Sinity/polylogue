# Polylogue 0x7nh reindex canary differ implementation packet

Date: 2026-08-09

Branch: `feature/maintenance/reindex-canary-differ-roundout`

Status: implementation complete. The first production canary report and the
reviewed zero-unclassified-diff receipt remain open under `polylogue-0x7nh`.
The Bead is not closed by this packet.

## Scope and disposition

This lane owns the implementation-grade slice of the reindex canary differ.
It does not claim the first live report, production source remediation, or
manual review of that report. No live canary was run and no production archive
was mutated.

| Acceptance area | Disposition | Evidence |
| --- | --- | --- |
| Daemon-owned, inactive, no-promote replay | Satisfied | `run_daemon_canary_rebuild` submits the rebuild through `daemon_write_coordinator` with `promote=False`; the runner does not call the rebuild primitive directly. |
| Representative selection | Satisfied | Existing origin quotas are retained and explicit pathology plus live-anchor selections are carried through the selection receipt. |
| One canonical comparator | Satisfied | The production `compare_reindex_generations` route remains the comparison authority; no second row-difference vocabulary was introduced. |
| Evidence binding | Satisfied | Reports bind source root, current index, candidate index, selected raw/session IDs, replay closure, parser fingerprints, lowering fingerprint, and rebuild receipt. |
| Row coverage | Satisfied | Canonical relation discovery covers sessions, messages, blocks, `session_links`, and derived rows, with direct lineage coverage in tests. |
| Fail-closed classification | Satisfied | Durable reports require exact review coverage. Expected rows require a structured `bead:` or `delta:` authority; unexpected rows require a structured `successor:` authority. |
| First production zero-diff review | Open | Requires the first post-remediation production report and human review. No receipt was fabricated. |

## Historical transplant and supersession proof

Historical commit `1f334acee1c0e8c07addb9e6998c70805cf4caa1` is not an ancestor
of the fresh-master base. The ancestry check returned exit status 1. Its old
`polylogue/maintenance/reindex_canary.py` and unit test were not copied
blindly.

The capability was already substantially superseded on master by the current
canary route and its hardening sequence:

- `1ac438c1a` added the maintenance canary gate.
- `e1da54320` refused live-archive canary rebuilds.
- `0b943aaa4` rejected invalid candidates before insights.
- `3822bc771` separated canary replay evidence from durable state.
- `64edbf518` closed provenance-boundary gaps.
- `e6228af5c` isolated inactive candidate durable writes.

This branch therefore transplanted only the remaining differ/report contract:
daemon ownership, parser and source binding, structured classification, and
real-route coverage for the current architecture.

## Implementation commits

- `3ebcfe4d5` `feat: harden daemon-owned reindex canary differ`

The signed commit contains the production route, CLI adaptation, real-route
fixtures, and anti-vacuity tests.

## Anti-vacuity evidence

- Candidate construction is asserted through the captured inactive candidate
  request and `promote=False`; the real no-promote path also verifies that the
  active index remains unchanged.
- The active index is not accepted as both comparator inputs; the route and
  active-generation rotation tests preserve distinct current and candidate
  evidence.
- A `session_links` inheritance mutation produces a `session_links` row diff,
  proving the relation is not omitted from the canonical census.
- Parser fingerprint mutation and raw membership/logical-key expansion both
  reject changed evidence before comparison or approval.
- Missing review coverage and invalid authority kinds fail closed; a durable
  report cannot contain an unclassified difference.

## Verification

All focused tests used the managed harness with
`POLYLOGUE_PYTEST_WORKERS=1`:

```text
direnv exec . env POLYLOGUE_PYTEST_WORKERS=1 python -m devtools test tests/unit/maintenance/test_reindex_canary.py
45 passed

direnv exec . env POLYLOGUE_PYTEST_WORKERS=1 python -m devtools test tests/unit/cli/test_reindex_canary_cli.py
30 passed

direnv exec . python -m devtools render all --check
exit 0; generated surfaces sync OK

direnv exec . env POLYLOGUE_PYTEST_WORKERS=1 python -m devtools verify --quick
exit 0; format, lint, mypy, render, layering, and policy checks passed
```

`git diff --check` passed. The then-current tracker policy also reported an
inherited source-snapshot mismatch for `polylogue-7rds`. That generated policy
was retired later; this line records the historical branch result rather than
a command that remains available. This lane did not invoke `bd`, edit `.beads`,
or change that Bead state.

The full suite and `--seed-testmon` were not run. No production or live canary
was run.

## Publication boundary

This is a clean signed branch and a non-draft publication packet. It is ready
for publication when the coordinator's frontier permits a slot. Do not close
`polylogue-0x7nh` from this packet: the implementation is complete, but the
first production canary report and reviewed zero-unclassified-diff receipt
remain an explicit live gate under that Bead. No named successor in the frozen
Bead record owns that residual work.
