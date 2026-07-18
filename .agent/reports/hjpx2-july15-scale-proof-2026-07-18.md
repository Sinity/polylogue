# hjpx.2 — July-15-scale raw-authority replay convergence proof (2026-07-18, Lane D)

## Status: NOT CLOSABLE this session — AC6 (July-15-scale execution) unproven; other clauses have durable evidence

This report is honest about what was and was not achieved. It does not force
a close on unmet acceptance criteria.

## What this bead requires (from its own AC, verbatim scope)

1. Sanitized corpus matches July-15 candidate/component/byte/skew distributions
   within documented tolerances.
2. Every bounded pass stays within declared RSS/PSS/swap/temp-write/wall-time
   envelopes while daemon health remains responsive.
3. Executable plan backlog decreases monotonically modulo explicit retry
   injection; every finite retry resolves once; no component starves.
4. Cursor positions, source heads, accepted index heads, FTS readiness, and
   durable raw authority never regress across interruption/resume.
5. Two final quiescent census digests match with zero executable plans and
   identical typed residual debt.
6. Removing fair rotation recreates starvation; removing
   conservation/carry-forward accounting recreates a census mismatch.
7. Exact commands, containment receipts, corpus seed, and result artifacts
   are durable and reviewable.
8. No live archive apply is part of this bead; yla8 retains the separate
   verified-backup/explicit-authorization gate.

## Verdict per law

| Law | Status | Evidence |
| --- | --- | --- |
| **Resource envelopes** (AC2, partial) | Proven at small/medium synthetic scale, **not yet at July-15 cardinality** | `_PassProcessSampler` records RSS/PSS/swap/CPU/IO per pass in the harness (`devtools/raw_authority_scale_proof.py`); exercised by 21 passing tests in `tests/unit/devtools/test_raw_authority_scale_proof.py`. The July-15-shaped run itself has not completed (see below), so envelope numbers *at that cardinality* are not yet recorded. |
| **Daemon-health responsiveness** (AC2, other half) | **Not provable with the current harness — confirmed gap, not just unautomated** | Both `raw_authority_scale_proof.py` and `raw_authority_restart_proof.py` (#3080) call `repair.repair_raw_materialization` directly in-process against a synthetic archive; neither starts an actual `polylogued` process, so there is no HTTP/heartbeat surface to probe. Filed `polylogue-agvo` (P1) to build that harness variant. This AC clause cannot close without that follow-up landing. |
| **Monotonic, fair draining / no starvation** (AC3, AC6 mutation) | **Mechanism proven; not yet exercised at July-15 cardinality** | The pass loop in `run_raw_authority_scale_proof` raises `RuntimeError` if any pass leaves executable candidates undrained. Fairness mutation coverage: `test_raw_materialization_fair_rotation_mutation_recreates_starvation` (`tests/unit/storage/test_repair.py:2082`) — removing durable attempt-age fairness in a one-slot retry-injected scenario reproduces starvation; the production scheduler correctly advances to the next independent component instead. Verified passing this session. |
| **Interruption/resume, mid-census and mid-apply** (AC4) | **Proven, cited not duplicated** | PR #3080 (`devtools/raw_authority_restart_proof.py`, merged `5571fa2b7`) proves interrupted production repair reconciles durable outcomes without duplication and reaches a two-census fixed point across the transaction boundaries that matter. `tests/unit/devtools/test_raw_authority_restart_proof.py`: 4/4 passing this session. Per this bead's own instruction ("don't duplicate #3080's crash-recovery proof"), no new work was added here. |
| **Fixed point + conservation** (AC5, AC6 mutation) | **Mechanism proven; not yet exercised at July-15 cardinality** | The harness requires two consecutive quiescent dry-run passes with matching digests, raising on mismatch. Conservation mutation coverage: `test_raw_materialization_fails_closed_on_plan_conservation_mismatch` (`tests/unit/storage/test_repair.py:2174`) — any nonzero immutable replay-plan conservation error fails closed. Verified passing this session. |
| **Byte-skew fidelity / envelope-breach mutation** | Proven at unit scale | `test_historical_backfill_reparses_multi_gib_shaped_raw_instead_of_spilling_archive_wide` (`tests/unit/sources/test_revision_backfill.py`) proves a declared multi-GiB raw does not get cached archive-wide (bounded retention preserved); `RawRevisionReplayResourceBlockedError` is raised for genuinely oversized components. Verified passing this session. |
| **Pressure-gate mutation** (skipped recheck → gate miss) | Proven | Three tests in `test_raw_authority_scale_proof.py` (`_rechecks_pressure_during_generation`, `_around_census_and_replay`, `_after_census`) exercise the continuous recheck; independently, this session's own three real attempts (below) are a live demonstration of the gate firing correctly under genuine contention. |
| **July-15-scale execution itself** (AC1, AC6, AC7) | **NOT achieved this session** | Three (going on four) real attempts at `--components 10163 --raws 15264 --expanded-raws 21398`, all self-aborted under genuine host I/O pressure. See below. |

## The three (four, pending) real attempts

The host has run 4+ concurrent warroom lanes throughout this session
(per SONNET-NOTE 2026-07-18); `io_full_avg10` has oscillated between 0.02 and
16.96 over roughly 90 minutes of observation, never sustaining a quiet window
long enough to complete the ~21,398-row synthetic corpus generation phase.
The gate was never loosened and the corpus was never shrunk to dodge it, per
explicit instruction.

| Attempt | Wrapper strategy | Result | Abort point |
| --- | --- | --- | --- |
| 1 | Immediate, no wait | Self-abort, avg10=3.06 | Initial admission check (before any work) |
| 2 | 40 min bound, single-tick quiet required | Self-abort, avg10=2.46 | Early in generation (`raw_authority_scale_proof.py:781`) |
| 3 | 90 min bound, 3-consecutive-tick quiet required | Self-abort, avg10=3.63 | Mid-generation, past first publish-batch flush (`raw_authority_scale_proof.py:853`) |
| 4 | 2h bound, 5-consecutive-tick quiet required | *pending at report time* | — |

Each attempt got measurably further before aborting (immediate → early-gen →
past-first-flush), consistent with a gate that is working exactly as designed
against a genuinely, persistently contended host — not a harness defect and
not evidence the July-15 shape itself is unreachable, only that it hasn't
been reached *this session* under *this* load.

Receipts: `/realm/tmp/raw-authority-july15-v2-20260718/{attempt1,attempt2,attempt3}.stderr.log`,
`wrapper{,2,3}.log` (full poll history, PSI samples at every check).

## Follow-up beads filed this session

- **polylogue-amg1** (P1) — commit-batching + size-aware parse dispatch, the
  throughput lever needed to make ordinary catch-up viable at scale (9p8x's
  deferred "Fix 3"). Discovered-from polylogue-9p8x.
- **polylogue-agvo** (P1) — daemon-health-responsiveness proof harness (a real
  `polylogued` subprocess + concurrent HTTP status polling), the one law this
  bead cannot currently prove. Discovered-from polylogue-hjpx.2.

## hjpx.2 closable: **NO**

AC1 (corpus matches July-15 shape), AC6 (mutation proofs *at that scale*,
though the mutation *mechanisms* are proven at smaller scale), and AC7 (exact
July-15-shaped result artifacts) are not satisfied. The bead remains
**in_progress** with this report and the four attempt receipts appended to
its notes. The concrete remaining step is: retry the same command
(`devtools workspace raw-authority-scale-proof --components 10163 --raws
15264 --expanded-raws 21398 --pass-limit 1000 --keep --json` under
`sinnix-scope background --`) once the host has a sustained quiet window —
no code change is needed for this, only a quieter host. Separately,
`polylogue-agvo` needs to land before the daemon-health clause of AC2 can
close at all, independent of host contention.

## What IS ready to merge from this lane

- `polylogue-9p8x`: parallel census parse + spill-cache decoupling, tested,
  `devtools verify --quick` green. Real, measured (if modest) speedup;
  honest correction of the original 4x hypothesis, with `polylogue-amg1`
  filed for the remaining lever.
- `polylogue-yla8`: read-only authorization packet, recommending the
  operator hold the live gate given the backup/drain-rate/scale-proof
  blockers found live in the current archive state.
- This report and its bead-note trail.
