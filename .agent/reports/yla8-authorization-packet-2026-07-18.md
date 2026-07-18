# yla8 authorization packet — 2026-07-18 (Lane D)

**Read-only.** No live gate, backup, catch-up, reset, replay, or SQL mutation
was run to produce this packet. All commands below were `--check`/`status`/
read-only queries against the live archive
(`POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue`).

**Provisional caveat (per SONNET-NOTE 2026-07-18):** the archive is mid-restore
from the same-day poisoned-index incident (two divergent `index.db` identities
— see `polylogue-yla8` notes). A fresh index generation
(`gen-1784381541560-2d4fc3f4`) was promoted and the daemon (build 0.3.0)
restarted at 17:51 CEST, ~9 minutes before this packet's status snapshot. Every
count below reflects that in-flight state, not a settled archive. Lane A
flagged this first; this packet does not repeat that diagnosis, only the
counts relevant to an authorization decision.

## Recommendation

**Do not authorize the live gate yet.** Four independent blockers, detailed
below. None is a new regression from this session — they are the current,
directly observed state.

## 1. Current population counts vs July-15 baseline

| Metric | 2026-07-15 (yla8 preflight) | 2026-07-18 18:00 CEST (this packet) | Comparable? |
| --- | --- | --- | --- |
| Build | source v11/index v36/embeddings v2/user v8/ops v1 | source v13/index v40/embeddings v3/user v10/ops v1 | schema advanced on every tier |
| Raw artifacts (source.db) | ~18,347 checked (frontier sample) | 79,571 (`raw_artifact_count`) | archive grew; live capture continued through the incident |
| Materialized (index) | not directly comparable (frontier used a different metric) | 170 (`materialized_raw_artifact_count`) | fresh index, 99.8% unconverged |
| Join gap (source rows with no index session) | — | 79,401 (`join_gap_count`) | **this is the backlog a live gate would need to drain** |
| Broken active heads | 1,890 / 18,347 checked | 0 / 170 checked (`broken_head_status: healthy`) | **NOT comparable** — sample is 170 of 79,571 rows (0.2%), not a population verdict |
| Cursor-ahead | 40 / 739 comparisons | unknown / 1 comparison | **NOT comparable** — same reason |
| Incomparable cursor/head | 34 | — (not reported at this sample size) | **NOT comparable** |
| Cursor-authority gaps | — | 26 (partial, early signal — mostly Antigravity brain metadata sidecars) | new projection, not in the 07-15 baseline |
| Critical archive-debt groups | 5, affecting 207 artifacts | excluded from this bounded snapshot (`archive_debt.available: false`) | not captured this pass |
| Replay backlog (raw-authority) | 15,264 direct / 21,398 expanded / 10,163 components / 4.788 GB | excluded from this bounded snapshot (`raw_replay_backlog.available: false, reason: excluded_from_bounded_status_snapshot`) | not captured this pass — needs a dedicated raw-authority census, deliberately not run live this session (see §5) |
| FTS readiness | — | **not ready**: 0/0 indexed, `messages_fts` freshness `stale`, "archive message FTS drift exceeds bounded startup reconciliation" | blocking search until convergence catches up |
| Quarantined raws | 28 (durable authority-debt) | 248 (`raw_quarantined`) | grew; not yet reconciled against the new generation |

**Bottom line:** the archive-wide frontier-integrity numbers from July-15 are
**not reproducible from current evidence** — the fresh index has materialized
0.2% of the source-tier raw population, so any "broken heads: 0" or
"cursor-ahead: unknown" reading right now is a sample-size artifact, not a
health verdict. A meaningful re-run of the July-15-style frontier census needs
to wait until materialization has substantially caught up, or needs its own
bounded sampling strategy that doesn't assume the index is populated.

## 2. Backup-manifest currency verdict — **BLOCKS authorization**

- Most recent **formal, verified** per-tier backup (`polylogue-sqlite-backup.service`,
  zstd-compressed, timestamped): **2026-07-12T03:17:17Z** — **6 days stale**.
  Predates the 2026-07-15 authority work, the entire hjpx/lkrc/yla8 program
  through today, and the 2026-07-18 poisoned-index incident and restore.
- An informal snapshot exists at
  `/realm/staging/polylogue-sqlite/pre-deploy-20260718T132033Z/` (source.db +
  user.db only, copied 15:20 CEST today) but it is **not** a
  `polylogue ops backup --profile full_evidence --verify` output: no
  `manifest.json`, no `verification-receipt.json`, missing the
  index/embeddings/ops tiers entirely. It does not satisfy this gate's backup
  prerequisite.
- `polylogue ops backup --output-dir /realm/staging/polylogue-sqlite/yla8-preflight-check-DRYRUN --profile full_evidence --check` (read-only, no directory created) reports **"Backup prerequisites: OK"** — the *mechanism* is available; no backup has actually been taken against the *current* (post-restore) archive identity.
- **Verdict: no current verified full-evidence backup exists. This alone blocks authorization** per this gate's own AC2.

## 3. Blocker inventory

1. **polylogue-5jak (P0, open, unclaimed)** — daemon raw-materialization
   conveyor processes `raw_artifact_limit=1` every 30s (`daemon/cli.py:70-72`),
   ~2 rows/min. Directly evidenced today: watcher-level catch-up (a related
   but distinct stage) measured `files_per_second: 0.184867`,
   `ingest_worker_count_max: 1` in the most recent ingestion batch receipt. At
   that rate, draining the current 79,401-row join gap takes **~4.97 days**
   for watcher catch-up alone; 5jak's own math for the separate
   raw_materialization conveyor (~1 row/30s) is **~27.6 days** for a
   comparably sized backlog. **"Just let the daemon drain it" is not a viable
   plan at current throughput** — this is not a hypothesis, it's today's
   measured rate applied to today's measured backlog.
2. **polylogue-emx2 (P1, open, unclaimed)** — watcher catch-up trusts
   `ingest_cursor` rows as "materialized" when they only mean "acquired."
   After today's index reset, cursors pointing at the now-empty fresh
   generation risk being treated as already-satisfied and skipped, pushing
   more of the drain onto the slow conveyor in (1) instead of the fast
   parallel batch pipeline. `cursor_authority_gap_count: 26` is an early,
   partial signal of this class of gap (not yet a population-wide count,
   for the same 170-row sample-size reason as §1).
3. **polylogue-5vft (P2, open, unclaimed)** — maintenance CLI hygiene
   (`--only-missing` promotion, `reset --index` managed-generation escape,
   self-healing preflight). Not directly blocking this gate, but is why the
   07-18 incident needed manual layout surgery instead of a sanctioned repair
   path; relevant context for "why we're here."
4. **polylogue-hjpx.2 (P1, in progress — this lane)** — the raw-authority
   replay convergence proof at July-15 scale has **not yet completed**. Corpus
   preparation is retrying under the continuous I/O pressure gate (host is
   contended: 4+ concurrent warroom lanes; `io_full_avg10` has ranged 3.8–11.2
   this session, above the 2.0 admission threshold). The executor this gate
   would authorize (`RawAuthorityReconciler` / `repair_raw_materialization`)
   has proof coverage at small-fixture scale (hjpx.1, merged) but not yet at
   the archive's actual cardinality. See hjpx.2 bead notes for live receipts
   as they land.

FTS convergence also shows one failed stage today (`messages_fts`, "startup
found inconsistent messages_fts ready freshness ledger", retry due) — likely
transient post-restore noise, not separately gating, but should clear before
re-checking readiness.

## 4. Immutable plan digest

**Not captured this session.** `raw_replay_backlog` and `archive_debt` were
both excluded from the bounded `ops status --json --full` snapshot
(`reason: excluded_from_bounded_status_snapshot`). Computing a raw-authority
census digest requires either (a) a dedicated `repair_raw_materialization(dry_run=True)`
call, or (b) waiting for materialization to progress. Given the daemon is
*actively* mid-drain right now (single writer, `raw_artifact_limit`-bounded
passes), this session deliberately did not invoke a competing manual
raw-authority operation against the live archive — that risks contending with
the daemon's own writer coordinator during an already-fragile restore. This is
a gap in this packet, not a finding of "no debt": treat the plan digest as
**unknown, not zero**, until a dedicated read-only census is run once the
daemon's own drain has progressed or is paused for that purpose.

## 5. Exact apply command (for reference — not authorized to run)

```
# Prerequisite: a fresh verified backup (see §2)
POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue ops backup --output-dir /realm/staging/polylogue-sqlite/yla8-<UTC-timestamp> \
  --profile full_evidence --verify

# The gate itself, once backup + hjpx.2 proof + blocker triage are done:
# ordinary packaged daemon catch-up under bounded journal capture — no
# cursor reset, force replay, evidence deletion, or ad hoc SQL writes.
# (Mechanism already running automatically; this gate is about whether to
# let it keep running unattended vs. pausing for 5jak/emx2 first.)
```

## 6. Expected duration / resources

From hjpx.2's proof envelopes: **not yet available** — the scale proof has not
completed a pass at July-15 cardinality this session (host pressure gate
self-aborted the first two attempts; see hjpx.2 bead notes for exact PSI
receipts). From today's *live* measurement instead: at the observed
0.185 files/sec watcher-catchup rate and 5jak's documented ~1-row/30s
raw-materialization rate, draining the current 79,401-row gap is a
**multi-day-to-multi-week** operation, not hours. This number will only
improve once 5jak's backlog-aware batch scaling lands.

## 7. Postflight checks (once authorized and run)

Re-run `ops status --json --full`; require `raw_frontier_integrity.overall_status`
computed over a population-representative sample (not the current 170-row
slice), `join_gap_count` trending to 0, `fts_readiness.invariant_ready: true`,
and zero unresolved `raw_authority_ledger_counts.unresolved_blockers`.

## 8. Abort conditions

Any of: I/O or memory pressure gate breach during the live pass; a new
`RawRevisionReplayResourceBlockedError` on a previously-clean component;
`convergence.failed_count` growing instead of shrinking; daemon heartbeat age
exceeding its configured staleness threshold; any cursor advancing past
unmaterialized content (the original no-shrink invariant this whole program
protects).

## 9. Rollback statement

The daemon's own convergence is already the "live" state — there is no
separate mutation to roll back from *authorizing continued automatic drain*.
If a *manual* gate action were taken (none is proposed here) and needed
rollback, the durable tiers (`source.db`, `user.db`) are restorable from the
most recent verified backup (currently 6 days stale — see §2); `index.db` and
`embeddings.db` are rebuildable-tier and would be regenerated from source, not
restored byte-for-byte.

---

## The operator's question

**Authorize the live raw-authority catch-up gate now, given: no current
verified backup (§2), a 79,401-row unconverged materialization gap that would
take days-to-weeks at today's measured throughput without polylogue-5jak
(§3.1), an uncomputed plan digest (§4), and hjpx.2's scale proof still
in-flight (§3.4)? (yes / no)**
