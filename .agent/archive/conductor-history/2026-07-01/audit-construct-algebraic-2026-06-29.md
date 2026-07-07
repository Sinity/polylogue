
---

## CORRECTION (2026-06-29, adversarial verification) — run-projection materialization is LOAD-BEARING

The "run-projection trio materialization is UNREAD / safe to cut" finding above is **WRONG** and must
not be acted on. Adversarial verification proved the tables ARE read by the `run`/`observed-event`/
`context-snapshot` query units across CLI/MCP/API/daemon:
- SELECTs: `storage/sqlite/archive_tiers/archive.py:4071` (session_runs), `:4120` (session_observed_events),
  `:4168` (session_context_snapshots).
- Query-unit wiring: `archive/query/metadata.py:779/792/805` (`sql_query_method=query_runs/...`).
- Dispatch: `archive/query/unit_results.py:248-252`; reached by CLI `cli/archive_query.py:1591`,
  MCP `mcp/archive_support.py:413`, API `api/archive.py:2458`, daemon `daemon/http.py:2823`.
- Parity contract: `tests/unit/insights/test_run_projection_materialization.py` asserts read-back parity (#2384).
- Recovery MCP tools (`recovery_report`/`recovery_work_packet`) DO recompute via `compile_recovery_digest`
  (api/archive.py:1649/1670) — independent of the tables.

**Correct fix (not deletion):** the ~76% cost is the regex prose-mining in
`compile_recovery_digest._events_from_text`, NOT the storage. Keep the query units; REBUILD their events from
the keystone's structured tool outcomes (blocks.tool_result_is_error / tool_result_exit_code, schema v16) and/or
make the mining lazy/on-read. `POLYLOGUE_SKIP_RUN_PROJECTION=1` (rebuild.py dev stopgap) degrades the units to
empty — it is a dev shortcut, NOT the real fix. Already shipped (3fc78f492): the recovery REPORT renders these
events as unverified heuristic candidates so the surface stops asserting them as fact.

---

## RE-FRAME (2026-06-29, operator correction) — reference-count ≠ legitimacy; CUT the invalid layer

The "CORRECTION" above (don't cut, it's load-bearing) MISSED THE POINT. The reason to look at the
run-projection layer was architectural cleanliness/coherence/simplicity — not just perf, and NOT "is it
referenced?". The adversarial check verified the three tables are READ by the run/observed-event/
context-snapshot query units — and I wrongly concluded "preserve." That is preserving extant structure at all
costs, the exact anti-goal.

Correct architectural judgment:
- `run` / `observed-event` / `context-snapshot` query units are derived from the regex prose-miner
  (`_events_from_text`) — the SAME construct-invalid pattern as `work_events`: text shape promoted to fact
  (e.g. "PR #123 merged" fabricated). Being wired into CLI/MCP/API/daemon makes them MORE invalid-insight
  surface, not legitimate.
- They are ALSO a parallel duplicate of the recovery digest, which recomputes the same projection live
  (api/archive.py:1670). Two paths for one capability = the duplication the "algebra over silos" lens says to
  collapse.

THE MOVE (coherence + honesty + simplicity), batch into v18 reingest:
1. DELETE the regex prose-miner `_events_from_text` and the prose-mined event kinds that have no structured or
   external-truth basis (pr_merged/issue_closed/test counts from prose).
2. Recovery/observed events come ONLY from STRUCTURED evidence: the keystone (blocks.tool_result_is_error/
   exit_code) for command/test outcomes; external truths (PR/issue/CI) come from git/GitHub, NOT transcript prose
   (so they leave the transcript-derived layer entirely).
3. COLLAPSE the duplication: one projection path. Either the query units lower to the (keystone-sourced) digest,
   or drop the units + the three materialized tables + the parity test outright if the digest already serves the
   need. Default to removal unless a unit proves distinct value over the digest.
Net: less code, fewer tables, no fabricated facts, one coherent projection. That is the win — not preserving the
wired-up invalid layer.
