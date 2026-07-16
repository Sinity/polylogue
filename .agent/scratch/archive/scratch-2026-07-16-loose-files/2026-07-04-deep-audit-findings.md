---
created: 2026-07-04
purpose: Single-threaded deep-read findings -> beads (goal: figure out unencoded work)
status: active
project: polylogue
---
# Deep-audit findings (single-threaded)
Rhythm: read a subsystem -> find concrete unencoded issue -> file bead -> log here.

## dispatch.py (read 2026-07-05)
- `_generic_messages_session` (710-729): hardcodes created_at=None/updated_at=None, drops any timestamp present in generic `{messages,...}` payloads. CONFIRMED minor data-loss for drive-like/fallback message docs. -> bead.
- `_merge_duplicate_parsed_sessions` (414-418): sums reported_duration_ms across merged same-session fragments; if a fragment reports session-total (not per-fragment) duration this double-counts. VERIFY vs claude.parse_code_stream.
- same fn (396): active_leaf = last message in concatenated file order; true tip may not be last-in-file for sidechained Claude Code. VERIFY.

## write_effects.py (read 2026-07-05)
- commit_archive_write_effects (line 125): WriteResult.operation_id=str(uuid4()) is a fresh random, unrelated to payload _operation_id (the lease/telemetry id). Shown to user at cli/commands/maintenance.py:378. Minor traceability smell -> maybe fold into a write-effects polish bead.
- blob-lease acquire/release + failure re-release is CORRECT (#1746). No bug.

## HIGH-LEVERAGE FEATURE FILED
- Set-algebra over query results (union/intersect/except) -> child of fnm, P2. The read surface becomes Set(Query) x Projection x Render. Grounded in expression.py:623 grammar + fnm.9 subquery infra.

## convergence_stages.py (read 2026-07-05) — SYSTEMIC BUG FOUND
- Freshness PROBE handlers fail-closed to 'converged' + no log: fts check(105-106) return False, check_many(161-162) return set(); insights (342,412) same. A probe error -> silent permanent staleness. Filed P2 bug under 1xc (relates 1xc.9). Distinct from false_means_pending execute-path.
- coordination/envelope.py:572 `except Exception: return None` — broad catch in new code; archive probe errors -> silent 'archive absent'. (minor, noted)

## coordination/envelope.py newer parts (read 2026-07-05)
- SYSTEMIC connection leak: `with sqlite3.connect() as conn` commits-not-closes at ~9 sites (envelope:591 x3/build, user_state_resolver:59/67/91, api/archive:2931/4626, raw_payload/decode:309, repair:112, demo/seed:82). otlp_correlation:116 already fixed w/ comment. Filed P2 under a7xr.
- s7ae.4 ALREADY IMPLEMENTED: _archive_evidence_payloads wired at envelope.py:81, composes tree/activity/proof/context-flow. Commented as close-candidate.

## message_query_reads.py lineage composition (read 2026-07-05)
- Silent truncation x2: _MAX_LINEAGE_DEPTH=64 (deep acompact chains) + dangling-branch-point fallback-to-own (tail only, missing shared prefix). No completeness signal in read envelope. Filed P2 under 4ts.
- async blocking scan of daemon: CLEAN. mutable-defaults/naive-datetime scans: CLEAN. (codebase well-disciplined)
