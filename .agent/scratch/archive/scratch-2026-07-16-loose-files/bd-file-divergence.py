#!/usr/bin/env python3
"""File the verified divergence-audit findings as a7xr children. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:120] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:250])


COMMON = ["--parent", "polylogue-a7xr", "-l", "area:substrate,horizon:frontier,tech-tree"]

bd(
    "create",
    "Converger and repair disagree on session_profile staleness for NULL-sort-key sessions",
    *COMMON,
    "--type",
    "bug",
    "-p",
    "2",
    "-d",
    "VERIFIED LIVE 2026-07-06 (divergence audit): daemon/convergence_stages.py:829-836 and "
    "storage/repair.py:566-584 encode DIFFERENT staleness predicates for the same derived rows. "
    "For sessions with sort_key_ms IS NULL the converger compares "
    "strftime of source_updated_at vs updated_at_ms/1000 as strings, while repair COALESCEs the "
    "NULL to 0.0 and applies the 1e-6 epsilon against source_sort_key — a NULL-sort-key session "
    "with non-zero source_sort_key is permanently stale to repair and possibly fresh to the "
    "converger. Consequence: repeated repair churn or missed rebuilds, and the two paths also "
    "source the materializer version differently (constant vs helper call). The two derived-model "
    "maintenance paths can disagree about the same row indefinitely.",
    "--design",
    "One session_profile_stale_predicate(sessions_alias, profile_alias) -> str SQL-fragment "
    "builder in storage/insights/session/runtime.py (next to "
    "SESSION_INSIGHT_MATERIALIZATION_TYPES); both convergence_stages.py and repair.py compose "
    "their queries from it; repair's UNION arms for session_latency_profiles reuse the same "
    "fragment with the lp alias. Materializer version comes from one accessor. Decide the "
    "NULL-sort-key semantics ONCE (the converger's updated_at comparison is the better-considered "
    "branch) and encode it in the fragment. Ties into the cpf temporal doctrine (timeless "
    "sessions).",
    "--acceptance",
    "rg shows exactly one definition of the staleness predicate; a fixture with sort_key_ms NULL "
    "+ source_sort_key set is classified identically by a convergence pass and an ops repair "
    "pass (regression test asserting agreement); no repair churn on a converged archive "
    "(idempotence test: repair immediately after convergence selects zero rows). Verify: "
    "devtools test -k 'staleness or repair'.",
)

bd(
    "create",
    "message_type_backfill reconstructs prose unordered and unfiltered; message-prose SQL exists 5x",
    *COMMON,
    "--type",
    "bug",
    "-p",
    "2",
    "-d",
    "VERIFIED LIVE 2026-07-06: storage/message_type_backfill.py:54-64 claims (comment) to "
    "concatenate block text in position order, but its GROUP_CONCAT has no inner ORDER BY — "
    "SQLite GROUP_CONCAT is unordered, so the #839 classifier can receive scrambled prose. It "
    "also omits the block_type='text' filter (thinking/tool text leaks into classification) and "
    "uses a single-newline separator, while the embeddings/demo family "
    "(storage/embeddings/materialization.py:535/754/923, demo/seed.py:607, demo/constructs.py:240) "
    "uses double-newline + block_type filter + min-length HAVING. Five paste sites, one concept, "
    "one real ordering bug — and demo/constructs.py exists to VERIFY the embedding selector but "
    "pastes the SQL instead of importing it, so drift silently breaks the verification.",
    "--design",
    "message_prose_sql(alias, *, separator, block_types, min_chars) fragment builder next to "
    "archive_embeddable_message_where (the factoring pattern already proven there); backfill "
    "gains ordered concatenation via correlated subquery (SELECT GROUP_CONCAT(text, sep) FROM "
    "(SELECT text FROM blocks WHERE message_id=m.message_id AND ... ORDER BY position)); all "
    "five sites compose the builder; demo/constructs.py imports it (verification becomes real).",
    "--acceptance",
    "One builder; backfill output for a multi-block fixture is position-ordered (regression "
    "test with 3+ blocks inserted out of order); block_type filter applied on the classifier "
    "path; embeddings selection output unchanged (golden). Verify: devtools test -k 'backfill "
    "or message_type or embeddable'.",
)

bd(
    "create",
    "One percentile implementation: three algorithms across five copies skew operator-facing stats",
    *COMMON,
    "--type",
    "task",
    "-p",
    "3",
    "-d",
    "Divergence audit: _percentile exists 5x with three algorithms — linear interpolation "
    "(daemon/status.py:1306, daemon/live_ingest_attempt_progress.py:167, "
    "daemon/cursor_lag_baseline.py:320), nearest-rank q-in-[0,1] (insights/portfolio.py:107), "
    "nearest-rank p-in-[0,100] (archive/semantic/timing.py:37). "
    "live_ingest_attempt_progress.py:170 literally documents copy-discipline ('Matches "
    "cursor_lag_baseline._percentile so operator-facing percentiles stay comparable') instead of "
    "importing. Small samples produce visibly different p50/p95 across surfaces shown side by "
    "side.",
    "--design",
    "core/stats.py: percentile(sorted_values, q, *, method='linear'|'nearest') (core/metrics.py "
    "is host-metrics only — new module is right); timing.py's 0-100 scale becomes call-site "
    "conversion; five deletions. Pick 'linear' as the default operator-facing method (matches "
    "the daemon trio, the majority + the latency surfaces).",
    "--acceptance",
    "One implementation; five sites import it; a small-sample fixture (n=5) yields identical "
    "p95 across status/portfolio/timing paths (test). Verify: devtools test -k percentile.",
)

bd(
    "create",
    "FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle repair copies",
    *COMMON,
    "--type",
    "task",
    "-p",
    "2",
    "-d",
    "Same class as the closed fts_freshness_state double-declaration, three more objects: "
    "trigger DDL for messages_fts/session_work_events_fts/threads_fts lives in BOTH "
    "storage/sqlite/archive_tiers/index.py (:307-324, :729-767, :449-464) and "
    "storage/fts/fts_lifecycle.py (:198-233+ as _BLOCKS/_SESSION_WORK_EVENT/_THREAD trigger DDL "
    "constants used by drop-and-recreate repair). Byte-equivalent today; any future edit forks "
    "trigger behavior between fresh DBs and repaired DBs. No test couples the two sources.",
    "--design",
    "Move trigger DDL lists to storage/fts/sql.py (already holds FTS_INDEX_EXISTS_SQL) as the "
    "single source; archive_tiers/index.py composes its DDL script from them; fts_lifecycle "
    "imports them. Derived-tier regime: pure code move, no schema bump (emitted DDL identical — "
    "assert via normalized-text comparison in the PR). Relates 1xc.12 (drift gauges family).",
    "--acceptance",
    "rg finds each trigger body in exactly one module; a drift test asserts fresh-DB and "
    "repair-path trigger text are identical (normalized); rebuild + repair smoke green. "
    "Verify: devtools test -k fts.",
)

bd(
    "create",
    "parse_archive_datetime: 6 copies, one with different tz semantics (naive/aware time bomb)",
    *COMMON,
    "--type",
    "bug",
    "-p",
    "2",
    "-d",
    "Divergence audit: identical _parse_archive_datetime copies in context/selection.py:285, "
    "mcp/archive_support.py:492, cli/read_views/standard.py:232, api/archive.py:514, "
    "archive/query/archive_execution.py:113 (naive stays naive; empty string raises) vs a "
    "DIVERGENT copy in storage/insights/session/rebuild.py:763 (empty->None; naive FORCED to "
    "UTC). The same stored string parses to offset-naive or offset-aware depending on surface — "
    "a latent TypeError (cannot compare naive and aware) across insight vs read paths. Also "
    "_iso_from_epoch_ms x5 with a strict/lenient split (daemon/provenance.py:84, "
    "storage/embeddings/status_payload.py:338 lenient; three strict one-liners).",
    "--design",
    "core/timestamps.py is the designated home (docstring: unified timestamp parsing, all "
    "operations UTC): add parse_archive_datetime() with the rebuild copy's UTC-forcing "
    "semantics (matches the module contract) + iso_from_epoch_ms(); delete all copies. Audit "
    "each call site for naive-datetime comparisons that silently relied on naive semantics "
    "(mypy + tests are the net). Part of the cpf temporal doctrine surface.",
    "--acceptance",
    "One definition each; all six+five sites import core/timestamps; a test asserts the parsed "
    "value is ALWAYS tz-aware UTC; no naive-vs-aware comparison remains reachable (grep + "
    "focused tests). Verify: devtools test -k timestamp.",
)

bd(
    "create",
    "Role synonym vocabulary maintained by hand in two directions + normalize_role name collision",
    *COMMON,
    "--type",
    "task",
    "-p",
    "3",
    "-d",
    "core/enums.py:127-134 Role.normalize maps synonyms->canonical; "
    "archive/message/roles.py:18-24 ROLE_SQL_VALUES holds the SAME sets inverted for SQL "
    "role-filter expansion. Adding a synonym to one without the other makes --role filters "
    "silently miss rows (developer/progress/result were evidently added by hand to both). No "
    "coupling test. BONUS COLLISION: two unrelated exported functions named normalize_role — "
    "surfaces/payloads.py:431 (pass-through, ''->'unknown') vs archive/message/roles.py:11 "
    "(canonicalizing) — wrong-import failure mode.",
    "--design",
    "ROLE_SYNONYMS: dict[Role, frozenset[str]] once in core/enums.py; Role.normalize iterates "
    "it; ROLE_SQL_VALUES becomes a derivation/re-export (~20 lines). Rename the payloads "
    "function to role_label (its actual semantics). Coupling test: every synonym in "
    "ROLE_SYNONYMS round-trips through Role.normalize.",
    "--acceptance",
    "One synonym table; SQL expansion derived; rename done with call sites updated; coupling "
    "test in place. Verify: devtools test -k role.",
)

bd(
    "create",
    "Index-tier sibling-path derivation pasted ~7x with divergent existence rules",
    *COMMON,
    "--type",
    "task",
    "-p",
    "3",
    "-d",
    "Seven daemon/CLI sites re-derive 'the index.db next to this anchor' with different "
    "fallback behavior — provenance.py:194 uses the path even when absent, fts_status.py:156 "
    "returns None when missing, embedding_backlog.py:60 builds a candidate list + requires a "
    "sessions table — while paths/_roots.py:76 already exports resolve_active_index_db_path. "
    "Status surfaces can disagree about whether the archive exists.",
    "--design",
    "Extend polylogue/paths with sibling_index_db(anchor, *, require_exists: bool) and sweep "
    "the seven sites (convergence_stages.py:868, similarity.py:324, embedding_backlog.py:60, "
    "fts_status.py:156, provenance.py:194, cli/commands/status.py:189, daemon/cli.py:182); "
    "embedding_backlog keeps its table probe locally on top of the resolved path.",
    "--acceptance",
    "One derivation; seven sites swept; a missing-index fixture yields the SAME verdict from "
    "every status surface (test). Verify: devtools test -k 'status or paths'.",
)

bd(
    "create",
    "Mechanical helper dedup sweep: scalar coercion quadruplet, _table_exists x40, provenance vocab x6, title/tags mixin",
    *COMMON,
    "--type",
    "chore",
    "-p",
    "3",
    "-d",
    "Bundle of zero-risk verbatim-copy consolidations from the divergence audit: (a) daemon "
    "scalar-coercion helpers (_required_str/_optional_str/_row_int/_row_float) copied across "
    "5+ status modules with ALREADY-diverged signatures (row_float -> float vs float|None) "
    "while core/payload_coercion.py is the designated home; (b) _table_exists defined 40x "
    "(41 with the schema variant) — the codebase's single most duplicated function; (c) "
    "_range_timing_provenance/_date_provenance emitting the timestamped_range/... vocabulary "
    "verbatim in SIX modules across four packages; (d) Session vs SessionSummary duplicating "
    "display_title/tags/summary property logic including the pasted #1240 comment "
    "(domain_runtime.py:64-87 vs summary_runtime.py:36-55).",
    "--design",
    "(a) add row_int/row_float/required_str raising variants to core/payload_coercion.py, "
    "sweep daemon modules; (b) table_exists(conn, name, *, schema='main') + async twin in "
    "storage/sqlite/, mechanical sweep; (c) define once in archive/session/provenance.py with "
    "object|None signature, five deletions; (d) shared mixin for the title/tags precedence "
    "rules. All four are boilerplate-agent-shaped; mypy --strict is the net. One PR per letter "
    "or one sweep PR — batching judgment to the executor.",
    "--acceptance",
    "rg counts: one definition each for the swept helpers; mypy --strict green; no behavior "
    "goldens change. Verify: devtools verify (testmon picks affected).",
)

print("--- divergence beads filed")
