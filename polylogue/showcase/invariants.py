"""Showcase-level invariant checks.

Defines universal invariants that hold across ALL showcase exercises.
These are the showcase equivalent of property tests: conditions that
must be true regardless of which exercise produced the output.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.outcomes import OutcomeStatus
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.runner import ExerciseResult

# Sentinel for skipping an invariant check on a particular exercise
SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class Invariant:
    """A universal invariant that must hold across showcase exercises."""

    name: str
    description: str
    check: Callable[..., str | None]

    def applies_to(self, exercise: Exercise) -> bool:
        return True


@dataclass(slots=True)
class InvariantResult:
    invariant_name: str
    exercise_name: str
    status: OutcomeStatus
    error: str | None = None


def _check_json_valid(result: ExerciseResult) -> str | None:
    args_str = result.exercise.args_text
    if "-f json" not in args_str and "--format json" not in args_str:
        return SKIP
    if result.exercise.output_ext not in (".json", ".jsonl"):
        return SKIP
    if not result.output.strip():
        return SKIP
    if result.exercise.output_ext == ".jsonl":
        for i, line in enumerate(result.output.strip().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                return f"Invalid JSON on line {i}: {e}"
        return None
    try:
        json.loads(result.output)
        return None
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


def _check_exit_code(result: ExerciseResult) -> str | None:
    expected = result.exercise.assertion.exit_code
    if expected is None:
        return SKIP
    if result.exit_code != expected:
        return f"exit code {result.exit_code}, expected {expected}"
    return None


def _check_nonempty_output(result: ExerciseResult) -> str | None:
    if result.exercise.writes:
        return SKIP
    args_str = result.exercise.args_text
    if "--count" in args_str:
        return SKIP
    if "--help" in args_str or "--version" in args_str:
        return SKIP
    if not result.output.strip():
        return "Empty output for read command"
    return None


SHOWCASE_INVARIANTS: list[Invariant] = [
    Invariant("json_valid", "All -f json output parses as valid JSON", _check_json_valid),
    Invariant("exit_code", "Exit code matches validation spec", _check_exit_code),
    Invariant("nonempty_output", "Non-count read commands produce non-empty output", _check_nonempty_output),
]


def check_invariants(results: list[ExerciseResult]) -> list[InvariantResult]:
    invariant_results: list[InvariantResult] = []
    for result in results:
        if result.skipped:
            continue
        for invariant in SHOWCASE_INVARIANTS:
            if not invariant.applies_to(result.exercise):
                invariant_results.append(InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.SKIP))
                continue
            try:
                error = invariant.check(result)
            except Exception as e:
                error = f"invariant check crashed: {e}"
            if error == SKIP:
                invariant_results.append(InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.SKIP))
            elif error is None:
                invariant_results.append(InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.OK))
            else:
                invariant_results.append(
                    InvariantResult(invariant.name, result.exercise.name, OutcomeStatus.ERROR, error=error)
                )
    return invariant_results


def format_invariant_summary(results: list[InvariantResult]) -> str:
    passed = sum(1 for r in results if r.status is OutcomeStatus.OK)
    failed = sum(1 for r in results if r.status is OutcomeStatus.ERROR)
    skipped = sum(1 for r in results if r.status is OutcomeStatus.SKIP)
    lines = [f"Invariant Checks: {passed} pass, {failed} fail, {skipped} skip"]
    failures = [r for r in results if r.status is OutcomeStatus.ERROR]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for f in failures:
            lines.append(f"  {f.invariant_name} @ {f.exercise_name}: {f.error}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Database-level semantic invariants
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DbInvariantResult:
    invariant_name: str
    status: OutcomeStatus
    error: str | None = None


def check_fts_integrity(db_path: str | Path) -> DbInvariantResult:
    from polylogue.storage.sqlite.connection import open_connection

    try:
        with open_connection(str(db_path)) as conn:
            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            src_count = conn.execute("SELECT COUNT(*) FROM messages WHERE text IS NOT NULL AND text != ''").fetchone()[
                0
            ]
    except Exception as e:
        return DbInvariantResult("fts_integrity", OutcomeStatus.ERROR, f"query failed: {e}")
    if fts_count != src_count:
        return DbInvariantResult(
            "fts_integrity",
            OutcomeStatus.ERROR,
            f"FTS count {fts_count} != source count {src_count} (delta: {src_count - fts_count})",
        )
    return DbInvariantResult("fts_integrity", OutcomeStatus.OK)


def check_provider_meta_roundtrip(db_path: str | Path) -> DbInvariantResult:
    import json as _json

    from polylogue.storage.sqlite.connection import open_connection

    try:
        with open_connection(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT id, provider_meta FROM messages WHERE provider_meta IS NOT NULL LIMIT 1000"
            ).fetchall()
    except Exception as e:
        return DbInvariantResult("provider_meta_roundtrip", OutcomeStatus.ERROR, f"query failed: {e}")
    if not rows:
        return DbInvariantResult(
            "provider_meta_roundtrip", OutcomeStatus.SKIP, "no messages with provider_meta to verify"
        )
    mismatches = 0
    for row in rows:
        try:
            meta = _json.loads(row["provider_meta"])
            re_serialized = _json.dumps(meta, sort_keys=True)
            re_parsed = _json.loads(re_serialized)
            re_re_serialized = _json.dumps(re_parsed, sort_keys=True)
            if re_serialized != re_re_serialized:
                mismatches += 1
        except Exception:
            mismatches += 1
    if mismatches:
        return DbInvariantResult(
            "provider_meta_roundtrip",
            OutcomeStatus.ERROR,
            f"provider_meta roundtrip divergence in {mismatches}/{len(rows)} message(s)",
        )
    return DbInvariantResult("provider_meta_roundtrip", OutcomeStatus.OK)


def check_schema_consistency(db_path: str | Path) -> DbInvariantResult:
    from polylogue.storage.sqlite.connection import open_connection

    _allow_empty = frozenset({"action_events_fts", "_litestream_seq", "_litestream_lock"})
    try:
        with open_connection(str(db_path)) as conn:
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_litestream%'"
            ).fetchall()
    except Exception as e:
        return DbInvariantResult("schema_consistency", OutcomeStatus.ERROR, f"query failed: {e}")
    empty_tables: list[str] = []
    for (table_name,) in table_rows:
        if table_name in _allow_empty:
            continue
        try:
            with open_connection(str(db_path)) as conn:
                count = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
            if count == 0:
                empty_tables.append(table_name)
        except Exception:
            pass
    if empty_tables:
        return DbInvariantResult(
            "schema_consistency", OutcomeStatus.ERROR, f"empty tables not in allowlist: {', '.join(empty_tables)}"
        )
    return DbInvariantResult("schema_consistency", OutcomeStatus.OK)


db_semantic_invariant_checks: dict[str, Callable[[str | Path], DbInvariantResult]] = {
    "fts_integrity": check_fts_integrity,
    "provider_meta_roundtrip": check_provider_meta_roundtrip,
    "schema_consistency": check_schema_consistency,
}


def run_db_invariants(db_path: str | Path) -> list[DbInvariantResult]:
    results: list[DbInvariantResult] = []
    for name, check_fn in db_semantic_invariant_checks.items():
        try:
            result = check_fn(db_path)
        except Exception as e:
            result = DbInvariantResult(name, OutcomeStatus.ERROR, f"invariant check crashed: {e}")
        results.append(result)
    return results


def format_db_invariant_summary(results: list[DbInvariantResult]) -> str:
    passed = sum(1 for r in results if r.status is OutcomeStatus.OK)
    failed = sum(1 for r in results if r.status is OutcomeStatus.ERROR)
    skipped = sum(1 for r in results if r.status is OutcomeStatus.SKIP)
    lines = [f"DB Invariant Checks: {passed} pass, {failed} fail, {skipped} skip"]
    failures = [r for r in results if r.status is OutcomeStatus.ERROR]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for f in failures:
            lines.append(f"  {f.invariant_name}: {f.error}")
    return "\n".join(lines)


__all__ = [
    "DbInvariantResult",
    "Invariant",
    "InvariantResult",
    "SHOWCASE_INVARIANTS",
    "check_fts_integrity",
    "check_invariants",
    "check_provider_meta_roundtrip",
    "check_schema_consistency",
    "db_semantic_invariant_checks",
    "format_db_invariant_summary",
    "format_invariant_summary",
    "run_db_invariants",
]
