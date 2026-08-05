from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from devtools.preflight_ledger import build_preflight_ledger
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import add_convergence_debt
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _initialize_all_tiers(root: Path) -> None:
    for tier in ArchiveTier:
        initialize_archive_database(root / f"{tier.value}.db", tier)


def _insert_raws(
    root: Path,
    *,
    origin: str,
    quarantined_count: int,
    missing_census: int,
    missing_quarantine_count: int,
    quarantined_bytes: int,
    missing_census_bytes: int,
) -> None:
    source_db = root / "source.db"
    count = missing_census + quarantined_count - missing_quarantine_count
    missing_quarantined_count = missing_quarantine_count
    regular_count = quarantined_count - missing_quarantined_count
    regular_bytes = quarantined_bytes - missing_census_bytes

    def sizes(total: int, rows: int) -> list[int]:
        if rows == 0:
            assert total == 0
            return []
        quotient, remainder = divmod(total, rows)
        return [quotient + (index < remainder) for index in range(rows)]

    missing_sizes = sizes(missing_census_bytes, missing_quarantined_count) + [0] * (
        missing_census - missing_quarantined_count
    )
    regular_sizes = sizes(regular_bytes, regular_count)
    all_sizes = missing_sizes + regular_sizes
    with sqlite3.connect(source_db) as conn:
        conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'quarantined')
            """,
            [
                (
                    f"{origin}-raw-{index}",
                    origin,
                    f"{origin}-native-{index}",
                    f"/{origin}/{index}.jsonl",
                    (f"{origin}-{index}".encode() + b"x" * 32)[:32],
                    all_sizes[index],
                    index + 1,
                )
                for index in range(count)
            ],
        )
        conn.executemany(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms
            ) VALUES (?, 'test', 'complete', 1, 1)
            """,
            [(f"{origin}-raw-{index}",) for index in range(missing_census, count)],
        )
        conn.execute(
            """
            UPDATE raw_sessions SET revision_authority = 'byte_proven'
            WHERE origin = ?
              AND CAST(substr(raw_id, instr(raw_id, 'raw-') + 4) AS INTEGER) >= ?
              AND CAST(substr(raw_id, instr(raw_id, 'raw-') + 4) AS INTEGER) < ?
            """,
            (origin, missing_quarantined_count, missing_census),
        )
        conn.commit()
    assert sum(missing_sizes[:missing_quarantined_count]) + sum(regular_sizes) == quarantined_bytes
    assert sum(all_sizes[:missing_census]) == missing_census_bytes
    assert regular_count == quarantined_count - missing_quarantined_count
    assert regular_bytes == quarantined_bytes - missing_census_bytes


@pytest.mark.integration
def test_preflight_reports_quarantine_and_missing_census_by_origin(tmp_path: Path) -> None:
    _initialize_all_tiers(tmp_path)
    gib = 1024**3
    _insert_raws(
        tmp_path,
        origin="codex-session",
        quarantined_count=5_203,
        missing_census=922,
        missing_quarantine_count=922,
        quarantined_bytes=round(46.8 * gib),
        missing_census_bytes=round(9.602 * gib),
    )
    _insert_raws(
        tmp_path,
        origin="claude-code-session",
        quarantined_count=3_829,
        missing_census=6_379,
        missing_quarantine_count=922,
        quarantined_bytes=round(7.9 * gib),
        missing_census_bytes=round(2.275 * gib),
    )

    report = build_preflight_ledger(tmp_path, limit=4, now=datetime(2026, 8, 5, tzinfo=UTC))
    source = report["checks"]["source"]
    by_origin = {item["origin"]: item for item in source["by_origin"]}

    assert report["read_only"] is True
    assert report["mutation_operations"] == []
    assert report["state"] == "blocked"
    assert source["state"] == "unknown"
    assert source["totals"]["raw_count"] == 14_489
    assert source["totals"]["quarantined_count"] == 9_032
    assert source["totals"]["missing_census_count"] == 7_301
    assert source["totals"]["quarantined_size"]["bytes"] == round((46.8 + 7.9) * gib)
    assert source["totals"]["missing_census_size"]["bytes"] == round((9.602 + 2.275) * gib)
    assert by_origin["codex-session"]["quarantine"]["count"] == 5_203
    assert by_origin["codex-session"]["quarantine"]["size"]["gib"] == pytest.approx(46.8, abs=0.001)
    assert by_origin["codex-session"]["census_coverage"]["missing_count"] == 922
    assert by_origin["codex-session"]["census_coverage"]["missing_size"]["bytes"] == round(9.602 * gib)
    assert by_origin["codex-session"]["census_coverage"]["missing_size"]["gib"] == pytest.approx(9.602, abs=0.001)
    assert by_origin["claude-code-session"]["quarantine"]["count"] == 3_829
    assert by_origin["claude-code-session"]["quarantine"]["size"]["gib"] == pytest.approx(7.9, abs=0.001)
    assert by_origin["claude-code-session"]["census_coverage"]["missing_count"] == 6_379
    assert by_origin["claude-code-session"]["census_coverage"]["missing_size"]["bytes"] == round(2.275 * gib)
    assert by_origin["claude-code-session"]["census_coverage"]["missing_size"]["gib"] == pytest.approx(2.275, abs=0.001)
    assert source["semantics"]["quarantine"] == "authority_pending, not automatically bad"
    assert (
        source["semantics"]["missing_census"]
        == "coverage_unknown, never terminal or actionable without a census verdict"
    )


def test_preflight_fails_closed_on_missing_census_relation(tmp_path: Path) -> None:
    _initialize_all_tiers(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DROP TABLE raw_membership_census")
        conn.commit()

    report = build_preflight_ledger(tmp_path)

    source = report["checks"]["source"]
    assert source["state"] == "unknown"
    assert report["ok"] is False
    assert "source" in report["blocking_checks"]
    assert "raw_membership_census" in source["reason"]


def test_preflight_exposes_schema_and_convergence_failures_without_writing(tmp_path: Path) -> None:
    _initialize_all_tiers(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("PRAGMA user_version = 24")
        conn.commit()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        for index in range(3):
            add_convergence_debt(
                conn,
                stage="fts",
                target_type="session_id",
                target_id=f"session-{index}",
                status="failed",
                attempts=1,
                created_at_ms=1,
                updated_at_ms=1,
            )
        conn.commit()
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.db")}

    report = build_preflight_ledger(tmp_path)

    after = {path.name: path.read_bytes() for path in tmp_path.glob("*.db")}
    assert report["checks"]["schema"]["state"] == "fail"
    assert "source" in report["checks"]["schema"]["schema_mismatches"]
    assert report["checks"]["convergence_debt"]["failed_count"] == 3
    assert report["checks"]["convergence_debt"]["state"] == "fail"
    assert before == after
