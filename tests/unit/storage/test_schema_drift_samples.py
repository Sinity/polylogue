"""Bounded ops.db format-drift sample storage (polylogue-da1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    SCHEMA_DRIFT_SAMPLE_RETENTION_MS,
    SCHEMA_DRIFT_SAMPLE_ROW_CAP,
    list_schema_drift_samples,
    record_schema_drift_sample,
    summarize_schema_drift_since,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _ops_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path / "ops.db"))
    initialize_archive_tier(conn, ArchiveTier.OPS)
    return conn


class TestSchemaDriftSamples:
    def test_record_and_list_round_trip(self, tmp_path: Path) -> None:
        conn = _ops_conn(tmp_path)
        try:
            record_schema_drift_sample(
                conn,
                origin="claude-code-session",
                element_kind="session_record",
                classification="new_field",
                unseen_key_signature="metadata.newThing",
                native_id_example="raw-1",
                raw_id="raw-1",
                observed_at_ms=1_000,
            )
            record_schema_drift_sample(
                conn,
                origin="claude-code-session",
                element_kind="session_record",
                classification="field_changed",
                unseen_key_signature="",
                native_id_example="raw-2",
                raw_id="raw-2",
                observed_at_ms=2_000,
            )
            samples = list_schema_drift_samples(conn, origin="claude-code-session")
            assert len(samples) == 2
            newest = samples[0]
            assert newest.observed_at_ms == 2_000
            assert newest.classification == "field_changed"
            assert newest.raw_id == "raw-2"
        finally:
            conn.close()

    def test_retention_prunes_old_samples(self, tmp_path: Path) -> None:
        conn = _ops_conn(tmp_path)
        try:
            record_schema_drift_sample(
                conn,
                origin="codex-session",
                element_kind="session_record",
                classification="new_field",
                unseen_key_signature="x",
                native_id_example="raw-old",
                raw_id="raw-old",
                observed_at_ms=0,
            )
            # A sample recorded well beyond the retention window must prune
            # the ancient row on the next write.
            record_schema_drift_sample(
                conn,
                origin="codex-session",
                element_kind="session_record",
                classification="new_field",
                unseen_key_signature="y",
                native_id_example="raw-new",
                raw_id="raw-new",
                observed_at_ms=SCHEMA_DRIFT_SAMPLE_RETENTION_MS * 2,
            )
            samples = list_schema_drift_samples(conn, origin="codex-session", limit=100)
            assert len(samples) == 1
            assert samples[0].raw_id == "raw-new"
        finally:
            conn.close()

    def test_row_cap_prunes_oldest_first(self, tmp_path: Path) -> None:
        conn = _ops_conn(tmp_path)
        try:
            for i in range(SCHEMA_DRIFT_SAMPLE_ROW_CAP + 5):
                record_schema_drift_sample(
                    conn,
                    origin="codex-session",
                    element_kind="session_record",
                    classification="new_field",
                    unseen_key_signature="x",
                    native_id_example=f"raw-{i}",
                    raw_id=f"raw-{i}",
                    observed_at_ms=i,
                )
            total = conn.execute("SELECT COUNT(*) FROM schema_drift_samples").fetchone()[0]
            assert total <= SCHEMA_DRIFT_SAMPLE_ROW_CAP
            # The oldest rows (lowest observed_at_ms) are the ones pruned.
            remaining_min = conn.execute("SELECT MIN(observed_at_ms) FROM schema_drift_samples").fetchone()[0]
            assert remaining_min >= 5
        finally:
            conn.close()

    def test_summarize_since_windows_and_classifies_risky_vs_benign(self, tmp_path: Path) -> None:
        conn = _ops_conn(tmp_path)
        try:
            # Two benign, one risky, all within the window.
            for i, (classification, raw_id) in enumerate(
                [
                    ("new_field", "raw-benign-1"),
                    ("new_field", "raw-benign-2"),
                    ("field_changed", "raw-risky-1"),
                ]
            ):
                record_schema_drift_sample(
                    conn,
                    origin="claude-code-session",
                    element_kind="session_record",
                    classification=classification,
                    unseen_key_signature="",
                    native_id_example=raw_id,
                    raw_id=raw_id,
                    observed_at_ms=10_000 + i,
                )
            # One sample well before the window -- must not count.
            record_schema_drift_sample(
                conn,
                origin="claude-code-session",
                element_kind="session_record",
                classification="field_changed",
                unseen_key_signature="",
                native_id_example="raw-stale",
                raw_id="raw-stale",
                observed_at_ms=1,
            )
            summaries = summarize_schema_drift_since(conn, since_ms=10_000)
            assert len(summaries) == 1
            summary = summaries[0]
            assert summary.origin == "claude-code-session"
            assert summary.total == 3
            assert summary.risky == 1
            assert summary.benign == 2
            assert summary.risky_rate == 1 / 3
            # Risky examples surface before benign ones.
            assert summary.example_native_ids[0] == "raw-risky-1"
        finally:
            conn.close()

    def test_summarize_since_omits_origins_with_no_samples_in_window(self, tmp_path: Path) -> None:
        conn = _ops_conn(tmp_path)
        try:
            record_schema_drift_sample(
                conn,
                origin="chatgpt-export",
                element_kind="conversation",
                classification="new_field",
                unseen_key_signature="",
                native_id_example="raw-1",
                raw_id="raw-1",
                observed_at_ms=1,
            )
            summaries = summarize_schema_drift_since(conn, since_ms=1_000_000)
            assert summaries == ()
        finally:
            conn.close()

    def test_drift_sample_writer_rejects_unknown_origin(self, tmp_path: Path) -> None:
        conn = _ops_conn(tmp_path)
        try:
            try:
                record_schema_drift_sample(
                    conn,
                    origin="not-a-real-origin",
                    element_kind="session_record",
                    classification="new_field",
                    unseen_key_signature="",
                    native_id_example="raw-1",
                    raw_id="raw-1",
                    observed_at_ms=1,
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise AssertionError("expected a CHECK constraint violation for an unrecognized origin token")
        finally:
            conn.close()
