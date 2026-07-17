from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import record_query_run
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_query_run_stores_only_bounded_result_relation_metadata() -> None:
    conn = sqlite3.connect(":memory:")
    initialize_archive_tier(conn, ArchiveTier.OPS)
    record_query_run(
        conn,
        run_id="qr_01JTEST",
        query_hash="a" * 64,
        actor="agent:codex",
        surface="mcp",
        verb="search",
        request={"query": "hello"},
        lowered_spec={"unit": "session"},
        archive_epoch="index:1",
        started_at_ms=1,
        duration_ms=2,
        status="ok",
        degraded=None,
        unit="session",
        member_count=2,
        exactness="exact",
        result_fingerprint="b" * 64,
        sample_refs=("session:a", "session:b"),
    )

    row = conn.execute("SELECT run_id, sample_refs_json FROM query_runs").fetchone()
    assert row == ("qr_01JTEST", '["session:a","session:b"]')
    with pytest.raises(ValueError, match="capped"):
        record_query_run(
            conn,
            run_id="qr_overflow",
            query_hash="c" * 64,
            actor=None,
            surface="api",
            verb=None,
            request=None,
            lowered_spec=None,
            archive_epoch=None,
            started_at_ms=1,
            duration_ms=0,
            status="ok",
            degraded=None,
            unit=None,
            member_count=0,
            exactness="exact",
            result_fingerprint=None,
            sample_refs=tuple(f"session:{index}" for index in range(21)),
        )
