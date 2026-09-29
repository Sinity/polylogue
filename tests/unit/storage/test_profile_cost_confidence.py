"""Profile cost confidence reads the evidence the canonical writer stores."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.queries.mappers_insight_profiles import _row_to_session_profile_record
from tests.infra.session_profiles import write_session_profile


@pytest.mark.parametrize(("estimated", "provenance"), [(True, "unknown"), (False, "provider_reported")])
def test_stored_evidence_decides_cost_confidence(tmp_path: Path, estimated: bool, provenance: str) -> None:
    """Ignoring the stored evidence changes reported charges into estimates."""
    path = tmp_path / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('cost', 'codex-session', ?)", (bytes(32),)
        )
        write_session_profile(
            conn, "codex-session:cost", evidence={"cost_is_estimated": estimated, "cost_provenance": provenance}
        )
        row = conn.execute("SELECT * FROM session_profiles WHERE session_id = 'codex-session:cost'").fetchone()
        record = _row_to_session_profile_record(row)
    assert record.cost_is_estimated is estimated
    assert record.cost_provenance == provenance
