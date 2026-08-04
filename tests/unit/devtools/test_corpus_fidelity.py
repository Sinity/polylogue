"""Focused tests for the production corpus-fidelity devtools command."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from devtools import corpus_fidelity
from tests.infra.workload_artifacts import SeededArchiveArtifact, clone_seeded_archive


def test_command_runs_registered_gate_against_real_seeded_archive(
    corpus_fidelity_archive: SeededArchiveArtifact,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = corpus_fidelity.main(["--archive-root", str(corpus_fidelity_archive.root)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Corpus fidelity:" in output
    assert "[OK] corpus-absences:" in output
    assert "[OK] corpus-attachment-fidelity:" in output
    assert "[OK] corpus-revision-fidelity:" in output
    assert "clear" in output


def _clone(artifact: SeededArchiveArtifact, destination: Path) -> Path:
    return clone_seeded_archive(artifact, destination).root


@pytest.mark.parametrize(
    ("violation", "expected_check"),
    (
        ("absent", "corpus-absences"),
        ("attachment", "corpus-attachment-fidelity"),
        ("revision", "corpus-revision-fidelity"),
    ),
)
def test_command_blocks_real_seeded_archive_fidelity_violations(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    violation: str,
    expected_check: str,
) -> None:
    """The command drives the production registry over a writable clone of
    the real seeded SQLite archive. Removing its ``verify_archive`` call or
    any selected checker leaves the corresponding archive mutation green.
    """
    root = _clone(corpus_fidelity_archive, tmp_path / violation)
    with sqlite3.connect(root / "index.db") as index:
        session_row = index.execute("SELECT session_id FROM sessions ORDER BY session_id LIMIT 1").fetchone()
        message_row = index.execute(
            "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position LIMIT 1",
            session_row,
        ).fetchone()
        assert session_row is not None
        assert message_row is not None
        session_id = str(session_row[0])
        if violation == "attachment":
            index.execute(
                "INSERT INTO attachments(attachment_id, acquisition_status) VALUES ('fixture-unfetched', 'unfetched')"
            )
            index.execute(
                "INSERT INTO attachment_refs(attachment_id, session_id, message_id, position, upload_origin) "
                "VALUES ('fixture-unfetched', ?, ?, 99, 'drive')",
                (session_id, message_row[0]),
            )
    if violation in {"absent", "revision"}:
        origin, native_id = session_id.split(":", 1)
        raw_id = f"fixture-{violation}"
        provider_session_id = native_id if violation == "revision" else "absent"
        with sqlite3.connect(root / "source.db") as source:
            source.execute(
                "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, "
                "acquired_at_ms, logical_source_key) VALUES (?, ?, ?, ?, ?, 10, 100, ?)",
                (
                    raw_id,
                    origin,
                    provider_session_id,
                    f"/fixture/{raw_id}",
                    hashlib.sha256(raw_id.encode()).digest(),
                    f"fixture:{provider_session_id}",
                ),
            )
            source.execute(
                "INSERT INTO raw_session_memberships(raw_id, logical_source_key, provider_session_id, "
                "source_revision, normalized_content_hash, message_count) VALUES (?, ?, ?, 'fixture-revision', ?, ?)",
                (
                    raw_id,
                    f"fixture:{provider_session_id}",
                    provider_session_id,
                    b"f" * 32,
                    100_000 if violation == "revision" else 4,
                ),
            )

    exit_code = corpus_fidelity.main(["--archive-root", str(root), "--json"])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking"] is True
    assert {check["name"]: check["status"] for check in payload["checks"]}[expected_check] == "error"
