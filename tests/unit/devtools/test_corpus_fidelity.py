"""Focused tests for the production corpus-fidelity devtools command."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from devtools import click_dispatch
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from tests.infra.workload_artifacts import SeededArchiveArtifact, clone_seeded_archive


def _archive_tier_bytes(root: Path) -> dict[str, bytes]:
    """Capture exact bytes for every registered tier file present in a fixture."""
    snapshots: dict[str, bytes] = {}
    for spec in ARCHIVE_TIER_SPECS.values():
        path = root / spec.filename
        if path.is_file():
            snapshots[spec.filename] = path.read_bytes()
    assert snapshots, f"expected at least one archive tier under {root}"
    return snapshots


def _run_registered_route(root: Path, *, json_output: bool) -> int:
    argv = ["verify", "corpus-fidelity", "--archive-root", str(root)]
    if json_output:
        argv.append("--json")
    return click_dispatch.main(argv)


@pytest.mark.parametrize("json_output", (False, True), ids=("plain", "json"))
def test_registered_route_preserves_all_existing_archive_tiers(
    corpus_fidelity_archive: SeededArchiveArtifact,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    """Run the real registered route and prove it is read-only across tiers.

    Anti-vacuity: mutating the command to write ``user.db``, ``ops.db``, or
    ``embeddings.db`` would fail the exact-byte assertion below. The previous
    source/index-only snapshot left those unchecked-tier mutations green.
    """
    before = _archive_tier_bytes(corpus_fidelity_archive.root)

    exit_code = _run_registered_route(corpus_fidelity_archive.root, json_output=json_output)

    assert exit_code == 0
    output = capsys.readouterr().out
    if json_output:
        payload = json.loads(output)
        assert payload["verdict"] == "PASS"
        assert payload["blocking"] is False
    else:
        assert "Corpus fidelity:" in output
        assert "[OK] corpus-absences:" in output
        assert "[OK] corpus-attachment-fidelity:" in output
        assert "[OK] corpus-revision-fidelity:" in output
        assert "clear" in output
    assert _archive_tier_bytes(corpus_fidelity_archive.root) == before


def _clone(artifact: SeededArchiveArtifact, destination: Path) -> Path:
    return clone_seeded_archive(artifact, destination).root


@pytest.mark.parametrize(
    ("violation", "expected_check", "json_output"),
    (
        ("absent", "corpus-absences", False),
        ("absent", "corpus-absences", True),
        ("attachment", "corpus-attachment-fidelity", False),
        ("attachment", "corpus-attachment-fidelity", True),
        ("revision", "corpus-revision-fidelity", False),
        ("revision", "corpus-revision-fidelity", True),
    ),
)
def test_command_blocks_real_seeded_archive_fidelity_violations(
    corpus_fidelity_archive: SeededArchiveArtifact,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
    expected_check: str,
    json_output: bool,
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

    before = _archive_tier_bytes(root)
    exit_code = _run_registered_route(root, json_output=json_output)
    assert exit_code == 1
    output = capsys.readouterr().out
    if json_output:
        payload = json.loads(output)
        assert payload["verdict"] == "FAIL"
        assert payload["blocking"] is True
        assert {check["name"]: check["status"] for check in payload["checks"]}[expected_check] == "error"
    else:
        assert f"[FAIL] {expected_check}:" in output
        assert "BLOCKING" in output
    assert _archive_tier_bytes(root) == before
