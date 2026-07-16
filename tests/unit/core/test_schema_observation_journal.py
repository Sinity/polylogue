"""Behavioral contracts for the schema ObservationJournal lifecycle."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from polylogue.schemas.generation.observation_journal import (
    ObservationJournal,
    recover_stale_journals,
)
from polylogue.schemas.observation import SchemaUnit


def _unit() -> SchemaUnit:
    return SchemaUnit(
        cluster_payload={"type": "message", "content": {"kind": "tool"}},
        schema_samples=[{"type": "message"}, {"type": "tool_result", "exit_code": 0}],
        artifact_kind="session_record_stream",
        session_id="session-1",
        raw_id="raw-1",
        source_path="/private/source.jsonl",
        bundle_scope="scope-1",
        observed_at="2026-07-16T12:00:00+00:00",
        exact_structure_id="structure-1",
        profile_tokens=("field:type", "record:message"),
    )


def test_journal_roundtrips_units_samples_and_terminal_evidence(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    with ObservationJournal.create(root=root) as journal:
        journal.append_unit(_unit())
        journal.record_terminal(
            raw_id="raw-1",
            status="included",
            artifact_kind="session_record_stream",
            source_path="/private/source.jsonl",
            reason=None,
        )
        journal.record_terminal(
            raw_id="raw-quarantined",
            status="quarantined",
            artifact_kind="sqlite_evidence",
            source_path="/private/evidence.db",
            reason="artifact taxonomy excludes provider-wire decoding",
        )

        assert list(journal.iter_units()) == [_unit()]
        terminals = list(journal.iter_terminals())
        assert [(item.raw_id, item.status) for item in terminals] == [
            ("raw-1", "included"),
            ("raw-quarantined", "quarantined"),
        ]
        assert journal.unit_count == 1
        assert journal.sample_count == 2
        assert journal.terminal_counts() == {"included": 1, "quarantined": 1}

        journal_path = journal.path
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600

    assert not journal_path.exists()
    assert not journal_path.with_name(f"{journal_path.name}-wal").exists()
    assert not journal_path.with_name(f"{journal_path.name}-shm").exists()


def test_journal_cleanup_runs_for_exceptions(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    journal_path: Path | None = None

    with pytest.raises(RuntimeError, match="cancelled"):
        with ObservationJournal.create(root=root) as journal:
            journal_path = journal.path
            journal.append_unit(_unit())
            raise RuntimeError("cancelled")

    assert journal_path is not None
    assert not journal_path.exists()


def test_journal_replays_assignments_without_retaining_cluster_payloads(tmp_path: Path) -> None:
    first = _unit()
    second = SchemaUnit(
        cluster_payload={"large": ["private clustering content"] * 10},
        schema_samples=[{"type": "assistant"}],
        artifact_kind="session_record_stream",
        session_id="session-2",
        exact_structure_id="structure-2",
        profile_tokens=("field:type", "record:assistant"),
    )
    with ObservationJournal.create(root=tmp_path / "journals") as journal:
        first_id = journal.append_unit(first, retain_cluster_payload=False)
        second_id = journal.append_unit(second, retain_cluster_payload=False)
        journal.assign_profile_family(first_id, "family-a")
        journal.assign_profile_family(second_id, "family-b")

        # Simultaneous normalization must not cascade family-a through family-b.
        journal.normalize_profile_families({"family-a": "family-b", "family-b": "family-c"})
        journal.assign_package_family(first_id, "package-1")

        identified = list(journal.iter_identified_memberships())
        assert [membership.profile_family_id for _unit_id, membership in identified] == [
            "family-b",
            "family-c",
        ]
        assert all(membership.unit.cluster_payload == {} for _unit_id, membership in identified)
        package = journal.memberships(package_family_id="package-1")
        assert len(package) == 1
        assert package[0].unit.schema_samples == first.schema_samples


def test_journal_rejects_archive_or_broad_permission_roots(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    with pytest.raises(ValueError, match="forbidden"):
        ObservationJournal.create(root=archive_root / "journal", forbidden_roots=(archive_root,))

    broad_root = tmp_path / "broad"
    broad_root.mkdir(mode=0o777)
    broad_root.chmod(0o777)
    with pytest.raises(PermissionError, match="permissions"):
        ObservationJournal.create(root=broad_root, tighten_permissions=False)


def test_stale_recovery_removes_dead_owner_and_preserves_live_owner(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    code = """
import os
from pathlib import Path
from polylogue.schemas.generation.observation_journal import ObservationJournal
journal = ObservationJournal.create(root=Path(os.environ["JOURNAL_ROOT"]))
os.write(1, str(journal.path).encode("utf-8"))
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=os.environ | {"JOURNAL_ROOT": str(root)},
    )
    stale_path = Path(result.stdout.decode("utf-8"))
    os.utime(stale_path, (100.0, 100.0))

    live = ObservationJournal.create(root=root)
    live_path = live.path
    os.utime(live_path, (100.0, 100.0))

    removed = recover_stale_journals(root, now_s=10_000.0, minimum_age_s=60.0)

    assert removed == [stale_path]
    assert not stale_path.exists()
    assert live_path.exists()
    live.close()
