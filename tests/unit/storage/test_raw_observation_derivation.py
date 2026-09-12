"""Production raw-adapter laws independent of the legacy backlog census."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, converge
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _admit(root: Path, names: tuple[str, ...], *, path: str = "bundle.json") -> str:
    payload = [
        {
            "id": name,
            "title": name,
            "create_time": 1,
            "current_node": "m",
            "mapping": {
                "m": {
                    "id": "m",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": [name]},
                    },
                },
            },
        }
        for name in names
    ]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path=path,
            acquired_at_ms=1,
        )


def _run(root: Path) -> DerivationReport:
    return converge(DerivationRegistry((RawObservationDerivation(root),)), raw_observation_frame(root))


def _snapshot(root: Path) -> tuple[tuple[tuple[object, ...], ...], ...]:
    with sqlite3.connect(root / "source.db") as conn:
        return tuple(
            tuple(conn.execute(f"SELECT * FROM {table} ORDER BY raw_id"))
            for table in (
                "raw_sessions",
                "raw_session_memberships",
                "raw_membership_census",
                "raw_authority_parser_census",
            )
        )


def test_split_member_loss_is_recovered_by_kernel_without_legacy_scanner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: a raw-id/session-exists probe misses the lost split member."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("split-a", "split-b"))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy raw candidate scanner was called")

    monkeypatch.setattr("polylogue.storage.raw_convergence._raw_materialization_candidate_ids", forbidden)
    first = _run(tmp_path)
    assert first.done == 1 and first.failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE native_id = 'split-b'")
        conn.commit()
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "missing"
    restored = _run(tmp_path)
    assert restored.done == 1 and restored.failed == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
            ("split-a",),
            ("split-b",),
        ]
    before = _snapshot(tmp_path)
    unchanged = _run(tmp_path)
    assert unchanged.work.computed == unchanged.work.published == 0
    assert _snapshot(tmp_path) == before


def test_restart_without_ops_hints_recovers_index_loss_and_new_admission(tmp_path: Path) -> None:
    """Anti-vacuity: retained pending caches hide reset or later admissions."""
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, ("first",))
    assert _run(tmp_path).done == 1
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("DELETE FROM convergence_debt")
        conn.commit()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions")
        conn.commit()
    _admit(tmp_path, ("second",), path="second.json")
    restarted = _run(tmp_path)
    assert restarted.done == 2 and restarted.failed == 0
    assert _run(tmp_path).wrote_nothing


@pytest.mark.parametrize("mutation", ["descriptor", "blob", "generation"])
def test_publish_rejects_changed_source_or_generation(tmp_path: Path, mutation: str) -> None:
    """Anti-vacuity: bypassing publish revalidation materializes stale inputs."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("bound",))
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    prepared = adapter.compute(frame, raw_id)
    if mutation == "descriptor":
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("UPDATE raw_sessions SET source_path = 'changed.json' WHERE raw_id = ?", (raw_id,))
            conn.commit()
    elif mutation == "blob":
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            _provider, blob_hash, _path, _kind, _size = archive.raw_revision_descriptor(raw_id)
            blob_path = archive.blob_path_for_hash(blob_hash)
            assert blob_path is not None
            blob_path.write_bytes(b"changed")
    else:
        from dataclasses import replace

        frame = replace(frame, source_revision=str(tmp_path / "another-index.db"))
    assert adapter.publish(frame, prepared) is False
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_poison_observation_does_not_suppress_healthy_sibling(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _admit(tmp_path, ("healthy",), path="healthy.json")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=b"not json",
            source_path="poison.json",
            acquired_at_ms=1,
        )
    report = _run(tmp_path)
    assert report.done == 1 and report.failed == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("healthy",)]


def test_zero_output_requires_parser_evidence(tmp_path: Path) -> None:
    """Anti-vacuity: a bare empty index cannot certify a zero-output raw."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ())
    assert _run(tmp_path).done == 1
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"
    assert _run(tmp_path).work.published == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_membership_census WHERE raw_id = ?", (raw_id,))
        conn.execute("DELETE FROM raw_artifacts WHERE raw_id = ?", (raw_id,))
        conn.commit()
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"


def test_inspection_rejects_stale_parser_recipe_and_excess_identity(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("expected",))
    assert _run(tmp_path).done == 1
    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET parser_fingerprint = 'stale'")
        conn.commit()
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET native_id = 'excess'")
        conn.commit()
    assert adapter.inspect(frame, (raw_id,))[raw_id] == "excess"


def test_existing_head_does_not_certify_missing_observation_application(tmp_path: Path) -> None:
    """Anti-vacuity: a valid sibling head cannot replace this raw's decision."""
    bootstrap_archive_root(tmp_path)
    raw_id = _admit(tmp_path, ("application",))
    assert _run(tmp_path).done == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM raw_revision_applications WHERE raw_id = ?", (raw_id,))
        conn.commit()
    adapter = RawObservationDerivation(tmp_path)
    assert adapter.inspect(raw_observation_frame(tmp_path), (raw_id,))[raw_id] == "missing"
    assert _run(tmp_path).done == 1


def test_discovery_budget_bounds_raw_enumeration(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    for index in range(5):
        _admit(tmp_path, (f"session-{index}",), path=f"{index}.json")
    report = converge(
        DerivationRegistry((RawObservationDerivation(tmp_path),)),
        raw_observation_frame(tmp_path),
        budget=Budget(page=2, discovery=2, inspection=2, compute=2),
    )
    assert report.work.discovered == 2
    assert report.work.computed == report.work.published == 0
    assert not report.cursor.position("raw_observation").swept
