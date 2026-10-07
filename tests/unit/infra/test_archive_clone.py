"""Synthetic archive population owns destination trains and exact retained evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Generator
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite import durable_change_train, migration_runner
from polylogue.storage.sqlite.archive_population import ArchivePopulationError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import _template_key, clone_archive_template, finalize_archive_template
from tests.infra.durable_tier_fixtures import ship_synthetic_source_train
from tests.infra.workload_artifacts import ImmutableTreeArtifact


def _populated_template(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    # A released train gives the template history for population to detach.
    ship_synthetic_source_train(root.parent / "train-package", monkeypatch)
    initialize_active_archive_root(root)
    payload = b'{"synthetic_record":"retained"}\n'
    BlobStore(root / "blob").write_from_bytes(payload)
    with closing(sqlite3.connect(root / "source.db")) as conn:
        return write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            capture_mode=Provider.CODEX,
            source_path="/synthetic/exact",
            canonical_source_path="/synthetic/exact",
            native_id="native\x00suffix",
            source_index=0,
            payload=payload,
            acquired_at_ms=2,
        )


def test_populated_template_clone_keeps_rows_blobs_and_original_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    raw_id = _populated_template(source, monkeypatch)
    original_train = source / ".maintenance-state/durable-change-trains/source-002.json"
    original_bytes = original_train.read_bytes()
    finalize_archive_template(source)
    source_blobs = {
        str(path.relative_to(source)): path.read_bytes() for path in (source / "blob").rglob("*") if path.is_file()
    }
    clone_archive_template(source, destination)
    with (
        closing(sqlite3.connect(source / "source.db")) as original,
        closing(sqlite3.connect(destination / "source.db")) as clone,
    ):
        assert migration_runner._durable_literal_rows_digest(original) == migration_runner._durable_literal_rows_digest(
            clone
        )
        assert clone.execute("SELECT raw_id, native_id FROM raw_sessions").fetchall() == [(raw_id, "native\x00suffix")]
    assert original_train.read_bytes() == original_bytes
    assert (destination / ".maintenance-state/durable-change-trains/source-002.json").read_bytes() != original_bytes
    source_manifest_id = ImmutableTreeArtifact.adopt(source, key=_template_key(source)).manifest_id
    source_namespace = hashlib.sha256(source_manifest_id.encode()).hexdigest()
    provenance = destination / ".archive-population-provenance" / source_namespace / "source.json"
    provenance_record = json.loads(provenance.read_text())
    assert provenance_record["source_manifest_id"] == source_manifest_id
    assert provenance_record["owning_artifact"] is None
    assert (provenance.parent / "original-history/source-002.json").read_bytes() == original_bytes
    assert {
        str(path.relative_to(destination)): path.read_bytes()
        for path in (destination / "blob").rglob("*")
        if path.is_file()
    } == source_blobs
    with ArchiveStore(destination):
        pass
    second = tmp_path / "second"
    finalize_archive_template(destination)
    clone_archive_template(destination, second)
    with ArchiveStore(second):
        pass


def test_clone_validates_source_release_before_any_source_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _populated_template(source, monkeypatch)
    finalize_archive_template(source)
    verified = False
    verify = durable_change_train._verify_released_train_live_tier
    connect = sqlite3.connect
    backed_up: list[str] = []

    def verified_release(*args: Any, **kwargs: Any) -> object:
        nonlocal verified
        result = verify(*args, **kwargs)
        if args[0].execute("PRAGMA database_list").fetchone()[2] == str(source / "source.db"):
            verified = True
        return result

    # A finalized template is a relocated (detached) source: population admits
    # it against each released train's immutable historical proof instead of
    # verifying it as the live archive. Either check must precede the backup.
    historical = durable_change_train._historical_schema_evidence

    def verified_history(train: Any) -> object:
        nonlocal verified
        result = historical(train)
        if train.tier is ArchiveTier.SOURCE:
            verified = True
        return result

    from polylogue.storage.io_phase_metrics import _MeasuredConnection

    # Production opens tiers through its measured connection; the observer
    # extends that class rather than replacing it with a bare connection.
    class ObservedConnection(_MeasuredConnection):
        def backup(self, target: sqlite3.Connection, **kwargs: Any) -> None:
            path = self.execute("PRAGMA database_list").fetchone()[2]
            if path == str(source / "source.db"):
                assert verified
                backed_up.append(path)
            super().backup(target, **kwargs)

    def tracked_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return cast(sqlite3.Connection, connect(*args, **(kwargs | {"factory": ObservedConnection})))

    monkeypatch.setattr(durable_change_train, "_verify_released_train_live_tier", verified_release)
    monkeypatch.setattr(durable_change_train, "_historical_schema_evidence", verified_history)
    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    clone_archive_template(source, tmp_path / "destination")
    assert backed_up == [str(source / "source.db")]


@pytest.mark.parametrize("kind", ("schema", "pending"))
def test_clone_refuses_unreleased_or_custom_archive_without_population(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _populated_template(source, monkeypatch)
    if kind == "schema":
        with closing(sqlite3.connect(source / "source.db")) as conn:
            conn.execute("CREATE TABLE custom_unproved (value TEXT)")
            conn.commit()
    else:
        path = source / ".maintenance-state/durable-change-trains/source-002.json"
        train = durable_change_train.load_durable_change_train_manifest(path)
        assert train.proof is not None
        pending = replace(
            train,
            state=migration_runner.DurableChangeTrainState.PROVEN,
            released_at_ms=None,
            release_evidence_ref=None,
            proof_refs=train.proof.proof_refs,
        )
        path.write_text(json.dumps(migration_runner.durable_change_train_to_payload(pending)))
    finalize_archive_template(source)
    destination = tmp_path / "destination"
    with pytest.raises(ArchivePopulationError) as refusal:
        clone_archive_template(source, destination)
    assert refusal.value.code == ("unsupported_source_schema" if kind == "schema" else "unreleased_source_history")
    assert (destination / ".archive-population.pending").is_file()
    from polylogue.storage.sqlite.population_admission import ArchivePopulationPendingError

    with pytest.raises(ArchivePopulationPendingError):
        ArchiveStore.open_existing(destination)


@pytest.mark.parametrize("interrupt", (False, True))
def test_fixture_population_fences_concurrent_reader_and_retains_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    import threading

    from polylogue.storage.sqlite import archive_population
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    source = tmp_path / "source"
    _populated_template(source, monkeypatch)
    finalize_archive_template(source)
    destination = tmp_path / "destination"
    actual = archive_population._populate_authenticated_archive
    targets: list[Path] = []
    observed: list[str] = []
    failures: list[BaseException] = []

    def populate(original: Path, target: Path, **kwargs: Any) -> Any:
        proof = actual(original, target, **kwargs)
        assert target == destination
        targets.append(target)

        def reader() -> None:
            try:
                with pytest.raises(ArchivePopulationPendingError):
                    ArchiveStore.open_existing(target)
                observed.append("pending")
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=reader)
        thread.start()
        thread.join()
        if failures:
            raise failures[0]
        if interrupt:
            raise KeyboardInterrupt
        return proof

    monkeypatch.setattr(archive_population, "_populate_authenticated_archive", populate)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            clone_archive_template(source, destination)
        assert len(targets) == 1
        assert (targets[0] / POPULATION_PENDING).is_file()
        with pytest.raises(ArchivePopulationPendingError):
            initialize_active_archive_root(targets[0])
    else:
        clone_archive_template(source, destination)
        assert not (destination / POPULATION_PENDING).exists()
        with ArchiveStore.open_existing(destination):
            pass
    assert observed == ["pending"]


def test_workspace_fixture_teardown_retains_failed_population_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError
    from tests import conftest

    create_fixture = cast(
        Callable[[Path, pytest.MonkeyPatch], Generator[dict[str, Path], None, None]],
        vars(conftest.workspace_paths)["__wrapped__"],
    )
    fixture = create_fixture(tmp_path, monkeypatch)
    paths = next(fixture)
    root = paths["archive_root"]
    root.mkdir()
    marker = root / POPULATION_PENDING
    marker.write_text('{"fixture":"failed-population"}')
    retained = root / "partial-evidence"
    retained.write_bytes(b"retained synthetic evidence")
    fixture.close()
    assert marker.is_file()
    assert retained.read_bytes() == b"retained synthetic evidence"
    with pytest.raises(ArchivePopulationPendingError):
        ArchiveStore.open_existing(root)


@pytest.mark.parametrize("interrupt", (False, True))
def test_fixture_copy_fences_the_actual_destination_before_the_first_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    import subprocess
    import threading

    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    source = tmp_path / "source"
    _populated_template(source, monkeypatch)
    finalize_archive_template(source)
    destination = tmp_path / "destination"
    actual = subprocess.run
    observed: list[str] = []
    failures: list[BaseException] = []

    def copy(argv: list[str], **kwargs: Any) -> Any:
        if argv[:3] == ["cp", "-a", "--reflink=always"]:
            assert Path(argv[-1]) == destination
            assert (destination / POPULATION_PENDING).is_file()
            assert str(source / ".archive-ownership.lock") not in argv
            assert str(source / "daemon.pid") not in argv

            def reader() -> None:
                try:
                    with pytest.raises(ArchivePopulationPendingError):
                        ArchiveStore.open_existing(destination)
                    with pytest.raises(ArchivePopulationPendingError):
                        initialize_active_archive_root(destination)
                    observed.append("pending-before-copy")
                except BaseException as exc:
                    failures.append(exc)

            thread = threading.Thread(target=reader)
            thread.start()
            thread.join()
            if failures:
                raise failures[0]
            if interrupt:
                raise KeyboardInterrupt
        return actual(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", copy)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            clone_archive_template(source, destination)
        assert (destination / POPULATION_PENDING).is_file()
        with pytest.raises(ArchivePopulationPendingError):
            ArchiveStore.open_existing(destination)
    else:
        clone_archive_template(source, destination)
        assert not (destination / POPULATION_PENDING).exists()
        with ArchiveStore.open_existing(destination):
            pass
    assert observed == ["pending-before-copy"]
