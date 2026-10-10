"""Hygiene helpers for fixtures that author a durable tier by hand.

Two properties of a real archive make a hand-authored durable fixture lie
unless the fixture restates them, and both produce a green-looking or
mysteriously-empty test rather than an obvious error:

* Durable tiers run in WAL mode. Backup evidence readers open a tier with
  ``immutable=1``, which is correct for the checkpointed copy production hands
  them and which cannot see a ``-wal`` file at all. A fixture that seeds the
  *live* tier and then reads it as backup evidence measures an empty database.
* The archive format marker binds each durable tier to its schema
  fingerprint. Editing that schema on disk invalidates the marker exactly the
  way a transplanted historical tier would, so a fixture that deliberately
  authors the tier has to restate the evidence or every later refusal is a
  fixture artifact instead of the behaviour under test.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

__all__ = [
    "SYNTHETIC_SOURCE_TRAIN_TABLE",
    "checkpoint_durable_tier",
    "ship_synthetic_source_train",
    "rebind_archive_format_fingerprints",
    "refresh_archive_format_marker",
    "seed_durable_tier",
    "bootstrap_baseline_archive",
    "initialize_runtime_source_fixture",
]


def initialize_runtime_source_fixture(path: Path) -> None:
    """Build an empty unit schema through the installed numbered probe authority.

    This fixture is not an archive migration. Tests for populated historical
    archives use the actual owned train and authenticated package instead.
    """
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_archive_database,
        initialize_runtime_tier_probe,
    )
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as source:
        current = int(source.execute("PRAGMA user_version").fetchone()[0])
        if current == 0:
            initialize_runtime_tier_probe(source, ArchiveTier.SOURCE, probe_path=path)
        elif current != ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]:
            raise ValueError("historical Source fixture requires its owned migration train")
    initialize_archive_database(path, ArchiveTier.SOURCE)


def bootstrap_baseline_archive(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct the immutable baseline through its ordinary owned bootstrap."""
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_DDL_BY_TIER, ARCHIVE_BASELINE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        invalidate_active_archive_bootstrap,
    )

    with monkeypatch.context() as context:
        for owner in (
            "polylogue.storage.sqlite.archive_tiers",
            "polylogue.storage.sqlite.archive_tiers.bootstrap",
            "polylogue.storage.sqlite.migration_runner",
            "polylogue.operations.durable_change_train",
        ):
            context.setattr(f"{owner}.ARCHIVE_VERSION_BY_TIER", dict(ARCHIVE_BASELINE_VERSION_BY_TIER))
        for owner in ("polylogue.storage.sqlite.archive_tiers.bootstrap", "polylogue.storage.sqlite.migration_runner"):
            context.setattr(f"{owner}.ARCHIVE_DDL_BY_TIER", dict(ARCHIVE_BASELINE_DDL_BY_TIER))
        initialize_active_archive_root(root)
    # Restoring the runtime target models a different installed runtime;
    # production's process-local unchanged-code certificate cannot span it.
    invalidate_active_archive_bootstrap(root)


SYNTHETIC_SOURCE_TRAIN_TABLE = "synthetic_train_items"


def ship_synthetic_source_train(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    requires_backup: bool = False,
    index_replacement_sql: str | None = None,
) -> None:
    """Install one Source train (slot 002) and raise the runtime Source target to it.

    The archive ships no numbered durable migration, so tests of released-train
    history (backup, restore, clone) install this neutral step. The next owned
    bootstrap advances a v1 Source tier through it and releases
    ``source-002.json``; with ``requires_backup`` the step omits the additive
    claim, so bootstrap takes its verified pre-migration backup first. The
    runtime maps are patched in place, so every module that imported them
    observes the same raised target. ``index_replacement_sql`` installs a
    row-preserving replacement for the Source artifact partition indexes.
    """
    import json
    import re

    from polylogue.storage.sqlite import durable_change_train, migration_runner
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.migration_runner import (
        DurableChangeRider,
        DurableRuntimeConsumer,
        declare_durable_change_train,
        durable_change_train_to_payload,
        durable_migration_claim_for_sql,
    )

    tier = ArchiveTier.SOURCE
    table = SYNTHETIC_SOURCE_TRAIN_TABLE
    create = f"CREATE TABLE {table} (id INTEGER PRIMARY KEY) STRICT;\n"
    if index_replacement_sql is not None:
        assert not requires_backup
        sql = index_replacement_sql
        schema_objects: tuple[str, ...] = (
            "index:idx_raw_artifacts_source_identity",
            "index:idx_raw_artifacts_failure_identity",
        )
    else:
        sql = create if requires_backup else f"-- migration-safety: additive-no-backup\n{create}"
        schema_objects = (f"table:{table}",)
    # An imported package is cached in sys.modules, so each test gets its own name.
    package = "fixture_source_train_" + re.sub(r"\W", "_", str(package_root.resolve()))[-80:]
    tier_package = package_root / package / tier.value
    tier_package.mkdir(parents=True)
    (package_root / package / "__init__.py").write_text("", encoding="utf-8")
    (tier_package / "__init__.py").write_text("", encoding="utf-8")
    name = f"002_{table}.sql"
    (tier_package / name).write_text(sql, encoding="utf-8")
    claim = durable_migration_claim_for_sql(tier, name, sql, owner_ref="owner:synthetic-source-train")
    rider = DurableChangeRider(
        rider_id="rider:synthetic-source-train",
        owner_ref="owner:synthetic-source-train-rider",
        schema_objects=schema_objects,
        runtime_consumers=(
            DurableRuntimeConsumer(
                "bootstrap",
                "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_database",
                "proof:bootstrap",
                ("write",),
            ),
            DurableRuntimeConsumer(
                "daemon-health",
                "polylogue/storage/sqlite/archive_tiers/bootstrap.py:initialize_archive_tier",
                "proof:daemon-health",
                ("read",),
            ),
        ),
        behavior_proof_refs=("proof:bootstrap", "proof:daemon-health"),
    )
    declared = declare_durable_change_train(
        train_id="train:source:synthetic-v2",
        tier=tier,
        current_version=1,
        target_version=2,
        slot=2,
        owner_ref="owner:synthetic-source-train",
        migration=claim,
        riders=(rider,),
        backup_plan_ref="plan:synthetic-verified-backup" if requires_backup else None,
        declared_at_ms=1,
    )
    (tier_package / "002.train.json").write_text(json.dumps(durable_change_train_to_payload(declared)), "utf-8")
    monkeypatch.syspath_prepend(str(package_root))
    canonical_package = migration_runner._migration_package

    def package_for(observed: ArchiveTier) -> str:
        return f"{package}.{tier.value}" if observed is tier else canonical_package(observed)

    monkeypatch.setattr(migration_runner, "_migration_package", package_for)
    monkeypatch.setattr(durable_change_train, "_migration_package", package_for)
    monkeypatch.setitem(ARCHIVE_VERSION_BY_TIER, tier, 2)
    monkeypatch.setitem(
        ARCHIVE_DDL_BY_TIER,
        tier,
        f"{ARCHIVE_DDL_BY_TIER[tier]}\n{sql}",
    )


def checkpoint_durable_tier(path: Path) -> None:
    """Fold a tier's WAL back into its main file and leave no sidecar behind.

    The sidecars are removed explicitly rather than left to SQLite's
    last-close cleanup: a fixture that writes into a *backup* directory would
    otherwise leave a ``.db-wal`` there, and backup publication refuses an
    unbound SQLite sidecar by name. Removing them is only safe because the
    TRUNCATE checkpoint above has already folded every frame back into the
    main file and this connection is closed.
    """
    with closing(sqlite3.connect(path)) as conn:
        busy, log_frames, checkpointed = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert not busy and log_frames == checkpointed, (
            f"{path} could not be checkpointed ({busy=}, {log_frames=}, {checkpointed=}); "
            "an open writer would make removing its sidecars lose data"
        )
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


@contextmanager
def seed_durable_tier(path: Path) -> Iterator[sqlite3.Connection]:
    """Write to a live durable tier and leave nothing in its WAL.

    Use this wherever a fixture seeds a tier that the code under test reopens
    with ``immutable=1``: the committed rows are invisible otherwise.
    """
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            yield conn
    checkpoint_durable_tier(path)


def refresh_archive_format_marker(archive_root: Path) -> None:
    """Re-publish the format marker after a fixture rebuilt a durable tier.

    The marker records each durable tier's birth version and schema
    fingerprint, so replacing or editing the file on disk invalidates it.
    A fixture that deliberately authors the tier restates that evidence here.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_plan import (
        archive_format_marker_path,
        record_fresh_archive_format,
    )

    marker = archive_format_marker_path(archive_root)
    assert marker.is_file(), f"fixture must carry an archive format marker: {marker}"
    marker.unlink()
    record_fresh_archive_format(archive_root)


def rebind_archive_format_fingerprints(root: Path) -> None:
    """Re-fingerprint an existing marker in place, keeping its recorded versions.

    ``record_fresh_archive_format`` is for a live archive and requires all six
    tiers, which a backup directory deliberately does not have. A fixture that
    edits a *backup* tier's schema still has to restate the marker that
    fingerprints it, or the restore refuses on lineage before the behaviour
    under test runs. The fingerprint and digest functions are imported from
    production so this helper cannot drift from the check it satisfies.
    """
    import json

    from polylogue.storage.sqlite.archive_tiers.archive_plan import (
        _DURABLE_FORMAT_TIERS,
        _format_digest,
        _tier_schema_fingerprint,
        archive_format_marker_path,
    )
    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

    marker_path = archive_format_marker_path(root)
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload.pop("digest", None)
    payload["durable_schema_fingerprints"] = {
        tier.value: _tier_schema_fingerprint(root / ARCHIVE_TIER_SPECS[tier].filename) for tier in _DURABLE_FORMAT_TIERS
    }
    payload["digest"] = _format_digest(payload)
    marker_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
