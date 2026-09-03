"""A derived tier's identity is stamped durably and checked before it is used.

``PRAGMA user_version`` is a migration cursor, not a proof. The bootstrap
routes rewrite it while materialising, so a check that reads only that value
after materialisation validates what it just wrote. These tests pin the four
places that make the stamp trustworthy: it survives the connection that wrote
it, it is verified alongside the version, the version is admitted before
materialisation can restamp it, and a read that cannot inspect the schema is
distinguished from a read that inspected it and found drift.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.errors import SchemaSkewError, SchemaVersionMismatchError
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_archive_tier,
    open_initialized_tier_connection,
)
from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, read_schema_identity
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported
from polylogue.storage.sqlite.schema import assert_readable_archive_layout
from polylogue.storage.sqlite.schema_manifest import canonical_schema_manifest

INDEX_VERSION = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]


def test_schema_identity_stamp_survives_the_connection_that_wrote_it(tmp_path: Path) -> None:
    """The stamp is committed, not left in the writer's open transaction.

    Anti-vacuity: drop the ``conn.commit()`` after
    ``stamp_derived_schema_identity`` in ``initialize_archive_tier`` and this
    fails -- the stamp lands after the materialisation commit, so closing
    without committing rolls it back and the reopened tier reads ``None``.
    """
    path = tmp_path / "ops.db"
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.OPS)
    finally:
        conn.close()

    with sqlite3.connect(path) as reopened:
        assert read_schema_identity(reopened, DerivedTier.OPS) is not None


def test_tier_admission_checks_identity_and_not_only_the_version(tmp_path: Path) -> None:
    """A current version with a foreign identity is still refused.

    Anti-vacuity: remove the ``_assert_derived_identity_supported`` call from
    ``_assert_schema_supported`` and this fails with no exception raised --
    ``user_version`` is untouched here, so the version gate alone passes a
    tier stamped by a runtime whose read models this one cannot interpret.
    """
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute("UPDATE schema_identity SET identity = 'from-another-runtime' WHERE tier = 'index'")
        conn.commit()
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == INDEX_VERSION

        with pytest.raises(SchemaSkewError):
            assert_tier_schema_supported(conn, path, ArchiveTier.INDEX)
    finally:
        conn.close()


def test_a_skewed_tier_is_refused_before_materialisation_restamps_it(tmp_path: Path) -> None:
    """The stored version is admitted before it can be overwritten.

    Anti-vacuity: delete the stored-version admission from
    ``open_initialized_tier_connection`` and this fails with the connection
    returned instead of raising -- materialisation rewrites ``user_version``
    to the current spec, so the check that follows reads the value this route
    just wrote and can never observe the skew.
    """
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute(f"PRAGMA user_version = {INDEX_VERSION - 1}")

    with pytest.raises(SchemaSkewError) as caught:
        open_initialized_tier_connection(path, ArchiveTier.INDEX, daemon=False)

    assert caught.value.found == INDEX_VERSION - 1

    # The version on disk was refused, not rewritten.
    with sqlite3.connect(path) as after:
        assert int(after.execute("PRAGMA user_version").fetchone()[0]) == INDEX_VERSION - 1


@pytest.mark.parametrize(
    ("raised", "expected_action"),
    [
        pytest.param(sqlite3.OperationalError("disk I/O error"), "retry", id="runtime-failure"),
        pytest.param(RuntimeError("index schema semantic manifest mismatch: {}"), "rebuild_index", id="real-drift"),
    ],
)
def test_an_unreadable_schema_is_not_reported_as_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    expected_action: str,
) -> None:
    """A failed inspection does not prescribe the remedy drift prescribes.

    Anti-vacuity: merge the handlers back into
    ``except (RuntimeError, sqlite3.Error)`` in
    ``assert_readable_archive_layout`` and the runtime-failure case fails --
    both raises then report ``rebuild_index``, telling an operator to destroy
    a sound index over a read that never inspected its schema.
    """
    path = tmp_path / "index.db"
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.commit()

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise raised

        monkeypatch.setattr("polylogue.storage.sqlite.schema.schema_manifest_diff", _raise)

        with pytest.raises(SchemaVersionMismatchError) as caught:
            assert_readable_archive_layout(conn)
    finally:
        conn.close()

    assert caught.value.lifecycle_action == expected_action


def test_canonical_manifest_is_rendered_once_per_declared_schema() -> None:
    """The canonical manifest is cached, not re-executed on every read open.

    Anti-vacuity: remove the ``functools.cache`` from ``_canonical_schema_manifest``
    and this fails -- two calls would render two separate in-memory databases
    and return distinct objects rather than the same shared one.
    """
    first = canonical_schema_manifest(ArchiveTier.INDEX)
    second = canonical_schema_manifest(ArchiveTier.INDEX)

    assert first is second


def test_canonical_manifest_excludes_fts5_shadow_tables() -> None:
    """FTS5's own storage is not part of the archive's schema contract.

    Anti-vacuity: restore the bare ``NOT LIKE 'sqlite_%'`` filter in
    ``_projection`` and the shadow assertion fails -- ``messages_fts_data``
    and friends reappear, so an FTS5 build that reshapes them reports library
    drift as archive schema drift. Widen the filter from the exact shadow
    suffixes back to a ``messages_fts`` prefix and the ``messages_fts_identity``
    assertion fails instead: that is a declared table of ours, and dropping it
    from the contract would hide real drift in it.
    """
    manifest = canonical_schema_manifest(ArchiveTier.INDEX)
    names = {name for _kind, name, _sql in manifest.objects}

    assert "messages_fts" in names, "the declaring virtual table stays in the contract"
    assert "messages_fts_identity" in names, "a declared table sharing the prefix is not a shadow table"
    assert not (names & {f"messages_fts{suffix}" for suffix in ("_data", "_idx", "_content", "_docsize", "_config")})
