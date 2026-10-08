"""Six-tier bootstrap is paid once per archive generation (polylogue-q53j4 AC1).

``initialize_active_archive_root`` runs on every index-tier sync write open
and once per ingest batch. Its body opens and validates all six tiers, a
fixed per-call cost that a cold build repeats for every chunk while learning
nothing new. These tests are *cost* assertions: they count how many times the
validation body actually executes, so a memo that degenerates back into
per-open work is red even though every behavioural assertion still passes.

Anti-vacuity runs in both directions, because both failures are real:

* Removing the memo makes ``…validates_once_per_generation`` red (the counter
  tracks calls rather than generations).
* Turning the memo into a once-flag makes every ``…revalidates…`` test red.
  A bootstrap that never re-runs would let a replaced tier, a promoted index
  or a durable-train migration skip validation entirely, which the
  2026-09-16 acceptance addendum names as a defect rather than a fix.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    active_archive_bootstrap_validation_count,
    initialize_active_archive_root,
    invalidate_active_archive_bootstrap,
)


@pytest.fixture(autouse=True)
def _forget_bootstrap_generations() -> None:
    """Start every test with no memoized generation from an earlier one."""
    invalidate_active_archive_bootstrap()


def _validations(callable_: object) -> int:
    """Validations executed while ``callable_`` ran."""
    before = active_archive_bootstrap_validation_count()
    callable_()  # type: ignore[operator]
    return active_archive_bootstrap_validation_count() - before


def test_repeat_bootstrap_validates_once_per_generation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    assert _validations(lambda: initialize_active_archive_root(root)) == 1

    # Twelve more opens of the same unchanged archive: a cold build's chunk
    # loop. Nothing about the archive has changed, so nothing is revalidated.
    def twelve_more() -> None:
        for _ in range(12):
            initialize_active_archive_root(root)

    assert _validations(twelve_more) == 0


def test_replaced_durable_tier_revalidates(tmp_path: Path) -> None:
    """A tier file swapped for a different inode is a different generation."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    assert _validations(lambda: initialize_active_archive_root(root)) == 0

    source = root / "source.db"
    replacement = tmp_path / "source-copy.db"
    shutil.copy2(source, replacement)
    assert replacement.stat().st_ino != source.stat().st_ino
    os.replace(replacement, source)

    # No released train binds this tier's physical identity, so the same
    # schema under a new inode is admitted, but only after a fresh validation.
    assert _validations(lambda: initialize_active_archive_root(root)) == 1


def test_durable_train_manifest_change_revalidates(tmp_path: Path) -> None:
    """A durable migration's manifest write is the schema-change case."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    assert _validations(lambda: initialize_active_archive_root(root)) == 0

    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "0001.train.json").write_text("{}", encoding="utf-8")

    assert _validations(lambda: initialize_active_archive_root(root)) == 1


def test_appended_durable_train_manifest_revalidates(tmp_path: Path) -> None:
    """Growing an existing manifest in place is also a new generation.

    An inode-only token would miss this: the file keeps its identity while
    its content -- the recorded migration state bootstrap reconciles against
    -- changes underneath it.
    """
    root = tmp_path / "archive"
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_root / "0001.train.json"
    manifest.write_text("{}", encoding="utf-8")
    initialize_active_archive_root(root)
    assert _validations(lambda: initialize_active_archive_root(root)) == 0

    manifest.write_text('{"step": 2}', encoding="utf-8")

    assert _validations(lambda: initialize_active_archive_root(root)) == 1


def test_removed_format_marker_is_refused_not_memoized_past(tmp_path: Path) -> None:
    """Marker state selects the bootstrap branch, so it is part of the token.

    This is the sharpest statement of "not a once-flag": the archive was
    successfully bootstrapped and then had its lineage marker removed. A
    blanket memo returns silently. The generation token notices, re-runs the
    body, and the body refuses -- the same refusal an unmemoized bootstrap
    raises.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_plan import archive_format_marker_path

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    assert _validations(lambda: initialize_active_archive_root(root)) == 0

    marker = archive_format_marker_path(root)
    assert marker.is_file()
    marker.unlink()

    before = active_archive_bootstrap_validation_count()
    with pytest.raises(RuntimeError, match="archive format marker is missing"):
        initialize_active_archive_root(root)
    assert active_archive_bootstrap_validation_count() - before == 1


def test_explicit_invalidation_revalidates(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    assert _validations(lambda: initialize_active_archive_root(root)) == 0

    invalidate_active_archive_bootstrap(root)

    assert _validations(lambda: initialize_active_archive_root(root)) == 1


def test_distinct_roots_do_not_share_a_generation(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert _validations(lambda: initialize_active_archive_root(first)) == 1
    assert _validations(lambda: initialize_active_archive_root(second)) == 1
    assert _validations(lambda: initialize_active_archive_root(first)) == 0


def test_failed_bootstrap_is_not_memoized(tmp_path: Path) -> None:
    """A refusal must be re-raised on every later call, never memoized away."""
    root = tmp_path / "archive"
    (root / ".maintenance-state").mkdir(parents=True)
    (root / "source.db").symlink_to(tmp_path / "nowhere.db")

    for _ in range(3):
        before = active_archive_bootstrap_validation_count()
        with pytest.raises(RuntimeError):
            initialize_active_archive_root(root)
        # Each attempt ran the body: a memoized failure would raise nothing
        # the second time, or raise without doing the work that proves it.
        assert active_archive_bootstrap_validation_count() - before == 1


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
@pytest.mark.parametrize("text_factory", [str, bytes])
def test_supplied_lineage_handle_preserves_literal_schema_fingerprint(
    tmp_path: Path, row_factory: Any, text_factory: Any
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive_plan import (
        _assert_archive_format_tier_lineage,
        _read_archive_format_birth,
    )
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    birth = _read_archive_format_birth(root)
    with closing(sqlite3.connect(root / "user.db")) as conn:
        conn.row_factory = row_factory
        conn.text_factory = text_factory
        _assert_archive_format_tier_lineage(root, ArchiveTier.USER, conn, birth)
    with closing(sqlite3.connect(root / "source.db")) as other:
        with pytest.raises(RuntimeError, match="connection does not own"):
            _assert_archive_format_tier_lineage(root, ArchiveTier.USER, other, birth)
