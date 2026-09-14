"""An absent index tier is a typed refusal, not a raw driver exception.

A freshly wiped (or never-created) archive root is an ordinary first-run
state, and every read surface has a correct answer for it. SQLite reports the
open of a missing file under ``mode=ro`` as the opaque ``unable to open
database file`` -- the same text it uses for a permission failure and for a
broken directory -- so the read boundary decides from the file, not the
message, and raises the archive-shaped error the surfaces already map
(polylogue-wwjy6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def test_read_only_open_of_an_absent_index_tier_raises_the_typed_archive_error(tmp_path: Path) -> None:
    """Anti-vacuity: let ``sqlite3.OperationalError`` propagate from
    ``_initialize_store`` and this raises ``sqlite3.OperationalError`` instead
    -- which ``machine_main`` prints as ``unexpected error: OperationalError``
    with no path and no remedy, and which the CLI's missing-archive handlers
    do not catch.
    """
    root = tmp_path / "archive"
    root.mkdir()
    assert not (root / "index.db").exists()

    with pytest.raises(ArchiveTierUnavailableError) as raised:
        ArchiveStore.open_existing(root, read_only=True)

    error = raised.value
    assert error.tier == "index"
    assert error.path == str(root / "index.db")
    # The operator is told where it looked and what to do next, which the raw
    # driver error carried neither of.
    assert str(root / "index.db") in str(error)
    assert error.guidance


def test_a_present_but_unreadable_index_tier_is_not_reported_as_absent(tmp_path: Path) -> None:
    """The file check, not the driver message, decides.

    Anti-vacuity: drop the ``index_db_path.exists()`` guard and a permission
    or corruption failure is reported to the operator as "database file not
    found. run `polylogue ingest`" -- advice that would destroy nothing but
    points at the wrong problem entirely.
    """
    root = tmp_path / "archive"
    root.mkdir()
    index = root / "index.db"
    index.write_bytes(b"")
    index.chmod(0o000)
    try:
        with pytest.raises(Exception) as raised:  # the reason is the assertion
            ArchiveStore.open_existing(root, read_only=True)
        if isinstance(raised.value, ArchiveTierUnavailableError):
            assert raised.value.reason != "database file not found"
    finally:
        index.chmod(0o600)
