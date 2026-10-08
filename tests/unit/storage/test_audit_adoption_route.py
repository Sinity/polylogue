"""A lost ``audit.db`` is refused by name, and only for a proven lineage member.

Bootstrap creates every durable tier under one pending intent, so an
established archive without ``audit.db`` lost it outside Polylogue. Startup
refuses it with a lost-tier message and never recreates the tier. The
format-marker guard must still run first for the surviving durable pair: an
archive whose remaining tiers are not proven members of this lineage is
reported as that more severe finding instead.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _bootstrapped(root: Path) -> Path:
    initialize_active_archive_root(root)
    assert (root / ".polylogue-format.json").is_file()
    return root


def test_lost_audit_is_refused_by_name(tmp_path: Path) -> None:
    """A lineage member missing only audit.db is refused as a lost durable tier.

    Anti-vacuity: restoring the unconditional
    ``assert_archive_format_lineage(root)`` in ``_initialize_active_archive_root``
    makes this raise ``archive format marker names a missing durable tier``
    instead, and the ``match`` fails. The second assertion pins the other
    direction: the refusal must not become a silent re-creation of the durable
    tier.
    """
    root = _bootstrapped(tmp_path / "archive")
    (root / "audit.db").unlink()

    with pytest.raises(RuntimeError, match="established archive is missing audit.db"):
        initialize_active_archive_root(root)

    assert not (root / "audit.db").exists()


@pytest.mark.parametrize("damage", ["nomarker", "forged", "foreign", "nouser"])
def test_lost_audit_refusal_needs_lineage_proof(tmp_path: Path, damage: str) -> None:
    """Only a proven lineage member is reported as having lost audit.db.

    Anti-vacuity: dropping the scoped ``assert_archive_format_lineage`` call and
    raising the lost-tier message on ``audit.db`` absence alone turns every
    case here into the lost-tier refusal, reporting an archive whose surviving
    durable files were never shown to belong to this lineage as merely
    missing one tier.
    """
    root = _bootstrapped(tmp_path / "archive")
    (root / "audit.db").unlink()
    marker = root / ".polylogue-format.json"
    if damage == "nomarker":
        marker.unlink()
    elif damage == "forged":
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["tier_versions"]["source"] = 99
        marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    elif damage == "foreign":
        with closing(sqlite3.connect(root / "source.db")) as connection:
            connection.execute("CREATE TABLE intruder (x TEXT)")
            connection.commit()
    else:
        (root / "user.db").unlink()

    with pytest.raises(RuntimeError) as refusal:
        initialize_active_archive_root(root)

    assert "established archive is missing audit.db" not in str(refusal.value)
    if damage == "foreign":
        # The v1 Source file no longer matches its birth fingerprint, so the
        # lineage proof itself refuses it.
        assert "is not part of polylogue.archive-format" in str(refusal.value)
        with closing(sqlite3.connect(root / "source.db")) as connection:
            assert connection.execute("SELECT name FROM sqlite_schema WHERE name='intruder'").fetchone() == (
                "intruder",
            )
    assert not (root / "audit.db").exists()
