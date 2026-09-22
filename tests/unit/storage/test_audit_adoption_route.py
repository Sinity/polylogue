"""The archive-format guard must not hide the audit-adoption recovery route.

An archive that belongs to the current format lineage and has lost only
``audit.db`` is recoverable through ``maintenance migrate-tier audit
--adopt-established-audit``. The format-marker guard introduced with the
lineage floor inspects every durable tier file before that refusal is reached,
so an operator in exactly that state was told ``archive format marker names a
missing durable tier`` -- true, and actionable by nobody.
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


def test_lost_audit_names_the_adoption_route(tmp_path: Path) -> None:
    """A lineage member missing only audit.db is refused with its recovery command.

    Anti-vacuity: restoring the unconditional
    ``assert_archive_format_lineage(root)`` in ``_initialize_active_archive_root``
    makes this raise ``archive format marker names a missing durable tier``
    instead, and the ``match`` fails. The second assertion pins the other
    direction: the refusal must not become a silent re-creation of the durable
    tier the operator is being asked to adopt.
    """
    root = _bootstrapped(tmp_path / "archive")
    (root / "audit.db").unlink()

    with pytest.raises(RuntimeError, match="adopt-established-audit"):
        initialize_active_archive_root(root)

    assert not (root / "audit.db").exists()


@pytest.mark.parametrize("damage", ["nomarker", "forged", "foreign", "nouser"])
def test_adoption_route_needs_lineage_proof(tmp_path: Path, damage: str) -> None:
    """Only a proven lineage member reaches the adoption route.

    Anti-vacuity: dropping the scoped ``assert_archive_format_lineage`` call and
    raising the adoption message on ``audit.db`` absence alone turns every case
    here into ``adopt-established-audit``, which would invite an operator to
    adopt an audit tier into an archive whose surviving durable files were never
    shown to belong to this lineage.
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

    assert "adopt-established-audit" not in str(refusal.value)
    assert not (root / "audit.db").exists()
