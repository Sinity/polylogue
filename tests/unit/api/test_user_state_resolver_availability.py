"""An unreadable index tier refuses a target check; it never denies the target.

polylogue-p707n's class at the user-state write boundary: "the tier could not
be read" and "the target is not materialized" are different answers, and only
the second one is a fact about the target.

Anti-vacuity: restore ``except sqlite3.Error: return False`` in
``_row_exists``/``_index_db_path`` and the corrupt-tier case reports the target
as not materialized, matching the genuinely-empty case exactly.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.api.user_state_resolver import resolve_insight_target
from polylogue.core.user_state_targets import TARGET_SESSION


def _materialized_index(root: Path) -> Path:
    index_db = root / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE session_profiles(session_id TEXT PRIMARY KEY)")
    return index_db


def _resolve(root: Path) -> None:
    asyncio.run(
        resolve_insight_target(
            root,
            target_type=TARGET_SESSION,
            target_id="origin:native",
            session_id="origin:native",
        )
    )


def test_absent_profile_is_reported_as_not_materialized(tmp_path: Path) -> None:
    _materialized_index(tmp_path)

    with pytest.raises(ValueError, match="is not materialized"):
        _resolve(tmp_path)


def test_unreadable_index_refuses_instead_of_denying_the_target(tmp_path: Path) -> None:
    _materialized_index(tmp_path)
    (tmp_path / "index.db").write_bytes(b"this is not a sqlite database")

    with pytest.raises(ValueError, match="could not be checked") as excinfo:
        _resolve(tmp_path)

    assert "is not materialized" not in str(excinfo.value)


def test_missing_index_is_absence_not_unavailability(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not materialized"):
        _resolve(tmp_path)
