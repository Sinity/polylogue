from __future__ import annotations

import fcntl
import os
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import native_sql_owner_for_connection
from polylogue.storage.sqlite.write_lease import ARCHIVE_WRITE_CUSTODY_LOCK_NAME
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session


def _custody_lock_is_free(root: Path) -> bool:
    """Probe the archive custody lock without waiting on it.

    flock locks belong to an open file description, so a second description
    in this same process is refused while any live custody still holds it.
    """
    fd = os.open(root / ARCHIVE_WRITE_CUSTODY_LOCK_NAME, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def test_a_caller_index_handle_does_not_keep_the_lease_custody_locked(tmp_path: Path) -> None:
    """A long-lived caller connection must not pin the writer lease after it ends.

    The Index mutation scope registers a physical owner for an unowned caller
    connection under the lease's custody. Anti-vacuity: leave that owner
    registered after the seal settles and the caller's still-open handle keeps
    the archive custody's file lock, so the probe below is refused and every
    later writer of this archive waits on it forever.
    """
    conn = connect_measured(tmp_path / "index.db")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    try:
        write_fixture_index_session(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="borrowed-writer",
                title="borrowed writer",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="a caller handle outlives its lease",
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )
        assert native_sql_owner_for_connection(conn) is None
        assert _custody_lock_is_free(tmp_path)
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    finally:
        close_fixture_index_connection(conn)
