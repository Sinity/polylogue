"""Tests for polylogue-m3p9: bounded backfill of NULL session timestamps.

``write_parsed_session_to_archive`` (see
``tests/unit/storage/test_archive_tiers_write.py``) now derives
``sessions.created_at_ms``/``updated_at_ms`` from message evidence going
forward, but that fix only ever touches a session on its next write. These
tests cover the separate bounded backfill
(``polylogue.storage.session_timestamp_backfill``) for rows that were
already archived with NULL session timestamps before the write-path fix
landed -- simulated here by nulling the columns directly after seeding via
the real writer, since the writer itself no longer produces NULLs when
message evidence exists.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Config
from polylogue.storage.repair import repair_session_timestamp_backfill
from polylogue.storage.session_timestamp_backfill import count_sessions_missing_timestamps_sync
from tests.infra.archive_scenarios import open_index_db
from tests.infra.storage_records import SessionBuilder, db_setup


def _make_config(workspace_env: dict[str, Path], db_path: Path) -> Config:
    return Config(
        archive_root=Path(workspace_env["archive_root"]),
        render_root=Path(workspace_env["archive_root"]),
        sources=[],
        db_path=db_path,
    )


def _null_session_timestamps(db_path: Path, session_id: str) -> None:
    """Simulate a pre-#m3p9 legacy row: session timestamps NULL, message
    ``occurred_at_ms`` intact -- exactly what re-ingest no longer produces
    after the write-path fix, but what already sits in the live archive.
    """
    with open_index_db(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET created_at_ms = NULL, updated_at_ms = NULL WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()


class TestSessionTimestampBackfill:
    def test_backfill_derives_null_session_timestamps_from_messages(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = SessionBuilder(db_path, "legacy-1")
        builder.provider("codex").title("legacy row")
        builder.add_message(message_id="m1", role="user", text="first", timestamp="2026-01-01T00:00:00+00:00")
        builder.add_message(message_id="m2", role="assistant", text="last", timestamp="2026-01-01T00:10:00+00:00")
        builder.save()
        session_id = builder.native_session_id()
        _null_session_timestamps(db_path, session_id)

        with open_index_db(db_path) as conn:
            row = conn.execute(
                "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            assert row["created_at_ms"] is None
            assert row["updated_at_ms"] is None

        cfg = _make_config(workspace_env, db_path)
        result = repair_session_timestamp_backfill(cfg, dry_run=False)
        assert result.success, result.detail
        assert result.repaired_count == 1

        with open_index_db(db_path) as conn:
            row = conn.execute(
                "SELECT created_at_ms, updated_at_ms, sort_key_ms FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        assert row["created_at_ms"] == 1_767_225_600_000  # 2026-01-01T00:00:00Z
        assert row["updated_at_ms"] == 1_767_226_200_000  # 2026-01-01T00:10:00Z
        assert row["sort_key_ms"] == row["updated_at_ms"]

    def test_backfill_does_not_touch_sessions_with_both_timestamps_set(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = SessionBuilder(db_path, "already-timestamped")
        builder.provider("codex").title("already fine")
        builder.created_at("2020-06-01T00:00:00+00:00")
        builder.updated_at("2020-06-01T00:05:00+00:00")
        builder.add_message(message_id="m1", role="user", text="hi", timestamp="2026-01-01T00:00:00+00:00")
        builder.save()
        session_id = builder.native_session_id()

        cfg = _make_config(workspace_env, db_path)
        result = repair_session_timestamp_backfill(cfg, dry_run=False)
        assert result.success
        assert result.repaired_count == 0

        with open_index_db(db_path) as conn:
            row = conn.execute(
                "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        # Provider-supplied timestamps untouched -- not overwritten by the
        # (much later) message evidence.
        assert row["created_at_ms"] == 1_590_969_600_000  # 2020-06-01T00:00:00Z
        assert row["updated_at_ms"] == 1_590_969_900_000  # 2020-06-01T00:05:00Z

    def test_backfill_only_fills_the_missing_half_when_one_timestamp_is_set(
        self, workspace_env: dict[str, Path]
    ) -> None:
        """A session with created_at set but updated_at NULL (partial
        provider evidence) only has the missing field filled -- COALESCE on
        both sides means the existing value is never overwritten.
        """
        db_path = db_setup(workspace_env)
        builder = SessionBuilder(db_path, "half-timestamped")
        builder.provider("codex").title("half fine")
        builder.add_message(message_id="m1", role="user", text="first", timestamp="2026-01-01T00:00:00+00:00")
        builder.add_message(message_id="m2", role="assistant", text="last", timestamp="2026-01-01T00:10:00+00:00")
        builder.save()
        session_id = builder.native_session_id()
        with open_index_db(db_path) as conn:
            # Keep created_at_ms as a durable provider value; null only updated_at_ms.
            conn.execute(
                "UPDATE sessions SET created_at_ms = ?, updated_at_ms = NULL WHERE session_id = ?",
                (1_500_000_000_000, session_id),
            )
            conn.commit()

        cfg = _make_config(workspace_env, db_path)
        result = repair_session_timestamp_backfill(cfg, dry_run=False)
        assert result.success
        assert result.repaired_count == 1

        with open_index_db(db_path) as conn:
            row = conn.execute(
                "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        assert row["created_at_ms"] == 1_500_000_000_000  # untouched
        assert row["updated_at_ms"] == 1_767_226_200_000  # 2026-01-01T00:10:00Z, derived

    def test_backfill_is_idempotent(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = SessionBuilder(db_path, "legacy-idempotent")
        builder.provider("codex").title("legacy row")
        builder.add_message(message_id="m1", role="user", text="first", timestamp="2026-01-01T00:00:00+00:00")
        builder.save()
        session_id = builder.native_session_id()
        _null_session_timestamps(db_path, session_id)

        cfg = _make_config(workspace_env, db_path)
        first = repair_session_timestamp_backfill(cfg, dry_run=False)
        assert first.success
        assert first.repaired_count == 1

        second = repair_session_timestamp_backfill(cfg, dry_run=False)
        assert second.success
        assert second.repaired_count == 0

    def test_preview_count_matches_repaired_count(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = SessionBuilder(db_path, "legacy-preview")
        builder.provider("codex").title("legacy row")
        builder.add_message(message_id="m1", role="user", text="first", timestamp="2026-01-01T00:00:00+00:00")
        builder.save()
        session_id = builder.native_session_id()
        _null_session_timestamps(db_path, session_id)

        with open_index_db(db_path) as conn:
            preview_count = count_sessions_missing_timestamps_sync(conn)
        assert preview_count == 1

        cfg = _make_config(workspace_env, db_path)
        dry_run_result = repair_session_timestamp_backfill(cfg, dry_run=True)
        assert dry_run_result.repaired_count == preview_count

        # Dry-run does not mutate.
        with open_index_db(db_path) as conn:
            row = conn.execute("SELECT created_at_ms FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        assert row["created_at_ms"] is None

        actual = repair_session_timestamp_backfill(cfg, dry_run=False)
        assert actual.repaired_count == preview_count

    def test_backfill_leaves_undatable_sessions_null(self, workspace_env: dict[str, Path]) -> None:
        """A session with no message timestamps either (never datable) is
        not counted as a candidate and stays NULL -- never backdated.
        """
        db_path = db_setup(workspace_env)
        builder = SessionBuilder(db_path, "undatable")
        builder.provider("codex").title("undatable")
        builder.add_message(message_id="m1", role="user", text="hi", timestamp=None)
        builder.save()
        session_id = builder.native_session_id()
        _null_session_timestamps(db_path, session_id)

        with open_index_db(db_path) as conn:
            preview_count = count_sessions_missing_timestamps_sync(conn)
        assert preview_count == 0

        cfg = _make_config(workspace_env, db_path)
        result = repair_session_timestamp_backfill(cfg, dry_run=False)
        assert result.success
        assert result.repaired_count == 0

        with open_index_db(db_path) as conn:
            row = conn.execute(
                "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        assert row["created_at_ms"] is None
        assert row["updated_at_ms"] is None
