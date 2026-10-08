"""Session-profile selection, discovery, hydration, and status gating."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.derived.session.derivation import SessionProfileDerivation, publish_session_profile
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.derived.session.rebuild import hydrate_sessions, load_sync_batch
from polylogue.storage.derived.session.status import session_insight_status_sync
from polylogue.storage.derived.session.usage_rollup import (
    publish_session_usage_rollup,
    session_usage_rollup_recipe_version,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.storage_records import SessionBuilder


def _connection(index_path: Path) -> sqlite3.Connection:
    conn = open_connection(index_path)
    conn.row_factory = sqlite3.Row
    return conn


def _built_profile(root: Path) -> tuple[Path, str, SessionProfileDerivation]:
    initialize_active_archive_root(root)
    index_path = root / "index.db"
    builder = SessionBuilder(index_path, "worker-five-profile")
    builder.add_message(role="user", text="a real profile input")
    builder.add_message(role="assistant", text="a real profile output")
    builder.save()
    session_id = builder.native_session_id()
    with write_lease("test.w5.profile", archive_root=root), closing(_connection(index_path)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        publish_session_usage_rollup(
            conn,
            session_id,
            input_binding=binding,
            recipe_version=session_usage_rollup_recipe_version(),
        )
        assert publish_session_profile(conn, session_id, input_binding=binding)
    adapter = SessionProfileDerivation(
        lambda: _connection(index_path),
        lambda: _connection(index_path),
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        session_scope=lambda _frame: None,
        archive_root=root,
    )
    assert adapter.inspect(object(), (session_id,))[session_id] == "valid"
    return index_path, session_id, adapter


def test_selected_profile_facts_do_not_certify_pending_demand(tmp_path: Path) -> None:
    """Ordinary inspection was stale while selected inspection certified the same rows valid."""
    index_path, session_id, adapter = _built_profile(tmp_path / "archive")
    assert adapter.selected_part_facts(object(), session_id).status == "valid"
    with write_lease("test.w5.demand", archive_root=index_path.parent), closing(_connection(index_path)) as conn:
        conn.execute("INSERT INTO session_profile_demand(session_id, revision) VALUES (?, 1)", (session_id,))
        conn.commit()
    assert adapter.inspect(object(), (session_id,))[session_id] == "stale"
    selected = adapter.selected_part_facts(object(), session_id)
    assert selected.profiles == 1
    assert selected.status == "stale"


def test_required_profile_page_recovers_a_missing_latency_sibling(tmp_path: Path) -> None:
    """No demand and an intact profile previously hid its missing mandatory sibling."""
    index_path, session_id, adapter = _built_profile(tmp_path / "archive")
    assert adapter.required_page(object(), cursor=None, limit=10) == ((), None)
    with write_lease("test.w5.latency", archive_root=index_path.parent), closing(_connection(index_path)) as conn:
        conn.execute("DELETE FROM session_latency_profiles WHERE session_id = ?", (session_id,))
        conn.commit()
        profiles = conn.execute("SELECT count(*) FROM session_profiles WHERE session_id = ?", (session_id,))
        assert profiles.fetchone()[0] == 1
        demand = conn.execute("SELECT count(*) FROM session_profile_demand WHERE session_id = ?", (session_id,))
        assert demand.fetchone()[0] == 0
    keys, _ = adapter.required_page(object(), cursor=None, limit=10)
    assert session_id in keys
    assert adapter.inspect(object(), (session_id,))[session_id] == "stale"


def test_sync_hydration_preserves_stored_attachment_provenance(tmp_path: Path) -> None:
    """The actual sync loader used None defaults despite non-null stored provenance."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    with write_lease("test.w5.hydrate", archive_root=root), closing(_connection(root / "index.db")) as conn:
        session_id = write_fixture_index_session(
            conn,
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="worker-five-attachment",
                messages=[
                    ParsedMessage(
                        provider_message_id="tool",
                        role=Role.TOOL,
                        blocks=[ParsedContentBlock(type=BlockType.TOOL_RESULT, text="generated file", is_error=False)],
                    )
                ],
                attachments=[
                    ParsedAttachment(
                        provider_attachment_id="output",
                        message_provider_id="tool",
                        name="report.pdf",
                        mime_type="application/pdf",
                        path="report.pdf",
                        direction="model_output",
                        producer_ref="message:tool",
                    )
                ],
            ),
        )
        stored = conn.execute(
            "SELECT direction, producer_ref FROM attachment_refs WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert stored is not None and stored[0] == "model_output" and stored[1] is not None
        batch = load_sync_batch(conn, (session_id,))
        record = batch.attachments_by_session[session_id][0]
        assert (record.direction, record.producer_ref) == tuple(stored)
        hydrated = hydrate_sessions(batch)
        assert len(hydrated) == 1
        attachment = list(hydrated[0].messages)[0].attachments[0]
        assert (attachment.direction, attachment.producer_ref) == tuple(stored)


def test_status_counts_runs_without_unrelated_product_tables() -> None:
    """The old shared dependency gate incorrectly returned zero for two readable runs."""
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY, parent_session_id TEXT, origin TEXT,
                branch_type TEXT, title TEXT, git_branch TEXT, native_id TEXT,
                message_count INTEGER, tool_use_count INTEGER, sort_key_ms INTEGER,
                created_at_ms INTEGER, updated_at_ms INTEGER
            );
            INSERT INTO sessions(session_id, parent_session_id, sort_key_ms, updated_at_ms)
            VALUES ('root', NULL, 1000, 1775001600000), ('child', 'root', 2000, 1775001660000);
            """
        )
        status = session_insight_status_sync(conn)
    assert status.total_sessions == 2
    assert status.run_count == 2
    assert status.observed_event_count == 0
    assert status.context_snapshot_count == 0
