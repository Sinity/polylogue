"""Focused contract tests for the shared real-archive snapshot comparator."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.archive_canonical_snapshot import (
    RUN_LOCAL_NORMALIZATION_ALLOWLIST,
    RelationSnapshot,
    assert_canonical_snapshots_equal,
    capture_canonical_snapshot,
)
from tests.infra.convergence_harness import (
    ConvergenceArchive,
    converge_convergence_archive,
    ingest_convergence_pathology,
    initialize_active_archive,
    rich_convergence_pathology,
)
from tests.infra.pathology_composer import ComposedPathology


def _relation_keys(section: tuple[RelationSnapshot, ...]) -> set[tuple[str, str]]:
    return {(relation.database, relation.relation) for relation in section}


def _add_real_action_result(archive_root: Path) -> None:
    """Add one action/result pair through the production parsed-session writer."""
    tool_id = "canonical-snapshot-tool"
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="canonical-snapshot-action",
        title="Canonical snapshot action result",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="action-message",
                role=Role.ASSISTANT,
                position=0,
                text="run the action",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Task",
                        tool_id=tool_id,
                        tool_input={"prompt": "run the action"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id=tool_id,
                        text="command failed",
                        is_error=True,
                        exit_code=1,
                    ),
                ],
            )
        ],
    )
    with open_connection(archive_root / "index.db") as conn:
        write_parsed_session_to_archive(conn, session, content_hash=session_content_hash(session))


def _build_archive(root: Path, pathology: ComposedPathology | None = None) -> ConvergenceArchive:
    """Use production ingest and convergence without the unrelated blob audit."""
    selected_pathology = rich_convergence_pathology() if pathology is None else pathology
    initialize_active_archive(root)
    archive = ingest_convergence_pathology(
        root,
        selected_pathology,
        session_indexes=tuple(range(len(selected_pathology.sessions))),
        converge_after_each=False,
    )
    converge_convergence_archive(archive)
    return archive


def test_snapshot_covers_semantic_archive_and_public_read_surfaces(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path / "archive", rich_convergence_pathology())
    snapshot = capture_canonical_snapshot(archive.root, search_queries=("fixture",))

    assert ("index", "sessions") in _relation_keys(snapshot.canonical_rows)
    assert ("index", "messages") in _relation_keys(snapshot.canonical_rows)
    assert ("index", "blocks") in _relation_keys(snapshot.canonical_rows)
    assert ("index", "session_events") in _relation_keys(snapshot.provenance)
    assert ("source", "raw_sessions") in _relation_keys(snapshot.authority)
    assert ("source", "raw_authority_verdicts") in _relation_keys(snapshot.authority)
    assert ("index", "session_links") in _relation_keys(snapshot.links)
    assert ("index", "action_pairs") in _relation_keys(snapshot.links)
    assert ("index", "attachments") in _relation_keys(snapshot.attachments)
    assert ("index", "session_profiles") in _relation_keys(snapshot.derived_views)
    assert ("index", "actions") in _relation_keys(snapshot.derived_views)

    public_names = {name for name, _value in snapshot.public_projections}
    assert any(name.startswith("summary:") for name in public_names)
    assert any(name.startswith("profile:") for name in public_names)
    assert "actions" in public_names
    assert "threads" in public_names
    assert "search:fixture" in public_names


def test_only_allowlisted_materialization_stamps_are_ignored(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path / "archive", rich_convergence_pathology())
    before = capture_canonical_snapshot(archive.root)

    with sqlite3.connect(archive.root / "index.db") as conn:
        profile = conn.execute("SELECT session_id FROM session_profiles LIMIT 1").fetchone()
        materialization = conn.execute("SELECT session_id FROM insight_materialization LIMIT 1").fetchone()
        assert profile is not None
        assert materialization is not None
        conn.execute(
            "UPDATE session_profiles SET materialized_at = '2099-01-01T00:00:00+00:00', priced_at_ms = 2099 "
            "WHERE session_id = ?",
            (profile[0],),
        )
        conn.execute(
            "UPDATE insight_materialization SET materialized_at_ms = 2099 WHERE session_id = ?",
            (materialization[0],),
        )
        conn.commit()

    after = capture_canonical_snapshot(archive.root)
    assert_canonical_snapshots_equal(before, after)
    assert RUN_LOCAL_NORMALIZATION_ALLOWLIST["index.session_profiles"] == frozenset({"materialized_at", "priced_at_ms"})


@pytest.mark.parametrize(
    ("name", "mutate"),
    (
        (
            "material_origin",
            lambda conn: conn.execute(
                "UPDATE messages SET material_origin = 'runtime_protocol' "
                "WHERE message_id = (SELECT message_id FROM messages LIMIT 1)"
            ),
        ),
        (
            "link status",
            lambda conn: conn.execute(
                "UPDATE session_links SET status = CASE WHEN status = 'quarantined' THEN 'repaired' "
                "ELSE 'quarantined' END WHERE rowid = (SELECT rowid FROM session_links LIMIT 1)"
            ),
        ),
        (
            "provenance",
            lambda conn: conn.execute(
                "UPDATE session_profiles SET evidence_payload_json = '{\"red_mutation\":true}' "
                "WHERE session_id = (SELECT session_id FROM session_profiles LIMIT 1)"
            ),
        ),
        (
            "action result state",
            lambda conn: conn.execute(
                "UPDATE blocks SET tool_result_is_error = CASE WHEN tool_result_is_error = 1 THEN 0 ELSE 1 END "
                "WHERE block_id = (SELECT block_id FROM blocks WHERE block_type = 'tool_result' LIMIT 1)"
            ),
        ),
    ),
)
def test_semantic_mutations_are_red(
    tmp_path: Path, name: str, mutate: Callable[[sqlite3.Connection], sqlite3.Cursor]
) -> None:
    """The comparator must fail on semantic fields, not normalize them away."""
    pathology = rich_convergence_pathology()
    canonical = _build_archive(tmp_path / "canonical", pathology)
    mutated = _build_archive(tmp_path / "mutated", pathology)
    if name == "action result state":
        _add_real_action_result(canonical.root)
        _add_real_action_result(mutated.root)
        converge_convergence_archive(canonical)
        converge_convergence_archive(mutated)
    with sqlite3.connect(mutated.root / "index.db") as conn:
        result = mutate(conn)
        if result.rowcount != 1:
            raise AssertionError(f"{name} mutation did not update exactly one row")
        conn.commit()

    expected = capture_canonical_snapshot(canonical.root)
    actual = capture_canonical_snapshot(mutated.root)
    expected_marker = {
        "material_origin": "messages",
        "link status": "session_links",
        "provenance": "session_profiles",
        "action result state": "actions",
    }[name]
    with pytest.raises(AssertionError, match=expected_marker):
        assert_canonical_snapshots_equal(expected, actual)
