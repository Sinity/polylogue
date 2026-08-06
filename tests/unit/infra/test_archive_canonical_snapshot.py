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


def _add_web_construct(archive_root: Path) -> None:
    with sqlite3.connect(archive_root / "index.db") as conn:
        row = conn.execute("SELECT session_id, message_id, block_id FROM blocks LIMIT 1").fetchone()
        assert row is not None
        conn.execute(
            """
            INSERT INTO web_content_constructs (
                session_id, message_id, block_id, position, provider, construct_type,
                provider_key, title, url, text, task_type, rank
            ) VALUES (?, ?, ?, 99, 'codex', 'search_result', 'fixture', 'fixture', ?, ?, ?, ?)
            """,
            (*row, "https://example.test/canonical", "canonical construct", "browser_search", 1),
        )
        conn.commit()


def _add_revision_head(archive_root: Path) -> None:
    with sqlite3.connect(archive_root / "index.db") as index, sqlite3.connect(archive_root / "source.db") as source:
        session_id, raw_id = index.execute(
            "SELECT session_id, raw_id FROM sessions WHERE raw_id IS NOT NULL LIMIT 1"
        ).fetchone()
        logical_source_key, source_revision, blob_hash = source.execute(
            "SELECT logical_source_key, source_revision, blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone()
        index.execute(
            """
            INSERT INTO raw_revision_heads (
                logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, 'byte', 0, 0, NULL, 1)
            """,
            (logical_source_key or "canonical:head", session_id, raw_id, source_revision or "rev-1", blob_hash),
        )
        index.commit()


def test_equivalent_archives_under_different_roots_compare_equal(tmp_path: Path) -> None:
    canonical = _build_archive(tmp_path / "root-a", rich_convergence_pathology())
    relocated = _build_archive(tmp_path / "root-b", rich_convergence_pathology())

    assert_canonical_snapshots_equal(
        capture_canonical_snapshot(canonical.root), capture_canonical_snapshot(relocated.root)
    )


@pytest.mark.parametrize("column", ("blob_hash", "native_id"))
def test_raw_identity_mutations_are_red(tmp_path: Path, column: str) -> None:
    canonical = _build_archive(tmp_path / "canonical", rich_convergence_pathology())
    mutated = _build_archive(tmp_path / "mutated", rich_convergence_pathology())
    with sqlite3.connect(mutated.root / "source.db") as conn:
        if column == "blob_hash":
            conn.execute(
                "UPDATE raw_sessions SET blob_hash = zeroblob(32) WHERE raw_id = (SELECT raw_id FROM raw_sessions LIMIT 1)"
            )
        else:
            conn.execute(
                "UPDATE raw_sessions SET native_id = 'mutated-native-id' WHERE raw_id = (SELECT raw_id FROM raw_sessions LIMIT 1)"
            )
        conn.commit()

    with pytest.raises(AssertionError, match="authority|canonical_rows"):
        assert_canonical_snapshots_equal(
            capture_canonical_snapshot(canonical.root), capture_canonical_snapshot(mutated.root)
        )


def test_default_fts_projection_detects_real_posting_deletion(tmp_path: Path) -> None:
    canonical = _build_archive(tmp_path / "canonical", rich_convergence_pathology())
    mutated = _build_archive(tmp_path / "mutated", rich_convergence_pathology())
    before = capture_canonical_snapshot(mutated.root)
    before_searches = {name for name, _value in before.public_projections if name.startswith("search:")}
    assert before_searches

    query = sorted(name.removeprefix("search:") for name in before_searches)[0]
    with sqlite3.connect(mutated.root / "index.db") as conn:
        posting = conn.execute("SELECT rowid FROM messages_fts WHERE text MATCH ? LIMIT 1", (query,)).fetchone()
        assert posting is not None
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (posting[0],))
        conn.commit()

    after = capture_canonical_snapshot(mutated.root)
    with pytest.raises(AssertionError, match="public_projections"):
        assert_canonical_snapshots_equal(capture_canonical_snapshot(canonical.root), after)
    before_search = dict(before.public_projections)[f"search:{query}"]
    after_search = dict(after.public_projections)[f"search:{query}"]
    assert before_search != after_search


@pytest.mark.parametrize("table", ("assertions", "context_deliveries"))
def test_user_state_mutations_are_red(tmp_path: Path, table: str) -> None:
    canonical = _build_archive(tmp_path / "canonical", rich_convergence_pathology())
    mutated = _build_archive(tmp_path / "mutated", rich_convergence_pathology())
    with sqlite3.connect(mutated.root / "user.db") as conn:
        if table == "assertions":
            conn.execute(
                "INSERT INTO assertions (assertion_id, target_ref, kind, body_text, created_at_ms, updated_at_ms) "
                "VALUES ('canonical-red', 'session:fixture', 'note', 'changed', 1, 1)"
            )
        else:
            conn.execute(
                """
                INSERT INTO context_deliveries (
                    snapshot_ref, recipient_ref, boundary, context_image_json, context_image_sha256,
                    delivered_by_ref, delivered_at_ms
                ) VALUES ('snapshot:red', 'agent:test', 'test', '{}', ?, 'agent:test', 1)
                """,
                ("0" * 64,),
            )
        conn.commit()

    with pytest.raises(AssertionError, match="user_state"):
        assert_canonical_snapshots_equal(
            capture_canonical_snapshot(canonical.root), capture_canonical_snapshot(mutated.root)
        )


def test_web_construct_mutation_is_red(tmp_path: Path) -> None:
    canonical = _build_archive(tmp_path / "canonical", rich_convergence_pathology())
    mutated = _build_archive(tmp_path / "mutated", rich_convergence_pathology())
    _add_web_construct(canonical.root)
    _add_web_construct(mutated.root)
    with sqlite3.connect(mutated.root / "index.db") as conn:
        conn.execute("UPDATE web_content_constructs SET url = 'https://example.test/mutated'")
        conn.commit()

    with pytest.raises(AssertionError, match="canonical_rows/.*web_content_constructs"):
        assert_canonical_snapshots_equal(
            capture_canonical_snapshot(canonical.root), capture_canonical_snapshot(mutated.root)
        )


def test_excision_tombstone_mutation_is_red(tmp_path: Path) -> None:
    canonical = _build_archive(tmp_path / "canonical", rich_convergence_pathology())
    mutated = _build_archive(tmp_path / "mutated", rich_convergence_pathology())
    with sqlite3.connect(mutated.root / "source.db") as conn:
        blob_hash = conn.execute("SELECT blob_hash FROM raw_sessions LIMIT 1").fetchone()[0]
        conn.execute(
            "INSERT INTO excised_content (removed_hash, reason, actor, excised_at_ms) VALUES (?, 'test', 'test', 1)",
            (blob_hash,),
        )
        conn.commit()

    with pytest.raises(AssertionError, match="authority/.*excised_content"):
        assert_canonical_snapshots_equal(
            capture_canonical_snapshot(canonical.root), capture_canonical_snapshot(mutated.root)
        )


def test_index_raw_revision_head_mutation_is_red(tmp_path: Path) -> None:
    canonical = _build_archive(tmp_path / "canonical", rich_convergence_pathology())
    mutated = _build_archive(tmp_path / "mutated", rich_convergence_pathology())
    _add_revision_head(mutated.root)

    with pytest.raises(AssertionError, match="authority/.*raw_revision_heads"):
        assert_canonical_snapshots_equal(
            capture_canonical_snapshot(canonical.root), capture_canonical_snapshot(mutated.root)
        )


def test_snapshot_covers_semantic_archive_and_public_read_surfaces(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path / "archive", rich_convergence_pathology())
    snapshot = capture_canonical_snapshot(archive.root, search_queries=("fixture",))

    assert ("index", "sessions") in _relation_keys(snapshot.canonical_rows)
    assert ("index", "messages") in _relation_keys(snapshot.canonical_rows)
    assert ("index", "blocks") in _relation_keys(snapshot.canonical_rows)
    assert ("index", "web_content_constructs") in _relation_keys(snapshot.canonical_rows)
    assert ("index", "session_events") in _relation_keys(snapshot.provenance)
    assert ("source", "raw_sessions") in _relation_keys(snapshot.authority)
    assert ("source", "excised_content") in _relation_keys(snapshot.authority)
    assert ("source", "raw_authority_verdicts") in _relation_keys(snapshot.authority)
    assert ("index", "raw_revision_heads") in _relation_keys(snapshot.authority)
    assert ("source", "raw_revision_heads") not in _relation_keys(snapshot.authority)
    assert ("user", "assertions") in _relation_keys(snapshot.user_state)
    assert ("user", "context_deliveries") in _relation_keys(snapshot.user_state)
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
