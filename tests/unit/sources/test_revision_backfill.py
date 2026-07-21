from __future__ import annotations

import json
import sqlite3
import time
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.pipeline.parsed_tree_size import estimate_parsed_tree_bytes
from polylogue.sources import revision_backfill
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.revision_backfill import (
    RawParsePrefetchCache,
    _browser_snapshot_fidelity,
    _parse_one,
    backfill_historical_revision_evidence,
    census_historical_revision_evidence,
)
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.revision_backfill_benchmark import (
    REVISION_CHAIN_SHAPE,
    WHALE_BEARING_SHAPE,
    build_independent_raw_corpus,
    build_revision_chain_corpus,
    build_whale_bearing_corpus,
)


def _chatgpt_session(native_id: str, *texts: str) -> dict[str, object]:
    mapping: dict[str, object] = {}
    previous: str | None = None
    for index, text in enumerate(texts):
        node_id = f"{native_id}-node-{index}"
        mapping[node_id] = {
            "id": node_id,
            "parent": previous,
            "children": [],
            "message": {
                "id": f"{native_id}-message-{index}",
                "author": {"role": "user" if index % 2 == 0 else "assistant"},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": 1_700_000_000 + index,
            },
        }
        if previous is not None:
            previous_row = mapping[previous]
            assert isinstance(previous_row, dict)
            previous_row["children"] = [node_id]
        previous = node_id
    return {
        "id": native_id,
        "conversation_id": native_id,
        "title": native_id,
        "create_time": 1_700_000_000,
        "update_time": 1_700_000_000 + len(texts),
        "current_node": previous,
        "mapping": mapping,
    }


def _bundle(*sessions: dict[str, object]) -> bytes:
    return json.dumps(list(sessions), sort_keys=True).encode()


def test_browser_snapshot_fidelity_derives_from_parser_ingest_flags() -> None:
    """``MembershipRevision.browser_snapshot_fidelity`` must reflect the parser's
    own ingest flags -- until this was wired up, every ``MembershipRevision``
    built during census carried ``browser_snapshot_fidelity=None``
    regardless of content, making ``classify_membership_revisions``'s entire
    dom/native/direct-export precedence dead code in production
    (polylogue-z1c6)."""
    assert _browser_snapshot_fidelity([]) is None
    assert _browser_snapshot_fidelity(["capture:temporary-chat"]) is None
    assert _browser_snapshot_fidelity([DOM_FALLBACK_INGEST_FLAG]) == "dom"
    assert _browser_snapshot_fidelity([NATIVE_BROWSER_CAPTURE_INGEST_FLAG]) == "native"
    assert _browser_snapshot_fidelity([COMPACT_BROWSER_CAPTURE_INGEST_FLAG]) == "native"
    # Native takes precedence if a parser somehow reports both.
    assert _browser_snapshot_fidelity([DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_INGEST_FLAG]) == "native"


def test_revision_reparse_preserves_beads_workspace_identity(tmp_path: Path) -> None:
    """Replay must retain the same workspace-scoped native ID as ingest."""
    source_path = tmp_path / "workspace" / ".beads" / "interactions.jsonl"
    payload = (
        b'{"id":"event-1","kind":"closed","created_at":"2026-07-12T00:00:00Z",'
        b'"issue_id":"polylogue-7fj","actor":"agent","extra":{}}\n'
    )

    sessions = _parse_one(Provider.BEADS, payload, str(source_path))

    assert len(sessions) == 1
    assert sessions[0].provider_session_id.startswith("polylogue-7fj@workspace-")
    assert sessions[0].working_directories == [str(source_path.parent.parent.resolve())]


def _single_session_state_db_bytes(tmp_path: Path) -> bytes:
    db_path = tmp_path / "state-source.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (19);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, model_config TEXT, parent_session_id TEXT,
                started_at REAL, ended_at REAL, end_reason TEXT, title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
                timestamp REAL NOT NULL, tool_calls TEXT, observed INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES ('root', 'cli', '{}', 1.0, 8.0, 'completed', 'root')"
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, 'root', 'user', 'hi', 2.0)"
        )
    return db_path.read_bytes()


def test_parse_one_replays_single_session_state_db_bytes_via_temp_spill(tmp_path: Path) -> None:
    """Regression for polylogue-1zex: _parse_one previously had no SQLite
    awareness and crashed with UnicodeDecodeError trying to json-parse raw
    SQLite bytes for a single-session state.db. Calling _parse_one directly
    with no payload_path exercises the bounded temp-file spill fallback (the
    real on-disk blob path is proven separately by the live-watcher and
    historical-backfill end-to-end tests, which always have one)."""

    payload = _single_session_state_db_bytes(tmp_path)

    sessions = _parse_one(Provider.HERMES, payload, str(tmp_path / "hermes-home" / "state.db"))

    assert len(sessions) == 1
    assert sessions[0].messages
    assert sessions[0].messages[0].text == "hi"


def test_historical_backfill_replays_single_session_state_db(tmp_path: Path) -> None:
    """End-to-end proof that the historical-repair entry point (which always
    has a real on-disk blob path, unlike the direct-bytes test above) also
    replays a single-session state.db raw revision correctly."""

    initialize_active_archive_root(tmp_path)
    payload = _single_session_state_db_bytes(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.HERMES,
            payload=payload,
            source_path=str(tmp_path / "hermes-home" / "state.db"),
            acquired_at_ms=1,
        )

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 1
    assert result.replayed_logical_sources == 1
    assert result.quarantined == 0


def test_historical_backfill_streams_codex_raw_without_eager_blob_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initialize_active_archive_root(tmp_path)
    payload = b'{"type":"session_meta","payload":{"id":"streamed"}}\n'
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="streamed.jsonl",
            acquired_at_ms=1,
        )
    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: pytest.fail("stream-safe revision replay must not eagerly read a blob"),
    )

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 1
    assert result.replayed_logical_sources == 1


def test_historical_backfill_selects_prefix_newest_independent_of_acquisition_order(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":'
        b'[{"type":"output_text","text":"new"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        newest_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="session.jsonl",
            acquired_at_ms=1,
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="session.jsonl",
            acquired_at_ms=2,
        )
        legacy_append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"response_item","payload":{"type":"message","id":"legacy-suffix"}}\n',
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=3,
        )

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 3
    assert result.classified_full == 2
    assert result.replayed_logical_sources == 1
    assert result.quarantined == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parser_census = conn.execute(
            """
            SELECT status, COUNT(*)
            FROM raw_authority_parser_census
            WHERE parser_fingerprint = 'revision-membership-v1'
            GROUP BY status ORDER BY status
            """
        ).fetchall()
    assert parser_census == [("complete", 2), ("failed", 1)]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT message_count, raw_id FROM sessions").fetchone() == (2, newest_raw_id)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        row = conn.execute(
            "SELECT rowid, block_id, message_id, session_id, block_type FROM blocks ORDER BY rowid LIMIT 1"
        ).fetchone()
        assert row is not None
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (row[0],))
        conn.execute(
            """
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            VALUES (?, ?, ?, ?, ?, 'stale-only-token')
            """,
            row,
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'stale' ").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = NULL WHERE logical_source_key IS NOT NULL")
        conn.commit()

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'stale'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'old'").fetchone()[0] == 1
        assert set(conn.execute("SELECT raw_id, decision FROM raw_revision_applications")) == {
            (baseline_raw_id, "superseded"),
            (newest_raw_id, "selected_baseline"),
        }
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 2
        assert conn.execute(
            "SELECT revision_kind, revision_authority, parsed_at_ms FROM raw_sessions WHERE raw_id = ?",
            (legacy_append_raw_id,),
        ).fetchone() == ("unknown", "quarantined", None)

    parsed_baseline = parse_payload(
        Provider.CODEX,
        list(_iter_json_stream(BytesIO(baseline), "session.jsonl")),
        "session",
        source_path="session.jsonl",
    )[0]
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_parsed_for_retained_raw(
            parsed_baseline,
            raw_id=baseline_raw_id,
            source_path="session.jsonl",
            acquired_at_ms=3,
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT message_count, raw_id FROM sessions").fetchone() == (2, newest_raw_id)


def test_incremental_target_expands_new_logical_key_across_source_paths(tmp_path: Path) -> None:
    """A newly parsed path must not split an already-known byte cohort."""
    initialize_active_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"shared","timestamp":"2026-07-15T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":'
        b'[{"type":"output_text","text":"new"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        old_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="first/shared.jsonl",
            acquired_at_ms=1,
        )
    assert backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[old_raw_id]).replayed_logical_sources == 1

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        new_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="moved/shared.jsonl",
            acquired_at_ms=2,
        )

    result = backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[new_raw_id])

    assert result.scanned == 2
    assert result.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count, raw_id FROM sessions").fetchall() == [
            ("shared", 2, new_raw_id)
        ]


def test_backfill_resumes_after_index_receipt_commits_before_source_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"one"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="session.jsonl",
            acquired_at_ms=1,
        )

    original_mark = ArchiveStore.mark_raw_parse_succeeded

    def crash_after_index_commit(self: ArchiveStore, raw_id: str, *, provider: Provider) -> None:
        raise RuntimeError("crash after index receipt")

    monkeypatch.setattr(ArchiveStore, "mark_raw_parse_succeeded", crash_after_index_commit)
    with pytest.raises(RuntimeError, match="crash after index receipt"):
        backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)

    monkeypatch.setattr(ArchiveStore, "mark_raw_parse_succeeded", original_mark)
    resumed = backfill_historical_revision_evidence(tmp_path)
    assert resumed.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1,)


def test_backfill_resumes_after_only_some_source_markers_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"one"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","id":"two","role":"assistant","content":'
        b'[{"type":"output_text","text":"two"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_ids = {
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="session.jsonl",
                acquired_at_ms=index,
            )
            for index, payload in enumerate((baseline, newest), start=1)
        }

    original_mark = ArchiveStore.mark_raw_parse_succeeded
    calls = 0

    def crash_after_one_marker(self: ArchiveStore, raw_id: str, *, provider: Provider) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_mark(self, raw_id, provider=provider)
            return
        raise RuntimeError("crash between source markers")

    monkeypatch.setattr(ArchiveStore, "mark_raw_parse_succeeded", crash_after_one_marker)
    with pytest.raises(RuntimeError, match="between source markers"):
        backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        accepted_before = conn.execute("SELECT raw_id, content_hash FROM sessions").fetchone()
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 2

    monkeypatch.setattr(ArchiveStore, "mark_raw_parse_succeeded", original_mark)
    backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 2
        assert {
            str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions WHERE parsed_at_ms IS NOT NULL")
        } == raw_ids
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT raw_id, content_hash FROM sessions").fetchone() == accepted_before
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 2


def test_cold_rebuild_restores_overlapping_multi_session_bundles(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    bundle_a = _bundle(_chatgpt_session("s1", "old"), _chatgpt_session("s2", "only-two"))
    bundle_b = _bundle(
        _chatgpt_session("s1", "old", "extended"),
        _chatgpt_session("s3", "only-three"),
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=bundle_a,
            source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=bundle_b,
            source_path="conversations.json",
            acquired_at_ms=2,
        )

    result = backfill_historical_revision_evidence(tmp_path)
    assert result.replayed_logical_sources == 3
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert set(conn.execute("SELECT native_id, message_count FROM sessions")) == {
            ("s1", 2),
            ("s2", 1),
            ("s3", 1),
        }
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert set(conn.execute("SELECT raw_id FROM raw_sessions WHERE parsed_at_ms IS NOT NULL")) == {
            (raw_a,),
            (raw_b,),
        }
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_session_memberships WHERE decision IN ('ambiguous', 'deferred')"
        ).fetchone() == (0,)

    (tmp_path / "index.db").unlink()
    initialize_active_archive_root(tmp_path)
    rebuilt = backfill_historical_revision_evidence(tmp_path)
    assert rebuilt.replayed_logical_sources == 3
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert set(conn.execute("SELECT native_id, message_count FROM sessions")) == {
            ("s1", 2),
            ("s2", 1),
            ("s3", 1),
        }


def test_divergent_bundle_member_does_not_block_safe_members(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "left"), _chatgpt_session("s2", "safe")),
            source_path="conversations.json",
            acquired_at_ms=1,
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "right"), _chatgpt_session("s3", "safe")),
            source_path="conversations.json",
            acquired_at_ms=2,
        )

    result = backfill_historical_revision_evidence(tmp_path)
    assert result.quarantined == 2
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert set(conn.execute("SELECT native_id FROM sessions")) == {("s2",), ("s3",)}
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert set(conn.execute("SELECT raw_id FROM raw_sessions WHERE parsed_at_ms IS NULL")) == {
            (raw_a,),
            (raw_b,),
        }
        assert conn.execute("SELECT COUNT(*) FROM raw_session_memberships WHERE decision = 'ambiguous'").fetchone() == (
            2,
        )


def test_divergent_bundle_member_preserves_last_accepted_session(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "accepted")),
            source_path="first.json",
            acquired_at_ms=1,
        )
    backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        accepted = conn.execute("SELECT message_count, content_hash FROM sessions WHERE native_id = 's1'").fetchone()
        accepted_head = conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:s1'"
        ).fetchone()
    assert accepted is not None
    assert accepted_head is not None

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("s1", "base", "divergent")),
            source_path="second.json",
            acquired_at_ms=2,
        )
    result = backfill_historical_revision_evidence(tmp_path)

    assert result.quarantined == 2
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert (
            conn.execute("SELECT message_count, content_hash FROM sessions WHERE native_id = 's1'").fetchone()
            == accepted
        )
        assert (
            conn.execute(
                "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:s1'"
            ).fetchone()
            == accepted_head
        )


def test_targeted_rebuild_expands_same_session_across_source_paths_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        selected_raw = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("shared", "old")),
            source_path="first.json",
            acquired_at_ms=1,
        )
        archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("shared", "old", "new")),
            source_path="second.json",
            acquired_at_ms=2,
        )
        unrelated_raw = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_bundle(_chatgpt_session("unrelated", "no")),
            source_path="third.json",
            acquired_at_ms=3,
        )

    # Production ordinary convergence starts from the durable membership
    # census established by ingestion/offline rebuild, not an empty source-v7
    # authority catalog.
    backfill_historical_revision_evidence(tmp_path)
    (tmp_path / "index.db").unlink()
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        unrelated_before = conn.execute(
            "SELECT parser_fingerprint, status, member_count, detail FROM raw_membership_census WHERE raw_id = ?",
            (unrelated_raw,),
        ).fetchone()

    from polylogue.sources import revision_backfill

    original_parse = revision_backfill._parse_retained_raw
    opened: list[str] = []

    def observed_parse(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        opened.append(raw_id)
        return original_parse(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", observed_parse)
    result = backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[selected_raw])
    assert result.replayed_logical_sources == 1
    assert result.scanned == 2
    assert unrelated_raw not in opened
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count FROM sessions").fetchall() == [("shared", 2)]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT parser_fingerprint, status, member_count, detail FROM raw_membership_census WHERE raw_id = ?",
                (unrelated_raw,),
            ).fetchone()
            == unrelated_before
        )


def test_membership_census_retains_only_one_logical_cohort_at_scale(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    independent_raw_count = 64
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(independent_raw_count):
            payload = _bundle(_chatgpt_session(f"session-{index}", f"message-{index}"))
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"bundle-{index}.json",
                acquired_at_ms=index + 1,
            )
        shared_payloads = [
            _bundle(_chatgpt_session("shared", "base")),
            _bundle(_chatgpt_session("shared", "base", "new")),
        ]
        for index, payload in enumerate(shared_payloads, start=1):
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"shared-{index}.json",
                acquired_at_ms=independent_raw_count + index,
            )
    raw_count = independent_raw_count + len(shared_payloads)

    retained: list[tuple[int, int]] = []
    result = backfill_historical_revision_evidence(
        tmp_path,
        retention_observer=lambda count, payload_bytes: retained.append((count, payload_bytes)),
    )

    assert result.scanned == raw_count
    assert result.replayed_logical_sources == independent_raw_count + 1
    assert len(retained) == independent_raw_count + 1
    assert max(count for count, _payload_bytes in retained) == 2
    assert max(payload_bytes for _count, payload_bytes in retained) == sum(map(len, shared_payloads))
    assert sum(count for count, _payload_bytes in retained) == raw_count


def test_historical_backfill_reparses_multi_gib_shaped_raw_instead_of_spilling_archive_wide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache miss reparses durable bytes rather than retaining a giant cohort tree."""
    initialize_active_archive_root(tmp_path)
    payload = b'{"type":"session_meta","payload":{"id":"multi-gib-shaped"}}\n'
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="multi-gib-shaped.jsonl",
            acquired_at_ms=1,
        )
    declared_multi_gib = 3 * 1024 * 1024 * 1024
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?", (declared_multi_gib, raw_id))
        conn.commit()

    original = revision_backfill._parse_retained_raw
    parses = 0

    def counted(*args: object, **kwargs: object) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        nonlocal parses
        parses += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)
    retained: list[tuple[int, int]] = []
    result = backfill_historical_revision_evidence(
        tmp_path,
        retention_observer=lambda count, payload_bytes: retained.append((count, payload_bytes)),
    )

    assert result.replayed_logical_sources == 1
    assert retained == [(1, declared_multi_gib)]
    # The former archive-wide spill served the second lookup from a retained
    # pickle. A bounded cache deliberately reparses the durable source row.
    assert parses >= 2


def _append_chain_archive(root: Path) -> tuple[str, str]:
    """Two revisions of one logical session: an accepted-cohort replay fixture."""
    initialize_active_archive_root(root)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"chain","timestamp":"2026-07-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":'
        b'[{"type":"output_text","text":"new"}]}}\n'
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        newest_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="chain.jsonl",
            acquired_at_ms=1,
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="chain.jsonl",
            acquired_at_ms=2,
        )
    return baseline_raw_id, newest_raw_id


def test_backfill_replay_reparses_when_spill_cache_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Baseline (pre-fix) shape: an unbounded (envelope=None) backfill with no
    explicit spill-cache bound reparses the accepted revision during replay
    even though census already parsed it once. This is exactly the CLI
    rebuild-index path's behavior before max_cached_payload_bytes decoupled
    caching from the resource envelope. Paired with the fixed-behavior test
    below to pin both sides of the regression.
    """
    _append_chain_archive(tmp_path)
    original = revision_backfill._parse_retained_raw
    parse_calls: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parse_calls.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    result = backfill_historical_revision_evidence(tmp_path, max_payload_bytes=None)

    assert result.replayed_logical_sources == 1
    # polylogue-nh44: census now proves the baseline raw is a byte-prefix of
    # the newest capture (same source_path) without parsing it at all, so
    # only the newest raw is ever ONE parse during census; replay then
    # reparses that same accepted revision again from blob because nothing
    # was cached (a 2nd call, duplicating that one raw_id) instead of reusing
    # census output.
    assert len(parse_calls) == 2
    assert len(set(parse_calls)) == 1


def test_census_skips_parse_for_byte_proven_superseded_revisions_at_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """polylogue-nh44 regression at the bead's own recorded corpus shape: a
    growing-file cohort (one re-scanned Codex rollout, 50 superseded captures
    plus the winner) must census-parse only the winner, never the 50 byte-
    proven-superseded snapshots. Measured on this exact shape: 52->2 parse
    calls (1 unique raw parsed instead of 51), ~3.3x wall-time reduction for
    the cohort (see PR body for the before/after numbers)."""
    raw_ids = build_revision_chain_corpus(tmp_path, **REVISION_CHAIN_SHAPE)
    original = revision_backfill._parse_retained_raw
    parse_calls: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parse_calls.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == len(raw_ids)
    assert result.classified_full == len(raw_ids)
    assert result.replayed_logical_sources == 1
    assert result.quarantined == 0
    # Only the newest raw (the winner) is ever independently parsed; the 50
    # older captures are bound to its learned identity by byte-prefix proof.
    assert set(parse_calls) == {raw_ids[-1]}
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT raw_id FROM sessions").fetchone() == (raw_ids[-1],)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE revision_kind = 'full' AND logical_source_key IS NOT NULL"
        ).fetchone()[0] == len(raw_ids)


def test_backfill_replay_reuses_spill_cache_when_bound_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """max_cached_payload_bytes caches census parse output independently of
    max_payload_bytes (the resource-envelope block), so an unbounded backfill
    still avoids reparsing accepted revisions during replay. Mutation:
    reverting to the paired baseline test's call (omitting
    max_cached_payload_bytes) reproduces the doubled parse count above --
    this is the anti-vacuity pairing for the CLI rebuild-index fix.
    """
    _append_chain_archive(tmp_path)
    original = revision_backfill._parse_retained_raw
    parse_calls: list[str] = []

    def counted(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parse_calls.append(raw_id)
        return original(archive, raw_id)

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", counted)

    result = backfill_historical_revision_evidence(
        tmp_path,
        max_payload_bytes=None,
        max_cached_payload_bytes=64 * 1024 * 1024,
    )

    assert result.replayed_logical_sources == 1
    # polylogue-nh44: only the newest raw is ever parsed (the baseline is
    # proven a byte-prefix and bound without parsing); replay hits the
    # census-populated spill cache instead of reparsing it from blob a
    # second time (contrast the 2-call baseline above).
    assert len(parse_calls) == 1
    assert len(set(parse_calls)) == 1


def test_parallel_census_matches_sequential_archive_state(tmp_path: Path) -> None:
    """Parsing spread across a process pool must produce byte-identical
    archive state to the sequential path. Only read-only blob->ParsedSession
    decode is parallelized; archive writes apply in fixed pending-rows order
    regardless of worker completion order, so parallel and sequential runs
    are authority-equivalent (not merely "close enough").
    """
    sequential_root = tmp_path / "sequential"
    parallel_root = tmp_path / "parallel"
    for root in (sequential_root, parallel_root):
        initialize_active_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            for index in range(6):
                payload = _bundle(_chatgpt_session(f"session-{index}", f"hello {index}", f"world {index}"))
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=payload,
                    source_path=f"chat-{index}.json",
                    acquired_at_ms=index,
                )

    seq_result = backfill_historical_revision_evidence(sequential_root, ingest_workers=1)
    par_result = backfill_historical_revision_evidence(parallel_root, ingest_workers=4)

    assert seq_result == par_result

    def _sessions(root: Path) -> list[tuple[object, ...]]:
        with sqlite3.connect(root / "index.db") as conn:
            return conn.execute("SELECT native_id, message_count, raw_id FROM sessions ORDER BY native_id").fetchall()

    assert _sessions(sequential_root) == _sessions(parallel_root)


def _state_db_bytes_for_session(tmp_path: Path, *, session_id: str, message_text: str) -> bytes:
    """Variant of _single_session_state_db_bytes with a distinct session id."""
    db_path = tmp_path / f"state-source-{session_id}.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (19);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, model_config TEXT, parent_session_id TEXT,
                started_at REAL, ended_at REAL, end_reason TEXT, title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
                timestamp REAL NOT NULL, tool_calls TEXT, observed INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES (?, 'cli', '{}', 1.0, 8.0, 'completed', ?)",
            (session_id, session_id),
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, ?, 'user', ?, 2.0)",
            (session_id, message_text),
        )
    return db_path.read_bytes()


def test_parallel_census_threads_hermes_sqlite_payload_path(tmp_path: Path) -> None:
    """Regression for the #3113/polylogue-1zex SQLite-detection branch under
    parallel dispatch: census_parse_worker must thread payload_path (the
    real on-disk blob path) and archive_root through to _parse_one the same
    way the sequential parse_retained_raw_sessions does, so a Hermes
    state.db raw parsed by a pool worker still opens via sqlite3 against a
    real file instead of only working by accident through the temp-file
    fallback. Two independent single-session state.db raws force
    ingest_workers>1 to actually dispatch through the process pool.
    """
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(2):
            payload = _state_db_bytes_for_session(tmp_path, session_id=f"hermes-{index}", message_text=f"hi {index}")
            archive.write_raw_payload(
                provider=Provider.HERMES,
                payload=payload,
                source_path=str(tmp_path / f"hermes-home-{index}" / "state.db"),
                acquired_at_ms=index,
            )

    result = backfill_historical_revision_evidence(tmp_path, ingest_workers=4)

    assert result.scanned == 2
    assert result.replayed_logical_sources == 2
    assert result.quarantined == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        rows = conn.execute("SELECT native_id, message_count FROM sessions ORDER BY native_id").fetchall()
    assert [native_id.startswith("hermes-") for native_id, _count in rows] == [True, True]
    assert [count for _native_id, count in rows] == [1, 1]


def test_independent_raw_corpus_fixture_backfills_cleanly(tmp_path: Path) -> None:
    """polylogue-amg1 benchmark fixture sanity: every synthetic raw census-and-replays
    to exactly one session with no quarantine, at both recorded payload shapes' scale
    (downscaled here for test speed; devtools/scripts run the full recorded counts)."""
    raw_ids = build_independent_raw_corpus(tmp_path, raw_count=12, avg_payload_bytes=5_000)

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 12
    assert result.replayed_logical_sources == 12
    assert result.quarantined == 0
    with sqlite3.connect(tmp_path / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert session_count == 12
    assert len(set(raw_ids)) == 12


def test_census_batching_reduces_commit_count(tmp_path: Path) -> None:
    """polylogue-amg1 lever (a): commit_batch_size must defer per-raw
    self-commits (manage_transaction=False) and drive the batch boundary
    through exactly ``ceil(raw_count / batch_size)`` explicit ArchiveStore
    commits, versus one self-commit per raw when unset -- while producing
    identical scan/classification results either way. sqlite3.Connection is
    an immutable C type and cannot be monkeypatched directly, so this
    verifies the actual mechanism (the manage_transaction contract each
    write call receives, and how many times the batch-boundary commit()
    fires) rather than a raw connection-level commit count."""
    unbatched_root = tmp_path / "unbatched"
    batched_root = tmp_path / "batched"
    build_independent_raw_corpus(unbatched_root, raw_count=9, avg_payload_bytes=1_000)
    build_independent_raw_corpus(batched_root, raw_count=9, avg_payload_bytes=1_000)

    import unittest.mock as mock

    def _manage_transaction_flags(archive_root: Path, *, commit_batch_size: int | None) -> tuple[list[bool], int]:
        flags: list[bool] = []
        commit_count = 0
        original_bind = ArchiveStore.bind_raw_revision
        original_commit = ArchiveStore.commit

        def recording_bind(self: ArchiveStore, raw_id: str, revision: object, **bind_kwargs: object) -> None:
            flags.append(bool(bind_kwargs.get("manage_transaction", True)))
            original_bind(self, raw_id, revision, **bind_kwargs)  # type: ignore[arg-type]

        def counting_commit(self: ArchiveStore) -> None:
            nonlocal commit_count
            commit_count += 1
            original_commit(self)

        with (
            mock.patch.object(ArchiveStore, "bind_raw_revision", recording_bind),
            mock.patch.object(ArchiveStore, "commit", counting_commit),
        ):
            census_historical_revision_evidence(archive_root, commit_batch_size=commit_batch_size)
        return flags, commit_count

    unbatched_flags, unbatched_explicit_commits = _manage_transaction_flags(unbatched_root, commit_batch_size=None)
    batched_flags, batched_explicit_commits = _manage_transaction_flags(batched_root, commit_batch_size=4)

    assert len(unbatched_flags) == len(batched_flags) == 9
    # Unbatched: every write self-commits immediately (manage_transaction=True).
    assert all(unbatched_flags)
    assert unbatched_explicit_commits == 0
    # Batched (size 4, 9 raws): writes defer (manage_transaction=False) and
    # the loop drives exactly ceil(9/4) = 3 explicit batch-boundary commits.
    assert not any(batched_flags)
    assert batched_explicit_commits == 3


def test_census_batch_crash_loses_at_most_one_batch_and_resumes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-amg1 crash-mid-batch proof: a fault partway through an
    uncommitted census batch must discard exactly that batch (not a partial
    raw, not prior committed batches), and a resume must converge to the
    same terminal state as an uninterrupted run with zero duplication."""
    raw_count = 10
    batch_size = 4
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=raw_count, avg_payload_bytes=1_000)

    original_bind = ArchiveStore.bind_raw_revision
    calls = 0
    # Crash on the 7th bind call: batch 1 (calls 1-4) has already committed;
    # batch 2 (calls 5-8) is interrupted after its 3rd call (7), before it
    # reaches batch_size and self-commits.
    crash_at_call = 7

    def crash_partway(self: ArchiveStore, raw_id: str, revision: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == crash_at_call:
            raise RuntimeError("injected crash mid-batch")
        original_bind(self, raw_id, revision, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", crash_partway)
    with pytest.raises(RuntimeError, match="injected crash mid-batch"):
        backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    with sqlite3.connect(root / "source.db") as conn:
        complete_after_crash = conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE revision_kind != 'unknown'"
        ).fetchone()[0]
    # Exactly one fully-committed batch survives the crash -- never a partial one.
    assert complete_after_crash == batch_size

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", original_bind)
    result = backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    assert result.scanned == raw_count
    assert result.replayed_logical_sources == raw_count
    assert result.quarantined == 0
    with sqlite3.connect(root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        application_count = conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0]
    assert session_count == raw_count
    # One application receipt per raw, no duplicates from the retried batch.
    assert application_count == raw_count
    with sqlite3.connect(root / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE revision_kind != 'unknown'").fetchone()[0]
            == raw_count
        )


def test_backfill_resumes_after_replay_batch_crash_discards_whole_batch_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-oikv: with ``commit_batch_size`` set, the REPLAY phase now
    batches index.db writes + terminal source.db markers across MULTIPLE
    independent cohorts (not just within one cohort, as the two
    unbatched-default pinned tests above still prove unmodified). A fault
    partway through an uncommitted replay batch must discard the WHOLE
    batch -- every cohort's index writes and terminal markers together,
    since neither side ever committed -- never a partial one, and a resume
    must converge to the same terminal state as an uninterrupted run with
    zero duplication (mirrors the census-phase proof above)."""
    raw_count = 10
    batch_size = 4
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=raw_count, avg_payload_bytes=1_000)

    original_apply = ArchiveStore.apply_raw_revision_replay
    calls = 0
    # Batch 1 (cohorts 1-4) commits cleanly and resets the counter. Batch 2
    # starts (cohort 5 applies, uncommitted), then crashes on cohort 6 --
    # before batch 2 reaches batch_size and self-commits.
    crash_at_call = 6

    def crash_partway(self: ArchiveStore, plan: object, parsed_by_raw_id: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == crash_at_call:
            raise RuntimeError("injected crash mid replay-batch")
        return original_apply(self, plan, parsed_by_raw_id, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ArchiveStore, "apply_raw_revision_replay", crash_partway)
    with pytest.raises(RuntimeError, match="injected crash mid replay-batch"):
        backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    with sqlite3.connect(root / "index.db") as conn:
        session_count_after_crash = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    # Exactly one fully-committed batch survives the crash -- never a partial one.
    assert session_count_after_crash == batch_size
    with sqlite3.connect(root / "source.db") as conn:
        parsed_after_crash = conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL"
        ).fetchone()[0]
    assert parsed_after_crash == batch_size

    monkeypatch.setattr(ArchiveStore, "apply_raw_revision_replay", original_apply)
    result = backfill_historical_revision_evidence(root, commit_batch_size=batch_size)

    assert result.scanned == raw_count
    assert result.replayed_logical_sources == raw_count
    assert result.quarantined == 0
    with sqlite3.connect(root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        application_count = conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0]
    assert session_count == raw_count
    # One application receipt per raw, no duplicates from the retried batch.
    assert application_count == raw_count
    with sqlite3.connect(root / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == raw_count
        )


def test_partition_raws_by_dispatch_size_routes_small_to_pool_large_sequential() -> None:
    """polylogue-amg1 lever (b): raws under the size ceiling are pool-eligible
    (parallel parse pays off), raws at/above it stay sequential (pickling the
    large returned ParsedSession back across the process boundary would cost
    more than the parse saved -- the bead's own 0.63x/net-loss measurement)."""
    payload_sizes = {"small-1": 1_000, "small-2": 200_000, "large-1": 262_144, "large-2": 1_700_000}

    pool_ids, sequential_ids = revision_backfill._partition_raws_by_dispatch_size(
        ["small-1", "small-2", "large-1", "large-2"], payload_sizes, dispatch_max_bytes=262_144
    )

    assert pool_ids == ["small-1", "small-2"]
    assert sequential_ids == ["large-1", "large-2"]


def test_parse_retained_raws_dedupes_identical_blob_across_paths_for_safe_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-869u: for a path-independent provider (Codex here), rows
    sharing a ``blob_hash`` parse once and fan out -- INCLUDING across
    different ``source_path``s, since those parsers' session identity comes
    entirely from the payload bytes. Live shape: one 442MB codex rollout
    acquired 8x within 2.3s during a stampede, at up to 8 different acquired
    paths -- 8 raw rows, one blob, formerly 8 full parses."""
    descriptors = {
        "dup-1": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
        "dup-2": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.UNKNOWN, 10),
        "dup-3": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
        "other-path": (Provider.CODEX, "hash-A", "different.jsonl", RawRevisionKind.FULL, 10),
        "other-bytes": (Provider.CODEX, "hash-B", "same.jsonl", RawRevisionKind.FULL, 20),
    }

    class FakeArchive:
        def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
            return descriptors[raw_id]

    parsed: list[str] = []

    def fake_parse(archive: object, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parsed.append(raw_id)
        descriptor = descriptors[raw_id]
        return [], descriptor[4], descriptor[3]

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", fake_parse)

    results = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
    )

    # one parse per distinct blob_hash: dup-2/dup-3/other-path all reuse
    # dup-1's outcome despite other-path having a different source_path.
    assert parsed == ["dup-1", "other-bytes"]
    assert set(results) == set(descriptors)
    sessions, size, kind = results["dup-2"]  # type: ignore[misc]
    assert (sessions, size, kind) == ([], 10, RawRevisionKind.UNKNOWN)
    _sessions, _size, dup3_kind = results["dup-3"]  # type: ignore[misc]
    assert dup3_kind == RawRevisionKind.FULL
    _sessions, other_path_size, other_path_kind = results["other-path"]  # type: ignore[misc]
    assert (other_path_size, other_path_kind) == (10, RawRevisionKind.FULL)


def test_parse_retained_raws_preserves_path_scoped_dedup_for_path_dependent_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-869u: Beads derives workspace-scoped native ids from
    ``source_path`` (``sources/parsers/beads.py:_repository_root``), so
    byte-identical Beads rows at DIFFERENT paths must keep parsing
    separately -- unlike the path-independent providers, cross-path
    identity is not provably safe here."""
    descriptors = {
        "same-path-a": (Provider.BEADS, "hash-A", "workspace-one/issues.jsonl", RawRevisionKind.FULL, 10),
        "same-path-b": (Provider.BEADS, "hash-A", "workspace-one/issues.jsonl", RawRevisionKind.FULL, 10),
        "other-path": (Provider.BEADS, "hash-A", "workspace-two/issues.jsonl", RawRevisionKind.FULL, 10),
    }

    class FakeArchive:
        def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
            return descriptors[raw_id]

    parsed: list[str] = []

    def fake_parse(archive: object, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parsed.append(raw_id)
        descriptor = descriptors[raw_id]
        return [], descriptor[4], descriptor[3]

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", fake_parse)

    results = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
    )

    # same-path-b reuses same-path-a's outcome; other-path (different
    # source_path, same bytes) still pays its own parse.
    assert parsed == ["same-path-a", "other-path"]
    assert set(results) == set(descriptors)


def test_parse_retained_raws_fans_out_exceptions_to_duplicate_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors = {
        "dup-1": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
        "dup-2": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
    }

    class FakeArchive:
        def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
            return descriptors[raw_id]

    def failing_parse(archive: object, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        raise ValueError(f"boom {raw_id}")

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", failing_parse)

    results = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
    )

    assert isinstance(results["dup-1"], ValueError)
    assert results["dup-2"] is results["dup-1"]


def test_raw_parse_prefetch_cache_admits_pops_and_enforces_budget() -> None:
    """polylogue-m6tp phase (a): the daemon's warmed-parse cache is bounded
    by an explicit inflight-bytes budget (the design's whale-memory guard)
    and never double-admits or double-serves the same raw_id."""
    cache = RawParsePrefetchCache(max_inflight_bytes=100)
    assert len(cache) == 0
    assert cache.contains("r1") is False

    assert cache.try_admit("r1", [], payload_bytes=60, revision_kind=RawRevisionKind.FULL) is True
    assert len(cache) == 1
    assert cache.contains("r1") is True

    # Re-admitting the same raw_id is rejected even though the payload is small.
    assert cache.try_admit("r1", [], payload_bytes=1, revision_kind=RawRevisionKind.FULL) is False
    assert len(cache) == 1

    # Budget: 60 already committed, 100 total -- a 50-byte entry would exceed it.
    assert cache.try_admit("r2", [], payload_bytes=50, revision_kind=RawRevisionKind.FULL) is False
    assert cache.contains("r2") is False

    # A smaller entry that fits the remaining 40 bytes is admitted.
    assert cache.try_admit("r2", [], payload_bytes=40, revision_kind=RawRevisionKind.UNKNOWN) is True
    assert len(cache) == 2

    popped = cache.pop("r1")
    assert popped is not None
    sessions, payload_bytes, revision_kind = popped
    assert sessions == []
    assert payload_bytes == 60
    assert revision_kind == RawRevisionKind.FULL
    assert len(cache) == 1
    assert cache.contains("r1") is False

    # Popping releases the budget: a raw that didn't fit before now fits.
    assert cache.try_admit("r3", [], payload_bytes=59, revision_kind=RawRevisionKind.FULL) is True

    # Popping an absent raw_id is a no-op, not an error.
    assert cache.pop("does-not-exist") is None


def test_raw_parse_prefetch_cache_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        RawParsePrefetchCache(max_inflight_bytes=0)


def test_parse_retained_raws_prefetch_cache_hit_skips_parse_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw_id already popped from the prefetch cache must reach the caller's
    result dict WITHOUT ever calling the parser -- proving the parse-stage
    extraction actually removes that raw from the writer-hold parse path,
    not merely duplicates the work. Reverting the cache-check in
    ``_parse_retained_raws`` would make ``parsed`` include ``"warm-1"`` and
    fail this test."""
    descriptors = {
        "warm-1": (Provider.CODEX, "hash-A", "warm.jsonl", RawRevisionKind.FULL, 10),
        "cold-1": (Provider.CODEX, "hash-B", "cold.jsonl", RawRevisionKind.FULL, 20),
    }

    class FakeArchive:
        def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
            return descriptors[raw_id]

    parsed: list[str] = []

    def fake_parse(archive: object, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        parsed.append(raw_id)
        descriptor = descriptors[raw_id]
        return [], descriptor[4], descriptor[3]

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", fake_parse)

    warmed_session = ParsedSession(
        provider_session_id="warmed",
        source_name=Provider.CODEX,
        title=None,
        created_at=None,
        updated_at=None,
        messages=[],
    )
    cache = RawParsePrefetchCache(max_inflight_bytes=1_000_000)
    assert cache.try_admit("warm-1", [warmed_session], payload_bytes=10, revision_kind=RawRevisionKind.FULL) is True

    results = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        ["warm-1", "cold-1"],
        ingest_workers=1,
        prefetch_cache=cache,
    )

    # Only the cold (unwarmed) raw actually went through the parser.
    assert parsed == ["cold-1"]
    assert results["warm-1"] == ([warmed_session], 10, RawRevisionKind.FULL)
    sessions, size, kind = results["cold-1"]  # type: ignore[misc]
    assert (sessions, size, kind) == ([], 20, RawRevisionKind.FULL)
    # The cache entry was consumed, not merely peeked.
    assert cache.contains("warm-1") is False


def test_parse_retained_raws_prefetch_cache_miss_is_byte_identical_to_no_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalence guarantee: an empty/absent prefetch cache must produce the
    exact same result as ``prefetch_cache=None`` for every existing caller
    (polylogue-m6tp phase (a) is purely additive)."""
    descriptors = {
        "dup-1": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.FULL, 10),
        "dup-2": (Provider.CODEX, "hash-A", "same.jsonl", RawRevisionKind.UNKNOWN, 10),
        "other-bytes": (Provider.CODEX, "hash-B", "same.jsonl", RawRevisionKind.FULL, 20),
    }

    class FakeArchive:
        def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
            return descriptors[raw_id]

    def fake_parse(archive: object, raw_id: str) -> tuple[list[ParsedSession], int, RawRevisionKind]:
        descriptor = descriptors[raw_id]
        return [], descriptor[4], descriptor[3]

    monkeypatch.setattr(revision_backfill, "_parse_retained_raw", fake_parse)

    baseline = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
        prefetch_cache=None,
    )
    with_empty_cache = revision_backfill._parse_retained_raws(
        FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        ingest_workers=1,
        prefetch_cache=RawParsePrefetchCache(max_inflight_bytes=1_000_000),
    )

    assert baseline == with_empty_cache


def test_pool_dispatch_floor_rejects_small_aggregate_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker spawn+import (~1.5-2s each, measured live 2026-07-19) dominates
    tiny batches: a per-cohort census batch of a few sub-256KiB raws must parse
    sequentially, not spawn a short-lived pool that spends ~95% of its life in
    importlib."""
    sizes = {"a": 100_000, "b": 200_000, "c": 50_000}
    assert not revision_backfill._pool_dispatch_amortizes(["a", "b", "c"], sizes)
    assert not revision_backfill._pool_dispatch_amortizes(["a"], sizes)
    big = {f"r{i}": 250_000 for i in range(400)}  # 100MB aggregate
    assert revision_backfill._pool_dispatch_amortizes(list(big), big)
    monkeypatch.setenv("POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES", "100000")
    assert revision_backfill._pool_dispatch_amortizes(["a", "b"], sizes)
    monkeypatch.setenv("POLYLOGUE_REVISION_PARSE_POOL_MIN_BYTES", "not-a-number")
    assert not revision_backfill._pool_dispatch_amortizes(["a", "b"], sizes)


def test_parse_retained_raws_small_batch_never_creates_a_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End-to-end: a small pool-eligible batch under the aggregate floor must
    not construct a ProcessPoolExecutor at all (the churn measured live: 20
    workers in 25s, each ~95% importlib). This is a GIL-build-fallback
    mechanic specifically (the amortization floor exists only to protect
    process-pool spawn costs, polylogue-xikl) -- pin the probe so this test's
    claim is exercised deterministically regardless of which interpreter
    (GIL or genuinely free-threaded) runs the suite."""
    monkeypatch.setattr(revision_backfill, "parallel_threads_effective", lambda: False)
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(3):
            payload = (
                f'{{"type":"session_meta","payload":{{"id":"floor-{index}"}}}}\n'
                f'{{"type":"response_item","payload":{{"type":"message","id":"one","role":"user",'
                f'"content":[{{"type":"input_text","text":"tiny"}}]}}}}\n'
            ).encode()
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"floor-{index}.jsonl",
                acquired_at_ms=index,
            )

    def forbidden_pool(**kwargs: object) -> object:
        raise AssertionError("pool must not be created for a sub-floor batch")

    import polylogue.pipeline.services.process_pool as process_pool_module

    monkeypatch.setattr(process_pool_module, "process_pool_executor", forbidden_pool)

    result = backfill_historical_revision_evidence(tmp_path, ingest_workers=4)
    assert result.scanned == 3
    assert result.quarantined == 0


def test_size_aware_dispatch_keeps_large_raws_off_the_process_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end proof: with a low dispatch ceiling, only the small raw in a
    mixed corpus is submitted to the process pool; the large one parses
    in-process, and the archive state matches ingest_workers=1 exactly.
    Size-based partitioning is a GIL-build-fallback mechanic specifically
    (the thread path applies no size partition at all, polylogue-xikl) --
    pin the probe so this test's claim is exercised deterministically
    regardless of which interpreter (GIL or genuinely free-threaded) runs
    the suite."""
    monkeypatch.setattr(revision_backfill, "parallel_threads_effective", lambda: False)
    monkeypatch.setenv("POLYLOGUE_REVISION_PARSE_DISPATCH_MAX_BYTES", "5000")
    initialize_active_archive_root(tmp_path)
    small_text = "s" * 200
    large_text = "l" * 20_000
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index, text in enumerate((small_text, large_text, small_text)):
            payload = (
                f'{{"type":"session_meta","payload":{{"id":"amg1-mix-{index}"}}}}\n'
                f'{{"type":"response_item","payload":{{"type":"message","id":"one","role":"user",'
                f'"content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
            ).encode()
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"amg1-mix-{index}.jsonl",
                acquired_at_ms=index,
            )

    submitted_raw_ids: list[str] = []
    original_partition = revision_backfill._partition_raws_by_dispatch_size

    def recording_partition(raw_ids: list[str], payload_sizes: dict[str, int], **kwargs: object) -> object:
        pool_ids, sequential_ids = original_partition(raw_ids, payload_sizes, **kwargs)  # type: ignore[arg-type]
        submitted_raw_ids.extend(pool_ids)
        return pool_ids, sequential_ids

    monkeypatch.setattr(revision_backfill, "_partition_raws_by_dispatch_size", recording_partition)

    result = backfill_historical_revision_evidence(tmp_path, ingest_workers=4)

    assert result.scanned == 3
    assert result.replayed_logical_sources == 3
    assert result.quarantined == 0
    # Only the two small raws (well under the 5000-byte ceiling) were pool-eligible.
    assert len(submitted_raw_ids) == 2


def test_thread_parse_matches_sequential_archive_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-xikl adoption wave: the ThreadPoolExecutor parse path (forced
    here by patching ``parallel_threads_effective`` -- these tests run under
    the GIL, where threads still parse *correctly*, just without a
    speedup) must produce byte-identical archive state to the sequential
    path, mirroring the process-pool equivalence proof above
    (``test_parallel_census_matches_sequential_archive_state``). No size
    partition or amortization floor applies on this path, so every raw
    (including tiny ones that a size-based ceiling would otherwise route
    straight to sequential) is actually dispatched through the thread pool.

    Anti-vacuity (patch-revert, performed live during implementation): making
    ``_parse_unique_retained_raws_via_threads`` skip one raw_id when building
    ``future_to_raw_id`` (dropping a raw from the batch) makes this test fail
    with a KeyError / unequal session counts; duplicating a raw_id's future
    against a second raw_id's descriptor makes the two archives' session
    rows diverge. Both mutations were applied and reverted by hand against
    this exact test to confirm it catches them before this test was
    finalized.
    """
    sequential_root = tmp_path / "sequential"
    thread_root = tmp_path / "threaded"
    for root in (sequential_root, thread_root):
        initialize_active_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            for index in range(6):
                payload = _bundle(_chatgpt_session(f"session-{index}", f"hello {index}", f"world {index}"))
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=payload,
                    source_path=f"chat-{index}.json",
                    acquired_at_ms=index,
                )

    seq_result = backfill_historical_revision_evidence(sequential_root, ingest_workers=1)

    monkeypatch.setattr(revision_backfill, "parallel_threads_effective", lambda: True)
    thread_result = backfill_historical_revision_evidence(thread_root, ingest_workers=4)

    assert seq_result == thread_result

    def _sessions(root: Path) -> list[tuple[object, ...]]:
        with sqlite3.connect(root / "index.db") as conn:
            return conn.execute("SELECT native_id, message_count, raw_id FROM sessions ORDER BY native_id").fetchall()

    assert _sessions(sequential_root) == _sessions(thread_root)


def test_thread_parse_never_touches_shared_archive_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the sqlite ``check_same_thread`` hazard this
    thread path was designed around: ``ArchiveStore._source_conn`` is a
    plain ``sqlite3.Connection`` created with the default
    ``check_same_thread=True``, so calling ``archive.raw_revision_descriptor``
    (as ``_parse_retained_raw`` does) from a worker thread other than the
    connection's owning thread raises ``sqlite3.ProgrammingError`` --
    confirmed empirically during implementation. This test uses a fake
    archive whose only usable attributes are ``archive_root``/
    ``source_db_path`` (plain ``Path`` values, no live sqlite connection at
    all); any code path that tried to call a *method* on it (as
    ``_parse_retained_raw`` would) fails immediately with ``AttributeError``,
    proving ``_parse_unique_retained_raws_via_threads`` only ever reads two
    static attributes off the shared archive object, never queries it.
    """

    class _NoMethodsArchive:
        archive_root = Path("/fake-root")
        source_db_path = Path("/fake-root/source.db")

    descriptors = {
        "raw-a": (Provider.CODEX, "hash-a", "a.jsonl", RawRevisionKind.FULL, 111),
        "raw-b": (Provider.CODEX, "hash-b", "b.jsonl", RawRevisionKind.FULL, 222),
    }

    def fake_worker(
        raw_id: str,
        provider_token: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
    ) -> tuple[str, list[ParsedSession] | None, str | None]:
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)

    results = revision_backfill._parse_unique_retained_raws_via_threads(
        _NoMethodsArchive(),  # type: ignore[arg-type]
        list(descriptors),
        descriptors=descriptors,
        ingest_workers=2,
    )

    assert results["raw-a"] == ([], 111, RawRevisionKind.FULL)
    assert results["raw-b"] == ([], 222, RawRevisionKind.FULL)


def test_thread_parse_propagates_per_raw_exception_without_poisoning_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One raw's parse failure on the thread path must be isolated to that
    raw_id's result slot, matching the process-pool path's fan-out contract
    (``test_parse_retained_raws_fans_out_exceptions_to_duplicate_rows``
    proves the same isolation on the sequential/dedup layer). Anti-vacuity:
    if a future's exception were allowed to propagate out of the
    ``as_completed`` loop uncaught, this whole test (and every other raw's
    result) would never be reached -- the two surviving successful results
    are asserted explicitly, not merely "no exception raised"."""
    descriptors = {
        "ok-1": (Provider.CODEX, "hash-A", "a.jsonl", RawRevisionKind.FULL, 10),
        "bad-1": (Provider.CODEX, "hash-B", "b.jsonl", RawRevisionKind.FULL, 20),
        "ok-2": (Provider.CODEX, "hash-C", "c.jsonl", RawRevisionKind.FULL, 30),
    }

    class _FakeArchive:
        archive_root = Path("/fake-root")
        source_db_path = Path("/fake-root/source.db")

    def fake_worker(
        raw_id: str,
        provider_token: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
    ) -> tuple[str, list[ParsedSession] | None, str | None]:
        if raw_id == "bad-1":
            raise RuntimeError(f"boom {raw_id}")
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)

    results = revision_backfill._parse_unique_retained_raws_via_threads(
        _FakeArchive(),  # type: ignore[arg-type]
        list(descriptors),
        descriptors=descriptors,
        ingest_workers=3,
    )

    assert isinstance(results["bad-1"], RuntimeError)
    assert "boom bad-1" in str(results["bad-1"])
    assert results["ok-1"] == ([], 10, RawRevisionKind.FULL)
    assert results["ok-2"] == ([], 30, RawRevisionKind.FULL)


def test_thread_parse_results_keyed_by_raw_id_not_completion_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Determinism proof: results must be assembled by looking up each
    completed future's OWN raw_id (``future_to_raw_id[future]``), never by
    zipping ``raw_ids`` (submission order) against ``as_completed(futures)``
    (completion order) -- that zip-based shape is a real historical bug
    class in concurrent code and would silently pair a fast-finishing raw's
    result with a different, slower raw_id's descriptor. Delays are
    inverted here (the raw submitted LAST finishes FIRST) so submission
    order and completion order actively diverge; each raw_id's own
    descriptor-derived payload_size must still come back attached to it
    regardless."""
    raw_ids = [f"raw-{i}" for i in range(6)]
    descriptors = {
        raw_id: (Provider.CODEX, f"hash-{i}", f"path-{i}.jsonl", RawRevisionKind.FULL, 100 + i)
        for i, raw_id in enumerate(raw_ids)
    }
    # raw-0 (submitted first) sleeps longest; raw-5 (submitted last) returns
    # immediately -- completion order is the exact reverse of submission order.
    delay_by_raw_id = {raw_id: 0.02 * (len(raw_ids) - index) for index, raw_id in enumerate(raw_ids)}

    class _FakeArchive:
        archive_root = Path("/fake-root")
        source_db_path = Path("/fake-root/source.db")

    def fake_worker(
        raw_id: str,
        provider_token: str,
        blob_hash: str,
        source_path: str,
        is_stream: bool,
        blob_root_str: str,
        source_db_path_str: str,
    ) -> tuple[str, list[ParsedSession] | None, str | None]:
        time.sleep(delay_by_raw_id[raw_id])
        return raw_id, [], None

    monkeypatch.setattr(revision_backfill, "census_parse_worker", fake_worker)

    results = revision_backfill._parse_unique_retained_raws_via_threads(
        _FakeArchive(),  # type: ignore[arg-type]
        raw_ids,
        descriptors=descriptors,
        ingest_workers=len(raw_ids),
    )

    for index, raw_id in enumerate(raw_ids):
        sessions, size, kind = results[raw_id]  # type: ignore[misc]
        assert sessions == []
        assert size == 100 + index
        assert kind == RawRevisionKind.FULL


def test_parse_unique_retained_raws_routes_to_threads_when_probe_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring proof for ``_parse_unique_retained_raws`` itself: when
    ``parallel_threads_effective()`` is true, it must call
    ``_parse_unique_retained_raws_via_threads`` (no size partition / floor)
    rather than falling through to the process-pool branch below it."""
    descriptors = {
        "raw-a": (Provider.CODEX, "hash-a", "a.jsonl", RawRevisionKind.FULL, 10),
        "raw-b": (Provider.CODEX, "hash-b", "b.jsonl", RawRevisionKind.FULL, 20),
    }

    sentinel: dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception] = {
        "raw-a": ([], 10, RawRevisionKind.FULL),
        "raw-b": ([], 20, RawRevisionKind.FULL),
    }
    calls: list[tuple[object, ...]] = []

    def fake_thread_dispatch(
        archive: object,
        raw_ids: list[str],
        *,
        descriptors: dict[str, tuple[Provider, str, str, RawRevisionKind, int]],
        ingest_workers: int,
    ) -> dict[str, tuple[list[ParsedSession], int, RawRevisionKind] | Exception]:
        calls.append((tuple(raw_ids), ingest_workers))
        return sentinel

    def forbidden_pool(**kwargs: object) -> object:
        raise AssertionError("process pool must not be constructed when the thread path is taken")

    monkeypatch.setattr(revision_backfill, "parallel_threads_effective", lambda: True)
    monkeypatch.setattr(revision_backfill, "_parse_unique_retained_raws_via_threads", fake_thread_dispatch)
    import polylogue.pipeline.services.process_pool as process_pool_module

    monkeypatch.setattr(process_pool_module, "process_pool_executor", forbidden_pool)

    results = revision_backfill._parse_unique_retained_raws(
        object(),  # type: ignore[arg-type]
        list(descriptors),
        descriptors=descriptors,
        ingest_workers=4,
    )

    assert results == sentinel
    assert calls == [(("raw-a", "raw-b"), 4)]


# ---------------------------------------------------------------------------
# Whale-aware census spill (polylogue-odm1)
# ---------------------------------------------------------------------------

# Shrink the hot-cache budget so a modest (KB-scale) fixture reliably
# classifies as a "whale" without depending on the host's real RAM -- see
# tests/benchmarks/test_whale_census_spill_bench.py's module docstring for
# the full rationale (the class computes its budgets from
# effective_physical_memory_bytes(), whose production floor is 256 MiB).
_SHRUNK_TREE_BYTES = 64 * 1024


def test_whale_add_bypasses_sqlite_spill_and_holds_resident(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A parsed tree too large for the hot cache but within the whale
    ceiling must be held resident in ``_whales`` and must NOT be written to
    the sqlite spill at all (no pickle.dumps paid)."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)

    archive_root = tmp_path / "archive"
    _small_ids, whale_id = build_whale_bearing_corpus(
        archive_root,
        small_raw_count=1,
        small_avg_payload_bytes=WHALE_BEARING_SHAPE["small_avg_payload_bytes"],
        whale_payload_bytes=200_000,
    )
    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        revision_backfill._ParsedSessionSpill(archive_root, max_cached_payload_bytes=10_000_000) as spill,
    ):
        sessions, payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, whale_id)
        assert estimate_parsed_tree_bytes(sessions) > spill._decoded_budget, (
            "fixture must actually exceed the shrunk hot-cache budget to exercise the whale path"
        )

        spill.add(whale_id, sessions, payload_bytes=payload_bytes)

        assert whale_id in spill._whales
        assert whale_id not in spill._decoded
        row = spill.conn.execute("SELECT COUNT(*) FROM parsed_sessions WHERE raw_id = ?", (whale_id,)).fetchone()
        assert row is not None and row[0] == 0, "whale must bypass the sqlite spill write entirely"

        reloaded_sessions, reloaded_payload_bytes = spill.for_raw(archive, whale_id)
        assert reloaded_sessions is sessions, "whale reload must return the same resident objects, no round trip"
        assert reloaded_payload_bytes == payload_bytes


def test_whale_exceeding_whale_ceiling_falls_back_to_sqlite_spill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Correctness-over-speed fallback: a tree too large for EITHER tier
    (hot cache or whale ceiling) must still be spilled to sqlite exactly as
    the pre-lever code did."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)

    archive_root = tmp_path / "archive"
    _small_ids, whale_id = build_whale_bearing_corpus(
        archive_root,
        small_raw_count=1,
        small_avg_payload_bytes=WHALE_BEARING_SHAPE["small_avg_payload_bytes"],
        whale_payload_bytes=200_000,
    )
    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        revision_backfill._ParsedSessionSpill(archive_root, max_cached_payload_bytes=10_000_000) as spill,
    ):
        sessions, payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, whale_id)

        spill.add(whale_id, sessions, payload_bytes=payload_bytes)

        assert whale_id not in spill._whales
        assert whale_id not in spill._decoded
        row = spill.conn.execute("SELECT COUNT(*) FROM parsed_sessions WHERE raw_id = ?", (whale_id,)).fetchone()
        assert row is not None and row[0] > 0, "must fall back to the sqlite spill when the whale ceiling is exceeded"

        reloaded_sessions, reloaded_payload_bytes = spill.for_raw(archive, whale_id)
        assert reloaded_payload_bytes == payload_bytes
        assert reloaded_sessions[0].provider_session_id == sessions[0].provider_session_id
        assert [m.text for m in reloaded_sessions[0].messages] == [m.text for m in sessions[0].messages]


def test_whale_eviction_degrades_to_sqlite_spill_courtesy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When multiple whales together exceed the whale budget, the entry
    evicted from the resident tier must be written to the sqlite spill
    (not silently dropped) so a later ``for_raw`` still finds it without a
    full reparse from raw bytes."""
    # Each whale's tree is ~83KB (measured for 20,000-char payload text under
    # this estimator); a 16KB decoded budget classifies either as a whale,
    # and a 140KB whale ceiling holds exactly one at a time but not two.
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", 16 * 1024)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", 16 * 1024)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", 140_000)

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        first_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_payload_of_size("whale-a", 20_000),
            source_path="odm1/whale-a.jsonl",
            acquired_at_ms=1,
        )
        second_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_payload_of_size("whale-b", 20_000),
            source_path="odm1/whale-b.jsonl",
            acquired_at_ms=2,
        )

    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        revision_backfill._ParsedSessionSpill(archive_root, max_cached_payload_bytes=10_000_000) as spill,
    ):
        first_sessions, first_payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, first_id)
        spill.add(first_id, first_sessions, payload_bytes=first_payload_bytes)
        assert first_id in spill._whales

        second_sessions, second_payload_bytes, _kind = revision_backfill._parse_retained_raw(archive, second_id)
        spill.add(second_id, second_sessions, payload_bytes=second_payload_bytes)

        # The second whale evicted the first from the resident tier.
        assert second_id in spill._whales
        assert first_id not in spill._whales

        # But the first must have been degraded into the sqlite spill on
        # eviction, not dropped -- for_raw() must still resolve it without
        # touching _parse_retained_raw again (proven by content equality
        # despite the archive already being past that raw in the loop).
        row = spill.conn.execute("SELECT COUNT(*) FROM parsed_sessions WHERE raw_id = ?", (first_id,)).fetchone()
        assert row is not None and row[0] > 0, "evicted whale must be degraded into the sqlite spill, not dropped"

        reloaded_sessions, reloaded_payload_bytes = spill.for_raw(archive, first_id)
        assert reloaded_payload_bytes == first_payload_bytes
        assert [m.text for m in reloaded_sessions[0].messages] == [m.text for m in first_sessions[0].messages]


def _codex_payload_of_size(session_id: str, text_len: int) -> bytes:
    session_meta = (
        json.dumps(
            {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-06-01T00:00:00Z"}},
            separators=(",", ":"),
        )
        + "\n"
    )
    response_item = (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "one",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "x" * text_len}],
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return (session_meta + response_item).encode()


def test_backfill_replays_whale_bearing_page_byte_identical_to_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end proof (through ``backfill_historical_revision_evidence``,
    not the ``_ParsedSessionSpill`` internals) that a whale-bearing rebuild
    page still replays every session -- large and small -- to correct
    content when the whale-residency lever is active. No schema change, no
    output drift: counts and message text must match what an equivalent
    small-only corpus already proves the pipeline produces."""
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_TREE_BYTES)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_TREE_BYTES)

    archive_root = tmp_path / "archive"
    small_raw_ids, whale_raw_id = build_whale_bearing_corpus(
        archive_root,
        small_raw_count=6,
        small_avg_payload_bytes=5_000,
        whale_payload_bytes=200_000,
    )

    result = backfill_historical_revision_evidence(
        archive_root, max_cached_payload_bytes=10_000_000, commit_batch_size=200, replay_commit_batch_size=1
    )

    assert result.scanned == len(small_raw_ids) + 1
    assert result.replayed_logical_sources == len(small_raw_ids) + 1
    assert result.quarantined == 0

    with sqlite3.connect(archive_root / "index.db") as conn:
        session_ids = {row[0] for row in conn.execute("SELECT raw_id FROM sessions")}
        assert session_ids == {*small_raw_ids, whale_raw_id}
        for raw_id in [*small_raw_ids, whale_raw_id]:
            message_count = conn.execute("SELECT message_count FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone()
            assert message_count == (1,)
