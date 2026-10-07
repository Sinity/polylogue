"""Retained SQLite production preserves the complete parser result."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources import revision_backfill
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.retained_sqlite import collect_sqlite_sessions
from polylogue.sources.sqlite_export import logical_export_bytes
from polylogue.sources.streamed_event_payload import iter_json_value
from polylogue.storage.blob_store import BlobStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_jsonl import retained_parser_fixture
from tests.infra.retained_parser_payloads import _single_session_state_db_bytes


def _projection(session: ParsedSession) -> tuple[object, ...]:
    return (
        session.model_dump(mode="json", exclude={"messages", "session_events", "unit_accounting"}),
        [message.model_dump(mode="json") for message in session.messages],
        [
            (
                event.model_dump(mode="json", exclude={"payload"}),
                json.loads("".join(iter_json_value(event.payload, ensure_ascii=False))),
            )
            for event in session.session_events
        ],
        None if session.unit_accounting is None else session.unit_accounting.stable_binding_digest(),
    )


@pytest.mark.parametrize("provider", [Provider.HERMES, Provider.ANTIGRAVITY])
def test_retained_sqlite_preparation_streams_complete_parser_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider
) -> None:
    """Using the collecting replay or dropping events/accounting makes this red."""
    if provider is Provider.HERMES:
        payload = _single_session_state_db_bytes(tmp_path)
        source_path = tmp_path / "hermes-home" / "state.db"
    else:
        database = tmp_path / "trajectory.db"
        with sqlite3.connect(database) as connection:
            connection.executescript(
                "CREATE TABLE trajectory_meta(trajectory_id TEXT, cascade_id TEXT);"
                "CREATE TABLE steps(idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT, trajectory_id TEXT);"
                "CREATE TABLE parent_references(cascade_id TEXT, parent_id TEXT);"
                "CREATE TABLE conversation_summaries(cascade_id TEXT, title TEXT, last_modified_time TEXT);"
                "INSERT INTO trajectory_meta VALUES ('trajectory-1','cascade-1');"
                "INSERT INTO conversation_summaries VALUES ('cascade-1','Complete title','2026-01-01T00:00:00Z');"
                "INSERT INTO steps VALUES (0,'message','v1','{\"role\":\"user\",\"text\":\"retained text\"}','trajectory-1');"
                "INSERT INTO steps VALUES (1,'future_step','v2','{\"opaque\":\"retained evidence\"}','trajectory-1');"
                "INSERT INTO parent_references VALUES ('cascade-1','parent-a');"
                "INSERT INTO parent_references VALUES ('cascade-1','parent-b');"
            )
        payload = logical_export_bytes(database)
        source_path = tmp_path / "antigravity" / "trajectory.db"
    archive = tmp_path / "archive"
    bootstrap_archive_root(archive)
    blob_hash, _ = BlobStore(archive / "blob").write_from_bytes(payload)
    original_prepare = revision_backfill.prepare_retained_non_json_artifact

    def refuse_collecting(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("production SQLite preparation used collecting replay")

    monkeypatch.setattr(revision_backfill, "parse_retained_raw_sessions", refuse_collecting)
    monkeypatch.setattr(revision_backfill, "collect_sqlite_sessions", refuse_collecting)
    with retained_parser_fixture(
        root=archive,
        provider=provider,
        blob_hash=blob_hash,
        source_path=str(source_path),
        directory=tmp_path / "prepared",
        prepare=original_prepare,
    ) as (artifact, _reader):
        assert artifact.error is None
        expected = collect_sqlite_sessions(
            provider,
            BlobStore(archive / "blob").blob_path(blob_hash),
            fallback_id=source_path.stem,
            profile_identity=artifact.captured_profile_key,
        )
        expected = [normalize_session_timestamps(session) for session in expected]
        for session in expected:
            session.content_hash = session_content_hash(session)
        actual = list(artifact.iter_sessions())
        assert len(actual) == len(expected)
        for observed, wanted in zip(actual, expected, strict=True):
            for label, left, right in zip(
                ("metadata", "messages", "events", "accounting"),
                _projection(observed),
                _projection(wanted),
                strict=True,
            ):
                assert left == right, f"{label}: observed={left!r}; expected={right!r}"
        assert actual[0].session_events

        events = {event.event_type: event for event in actual[0].session_events}
        assert actual[0].messages[0].text == ("hi" if provider is Provider.HERMES else "retained text")
        if provider is Provider.HERMES:
            assert actual[0].title == "root"
            assert events["hermes_session_metadata"].payload["source"] == "cli"
            assert events["hermes_message_state"].payload["active"] is True
        else:
            assert actual[0].unit_accounting is not None
            actual[0].unit_accounting.assert_conserved()
            assert actual[0].title == "Complete title"
            assert events["antigravity_unsupported_step"].payload["payload"] == {"opaque": "retained evidence"}
            parent_payload = json.loads(
                "".join(iter_json_value(events["antigravity_parent_reference"].payload, ensure_ascii=False))
            )
            assert parent_payload["parent_provider_ids"] == ["parent-a", "parent-b"]
            assert parent_payload["references"] == [
                {"cascade_id": "cascade-1", "parent_id": "parent-a"},
                {"cascade_id": "cascade-1", "parent_id": "parent-b"},
            ]
            assert parent_payload["parent_provider_id"] is None
            assert actual[0].unit_accounting.expected == {"part": 2}
            assert [(outcome.ordinal, outcome.disposition.value) for outcome in actual[0].unit_accounting.outcomes] == [
                (0, "accepted"),
                (1, "typed_unknown"),
            ]
