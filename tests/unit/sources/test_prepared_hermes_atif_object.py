"""A single Hermes ATIF trajectory streams its steps through sealed preparation."""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any

import ijson
import pytest

import polylogue.sources.prepared_jsonl as prepared_jsonl
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import spill_member_arrays
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_payload
from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import _hermes_atif_envelope, prepare_jsonl_blob
from polylogue.sources.prepared_message_sink import SqliteSessionEventSink
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.retained_jsonl import retained_raw_fixture

_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "hermes" / "atif" / "nemo_relay_atif_v1.7_real_redacted.json"
)


def _trajectory(step_count: int = 300, *, subagents: bool = True) -> dict[str, object]:
    document: dict[str, object] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    fixture_steps = document["steps"]
    assert isinstance(fixture_steps, list)
    steps: list[object] = [
        {**fixture_steps[index % len(fixture_steps)], "step_id": index + 1} for index in range(step_count)
    ]
    steps.append("not a step object")
    steps.append({})
    document["steps"] = steps
    if subagents:
        document["subagent_trajectories"] = [
            {
                "session_id": "neutral-child",
                "agent": {"name": "Neutral agent"},
                "steps": [
                    {**fixture_steps[index % len(fixture_steps)], "step_id": index + 1} for index in range(step_count)
                ]
                + ["not a step object"],
            },
            {"session_id": document["session_id"], "steps": []},
            "not a subagent object",
            ["not", "a", "subagent"],
            {"steps": [{"source": "agent", "message": "No child identity"}]},
            {"session_id": "neutral-scalar-steps", "steps": "not a list"},
            {},
        ]
    return document


def _source(tmp_path: Path, document: dict[str, object]) -> Path:
    # A Hermes install root supplies the profile key that lets subagent
    # trajectories become child sessions.
    source = tmp_path / "hermes" / "trajectory.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(document), encoding="utf-8")
    return source


def _expected(document: dict[str, object], source: Path) -> list[ParsedSession]:
    sessions = admit_parsed_sessions_for_publication(
        parse_payload(Provider.HERMES, [document], "fallback", source_path=str(source)),
        provider=Provider.HERMES,
        source_path=str(source),
    )
    for session in sessions:
        session.content_hash = session_content_hash(session)
    return sessions


def _assert_same_publication(
    actual: list[ParsedSession], expected: list[ParsedSession], shard_path: Path, tmp_path: Path
) -> None:
    assert [session.provider_session_id for session in actual] == [session.provider_session_id for session in expected]
    for left, right in zip(actual, expected, strict=True):
        assert left.content_hash == right.content_hash
        assert left.unit_accounting == right.unit_accounting
        assert [event.model_dump(mode="json") for event in left.session_events] == [
            event.model_dump(mode="json") for event in right.session_events
        ]
        exclude = {"messages", "session_events", "attachments"}
        assert left.model_dump(mode="json", exclude=exclude) == right.model_dump(mode="json", exclude=exclude)
    expected_shard = prepare_session_shard(tmp_path / "expected", expected)
    with sqlite3.connect(expected_shard.path) as baseline, sqlite3.connect(shard_path) as prepared:
        for table in ("messages", "blocks", "shard_session"):
            assert (
                prepared.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                == baseline.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            )


def _refuse_whole_document(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("whole-document decode or parse was used")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.owned_json_records", refuse)
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_payload", refuse)


@pytest.mark.parametrize("subagents", [True, False])
def test_atif_trajectory_streams_step_events_before_eof_with_parser_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subagents: bool
) -> None:
    document = _trajectory(subagents=subagents)
    source = _source(tmp_path, document)
    expected = _expected(document, source)
    assert len(expected) == (3 if subagents else 1)
    assert len(expected[0].session_events) > 300
    if subagents:
        assert len(expected[1].session_events) > 300

    _refuse_whole_document(monkeypatch)
    decoded = 0
    first_event_after: int | None = None
    original_items = ijson.items
    original_append = SqliteSessionEventSink.append

    def tracked_items(*args: object, **kwargs: object) -> object:
        nonlocal decoded
        for item in original_items(*args, **kwargs):
            decoded += 1
            yield item

    def tracked_append(self: SqliteSessionEventSink, value: ParsedSessionEvent) -> None:
        nonlocal first_event_after
        # The first row is the document correlation event; the second is a step.
        if first_event_after is None and value.event_type != "hermes_observer_trace_correlation":
            first_event_after = decoded
        original_append(self, value)

    monkeypatch.setattr(ijson, "items", tracked_items)
    monkeypatch.setattr(SqliteSessionEventSink, "append", tracked_append)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.positive_evidence_filtered
    assert first_event_after == 1
    actual = list(artifact.iter_sessions())
    assert all(isinstance(session.session_events, SqliteSessionEventSink) for session in actual)
    assert artifact.shard_path is not None
    _assert_same_publication(actual, expected, artifact.shard_path, tmp_path)


def test_streamed_atif_trajectory_carries_the_dispatch_admission_proof(tmp_path: Path) -> None:
    """A step of a future kind beyond the classifier's witness is still a typed unknown.

    Fails if the streamed carrier reaches the writer with no conservation
    proof, or proves only the sampled prefix of the steps.
    """
    document = _trajectory(200, subagents=False)
    steps = document["steps"]
    assert isinstance(steps, list)
    steps[150] = {**steps[150], "kind": "future_step_kind"}
    source = _source(tmp_path, document)
    expected = _expected(document, source)
    assert expected[0].unit_accounting is not None
    assert expected[0].unit_accounting.outcomes

    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    actual = list(artifact.iter_sessions())
    assert [session.unit_accounting for session in actual] == [session.unit_accounting for session in expected]
    assert [
        event.payload
        for session in actual
        for event in session.session_events
        if event.event_type == "hermes_unknown_input"
    ] == [{"source_index": 1, "wire_type": "future_step_kind"}]


@pytest.mark.parametrize(
    "extra",
    [
        {"atof_version": "0.1"},
        {"polylogue_artifact": "hermes_state_db", "state_db_path": "missing.db"},
        {"polylogue_artifact": "hermes_verification_evidence_db", "verification_db_path": "missing.db"},
        {"schema_version": "not-atif"},
        {"steps": {"not": "an array"}},
    ],
)
def test_atif_probe_leaves_other_hermes_lowerings_to_the_object_parser(extra: dict[str, object]) -> None:
    document = {**_trajectory(3), **extra}
    assert _hermes_atif_envelope(BytesIO(json.dumps(document).encode())) is None


def test_atif_trajectory_corrupt_suffix_leaves_no_artifact(tmp_path: Path) -> None:
    source = _source(tmp_path, _trajectory(20))
    source.write_text(source.read_text(encoding="utf-8")[:-3] + ", {broken", encoding="utf-8")
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert list(directory.glob("*.db")) == []


@pytest.mark.parametrize("failure", ["mutation", "parser"])
def test_atif_trajectory_failure_after_spill_discards_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = _source(tmp_path, _trajectory(20))
    original_append = SqliteSessionEventSink.append
    written = 0

    def append_then_fail(self: SqliteSessionEventSink, value: ParsedSessionEvent) -> None:
        nonlocal written
        original_append(self, value)
        written += 1
        if written == 5:
            if failure == "mutation":
                source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
            else:
                raise RuntimeError("synthetic parse worker failure")

    monkeypatch.setattr(SqliteSessionEventSink, "append", append_then_fail)
    directory = tmp_path / "prepared"
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(directory),
    )
    assert written >= 5
    assert artifact.error is not None
    assert artifact.sessions_path is None
    assert artifact.deferred is (failure == "mutation")
    assert list(directory.glob("*.db")) == []


def test_retained_atif_trajectory_uses_streamed_replay_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.sources.revision_backfill as revision_backfill

    document = _trajectory()
    blob_root = tmp_path / "blob"
    blob_hash, _size = BlobStore(blob_root).write_from_bytes(json.dumps(document).encode())
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.HERMES,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "hermes" / "trajectories" / "trajectory.json"),
        file_mtime="2025-01-02T03:04:05Z",
    ) as (reader, raw_id):
        _refuse_whole_document(monkeypatch)
        artifact = revision_backfill.prepare_retained_jsonl_artifact(
            reader, raw_id, directory=BlobStore(blob_root)._ensure_private_staging_root() / "prepared"
        )
        try:
            assert artifact.error is None, artifact.error
            assert artifact.positive_evidence_filtered
            sessions = list(artifact.iter_sessions())
            assert len(sessions) == 3
            assert len(sessions[0].session_events) > 300
        finally:
            artifact.discard()


def test_atif_subagent_entries_arrive_without_their_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each child step is spilled on its own; no entry is decoded with its step list."""
    document = _trajectory(40)
    # A root key containing dots must not pose as a path into the entries.
    document["subagent_trajectories.item"] = {"session_id": "posing-child", "steps": [{"source": "agent"}]}
    source = _source(tmp_path, document)
    expected = _expected(document, source)
    members: list[tuple[int, object, int | None]] = []
    spilled_steps: dict[int, int] = {}
    original_walk = spill_member_arrays

    def tracked_walk(handle: Any, container: str, nested: str, *, on_member: Any, on_nested_item: Any) -> bool:
        def record_member(index: int, fields: object, count: int | None) -> None:
            members.append((index, fields, count))
            on_member(index, fields, count)

        def record_step(index: int, ordinal: int, step: object) -> None:
            # A step is spilled before its entry has been reported.
            assert all(member[0] != index for member in members)
            spilled_steps[index] = spilled_steps.get(index, 0) + 1
            on_nested_item(index, ordinal, step)

        return original_walk(handle, container, nested, on_member=record_member, on_nested_item=record_step)

    monkeypatch.setattr(prepared_jsonl, "spill_member_arrays", tracked_walk)
    _refuse_whole_document(monkeypatch)
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    entries = document["subagent_trajectories"]
    assert isinstance(entries, list)
    assert [member[0] for member in members] == list(range(len(entries)))
    for index, fields, count in members:
        entry = entries[index]
        if isinstance(entry, dict) and isinstance(entry.get("steps"), list):
            assert isinstance(fields, dict) and "steps" not in fields
            assert count == spilled_steps.get(index, 0) == len(entry["steps"])
        else:
            assert count is None
    assert spilled_steps[0] == 41
    assert artifact.shard_path is not None
    _assert_same_publication(list(artifact.iter_sessions()), expected, artifact.shard_path, tmp_path)
    assert artifact.sessions_path is not None
    with sqlite3.connect(artifact.sessions_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not {table for table in tables if table.startswith("atif_")}


def test_atif_subagent_repeating_steps_keeps_collecting_parity(tmp_path: Path) -> None:
    """The decoder keeps a repeated key's last value, so the walk refuses it."""
    document = _trajectory(10)
    text = json.dumps(document).replace(
        '"session_id": "neutral-child", ',
        '"session_id": "neutral-child", "steps": [{"source": "agent", "message": "overwritten"}], ',
        1,
    )
    assert text.count('"overwritten"') == 1
    source = _source(tmp_path, document)
    source.write_text(text, encoding="utf-8")
    expected = _expected(json.loads(text), source)
    with source.open("rb") as handle:
        refused = prepared_jsonl._spill_atif_subagents(handle, sqlite3.connect(":memory:"))
    assert refused is False
    artifact = prepare_jsonl_blob(
        str(source),
        str(source),
        Provider.HERMES.value,
        "fallback",
        is_stream=False,
        shard_directory=str(tmp_path / "prepared"),
    )
    assert artifact.error is None
    assert artifact.shard_path is not None
    _assert_same_publication(list(artifact.iter_sessions()), expected, artifact.shard_path, tmp_path)
