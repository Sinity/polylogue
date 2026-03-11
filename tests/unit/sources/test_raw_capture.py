"""Operational contracts for raw source iteration that are not covered by the law suites."""

from __future__ import annotations

import json

import pytest

from polylogue.config import Source
from polylogue.sources.source import iter_source_conversations_with_raw
from tests.infra.helpers import GenericConversationBuilder


def test_iter_source_conversations_with_raw_tracks_failures_without_stopping(tmp_path) -> None:
    (
        GenericConversationBuilder("good")
        .add_message("user", "valid", text="valid")
        .write_to(tmp_path / "good.json")
    )
    (tmp_path / "bad.json").write_text("{ broken json", encoding="utf-8")

    cursor_state: dict[str, object] = {}
    results = list(iter_source_conversations_with_raw(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert len(results) == 1
    raw_data, conversation = results[0]
    assert raw_data is not None
    assert conversation.provider_conversation_id == "good"
    assert cursor_state["file_count"] == 2
    assert cursor_state["failed_count"] >= 1
    assert any("bad.json" in item["path"] for item in cursor_state["failed_files"])


@pytest.mark.parametrize("skip_dir_name", ["analysis", "__pycache__"])
def test_iter_source_conversations_with_raw_prunes_skip_dirs(tmp_path, skip_dir_name: str) -> None:
    skip_dir = tmp_path / skip_dir_name
    skip_dir.mkdir()
    GenericConversationBuilder("skipped").write_to(skip_dir / "data.json")

    results = list(iter_source_conversations_with_raw(Source(name="test", path=tmp_path)))

    assert results == []


def test_iter_source_conversations_with_raw_follows_symlinked_directories(tmp_path) -> None:
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    GenericConversationBuilder("linked").write_to(subdir / "conv.json")

    link = tmp_path / "link"
    try:
        link.symlink_to(subdir)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this system")

    results = list(iter_source_conversations_with_raw(Source(name="test", path=tmp_path)))

    assert results
    assert any(conversation.provider_conversation_id == "linked" for _, conversation in results)


def test_iter_source_conversations_with_raw_accepts_single_file_source(tmp_path) -> None:
    file_path = tmp_path / "single.json"
    GenericConversationBuilder("single").write_to(file_path)

    results = list(iter_source_conversations_with_raw(Source(name="test", path=file_path)))

    assert len(results) == 1
    raw_data, conversation = results[0]
    assert raw_data is not None
    assert raw_data.source_path == str(file_path)
    assert conversation.provider_conversation_id == "single"


def test_iter_source_conversations_with_raw_grouped_provider_keeps_whole_file_bytes(tmp_path) -> None:
    records = [
        {"type": "user", "uuid": "u1", "sessionId": "s1", "message": {"content": "Hello"}},
        {"type": "assistant", "uuid": "a1", "sessionId": "s1", "message": {"content": "Hi"}},
    ]
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    results = list(iter_source_conversations_with_raw(Source(name="claude-code", path=tmp_path)))

    assert len(results) == 1
    raw_data, conversation = results[0]
    assert raw_data is not None
    assert raw_data.source_index is None
    assert b"Hello" in raw_data.raw_bytes
    assert b"Hi" in raw_data.raw_bytes
    assert conversation.provider_name == "claude-code"


def test_iter_source_conversations_with_raw_grouped_zip_keeps_entry_bytes(tmp_path) -> None:
    import json
    import zipfile

    records = [
        {"type": "user", "uuid": "u1", "sessionId": "s1", "message": {"content": "From zip"}},
    ]
    archive_path = tmp_path / "claude-code.zip"
    content = "\n".join(json.dumps(record) for record in records) + "\n"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("session.jsonl", content)

    results = list(iter_source_conversations_with_raw(Source(name="claude-code", path=archive_path)))

    assert len(results) == 1
    raw_data, conversation = results[0]
    assert raw_data is not None
    assert raw_data.source_path == f"{archive_path}:session.jsonl"
    assert raw_data.source_index is None
    assert b"From zip" in raw_data.raw_bytes
    assert conversation.provider_name == "claude-code"


def test_iter_source_conversations_with_raw_zip_assigns_source_indexes_per_emitted_conversation(tmp_path) -> None:
    import json
    import zipfile

    archive_path = tmp_path / "multi.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "data.json",
            json.dumps(
                [
                    {"id": "c1", "messages": [{"id": "m1", "role": "user", "text": "Q1"}]},
                    {"id": "c2", "messages": [{"id": "m2", "role": "user", "text": "Q2"}]},
                ]
            ),
        )

    results = list(iter_source_conversations_with_raw(Source(name="test", path=archive_path)))

    assert len(results) == 2
    assert [raw_data.source_index for raw_data, _ in results if raw_data is not None] == [0, 1]


def test_iter_source_conversations_with_raw_tracks_unicode_decode_failures(tmp_path) -> None:
    bad_file = tmp_path / "bad_encoding.json"
    bad_file.write_bytes(b"\xff\xfe invalid utf-8 { bad json")

    cursor_state: dict[str, object] = {}
    list(iter_source_conversations_with_raw(Source(name="test", path=tmp_path), cursor_state=cursor_state))

    assert cursor_state["failed_count"] >= 1
    assert any("bad_encoding.json" in item["path"] for item in cursor_state["failed_files"])
