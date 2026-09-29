"""Neutral synthetic source payloads and builders for parser contract tests."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/source_parser_cases/cases.json"


def case(*keys: str) -> Any:
    value = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for key in keys:
        value = value[key]
    return value


def trajectory_db(path: Path, name: str, *, ambiguous_parent: bool = False) -> Path:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, trajectory_id TEXT, step_type TEXT,
                                step_format TEXT, step_payload TEXT, exit_code INTEGER);
            CREATE TABLE parent_references (cascade_id TEXT, parent_id TEXT);
        """)
        if name == "parent":
            conn.executemany(
                "INSERT INTO trajectory_meta VALUES (?, ?)", [("parent", "parent-alias"), ("child", "child-alias")]
            )
            refs = [("child", "parent"), ("child", "parent-alias")]
            if ambiguous_parent:
                refs.append(("child", "other-parent"))
            conn.executemany("INSERT INTO parent_references VALUES (?, ?)", refs)
        else:
            conn.execute("INSERT INTO trajectory_meta VALUES (?, ?)", ("trajectory-1", "cascade-1"))
        for ordinal, row in enumerate(case("antigravity", name)):
            conn.execute(
                "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ordinal,
                    row.get("trajectory_id", "trajectory-1"),
                    row["step_type"],
                    "v1",
                    json.dumps(row["payload"]),
                    row.get("exit_code"),
                ),
            )
    return path


def capture_with_native(native: dict[str, Any], attachments: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = case("browser")
    payload["raw_provider_payload"] = native
    payload["session"]["attachments"] = attachments
    return payload


def chatgpt_run(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = case("chatgpt")
    payload["mapping"]["result"]["message"]["metadata"]["aggregate_result"] = case("chatgpt_runs", name)
    return payload


def chatgpt_parts(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = case("chatgpt")
    msg = payload["mapping"]["result"]["message"]
    msg["author"]["role"] = "user"
    msg["content"] = {"content_type": "multimodal_text", "parts": ["kept", *case("structured_parts", name)]}
    return payload


def goals_db(path: Path, rowid: int) -> Path:
    goal = case("goals")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE thread_goals (thread_id TEXT, goal_id TEXT, objective TEXT)")
        conn.execute(
            "INSERT INTO thread_goals (rowid, thread_id, goal_id, objective) VALUES (?, ?, ?, ?)",
            (rowid, goal["thread_id"], goal["goal_id"], goal["objective"]),
        )
    return path


class _OrderedScandir:
    """A scandir result in a chosen order, usable the way ``os.scandir`` is."""

    def __init__(self, entries: list[os.DirEntry[str]]) -> None:
        self._entries = entries

    def __enter__(self) -> _OrderedScandir:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        return iter(self._entries)


def ordered_scandir(path: Path, *, reverse: bool = False) -> _OrderedScandir:
    """List ``path`` in name order, or reversed, to vary enumeration order."""
    with os.scandir(path) as scan:
        return _OrderedScandir(sorted(scan, key=lambda entry: entry.name, reverse=reverse))


def browser_thinking_turn() -> dict[str, Any]:
    """The generic capture route leaves authoredness to ParsedMessage."""
    payload: dict[str, Any] = case("browser")
    payload["session"].update(provider="chatgpt", provider_session_id="review-chat")
    payload["provenance"].update(source_url="https://chatgpt.com/c/review-chat", adapter_name="chatgpt-dom-v1")
    thinking, text = case("thinking_content")
    payload["session"]["turns"] = [
        {
            "provider_turn_id": "answer",
            "role": "assistant",
            "ordinal": 0,
            "text": f"<thinking>{thinking['thinking']}</thinking>\n{text['text']}",
            "blocks": [{"type": "thinking", "text": thinking["thinking"]}, text],
        }
    ]
    return payload


def native_and_envelope_files(*, repeated_synthetic_id: bool = False) -> dict[str, Any]:
    native: dict[str, Any] = case("claude_ai")
    envelope = []
    for index in range(2 if repeated_synthetic_id else 1):
        native_id = f"file-{index + 1}"
        name = f"file-{index + 1}.txt" if repeated_synthetic_id else "same.txt"
        native_file = case("attachment")
        native_file.update(file_uuid=native_id, file_name=name)
        if repeated_synthetic_id:
            native_file["extracted_content"] = "aaaa" if index == 0 else "bbbb"
        native["chat_messages"][0].setdefault("files", []).append(native_file)
        projection = {
            "provider_attachment_id": "claude-attachment:shared"
            if repeated_synthetic_id
            else f"claude-file:{native_id}",
            "message_provider_id": "u1",
            "name": name,
            "mime_type": "text/plain",
            "size_bytes": 4,
        }
        if repeated_synthetic_id:
            projection["extracted_content"] = native_file["extracted_content"]
        envelope.append(projection)
    return capture_with_native(native, envelope)


def claude_conversation_attachment(*, byte_relation: str) -> dict[str, Any]:
    payload: dict[str, Any] = case("claude_ai")
    owned, top = case("attachment"), case("attachment")
    owned["id"] = "owned-file"
    if byte_relation != "absent":
        owned["extracted_content"] = "same"
        top["extracted_content"] = "same" if byte_relation == "equal" else "diff"
    payload["chat_messages"][0]["attachments"] = [owned]
    payload["attachments"] = [top]
    return payload


def counter_payload(provider: str, key: str, value: object) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = case(provider)
    message = payload[0]["message"] if provider == "claude_code" else payload[1]["payload"]
    message["usage"][key] = value
    return payload
