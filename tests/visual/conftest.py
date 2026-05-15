"""Shared fixtures for reader visual/DOM evidence tests."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from http.server import HTTPServer
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

POLYLOGUE_LOCAL_PATH_PREFIXES = ("/home/", "/Users/", "/realm/", "/var/", "/etc/")


@dataclass(frozen=True)
class ReaderWorkspace:
    data_root: Path
    archive_root: Path
    state_dir: Path

    def as_env(self) -> dict[str, Path]:
        return {
            "data_root": self.data_root,
            "archive_root": self.archive_root,
            "state_dir": self.state_dir,
        }


class DOMSummary(HTMLParser):
    """Minimal DOM summary for browserless structural assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.classes: set[str] = set()
        self.scripts = 0
        self.styles = 0
        self.text_parts: list[str] = []
        self.meta_viewport = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        if tag == "script":
            self.scripts += 1
        if tag == "style":
            self.styles += 1
        if tag == "meta" and attr_map.get("name") == "viewport":
            self.meta_viewport = True
        if "id" in attr_map:
            self.ids.add(attr_map["id"])
        for cls in attr_map.get("class", "").split():
            self.classes.add(cls)

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.text_parts.append(text)

    @property
    def text(self) -> str:
        return " ".join(self.text_parts)


def parse_dom(html: str) -> DOMSummary:
    parser = DOMSummary()
    parser.feed(html)
    return parser


def assert_no_private_paths(text: str, *, context: str) -> None:
    for prefix in POLYLOGUE_LOCAL_PATH_PREFIXES:
        assert prefix not in text, f"{context} leaked absolute local path with prefix {prefix!r}"


def write_evidence_manifest(
    path: Path,
    *,
    artifact_id: str,
    route: str,
    fixture_id: str,
    checks: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "command": "pytest -q tests/visual",
        "fixture_id": fixture_id,
        "route": route,
        "evidence_kind": "browserless-dom",
        "generated_path": path.name,
        "checks": checks,
        "manual_review_required": False,
        "browser_gate_followup": "#865-playwright-screenshot-lane",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    loaded = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    assert loaded["artifact_id"] == artifact_id
    assert loaded["checks"] == checks
    return loaded


@pytest.fixture
def reader_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReaderWorkspace:
    workspace = ReaderWorkspace(
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        state_dir=tmp_path / "state",
    )
    monkeypatch.setenv("XDG_DATA_HOME", str(workspace.data_root))
    monkeypatch.setenv("XDG_STATE_HOME", str(workspace.state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(workspace.archive_root))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    return workspace


def archive_db_path(workspace: ReaderWorkspace) -> Path:
    return workspace.data_root / "polylogue" / "polylogue.db"


def seed_reader_archive(
    workspace: ReaderWorkspace,
    *,
    conversations: bool = True,
    message_fts: bool = True,
) -> None:
    from polylogue.storage.sqlite.schema_ddl_archive import (
        ARCHIVE_STORAGE_DDL,
        MESSAGE_FTS_DDL,
        SAVED_VIEWS_DDL,
        USER_ANNOTATIONS_DDL,
        USER_MARKS_DDL,
    )

    db = archive_db_path(workspace)
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(str(db))
    conn.executescript(ARCHIVE_STORAGE_DDL)
    conn.executescript(USER_MARKS_DDL)
    conn.executescript(USER_ANNOTATIONS_DDL)
    conn.executescript(SAVED_VIEWS_DDL)
    if message_fts:
        conn.executescript(MESSAGE_FTS_DDL)
    if conversations:
        records = [
            (
                "reader-c1",
                "claude-code",
                "MK3 reader target contract",
                {"repo": "polylogue", "cwd_display": "project/polylogue", "model": "claude-sonnet"},
                [
                    ("reader-c1-m1", "user", "Hello reader, show the target reference contract.", "message", 0, 0, 0),
                    (
                        "reader-c1-m2",
                        "assistant",
                        "The reader exposes stable anchors and action states.",
                        "message",
                        0,
                        0,
                        0,
                    ),
                    (
                        "reader-c1-m3",
                        "tool",
                        "python -m pytest tests/visual\nstatus: passed",
                        "tool_result",
                        1,
                        0,
                        0,
                    ),
                ],
            ),
            (
                "reader-c2",
                "chatgpt",
                "No-results smoke seed",
                {"repo": "polylogue", "cwd_display": "project/polylogue", "model": "gpt"},
                [("reader-c2-m1", "user", "Facet query and empty state fixture.", "message", 0, 0, 0)],
            ),
            (
                "reader-c3",
                "claude-ai",
                "Paste and privacy fixture",
                {"repo": "polylogue", "cwd_display": "project/polylogue", "model": "claude"},
                [("reader-c3-m1", "user", "A synthetic paste-like block with safe content.", "message", 0, 0, 1)],
            ),
        ]
        for conv_id, provider, title, provider_meta, messages in records:
            conn.execute(
                """
                INSERT INTO conversations(
                    conversation_id, provider_name, provider_conversation_id, title,
                    content_hash, provider_meta, version
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (conv_id, provider, f"provider-{conv_id}", title, f"hash-{conv_id}", json.dumps(provider_meta), 1),
            )
            for index, (message_id, role, text, message_type, has_tool, has_thinking, has_paste) in enumerate(messages):
                conn.execute(
                    """
                    INSERT INTO messages(
                        message_id, conversation_id, role, text, sort_key, provider_name,
                        content_hash, version, word_count, has_tool_use, has_thinking,
                        has_paste, message_type
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        conv_id,
                        role,
                        text,
                        float(index),
                        provider,
                        f"hash-{message_id}",
                        1,
                        len(text.split()),
                        has_tool,
                        has_thinking,
                        has_paste,
                        message_type,
                    ),
                )
        conn.execute(
            """
            INSERT INTO user_marks(target_type, target_id, conversation_id, message_id, mark_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("conversation", "reader-c1", "reader-c1", None, "star", "2026-05-15T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO user_marks(target_type, target_id, conversation_id, message_id, mark_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("conversation", "reader-c1", "reader-c1", None, "pin", "2026-05-15T00:01:00+00:00"),
        )
        conn.execute(
            "INSERT INTO saved_views(view_id, name, query_json, created_at) VALUES (?, ?, ?, ?)",
            (
                "reader-view-claude-code",
                "Claude Code reader fixtures",
                json.dumps({"provider": "claude-code", "query": "Hello", "limit": 100, "offset": 0}, sort_keys=True),
                "2026-05-15T00:02:00+00:00",
            ),
        )
    conn.commit()
    conn.close()


@contextmanager
def running_reader_server(
    workspace: ReaderWorkspace,
    *,
    conversations: bool = True,
    message_fts: bool = True,
) -> Iterator[tuple[HTTPServer, str]]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    seed_reader_archive(workspace, conversations=conversations, message_fts=message_fts)
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="reader-visual-smoke", daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def get_json(base_url: str, path: str) -> object:
    req = Request(f"{base_url}{path}")
    with urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        return json.loads(resp.read())


def get_text(base_url: str, path: str) -> tuple[int, str, str]:
    req = Request(f"{base_url}{path}")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
    except HTTPError as exc:
        body = exc.read().decode() if exc.fp else ""
        return exc.code, exc.headers.get("Content-Type", ""), body
