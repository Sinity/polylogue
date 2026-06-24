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

from tests.infra.archive_scenarios import native_session_id_for

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
        "command": "uv run devtools test tests/visual",
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


def index_db_path(workspace: ReaderWorkspace) -> Path:
    """index database the daemon reader resolves against."""
    return workspace.archive_root / "index.db"


# Archive ``<origin>:ext-<session>`` session ids for the seeded reader
# fixtures. The daemon web reader routes through the archive adapter and
# returns these canonical ids in every payload, target_ref, and anchor, so the
# reader tests key their URLs and assertions on these rather than the raw
# builder tokens.
READER_C1 = native_session_id_for("claude-code", "reader-c1")
READER_C2 = native_session_id_for("chatgpt", "reader-c2")
READER_C3 = native_session_id_for("claude-ai", "reader-c3")
READER_C1_M1 = f"{READER_C1}:reader-c1-m1"
READER_C1_M2 = f"{READER_C1}:reader-c1-m2"
READER_C1_M3 = f"{READER_C1}:reader-c1-m3"
READER_C3_M1 = f"{READER_C3}:reader-c3-m1"
READER_C3_DIFF = f"{READER_C3}:reader-c3-diff"


def _attachment_native_id(message_id: str, attachment_id: str) -> str:
    """Archive attachment id the reader emits: ``<session>:attachment:<id>``."""
    session_id = message_id.rsplit(":", 1)[0]
    return f"{session_id}:attachment:{attachment_id}"


# Archive attachment ids for the six MK3-state attachments seeded on
# ``reader-c1-m1`` by :func:`seed_reader_attachments`.
ATT_OK = _attachment_native_id(READER_C1_M1, "att-ok")
ATT_MISSING = _attachment_native_id(READER_C1_M1, "att-missing")
ATT_UNSUPPORTED = _attachment_native_id(READER_C1_M1, "att-unsupported")
ATT_TOOLARGE = _attachment_native_id(READER_C1_M1, "att-toolarge")
ATT_QUARANTINED = _attachment_native_id(READER_C1_M1, "att-quarantined")
ATT_RAWHTML = _attachment_native_id(READER_C1_M1, "att-rawhtml")


def _build_reader_c1(workspace: ReaderWorkspace, *, attachments: bool = False) -> None:
    """(Re)ingest the ``reader-c1`` session, optionally with the six
    MK3-state attachments linked to its first message."""
    from polylogue.core.enums import BlockType
    from polylogue.daemon.web_shell_attachments import PREVIEW_SIZE_BUDGET
    from tests.infra.storage_records import SessionBuilder

    builder = (
        SessionBuilder(index_db_path(workspace), "reader-c1")
        .provider("claude-code")
        .title("MK3 reader target contract")
        .created_at("2026-05-15T00:00:00+00:00")
        .updated_at("2026-05-15T00:02:00+00:00")
        .add_message(
            "reader-c1-m1",
            role="user",
            text="Hello reader, show the target reference contract.",
            timestamp="2026-05-15T00:00:00+00:00",
        )
        .add_message(
            "reader-c1-m2",
            role="assistant",
            text="The reader exposes stable anchors and action states.",
            timestamp="2026-05-15T00:01:00+00:00",
        )
        .add_message(
            "reader-c1-m3",
            role="tool",
            text="python -m pytest tests/visual\nstatus: passed",
            timestamp="2026-05-15T00:02:00+00:00",
            message_type="tool_result",
            blocks=[
                {
                    "type": BlockType.TOOL_RESULT.value,
                    "text": "python -m pytest tests/visual\nstatus: passed",
                }
            ],
        )
    )
    if attachments:
        attachment_specs: tuple[tuple[str, str, int, str | None, dict[str, object]], ...] = (
            ("att-ok", "text/plain", 1024, "blob/aa/aaaa-ok", {"name": "notes.txt"}),
            ("att-missing", "image/png", 2048, None, {"name": "screenshot.png"}),
            ("att-unsupported", "application/zip", 4096, "blob/bb/bbbb-zip", {"name": "bundle.zip"}),
            ("att-toolarge", "video/mp4", PREVIEW_SIZE_BUDGET + 1, "blob/cc/cccc-video", {"name": "recording.mp4"}),
            ("att-quarantined", "text/html", 512, "blob/dd/dddd-html", {"name": "suspect.html", "quarantined": True}),
            ("att-rawhtml", "text/html", 2048, "blob/ee/eeee-html", {"name": "<script>alert('xss')</script>.html"}),
        )
        for attachment_id, mime_type, size_bytes, path, meta in attachment_specs:
            builder = builder.add_attachment(
                attachment_id,
                message_id="reader-c1-m1",
                mime_type=mime_type,
                size_bytes=size_bytes,
                path=path,
                display_name=str(meta["name"]),
            )
    builder.save()


def seed_reader_attachments(workspace: ReaderWorkspace) -> None:
    """Re-ingest ``reader-c1`` with the six MK3-state attachments.

    Called by attachment tests after the server is running; re-ingest is an
    idempotent upsert that replaces the attachment-free ``reader-c1`` seeded by
    :func:`seed_reader_archive`."""
    _build_reader_c1(workspace, attachments=True)


def seed_reader_diff_paste(workspace: ReaderWorkspace) -> None:
    """Re-ingest ``reader-c3`` with an added unified-diff paste message so the
    paste-spans surface has both a per-span diff fold and the whole-message
    prose-paste fallback."""
    from tests.infra.storage_records import SessionBuilder

    diff_text = "Before the diff:\n@@ -1,3 +1,4 @@\n context\n-old\n+new\n+added\n"
    (
        SessionBuilder(index_db_path(workspace), "reader-c3")
        .provider("claude-ai")
        .title("Paste and privacy fixture")
        .created_at("2026-05-15T00:04:00+00:00")
        .updated_at("2026-05-15T00:05:00+00:00")
        .add_message(
            "reader-c3-m1",
            role="user",
            text="A synthetic paste-like block with safe content.",
            timestamp="2026-05-15T00:04:00+00:00",
        )
        .add_message(
            "reader-c3-diff",
            role="user",
            text=diff_text,
            timestamp="2026-05-15T00:05:00+00:00",
        )
        .save()
    )
    _force_has_paste(workspace, (READER_C3_M1, READER_C3_DIFF))


def _build_reader_sessions(workspace: ReaderWorkspace) -> None:
    """Seed the three reader sessions into the archive `index.db`."""
    from tests.infra.storage_records import SessionBuilder

    db = index_db_path(workspace)
    _build_reader_c1(workspace)
    (
        SessionBuilder(db, "reader-c2")
        .provider("chatgpt")
        .title("No-results smoke seed")
        .created_at("2026-05-15T00:03:00+00:00")
        .updated_at("2026-05-15T00:03:00+00:00")
        .add_message(
            "reader-c2-m1",
            role="user",
            text="Facet query and empty state fixture.",
            timestamp="2026-05-15T00:03:00+00:00",
        )
        .save()
    )
    (
        SessionBuilder(db, "reader-c3")
        .provider("claude-ai")
        .title("Paste and privacy fixture")
        .created_at("2026-05-15T00:04:00+00:00")
        .updated_at("2026-05-15T00:04:00+00:00")
        .add_message(
            "reader-c3-m1",
            role="user",
            text="A synthetic paste-like block with safe content.",
            timestamp="2026-05-15T00:04:00+00:00",
        )
        .save()
    )
    # The reader paste surface keys on the ``has_paste`` column. The prose
    # fixture carries no detectable paste marker, so flag it directly to mirror
    # a whole-message paste with no per-span diff structure.
    _force_has_paste(workspace, (READER_C3_M1,))


def _force_has_paste(workspace: ReaderWorkspace, message_ids: tuple[str, ...]) -> None:
    db = index_db_path(workspace)
    conn = sqlite3.connect(str(db))
    try:
        for message_id in message_ids:
            conn.execute("UPDATE messages SET has_paste = 1 WHERE message_id = ?", (message_id,))
        conn.commit()
    finally:
        conn.close()


def _seed_reader_user_state(workspace: ReaderWorkspace) -> None:
    """Seed marks, annotation, and a saved view through archive write paths."""
    import asyncio

    from polylogue.api import Polylogue
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_saved_view

    root = workspace.archive_root

    async def _seed() -> None:
        async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
            await poly.add_mark(READER_C1, "star")
            await poly.add_mark(READER_C1, "pin")
            await poly.save_annotation(
                "reader-ann-c1",
                READER_C1,
                "This session anchors the MK3 reader evidence.",
            )

    asyncio.run(_seed())

    user_db = root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    user_conn = sqlite3.connect(user_db)
    try:
        upsert_saved_view(
            user_conn,
            "Claude Code reader fixtures",
            {"provider": "claude-code", "query": "Hello", "limit": 100, "offset": 0},
        )
        user_conn.commit()
    finally:
        user_conn.close()


def seed_reader_assertion_claims(workspace: ReaderWorkspace) -> None:
    """Seed one assertion-backed overlay for the #1846 evidence panel."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

    user_db = workspace.archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as user_conn:
        upsert_assertion(
            user_conn,
            assertion_id="reader-evidence-decision",
            target_ref=f"session:{READER_C1}",
            scope_ref="repo:polylogue",
            kind=AssertionKind.DECISION,
            body_text="The evidence tab renders shared assertion claims.",
            author_ref="agent:poly-07",
            author_kind="agent",
            evidence_refs=[f"message:{READER_C1_M1}"],
            status="active",
            visibility="private",
            context_policy={"inject": False},
            now_ms=1_760_000_000_000,
        )
        user_conn.commit()


def _degrade_message_fts(workspace: ReaderWorkspace) -> None:
    """Drop the native message FTS virtual table and its sync triggers so a real
    query degrades to an explicit "Search index" route-state response, mirroring an
    interrupted bulk import that never rebuilt the search index."""
    db = index_db_path(workspace)
    conn = sqlite3.connect(str(db))
    try:
        for trigger in ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.commit()
    finally:
        conn.close()


def seed_reader_archive(
    workspace: ReaderWorkspace,
    *,
    sessions: bool = True,
    message_fts: bool = True,
) -> None:
    workspace.archive_root.mkdir(parents=True, exist_ok=True)
    if sessions:
        _build_reader_sessions(workspace)
        _rebuild_reader_insights(workspace)
        _seed_reader_user_state(workspace)
        if not message_fts:
            _degrade_message_fts(workspace)
    else:
        # An empty archive still needs the index.db to exist (with its
        # full schema, including messages_fts) so the daemon routes through the
        # archive reader.
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        initialize_archive_database(index_db_path(workspace), ArchiveTier.INDEX)


def _rebuild_reader_insights(workspace: ReaderWorkspace) -> None:
    """Materialize archive session insights so cost/insights reader panels read
    a populated profile/timeline/phases/threads set."""
    from polylogue.api.archive import _rebuild_archive_session_insights
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore.open_existing(workspace.archive_root, read_only=False) as archive:
        _rebuild_archive_session_insights(archive)


@contextmanager
def running_reader_server(
    workspace: ReaderWorkspace,
    *,
    sessions: bool = True,
    message_fts: bool = True,
) -> Iterator[tuple[HTTPServer, str]]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    seed_reader_archive(workspace, sessions=sessions, message_fts=message_fts)
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
        body = exc.read().decode()
        return exc.code, exc.headers.get("Content-Type", ""), body
