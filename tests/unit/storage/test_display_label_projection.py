"""Production-route coverage for the read-time session display label."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, TitleSource
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _edit_message(index: int, path: Path) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=f"assistant-{index}",
        role=Role.ASSISTANT,
        position=index,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Edit",
                tool_id=f"tool-{index}",
                tool_input={"file_path": str(path)},
            ),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=f"tool-{index}",
                text="ok",
                is_error=False,
                exit_code=0,
            ),
        ],
    )


def _session(
    native_id: str,
    root: Path,
    paths: tuple[str, ...],
    *,
    title: str | None = None,
    title_source: TitleSource | None = None,
    created_at: str = "2026-08-06T10:00:00+00:00",
) -> ParsedSession:
    messages = [
        ParsedMessage(
            provider_message_id="user-0",
            role=Role.USER,
            position=0,
            text="work on the repository",
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="work on the repository")],
        )
    ]
    messages.extend(_edit_message(index + 1, root / path) for index, path in enumerate(paths))
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=native_id,
        title=title,
        title_source=title_source,
        working_directories=[str(root)],
        created_at=created_at,
        updated_at=created_at,
        messages=messages,
    )


def _write_sessions(db_path: Path, sessions: list[ParsedSession]) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for session in sessions:
            write_parsed_session_to_archive(conn, session)
        conn.commit()
    finally:
        conn.close()


def test_display_label_is_fresh_and_keeps_stored_title_absent(tmp_path: Path) -> None:
    root = tmp_path / "polylogue"
    (root / ".git").mkdir(parents=True)
    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass

    _write_sessions(db_path, [_session("fresh", root, ("archive.py",))])
    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        session_id = archive.resolve_session_id("fresh")
        first = archive.read_summary(session_id)
        listed = archive.list_summaries(repo_names=("polylogue",), limit=10, offset=0)

    assert first.title is None
    assert first.title_source is None
    assert first.display_label == "polylogue · 1 file · 2 msgs · 2026-08-06"
    assert [item.display_label for item in listed if item.session_id == session_id] == [first.display_label]

    _write_sessions(
        db_path,
        [_session("fresh", root, ("archive.py", "query.py"))],
    )
    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        refreshed = archive.read_summary(session_id)

    assert refreshed.title is None
    assert refreshed.title_source is None
    assert refreshed.display_label == "polylogue · 2 files · 3 msgs · 2026-08-06"

    raw = (
        sqlite3.connect(db_path)
        .execute(
            "SELECT title, title_source FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        .fetchone()
    )
    assert raw == (None, None)


def test_provider_title_and_provenance_survive_the_projection(tmp_path: Path) -> None:
    root = tmp_path / "polylogue"
    (root / ".git").mkdir(parents=True)
    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass

    _write_sessions(
        db_path,
        [
            _session(
                "provider-title", root, ("archive.py",), title="Repair the read path", title_source=TitleSource.ORIGIN
            )
        ],
    )

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        summary = archive.read_summary(archive.resolve_session_id("provider-title"))

    assert summary.title == "Repair the read path"
    assert summary.display_label == "Repair the read path"
    assert summary.title_source == TitleSource.ORIGIN.value


def test_structural_label_collisions_are_small_and_explicit_on_list_route(tmp_path: Path) -> None:
    root = tmp_path / "polylogue"
    (root / ".git").mkdir(parents=True)
    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass

    _write_sessions(
        db_path,
        [
            _session("collision-a", root, ("same.py",)),
            _session("collision-b", root, ("other.py",)),
            _session("distinct", root, ("one.py", "two.py")),
        ],
    )

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        summaries = archive.list_summaries(repo_names=("polylogue",), limit=10, offset=0)

    labels = {summary.native_id: summary.display_label for summary in summaries}
    assert labels["collision-a"] == labels["collision-b"] == "polylogue · 1 file · 2 msgs · 2026-08-06"
    assert labels["distinct"] == "polylogue · 2 files · 3 msgs · 2026-08-06"
    assert len(set(labels.values())) == 2
    assert sum(1 for label in labels.values() if label and "polylogue" in label) == 3
