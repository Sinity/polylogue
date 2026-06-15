from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import build_demo_corpus_specs
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

EXPECTED_DEMO_SESSIONS = (
    (
        "chatgpt-export:dc13ca54-0bba-4298-a38f-09068c2ef2c5",
        "chatgpt-export",
        "dc13ca54-0bba-4298-a38f-09068c2ef2c5",
        "Debugging flaky async pipeline tests",
        1746826781690,
        1746826901690,
        3,
    ),
    (
        "claude-code-session:63705dcc-f3e5-4378-8118-8bc21e53bbb6",
        "claude-code-session",
        "63705dcc-f3e5-4378-8118-8bc21e53bbb6",
        '{"id": "fbc67a83-8c88-4491-853a-2fd7a7769e93", "input": {"query": "Can you help ...',
        1730589115737,
        1730589655737,
        10,
    ),
    (
        "codex-session:demo-00",
        "codex-session",
        "demo-00",
        "Could you review this code for potential issues?",
        None,
        1705985522161,
        6,
    ),
)

EXPECTED_DEMO_SOURCE_PATHS = (
    "chatgpt/demo-00.json",
    "claude-code/demo-00.jsonl",
    "codex/demo-00.jsonl",
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _session_rows(archive_root: Path) -> tuple[tuple[object, ...], ...]:
    with _connect(archive_root / "index.db") as conn:
        rows = conn.execute(
            """
            SELECT session_id, origin, native_id, title, created_at_ms, updated_at_ms, message_count
            FROM sessions
            ORDER BY origin, native_id
            """
        ).fetchall()
    return tuple(tuple(row) for row in rows)


def _row_count(db_path: Path, table: str) -> int:
    with _connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _raw_source_paths(archive_root: Path) -> tuple[str, ...]:
    with _connect(archive_root / "source.db") as conn:
        rows = conn.execute("SELECT source_path FROM raw_sessions ORDER BY origin, native_id").fetchall()
    return tuple(str(row["source_path"]) for row in rows)


def _stored_text_values(archive_root: Path) -> Iterable[str]:
    with _connect(archive_root / "index.db") as conn:
        for table, columns in (
            ("sessions", ("title", "git_repository_url", "instructions_text")),
            ("blocks", ("text", "tool_input", "tool_path", "tool_command")),
            ("attachments", ("display_name",)),
            ("attachment_refs", ("source_url", "caption")),
        ):
            for row in conn.execute(f"SELECT {', '.join(columns)} FROM {table}"):
                for value in row:
                    if value:
                        yield str(value)

    yield from _raw_source_paths(archive_root)


@pytest.mark.asyncio
async def test_demo_fixture_world_converges_into_deterministic_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    source_root = tmp_path / "demo-fixture-world-source"
    written = SyntheticCorpus.write_specs_artifacts(
        build_demo_corpus_specs(),
        source_root,
        prefix="demo",
    )
    source_paths = tuple(path for batch in written for path in batch.files)
    sources = [Source(name=path.parent.name, path=path.relative_to(source_root)) for path in source_paths]

    monkeypatch.chdir(source_root)
    result = await parse_sources_archive(archive_root, sources)

    assert sorted(result.processed_ids) == [row[0] for row in EXPECTED_DEMO_SESSIONS]
    assert result.counts["sessions"] == 3
    assert result.counts["messages"] == 19
    assert result.changed_counts["sessions"] == 3
    assert _session_rows(archive_root) == EXPECTED_DEMO_SESSIONS
    assert _raw_source_paths(archive_root) == EXPECTED_DEMO_SOURCE_PATHS

    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        hits = archive.search_summaries("pytest", limit=5)
    assert hits
    assert {hit.session_id for hit in hits} == {
        "claude-code-session:63705dcc-f3e5-4378-8118-8bc21e53bbb6",
    }

    forbidden_fragments = (str(tmp_path), "/tmp/", "/realm/", "/home/")
    stored_values = tuple(_stored_text_values(archive_root))
    assert stored_values
    assert all(not Path(path).is_absolute() for path in _raw_source_paths(archive_root))
    assert all(fragment not in value for value in stored_values for fragment in forbidden_fragments)

    before_counts = {
        "raw_sessions": _row_count(archive_root / "source.db", "raw_sessions"),
        "sessions": _row_count(archive_root / "index.db", "sessions"),
        "messages": _row_count(archive_root / "index.db", "messages"),
        "blocks": _row_count(archive_root / "index.db", "blocks"),
    }

    repeat = await parse_sources_archive(archive_root, sources)

    assert sorted(repeat.processed_ids) == [row[0] for row in EXPECTED_DEMO_SESSIONS]
    assert _session_rows(archive_root) == EXPECTED_DEMO_SESSIONS
    assert {
        "raw_sessions": _row_count(archive_root / "source.db", "raw_sessions"),
        "sessions": _row_count(archive_root / "index.db", "sessions"),
        "messages": _row_count(archive_root / "index.db", "messages"),
        "blocks": _row_count(archive_root / "index.db", "blocks"),
    } == before_counts
