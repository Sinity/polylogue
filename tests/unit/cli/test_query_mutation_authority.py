"""The root query's tag/metadata writes go through the mutation authority.

Anti-vacuity for every test here: a route that calls ``ArchiveStore``'s tag or
metadata writers directly still changes ``user.db``, but leaves no
``operation_runs`` row, so :func:`test_add_tag_records_a_cli_operation_run` and
:func:`test_set_metadata_records_a_cli_operation_run` go red.  A route that
mutates while the query's read snapshot is still open cannot open its writable
connection at all, so the persistence tests go red with
``ReadOnlyArchiveError``/``database is locked``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder

_ORIGIN_FILTER = "origin:claude-ai-export"


@pytest.fixture
def tagged_archive(tmp_path: Path) -> Path:
    """Return an archive root holding three plain sessions."""
    initialize_active_archive_root(tmp_path)
    index_db = tmp_path / "index.db"
    for index in range(3):
        (
            SessionBuilder(index_db, f"conv-{index}")
            .provider("claude-ai")
            .title(f"Session {index}")
            .add_message(f"m{index}", role="user", text="hello alpha bravo")
            .save()
        )
    return tmp_path


def _run(archive_root: Path, *args: str) -> Result:
    env = {
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_DB_PATH": str(archive_root / "index.db"),
        "POLYLOGUE_NO_DAEMON": "1",
    }
    return CliRunner().invoke(cli, ["--no-daemon", *args], env=env)


def _payload(result: Result) -> dict[str, object]:
    assert result.exit_code == 0, result.output or result.exception
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    return payload


def _operation_runs(archive_root: Path) -> list[tuple[str, str, str, int]]:
    connection = sqlite3.connect(f"file:{archive_root / 'audit.db'}?mode=ro", uri=True)
    try:
        return [
            (str(row[0]), str(row[1]), str(row[2]), int(row[3]))
            for row in connection.execute(
                "SELECT operation_name, surface, status, affected_count FROM operation_runs ORDER BY operation_name"
            )
        ]
    finally:
        connection.close()


def test_add_tag_persists_across_the_matched_page(tagged_archive: Path) -> None:
    """``--add-tag`` writes the tag and a later query reads it back."""
    payload = _payload(_run(tagged_archive, "--add-tag", "triage", "find", _ORIGIN_FILTER))
    assert payload["operation"] == "add_tag"
    assert payload["affected_count"] == 3

    listed = _payload(_run(tagged_archive, "--format", "json", "find", "tag:triage"))
    items = listed["items"]
    assert isinstance(items, list)
    assert len(items) == 3
    for item in items:
        assert "triage" in item["tags"]


def test_add_tag_counts_session_tag_pairs(tagged_archive: Path) -> None:
    """Two tags over three sessions report six written pairs, not three."""
    payload = _payload(_run(tagged_archive, "--add-tag", "alpha", "--add-tag", "beta", "find", _ORIGIN_FILTER))
    assert payload["affected_count"] == 6


def test_add_tag_is_idempotent(tagged_archive: Path) -> None:
    """Re-adding a present tag reports nothing written."""
    _payload(_run(tagged_archive, "--add-tag", "triage", "find", _ORIGIN_FILTER))
    repeated = _payload(_run(tagged_archive, "--add-tag", "triage", "find", _ORIGIN_FILTER))
    assert repeated["affected_count"] == 0


def test_add_tag_records_a_cli_operation_run(tagged_archive: Path) -> None:
    """The write is journaled as one executor-routed CLI operation."""
    _payload(_run(tagged_archive, "--add-tag", "triage", "find", _ORIGIN_FILTER))
    assert _operation_runs(tagged_archive) == [("mutate-bulk-tag-sessions", "cli", "completed", 3)]


def test_set_metadata_persists_and_is_idempotent(tagged_archive: Path) -> None:
    """``--set`` writes metadata once and reports nothing on the repeat."""
    payload = _payload(_run(tagged_archive, "--set", "lane", "triage", "find", _ORIGIN_FILTER))
    assert payload["operation"] == "set_meta"
    assert payload["affected_count"] == 3

    repeated = _payload(_run(tagged_archive, "--set", "lane", "triage", "find", _ORIGIN_FILTER))
    assert repeated["affected_count"] == 0


def test_set_metadata_records_a_cli_operation_run(tagged_archive: Path) -> None:
    """The metadata write is journaled as one executor-routed CLI operation."""
    _payload(_run(tagged_archive, "--set", "lane", "triage", "find", _ORIGIN_FILTER))
    assert _operation_runs(tagged_archive) == [("mutate-bulk-set-metadata", "cli", "completed", 3)]


def test_combined_tag_and_metadata_reports_both_halves(tagged_archive: Path) -> None:
    """A combined mutation reports the tag and metadata counts separately."""
    payload = _payload(_run(tagged_archive, "--add-tag", "triage", "--set", "lane", "x", "find", _ORIGIN_FILTER))
    assert payload["operation"] == "mutate"
    assert payload["tag_count"] == 3
    assert payload["applied_count"] == 3
    assert {name for name, _surface, _status, _count in _operation_runs(tagged_archive)} == {
        "mutate-bulk-set-metadata",
        "mutate-bulk-tag-sessions",
    }


def test_single_session_tag_route_uses_the_same_authority(tagged_archive: Path) -> None:
    """Tagging one resolved session is journaled like the matched-page route."""
    _payload(_run(tagged_archive, "--add-tag", "triage", "find", "claude-ai-export:ext-conv-1"))
    assert _operation_runs(tagged_archive) == [("mutate-bulk-tag-sessions", "cli", "completed", 1)]
