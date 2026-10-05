"""Destructive CLI verbs act on exactly the sessions the operator selected.

Every case runs the production CLI against a real daemon over a synthetic
archive: the selection read, the refusal, the daemon's preview, and the write
are the shipped routes. The archive holds three sessions that mention
``TOKEN`` plus, for each, an unselected sibling whose id extends the selected
id, so any prefix resolution or query widening is observable as a sibling that
changed.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from tests.infra.daemon_operations import DaemonOperationStack, cli_daemon_archive
from tests.infra.storage_records import SessionBuilder

pytestmark = pytest.mark.uses_real_clock(
    "starts the real UDS listener and coordinator loop; wall-clock events bound socket and writer ownership waits"
)

TOKEN = "zzexactselectiontoken"
#: Ref-shaped (long, has a digit) but names no session; every selected
#: session's text mentions it, so a text-search fallback would match them all.
ABSENT_REF = "zzabsentref0001"


class _Archive:
    def __init__(self, stack: DaemonOperationStack) -> None:
        self.stack = stack
        self.selected: list[str] = []  # oldest first
        self.siblings: list[str] = []

    @property
    def newest(self) -> str:
        return self.selected[-1]

    def remaining(self) -> set[str]:
        return {sid for sid in (*self.selected, *self.siblings) if self.stack.session_exists(sid)}

    def user_tags(self) -> dict[str, set[str]]:
        tags: dict[str, set[str]] = {}
        user_db = self.stack.archive_root / "user.db"
        with sqlite3.connect(user_db) as conn:
            for target_ref, key in conn.execute(
                "SELECT target_ref, key FROM assertions WHERE kind = 'tag' AND status != 'deleted'"
            ):
                tags.setdefault(str(target_ref).removeprefix("session:"), set()).add(str(key))
        return tags


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Archive]:
    selected: list[str] = []
    siblings: list[str] = []

    def seed(root: Path) -> None:
        for number in range(3):
            stamp = f"2026-0{number + 1}-01T00:00:00+00:00"
            chosen = (
                SessionBuilder(root / "index.db", f"picked-{number}")
                .provider("codex")
                .title(f"Selected session {number}")
                .updated_at(stamp)
                .add_message(text=f"{TOKEN} {ABSENT_REF} selected body {number}", timestamp=stamp)
            )
            chosen.save()
            selected.append(chosen.native_session_id())
            sibling = (
                SessionBuilder(root / "index.db", f"picked-{number}-keep")
                .provider("codex")
                .title(f"Unselected sibling {number}")
                .updated_at(stamp)
                .add_message(text=f"unrelated sibling body {number}", timestamp=stamp)
            )
            sibling.save()
            siblings.append(sibling.native_session_id())
        from polylogue.storage.sqlite.connection import _clear_connection_cache

        _clear_connection_cache()

    with cli_daemon_archive(tmp_path / "archive", monkeypatch, seed_archive=seed, home=tmp_path / "home") as stack:
        state = _Archive(stack)
        state.selected, state.siblings = selected, siblings
        assert all(sibling.startswith(chosen) for chosen, sibling in zip(selected, siblings, strict=True))
        yield state


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(cli, ["--plain", *args], catch_exceptions=False)


def _json_document(result: Result) -> dict[str, object]:
    """The last JSON document the command printed (a document starts at column 0)."""
    lines = result.stdout.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("{")]
    assert starts, f"no JSON document in output:\n{result.output}"
    payload, _end = json.JSONDecoder().raw_decode("\n".join(lines[starts[-1] :]))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--latest", "find", TOKEN, "then", "delete", "--yes", "--all"], id="latest-with-all-delete"),
        pytest.param(["--latest", "find", TOKEN, "then", "delete", "--dry-run", "--all"], id="latest-with-all-preview"),
        pytest.param(["--limit", "1", "find", TOKEN, "then", "delete", "--yes", "--all"], id="limit-delete"),
        pytest.param(["--offset", "1", "find", TOKEN, "then", "delete", "--yes", "--all"], id="offset-delete"),
        pytest.param(["--latest", "find", TOKEN, "then", "mark", "--tag-add", "wide", "--all"], id="latest-mark"),
        pytest.param(["--limit", "1", "find", TOKEN, "then", "mark", "--tag-add", "wide", "--all"], id="limit-mark"),
    ],
)
def test_over_selecting_combinations_are_refused_before_anything_changes(archive: _Archive, argv: list[str]) -> None:
    """A selector the verb cannot honour exactly is a typed refusal, not a wider set.

    Anti-vacuity: drop ``require_exact_mutation_selection`` and
    ``--latest --all`` walks the whole filter (the list page reported the
    filter's total, so continuation never stopped) while ``--limit``/``--offset``
    are ignored by the complete walk -- every selected session is deleted or
    tagged and the ``remaining``/tag assertions go red.
    """

    before = archive.remaining()
    result = _invoke(argv)

    assert result.exit_code == 2, result.output
    assert "does not combine" in result.output
    assert archive.remaining() == before
    assert archive.user_tags() == {}


def test_latest_selects_exactly_the_newest_session_for_preview_and_delete(archive: _Archive) -> None:
    """``--latest`` is one session through the guard, the preview, and the write.

    Anti-vacuity: report the filter's full count as the ``--latest`` page's
    total and the verb's resolution collects all three matches, so the
    singleton delete refuses as ambiguous instead of deleting the newest.
    """

    preview = _json_document(_invoke(["--latest", "find", TOKEN, "then", "delete", "--dry-run"]))
    assert preview["status"] == "preview"
    assert preview["session_count"] == 1
    assert preview["session_ids_sample"] == [archive.newest]

    deleted = _json_document(_invoke(["--latest", "find", TOKEN, "then", "delete", "--yes"]))

    assert deleted["affected_count"] == 1
    assert archive.remaining() == {*archive.selected[:-1], *archive.siblings}


def test_preview_count_equals_applied_count_and_spares_prefix_siblings(archive: _Archive) -> None:
    """The previewed set is the deleted set; unselected prefix siblings survive.

    Anti-vacuity: re-resolve a previewed id by prefix at apply and the sibling
    sharing that prefix is deleted with it; widen the selection and
    ``affected_count`` exceeds the preview's ``session_count``.
    """

    preview = _json_document(_invoke(["find", TOKEN, "then", "delete", "--dry-run", "--all"]))
    assert preview["session_count"] == len(archive.selected)
    # Three matches fit inside the preview's bounded sample.
    previewed = preview["session_ids_sample"]
    assert isinstance(previewed, list)
    assert set(previewed) == set(archive.selected)

    applied = _json_document(_invoke(["find", TOKEN, "then", "delete", "--yes", "--all"]))

    assert applied["affected_count"] == preview["session_count"]
    assert archive.remaining() == set(archive.siblings)


def test_root_mutation_on_an_unknown_ref_is_refused_not_widened_to_a_text_search(archive: _Archive) -> None:
    """A ref-shaped token that names no session never becomes a mutation's text search.

    Anti-vacuity: let the missed transcript probe fall through to the page
    query and ``ABSENT_REF`` matches every selected session's text, so all
    three are tagged although the operator named one (absent) session.
    """

    result = _invoke(["--add-tag", "widened", "find", ABSENT_REF])

    assert result.exit_code == 1, result.output
    assert f"Session not found: {ABSENT_REF}" in result.output
    assert archive.user_tags() == {}


def test_root_mutation_on_a_resolving_ref_tags_exactly_that_session(archive: _Archive) -> None:
    """A ref-shaped token that names a session mutates that session alone.

    Anti-vacuity: lower the token into the page selection as text instead and
    no session's text mentions its own id, so nothing is tagged; resolve it by
    prefix and the unselected sibling sharing the prefix is tagged too.
    """

    target = archive.selected[0]
    result = _invoke(["--add-tag", "exact", "find", target])

    assert result.exit_code == 0, result.output
    assert archive.user_tags() == {target: {"exact"}}
