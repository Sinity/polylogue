"""Tests for the session structural label projection (polylogue-cijx.4).

Covers: decision 2 (repo-relative paths) and decision 3 (the label is a
read-time projection, never a stored column).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.archive.session.repo_identity import repo_relative_path
from polylogue.core.enums import BlockType, Provider
from polylogue.insights.session_label import (
    SessionLabelInputs,
    compute_session_structural_label,
    dominant_repo_relative_path_for_session,
    session_structural_label_for_session,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _read_message(idx: int, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=f"u{idx}",
        role=Role.USER,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _edit_call(idx: int, file_path: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=f"m{idx}",
        role=Role.ASSISTANT,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Edit",
                tool_id=f"tool-{idx}",
                tool_input={"file_path": file_path},
            ),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=f"tool-{idx}",
                text="ok",
                is_error=False,
                exit_code=0,
            ),
        ],
    )


# ── compute_session_structural_label: pure format tests ────────────────────


def test_provider_title_wins_over_structural_form() -> None:
    inputs = SessionLabelInputs(
        provider_title="Fix the ingest race",
        repo_name="polylogue",
        is_directory=False,
        dominant_path="pipeline/services/ingest_batch/_core.py",
        additional_file_count=26,
        message_count=499,
    )
    assert compute_session_structural_label(inputs) == "Fix the ingest race"


def test_structural_label_matches_bead_evidence_shape() -> None:
    """Reproduces the exact example from polylogue-cijx.4's evidence:
    'polylogue · pipeline/services/ingest_batch/_core.py +26 · 499 msgs'."""
    inputs = SessionLabelInputs(
        provider_title=None,
        repo_name="polylogue",
        is_directory=False,
        dominant_path="pipeline/services/ingest_batch/_core.py",
        additional_file_count=26,
        message_count=499,
    )
    assert (
        compute_session_structural_label(inputs) == "polylogue · pipeline/services/ingest_batch/_core.py +26 · 499 msgs"
    )


def test_structural_label_omits_plus_n_when_only_one_file_touched() -> None:
    inputs = SessionLabelInputs(
        provider_title=None,
        repo_name="sinex",
        is_directory=False,
        dominant_path="src/main.rs",
        additional_file_count=0,
        message_count=12,
    )
    assert compute_session_structural_label(inputs) == "sinex · src/main.rs · 12 msgs"


def test_structural_label_degrades_to_message_count_only_with_no_evidence() -> None:
    inputs = SessionLabelInputs(
        provider_title=None,
        repo_name=None,
        is_directory=True,
        dominant_path=None,
        additional_file_count=0,
        message_count=3,
    )
    assert compute_session_structural_label(inputs) == "3 msgs"


def test_structural_label_suppresses_repo_name_when_marked_directory() -> None:
    """polylogue-cijx.2 AC4: a bare directory (no git evidence) must never
    display as if it were a repository. This reproduces the exact bug this
    AC exists to kill -- a session whose cwd happens to be e.g.
    ``/home/sinity`` must not surface "sinity" as a repo name -- as a defensive
    read-layer check: even if ``repo_name`` is populated while ``is_directory``
    is True (contradicting the write-path invariant), the label must not leak
    it."""
    inputs = SessionLabelInputs(
        provider_title=None,
        repo_name="sinity",
        is_directory=True,
        dominant_path=None,
        additional_file_count=0,
        message_count=7,
    )
    assert compute_session_structural_label(inputs) == "7 msgs"


# ── repo_relative_path ──────────────────────────────────────────────────────


def test_repo_relative_path_strips_checkout_root() -> None:
    assert (
        repo_relative_path(
            "/realm/worktrees/polylogue-x/polylogue/pipeline/services/ingest_batch/_core.py",
            "/realm/worktrees/polylogue-x",
        )
        == "polylogue/pipeline/services/ingest_batch/_core.py"
    )


def test_repo_relative_path_returns_unchanged_outside_root() -> None:
    assert repo_relative_path("/etc/hosts", "/realm/project/polylogue") == "/etc/hosts"


def test_repo_relative_path_handles_missing_root() -> None:
    assert repo_relative_path("polylogue/foo.py", "") == "polylogue/foo.py"


# ── read-path integration: dominant_repo_relative_path_for_session ─────────


def test_dominant_path_is_repo_relative_across_two_worktree_checkouts(tmp_path: Path) -> None:
    """The same logical file, edited from two different worktree checkouts
    of ONE repository, must resolve to the SAME repo-relative dominant path
    once each session's checkout root is known -- decision 2."""
    conn = _connect(tmp_path / "index.db")
    repo_root_a = tmp_path / "checkout-a"
    repo_root_b = tmp_path / "checkout-b"
    (repo_root_a / ".git").mkdir(parents=True)
    (repo_root_b / ".git").mkdir(parents=True)

    session_a = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="session-a",
        working_directories=[str(repo_root_a)],
        messages=[
            _read_message(0, "edit the ingest core"),
            _edit_call(1, str(repo_root_a / "pipeline" / "_core.py")),
        ],
    )
    session_b = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="session-b",
        working_directories=[str(repo_root_b)],
        messages=[
            _read_message(0, "edit the ingest core"),
            _edit_call(1, str(repo_root_b / "pipeline" / "_core.py")),
        ],
    )
    session_a_id = write_parsed_session_to_archive(conn, session_a)
    session_b_id = write_parsed_session_to_archive(conn, session_b)

    path_a, extra_a = dominant_repo_relative_path_for_session(conn, session_a_id)
    path_b, extra_b = dominant_repo_relative_path_for_session(conn, session_b_id)

    assert path_a == "pipeline/_core.py"
    assert path_b == "pipeline/_core.py"
    assert extra_a == 0
    assert extra_b == 0


def test_dominant_path_picks_most_touched_file_and_counts_the_rest(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    repo_root = tmp_path / "myrepo"
    (repo_root / ".git").mkdir(parents=True)

    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="session-multi-file",
        working_directories=[str(repo_root)],
        messages=[
            _read_message(0, "touch several files"),
            _edit_call(1, str(repo_root / "a.py")),
            _edit_call(2, str(repo_root / "a.py")),
            _edit_call(3, str(repo_root / "b.py")),
            _edit_call(4, str(repo_root / "c.py")),
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)

    dominant_path, additional = dominant_repo_relative_path_for_session(conn, session_id)

    assert dominant_path == "a.py"
    assert additional == 2  # b.py, c.py


def test_session_structural_label_for_session_end_to_end(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    repo_root = tmp_path / "myrepo"
    (repo_root / ".git").mkdir(parents=True)

    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="session-e2e",
        working_directories=[str(repo_root)],
        messages=[
            _read_message(0, "work on the core module"),
            _edit_call(1, str(repo_root / "core.py")),
        ],
    )
    session_id = write_parsed_session_to_archive(conn, session)

    label = session_structural_label_for_session(
        conn,
        session_id,
        message_count=2,
        provider_title=None,
    )
    assert label == "myrepo · core.py · 2 msgs"

    # A real provider title always wins over the structural form.
    titled_label = session_structural_label_for_session(
        conn,
        session_id,
        message_count=2,
        provider_title="Refactor the core module",
    )
    assert titled_label == "Refactor the core module"
