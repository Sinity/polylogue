"""Correlation over real Git objects and canonical hydrated path evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.analysis.session_commit import detect_session_commits
from polylogue.api.archive import PolylogueArchiveMixin
from polylogue.archive.hydration import archive_block_to_domain
from polylogue.core.errors import PolylogueError
from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow
from tests.infra.builders import make_conv, make_msg
from tests.unit.insights.test_session_commit import _commit, _init_git_repo

pytestmark = pytest.mark.uses_real_clock("Neutral Git commit timestamps define the external scan window.")


def _block(path: str) -> dict[str, Any]:
    return archive_block_to_domain(
        ArchiveBlockRow(
            block_id="b",
            message_id="m",
            block_type="tool_use",
            text=None,
            content_identity=hashlib.sha256(b"neutral").hexdigest(),
            content_occurrence=0,
            tool_name="Read",
            tool_id="call",
            tool_input=json.dumps({"file_path": path}),
        )
    )


@pytest.mark.parametrize(
    "filename",
    ["plain.py", "zażółć.py", "space name.py", 'quote"name.py', "line\nbreak.py", "\nleading.py", " leading.py ", " "],
)
@pytest.mark.parametrize("absolute", [False, True])
def test_hydrated_paths_match_exact_git_filenames(tmp_path: Path, filename: str, absolute: bool) -> None:
    _init_git_repo(tmp_path)
    sha = _commit(tmp_path, filename, "neutral")
    now = datetime.now(timezone.utc)
    path = str(tmp_path / filename) if absolute else filename
    edges = detect_session_commits(
        "neutral", [{"text": "", "content_blocks": [_block(path)]}], now, now, repo_path=str(tmp_path)
    )
    assert [(edge.commit_sha, edge.file_overlap_count, edge.detection_method) for edge in edges] == [
        (sha, 1, "file_overlap")
    ]


@pytest.mark.parametrize("length", [7, 8, 9, 12, 39, 40])
def test_every_admitted_unique_prefix_matches_git_identity(tmp_path: Path, length: int) -> None:
    _init_git_repo(tmp_path)
    sha = _commit(tmp_path, "neutral.py", "neutral")
    prefix = sha[:length]
    assert (
        subprocess.check_output(
            ["git", "-C", str(tmp_path), "rev-parse", "--verify", prefix + "^{commit}"], text=True
        ).strip()
        == sha
    )
    now = datetime.now(timezone.utc)
    edges = detect_session_commits("neutral", [{"text": "Commit " + prefix}], now, now, repo_path=str(tmp_path))
    assert [(edge.commit_sha, edge.detection_method) for edge in edges] == [(sha, "explicit_ref")]


def test_public_correlation_uses_checkout_not_remote_identity(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    sha = _commit(tmp_path, "neutral.py", "neutral")
    now = datetime.now(timezone.utc)
    session = make_conv(
        id="codex-session:neutral",
        origin="codex-session",
        created_at=now,
        updated_at=now,
        working_directories=[str(tmp_path)],
        git_repository_url="https://example.invalid/neutral/repo.git",
        messages=[make_msg(id="m", origin="codex-session", role="assistant", text="Commit " + sha, timestamp=now)],
    )

    class Repository:
        async def get_session_refs(self, session_id: str) -> list[Any]:
            return []

        async def get_session_commits(self, session_id: str) -> list[Any]:
            return []

    class Reader:
        repository = Repository()

        async def get_session(self, session_id: str) -> Any:
            return session

    result = asyncio.run(
        PolylogueArchiveMixin.session_correlation_payload(cast(PolylogueArchiveMixin, Reader()), session.id)
    )
    assert result is not None
    assert result["repo"] == str(tmp_path)
    commits = result["commits"]
    assert isinstance(commits, list)
    assert all(isinstance(edge, dict) for edge in commits)
    assert [cast(dict[str, Any], edge)["commit_sha"] for edge in commits] == [sha]


def test_unavailable_git_lookup_is_not_successful_empty(tmp_path: Path) -> None:
    with pytest.raises(PolylogueError):
        detect_session_commits("neutral", [], repo_path=str(tmp_path / "missing"))
    _init_git_repo(tmp_path)
    _commit(tmp_path, "neutral.py", "neutral")
    now = datetime.now(timezone.utc)
    assert detect_session_commits("neutral", [], now, now, repo_path=str(tmp_path)) == []


def test_ambiguous_commit_prefix_is_explicitly_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.analysis.session_commit as owner

    _init_git_repo(tmp_path)
    _commit(tmp_path, "neutral.py", "neutral")
    real_read = owner._git_read

    def read(repo: str, *arguments: str) -> bytes:
        if arguments == ("rev-parse", "--disambiguate=abcdef0"):
            return b"abcdef0" + b"1" * 33 + b"\nabcdef0" + b"2" * 33 + b"\n"
        return real_read(repo, *arguments)

    monkeypatch.setattr(owner, "_git_read", read)
    with pytest.raises(owner.GitCorrelationUnavailableError, match="commit_prefix_ambiguous"):
        detect_session_commits("neutral", [{"text": "Commit abcdef0"}], repo_path=str(tmp_path))


def test_pinned_reader_selects_checkout_and_canonical_tool_input(tmp_path: Path) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.operations.read_view_extras import execute_correlation_read
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    sha = _commit(repo, "zażółć.py", "neutral")
    now = datetime.now(timezone.utc).isoformat()
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="neutral",
                title="neutral",
                created_at=now,
                updated_at=now,
                working_directories=[str(repo)],
                git_repository_url="https://example.invalid/neutral.git",
                messages=[
                    ParsedMessage(
                        provider_message_id="m",
                        role=Role.ASSISTANT,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Read",
                                tool_input={"file_path": str(repo / "zażółć.py")},
                            )
                        ],
                    )
                ],
            ),
        )
    with ArchiveStore.open_existing(root) as archive:
        result = execute_correlation_read({"session_id": session_id}, archive=archive)
    payload = result["payload"]
    assert isinstance(payload, dict)
    assert payload["repo"] == str(repo)
    assert [edge["commit_sha"] for edge in payload["commits"]] == [sha]
    assert payload["file_paths"] == [str(repo / "zażółć.py")]


def test_subdirectory_relative_paths_and_multiple_git_records(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    first = _commit(tmp_path, "sub/child.py", "first")
    second = _commit(tmp_path, "sub/zażółć.py", "second")
    now = datetime.now(timezone.utc)
    edges = detect_session_commits(
        "neutral",
        [{"text": "", "content_blocks": [_block("child.py"), _block("zażółć.py")]}],
        now,
        now,
        repo_path=str(tmp_path / "sub"),
    )
    assert [(edge.commit_sha, edge.file_overlap_count) for edge in edges] == [(second, 1), (first, 1)]
    assert (
        detect_session_commits(
            "neutral",
            [{"text": "", "content_blocks": [_block(str(tmp_path.parent / "sibling" / "child.py"))]}],
            now,
            now,
            repo_path=str(tmp_path),
        )
        == []
    )


def test_empty_git_records_do_not_consume_following_file_headers(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    first = _commit(tmp_path, "\nleading.py", "first")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-q", "-m", "empty-middle"], check=True)
    second = _commit(tmp_path, "other.py", "second")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-q", "-m", "empty-first"], check=True)
    now = datetime.now(timezone.utc)
    edges = detect_session_commits(
        "neutral",
        [{"text": "", "content_blocks": [_block("\nleading.py"), _block("other.py")]}],
        now,
        now,
        repo_path=str(tmp_path),
    )
    assert [(edge.commit_sha, edge.file_overlap_count) for edge in edges] == [(second, 1), (first, 1)]
