"""Tests for session-to-git-commit attribution (#1690 phase 2-3).

Covers:
- Time window derivation from session timestamps
- File overlap scoring
- Confidence threshold filtering
- Empty session (no files referenced) → no edges
- Session with no repo → graceful handling
- Issue/PR reference extraction from message text
- GitHubRef deduplication
- Correlation result payload shape
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from polylogue.core.refs import ObjectRef
from polylogue.insights.session_commit import (
    GitHubRef,
    SessionCommitEdge,
    SessionCorrelationResult,
    build_correlation_result,
    correlation_result_to_payload,
    derive_scan_window,
    detect_session_commits,
    extract_github_refs,
    extract_referenced_files,
    score_file_overlap,
)


class TestDeriveScanWindow:
    def test_normal_session(self) -> None:
        created = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        updated = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        win_start, win_end = derive_scan_window(created, updated, before_hours=2, after_hours=2)
        assert win_start == datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        assert win_end == datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

    def test_none_timestamps_uses_now(self) -> None:
        win_start, win_end = derive_scan_window(None, None, before_hours=1, after_hours=1)
        # Window should be roughly centered on now with ±1 hour
        delta = (win_end - win_start).total_seconds()
        assert 7100 <= delta <= 7300  # ~2 hours plus some tolerance

    def test_naive_datetime_gets_tz(self) -> None:
        created = datetime(2024, 1, 15, 10, 0, 0)
        win_start, _ = derive_scan_window(created, created, before_hours=1, after_hours=1)
        assert win_start.tzinfo is not None

    def test_custom_window_sizes(self) -> None:
        created = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        win_start, win_end = derive_scan_window(created, created, before_hours=0, after_hours=0)
        assert win_start == created
        assert win_end == created


class TestScoreFileOverlap:
    def test_full_overlap(self) -> None:
        session_files = {"a.py", "b.py"}
        commit_files = {"a.py", "b.py", "c.py"}
        assert score_file_overlap(commit_files, session_files) == 1.0

    def test_partial_overlap(self) -> None:
        session_files = {"a.py", "b.py", "c.py", "d.py"}
        commit_files = {"a.py", "b.py"}
        assert score_file_overlap(commit_files, session_files) == 0.5

    def test_no_overlap(self) -> None:
        session_files = {"a.py", "b.py"}
        commit_files = {"x.py", "y.py"}
        assert score_file_overlap(commit_files, session_files) == 0.0

    def test_empty_session_files(self) -> None:
        session_files: set[str] = set()
        commit_files = {"a.py"}
        assert score_file_overlap(commit_files, session_files) == 0.0

    def test_empty_commit_files(self) -> None:
        session_files = {"a.py"}
        commit_files: set[str] = set()
        assert score_file_overlap(commit_files, session_files) == 0.0


class TestExtractReferencedFiles:
    def test_extracts_from_tool_use_with_affected_paths(self) -> None:
        messages = [
            {
                "id": "msg1",
                "text": "Reading file",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "affected_paths": ["src/main.py", "tests/test_main.py"],
                    }
                ],
            }
        ]
        files = extract_referenced_files(messages)
        assert "src/main.py" in files
        assert "tests/test_main.py" in files

    def test_extracts_from_tool_use_input_fields(self) -> None:
        messages = [
            {
                "id": "msg1",
                "text": "Using Edit",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {
                            "file_path": "/absolute/path/to/file.py",
                        },
                    }
                ],
            }
        ]
        files = extract_referenced_files(messages)
        assert "/absolute/path/to/file.py" in files

    def test_empty_messages(self) -> None:
        assert extract_referenced_files([]) == set()

    def test_no_content_blocks(self) -> None:
        messages = [{"id": "msg1", "text": "Hello, how are you?"}]
        files = extract_referenced_files(messages)
        assert files == set()

    def test_non_dict_blocks_are_skipped(self) -> None:
        messages = [
            {
                "id": "msg1",
                "text": "test",
                "content_blocks": ["not a dict", 123, None],
            }
        ]
        files = extract_referenced_files(messages)
        assert files == set()


class TestExtractGithubRefs:
    def test_full_issue_url(self) -> None:
        refs = extract_github_refs("See https://github.com/Sinity/polylogue/issues/1690 for details")
        assert len(refs) >= 1
        issue = next(r for r in refs if r.number == 1690)
        assert issue.owner == "Sinity"
        assert issue.repo == "polylogue"
        assert issue.kind == "issue"

    def test_full_pr_url(self) -> None:
        refs = extract_github_refs("PR: https://github.com/Sinity/polylogue/pull/1700")
        prs = [r for r in refs if r.kind == "pr"]
        assert len(prs) == 1
        assert prs[0].owner == "Sinity"
        assert prs[0].repo == "polylogue"
        assert prs[0].number == 1700

    def test_shorthand_owner_repo_ref(self) -> None:
        refs = extract_github_refs("Fixed in Sinity/polylogue#1690")
        matching = [r for r in refs if r.owner == "Sinity"]
        assert len(matching) >= 1
        assert matching[0].repo == "polylogue"
        assert matching[0].number == 1690

    def test_bare_number_ref(self) -> None:
        refs = extract_github_refs("As discussed in #1234, we need to fix this")
        bare = [r for r in refs if r.number == 1234 and r.owner is None]
        assert len(bare) >= 1

    def test_multiple_refs_in_one_text(self) -> None:
        text = "See #100 and #200 and https://github.com/foo/bar/issues/300"
        refs = extract_github_refs(text)
        numbers = {r.number for r in refs}
        assert 100 in numbers
        assert 200 in numbers
        assert 300 in numbers

    def test_no_refs(self) -> None:
        refs = extract_github_refs("Nothing to see here.")
        assert refs == []

    def test_with_message_id(self) -> None:
        refs = extract_github_refs("See #1234", message_id="msg-1")
        assert len(refs) >= 1
        assert refs[0].message_id == "msg-1"


class TestDetectSessionCommits:
    def test_empty_messages_no_git_available(self) -> None:
        """When no git repo exists, detection returns empty list gracefully."""
        edges = detect_session_commits(
            session_id="test-session",
            messages=[],
            repo_path="/nonexistent/path/should/not/exist",
            before_hours=2,
            after_hours=2,
        )
        assert edges == []

    def test_no_files_referenced(self) -> None:
        """Session with no file references produces no edges in repos with no commits."""
        edges = detect_session_commits(
            session_id="test-session",
            messages=[{"id": "m1", "text": "Hello", "content_blocks": []}],
            repo_path="/nonexistent/path",
        )
        assert edges == []


class TestSessionCorrelationResult:
    def test_empty_result(self) -> None:
        result = SessionCorrelationResult(
            session_id="test",
            window_start="2024-01-01T00:00:00+00:00",
            window_end="2024-01-01T02:00:00+00:00",
        )
        assert result.session_id == "test"
        assert result.commits == []
        assert result.issue_refs == []
        assert result.pr_refs == []

    def test_with_commits_and_refs(self) -> None:
        result = SessionCorrelationResult(
            session_id="test",
            window_start="2024-01-01T00:00:00+00:00",
            window_end="2024-01-01T02:00:00+00:00",
            commits=[
                SessionCommitEdge(
                    session_id="test",
                    commit_sha="abc123def456",
                    detection_method="file_overlap",
                    confidence=0.75,
                    file_overlap_count=3,
                )
            ],
            issue_refs=[GitHubRef(owner="foo", repo="bar", number=42, kind="issue", raw_match="#42")],
            pr_refs=[],
            file_paths=["src/main.py"],
        )
        assert len(result.commits) == 1
        assert len(result.issue_refs) == 1


class TestCorrelationResultToPayload:
    def test_payload_shape(self) -> None:
        result = SessionCorrelationResult(
            session_id="test",
            window_start="2024-01-01T00:00:00+00:00",
            window_end="2024-01-01T02:00:00+00:00",
            repo="/path/to/repo",
            commits=[
                SessionCommitEdge(
                    session_id="test",
                    commit_sha="abc123456789",
                    detection_method="file_overlap",
                    confidence=0.8,
                    file_overlap_count=4,
                )
            ],
            issue_refs=[GitHubRef(owner="owner", repo="repo", number=1, kind="issue", raw_match="#1")],
            pr_refs=[GitHubRef(owner="owner", repo="repo", number=2, kind="pr", raw_match="#2")],
            file_paths=["a.py", "b.py"],
        )
        payload = correlation_result_to_payload(result)
        assert payload["session_id"] == "test"
        assert payload["repo"] == "/path/to/repo"
        commits = cast(list[dict[str, Any]], payload["commits"])
        assert len(commits) == 1
        assert commits[0]["short_sha"] == "abc12345"
        assert commits[0]["object_ref"] == "commit:abc123456789"
        assert ObjectRef.parse(str(commits[0]["object_ref"])).kind == "commit"
        assert commits[0]["confidence"] == 0.8
        issue_refs = cast(list[dict[str, Any]], payload["issue_refs"])
        assert len(issue_refs) == 1
        assert issue_refs[0]["object_ref"] == "github-issue:owner/repo#1"
        pr_refs = cast(list[dict[str, Any]], payload["pr_refs"])
        assert len(pr_refs) == 1
        assert pr_refs[0]["object_ref"] == "github-pr:owner/repo#2"
        file_paths = cast(list[str], payload["file_paths"])
        assert len(file_paths) == 2
        assert payload["file_refs"] == ["file:a.py", "file:b.py"]
        object_refs = cast(list[str], payload["object_refs"])
        assert object_refs == [
            "commit:abc123456789",
            "github-issue:owner/repo#1",
            "github-pr:owner/repo#2",
            "file:a.py",
            "file:b.py",
        ]
        assert all(ObjectRef.parse(ref).format() == ref for ref in object_refs)


class TestBuildCorrelationResult:
    def test_builds_for_session_with_no_repo(self) -> None:
        """Session with no repo path still produces a valid result."""
        result = build_correlation_result(
            session_id="test",
            messages=[],
            repo_path="/nonexistent/path",
        )
        assert result.session_id == "test"
        assert result.commits == []  # No git repo available
        assert result.issue_refs == []
        assert result.pr_refs == []

    def test_extracts_refs_from_messages(self) -> None:
        """Issue and PR refs are extracted even without git access."""
        messages = [
            {
                "id": "m1",
                "text": "Closes #1690 and refs Sinity/polylogue#100",
                "content_blocks": [],
            }
        ]
        result = build_correlation_result(
            session_id="test",
            messages=messages,
            repo_path="/nonexistent/path",
        )
        # Issue refs should be extracted from text
        assert len(result.issue_refs) >= 1
        numbers = {r.number for r in result.issue_refs}
        assert 1690 in numbers


class TestSessionCommitEdge:
    def test_detection_method_values(self) -> None:
        for method in ("time_window", "file_overlap", "explicit_ref"):
            edge = SessionCommitEdge(
                session_id="s",
                commit_sha="abc",
                detection_method=method,
                confidence=0.5,
            )
            assert edge.detection_method == method

    def test_confidence_bounds(self) -> None:
        edge_min = SessionCommitEdge(session_id="s", commit_sha="abc", detection_method="time_window", confidence=0.0)
        assert edge_min.confidence == 0.0
        edge_max = SessionCommitEdge(session_id="s", commit_sha="abc", detection_method="file_overlap", confidence=1.0)
        assert edge_max.confidence == 1.0
