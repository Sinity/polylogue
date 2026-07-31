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
- polylogue-l9su: typed evidence (Claude-Session commit trailer,
  session_refs-derived GitHubRef) takes priority over the regex/
  time-window heuristics, and disagreements between the two are surfaced.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from polylogue.core.refs import ObjectRef
from polylogue.insights.session_commit import (
    SOURCE_HEURISTIC,
    SOURCE_TYPED,
    GitHubRef,
    SessionCommitEdge,
    SessionCorrelationResult,
    bridge_session_ids_from_events,
    build_correlation_result,
    correlation_result_to_payload,
    derive_scan_window,
    detect_session_commits,
    extract_claude_session_trailer_tokens,
    extract_github_refs,
    extract_referenced_files,
    score_file_overlap,
    typed_refs_from_session_refs,
)


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "agent@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Agent"], check=True)


def _commit(path: Path, filename: str, message: str) -> str:
    target = path / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", filename], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", message], check=True)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


class TestExtractClaudeSessionTrailerTokens:
    def test_extracts_token_from_trailer(self) -> None:
        body = (
            "fix: ship it\n\n"
            "Co-Authored-By: Claude <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_0182HDxDpJpsbn2qcKWK6Fsf\n"
        )
        assert extract_claude_session_trailer_tokens(body) == {"0182HDxDpJpsbn2qcKWK6Fsf"}

    def test_no_trailer_yields_empty_set(self) -> None:
        assert extract_claude_session_trailer_tokens("fix: unrelated change\n") == set()


class TestTrailerTakesPriorityOverHeuristics:
    """polylogue-l9su AC1/AC4: a matching Claude-Session trailer is
    authoritative and supersedes file_overlap/time_window/explicit_ref for
    that commit; a foreign trailer is surfaced as a disagreement rather than
    silently accepted or dropped.
    """

    def test_matching_trailer_wins_as_origin_reported(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        token = "0182HDxDpJpsbn2qcKWK6Fsf"
        sha = _commit(
            tmp_path,
            "src/main.py",
            f"fix: ship it\n\nClaude-Session: https://claude.ai/code/session_{token}\n",
        )
        now = datetime.now(timezone.utc)

        # A message referencing the same file would otherwise also earn a
        # file_overlap edge -- the trailer match must win instead.
        messages = [
            {
                "id": "m1",
                "text": "editing src/main.py",
                "content_blocks": [{"type": "tool_use", "name": "Edit", "affected_paths": ["src/main.py"]}],
            }
        ]

        edges = detect_session_commits(
            session_id="claude-code-session:own-session",
            messages=messages,
            session_created_at=now - timedelta(minutes=5),
            session_updated_at=now,
            repo_path=str(tmp_path),
            bridge_session_ids=[f"cse_{token}"],
        )

        assert len(edges) == 1
        edge = edges[0]
        assert edge.commit_sha == sha
        assert edge.detection_method == "origin_reported"
        assert edge.confidence == 1.0
        assert edge.disagreement_note is None

    def test_foreign_trailer_flags_disagreement_but_keeps_heuristic_edge(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        other_token = "01OtherSessionToken0000000"
        sha = _commit(
            tmp_path,
            "src/main.py",
            f"fix: ship it\n\nClaude-Session: https://claude.ai/code/session_{other_token}\n",
        )
        now = datetime.now(timezone.utc)

        messages = [
            {
                "id": "m1",
                "text": "editing src/main.py",
                "content_blocks": [{"type": "tool_use", "name": "Edit", "affected_paths": ["src/main.py"]}],
            }
        ]

        edges = detect_session_commits(
            session_id="claude-code-session:own-session",
            messages=messages,
            session_created_at=now - timedelta(minutes=5),
            session_updated_at=now,
            repo_path=str(tmp_path),
            bridge_session_ids=["cse_this-session-does-not-match"],
        )

        assert len(edges) == 1
        edge = edges[0]
        assert edge.commit_sha == sha
        assert edge.detection_method == "file_overlap"
        assert edge.disagreement_note is not None
        assert other_token in edge.disagreement_note

    def test_no_bridge_session_ids_falls_back_to_heuristics_unflagged(self, tmp_path: Path) -> None:
        """A session with no claude_bridge_session evidence of its own gets
        the plain heuristic result -- untagged commits are not disagreements."""
        _init_git_repo(tmp_path)
        sha = _commit(tmp_path, "src/main.py", "fix: ship it, no trailer at all\n")
        now = datetime.now(timezone.utc)

        messages = [
            {
                "id": "m1",
                "text": "editing src/main.py",
                "content_blocks": [{"type": "tool_use", "name": "Edit", "affected_paths": ["src/main.py"]}],
            }
        ]

        edges = detect_session_commits(
            session_id="claude-code-session:own-session",
            messages=messages,
            session_created_at=now - timedelta(minutes=5),
            session_updated_at=now,
            repo_path=str(tmp_path),
        )

        assert len(edges) == 1
        assert edges[0].commit_sha == sha
        assert edges[0].detection_method == "file_overlap"
        assert edges[0].disagreement_note is None

    def test_foreign_trailer_present_but_no_own_bridge_ids_is_not_a_disagreement(self, tmp_path: Path) -> None:
        """polylogue-2vor finding 3: when this session has no
        bridge_session_ids of its own, there is no typed identity to
        compare a commit's Claude-Session trailer against -- a trailer
        naming *some other* session is the expected case for any commit
        that isn't this session's, not evidence of misattribution. Unlike
        ``test_no_bridge_session_ids_falls_back_to_heuristics_unflagged``
        above (whose commit carries no trailer at all), this commit DOES
        carry a foreign trailer, which is exactly the shape that used to
        false-positive."""
        _init_git_repo(tmp_path)
        other_token = "01OtherSessionToken0000000"
        sha = _commit(
            tmp_path,
            "src/main.py",
            f"fix: ship it\n\nClaude-Session: https://claude.ai/code/session_{other_token}\n",
        )
        now = datetime.now(timezone.utc)

        messages = [
            {
                "id": "m1",
                "text": "editing src/main.py",
                "content_blocks": [{"type": "tool_use", "name": "Edit", "affected_paths": ["src/main.py"]}],
            }
        ]

        edges = detect_session_commits(
            session_id="claude-code-session:own-session",
            messages=messages,
            session_created_at=now - timedelta(minutes=5),
            session_updated_at=now,
            repo_path=str(tmp_path),
            bridge_session_ids=None,  # this session asserts no bridge identity of its own
        )

        assert len(edges) == 1
        edge = edges[0]
        assert edge.commit_sha == sha
        assert edge.detection_method == "file_overlap"
        assert edge.disagreement_note is None


class TestTypedRefsPreferredOverRegex:
    """polylogue-l9su AC2/AC4: typed session_refs-derived GitHubRefs are
    authoritative; the regex scan runs only to detect disagreement."""

    def test_typed_pr_ref_used_when_present(self) -> None:
        messages = [{"id": "m1", "text": "see #999 for context", "content_blocks": []}]
        typed_pr = GitHubRef(owner="Sinity", repo="polylogue", number=3265, kind="pr", url="https://x/pull/3265")

        result = build_correlation_result(
            session_id="s",
            messages=messages,
            repo_path="/nonexistent/path",
            typed_pr_refs=[typed_pr],
        )

        assert [r.number for r in result.pr_refs] == [3265]
        assert result.pr_refs[0].source == SOURCE_TYPED

    def test_regex_result_tagged_heuristic_when_no_typed_evidence(self) -> None:
        messages = [{"id": "m1", "text": "https://github.com/foo/bar/pull/42", "content_blocks": []}]
        result = build_correlation_result(session_id="s", messages=messages, repo_path="/nonexistent/path")
        assert len(result.pr_refs) == 1
        assert result.pr_refs[0].source == SOURCE_HEURISTIC

    def test_disagreement_recorded_when_regex_finds_untyped_pr(self) -> None:
        messages = [{"id": "m1", "text": "https://github.com/foo/bar/pull/42", "content_blocks": []}]
        typed_pr = GitHubRef(owner="Sinity", repo="polylogue", number=3265, kind="pr", url="https://x/pull/3265")

        result = build_correlation_result(
            session_id="s",
            messages=messages,
            repo_path="/nonexistent/path",
            typed_pr_refs=[typed_pr],
        )

        assert [r.number for r in result.pr_refs] == [3265]  # typed still wins
        assert any(d.kind == "pr_ref" for d in result.disagreements)
        pr_disagreement = next(d for d in result.disagreements if d.kind == "pr_ref")
        assert "42" in pr_disagreement.heuristic_values

    def test_disagreement_recorded_for_same_number_different_repo(self) -> None:
        """polylogue-2vor finding 2: comparing PR/issue identity by bare
        number alone means acme/product#42 and other/repo#42 compare
        equal, silently swallowing a real disagreement across
        differently-named repos. Typed evidence names Sinity/polylogue#42;
        the message text names a *different* repo's #42 -- that must still
        surface as a disagreement even though the numbers match."""
        messages = [{"id": "m1", "text": "https://github.com/other/repo/pull/42", "content_blocks": []}]
        typed_pr = GitHubRef(owner="Sinity", repo="polylogue", number=42, kind="pr", url="https://x/pull/42")

        result = build_correlation_result(
            session_id="s",
            messages=messages,
            repo_path="/nonexistent/path",
            typed_pr_refs=[typed_pr],
        )

        assert [r.number for r in result.pr_refs] == [42]  # typed still wins
        pr_disagreements = [d for d in result.disagreements if d.kind == "pr_ref"]
        assert len(pr_disagreements) == 1, (
            "other/repo#42 must not be silently treated as corroborating Sinity/polylogue#42"
        )
        assert "42" in pr_disagreements[0].heuristic_values


class TestTypedRefsFromSessionRefs:
    def test_converts_pull_request_kind(self) -> None:
        class _FakeRef:
            kind = "pull_request"
            repo = "Sinity/polylogue"
            number = 3265
            url = "https://github.com/Sinity/polylogue/pull/3265"

        pr_refs, issue_refs = typed_refs_from_session_refs([_FakeRef()])
        assert issue_refs == []
        assert len(pr_refs) == 1
        assert pr_refs[0].owner == "Sinity"
        assert pr_refs[0].repo == "polylogue"
        assert pr_refs[0].number == 3265
        assert pr_refs[0].kind == "pr"
        assert pr_refs[0].source == SOURCE_TYPED

    def test_ignores_unknown_kind(self) -> None:
        class _FakeRef:
            kind = "unknown"
            repo = None
            number = None
            url = None

        pr_refs, issue_refs = typed_refs_from_session_refs([_FakeRef()])
        assert pr_refs == []
        assert issue_refs == []

    def test_missing_number_with_non_github_url_is_skipped_not_pr_zero(self) -> None:
        """polylogue-2vor finding 1: Codex Cloud's
        chatgpt_codex_sidecar._pull_request_ref() stores
        external_pull_request_id in ``url`` and leaves ``repo``/``number``
        unset. Coercing that to number=0 fabricates a bogus "PR #0" that,
        because typed evidence is authoritative over the regex fallback,
        would silently suppress a correctly-parsed regex result. The row
        must be skipped instead of appearing as PR #0."""

        class _FakeRef:
            kind = "pull_request"
            repo = None
            number = None
            url = "task_e_68abcdef"  # opaque Codex Cloud id, not a github.com URL

        pr_refs, issue_refs = typed_refs_from_session_refs([_FakeRef()])
        assert pr_refs == []
        assert issue_refs == []
        assert all(ref.number != 0 for ref in pr_refs)

    def test_missing_number_is_parsed_from_github_url(self) -> None:
        """When the row lacks a typed number but ``url`` is a genuine
        github.com PR/issue URL, recover the number (and owner/repo, if
        not already set) from the URL rather than skipping it."""

        class _FakeRef:
            kind = "pull_request"
            repo = None
            number = None
            url = "https://github.com/Sinity/polylogue/pull/3265"

        pr_refs, issue_refs = typed_refs_from_session_refs([_FakeRef()])
        assert issue_refs == []
        assert len(pr_refs) == 1
        assert pr_refs[0].number == 3265
        assert pr_refs[0].owner == "Sinity"
        assert pr_refs[0].repo == "polylogue"
        assert pr_refs[0].source == SOURCE_TYPED


class TestBridgeSessionIdsFromEvents:
    def test_extracts_bridge_session_id(self) -> None:
        class _FakeEvent:
            event_type = "claude_bridge_session"
            payload = {"bridge_session_id": "cse_abc123"}

        assert bridge_session_ids_from_events([_FakeEvent()]) == ["cse_abc123"]

    def test_ignores_other_event_types(self) -> None:
        class _FakeEvent:
            event_type = "claude_pr_link"
            payload = {"pr_number": 1}

        assert bridge_session_ids_from_events([_FakeEvent()]) == []
