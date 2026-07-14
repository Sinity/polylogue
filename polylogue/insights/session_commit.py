"""Session-to-git-commit attribution (#1690 phase 2).

Detection of git commits likely produced by archived AI coding sessions
through time-window analysis, file-overlap scoring, and explicit reference
detection. Also extracts GitHub issue/PR references from session message text
(#1690 phase 3).

Edge insertion is idempotent by (session_id, commit_sha): re-running
detection for the same session replaces existing edges.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from polylogue.core.refs import ObjectRef

# ── GitHub Issue / PR reference extraction (#1690 phase 3) ──────────────

# Match full GitHub URLs: https://github.com/owner/repo/issues/123
_GITHUB_ISSUE_URL_RE = re.compile(r"https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)/issues/(\d+)")
_GITHUB_PR_URL_RE = re.compile(r"https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)/pull/(\d+)")

# Match shorthand owner/repo#NNN or owner/repo#NNN
_SHORTHAND_REPO_REF_RE = re.compile(r"\b([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)#(\d+)\b")

# Match bare #NNN (must be preceded by word boundary, not part of heading)
_BARE_NUM_REF_RE = re.compile(r"(?<!\w)#(\d{1,6})\b")

# Match commit SHA references (full 40-char or short 7-14 char)
_COMMIT_SHA_RE = re.compile(r"\b([0-9a-f]{7,40})\b", re.IGNORECASE)


@dataclass(frozen=True)
class GitHubRef:
    """A GitHub issue or PR reference extracted from session text."""

    owner: str | None = None
    repo: str | None = None
    number: int = 0
    kind: str = "issue"  # "issue" or "pr"
    url: str | None = None
    raw_match: str = ""
    message_id: str | None = None
    """ID of the message where this ref was found, if known."""


def extract_github_refs(text: str, *, message_id: str | None = None) -> list[GitHubRef]:
    """Extract all GitHub issue and PR references from text.

    Detects: full URLs, owner/repo#NNN, and bare #NNN references.
    """
    results: list[GitHubRef] = []
    seen: set[tuple[str, int]] = set()

    # Full issue URLs
    for match in _GITHUB_ISSUE_URL_RE.finditer(text):
        owner = match.group(1)
        repo = match.group(2)
        number = int(match.group(3))
        key = (f"{owner}/{repo}", number)
        if key not in seen:
            seen.add(key)
            results.append(
                GitHubRef(
                    owner=owner,
                    repo=repo,
                    number=number,
                    kind="issue",
                    url=match.group(0),
                    raw_match=match.group(0),
                    message_id=message_id,
                )
            )

    # Full PR URLs
    for match in _GITHUB_PR_URL_RE.finditer(text):
        owner = match.group(1)
        repo = match.group(2)
        number = int(match.group(3))
        key = (f"{owner}/{repo}", number)
        if key not in seen:
            seen.add(key)
            results.append(
                GitHubRef(
                    owner=owner,
                    repo=repo,
                    number=number,
                    kind="pr",
                    url=match.group(0),
                    raw_match=match.group(0),
                    message_id=message_id,
                )
            )

    # Shorthand owner/repo#NNN
    for match in _SHORTHAND_REPO_REF_RE.finditer(text):
        owner = match.group(1)
        repo = match.group(2)
        number = int(match.group(3))
        key = (f"{owner}/{repo}", number)
        if key not in seen:
            seen.add(key)
            results.append(
                GitHubRef(
                    owner=owner,
                    repo=repo,
                    number=number,
                    kind="issue",
                    raw_match=match.group(0),
                    message_id=message_id,
                )
            )

    # Bare #NNN
    for match in _BARE_NUM_REF_RE.finditer(text):
        number = int(match.group(1))
        # Skip if already captured (e.g., inside owner/repo#NNN)
        # The shorthand regex has already consumed these.
        full_match = match.group(0)
        # Conservatively skip 4-digit numbers that could be heading anchors
        # and very large numbers that cannot be real issue numbers.
        if number < 1 or number > 999999:
            continue
        key = ("_bare", number)
        if key not in seen:
            seen.add(key)
            results.append(
                GitHubRef(
                    number=number,
                    kind="issue",
                    raw_match=full_match,
                    message_id=message_id,
                )
            )

    return results


# ── File path extraction ────────────────────────────────────────────────


def extract_referenced_files(messages: Sequence[dict[str, Any]]) -> set[str]:
    """Extract file paths referenced in tool calls across session messages.

    Scans content_blocks for tool_use blocks that carry ``affected_paths``
    and for text content that contains file path patterns.
    """
    paths: set[str] = set()

    for msg in messages:
        content_blocks = msg.get("content_blocks") or []
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    # Check for affected_paths in the block dict
                    affected = block.get("affected_paths")
                    if isinstance(affected, list):
                        for p in affected:
                            if isinstance(p, str) and p.strip():
                                paths.add(p.strip())
                    # Also check input dict for path-like fields
                    inp = block.get("input")
                    if isinstance(inp, dict):
                        for key in ("file_path", "filePath", "path", "target_file"):
                            val = inp.get(key)
                            if isinstance(val, str) and val.strip():
                                paths.add(val.strip())

        # Also scan message text for file paths (common in tool calls)
        text = msg.get("text")
        if isinstance(text, str) and text:
            # Rough path detection: lines that look like file references
            for line in text.split("\n"):
                line = line.strip()
                if line and ("/" in line or line.endswith(".py") or line.endswith(".rs")):
                    paths.add(line)

    return paths


# ── Commit detection ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionCommitEdge:
    """A detected link between a session and a git commit."""

    session_id: str
    commit_sha: str
    detection_method: str  # "time_window", "file_overlap", "explicit_ref"
    confidence: float  # 0.0 – 1.0
    file_overlap_count: int = 0
    repo_path: str | None = None


@dataclass(frozen=True)
class SessionCorrelationResult:
    """Full correlation result for a session (#1690 phase 2+3)."""

    session_id: str
    window_start: str
    window_end: str
    repo: str | None = None
    commits: list[SessionCommitEdge] = field(default_factory=list)
    issue_refs: list[GitHubRef] = field(default_factory=list)
    pr_refs: list[GitHubRef] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)


def derive_scan_window(
    created_at: datetime | None,
    updated_at: datetime | None,
    *,
    before_hours: int = 2,
    after_hours: int = 2,
) -> tuple[datetime, datetime]:
    """Derive a git-scan time window from session timestamps.

    Returns (window_start, window_end) as timezone-aware datetimes.
    The window extends ``before_hours`` before ``created_at`` and
    ``after_hours`` after ``updated_at``.
    """
    now = datetime.now(timezone.utc)
    start = created_at or now
    end = updated_at or now
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (start - timedelta(hours=before_hours), end + timedelta(hours=after_hours))


def score_file_overlap(commit_files: set[str], session_files: set[str]) -> float:
    """Score a commit by file overlap with session-referenced files.

    Returns a float 0.0–1.0 where 1.0 means every session file appears
    in the commit's changed files.
    """
    if not session_files:
        return 0.0
    overlap = session_files & commit_files
    return len(overlap) / len(session_files)


_DEFAULT_CONFIDENCE_THRESHOLD = 0.3


def detect_session_commits(
    session_id: str,
    messages: Sequence[dict[str, Any]],
    session_created_at: datetime | None = None,
    session_updated_at: datetime | None = None,
    *,
    repo_path: str = ".",
    before_hours: int = 2,
    after_hours: int = 2,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[SessionCommitEdge]:
    """Detect git commits likely produced by an archived AI coding session.

    Steps:
    1. Derive scan window from session timestamps (±window hours)
    2. Extract referenced files from session tool calls
    3. Find commits in the window via git log
    4. Score each commit by file overlap with session files
    5. Return edges above the confidence threshold
    """
    window_start, window_end = derive_scan_window(
        session_created_at,
        session_updated_at,
        before_hours=before_hours,
        after_hours=after_hours,
    )

    # Extract session files
    session_files = extract_referenced_files(messages)

    # Collect all message texts for explicit ref detection
    all_text = " ".join(msg.get("text", "") or "" for msg in messages if isinstance(msg.get("text"), str))
    commit_sha_refs = {cast(str, m.group(1)).lower() for m in _COMMIT_SHA_RE.finditer(all_text)}

    # Run git log
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "log",
                "--since",
                window_start.isoformat(),
                "--until",
                window_end.isoformat(),
                "--format=%H%n%ai%n%s%n---",
                "--name-only",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    commits = _parse_git_log_blocks(result.stdout)

    edges: list[SessionCommitEdge] = []
    for commit_data in commits:
        commit_files: set[str] = commit_data["files"]
        sha: str = commit_data["hash"]

        # Check for explicit ref first (highest confidence)
        if sha.lower() in commit_sha_refs or sha[:8].lower() in commit_sha_refs:
            edges.append(
                SessionCommitEdge(
                    session_id=session_id,
                    commit_sha=sha,
                    detection_method="explicit_ref",
                    confidence=0.95,
                    file_overlap_count=len(session_files & commit_files),
                    repo_path=repo_path,
                )
            )
            continue

        # File overlap scoring
        if session_files:
            overlap = session_files & commit_files
            confidence = score_file_overlap(commit_files, session_files)
            if confidence >= confidence_threshold:
                edges.append(
                    SessionCommitEdge(
                        session_id=session_id,
                        commit_sha=sha,
                        detection_method="file_overlap",
                        confidence=round(confidence, 4),
                        file_overlap_count=len(overlap),
                        repo_path=repo_path,
                    )
                )
            elif commit_files:
                # Time-window only (low-confidence fallback)
                tw_confidence = 0.1
                if tw_confidence >= confidence_threshold:
                    edges.append(
                        SessionCommitEdge(
                            session_id=session_id,
                            commit_sha=sha,
                            detection_method="time_window",
                            confidence=tw_confidence,
                            file_overlap_count=0,
                            repo_path=repo_path,
                        )
                    )

    return edges


def _parse_git_log_blocks(output: str) -> list[dict[str, Any]]:
    """Parse git log output into commit dicts including changed files."""
    commits: list[dict[str, Any]] = []
    blocks = output.split("\n---\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        commit_hash = lines[0].strip()
        changed_files = {f.strip() for f in lines[3:] if f.strip()}
        commits.append(
            {
                "hash": commit_hash,
                "date": lines[1].strip(),
                "subject": lines[2].strip(),
                "files": changed_files,
            }
        )
    return commits


def build_correlation_result(
    session_id: str,
    messages: Sequence[dict[str, Any]],
    session_created_at: datetime | None = None,
    session_updated_at: datetime | None = None,
    *,
    repo_path: str = ".",
    before_hours: int = 2,
    after_hours: int = 2,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> SessionCorrelationResult:
    """Build a complete SessionCorrelationResult for a session.

    Combines Phase 2 (commit attribution) and Phase 3 (issue/PR extraction).
    """
    window_start, window_end = derive_scan_window(
        session_created_at,
        session_updated_at,
        before_hours=before_hours,
        after_hours=after_hours,
    )

    # Detect commits
    edges = detect_session_commits(
        session_id=session_id,
        messages=messages,
        session_created_at=session_created_at,
        session_updated_at=session_updated_at,
        repo_path=repo_path,
        before_hours=before_hours,
        after_hours=after_hours,
        confidence_threshold=confidence_threshold,
    )

    # Extract issue/PR references from message text
    issue_refs: list[GitHubRef] = []
    pr_refs: list[GitHubRef] = []

    for msg in messages:
        msg_id = msg.get("id")
        msg_id_str = str(msg_id) if msg_id else None
        text = msg.get("text")
        if isinstance(text, str):
            refs = extract_github_refs(text, message_id=msg_id_str)
            for ref in refs:
                if ref.kind == "pr":
                    pr_refs.append(ref)
                else:
                    issue_refs.append(ref)

    # Extract session file paths
    session_files = sorted(extract_referenced_files(messages))

    return SessionCorrelationResult(
        session_id=session_id,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        repo=repo_path if repo_path != "." else None,
        commits=edges,
        issue_refs=issue_refs,
        pr_refs=pr_refs,
        file_paths=session_files,
    )


# ── Database operations ─────────────────────────────────────────────────


def correlation_result_to_payload(
    result: SessionCorrelationResult,
) -> dict[str, object]:
    """Convert a SessionCorrelationResult to a JSON-serializable dict."""
    commit_refs = [ObjectRef(kind="commit", object_id=commit.commit_sha).format() for commit in result.commits]
    issue_object_refs = [_github_ref_object_ref(ref).format() for ref in result.issue_refs]
    pr_object_refs = [_github_ref_object_ref(ref).format() for ref in result.pr_refs]
    file_refs = [ObjectRef(kind="file", object_id=path).format() for path in result.file_paths]

    return {
        "session_id": result.session_id,
        "window_start": result.window_start,
        "window_end": result.window_end,
        "repo": result.repo,
        "commits": [
            {
                "commit_sha": c.commit_sha,
                "short_sha": c.commit_sha[:8],
                "object_ref": ObjectRef(kind="commit", object_id=c.commit_sha).format(),
                "detection_method": c.detection_method,
                "confidence": c.confidence,
                "file_overlap_count": c.file_overlap_count,
            }
            for c in result.commits
        ],
        "issue_refs": [
            {
                "owner": r.owner,
                "repo": r.repo,
                "number": r.number,
                "kind": r.kind,
                "url": r.url or f"https://github.com/{r.owner}/{r.repo}/issues/{r.number}"
                if r.owner and r.repo
                else None,
                "raw_match": r.raw_match,
                "message_id": r.message_id,
                "object_ref": _github_ref_object_ref(r).format(),
            }
            for r in result.issue_refs
        ],
        "pr_refs": [
            {
                "owner": r.owner,
                "repo": r.repo,
                "number": r.number,
                "kind": r.kind,
                "url": r.url or f"https://github.com/{r.owner}/{r.repo}/pull/{r.number}"
                if r.owner and r.repo
                else None,
                "raw_match": r.raw_match,
                "message_id": r.message_id,
                "object_ref": _github_ref_object_ref(r).format(),
            }
            for r in result.pr_refs
        ],
        "file_paths": result.file_paths,
        "file_refs": file_refs,
        "object_refs": [*commit_refs, *issue_object_refs, *pr_object_refs, *file_refs],
    }


def _github_ref_object_ref(ref: GitHubRef) -> ObjectRef:
    object_id = f"{ref.owner}/{ref.repo}#{ref.number}" if ref.owner and ref.repo else ref.raw_match
    if ref.kind == "pr":
        return ObjectRef(kind="github-pr", object_id=object_id)
    return ObjectRef(kind="github-issue", object_id=object_id)


__all__ = [
    "GitHubRef",
    "SessionCommitEdge",
    "SessionCorrelationResult",
    "build_correlation_result",
    "correlation_result_to_payload",
    "derive_scan_window",
    "detect_session_commits",
    "extract_github_refs",
    "extract_referenced_files",
    "score_file_overlap",
    "_DEFAULT_CONFIDENCE_THRESHOLD",
]
