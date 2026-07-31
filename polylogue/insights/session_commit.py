"""Session-to-git-commit attribution (#1690 phase 2).

Detection of git commits likely produced by archived AI coding sessions
through time-window analysis, file-overlap scoring, and explicit reference
detection. Also extracts GitHub issue/PR references from session message text
(#1690 phase 3).

Edge insertion is idempotent by (session_id, commit_sha): re-running
detection for the same session replaces existing edges.

polylogue-l9su: this module's regex/time-window scan is a FALLBACK, not the
primary signal. Two typed evidence sources rank above it when the caller
supplies them:

- ``typed_pr_refs`` / ``typed_issue_refs`` -- ``GitHubRef`` rows built from
  ``session_refs`` (kind ``pull_request``/``issue``, itself sourced from the
  Claude Code ``pr-link`` sidecar record). When present for a session, the
  text-regex ``extract_github_refs`` scan of that session's messages is
  computed only to cross-check for disagreement, never used as the primary
  result.
- ``bridge_session_ids`` -- the session's own ``claude_bridge_session``
  event payloads (``cse_<token>``). Commits in this repo (and others
  following the same convention) carry a ``Claude-Session:
  https://claude.ai/code/session_<token>`` trailer; when a commit's trailer
  token matches one of the session's own bridge-session ids, that commit is
  attributed with ``detection_method="origin_reported"`` and
  ``confidence=1.0``, superseding the file-overlap/time-window/explicit-ref
  heuristics for that commit sha.

When a commit's trailer names a *different* session than the one currently
being scored, but the file-overlap/time-window heuristic would still have
attributed it to the current session, that is recorded as a disagreement
(``SessionCommitEdge.disagreement_note`` / ``SessionCorrelationResult
.disagreements``) rather than silently resolved either way.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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

# Match this repo's (and sinnix/sinex/lynchpin's) commit-trailer convention:
# "Claude-Session: https://claude.ai/code/session_<token>". The <token> is
# the same base62 id the Claude Code sidecar reports as
# ``claude_bridge_session``'s ``bridge_session_id`` with a ``cse_`` prefix
# instead of ``session_`` -- verified against the live archive (index.db
# session_events payload ``bridge_session_id":"cse_0182HDxDpJpsbn2qcKWK6Fsf"``
# matches this repo's own commit trailer ``session_0182HDxDpJpsbn2qcKWK6Fsf``
# byte-for-byte apart from the prefix).
_CLAUDE_SESSION_TRAILER_RE = re.compile(r"Claude-Session:\s*\S*?session_([A-Za-z0-9]+)")

# Sources a GitHubRef/SessionCommitEdge can carry, distinguishing typed
# provider evidence from the regex/time-window reconstruction fallback
# (polylogue-l9su: disagreements between the two must be visible, not
# silently resolved).
SOURCE_TYPED = "typed_session_ref"
SOURCE_HEURISTIC = "heuristic_regex"


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
    source: str = SOURCE_HEURISTIC
    """SOURCE_TYPED (from session_refs) or SOURCE_HEURISTIC (regex scan)."""


def extract_claude_session_trailer_tokens(commit_body: str) -> set[str]:
    """Extract ``Claude-Session:`` trailer tokens from a commit message body.

    Returns the raw ``<token>`` from ``.../code/session_<token>`` URLs, with
    no ``session_``/``cse_`` prefix -- callers compare against
    ``bridge_session_ids`` stripped of their ``cse_`` prefix the same way.
    """
    return {m.group(1) for m in _CLAUDE_SESSION_TRAILER_RE.finditer(commit_body)}


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
    detection_method: str  # "time_window", "file_overlap", "explicit_ref", "origin_reported"
    confidence: float  # 0.0 – 1.0
    file_overlap_count: int = 0
    repo_path: str | None = None
    disagreement_note: str | None = None
    """Set when a heuristic edge conflicts with a typed Claude-Session
    trailer naming a *different* session for the same commit -- the
    heuristic result is kept (not silently dropped) but flagged."""


@dataclass(frozen=True)
class CorrelationDisagreement:
    """A case where typed provider evidence and the regex/time-window
    heuristic disagree about the same session's git/GitHub evidence.

    polylogue-l9su AC4: disagreements must be surfaced, never silently
    resolved by preferring one signal.
    """

    kind: str  # "commit", "pr_ref", "issue_ref"
    session_id: str
    typed_values: tuple[str, ...]
    heuristic_values: tuple[str, ...]
    detail: str = ""


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
    disagreements: list[CorrelationDisagreement] = field(default_factory=list)


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


def _strip_bridge_session_prefix(bridge_session_id: str) -> str:
    """Normalize a ``claude_bridge_session`` id to the bare trailer token.

    ``claude_bridge_session`` payloads carry ``cse_<token>``; the
    ``Claude-Session:`` commit trailer carries ``session_<token>``. Both
    identify the same underlying id -- strip whichever prefix is present.
    """
    for prefix in ("cse_", "session_"):
        if bridge_session_id.startswith(prefix):
            return bridge_session_id[len(prefix) :]
    return bridge_session_id


def _git_log_commit_bodies(
    repo_path: str,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, str]:
    """Fetch full commit message bodies in the window, keyed by full sha.

    A separate invocation from the file-list scan: mixing ``--name-only``
    file lists with a multi-line ``%B`` body in one format string is not
    reliably separable, so this pass uses ASCII field/record separators
    that cannot collide with commit text.
    """
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
                "--format=%H%x1f%B%x1e",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}
    bodies: dict[str, str] = {}
    for block in result.stdout.split("\x1e"):
        block = block.strip("\n")
        if not block:
            continue
        sha, sep, body = block.partition("\x1f")
        sha = sha.strip()
        if sep and sha:
            bodies[sha] = body
    return bodies


def _flag_foreign_trailer(
    edge: SessionCommitEdge,
    *,
    foreign_trailer: bool,
    sha: str,
    session_id: str,
    trailer_tokens: set[str],
) -> SessionCommitEdge:
    """Attach a disagreement note when a heuristic edge's commit carries a
    Claude-Session trailer naming a *different* session than the one it was
    just heuristically attributed to."""
    if not foreign_trailer:
        return edge
    note = (
        f"commit {sha[:12]} heuristically attributed to {session_id} via "
        f"{edge.detection_method}, but its Claude-Session trailer names a "
        f"different session (token(s): {', '.join(sorted(trailer_tokens))})"
    )
    return replace(edge, disagreement_note=note)


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
    bridge_session_ids: Sequence[str] | None = None,
) -> list[SessionCommitEdge]:
    """Detect git commits likely produced by an archived AI coding session.

    Steps:
    1. Derive scan window from session timestamps (±window hours)
    2. Extract referenced files from session tool calls
    3. Find commits in the window via git log
    4. Check each commit's ``Claude-Session:`` trailer against the
       session's own ``bridge_session_ids`` (typed, highest priority --
       polylogue-l9su); a match short-circuits the heuristics below.
    5. Score remaining commits by file overlap / explicit sha ref /
       time window, flagging any commit whose trailer names a *different*
       session as a disagreement rather than silently accepting or
       dropping it.
    6. Return edges above the confidence threshold
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

    own_trailer_tokens = {_strip_bridge_session_prefix(bsid) for bsid in (bridge_session_ids or ()) if bsid}

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
                "--format=%x1e%H%x1f%ai%x1f%s",
                "--name-only",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    commits = _parse_git_log_blocks(result.stdout)
    if not commits:
        return []

    commit_bodies = _git_log_commit_bodies(repo_path, window_start, window_end)

    edges: list[SessionCommitEdge] = []
    for commit_data in commits:
        commit_files: set[str] = commit_data["files"]
        sha: str = commit_data["hash"]

        trailer_tokens = extract_claude_session_trailer_tokens(commit_bodies.get(sha, ""))

        # Typed evidence first: the commit's own trailer names this session.
        if trailer_tokens and (trailer_tokens & own_trailer_tokens):
            edges.append(
                SessionCommitEdge(
                    session_id=session_id,
                    commit_sha=sha,
                    detection_method="origin_reported",
                    confidence=1.0,
                    file_overlap_count=len(session_files & commit_files),
                    repo_path=repo_path,
                )
            )
            continue

        # A disagreement requires something to disagree *with*: when this
        # session has no bridge/trailer tokens of its own (own_trailer_tokens
        # empty), there is no typed identity to compare a commit's trailer
        # against, so a trailer naming *some* other session is not evidence
        # of misattribution -- it is simply the expected case for a commit
        # from any other session (polylogue-2vor finding 3).
        foreign_trailer = (
            bool(trailer_tokens) and bool(own_trailer_tokens) and not (trailer_tokens & own_trailer_tokens)
        )

        # Check for explicit ref first (highest confidence)
        if sha.lower() in commit_sha_refs or sha[:8].lower() in commit_sha_refs:
            edges.append(
                _flag_foreign_trailer(
                    SessionCommitEdge(
                        session_id=session_id,
                        commit_sha=sha,
                        detection_method="explicit_ref",
                        confidence=0.95,
                        file_overlap_count=len(session_files & commit_files),
                        repo_path=repo_path,
                    ),
                    foreign_trailer=foreign_trailer,
                    sha=sha,
                    session_id=session_id,
                    trailer_tokens=trailer_tokens,
                )
            )
            continue

        # File overlap scoring
        if session_files:
            overlap = session_files & commit_files
            confidence = score_file_overlap(commit_files, session_files)
            if confidence >= confidence_threshold:
                edges.append(
                    _flag_foreign_trailer(
                        SessionCommitEdge(
                            session_id=session_id,
                            commit_sha=sha,
                            detection_method="file_overlap",
                            confidence=round(confidence, 4),
                            file_overlap_count=len(overlap),
                            repo_path=repo_path,
                        ),
                        foreign_trailer=foreign_trailer,
                        sha=sha,
                        session_id=session_id,
                        trailer_tokens=trailer_tokens,
                    )
                )
            elif commit_files:
                # Time-window only (low-confidence fallback)
                tw_confidence = 0.1
                if tw_confidence >= confidence_threshold:
                    edges.append(
                        _flag_foreign_trailer(
                            SessionCommitEdge(
                                session_id=session_id,
                                commit_sha=sha,
                                detection_method="time_window",
                                confidence=tw_confidence,
                                file_overlap_count=0,
                                repo_path=repo_path,
                            ),
                            foreign_trailer=foreign_trailer,
                            sha=sha,
                            session_id=session_id,
                            trailer_tokens=trailer_tokens,
                        )
                    )

    return edges


def _parse_git_log_blocks(output: str) -> list[dict[str, Any]]:
    """Parse git log output into commit dicts including changed files.

    Expects ``--format=%x1e%H%x1f%ai%x1f%s --name-only``: each commit's
    header fields are ASCII-unit-separator-delimited on one line, the
    per-commit block is ASCII-record-separator-delimited, and the changed
    files (if any) follow on their own lines. Using real field/record
    separators (rather than a literal ``---`` token, which cannot be
    distinguished from git's own blank-line entry separator once
    ``--name-only`` appends a file list after the format text) is what
    makes ``changed_files`` non-empty -- a bare ``\\n---\\n`` split
    previously left every commit's file set empty because the file list
    for commit N lived in the *next* split segment, past the point the
    parser had already stopped consuming header lines for commit N.
    """
    commits: list[dict[str, Any]] = []
    for block in output.split("\x1e"):
        if not block:
            continue
        header, _, rest = block.partition("\n")
        parts = header.split("\x1f")
        if len(parts) != 3:
            continue
        commit_hash, date, subject = parts
        commit_hash = commit_hash.strip()
        if not commit_hash:
            continue
        changed_files = {f.strip() for f in rest.split("\n") if f.strip()}
        commits.append(
            {
                "hash": commit_hash,
                "date": date.strip(),
                "subject": subject.strip(),
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
    typed_pr_refs: Sequence[GitHubRef] | None = None,
    typed_issue_refs: Sequence[GitHubRef] | None = None,
    bridge_session_ids: Sequence[str] | None = None,
) -> SessionCorrelationResult:
    """Build a complete SessionCorrelationResult for a session.

    Combines Phase 2 (commit attribution) and Phase 3 (issue/PR extraction).

    ``typed_pr_refs``/``typed_issue_refs`` (from ``session_refs``, kind
    ``pull_request``/``issue``) and ``bridge_session_ids`` (from the
    session's own ``claude_bridge_session`` events) are typed provider
    evidence -- see the module docstring. When supplied they are the
    authoritative result for that kind; the regex scan of message text is
    still run (it is needed for file-path/window extraction regardless) and
    is used only as a fallback, or to detect a disagreement against typed
    evidence, never to override it silently.
    """
    window_start, window_end = derive_scan_window(
        session_created_at,
        session_updated_at,
        before_hours=before_hours,
        after_hours=after_hours,
    )

    # Detect commits (typed Claude-Session trailer match takes priority
    # inside detect_session_commits when bridge_session_ids is supplied).
    edges = detect_session_commits(
        session_id=session_id,
        messages=messages,
        session_created_at=session_created_at,
        session_updated_at=session_updated_at,
        repo_path=repo_path,
        before_hours=before_hours,
        after_hours=after_hours,
        confidence_threshold=confidence_threshold,
        bridge_session_ids=bridge_session_ids,
    )

    disagreements: list[CorrelationDisagreement] = []
    for edge in edges:
        if edge.disagreement_note:
            disagreements.append(
                CorrelationDisagreement(
                    kind="commit",
                    session_id=session_id,
                    typed_values=(),
                    heuristic_values=(edge.commit_sha,),
                    detail=edge.disagreement_note,
                )
            )

    # Regex scan of message text -- always computed (it is the only source
    # for file_paths and remains the fallback for sessions with no typed
    # session_refs evidence).
    heuristic_issue_refs: list[GitHubRef] = []
    heuristic_pr_refs: list[GitHubRef] = []

    for msg in messages:
        msg_id = msg.get("id")
        msg_id_str = str(msg_id) if msg_id else None
        text = msg.get("text")
        if isinstance(text, str):
            refs = extract_github_refs(text, message_id=msg_id_str)
            for ref in refs:
                if ref.kind == "pr":
                    heuristic_pr_refs.append(ref)
                else:
                    heuristic_issue_refs.append(ref)

    pr_refs, pr_disagreement = _resolve_refs(
        kind="pr_ref",
        session_id=session_id,
        typed_refs=typed_pr_refs,
        heuristic_refs=heuristic_pr_refs,
    )
    if pr_disagreement is not None:
        disagreements.append(pr_disagreement)

    issue_refs, issue_disagreement = _resolve_refs(
        kind="issue_ref",
        session_id=session_id,
        typed_refs=typed_issue_refs,
        heuristic_refs=heuristic_issue_refs,
    )
    if issue_disagreement is not None:
        disagreements.append(issue_disagreement)

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
        disagreements=disagreements,
    )


def _refs_match(a: GitHubRef, b: GitHubRef) -> bool:
    """True when two refs plausibly identify the same PR/issue.

    Bare number equality is not sufficient once a ref carries repo
    identity: ``acme/product#42`` and ``other/repo#42`` are different
    objects that happen to share a number (polylogue-2vor finding 2).
    Compare the full ``(owner, repo, number)`` identity whenever *both*
    refs are repo-qualified; fall back to number-only equality when either
    side lacks repo identity (e.g. a bare ``#42`` regex mention), since the
    number is the only signal available there.
    """
    if a.number != b.number:
        return False
    if a.owner and a.repo and b.owner and b.repo:
        return a.owner.lower() == b.owner.lower() and a.repo.lower() == b.repo.lower()
    return True


def _resolve_refs(
    *,
    kind: str,
    session_id: str,
    typed_refs: Sequence[GitHubRef] | None,
    heuristic_refs: Sequence[GitHubRef],
) -> tuple[list[GitHubRef], CorrelationDisagreement | None]:
    """Prefer typed refs over the regex/heuristic scan for one ref kind.

    Returns the resolved ref list plus a disagreement record when the
    heuristic scan found numbers the typed evidence does not corroborate
    (polylogue-l9su AC2/AC4: typed is authoritative, heuristic is fallback
    only, disagreements are surfaced rather than silently dropped).
    """
    if not typed_refs:
        return list(heuristic_refs), None

    resolved = [replace(ref, source=SOURCE_TYPED) for ref in typed_refs]
    extra = [h for h in heuristic_refs if not any(_refs_match(h, t) for t in resolved)]
    if not extra:
        return resolved, None
    typed_numbers = {ref.number for ref in resolved}
    extra_numbers = {ref.number for ref in extra}
    return (
        resolved,
        CorrelationDisagreement(
            kind=kind,
            session_id=session_id,
            typed_values=tuple(sorted(str(n) for n in typed_numbers)),
            heuristic_values=tuple(sorted(str(n) for n in extra_numbers)),
            detail=(
                f"regex-scanned message text references {kind} number(s) "
                f"{sorted(extra_numbers)} not present in typed session_refs evidence "
                f"({sorted(typed_numbers)})"
            ),
        ),
    )


def typed_refs_from_session_refs(refs: Sequence[Any]) -> tuple[list[GitHubRef], list[GitHubRef]]:
    """Convert ``session_refs`` rows into typed ``(pr_refs, issue_refs)``.

    Accepts anything exposing ``.kind``/``.repo``/``.number``/``.url``
    (``SessionRefRecord``) so callers on either the sync or async
    repository path can build the same typed evidence without this module
    importing the storage layer's record types (kept dependency-light /
    testable without a DB).
    """
    pr_refs: list[GitHubRef] = []
    issue_refs: list[GitHubRef] = []
    for ref in refs:
        kind = getattr(ref, "kind", None)
        if kind not in ("pull_request", "issue"):
            continue
        repo = getattr(ref, "repo", None)
        owner_name: str | None = None
        repo_name: str | None = None
        if isinstance(repo, str) and "/" in repo:
            owner_name, _, repo_name = repo.partition("/")
        elif isinstance(repo, str) and repo:
            repo_name = repo
        url = getattr(ref, "url", None)
        url_str = url if isinstance(url, str) else None

        raw_number = getattr(ref, "number", None)
        number: int | None = raw_number if isinstance(raw_number, int) and raw_number > 0 else None
        if number is None and url_str is not None:
            # No typed number on the row (observed for Codex Cloud's
            # chatgpt_codex_sidecar._pull_request_ref(), which stores
            # external_pull_request_id in url and leaves repo/number
            # unset). Try to recover a real number by parsing a genuine
            # github.com PR/issue URL out of it.
            url_pattern = _GITHUB_PR_URL_RE if kind == "pull_request" else _GITHUB_ISSUE_URL_RE
            match = url_pattern.search(url_str)
            if match:
                owner_name = owner_name or match.group(1)
                repo_name = repo_name or match.group(2)
                number = int(match.group(3))
        if number is None:
            # Still nothing usable (e.g. url holds an opaque non-GitHub
            # id). Coercing this to number=0 would fabricate a bogus
            # "PR #0" that -- because typed evidence is authoritative --
            # would silently outrank a correctly-parsed regex fallback
            # result (polylogue-2vor finding 1). Skip the row instead.
            continue

        built = GitHubRef(
            owner=owner_name,
            repo=repo_name,
            number=number,
            kind="pr" if kind == "pull_request" else "issue",
            url=url_str,
            raw_match=url_str or "",
            source=SOURCE_TYPED,
        )
        if kind == "pull_request":
            pr_refs.append(built)
        else:
            issue_refs.append(built)
    return pr_refs, issue_refs


def bridge_session_ids_from_events(events: Sequence[Any]) -> list[str]:
    """Extract ``claude_bridge_session`` ids from a session's own events.

    Accepts anything exposing ``.event_type``/``.payload`` (``SessionEvent``
    or ``SessionEventRecord``).
    """
    ids: list[str] = []
    for event in events:
        if getattr(event, "event_type", None) != "claude_bridge_session":
            continue
        payload = getattr(event, "payload", None) or {}
        bridge_session_id = payload.get("bridge_session_id") if isinstance(payload, dict) else None
        if isinstance(bridge_session_id, str) and bridge_session_id:
            ids.append(bridge_session_id)
    return ids


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
                "disagreement_note": c.disagreement_note,
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
                "source": r.source,
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
                "source": r.source,
                "object_ref": _github_ref_object_ref(r).format(),
            }
            for r in result.pr_refs
        ],
        "file_paths": result.file_paths,
        "file_refs": file_refs,
        "object_refs": [*commit_refs, *issue_object_refs, *pr_object_refs, *file_refs],
        "disagreements": [
            {
                "kind": d.kind,
                "session_id": d.session_id,
                "typed_values": list(d.typed_values),
                "heuristic_values": list(d.heuristic_values),
                "detail": d.detail,
            }
            for d in result.disagreements
        ],
    }


def _github_ref_object_ref(ref: GitHubRef) -> ObjectRef:
    object_id = f"{ref.owner}/{ref.repo}#{ref.number}" if ref.owner and ref.repo else ref.raw_match
    if ref.kind == "pr":
        return ObjectRef(kind="github-pr", object_id=object_id)
    return ObjectRef(kind="github-issue", object_id=object_id)


__all__ = [
    "CorrelationDisagreement",
    "GitHubRef",
    "SOURCE_HEURISTIC",
    "SOURCE_TYPED",
    "SessionCommitEdge",
    "SessionCorrelationResult",
    "bridge_session_ids_from_events",
    "build_correlation_result",
    "correlation_result_to_payload",
    "derive_scan_window",
    "detect_session_commits",
    "extract_claude_session_trailer_tokens",
    "extract_github_refs",
    "extract_referenced_files",
    "score_file_overlap",
    "typed_refs_from_session_refs",
    "_DEFAULT_CONFIDENCE_THRESHOLD",
]
