"""Path and repository attribution for conversations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from polylogue.archive.action_event.action_events import ActionEvent
from polylogue.archive.semantic.facts import ConversationSemanticFacts, build_conversation_semantic_facts
from polylogue.archive.viewport.tools import looks_like_path_candidate
from polylogue.lib.repo_identity import (
    normalize_repo_name,
    normalize_repo_names,
    normalize_repo_path,
    normalize_repo_paths,
)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation

_LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".nix": "nix",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".zsh": "zsh",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".sql": "sql",
    ".r": "r",
    ".R": "r",
    ".toml": "toml",
    ".yaml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
}
_DIALOGUE_LANGUAGE_HINT_PATTERNS = {
    lang_name: re.compile(rf"\b{re.escape(lang_name)}\b")
    for lang_name in ("python", "rust", "typescript", "javascript", "nix", "go", "java", "ruby", "sql")
}
_IGNORED_ABSOLUTE_PATH_PARTS = frozenset({".snapshot", ".snapshots", "tool-results"})
_IGNORED_RELATIVE_PATH_PREFIXES = (
    ".btrfs/snapshot",
    ".claude",
    ".codex",
    ".snapshot",
    ".snapshots",
)


def _is_ignored_absolute_path(path: PurePosixPath) -> bool:
    parts = tuple(part for part in path.parts if part != "/")
    if not parts:
        return True
    if any(part in _IGNORED_ABSOLUTE_PATH_PARTS for part in parts):
        return True
    if parts[:2] == ("nix", "store"):
        return True
    if parts[0] == "tmp" and any(part.startswith("claude-") or part.startswith("codex-") for part in parts[1:]):
        return True
    if parts[:2] == ("home", Path.home().name):
        if parts[2:3] in ((".claude",), (".codex",)):
            return True
        if parts[2:5] == (".config", "claude", "projects"):
            return True
        if parts[2:4] in ((".config", "claude"), (".config", "codex")):
            return True
        if parts[2:4] == (".claude", "projects"):
            return True
        if parts[2:4] == (".codex", "sessions"):
            return True
    return False


def _repo_root_from_path(path: str) -> str | None:
    """Derive a likely repository root from a file path."""
    return normalize_repo_path(path)


def _is_ignored_repo_relative_path(path: PurePosixPath) -> bool:
    candidate = path.as_posix()
    return any(candidate == prefix or candidate.startswith(f"{prefix}/") for prefix in _IGNORED_RELATIVE_PATH_PREFIXES)


def _language_from_path(path: str) -> str | None:
    """Detect programming language from file extension."""
    suffix = PurePosixPath(path).suffix.lower()
    return _LANGUAGE_EXTENSIONS.get(suffix)


def _clean_attributed_path(path: str) -> str | None:
    candidate = path.rstrip(".,;:)'\">`*_")
    if not candidate:
        return None
    if "<" in candidate or ">" in candidate:
        return None
    if any(ch in candidate for ch in "*?[]{}"):
        return None
    if set(candidate) <= {".", "/"}:
        return None
    if not candidate.startswith(("/", "~/")):
        if not looks_like_path_candidate(candidate):
            return None
        if _is_ignored_repo_relative_path(PurePosixPath(candidate)):
            return None
        return candidate
    expanded_path = Path(candidate).expanduser()
    expanded = str(expanded_path)
    resolved = str(expanded_path.resolve(strict=False))
    repo_root = _repo_root_from_path(resolved)
    if repo_root is not None:
        try:
            repo_relative = PurePosixPath(Path(resolved).relative_to(repo_root).as_posix())
        except ValueError:
            repo_relative = None
        if repo_relative is not None and _is_ignored_repo_relative_path(repo_relative):
            return None
        return resolved

    pure_path = PurePosixPath(expanded)
    if _is_ignored_absolute_path(pure_path):
        return None

    parts = [part for part in pure_path.parts if part != "/"]
    if len(parts) < 2:
        return None
    if pure_path.suffix:
        return expanded
    return None


def _add_repo_candidate_from_path(value: str, *, repo_paths: set[str], repo_names: set[str]) -> None:
    repo = _repo_root_from_path(value)
    if repo is not None:
        repo_paths.add(repo)
        repo_name = normalize_repo_name(repo)
        if repo_name:
            repo_names.add(repo_name)
        return


def _add_repo_candidate_from_hint(value: str, *, repo_paths: set[str], repo_names: set[str]) -> None:
    _add_repo_candidate_from_path(value, repo_paths=repo_paths, repo_names=repo_names)
    repo_name = normalize_repo_name(value)
    if repo_name:
        repo_names.add(repo_name)


@dataclass(frozen=True)
class ConversationAttribution:
    """Raw path/repo/branch attribution extracted from a conversation."""

    repo_paths: tuple[str, ...]
    repo_names: tuple[str, ...]
    cwd_paths: tuple[str, ...]
    branch_names: tuple[str, ...]
    file_paths_touched: tuple[str, ...]
    languages_detected: tuple[str, ...]


def extract_attribution_from_action_events(
    actions: tuple[ActionEvent, ...] | list[ActionEvent],
    *,
    provider_meta: dict[str, object] | None = None,
) -> ConversationAttribution:
    """Extract attribution from canonical action events plus optional provider metadata."""
    repo_paths: set[str] = set()
    repo_names: set[str] = set()
    cwd_paths: set[str] = set()
    branch_names: set[str] = set()
    file_paths: set[str] = set()
    languages: set[str] = set()

    provider_meta = provider_meta if isinstance(provider_meta, dict) else {}
    cwd_value = provider_meta.get("cwd")
    if isinstance(cwd_value, str) and cwd_value:
        cwd_paths.add(cwd_value)
        _add_repo_candidate_from_path(cwd_value, repo_paths=repo_paths, repo_names=repo_names)
    working_dirs = provider_meta.get("working_directories") or []
    if isinstance(working_dirs, (list, tuple)):
        for working_directory in working_dirs:
            if isinstance(working_directory, str) and working_directory:
                cwd_paths.add(working_directory)
                _add_repo_candidate_from_path(working_directory, repo_paths=repo_paths, repo_names=repo_names)

    git_branch = provider_meta.get("gitBranch")
    if isinstance(git_branch, str) and git_branch:
        branch_names.add(git_branch)

    git_meta = provider_meta.get("git")
    if isinstance(git_meta, dict):
        branch = git_meta.get("branch")
        if isinstance(branch, str) and branch:
            branch_names.add(branch)
        repository_url = git_meta.get("repository_url")
        if isinstance(repository_url, str) and repository_url:
            _add_repo_candidate_from_hint(repository_url, repo_paths=repo_paths, repo_names=repo_names)

    for action in actions:
        if action.cwd_path:
            cwd_paths.add(action.cwd_path)
            _add_repo_candidate_from_path(action.cwd_path, repo_paths=repo_paths, repo_names=repo_names)
        for branch in action.branch_names:
            branch_names.add(branch)
        for path in action.affected_paths:
            clean_path = _clean_attributed_path(path)
            if clean_path is None:
                continue
            file_paths.add(clean_path)
            lang = _language_from_path(clean_path)
            if lang:
                languages.add(lang)
            _add_repo_candidate_from_path(clean_path, repo_paths=repo_paths, repo_names=repo_names)

    normalized_repo_paths = normalize_repo_paths(repo_paths)

    return ConversationAttribution(
        repo_paths=normalized_repo_paths,
        repo_names=normalize_repo_names(repo_names, repo_paths=normalized_repo_paths),
        cwd_paths=tuple(sorted(cwd_paths)),
        branch_names=tuple(sorted(branch_names)),
        file_paths_touched=tuple(sorted(file_paths)),
        languages_detected=tuple(sorted(languages)),
    )


def extract_attribution(
    conversation: Conversation,
    *,
    facts: ConversationSemanticFacts | None = None,
) -> ConversationAttribution:
    """Extract path/repo/branch attribution from all tool calls in a conversation."""
    semantic_facts = facts or build_conversation_semantic_facts(conversation)
    base = extract_attribution_from_action_events(
        semantic_facts.action_events,
        provider_meta=conversation.provider_meta if isinstance(conversation.provider_meta, dict) else None,
    )
    repo_paths = set(base.repo_paths)
    repo_names = set(base.repo_names)
    cwd_paths = set(base.cwd_paths)
    branch_names = set(base.branch_names)
    file_paths = set(base.file_paths_touched)
    languages = set(base.languages_detected)

    # Dialogue text can mention transcript-store or persisted-output paths that are not
    # evidence of the repos actually worked on. Keep only low-risk language hints here.
    for message in semantic_facts.message_facts:
        if not message.is_dialogue or not message.text:
            continue
        text_lower = message.text.lower()
        for lang_name, pattern in _DIALOGUE_LANGUAGE_HINT_PATTERNS.items():
            if pattern.search(text_lower):
                languages.add(lang_name)

    normalized_repo_paths = normalize_repo_paths(repo_paths)
    normalized_repo_names = {str(name).strip() for name in repo_names if isinstance(name, str) and name.strip()}
    normalized_repo_names.update(normalize_repo_names(repo_paths=normalized_repo_paths))
    return ConversationAttribution(
        repo_paths=normalized_repo_paths,
        repo_names=tuple(sorted(normalized_repo_names)),
        cwd_paths=tuple(sorted(cwd_paths)),
        branch_names=tuple(sorted(branch_names)),
        file_paths_touched=tuple(sorted(file_paths)),
        languages_detected=tuple(sorted(languages)),
    )
