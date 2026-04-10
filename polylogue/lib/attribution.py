"""Path and repository attribution for conversations.

Extracts raw file paths, repository roots, working directories,
branch names, and detected languages from tool calls within a conversation.
Polylogue does NOT resolve these to project names — that's Lynchpin's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from polylogue.lib.action_events import ActionEvent
from polylogue.lib.project_normalization import normalize_project_names, normalize_repo_path, normalize_repo_paths
from polylogue.lib.semantic_facts import ConversationSemanticFacts, build_conversation_semantic_facts

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


def _repo_root_from_path(path: str) -> str | None:
    """Derive a likely repository root from a file path."""
    return normalize_repo_path(path)


def _language_from_path(path: str) -> str | None:
    """Detect programming language from file extension."""
    suffix = PurePosixPath(path).suffix.lower()
    return _LANGUAGE_EXTENSIONS.get(suffix)


def _clean_dialogue_path(path: str) -> str | None:
    candidate = path.rstrip(".,;:)'\">`*_")
    if not candidate:
        return None
    if candidate.rstrip("/") == "/realm/project":
        return None
    if "<" in candidate or ">" in candidate:
        return None
    if any(ch in candidate for ch in "*?[]{}"):
        return None
    if set(candidate) <= {".", "/"}:
        return None
    return candidate


@dataclass(frozen=True)
class ConversationAttribution:
    """Raw path/repo/branch attribution extracted from a conversation.

    Polylogue extracts raw signals; canonical_projects resolves /realm/project/<X>
    paths to bare project names for direct consumption by Lynchpin.
    """

    repo_paths: tuple[str, ...]
    cwd_paths: tuple[str, ...]
    branch_names: tuple[str, ...]
    file_paths_touched: tuple[str, ...]
    languages_detected: tuple[str, ...]
    canonical_projects: tuple[str, ...]  # resolved project names (e.g. "sinex", "sinnix")


def extract_attribution_from_action_events(
    actions: tuple[ActionEvent, ...] | list[ActionEvent],
    *,
    provider_meta: dict[str, object] | None = None,
) -> ConversationAttribution:
    """Extract attribution from canonical action events plus optional provider metadata."""
    repo_paths: set[str] = set()
    cwd_paths: set[str] = set()
    branch_names: set[str] = set()
    file_paths: set[str] = set()
    languages: set[str] = set()

    provider_meta = provider_meta if isinstance(provider_meta, dict) else {}
    cwd_value = provider_meta.get("cwd")
    if isinstance(cwd_value, str) and cwd_value:
        cwd_paths.add(cwd_value)
        repo = _repo_root_from_path(cwd_value)
        if repo:
            repo_paths.add(repo)

    git_branch = provider_meta.get("gitBranch")
    if isinstance(git_branch, str) and git_branch:
        branch_names.add(git_branch)

    git_meta = provider_meta.get("git")
    if isinstance(git_meta, dict):
        branch = git_meta.get("branch")
        if isinstance(branch, str) and branch:
            branch_names.add(branch)
        repository_url = git_meta.get("repository_url")
        if isinstance(repository_url, str) and repository_url.startswith("/"):
            repo = _repo_root_from_path(repository_url)
            if repo:
                repo_paths.add(repo)

    for action in actions:
        if action.cwd_path:
            cwd_paths.add(action.cwd_path)
            repo = _repo_root_from_path(action.cwd_path)
            if repo:
                repo_paths.add(repo)
        for branch in action.branch_names:
            branch_names.add(branch)
        for path in action.affected_paths:
            file_paths.add(path)
            lang = _language_from_path(path)
            if lang:
                languages.add(lang)
            repo = _repo_root_from_path(path)
            if repo:
                repo_paths.add(repo)

    normalized_repo_paths = normalize_repo_paths(repo_paths)

    return ConversationAttribution(
        repo_paths=normalized_repo_paths,
        cwd_paths=tuple(sorted(cwd_paths)),
        branch_names=tuple(sorted(branch_names)),
        file_paths_touched=tuple(sorted(file_paths)),
        languages_detected=tuple(sorted(languages)),
        canonical_projects=normalize_project_names(repo_paths=normalized_repo_paths),
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
    cwd_paths = set(base.cwd_paths)
    branch_names = set(base.branch_names)
    file_paths = set(base.file_paths_touched)
    languages = set(base.languages_detected)

    # Scan dialogue text for file paths and language mentions (catches pure-conversation sessions)
    realm_path_pattern = re.compile(r"/realm/project/[^\s,;:)\]]+")
    language_names = {"python", "rust", "typescript", "javascript", "nix", "go", "java", "ruby", "sql", "r"}
    for message in semantic_facts.message_facts:
        if not message.is_dialogue or not message.text:
            continue
        for match in realm_path_pattern.finditer(message.text):
            path = _clean_dialogue_path(match.group())
            if path is None:
                continue
            file_paths.add(path)
            lang = _language_from_path(path)
            if lang:
                languages.add(lang)
            repo = _repo_root_from_path(path)
            if repo:
                repo_paths.add(repo)
        text_lower = message.text.lower()
        for lang_name in language_names:
            if lang_name in text_lower:
                languages.add(lang_name)

    normalized_repo_paths = normalize_repo_paths(repo_paths)
    return ConversationAttribution(
        repo_paths=normalized_repo_paths,
        cwd_paths=tuple(sorted(cwd_paths)),
        branch_names=tuple(sorted(branch_names)),
        file_paths_touched=tuple(sorted(file_paths)),
        languages_detected=tuple(sorted(languages)),
        canonical_projects=normalize_project_names(
            base.canonical_projects,
            repo_paths=normalized_repo_paths,
        ),
    )
