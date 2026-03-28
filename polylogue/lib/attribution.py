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

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, Message
    from polylogue.lib.viewports import ToolCall

_BRANCH_PATTERN = re.compile(r"git\s+(?:checkout|switch)\s+(?:-[bc]\s+)?(\S+)")
_LANGUAGE_EXTENSIONS = {
    ".py": "python", ".rs": "rust", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".go": "go", ".nix": "nix",
    ".java": "java", ".rb": "ruby", ".sh": "bash", ".zsh": "zsh",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".sql": "sql", ".r": "r", ".R": "r", ".toml": "toml", ".yaml": "yaml",
    ".json": "json", ".md": "markdown", ".html": "html", ".css": "css",
}


def _get_tool_calls(message: Message) -> list[ToolCall]:
    """Extract tool calls from a message's harmonized viewport."""
    harmonized = message.harmonized
    if harmonized is None:
        return []
    calls = getattr(harmonized, "tool_calls", None)
    return list(calls) if calls else []


def _repo_root_from_path(path: str) -> str | None:
    """Derive a likely repository root from a file path."""
    parts = PurePosixPath(path).parts
    # Look for /realm/project/<name> pattern
    for i, part in enumerate(parts):
        if part == "project" and i > 0 and i + 1 < len(parts):
            return str(PurePosixPath(*parts[: i + 2]))
    return None


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


def extract_attribution(conversation: Conversation) -> ConversationAttribution:
    """Extract path/repo/branch attribution from all tool calls in a conversation."""
    repo_paths: set[str] = set()
    cwd_paths: set[str] = set()
    branch_names: set[str] = set()
    file_paths: set[str] = set()
    languages: set[str] = set()

<<<<<<< HEAD
    for message in conversation.messages:
        for tc in _get_tool_calls(message):
||||||| parent of 171d6b0a (report: clarify SQLite source material design was always intended)
    for message in semantic_facts.message_facts:
        for tc in message.tool_calls:
=======
    provider_meta = conversation.provider_meta if isinstance(conversation.provider_meta, dict) else {}
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

    for message in semantic_facts.message_facts:
        for tc in message.tool_calls:
>>>>>>> 171d6b0a (report: clarify SQLite source material design was always intended)
            # Collect affected paths
            for path in tc.affected_paths:
                file_paths.add(path)
                lang = _language_from_path(path)
                if lang:
                    languages.add(lang)
                repo = _repo_root_from_path(path)
                if repo:
                    repo_paths.add(repo)

            # Extract cwd from shell commands
            cmd = tc.input.get("command", "")
            if isinstance(cmd, str):
                # Look for cd commands
                for match in re.finditer(r'cd\s+"?(/[^\s"]+)', cmd):
                    cwd_paths.add(match.group(1))
                # Look for git branch operations
                for match in _BRANCH_PATTERN.finditer(cmd):
                    branch = match.group(1)
                    if not branch.startswith("-"):
                        branch_names.add(branch)

<<<<<<< HEAD
    # Scan assistant text for file paths and language mentions (catches pure-conversation sessions)
    _REALM_PATH_PATTERN = re.compile(r'/realm/project/[^\s,;:)\]]+')
    _LANGUAGE_NAMES = {"python", "rust", "typescript", "javascript", "nix", "go", "java", "ruby", "sql", "r"}
    for message in conversation.messages:
        if not message.is_assistant or not message.text:
||||||| parent of e4b406d6 (fix: improve gemini and codex archive fidelity)
    # Scan assistant text for file paths and language mentions (catches pure-conversation sessions)
    realm_path_pattern = re.compile(r'/realm/project/[^\s,;:)\]]+')
    language_names = {"python", "rust", "typescript", "javascript", "nix", "go", "java", "ruby", "sql", "r"}
    for message in semantic_facts.message_facts:
        if not message.is_assistant or not message.text:
=======
    # Scan dialogue text for file paths and language mentions (catches pure-conversation sessions)
    realm_path_pattern = re.compile(r'/realm/project/[^\s,;:)\]]+')
    language_names = {"python", "rust", "typescript", "javascript", "nix", "go", "java", "ruby", "sql", "r"}
    for message in semantic_facts.message_facts:
        if not message.is_dialogue or not message.text:
>>>>>>> e4b406d6 (fix: improve gemini and codex archive fidelity)
            continue
<<<<<<< HEAD
        for match in _REALM_PATH_PATTERN.finditer(message.text):
            path = match.group().rstrip(".,;:)'\">`")
            # Skip template/placeholder paths like /realm/project/<name>
            if "<" in path or ">" in path:
||||||| parent of ba2804ba (refactor: strengthen semantic tool understanding)
        for match in realm_path_pattern.finditer(message.text):
            path = match.group().rstrip(".,;:)'\">`")
            # Skip template/placeholder paths like /realm/project/<name>
            if "<" in path or ">" in path:
=======
        for match in realm_path_pattern.finditer(message.text):
            path = _clean_dialogue_path(match.group())
            if path is None:
>>>>>>> ba2804ba (refactor: strengthen semantic tool understanding)
                continue
            file_paths.add(path)
            lang = _language_from_path(path)
            if lang:
                languages.add(lang)
            repo = _repo_root_from_path(path)
            if repo:
                repo_paths.add(repo)
        text_lower = message.text.lower()
        for lang_name in _LANGUAGE_NAMES:
            if lang_name in text_lower:
                languages.add(lang_name)

    # Resolve repo_paths to canonical project names (basename of /realm/project/<X>)
    canonical: set[str] = set()
    for rp in repo_paths:
        parts = PurePosixPath(rp).parts
        for i, part in enumerate(parts):
            if part == "project" and i + 1 < len(parts):
                name = parts[i + 1]
                if name and not name.startswith("."):
                    canonical.add(name)
                break

    return ConversationAttribution(
        repo_paths=tuple(sorted(repo_paths)),
        cwd_paths=tuple(sorted(cwd_paths)),
        branch_names=tuple(sorted(branch_names)),
        file_paths_touched=tuple(sorted(file_paths)),
        languages_detected=tuple(sorted(languages)),
        canonical_projects=tuple(sorted(canonical)),
    )
