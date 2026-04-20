"""Tool classification and path-cleaning helpers for harmonized viewports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath

from polylogue.lib.json import JSONValue
from polylogue.lib.viewport_enums import ToolCategory

PATH_PATTERN = re.compile(r'(?:^|[\s"\'])(/[^\s"\']+|[./][^\s"\']+)')
NOISE_PATH_TOKENS = frozenset({"...", "//", "/dev/null", "|", "||", "&&", ";"})
METADATA_FILE_BASENAME_ALLOWLIST = frozenset({"LICENSE", "COPYING", "NOTICE", "Makefile", "Dockerfile", "Justfile"})


def classify_tool(name: str, input_data: Mapping[str, JSONValue]) -> ToolCategory:
    name_lower = name.lower()

    if name_lower in ("read", "view", "cat"):
        return ToolCategory.FILE_READ
    if "__get_file_contents" in name_lower:
        return ToolCategory.FILE_READ
    if name_lower in ("write", "create"):
        return ToolCategory.FILE_WRITE
    if (
        name_lower in ("edit", "patch", "sed", "notebookedit", "multiedit")
        or "__edit_file" in name_lower
        or "__create_or_update_file" in name_lower
    ):
        return ToolCategory.FILE_EDIT
    if name_lower in ("glob", "grep", "search", "find", "file_search", "ls", "toolsearch") or any(
        marker in name_lower
        for marker in (
            "__query-docs",
            "__get-library-docs",
            "__resolve-library-id",
            "__search_code",
            "__search_tool",
            "__search_repositories",
            "__search_users",
            "__search_issues",
            "__search_pull_requests",
            "__find_definition",
            "__find_references",
            "__find_referencing_symbols",
            "__get_symbols_in_file",
            "__get_diagnostics",
        )
    ):
        return ToolCategory.SEARCH
    if name_lower in ("bash", "shell", "terminal", "run"):
        cmd = input_data.get("command", "")
        if isinstance(cmd, str) and cmd.strip().startswith("git "):
            return ToolCategory.GIT
        return ToolCategory.SHELL
    if name_lower in ("killshell",):
        return ToolCategory.SHELL
    if name_lower in ("task", "subagent"):
        return ToolCategory.SUBAGENT
    if (
        name_lower == "agent"
        or name_lower.startswith(("todo", "task"))
        or name_lower
        in (
            "askuserquestion",
            "enterplanmode",
            "exitplanmode",
            "skill",
            "batch",
            "mcp__sequential-thinking__sequentialthinking",
            "mcp__cclsp__restart_server",
        )
    ):
        return ToolCategory.AGENT
    if "tabs_context" in name_lower:
        return ToolCategory.WEB
    if name_lower in ("web", "fetch", "browse", "webfetch", "websearch") or any(
        marker in name_lower for marker in ("__web_search", "__webfetch")
    ):
        return ToolCategory.WEB
    return ToolCategory.OTHER


def clean_path_candidate(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().strip("\"'`")
    candidate = candidate.rstrip(",;:")
    if not candidate:
        return None
    if candidate in NOISE_PATH_TOKENS:
        return None
    if set(candidate) <= {".", "/"}:
        return None
    if candidate.startswith("-"):
        return None
    if any(ch in candidate for ch in "*?[]{}"):
        return None
    return candidate


def looks_like_path_candidate(value: object) -> bool:
    candidate = clean_path_candidate(value)
    if candidate is None:
        return False
    if candidate.startswith(("/", "~/", "./", "../")):
        return True
    if "/" in candidate:
        return True
    if candidate.startswith("."):
        return True
    if PurePosixPath(candidate).suffix:
        return True
    return candidate in METADATA_FILE_BASENAME_ALLOWLIST


def clean_metadata_path_candidate(value: object) -> str | None:
    candidate = clean_path_candidate(value)
    if candidate is None:
        return None
    if not looks_like_path_candidate(candidate):
        return None
    return candidate


def clean_shell_path_candidate(value: object) -> str | None:
    candidate = clean_path_candidate(value)
    if candidate is None:
        return None
    if candidate.startswith(".") and "/" not in candidate:
        return None
    return candidate


__all__ = [
    "METADATA_FILE_BASENAME_ALLOWLIST",
    "NOISE_PATH_TOKENS",
    "PATH_PATTERN",
    "classify_tool",
    "clean_metadata_path_candidate",
    "clean_path_candidate",
    "clean_shell_path_candidate",
    "looks_like_path_candidate",
]
