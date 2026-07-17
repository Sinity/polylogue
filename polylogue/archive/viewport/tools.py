"""Tool classification and path-cleaning helpers for harmonized viewports."""

from __future__ import annotations

import re
from collections.abc import Mapping

from polylogue.archive.viewport.enums import ToolCategory
from polylogue.core.json import JSONValue

PATH_PATTERN = re.compile(r'(?:^|[\s"\'])(/[^\s"\']+|[./][^\s"\']+)')
NOISE_PATH_TOKENS = frozenset({"...", "//", "/dev/null", "|", "||", "&&", ";"})
METADATA_FILE_BASENAME_ALLOWLIST = frozenset({"LICENSE", "COPYING", "NOTICE", "Makefile", "Dockerfile", "Justfile"})

# Match a sed-style substitution token (``s/find/replace/flags``) — these
# come out of shell-command path extraction with a leading slash and look
# like real paths to the naive PATH_PATTERN. The trailing flags are
# optional. The leading address part (``[!]?[0-9]*[a-z]?``) is intentionally
# tolerant since real sed expressions vary widely.
_SED_SUBSTITUTION_RE = re.compile(r"(?:^|/)!?(s|y)/[^/]*?/[^/]*?/[a-z0-9]*$")

# Path-extension shape: a trailing ``.`` followed by a leading letter
# and 0-5 more alphanumeric characters. Catches ``.py``, ``.md``,
# ``.json``, ``.tsx``, ``.yaml`` etc. without locking the codebase
# into an enumerated allowlist. Rejects long or punctuation-bearing
# suffixes that ``PurePosixPath.suffix`` would otherwise treat as
# paths (``.provider_name``), and digit-only suffixes that come from
# version strings (``.7`` in ``4.7``). Requiring a leading letter is
# the cheap discriminator that handles both.
_PATH_EXTENSION_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,5}$")


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
    if name_lower in (
        "glob",
        "grep",
        "search",
        "find",
        "file_search",
        "ls",
        "toolsearch",
        "find_symbol",
        "find_referencing_symbols",
        "search_for_pattern",
        "get_symbols_overview",
        "query_docs",
        "resolve_library_id",
    ) or any(
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
    if name_lower in (
        "bash",
        "shell",
        "terminal",
        "run",
        "exec",
        "exec_command",
        "functions.exec",
        "functions.exec_command",
        "shell_command",
    ):
        cmd = input_data.get("command", input_data.get("cmd", ""))
        if isinstance(cmd, str) and cmd.strip().startswith("git "):
            return ToolCategory.GIT
        return ToolCategory.SHELL
    if name_lower in ("killshell",):
        return ToolCategory.SHELL
    if name_lower in ("task", "subagent", "spawn_agent"):
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
            "update_plan",
            "write_stdin",
            "send_input",
            "wait_agent",
            "close_agent",
            "initial_instructions",
            "activate_project",
            "get_current_config",
            "get_goal",
            "update_goal",
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
    if any(ch in candidate for ch in "*?[]{}<>"):
        return None
    if _SED_SUBSTITUTION_RE.search(candidate):
        return None
    return candidate


def looks_like_path_candidate(value: object) -> bool:
    candidate = clean_path_candidate(value)
    if candidate is None:
        return False
    # Absolute or explicit-relative paths are accepted as-is — the
    # leading prefix is itself the path signal.
    if candidate.startswith(("/", "~/", "./", "../")):
        return True
    # Otherwise require a *file-extension-shaped* suffix (``.py``,
    # ``.md``, ``.tsx`` …). This rejects Python attribute access
    # (``insight.provider_name``), version numbers (``4.7``), email
    # TLDs in brackets, and other false-positive ``.suffix`` matches
    # that ``PurePosixPath`` would otherwise treat as paths. Tokens
    # like ``kwargs/fields`` and ``dataclass/function`` (slash but no
    # extension) also stop here — they're code, not paths.
    if _PATH_EXTENSION_RE.search(candidate):
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
