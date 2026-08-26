"""Authoritative declaration of structural schema subjects.

Schema subjects are wire contracts, not necessarily runtime providers.  The
browser receiver is therefore represented alongside provider payloads without
inventing a ``Provider.UNKNOWN`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class SchemaSubjectSpec:
    """One structural subject and its package/admission disposition."""

    token: str
    package_dir: str
    provider: str | None
    origins: tuple[str, ...]
    requires_package: bool = True
    package_not_required_reason: str | None = None


SCHEMA_SUBJECTS: Final[tuple[SchemaSubjectSpec, ...]] = (
    SchemaSubjectSpec("chatgpt", "chatgpt", "chatgpt", ("chatgpt-export",)),
    SchemaSubjectSpec("claude-ai", "claude-ai", "claude-ai", ("claude-ai-export",)),
    SchemaSubjectSpec("claude-design", "claude-design", "claude-design", ("claude-design-session",)),
    SchemaSubjectSpec("claude-code", "claude-code", "claude-code", ("claude-code-session",)),
    SchemaSubjectSpec("codex", "codex", "codex", ("codex-session",)),
    SchemaSubjectSpec("gemini", "gemini", "gemini", ("aistudio-drive",)),
    SchemaSubjectSpec("gemini-cli", "gemini-cli", "gemini-cli", ("gemini-cli-session",)),
    SchemaSubjectSpec("hermes", "hermes", "hermes", ("hermes-session",)),
    SchemaSubjectSpec("antigravity", "antigravity", "antigravity", ("antigravity-session",)),
    SchemaSubjectSpec(
        "beads",
        "beads",
        "beads",
        ("beads-issue",),
        requires_package=False,
        package_not_required_reason="Beads is admitted from repository issue records but has no harvested schema package",
    ),
    SchemaSubjectSpec("grok", "grok", "grok", ("grok-export",)),
    SchemaSubjectSpec(
        "browser-capture",
        "browser-capture",
        None,
        ("unknown-export",),
        package_not_required_reason="first-party transport envelope, not a provider payload",
    ),
)

SCHEMA_SUBJECT_BY_TOKEN: Final[dict[str, SchemaSubjectSpec]] = {item.token: item for item in SCHEMA_SUBJECTS}
CORE_SCHEMA_PROVIDERS: Final[tuple[str, ...]] = tuple(
    item.token for item in SCHEMA_SUBJECTS if item.provider is not None and item.requires_package
)
SCHEMA_PACKAGE_DIRECTORIES: Final[tuple[str, ...]] = tuple(
    item.package_dir for item in SCHEMA_SUBJECTS if item.requires_package
)
CORE_SCHEMA_ORIGINS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(origin for item in SCHEMA_SUBJECTS for origin in item.origins)
)


def schema_subject(token: str) -> SchemaSubjectSpec | None:
    """Return the declared subject for a normalized token."""

    return SCHEMA_SUBJECT_BY_TOKEN.get(token.strip().lower().replace("_", "-"))


__all__ = [
    "CORE_SCHEMA_ORIGINS",
    "CORE_SCHEMA_PROVIDERS",
    "SCHEMA_PACKAGE_DIRECTORIES",
    "SCHEMA_SUBJECTS",
    "SCHEMA_SUBJECT_BY_TOKEN",
    "SchemaSubjectSpec",
    "schema_subject",
]
