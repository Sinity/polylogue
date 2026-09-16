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
    member_prefixes: tuple[str, ...] = ()
    excluded_member_prefixes: tuple[str, ...] = ()

    def admits_member(self, member_path: str) -> bool:
        """Decide whether one member of a shared export root belongs to this subject.

        Two subjects can declare the same physical root (the Claude account
        export ships both GDPR conversations and design chats).  Without a
        member rule each would fold the other's material and describe the
        wrong wire format.  ``member_prefixes`` is a positive admission list;
        ``excluded_member_prefixes`` removes members owned by a sibling
        subject.  A subject declaring neither admits every member.
        """

        normalized = member_path.replace("\\", "/").lstrip("/")
        if any(normalized.startswith(prefix) for prefix in self.excluded_member_prefixes):
            return False
        if not self.member_prefixes:
            return True
        return any(normalized.startswith(prefix) for prefix in self.member_prefixes)


SCHEMA_SUBJECTS: Final[tuple[SchemaSubjectSpec, ...]] = (
    SchemaSubjectSpec("chatgpt", "chatgpt", "chatgpt", ("chatgpt-export",)),
    SchemaSubjectSpec(
        "claude-ai",
        "claude-ai",
        "claude-ai",
        ("claude-ai-export",),
        excluded_member_prefixes=("design_chats/",),
    ),
    SchemaSubjectSpec(
        "claude-design",
        "claude-design",
        "claude-design",
        ("claude-design-session",),
        member_prefixes=("design_chats/",),
    ),
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
        package_not_required_reason="Reserved origin with no admitted Beads session wire format",
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


def subject_admits_member(token: str, member_path: str) -> bool:
    """Admit a member for an undeclared subject; enforce the rule for a declared one."""

    subject = schema_subject(token)
    return True if subject is None else subject.admits_member(member_path)


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
    "subject_admits_member",
]
