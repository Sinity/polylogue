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
    #: Why this subject is outside the schema-*inference* denominator. Set only
    #: for a subject whose wire format this repository authors: its shape is a
    #: decision made here, changed together with its writer, never a discovery
    #: made from evidence. An excluded subject has no source-evidence adapter
    #: by design -- writing one would infer a schema for a format we control --
    #: so the inference route refuses the token outright rather than admitting
    #: its material and refusing it member by member.
    inference_excluded_reason: str | None = None


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
        package_not_required_reason="Reserved origin with no admitted Beads session wire format",
    ),
    SchemaSubjectSpec(
        "grok",
        "grok",
        "grok",
        ("grok-export",),
        requires_package=False,
        package_not_required_reason=(
            "No admitted Grok export evidence. The package committed under this subject was "
            "Claude.ai's export folded in by a source-selection defect -- its elements carried "
            "$id polylogue://schemas/claude-ai/... and Claude.ai's document shape (chat_messages, "
            "uuid, account, project) -- so it was removed rather than relabelled (polylogue-n61h5). "
            "Regenerate from real Grok artifacts to restore a package; "
            "`devtools gate schema-provider-identity` refuses another subject's elements landing here."
        ),
    ),
    SchemaSubjectSpec(
        "browser-capture",
        "browser-capture",
        None,
        ("unknown-export",),
        requires_package=False,
        package_not_required_reason=(
            "first-party transport envelope, not a provider payload; its structural contract is the authored "
            "Pydantic model in polylogue/browser_capture/models.py, not an inferred package"
        ),
        inference_excluded_reason=(
            "browser-capture is authored in this repository -- the browser-extension/ writer and the "
            "polylogue-browser-capture-native-host reader are both ours -- so its shape is a decision, not "
            "evidence to discover. No source-evidence adapter should exist for it; the envelope model and its "
            "parser change together."
        ),
    ),
)

SCHEMA_SUBJECT_BY_TOKEN: Final[dict[str, SchemaSubjectSpec]] = {item.token: item for item in SCHEMA_SUBJECTS}
#: Subjects declared outside the schema-inference denominator, with their reason.
INFERENCE_EXCLUDED_SUBJECTS: Final[dict[str, str]] = {
    item.token: reason for item in SCHEMA_SUBJECTS if (reason := item.inference_excluded_reason) is not None
}
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


def inference_exclusion_reason(token: str) -> str | None:
    """Return why this subject is outside the inference denominator, if it is.

    A declared exclusion is the authority for a zero denominator: the subject
    contributes no eligible material because it was never admissible, which is
    a different claim from "every member was refused".
    """

    return INFERENCE_EXCLUDED_SUBJECTS.get(token.strip().lower().replace("_", "-"))


__all__ = [
    "CORE_SCHEMA_ORIGINS",
    "CORE_SCHEMA_PROVIDERS",
    "INFERENCE_EXCLUDED_SUBJECTS",
    "SCHEMA_PACKAGE_DIRECTORIES",
    "SCHEMA_SUBJECTS",
    "SCHEMA_SUBJECT_BY_TOKEN",
    "SchemaSubjectSpec",
    "inference_exclusion_reason",
    "schema_subject",
]
