"""Hermes archival export contract (fs1.7): a versioned per-session export shape.

Status: v1 defined and implemented on the Polylogue side only
(``polylogue-fs1.7``). This module is the checked-in half of a two-sided
contract; the corresponding upstream Hermes-repo commit is external (open
source, not owned by this workspace) and could not be authored or merged
from this environment. ``docs/design/hermes-archival-export-contract.md`` is
the handoff proposal document a Hermes maintainer would need to review and
implement the producer side of.

What this is: a generic, versioned, per-session export Hermes could produce
from one consistent read transaction, covering every active/inactive/
compacted/rewound message, observed/addressing semantics, explicit parent
relationship and lifecycle, usage plus cost provenance, archive/handoff
state, and source/user scope (2026-07-10 Nous follow-up refinement, folded
into the acceptance criteria verbatim). Polylogue stores and parses the exact
export bytes; it never re-derives a lossy summary and calls it the record.

Design decisions this schema encodes, on purpose:

- ``session_revision_hash`` is the dedup key: two exports with an identical
  hash are the same revision (skip), a changed hash is new retained history
  (never an in-place overwrite) -- mirrors the archive-wide content-hash
  idempotency rule (``pipeline/ids.py``).
- Message bodies stay in the export, never in a runtime event
  (``sources.hooks``/``sources.parsers.hermes_lifecycle``): the export is the
  snapshot of record; lifecycle events are timing/outcome evidence about it.
- ``tool_results.output_preview`` is explicitly bounded (not the full tool
  output) -- long tool output belongs in the message content this schema
  already carries in full; the preview exists only for cheap display, and a
  full copy in both places would violate the no-duplicated-transcript rule
  this whole contract exists to uphold for the *event* side of the bridge.
- Every optional/absent field is ``None``, never a fabricated default --
  fidelity (``hermes_state.HermesImportFidelity``) is declared by the
  consumer from what is actually present, not assumed from the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

from polylogue.core.json import JSONValue

HERMES_EXPORT_SCHEMA_VERSION = 1

HermesMessageState: TypeAlias = Literal["active", "inactive", "rewound", "compacted", "observed"]
HermesParentRelationship: TypeAlias = Literal["fork", "resume", "subagent", "continuation"]
HermesArchiveState: TypeAlias = Literal["active", "archived", "handoff-pending", "handoff-complete"]


@dataclass(frozen=True, slots=True)
class HermesExportToolCall:
    action_id: str
    tool_name: str
    arguments_json: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {"action_id": self.action_id, "tool_name": self.tool_name, "arguments_json": self.arguments_json}

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> HermesExportToolCall:
        return HermesExportToolCall(
            action_id=str(payload["action_id"]),
            tool_name=str(payload["tool_name"]),
            arguments_json=_optional_str(payload.get("arguments_json")),
        )


@dataclass(frozen=True, slots=True)
class HermesExportToolResult:
    action_id: str
    is_error: bool | None = None
    exit_code: int | None = None
    output_preview: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "action_id": self.action_id,
            "is_error": self.is_error,
            "exit_code": self.exit_code,
            "output_preview": self.output_preview,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> HermesExportToolResult:
        return HermesExportToolResult(
            action_id=str(payload["action_id"]),
            is_error=cast(bool | None, payload.get("is_error")),
            exit_code=cast(int | None, payload.get("exit_code")),
            output_preview=_optional_str(payload.get("output_preview")),
        )


@dataclass(frozen=True, slots=True)
class HermesExportMessage:
    message_id: str
    role: str
    state: HermesMessageState
    timestamp: str | None = None
    text: str | None = None
    reasoning: str | None = None
    tool_calls: tuple[HermesExportToolCall, ...] = ()
    tool_results: tuple[HermesExportToolResult, ...] = ()
    token_count: int | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "state": self.state,
            "timestamp": self.timestamp,
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "token_count": self.token_count,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> HermesExportMessage:
        return HermesExportMessage(
            message_id=str(payload["message_id"]),
            role=str(payload["role"]),
            state=cast(HermesMessageState, str(payload["state"])),
            timestamp=_optional_str(payload.get("timestamp")),
            text=_optional_str(payload.get("text")),
            reasoning=_optional_str(payload.get("reasoning")),
            tool_calls=tuple(
                HermesExportToolCall.from_dict(cast("dict[str, JSONValue]", item))
                for item in cast("list[JSONValue]", payload.get("tool_calls") or [])
            ),
            tool_results=tuple(
                HermesExportToolResult.from_dict(cast("dict[str, JSONValue]", item))
                for item in cast("list[JSONValue]", payload.get("tool_results") or [])
            ),
            token_count=cast(int | None, payload.get("token_count")),
        )


@dataclass(frozen=True, slots=True)
class HermesExportUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    api_call_count: int = 0

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_call_count": self.api_call_count,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> HermesExportUsage:
        return HermesExportUsage(
            input_tokens=int(cast(int, payload.get("input_tokens", 0))),
            output_tokens=int(cast(int, payload.get("output_tokens", 0))),
            cache_read_tokens=int(cast(int, payload.get("cache_read_tokens", 0))),
            cache_write_tokens=int(cast(int, payload.get("cache_write_tokens", 0))),
            reasoning_tokens=int(cast(int, payload.get("reasoning_tokens", 0))),
            api_call_count=int(cast(int, payload.get("api_call_count", 0))),
        )


@dataclass(frozen=True, slots=True)
class HermesExportCostProvenance:
    billing_provider: str | None = None
    billing_mode: str | None = None
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    cost_status: str | None = None
    cost_source: str | None = None
    pricing_version: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "billing_provider": self.billing_provider,
            "billing_mode": self.billing_mode,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "cost_status": self.cost_status,
            "cost_source": self.cost_source,
            "pricing_version": self.pricing_version,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> HermesExportCostProvenance:
        return HermesExportCostProvenance(
            billing_provider=_optional_str(payload.get("billing_provider")),
            billing_mode=_optional_str(payload.get("billing_mode")),
            estimated_cost_usd=cast(float | None, payload.get("estimated_cost_usd")),
            actual_cost_usd=cast(float | None, payload.get("actual_cost_usd")),
            cost_status=_optional_str(payload.get("cost_status")),
            cost_source=_optional_str(payload.get("cost_source")),
            pricing_version=_optional_str(payload.get("pricing_version")),
        )


@dataclass(frozen=True, slots=True)
class HermesArchivalExportV1:
    """One versioned, self-contained Hermes session export.

    ``schema_version`` must equal ``HERMES_EXPORT_SCHEMA_VERSION`` for this
    dataclass shape; unrecognized higher versions must fail loudly rather
    than being silently coerced (see ``parse_hermes_export``).
    """

    schema_version: int
    producer: str
    profile_id: str
    session_id: str
    session_revision_hash: str
    created_at: str
    finalized: bool
    messages: tuple[HermesExportMessage, ...] = ()
    usage: HermesExportUsage = field(default_factory=HermesExportUsage)
    cost: HermesExportCostProvenance = field(default_factory=HermesExportCostProvenance)
    archive_state: HermesArchiveState = "active"
    ended_at: str | None = None
    end_reason: str | None = None
    parent_session_id: str | None = None
    parent_relationship: HermesParentRelationship | None = None
    handoff_platform: str | None = None
    repository_cwd: str | None = None
    git_branch: str | None = None
    git_repo_root: str | None = None
    source_user_scope: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "session_revision_hash": self.session_revision_hash,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "finalized": self.finalized,
            "end_reason": self.end_reason,
            "messages": [message.to_dict() for message in self.messages],
            "usage": self.usage.to_dict(),
            "cost": self.cost.to_dict(),
            "archive_state": self.archive_state,
            "parent_session_id": self.parent_session_id,
            "parent_relationship": self.parent_relationship,
            "handoff_platform": self.handoff_platform,
            "repository_cwd": self.repository_cwd,
            "git_branch": self.git_branch,
            "git_repo_root": self.git_repo_root,
            "source_user_scope": self.source_user_scope,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> HermesArchivalExportV1:
        schema_version = int(cast(int, payload["schema_version"]))
        if schema_version != HERMES_EXPORT_SCHEMA_VERSION:
            raise HermesExportSchemaError(
                f"unsupported Hermes archival export schema_version: {schema_version} "
                f"(this parser understands {HERMES_EXPORT_SCHEMA_VERSION})"
            )
        return HermesArchivalExportV1(
            schema_version=schema_version,
            producer=str(payload["producer"]),
            profile_id=str(payload["profile_id"]),
            session_id=str(payload["session_id"]),
            session_revision_hash=str(payload["session_revision_hash"]),
            created_at=str(payload["created_at"]),
            ended_at=_optional_str(payload.get("ended_at")),
            finalized=bool(payload["finalized"]),
            end_reason=_optional_str(payload.get("end_reason")),
            messages=tuple(
                HermesExportMessage.from_dict(cast("dict[str, JSONValue]", item))
                for item in cast("list[JSONValue]", payload.get("messages") or [])
            ),
            usage=HermesExportUsage.from_dict(cast("dict[str, JSONValue]", payload.get("usage") or {})),
            cost=HermesExportCostProvenance.from_dict(cast("dict[str, JSONValue]", payload.get("cost") or {})),
            archive_state=cast(HermesArchiveState, str(payload.get("archive_state", "active"))),
            parent_session_id=_optional_str(payload.get("parent_session_id")),
            parent_relationship=cast(HermesParentRelationship | None, payload.get("parent_relationship")),
            handoff_platform=_optional_str(payload.get("handoff_platform")),
            repository_cwd=_optional_str(payload.get("repository_cwd")),
            git_branch=_optional_str(payload.get("git_branch")),
            git_repo_root=_optional_str(payload.get("git_repo_root")),
            source_user_scope=_optional_str(payload.get("source_user_scope")),
        )

    def message_counts_by_state(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for message in self.messages:
            counts[message.state] = counts.get(message.state, 0) + 1
        return counts


class HermesExportSchemaError(ValueError):
    """A Hermes archival export document does not match a supported schema version."""


def parse_hermes_export(payload: dict[str, JSONValue]) -> HermesArchivalExportV1:
    """Parse and validate one Hermes archival export document.

    Raises :class:`HermesExportSchemaError` for an unrecognized
    ``schema_version`` rather than guessing a compatible shape -- a future
    Hermes-side migration that bumps the export version must fail loudly
    here, not silently drop fields (fs1.7 AC: "a Hermes internal DB migration
    does not break the export consumer" means the *failure is explicit*, not
    that every future version is retroactively supported by this parser).
    """
    return HermesArchivalExportV1.from_dict(payload)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "HERMES_EXPORT_SCHEMA_VERSION",
    "HermesArchivalExportV1",
    "HermesArchiveState",
    "HermesExportCostProvenance",
    "HermesExportMessage",
    "HermesExportSchemaError",
    "HermesExportToolCall",
    "HermesExportToolResult",
    "HermesExportUsage",
    "HermesMessageState",
    "HermesParentRelationship",
    "parse_hermes_export",
]
