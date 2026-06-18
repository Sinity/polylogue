"""Session read-view profile registry.

Read views are executable surface contracts, not only CLI strings.  This
registry describes the views that already exist so CLI, completion, MCP, and
future web/API surfaces can inspect the same vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.core.json import JSONDocument

ViewEvidencePolicy = Literal["required", "optional", "unavailable", "omitted"]
ViewLossiness = Literal["raw", "normalized", "filtered", "summarized", "derived", "browse-only"]


@dataclass(frozen=True)
class SessionViewProfile:
    """Contract metadata for an executable session read view."""

    view_id: str
    label: str
    owner: str
    purpose: str
    input_scope: str
    included_kinds: tuple[str, ...]
    lossiness: ViewLossiness
    evidence_policy: ViewEvidencePolicy
    privacy_policy: str
    formats: tuple[str, ...]
    machine_payload: str | None
    degraded_states: tuple[str, ...]
    successor_handoff: bool = False

    def to_payload(self) -> JSONDocument:
        """Return a JSON-native representation for CLI/API/MCP surfaces."""

        return {
            "view_id": self.view_id,
            "label": self.label,
            "owner": self.owner,
            "purpose": self.purpose,
            "input_scope": self.input_scope,
            "included_kinds": list(self.included_kinds),
            "lossiness": self.lossiness,
            "evidence_policy": self.evidence_policy,
            "privacy_policy": self.privacy_policy,
            "formats": list(self.formats),
            "machine_payload": self.machine_payload,
            "degraded_states": list(self.degraded_states),
            "successor_handoff": self.successor_handoff,
        }


READ_VIEW_PROFILES: tuple[SessionViewProfile, ...] = (
    SessionViewProfile(
        view_id="summary",
        label="Summary",
        owner="polylogue.cli.query",
        purpose="Compact human browse view for matched sessions.",
        input_scope="query result set",
        included_kinds=("session summaries", "titles", "dates", "origins", "counts"),
        lossiness="summarized",
        evidence_policy="optional",
        privacy_policy="does not expose raw provider payloads",
        formats=("markdown", "json", "ndjson", "html", "obsidian", "org", "yaml", "plaintext", "csv"),
        machine_payload="query result envelope",
        degraded_states=("empty result set",),
    ),
    SessionViewProfile(
        view_id="transcript",
        label="Transcript",
        owner="polylogue.cli.query",
        purpose="Human transcript rendering through the standard query output path.",
        input_scope="query result set",
        included_kinds=("messages", "blocks", "session metadata"),
        lossiness="normalized",
        evidence_policy="optional",
        privacy_policy="renders normalized archive content, not raw source payloads",
        formats=("markdown", "json", "ndjson", "html", "obsidian", "org", "yaml", "plaintext", "csv"),
        machine_payload="query result envelope",
        degraded_states=("empty result set",),
    ),
    SessionViewProfile(
        view_id="messages",
        label="Messages",
        owner="polylogue.cli.messages.run_messages",
        purpose="Paginated normalized message/block inspection for one session.",
        input_scope="single session id",
        included_kinds=("messages", "blocks", "roles", "content types"),
        lossiness="filtered",
        evidence_policy="optional",
        privacy_policy="projection flags can omit tool calls, outputs, file reads, code blocks, or non-prose",
        formats=("text", "json", "ndjson"),
        machine_payload="archive_messages_payload",
        degraded_states=("missing session", "offset beyond message count"),
    ),
    SessionViewProfile(
        view_id="raw",
        label="Raw",
        owner="polylogue.cli.messages.run_raw",
        purpose="Raw archived provider/source record inspection for one session.",
        input_scope="single session id",
        included_kinds=("raw payload", "raw messages", "source metadata"),
        lossiness="raw",
        evidence_policy="required",
        privacy_policy="may expose raw provider data; caller must choose it explicitly",
        formats=("json",),
        machine_payload="raw archive payload",
        degraded_states=("missing raw record", "missing session"),
    ),
    SessionViewProfile(
        view_id="context",
        label="Context",
        owner="polylogue.context.preamble.compose_context_preamble",
        purpose="Seed-session preamble with nearby related sessions for agent handoff.",
        input_scope="single seed session id",
        included_kinds=("seed session", "related sessions", "context preamble"),
        lossiness="derived",
        evidence_policy="optional",
        privacy_policy="uses normalized archive content and related-session limits",
        formats=("json",),
        machine_payload="context preamble document",
        degraded_states=("missing seed session", "no related sessions"),
        successor_handoff=True,
    ),
    SessionViewProfile(
        view_id="context-pack",
        label="Context Pack",
        owner="polylogue.context.pack.run_context_pack_view",
        purpose="Project/query-scoped multi-session context bundle.",
        input_scope="project path, repo, date/origin/query filters",
        included_kinds=("session excerpts", "messages", "project scope", "redacted paths"),
        lossiness="summarized",
        evidence_policy="optional",
        privacy_policy="redacts filesystem paths unless --no-redact is selected",
        formats=("markdown",),
        machine_payload=None,
        degraded_states=("empty scoped query", "archive unavailable"),
        successor_handoff=True,
    ),
    SessionViewProfile(
        view_id="recovery",
        label="Recovery",
        owner="polylogue.api.archive.Polylogue.recovery_digest / polylogue.api.archive.Polylogue.recovery_report",
        purpose="On-demand successor-agent recovery transform for one session.",
        input_scope="single session id",
        included_kinds=(
            "raw evidence refs",
            "size metrics",
            "tool summaries",
            "subagent reports",
            "run-state candidates",
            "event candidates",
            "decision candidates",
            "forensic index",
            "resume bundle",
            "continue/blame/work-packet reports",
        ),
        lossiness="derived",
        evidence_policy="required",
        privacy_policy="renders normalized evidence links and caveats; raw source payloads remain explicit raw view evidence",
        formats=("markdown", "json"),
        machine_payload="RecoveryDigest",
        degraded_states=("missing session", "insufficient evidence for a section"),
        successor_handoff=True,
    ),
    SessionViewProfile(
        view_id="neighbors",
        label="Neighbors",
        owner="polylogue.archive.session.neighbor_candidates",
        purpose="Explainable nearby/related-session candidates for a seed session or query.",
        input_scope="seed session id or query text",
        included_kinds=("session summaries", "neighbor reasons", "scores", "time windows"),
        lossiness="derived",
        evidence_policy="required",
        privacy_policy="uses summaries and relationship evidence, not full raw transcripts",
        formats=("text", "json"),
        machine_payload="SessionNeighborCandidatePayload",
        degraded_states=("missing seed", "no candidates", "neighbor discovery error"),
    ),
    SessionViewProfile(
        view_id="correlation",
        label="Correlation",
        owner="polylogue.insights.correlation_view.run_correlation_view",
        purpose="GitHub/Git/OTLP correlation evidence around one session.",
        input_scope="single session id plus repository/time-window options",
        included_kinds=("commits", "issues", "pull requests", "checks", "spans", "file overlap"),
        lossiness="derived",
        evidence_policy="required",
        privacy_policy="may call gh for external metadata when enabled",
        formats=("text", "json"),
        machine_payload="correlation payload",
        degraded_states=("missing session", "repo unavailable", "gh unavailable", "no external matches"),
    ),
)

READ_VIEW_PROFILE_BY_ID: dict[str, SessionViewProfile] = {profile.view_id: profile for profile in READ_VIEW_PROFILES}


def read_view_choices() -> tuple[str, ...]:
    """Return the stable read-view ids accepted by ``read --view``."""

    return tuple(profile.view_id for profile in READ_VIEW_PROFILES)


def get_read_view_profile(view_id: str) -> SessionViewProfile:
    """Return profile metadata for a registered read view."""

    return READ_VIEW_PROFILE_BY_ID[view_id]


def read_view_profile_payloads() -> list[JSONDocument]:
    """Return JSON-native payloads for all executable read-view profiles."""

    return [profile.to_payload() for profile in READ_VIEW_PROFILES]


__all__ = [
    "READ_VIEW_PROFILE_BY_ID",
    "READ_VIEW_PROFILES",
    "SessionViewProfile",
    "ViewEvidencePolicy",
    "ViewLossiness",
    "get_read_view_profile",
    "read_view_profile_payloads",
    "read_view_choices",
]
