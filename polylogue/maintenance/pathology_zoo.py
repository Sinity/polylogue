"""Private-data-free pathology-zoo contracts used by production maintenance.

Fixture construction belongs under :mod:`tests.infra.pathology_zoo`; this
module deliberately contains only the stable identities and read-only archive
invariants that a real canary or archive verifier must consume.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


def _tier_path(archive_root: Path, tier: str) -> Path:
    if tier == "index":
        from polylogue.storage.archive_identity import resolve_active_index_path

        return resolve_active_index_path(archive_root)
    return archive_root / f"{tier}.db"


class PathologyZooCanaryEligibility(StrEnum):
    """Whether a pathology has a replayable session identity."""

    SESSION_BACKED = "session_backed"
    NON_SESSION_ONLY = "non_session_only"


@dataclass(frozen=True, slots=True)
class PathologyZooInvariant:
    """One independently valuable, read-only SQLite invariant."""

    condition: str
    tier: str
    query: str
    parameters: tuple[object, ...]
    expected: tuple[object, ...] | None

    def is_satisfied(self, archive_root: Path, *, index_path_override: Path | None = None) -> bool:
        path = (
            index_path_override
            if self.tier == "index" and index_path_override is not None
            else _tier_path(archive_root, self.tier)
        )
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            return bool(connection.execute(self.query, self.parameters).fetchone() == self.expected)


@dataclass(frozen=True, slots=True)
class PathologyZooMember:
    """One pathology whose durable invariant is checked by maintenance."""

    member_id: str
    pathology: str
    motivating_beads: tuple[str, ...]
    session_ids: tuple[str, ...]
    raw_paths: tuple[str, ...]
    canary_eligibility: PathologyZooCanaryEligibility
    invariant: PathologyZooInvariant
    durable_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.canary_eligibility is PathologyZooCanaryEligibility.SESSION_BACKED and not self.session_ids:
            raise ValueError(f"session-backed pathology {self.member_id!r} must declare session_ids")
        if self.canary_eligibility is PathologyZooCanaryEligibility.NON_SESSION_ONLY and self.session_ids:
            raise ValueError(f"non-session pathology {self.member_id!r} must not declare session_ids")


def _invariant(
    condition: str, tier: str, query: str, parameters: tuple[object, ...], expected: tuple[object, ...] | None
) -> PathologyZooInvariant:
    return PathologyZooInvariant(condition, tier, query, parameters, expected)


_CODEX = "codex-session"
_CLAUDE_AI = "claude-ai-export"
_CLAUDE_CODE = "claude-code-session"
_CHATGPT = "chatgpt-export"
_DESIGN = "claude-design-session"
_GEMINI = "aistudio-drive"


PATHOLOGY_ZOO_MANIFEST: tuple[PathologyZooMember, ...] = (
    PathologyZooMember(
        "whale-component",
        "whale-component",
        ("polylogue-yazae",),
        (f"{_CODEX}:zoo-whale",),
        ("generated/zoo-00-00.jsonl",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the whale retains at least 48 materialized messages",
            "index",
            "SELECT message_count >= 48 FROM sessions WHERE session_id = ?",
            (f"{_CODEX}:zoo-whale",),
            (1,),
        ),
    ),
    PathologyZooMember(
        "append-self-describing",
        "append-revision-chain",
        ("polylogue-yazae",),
        (f"{_CODEX}:zoo-append-self",),
        ("generated/zoo-01-00.jsonl", "generated/zoo-02-00.jsonl"),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the self-describing append retains both raw revisions",
            "source",
            "SELECT COUNT(*) FROM raw_sessions WHERE native_id = 'zoo-append-self'",
            (),
            (2,),
        ),
    ),
    PathologyZooMember(
        "append-opaque",
        "append-revision-chain",
        ("polylogue-yazae",),
        (f"{_CODEX}:zoo-append-opaque",),
        ("generated/zoo-03-00.jsonl", "generated/zoo-04-00.jsonl"),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the opaque append retains both raw revisions",
            "source",
            "SELECT COUNT(*) FROM raw_sessions WHERE native_id = 'zoo-append-opaque'",
            (),
            (2,),
        ),
    ),
    PathologyZooMember(
        "fork-prefix-tail",
        "fork-resume-prefix-tail-lineage",
        ("polylogue-yazae",),
        (f"{_CODEX}:zoo-lineage-parent", f"{_CODEX}:zoo-lineage-child"),
        ("manual/lineage-parent.jsonl", "manual/lineage-child.jsonl"),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the fork child resolves to its prefix-sharing parent",
            "index",
            "SELECT resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
            (f"{_CODEX}:zoo-lineage-child",),
            (f"{_CODEX}:zoo-lineage-parent",),
        ),
    ),
    PathologyZooMember(
        "lineage-cycle",
        "lineage-cycle-candidate",
        ("polylogue-yazae",),
        (f"{_CODEX}:zoo-cycle-a", f"{_CODEX}:zoo-cycle-b"),
        ("manual/lineage-cycle-a.jsonl", "manual/lineage-cycle-b.jsonl", "manual/lineage-cycle-z-a-update.jsonl"),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the cyclic lineage candidate is quarantined",
            "index",
            "SELECT status, resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
            (f"{_CODEX}:zoo-cycle-a",),
            ("quarantined", None),
        ),
    ),
    PathologyZooMember(
        "grouped-jsonl",
        "multi-session-raw",
        ("polylogue-yazae",),
        (f"{_CLAUDE_CODE}:zoo-group-a", f"{_CLAUDE_CODE}:zoo-group-b"),
        ("manual/grouped.jsonl",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the grouped JSONL raw materializes both sessions",
            "index",
            "SELECT COUNT(*) FROM sessions WHERE session_id IN (?, ?)",
            (f"{_CLAUDE_CODE}:zoo-group-a", f"{_CLAUDE_CODE}:zoo-group-b"),
            (2,),
        ),
    ),
    PathologyZooMember(
        "quarantined-head",
        "quarantined-head-open-blocker",
        ("polylogue-yazae",),
        (f"{_CODEX}:zoo-orphan-head",),
        ("manual/lineage-orphan.jsonl",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the unresolved head remains an open blocker",
            "index",
            "SELECT status, resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
            (f"{_CODEX}:zoo-orphan-head",),
            (None, None),
        ),
    ),
    PathologyZooMember(
        "empty-session",
        "genuinely-empty-session",
        ("polylogue-yazae",),
        (f"{_CLAUDE_CODE}:zoo-empty",),
        ("manual/empty.jsonl",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the empty session has no materialized messages",
            "index",
            "SELECT message_count FROM sessions WHERE session_id = ?",
            (f"{_CLAUDE_CODE}:zoo-empty",),
            (0,),
        ),
    ),
    PathologyZooMember(
        "hook-event",
        "hook-event-raw",
        ("polylogue-yazae",),
        (),
        ("hook-spool/zoo-hook-event.json",),
        PathologyZooCanaryEligibility.NON_SESSION_ONLY,
        _invariant(
            "the hook spool becomes a durable PostToolUse event",
            "source",
            "SELECT session_native_id, event_type FROM raw_hook_events WHERE hook_event_id = 'hook:zoo-hook-event'",
            (),
            ("zoo-hook-parent", "PostToolUse"),
        ),
    ),
    PathologyZooMember(
        "claude-design",
        "claude-design-session-origin",
        ("polylogue-yazae",),
        (f"{_DESIGN}:zoo-design-session",),
        ("manual/design.json",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the design export materializes under its distinct origin",
            "index",
            "SELECT origin FROM sessions WHERE session_id = ?",
            (f"{_DESIGN}:zoo-design-session",),
            (_DESIGN,),
        ),
    ),
    PathologyZooMember(
        "vintage-reorder",
        "export-vintage-reorder",
        ("polylogue-yazae", "polylogue-slshy"),
        (f"{_CHATGPT}:zoo-vintage-reorder",),
        ("manual/vintage-old.json", "manual/vintage-new.json"),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the vintage reorder retains both raw revisions",
            "source",
            "SELECT COUNT(*) FROM raw_sessions WHERE native_id = 'zoo-vintage-reorder'",
            (),
            (2,),
        ),
    ),
    PathologyZooMember(
        "content-blocks-vintage",
        "content-blocks-vintage",
        ("polylogue-yazae", "polylogue-0qfy"),
        (f"{_CLAUDE_AI}:zoo-content-blocks-vintage",),
        ("manual/content-blocks-old.json", "manual/content-blocks-new.json"),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the content-block vintage keeps its parsed text block",
            "index",
            "SELECT COUNT(*) FROM blocks WHERE session_id = ?",
            (f"{_CLAUDE_AI}:zoo-content-blocks-vintage",),
            (1,),
        ),
    ),
    PathologyZooMember(
        "lifecycle-anchor-drift",
        "lifecycle-anchor-drift",
        ("polylogue-yazae", "polylogue-uqwd"),
        (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
        ("manual/lifecycle-old.json", "manual/lifecycle-new.json"),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the lifecycle event keeps the revised provider-message anchor",
            "index",
            "SELECT source_message_provider_id FROM session_events WHERE session_id = ? AND event_type = 'generation_lifecycle'",
            (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
            ("lifecycle-b",),
        ),
    ),
    PathologyZooMember(
        "non-stream-safe",
        "non-stream-safe-origin",
        ("polylogue-yazae",),
        ("antigravity-session:zoo-06-00:zoo-06-00.md",),
        ("generated",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the non-stream-safe source is retained under the Antigravity origin",
            "index",
            "SELECT origin FROM sessions WHERE session_id = ?",
            ("antigravity-session:zoo-06-00:zoo-06-00.md",),
            ("antigravity-session",),
        ),
    ),
    PathologyZooMember(
        "attachment-with-bytes",
        "attachment-with-acquired-bytes",
        ("polylogue-yazae",),
        (f"{_GEMINI}:zoo-attachment-bytes",),
        ("generated/zoo-05-00.json",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the acquired attachment retains bytes and a blob hash",
            "index",
            "SELECT a.byte_count > 0 AND a.blob_hash IS NOT NULL AND a.acquisition_status = 'acquired' FROM attachments a JOIN attachment_refs r ON r.attachment_id = a.attachment_id WHERE r.session_id = ?",
            (f"{_GEMINI}:zoo-attachment-bytes",),
            (1,),
        ),
    ),
    PathologyZooMember(
        "attachment-without-bytes",
        "attachment-without-acquired-bytes",
        ("polylogue-yazae",),
        (f"{_CLAUDE_AI}:zoo-attachment-metadata",),
        ("manual/attachment-metadata.json",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the unavailable attachment remains structured metadata without acquired bytes",
            "index",
            "SELECT a.byte_count = 0 AND a.blob_hash IS NULL AND a.acquisition_status = 'unfetched' FROM attachments a JOIN attachment_refs r ON r.attachment_id = a.attachment_id WHERE r.session_id = ?",
            (f"{_CLAUDE_AI}:zoo-attachment-metadata",),
            (1,),
        ),
    ),
    PathologyZooMember(
        "events-sidecars",
        "session-events-and-sidecars",
        ("polylogue-yazae",),
        (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
        ("manual/lifecycle-old.json",),
        PathologyZooCanaryEligibility.SESSION_BACKED,
        _invariant(
            "the lifecycle session retains its parsed sidecar event",
            "index",
            "SELECT COUNT(*) FROM session_events WHERE session_id = ?",
            (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
            (1,),
        ),
    ),
)


def pathology_zoo_manifest() -> tuple[PathologyZooMember, ...]:
    """Return the one manifest production consumers must share."""
    return PATHOLOGY_ZOO_MANIFEST


def pathology_zoo_session_ids() -> tuple[str, ...]:
    """Return every session-backed pathology, explicitly excluding hook-only members."""
    return tuple(
        sorted(
            {
                session_id
                for member in PATHOLOGY_ZOO_MANIFEST
                if member.canary_eligibility is PathologyZooCanaryEligibility.SESSION_BACKED
                for session_id in member.session_ids
            }
        )
    )


def pathology_zoo_is_present(archive_root: Path, *, index_path_override: Path | None = None) -> bool:
    """Return whether an archive contains any durable pathology-zoo evidence."""
    session_ids = pathology_zoo_session_ids()
    index_path = index_path_override if index_path_override is not None else _tier_path(archive_root, "index")
    if index_path.exists():
        placeholders = ", ".join("?" for _ in session_ids)
        try:
            with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
                if (
                    connection.execute(
                        f"SELECT 1 FROM sessions WHERE session_id IN ({placeholders}) LIMIT 1", session_ids
                    ).fetchone()
                    is not None
                ):
                    return True
        except sqlite3.Error:
            pass
    source_path = archive_root / "source.db"
    if source_path.exists():
        try:
            with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as connection:
                return (
                    connection.execute(
                        "SELECT 1 FROM raw_hook_events WHERE hook_event_id = 'hook:zoo-hook-event' LIMIT 1"
                    ).fetchone()
                    is not None
                )
        except sqlite3.Error:
            pass
    return False


__all__ = [
    "PATHOLOGY_ZOO_MANIFEST",
    "PathologyZooCanaryEligibility",
    "PathologyZooInvariant",
    "PathologyZooMember",
    "pathology_zoo_is_present",
    "pathology_zoo_manifest",
    "pathology_zoo_session_ids",
]
