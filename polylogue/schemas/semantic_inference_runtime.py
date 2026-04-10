"""Role-selection workflow for semantic schema inference."""

from __future__ import annotations

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.semantic_inference_conversation_scoring import score_title
from polylogue.schemas.semantic_inference_message_scoring import (
    score_body,
    score_container,
    score_role,
    score_timestamp,
)
from polylogue.schemas.semantic_inference_models import SEMANTIC_ROLES, SemanticCandidate

RECORD_STREAM_KINDS = frozenset(
    {
        "conversation_record_stream",
        "subagent_conversation_stream",
    }
)

# Roles that record-stream artifacts may infer. Title is excluded because
# individual records in a stream do not carry conversation-level metadata.
RECORD_STREAM_ELIGIBLE_ROLES = frozenset(
    {
        "message_container",
        "message_role",
        "message_body",
        "message_timestamp",
    }
)


def infer_semantic_roles(
    stats: dict[str, FieldStats],
    *,
    artifact_kind: str | None = None,
) -> list[SemanticCandidate]:
    """Score all field paths for all semantic roles.

    When *artifact_kind* is a record-stream kind, ``conversation_title``
    scoring is skipped because individual records do not carry titles.
    """
    is_record_stream = artifact_kind in RECORD_STREAM_KINDS
    candidates: list[SemanticCandidate] = []
    for path, field_stats in stats.items():
        for role in SEMANTIC_ROLES:
            if is_record_stream and role not in RECORD_STREAM_ELIGIBLE_ROLES:
                continue
            candidate = score_candidate(path, field_stats, role, stats)
            if candidate is not None and candidate.confidence > 0.1:
                candidates.append(candidate)
    candidates.sort(key=lambda candidate: -candidate.confidence)
    return candidates


def select_best_roles(candidates: list[SemanticCandidate]) -> dict[str, SemanticCandidate]:
    """Select the single best candidate for each semantic role."""
    best: dict[str, SemanticCandidate] = {}
    for candidate in candidates:
        if candidate.role not in best or candidate.confidence > best[candidate.role].confidence:
            best[candidate.role] = candidate
    return best


def score_candidate(
    path: str,
    field_stats: FieldStats,
    role: str,
    all_stats: dict[str, FieldStats],
) -> SemanticCandidate | None:
    """Score a single path against one semantic role."""
    match role:
        case "message_container":
            return score_container(path, field_stats, all_stats)
        case "message_role":
            return score_role(path, field_stats)
        case "message_body":
            return score_body(path, field_stats)
        case "message_timestamp":
            return score_timestamp(path, field_stats)
        case "conversation_title":
            return score_title(path, field_stats, all_stats)
    return None


__all__ = [
    "RECORD_STREAM_ELIGIBLE_ROLES",
    "RECORD_STREAM_KINDS",
    "infer_semantic_roles",
    "score_candidate",
    "select_best_roles",
]
