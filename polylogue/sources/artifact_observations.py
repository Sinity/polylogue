"""Producer-owned raw artifact observations.

Artifact taxonomy belongs at the source boundary.  Derived materializers may
read these observations, but they must not reconstruct them after acquisition.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact


def record_session_artifact_observation(
    archive: Any,
    *,
    raw_id: str,
    provider: Provider,
    source_path: str,
    source_index: int,
    observed_at_ms: int,
    manage_transaction: bool = True,
) -> bool:
    """Record a parsed session only when its path declares a session artifact.

    Content-aware callers invoke this after positive session evidence exists.
    A fact or raw-only path is deliberately ignored here: content evidence may
    override that declaration, and source-only acquisition must remain pending
    until replay has consumed the retained bytes.
    """

    classification = classify_artifact_path(source_path, provider=provider)
    if classification is None or not classification.parse_as_session:
        return False
    origin = origin_from_provider(provider)
    upsert_raw_artifact(
        archive._ensure_source_conn(),
        raw_id,
        ArchiveSourceArtifact(
            artifact_id=artifact_observation_id(
                source_name=origin.value,
                source_path=source_path,
                source_index=source_index,
            ),
            origin=origin,
            source_path=source_path,
            source_index=source_index,
            artifact_kind=classification.cohort,
            classification_reason=classification.reason,
            support_status="supported_parseable",
            parse_as_session=True,
            schema_eligible=classification.schema_eligible,
            first_observed_at_ms=observed_at_ms,
            last_observed_at_ms=observed_at_ms,
            link_group_key=_agent_link_group(source_path),
        ),
        manage_transaction=manage_transaction,
    )
    return True


def _agent_link_group(source_path: str) -> str | None:
    normalized = source_path.replace("\\", "/").lower()
    for suffix in (".meta.json", ".jsonl", ".ndjson"):
        if normalized.endswith(suffix) and PurePosixPath(normalized).name.startswith("agent-"):
            return normalized[: -len(suffix)]
    return None


__all__ = ["record_session_artifact_observation"]
