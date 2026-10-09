"""Producer-owned raw artifact observations.

Artifact taxonomy belongs at the source boundary.  Derived materializers may
read these observations, but they must not reconstruct them after acquisition.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from polylogue.archive.artifact_taxonomy import ArtifactClassification, classify_artifact_path
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.artifacts.inspection import artifact_observation_id
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    SourceArtifactProducer,
    _upsert_raw_artifact,
)


def record_session_artifact_observation(
    producer: SourceArtifactProducer,
    *,
    raw_id: str,
    provider: Provider,
    source_path: str,
    source_index: int,
    observed_at_ms: int,
    manage_transaction: bool = True,
    captured_classification: ArtifactClassification | None = None,
) -> bool:
    """Record a session artifact with captured or path-declared taxonomy.

    Captured parser taxonomy owns schema eligibility. Path-only positive
    session evidence conservatively requires validation.

    Content-aware callers invoke this after positive session-grammar evidence
    exists. A captured native grammar may yield zero session rows.
    A fact or raw-only path is deliberately ignored here: content evidence may
    override that declaration, and source-only acquisition must remain pending
    until replay has consumed the retained bytes.
    """

    classification = captured_classification
    if classification is not None and classification.provider is not provider:
        raise ValueError("captured session artifact provider changed")
    if classification is None:
        classification = classify_artifact_path(source_path, provider=provider)
    if classification is None or not classification.parse_as_session:
        return False
    origin = origin_from_provider(provider)
    _upsert_raw_artifact(
        producer,
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
            schema_eligible=classification.schema_eligible if captured_classification is not None else True,
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
