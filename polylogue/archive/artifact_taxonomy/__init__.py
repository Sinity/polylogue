"""Heuristic raw-artifact taxonomy for session-bearing payloads.

The taxonomy intentionally favors payload shape over path names. Path hints are
used only as strong evidence for well-known sidecars and weak evidence for
subagent streams.
"""

from __future__ import annotations

from polylogue.archive.artifact_taxonomy.models import ArtifactClassification, ArtifactKind
from polylogue.archive.artifact_taxonomy.runtime import (
    ArtifactStreamClassification,
    classify_artifact,
    classify_artifact_path,
    classify_artifact_records,
    classify_artifact_stream,
    declared_evidence_classification,
    fact_path_admits_session_content,
    strong_path_classification,
)

__all__ = [
    "ArtifactClassification",
    "ArtifactKind",
    "ArtifactStreamClassification",
    "classify_artifact",
    "classify_artifact_stream",
    "classify_artifact_records",
    "classify_artifact_path",
    "fact_path_admits_session_content",
    "declared_evidence_classification",
    "strong_path_classification",
]
