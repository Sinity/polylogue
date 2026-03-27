"""Heuristic raw-artifact taxonomy for conversation-bearing payloads.

The taxonomy intentionally favors payload shape over path names. Path hints are
used only as strong evidence for well-known sidecars and weak evidence for
subagent streams.
"""

from __future__ import annotations

from polylogue.lib.artifact_taxonomy_models import ArtifactClassification, ArtifactKind
from polylogue.lib.artifact_taxonomy_runtime import classify_artifact, classify_artifact_path

__all__ = [
    "ArtifactClassification",
    "ArtifactKind",
    "classify_artifact",
    "classify_artifact_path",
]
