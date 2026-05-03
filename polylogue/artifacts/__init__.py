"""Typed runtime artifact specifications shared across control-plane surfaces."""

from __future__ import annotations

from polylogue.artifacts.descriptors import ArtifactLayer, ArtifactNode, ArtifactPath
from polylogue.artifacts.runtime import (
    RUNTIME_ARTIFACT_NODES,
    RUNTIME_ARTIFACT_PATHS,
    build_runtime_artifact_nodes,
    build_runtime_artifact_paths,
)

__all__ = [
    "ArtifactLayer",
    "ArtifactNode",
    "ArtifactPath",
    "RUNTIME_ARTIFACT_NODES",
    "RUNTIME_ARTIFACT_PATHS",
    "build_runtime_artifact_nodes",
    "build_runtime_artifact_paths",
]
