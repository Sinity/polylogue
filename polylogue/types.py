"""Type aliases and legacy enum import paths for polylogue."""

from __future__ import annotations

from typing import NewType

from polylogue.core.enums import (
    ArtifactSupportStatus,
    ContentBlockType,
    ExerciseIOMode,
    PlanStage,
    Provider,
    SemanticBlockType,
    ValidationMode,
    ValidationStatus,
)

# Semantic ID types - provides compile-time distinction
SessionId = NewType("SessionId", str)
MessageId = NewType("MessageId", str)
AttachmentId = NewType("AttachmentId", str)
ContentHash = NewType("ContentHash", str)
SessionEventId = NewType("SessionEventId", str)


__all__ = [
    "AttachmentId",
    "ArtifactSupportStatus",
    "ContentBlockType",
    "ContentHash",
    "SessionId",
    "ExerciseIOMode",
    "MessageId",
    "PlanStage",
    "Provider",
    "SessionEventId",
    "SemanticBlockType",
    "ValidationMode",
    "ValidationStatus",
]
