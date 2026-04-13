"""Shared scenario models used across verification projections."""

from .corpus import (
    CorpusSourceKind,
    CorpusSpec,
    build_default_corpus_specs,
    build_inferred_corpus_specs,
    resolve_corpus_specs,
)
from .metadata import (
    ScenarioMetadata,
    declared_operation_target_names,
    runtime_artifact_graph,
    runtime_artifact_target_names,
    runtime_operation_target_names,
    runtime_path_target_names,
)
from .projections import (
    ScenarioProjectionEntry,
    ScenarioProjectionSourceKind,
)

__all__ = [
    "build_default_corpus_specs",
    "build_inferred_corpus_specs",
    "resolve_corpus_specs",
    "CorpusSourceKind",
    "CorpusSpec",
    "declared_operation_target_names",
    "ScenarioMetadata",
    "ScenarioProjectionEntry",
    "ScenarioProjectionSourceKind",
    "runtime_artifact_graph",
    "runtime_artifact_target_names",
    "runtime_operation_target_names",
    "runtime_path_target_names",
]
