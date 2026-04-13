"""Shared scenario models used across verification projections."""

from .corpus import (
    CorpusScenario,
    CorpusSourceKind,
    CorpusSpec,
    build_corpus_scenarios,
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
    ScenarioProjectionSource,
    ScenarioProjectionSourceKind,
    compile_projection_entries,
)

__all__ = [
    "build_default_corpus_specs",
    "build_corpus_scenarios",
    "build_inferred_corpus_specs",
    "resolve_corpus_specs",
    "CorpusSourceKind",
    "CorpusScenario",
    "CorpusSpec",
    "compile_projection_entries",
    "declared_operation_target_names",
    "ScenarioMetadata",
    "ScenarioProjectionEntry",
    "ScenarioProjectionSource",
    "ScenarioProjectionSourceKind",
    "runtime_artifact_graph",
    "runtime_artifact_target_names",
    "runtime_operation_target_names",
    "runtime_path_target_names",
]
