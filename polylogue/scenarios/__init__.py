"""Shared scenario models used across verification projections."""

from .corpus import (
    CorpusScenario,
    CorpusSourceKind,
    CorpusSpec,
    build_corpus_scenarios,
    build_default_corpus_specs,
    build_inferred_corpus_specs,
    flatten_corpus_specs,
    resolve_corpus_scenarios,
    resolve_corpus_specs,
)
from .executable import ExecutableScenario
from .execution import (
    ExecutionKind,
    ExecutionSpec,
    command_execution,
    composite_execution,
    pytest_execution,
    runner_execution,
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
from .sources import NamedScenarioSource
from .specs import ScenarioSpec

__all__ = [
    "build_default_corpus_specs",
    "build_corpus_scenarios",
    "build_inferred_corpus_specs",
    "command_execution",
    "composite_execution",
    "flatten_corpus_specs",
    "resolve_corpus_scenarios",
    "resolve_corpus_specs",
    "CorpusSourceKind",
    "CorpusScenario",
    "CorpusSpec",
    "compile_projection_entries",
    "declared_operation_target_names",
    "ExecutableScenario",
    "ExecutionKind",
    "ExecutionSpec",
    "ScenarioMetadata",
    "ScenarioProjectionEntry",
    "ScenarioProjectionSource",
    "ScenarioProjectionSourceKind",
    "ScenarioSpec",
    "NamedScenarioSource",
    "pytest_execution",
    "runner_execution",
    "runtime_artifact_graph",
    "runtime_artifact_target_names",
    "runtime_operation_target_names",
    "runtime_path_target_names",
]
