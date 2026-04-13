"""Shared scenario models used across verification projections."""

from .assertions import AssertionSpec
from .corpus import (
    CorpusRequest,
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
    PipelineProbeInputMode,
    PipelineProbeRequest,
    composite_execution,
    devtools_execution,
    memory_budget_execution,
    pipeline_probe_execution,
    polylogue_execution,
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
from .runtime import (
    ExecutionResult,
    dispatch_execution,
    resolve_execution_command,
    resolve_execution_runner,
    run_execution,
)
from .sources import NamedScenarioSource
from .specs import ScenarioSpec

__all__ = [
    "AssertionSpec",
    "build_default_corpus_specs",
    "build_corpus_scenarios",
    "build_inferred_corpus_specs",
    "composite_execution",
    "devtools_execution",
    "flatten_corpus_specs",
    "memory_budget_execution",
    "pipeline_probe_execution",
    "resolve_corpus_scenarios",
    "resolve_corpus_specs",
    "CorpusRequest",
    "CorpusSourceKind",
    "CorpusScenario",
    "CorpusSpec",
    "compile_projection_entries",
    "dispatch_execution",
    "declared_operation_target_names",
    "ExecutableScenario",
    "ExecutionKind",
    "ExecutionSpec",
    "PipelineProbeInputMode",
    "PipelineProbeRequest",
    "ScenarioMetadata",
    "ScenarioProjectionEntry",
    "ScenarioProjectionSource",
    "ScenarioProjectionSourceKind",
    "ScenarioSpec",
    "NamedScenarioSource",
    "ExecutionResult",
    "polylogue_execution",
    "pytest_execution",
    "resolve_execution_command",
    "resolve_execution_runner",
    "runner_execution",
    "run_execution",
    "runtime_artifact_graph",
    "runtime_artifact_target_names",
    "runtime_operation_target_names",
    "runtime_path_target_names",
]
