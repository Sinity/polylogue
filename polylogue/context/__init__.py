"""Context-oriented read-view implementations."""

from __future__ import annotations

__all__ = [
    "ContextImage",
    "ContextOmission",
    "ContextSegment",
    "ContextSnapshotRecord",
    "ContextSpec",
    "compose_context_preamble",
    "context_snapshot_record_from_image",
    "ContextAssembly",
    "ContextItem",
    "ContextLedgerRecord",
    "ContextLedgerRow",
    "ContextSource",
    "read_context_ledger",
    "schedule_context",
    "record_context_ledger",
    "ConfigurationArtifactVersion",
    "ConfigurationObservation",
    "ContextResolution",
    "StructuralInvocation",
    "EfficacyComparison",
    "artifact_from_bytes",
    "capture_path",
    "resolve_context",
    "join_invocations",
    "git_artifact_history",
    "compare_cohorts",
]


def __getattr__(name: str) -> object:
    if name in {
        "ConfigurationArtifactVersion",
        "ConfigurationObservation",
        "ContextResolution",
        "StructuralInvocation",
        "EfficacyComparison",
        "artifact_from_bytes",
        "capture_path",
        "resolve_context",
        "join_invocations",
        "git_artifact_history",
        "compare_cohorts",
    }:
        from polylogue.context import configuration_evidence

        return getattr(configuration_evidence, name)
    if name in {"ContextImage", "ContextOmission", "ContextSegment", "ContextSnapshotRecord", "ContextSpec"}:
        from polylogue.context import compiler as compiler_module

        return getattr(compiler_module, name)
    if name == "context_snapshot_record_from_image":
        from polylogue.context.compiler import context_snapshot_record_from_image

        return context_snapshot_record_from_image
    if name == "compose_context_preamble":
        from polylogue.context.preamble import compose_context_preamble

        return compose_context_preamble
    if name in {
        "ContextAssembly",
        "ContextItem",
        "ContextLedgerRecord",
        "ContextLedgerRow",
        "ContextSource",
        "read_context_ledger",
        "schedule_context",
        "record_context_ledger",
    }:
        from polylogue.context.scheduler import (
            ContextAssembly,
            ContextItem,
            ContextLedgerRecord,
            ContextLedgerRow,
            ContextSource,
            read_context_ledger,
            record_context_ledger,
            schedule_context,
        )

        return {
            "ContextAssembly": ContextAssembly,
            "ContextItem": ContextItem,
            "ContextLedgerRecord": ContextLedgerRecord,
            "ContextLedgerRow": ContextLedgerRow,
            "ContextSource": ContextSource,
            "read_context_ledger": read_context_ledger,
            "schedule_context": schedule_context,
            "record_context_ledger": record_context_ledger,
        }[name]
    raise AttributeError(name)
