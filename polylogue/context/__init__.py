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
]


def __getattr__(name: str) -> object:
    if name in {"ContextImage", "ContextOmission", "ContextSegment", "ContextSnapshotRecord", "ContextSpec"}:
        from polylogue.context import compiler as compiler_module

        return getattr(compiler_module, name)
    if name == "context_snapshot_record_from_image":
        from polylogue.context.compiler import context_snapshot_record_from_image

        return context_snapshot_record_from_image
    if name == "compose_context_preamble":
        from polylogue.context.preamble import compose_context_preamble

        return compose_context_preamble
    raise AttributeError(name)
