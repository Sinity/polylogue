"""Context-oriented read-view implementations."""

from __future__ import annotations

__all__ = [
    "ContextImage",
    "ContextOmission",
    "ContextSegment",
    "ContextSnapshotRecord",
    "ContextSpec",
    "compile_recovery_context",
    "compose_context_preamble",
    "context_image_from_recovery",
    "run_context_pack_view",
]


def __getattr__(name: str) -> object:
    if name in {"ContextImage", "ContextOmission", "ContextSegment", "ContextSnapshotRecord", "ContextSpec"}:
        from polylogue.context import compiler as compiler_module

        return getattr(compiler_module, name)
    if name == "compile_recovery_context":
        from polylogue.context.compiler import compile_recovery_context

        return compile_recovery_context
    if name == "context_image_from_recovery":
        from polylogue.context.compiler import context_image_from_recovery

        return context_image_from_recovery
    if name == "compose_context_preamble":
        from polylogue.context.preamble import compose_context_preamble

        return compose_context_preamble
    if name == "run_context_pack_view":
        from polylogue.context.pack import run_context_pack_view

        return run_context_pack_view
    raise AttributeError(name)
