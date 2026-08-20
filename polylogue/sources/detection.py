"""Compiled executable detector bindings declared by ``OriginSpec``.

The parser modules continue to own their structural predicates.  This module
only validates and orders the declaration-owned bindings that decide which
predicate runs for an acquired JSON record or record stream.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Protocol, cast

from polylogue.core.enums import Origin, Provider


class DetectionMode(StrEnum):
    """The normalized payload shape to which a detector binding applies."""

    RECORD = "record"
    SEQUENCE_DOCUMENT = "sequence-document"
    SEQUENCE_RECORD_STREAM = "sequence-record-stream"


@dataclass(frozen=True, slots=True)
class DetectorBinding:
    """One lazy, evidence-labelled provider claim owned by an ``OriginSpec``."""

    binding_id: str
    mode: DetectionMode
    predicate_path: str
    local_rank: int
    evidence_label: str
    #: Optional mode-specific precedence when a stream intentionally differs
    #: from the origin-wide record tightness order.
    mode_rank: int | None = None
    fixed_provider: Provider | None = None
    dynamic_provider_path: str | None = None
    dynamic_provider_allowlist: tuple[Provider, ...] = ()


class _OriginSpecLike(Protocol):
    @property
    def origin(self) -> Origin: ...

    @property
    def lifecycle(self) -> str: ...

    @property
    def provider_wires(self) -> tuple[Provider, ...]: ...

    @property
    def detector_tightness(self) -> int | None: ...

    @property
    def detector_bindings(self) -> tuple[DetectorBinding, ...]: ...


Predicate = Callable[[object], bool]
ProviderResolver = Callable[[object], Provider | None]


class DetectorBindingError(ValueError):
    """A declaration error with the owning origin and binding in its message."""


@dataclass(frozen=True, slots=True)
class _CompiledBinding:
    binding: DetectorBinding
    predicate: Predicate
    provider_resolver: ProviderResolver | None


@dataclass(frozen=True, slots=True)
class CompiledDetectorRegistry:
    """Validated detector bindings, ordered independently for each payload mode."""

    by_mode: dict[DetectionMode, tuple[_CompiledBinding, ...]]

    def detect(self, mode: DetectionMode, payload: object) -> tuple[Provider | None, str | None]:
        """Return the first declared detector result and its exact evidence label."""
        for compiled in self.by_mode.get(mode, ()):  # pragma: no branch - every enum mode is initialized
            if not compiled.predicate(payload):
                continue
            provider = (
                compiled.binding.fixed_provider
                if compiled.provider_resolver is None
                else compiled.provider_resolver(payload)
            )
            if provider is None:
                return None, compiled.binding.evidence_label
            if compiled.provider_resolver is not None and not isinstance(provider, Provider):
                raise DetectorBindingError(
                    f"{compiled.binding.binding_id}: dynamic provider resolver returned {provider!r}, not a Provider"
                )
            if provider not in compiled.binding.dynamic_provider_allowlist and compiled.provider_resolver is not None:
                raise DetectorBindingError(
                    f"{compiled.binding.binding_id}: dynamic provider {provider.value!r} is outside its declared "
                    "allowlist"
                )
            return provider, compiled.binding.evidence_label
        return None, None


def _resolve_symbol(path: str, *, binding_id: str, role: str) -> object:
    module_name, separator, symbol_path = path.partition(":")
    if not separator or not module_name or not symbol_path:
        raise DetectorBindingError(f"{binding_id}: {role} path must be 'module:Symbol', got {path!r}")
    try:
        resolved: object = import_module(module_name)
        for attribute in symbol_path.split("."):
            resolved = getattr(resolved, attribute)
    except (AttributeError, ImportError) as exc:
        raise DetectorBindingError(f"{binding_id}: cannot resolve {role} {path!r}: {exc}") from exc
    return resolved


def _validate_unary_callable(value: object, *, binding_id: str, role: str) -> Callable[[object], object]:
    if not callable(value):
        raise DetectorBindingError(f"{binding_id}: {role} is not callable")
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as exc:
        raise DetectorBindingError(f"{binding_id}: cannot inspect {role} signature: {exc}") from exc
    parameters = tuple(signature.parameters.values())
    if len(parameters) != 1 or parameters[0].kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise DetectorBindingError(f"{binding_id}: {role} must accept exactly one payload argument")
    return value


def _resolve_predicate(path: str, *, binding_id: str) -> Predicate:
    """Resolve a declaration's unary predicate after its signature is checked."""
    return cast(
        Predicate,
        _validate_unary_callable(
            _resolve_symbol(path, binding_id=binding_id, role="predicate"),
            binding_id=binding_id,
            role="predicate",
        ),
    )


def _resolve_provider_resolver(path: str, *, binding_id: str) -> ProviderResolver:
    """Resolve a declaration's unary provider resolver after its signature is checked."""
    return cast(
        ProviderResolver,
        _validate_unary_callable(
            _resolve_symbol(path, binding_id=binding_id, role="dynamic provider resolver"),
            binding_id=binding_id,
            role="dynamic provider resolver",
        ),
    )


def _compile_binding(spec: _OriginSpecLike, binding: DetectorBinding) -> _CompiledBinding:
    prefix = f"{spec.origin.value}/{binding.binding_id}"
    if not binding.binding_id:
        raise DetectorBindingError(f"{spec.origin.value}: detector binding id must be non-empty")
    if binding.local_rank < 0:
        raise DetectorBindingError(f"{prefix}: local rank must be non-negative")
    if not binding.evidence_label:
        raise DetectorBindingError(f"{prefix}: evidence label must be non-empty")
    has_fixed = binding.fixed_provider is not None
    has_dynamic = binding.dynamic_provider_path is not None
    if has_fixed == has_dynamic:
        raise DetectorBindingError(f"{prefix}: declare exactly one fixed provider or dynamic provider resolver")
    if has_fixed:
        assert binding.fixed_provider is not None
        if binding.fixed_provider not in spec.provider_wires:
            raise DetectorBindingError(
                f"{prefix}: fixed provider {binding.fixed_provider.value!r} is not declared by {spec.origin.value}"
            )
        if binding.dynamic_provider_allowlist:
            raise DetectorBindingError(f"{prefix}: fixed provider binding cannot declare a dynamic allowlist")
    else:
        if not binding.dynamic_provider_allowlist:
            raise DetectorBindingError(f"{prefix}: dynamic provider resolver requires a non-empty allowlist")

    predicate = _resolve_predicate(binding.predicate_path, binding_id=prefix)
    resolver: ProviderResolver | None = None
    if binding.dynamic_provider_path is not None:
        resolver = _resolve_provider_resolver(binding.dynamic_provider_path, binding_id=prefix)
    return _CompiledBinding(
        binding=binding,
        predicate=predicate,
        provider_resolver=resolver,
    )


def compile_detector_registry(specs: Iterable[_OriginSpecLike]) -> CompiledDetectorRegistry:
    """Compile executable OriginSpec bindings once, rejecting ambiguous declarations."""
    compiled: dict[DetectionMode, list[tuple[int, int, str, _CompiledBinding]]] = {mode: [] for mode in DetectionMode}
    binding_ids: set[str] = set()
    for spec in specs:
        if spec.lifecycle != "executable" and not spec.detector_bindings:
            continue
        if spec.lifecycle == "executable" and not spec.detector_bindings:
            raise DetectorBindingError(f"{spec.origin.value}: executable origin requires at least one detector binding")
        if spec.lifecycle == "executable" and spec.detector_tightness is None:
            raise DetectorBindingError(f"{spec.origin.value}: executable origin requires detector tightness")
        for binding in spec.detector_bindings:
            if binding.binding_id in binding_ids:
                raise DetectorBindingError(f"duplicate detector binding id {binding.binding_id!r}")
            binding_ids.add(binding.binding_id)
            compiled_binding = _compile_binding(spec, binding)
            compiled[binding.mode].append(
                (
                    binding.mode_rank
                    if binding.mode_rank is not None
                    else (spec.detector_tightness if spec.detector_tightness is not None else -1),
                    binding.local_rank,
                    binding.binding_id,
                    compiled_binding,
                )
            )
    return CompiledDetectorRegistry(
        by_mode={
            mode: tuple(item[-1] for item in sorted(bindings, key=lambda item: item[:3]))
            for mode, bindings in compiled.items()
        }
    )


__all__ = [
    "CompiledDetectorRegistry",
    "DetectionMode",
    "DetectorBinding",
    "DetectorBindingError",
    "compile_detector_registry",
]
