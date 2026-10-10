"""Compiled executable detector bindings declared by ``OriginSpec``.

The parser modules continue to own their structural predicates.  This module
only validates and orders the declaration-owned bindings that decide which
predicate runs for an acquired JSON record or record stream.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import IO, TYPE_CHECKING, Protocol, cast

from polylogue.core.enums import Origin, Provider

if TYPE_CHECKING:
    from polylogue.sources.detection_projection import DetectorProjection


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
    #: Parser-owned complete event projection, consumed by streaming detection.
    stream_projection_path: str | None = None


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

    def iter_record_event_detections(
        self,
        events_factory: Callable[[], Iterator[tuple[str, object]]],
        *,
        include_singleton: bool = True,
    ) -> Generator[tuple[Provider | None, str | None], None, None]:
        """Classify an owned event record and its optional singleton view.

        Exact declared projections share a read view only within this iterator.
        Actual root arrays retain each binding's independent predicate fold.
        Closing the iterator releases all projection resources before the
        caller retires the event/scalar owner.
        """
        from contextlib import ExitStack

        from polylogue.sources.detection_projection import DetectorProjection, _project, detection_read_view

        with ExitStack() as stack:
            first_events = events_factory()
            with ExitStack() as probe:
                close = getattr(first_events, "close", None)
                if close is not None:
                    probe.callback(close)
                first_event = next(first_events, None)
            root_array = first_event is not None and first_event[0] == "start_array"
            projections: dict[str, object] = {}
            for sequence in (False, True) if include_singleton else (False,):
                array_input = sequence or root_array
                modes = (
                    (DetectionMode.SEQUENCE_DOCUMENT, DetectionMode.SEQUENCE_RECORD_STREAM)
                    if array_input
                    else (DetectionMode.RECORD,)
                )
                result: tuple[Provider | None, str | None] = (None, None)
                matched = False
                for mode in modes:
                    for compiled in self.by_mode.get(mode, ()):
                        binding = compiled.binding
                        path = binding.stream_projection_path
                        if path is None:
                            raise DetectorBindingError(
                                f"{binding.binding_id}: complete stream projection is undeclared"
                            )
                        actual_array_fold = root_array and not sequence
                        with ExitStack() as binding_stack:
                            if actual_array_fold or path not in projections:
                                rule = _stream_projection(binding)

                                def array_predicate(item: object, predicate: Predicate = compiled.predicate) -> bool:
                                    return predicate([item])

                                root_rule = (
                                    DetectorProjection(item=rule, array_fold="any", array_predicate=array_predicate)
                                    if actual_array_fold
                                    else rule
                                )
                                with ExitStack() as traversal:
                                    events = events_factory()
                                    close = getattr(events, "close", None)
                                    if close is not None:
                                        traversal.callback(close)
                                    first = next(events, None)
                                    if first is None:
                                        continue
                                    projected = detection_read_view(
                                        _project(
                                            events,
                                            first[0],
                                            first[1],
                                            root_rule,
                                            binding_stack if actual_array_fold else stack,
                                        )
                                    )
                                if not actual_array_fold:
                                    projections[path] = projected
                            else:
                                projected = projections[path]
                            payload = [projected] if sequence else projected
                            if not compiled.predicate(payload):
                                continue
                            resolved_provider: object = (
                                binding.fixed_provider
                                if compiled.provider_resolver is None
                                else compiled.provider_resolver(payload)
                            )
                            if (
                                compiled.provider_resolver is not None
                                and resolved_provider is not None
                                and (
                                    not isinstance(resolved_provider, Provider)
                                    or resolved_provider not in binding.dynamic_provider_allowlist
                                )
                            ):
                                raise DetectorBindingError(f"{binding.binding_id}: invalid projected dynamic provider")
                            result = cast(Provider | None, resolved_provider), binding.evidence_label
                            matched = True
                            break
                    if matched:
                        break
                yield result

    def iter_record_detections(self, value: object) -> Iterator[tuple[Provider | None, str | None]]:
        """Classify a decoded record, then its singleton-sequence view.

        Both results retain their independent binding order. Identical declared
        projections share one read view of this record, held only until the
        iterator closes. Predicates and resolvers must only read those views.
        Actual root arrays retain each binding's own complete any-fold.
        """
        from polylogue.sources.detection_projection import (
            DetectorProjection,
            detection_read_view,
            project_detection_root,
        )

        projections: dict[str, object] = {}
        for sequence in (False, True):
            array_input = sequence or isinstance(value, list)
            modes = (
                (DetectionMode.SEQUENCE_DOCUMENT, DetectionMode.SEQUENCE_RECORD_STREAM)
                if array_input
                else (DetectionMode.RECORD,)
            )
            result: tuple[Provider | None, str | None] = (None, None)
            matched = False
            for mode in modes:
                for compiled in self.by_mode.get(mode, ()):
                    binding = compiled.binding
                    payload: object

                    if not sequence and array_input:
                        rule = _stream_projection(binding)

                        def array_predicate(item: object, predicate: Predicate = compiled.predicate) -> bool:
                            return predicate([item])

                        root_rule = DetectorProjection(item=rule, array_fold="any", array_predicate=array_predicate)
                        payload = detection_read_view(project_detection_root(value, root_rule))
                    else:
                        # A singleton sequence uses exactly the record rule;
                        # its outer any-fold has one possible witness. Cache by
                        # the declaration, never by a mutable object's ID or a
                        # union of different provider projections.
                        path = binding.stream_projection_path
                        if path is None:
                            raise DetectorBindingError(
                                f"{binding.binding_id}: complete stream projection is undeclared"
                            )
                        if path not in projections:
                            projections[path] = detection_read_view(
                                project_detection_root(value, _stream_projection(binding))
                            )
                        projected = projections[path]
                        payload = [projected] if sequence else projected
                    if not compiled.predicate(payload):
                        continue
                    resolved_provider: object = (
                        binding.fixed_provider
                        if compiled.provider_resolver is None
                        else compiled.provider_resolver(payload)
                    )
                    if (
                        compiled.provider_resolver is not None
                        and resolved_provider is not None
                        and (
                            not isinstance(resolved_provider, Provider)
                            or resolved_provider not in binding.dynamic_provider_allowlist
                        )
                    ):
                        raise DetectorBindingError(f"{binding.binding_id}: invalid projected dynamic provider")
                    result = cast(Provider | None, resolved_provider), binding.evidence_label
                    matched = True
                    break
                if matched:
                    break
            yield result

    def detect_record_stream(
        self, records: Iterable[object], *, check_stop: Callable[[], None] | None = None
    ) -> tuple[Provider | None, str | None]:
        """Classify a physical record stream in one pass over its records.

        Equivalent to applying each sequence binding, in registry order, to
        the whole stream: a binding claims the stream when any record
        satisfies it, and the earliest claiming binding wins. Every record is
        consumed, so the record source validates the complete input; only one
        record is held at a time. A record stream is never a single JSON
        document, so record-mode bindings do not apply.
        """
        from polylogue.sources.detection_projection import detection_read_view, project_detection_value

        candidates = (
            *self.by_mode.get(DetectionMode.SEQUENCE_DOCUMENT, ()),
            *self.by_mode.get(DetectionMode.SEQUENCE_RECORD_STREAM, ()),
        )
        rules = tuple(_stream_projection(compiled.binding) for compiled in candidates)
        best = len(candidates)
        witness: object = None
        seen = False
        for record in records:
            if check_stop is not None:
                check_stop()
            seen = True
            projections: dict[str, object] = {}
            # Only bindings at or before the current winner can change the
            # outcome; re-testing the winner keeps its last witness, as the
            # complete "any" fold does. Each predicate sees its own declared
            # projection, so no binding walks undeclared record content.
            for index in range(min(best + 1, len(candidates))):
                path = candidates[index].binding.stream_projection_path
                assert path is not None  # _stream_projection validated every declaration above.
                if path not in projections:
                    projections[path] = detection_read_view(project_detection_value(record, rules[index]))
                projected = projections[path]
                if candidates[index].predicate([projected]):
                    best, witness = index, projected
                    break
        if not seen or best == len(candidates):
            return None, None
        compiled = candidates[best]
        if compiled.provider_resolver is None:
            return compiled.binding.fixed_provider, compiled.binding.evidence_label
        resolved_provider: object = compiled.provider_resolver([witness])
        if resolved_provider is not None and (
            not isinstance(resolved_provider, Provider)
            or resolved_provider not in compiled.binding.dynamic_provider_allowlist
        ):
            raise DetectorBindingError(f"{compiled.binding.binding_id}: invalid projected dynamic provider")
        return resolved_provider, compiled.binding.evidence_label

    def detect_stream(
        self, handle: IO[bytes], *, check_stop: Callable[[], None] | None = None
    ) -> tuple[Provider | None, str | None]:
        """Apply the same registry order to complete parser-declared projections.

        Each candidate consumes and validates the complete input before its
        predicate decides. The handle is seekable acquired material; retrying
        a tighter predicate never reopens its mutable source coordinate.
        """
        from polylogue.sources.detection_projection import project_detection_input

        start = handle.tell()
        try:
            for mode in DetectionMode:
                for compiled in self.by_mode.get(mode, ()):
                    binding = compiled.binding
                    rule = _stream_projection(binding)

                    def array_predicate(item: object, predicate: Predicate = compiled.predicate) -> bool:
                        return predicate([item])

                    handle.seek(start)
                    stream_predicate = array_predicate if mode is not DetectionMode.RECORD else None
                    shape, payload = project_detection_input(
                        handle, rule, stream_predicate=stream_predicate, check_stop=check_stop
                    )
                    if (shape == "record") != (mode is DetectionMode.RECORD):
                        continue
                    if not compiled.predicate(payload):
                        continue
                    resolved_provider: object = (
                        binding.fixed_provider
                        if compiled.provider_resolver is None
                        else compiled.provider_resolver(payload)
                    )
                    if (
                        compiled.provider_resolver is not None
                        and resolved_provider is not None
                        and (
                            not isinstance(resolved_provider, Provider)
                            or resolved_provider not in binding.dynamic_provider_allowlist
                        )
                    ):
                        raise DetectorBindingError(f"{binding.binding_id}: invalid projected dynamic provider")
                    return cast(Provider | None, resolved_provider), binding.evidence_label
            return None, None
        finally:
            handle.seek(start)


@functools.cache
def _stream_projection(binding: DetectorBinding) -> DetectorProjection:
    """The binding's declared projection; a declaration is resolved once per process."""
    from polylogue.sources.detection_projection import DetectorProjection

    if binding.stream_projection_path is None:
        raise DetectorBindingError(f"{binding.binding_id}: complete stream projection is undeclared")
    factory = _resolve_symbol(binding.stream_projection_path, binding_id=binding.binding_id, role="stream projection")
    if not callable(factory):
        raise DetectorBindingError(f"{binding.binding_id}: stream projection factory is not callable")
    rule = factory()
    if not isinstance(rule, DetectorProjection):
        raise DetectorBindingError(f"{binding.binding_id}: invalid stream projection")
    return rule


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
