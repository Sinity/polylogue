"""Owned event-record projection reuse preserves independent interpretations."""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import replace

import pytest
from ijson.backends import python as exact_backend

from polylogue.core.enums import Provider
from polylogue.schemas.observation_spill import StreamedJSONDocument
from polylogue.sources import detection_projection
from polylogue.sources.detection import DetectionMode
from polylogue.sources.origin_specs import detector_registry


def _events(value: object) -> Iterator[tuple[str, object]]:
    return iter(exact_backend.basic_parse(io.BytesIO(json.dumps(value).encode())))


def test_event_record_reuses_only_declared_projections_and_releases_traversals() -> None:
    registry = detector_registry()
    opened = closed = 0
    value = {"unrelated": "x" * (1024 * 1024), "nested": [[{"opaque": "complete"}]]}

    def events() -> Iterator[tuple[str, object]]:
        nonlocal opened, closed
        opened += 1
        try:
            yield from _events(value)
        finally:
            closed += 1

    paths = {c.binding.stream_projection_path for candidates in registry.by_mode.values() for c in candidates}
    assert list(registry.iter_record_event_detections(events)) == list(registry.iter_record_detections(value))
    assert opened == closed == 1 + len(paths)  # one closed shape probe, then one pass per declaration
    assert list(registry.iter_record_event_detections(events)) == [(None, None), (None, None)]
    assert opened == closed == 2 * (1 + len(paths))  # no cross-record reuse


def test_event_record_reuses_projection_without_merging_binding_precedence() -> None:
    registry = detector_registry()
    compiled = registry.by_mode[DetectionMode.RECORD][0]
    losing = replace(compiled, predicate=lambda _payload: False, provider_resolver=None)
    winning = replace(
        compiled,
        predicate=lambda _payload: True,
        provider_resolver=None,
        binding=replace(compiled.binding, fixed_provider=Provider.CODEX, evidence_label="winner"),
    )
    selected = replace(
        registry, by_mode={DetectionMode.RECORD: (losing, winning), DetectionMode.SEQUENCE_DOCUMENT: (losing, winning)}
    )
    calls = 0

    def events() -> Iterator[tuple[str, object]]:
        nonlocal calls
        calls += 1
        return _events({"irrelevant": "complete"})

    assert list(selected.iter_record_event_detections(events)) == [(Provider.CODEX, "winner")] * 2
    assert calls == 2  # shape probe plus one shared record projection


def test_actual_array_keeps_binding_specific_complete_folds() -> None:
    registry = detector_registry()
    compiled = registry.by_mode[DetectionMode.SEQUENCE_DOCUMENT][0]
    losing = replace(compiled, predicate=lambda _payload: False, provider_resolver=None)
    winning = replace(
        compiled,
        predicate=lambda _payload: True,
        provider_resolver=None,
        binding=replace(compiled.binding, fixed_provider=Provider.CODEX, evidence_label="array-winner"),
    )
    selected = replace(registry, by_mode={DetectionMode.SEQUENCE_DOCUMENT: (losing, winning)})
    calls = 0

    def events() -> Iterator[tuple[str, object]]:
        nonlocal calls
        calls += 1
        return _events([{"irrelevant": 1}, {"irrelevant": 2}])

    assert list(selected.iter_record_event_detections(events, include_singleton=False)) == [
        (Provider.CODEX, "array-winner")
    ]
    assert calls == 3  # each binding owns its array predicate and witness


@pytest.mark.parametrize("settlement", ["complete", "early-close", "predicate-failure"])
def test_event_projection_stack_has_exact_record_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    settlement: str,
) -> None:
    registry = detector_registry()
    compiled = registry.by_mode[DetectionMode.RECORD][0]

    def predicate(_payload: object) -> bool:
        if settlement == "predicate-failure":
            raise ValueError("neutral predicate refusal")
        return True

    compiled = replace(compiled, predicate=predicate, provider_resolver=None)
    selected = replace(registry, by_mode={DetectionMode.RECORD: (compiled,)})
    original = detection_projection._project
    connections: list[sqlite3.Connection] = []

    def project(
        events: Iterator[tuple[str, object]],
        event: str,
        value: object,
        rule: detection_projection.DetectorProjection | None,
        stack: ExitStack,
    ) -> object:
        document = StreamedJSONDocument(None)
        stack.enter_context(document)
        connections.append(document.connection)
        return original(events, event, value, rule, stack)

    monkeypatch.setattr(detection_projection, "_project", project)
    detections = selected.iter_record_event_detections(lambda: _events({"ignored": 1}))
    if settlement == "predicate-failure":
        with pytest.raises(ValueError, match="neutral predicate refusal"):
            next(detections)
    else:
        next(detections)
        assert connections[0].execute("SELECT 1").fetchone() == (1,)
        if settlement == "complete":
            assert list(detections) == [(None, None)]
        else:
            detections.close()
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute("SELECT 1")


def test_projection_consumes_late_invalid_omitted_field() -> None:
    opened = closed = 0

    def events() -> Iterator[tuple[str, object]]:
        nonlocal opened, closed
        opened += 1
        try:
            yield "start_map", None
            yield "map_key", "irrelevant"
            yield "start_array", None
            yield "string", "complete"
            raise ValueError("neutral malformed tail")
        finally:
            closed += 1

    with pytest.raises(ValueError, match="neutral malformed tail"):
        list(detector_registry().iter_record_event_detections(events))
    assert opened == closed


def test_array_binding_releases_failed_projection_before_next_binding_and_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = detector_registry()
    compiled = registry.by_mode[DetectionMode.SEQUENCE_DOCUMENT][0]
    connections: list[sqlite3.Connection] = []
    original = detection_projection._project

    def project(
        events: Iterator[tuple[str, object]],
        event: str,
        value: object,
        rule: detection_projection.DetectorProjection | None,
        stack: ExitStack,
    ) -> object:
        # Recursive _project calls share the same owner; only instrument the root.
        if event == "start_array":
            if connections:
                with pytest.raises(sqlite3.ProgrammingError):
                    connections[-1].execute("SELECT 1")
            document = StreamedJSONDocument(None)
            stack.enter_context(document)
            connections.append(document.connection)
        return original(events, event, value, rule, stack)

    losing = replace(compiled, predicate=lambda _payload: False, provider_resolver=None)
    winning = replace(
        compiled,
        predicate=lambda _payload: True,
        provider_resolver=None,
        binding=replace(compiled.binding, fixed_provider=Provider.CODEX),
    )
    selected = replace(registry, by_mode={DetectionMode.SEQUENCE_DOCUMENT: (losing, winning)})
    monkeypatch.setattr(detection_projection, "_project", project)
    detections = selected.iter_record_event_detections(lambda: _events([{}, {}]), include_singleton=False)
    assert next(detections)[0] is Provider.CODEX
    assert len(connections) == 2
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    detections.close()
