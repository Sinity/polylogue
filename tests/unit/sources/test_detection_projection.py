"""Decoded projection retains event semantics without visiting ignored values."""

import io
import json
from decimal import Decimal
from typing import Literal

import pytest

from polylogue.core.json import JSONDocument, json_document_or_none
from polylogue.sources import detection_projection
from polylogue.sources.detection_projection import (
    DetectorProjection,
    detection_read_view,
    iter_projected_document_records,
    project_detection_root,
)


def test_declared_fields_do_not_visit_unselected_decoded_values(monkeypatch: pytest.MonkeyPatch) -> None:
    ignored = {"nested": ["opaque" * 100_000]}
    value = {"unknown": ignored, "selected": "yes"}
    rule = DetectorProjection(fields={"selected": DetectorProjection(), "absent": None})
    original = detection_projection._project_object
    visited: list[object] = []

    def observe(value: object, rule: DetectorProjection | None, *, scalarish_depth: int | None) -> tuple[object, bool]:
        visited.append(value)
        return original(value, rule, scalarish_depth=scalarish_depth)

    monkeypatch.setattr(detection_projection, "_project_object", observe)
    assert project_detection_root(value, rule) == {"selected": "yes"}
    assert not any(item is ignored for item in visited)


@pytest.mark.parametrize("fold", ["type", "first"])
@pytest.mark.parametrize("count", [0, 1, 8])
def test_decoded_type_and_first_arrays_do_not_project_discarded_items(
    fold: Literal["type", "first"], count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = [{"role": "user"} for _ in range(count)]
    rule = DetectorProjection(
        fields={
            "messages": DetectorProjection(
                item=DetectorProjection(fields={"role": DetectorProjection()}), array_fold=fold
            )
        }
    )
    value = {"messages": values}
    original = detection_projection._project_object
    visited: list[object] = []

    def observe(value: object, rule: DetectorProjection | None, *, scalarish_depth: int | None) -> tuple[object, bool]:
        visited.append(value)
        return original(value, rule, scalarish_depth=scalarish_depth)

    monkeypatch.setattr(detection_projection, "_project_object", observe)
    decoded = project_detection_root(value, rule)
    streamed = list(iter_projected_document_records(io.BytesIO(json.dumps(value).encode()), rule))
    assert decoded == streamed[0]
    assert len(visited) == (4 if count and fold == "first" else 2)
    assert not any(item is tail for item in visited for tail in values[1:])


@pytest.mark.parametrize("fold", ["type", "first"])
def test_decoded_type_and_first_arrays_keep_predicates_and_metadata_folds(fold: Literal["type", "first"]) -> None:
    values = [1, {"nested": [[{}]]}]
    observed: list[object] = []

    def predicate(value: object) -> bool:
        observed.append(value)
        return True

    for capture_metadata, with_predicate in ((False, True), (True, False), (True, True)):
        observed.clear()
        rule = DetectorProjection(
            fields={
                "selected": DetectorProjection(
                    item=DetectorProjection(), array_fold=fold, array_predicate=predicate if with_predicate else None
                )
            },
            preserve_mapping_size=True,
            capture_metadata_values=capture_metadata,
        )
        value = {"selected": values}
        decoded = project_detection_root(value, rule)
        assert len(observed) == (len(values) if with_predicate else 0)
        observed.clear()
        streamed = list(iter_projected_document_records(io.BytesIO(json.dumps(value).encode()), rule))
        assert len(observed) == (len(values) if with_predicate else 0)
        assert decoded == streamed[0]
        assert getattr(decoded, "metadata_values_scalarish", None) == getattr(
            streamed[0], "metadata_values_scalarish", None
        )
        if capture_metadata:
            assert getattr(decoded, "metadata_values_scalarish", None) is False


@pytest.mark.parametrize(
    "rule",
    [
        DetectorProjection(fields={"selected": DetectorProjection()}),
        DetectorProjection(fields={"selected": DetectorProjection()}, preserve_mapping_size=True),
        DetectorProjection(fields={"selected": DetectorProjection()}, capture_metadata_values=True),
        DetectorProjection(mapping_predicate=lambda item: item is None, mapping_witness="accepted"),
        DetectorProjection(mapping_key_predicate=lambda key: key == "unknown", mapping_witness={"match": True}),
    ],
)
def test_decoded_projection_matches_event_route_in_each_mapping_branch(rule: DetectorProjection) -> None:
    value = {"unknown": {"deep": [[{"opaque": "value"}]]}, "selected": "yes", "another": []}
    decoded = project_detection_root(value, rule)
    streamed = list(iter_projected_document_records(io.BytesIO(json.dumps(value).encode()), rule))
    assert len(streamed) == 1
    assert decoded == streamed[0]
    assert len(decoded) == len(streamed[0])  # type: ignore[arg-type]
    assert getattr(decoded, "metadata_values_scalarish", None) == getattr(
        streamed[0], "metadata_values_scalarish", None
    )


@pytest.mark.parametrize("scalar", ["synthetic", Decimal("2.5"), object()])
@pytest.mark.parametrize("preserve_size", [False, True])
def test_projection_read_view_validates_json_once(
    scalar: object, preserve_size: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatch predicates and dynamic resolution share conversion, including refusal."""
    from polylogue.sources.dispatch import _payload_record

    original = json_document_or_none
    calls: list[object] = []

    def observe(value: object) -> JSONDocument | None:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(detection_projection, "json_document_or_none", observe)
    rule = DetectorProjection(fields={"selected": DetectorProjection()}, preserve_mapping_size=preserve_size)
    projected = detection_read_view(project_detection_root({"selected": scalar, "unselected": None}, rule))
    first = _payload_record(projected)
    assert _payload_record(projected) is first
    assert len(calls) == 1
    if type(scalar) is object:
        assert first is None
    else:
        assert first is projected
        assert first is not None
        assert len(first) == (2 if preserve_size else 1)
        assert first["selected"] == (2.5 if isinstance(scalar, Decimal) else scalar)
