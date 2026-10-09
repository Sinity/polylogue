"""Decoded projection retains event semantics without visiting ignored values."""

import io
import json

import pytest

from polylogue.sources import detection_projection
from polylogue.sources.detection_projection import (
    DetectorProjection,
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
