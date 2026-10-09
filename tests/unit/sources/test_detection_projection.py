"""Decoded projection retains event semantics without visiting ignored values."""

import io
import json
from decimal import Decimal
from pathlib import Path
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
        DetectorProjection(mapping_key_prefix="unknown", mapping_witness={"match": True}),
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


@pytest.mark.parametrize("duplicate_accepted_last", [False, True])
def test_acquisition_key_folds_keep_exact_duplicates_without_key_materialization(
    duplicate_accepted_last: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import ExitStack

    from polylogue.schemas.observation_spill import SpilledKey
    from polylogue.sources.acquisition_boundary import _DocumentValidator, _RecordEvidence

    key = "giant" * (16 * 1024)
    first, last = (b"null", b"[[[]]]") if not duplicate_accepted_last else (b"[[[]]]", b"null")
    raw = b'{"' + key.encode() + b'":' + first + b',"' + key.encode() + b'":' + last + b',"selected":"yes"}'
    rules = [
        DetectorProjection(fields={"selected": DetectorProjection()}, preserve_mapping_size=True),
        DetectorProjection(
            fields={"selected": DetectorProjection()}, capture_metadata_values=True, preserve_mapping_size=True
        ),
        DetectorProjection(mapping_predicate=lambda value: value is None, mapping_witness="accepted"),
        DetectorProjection(mapping_key_prefix="giant", mapping_witness={"match": True}),
    ]
    expected = [project_detection_root(json.loads(raw), rule) for rule in rules]
    observed: list[object] = []

    def unselected_key(self: SpilledKey) -> str:
        raise AssertionError("projection reconstructed an unselected key")

    def validate(self: _RecordEvidence, _bound: object, *, record: bool) -> None:
        try:
            for rule in rules:
                with ExitStack() as stack:
                    events = self.events()
                    event, value = next(events)
                    observed.append(detection_projection._project(events, event, value, rule, stack))
        finally:
            self.close()

    monkeypatch.setattr(SpilledKey, "read", unselected_key)
    monkeypatch.setattr(_RecordEvidence, "validate", validate)
    validator = _DocumentValidator(None, records=True)
    try:
        for start in range(0, len(raw), 4096):
            validator.feed(raw[start : start + 4096])
        validator.finish()
        assert observed == expected
        assert [len(value) for value in observed] == [len(value) for value in expected]  # type: ignore[arg-type]
        assert getattr(observed[1], "metadata_values_scalarish", None) == getattr(
            expected[1], "metadata_values_scalarish", None
        )
        assert validator._tokens.connection.execute("SELECT COUNT(*) FROM projection_keys").fetchone()[0] == 0
    finally:
        validator.close()


def test_completed_tree_mapping_folds_do_not_read_original_giant_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.schemas.observation_spill import SpilledKey, StreamedJSONDocument

    key = "gen_ai." + "k" * (128 * 1024)
    document = {key: [[[]]], "selected": "yes"}
    path = tmp_path / "neutral.json"
    path.write_bytes(json.dumps(document).encode())
    rules = [
        DetectorProjection(fields={"selected": DetectorProjection()}, preserve_mapping_size=True),
        DetectorProjection(
            fields={"selected": DetectorProjection()}, capture_metadata_values=True, preserve_mapping_size=True
        ),
        DetectorProjection(mapping_key_prefix="gen_ai.", mapping_witness={"match": True}),
        DetectorProjection(mapping_predicate=lambda item: item is None, mapping_witness="accepted"),
    ]
    expected = [project_detection_root(document, rule) for rule in rules]

    def unselected_key(self: SpilledKey) -> str:
        raise AssertionError("decoded projection reconstructed an unknown key")

    with StreamedJSONDocument(path) as retained:
        monkeypatch.setattr(SpilledKey, "read", unselected_key)
        actual = [project_detection_root(retained, rule) for rule in rules]
        assert actual == expected
        assert [len(value) for value in actual] == [len(value) for value in expected]  # type: ignore[arg-type]
        assert getattr(actual[1], "metadata_values_scalarish", None) == getattr(
            expected[1], "metadata_values_scalarish", None
        )


@pytest.mark.parametrize("digits", [4301, 65537])
def test_stream_projection_accepts_and_conserves_exact_large_integer(digits: int) -> None:
    from polylogue.sources.detection_projection import project_detection_input

    integer = b"9" * digits
    selected = int(Decimal(integer.decode()))
    wire = b'{"selected":' + integer + b',"ignored":' + integer + b"}"
    rule = DetectorProjection(fields={"selected": DetectorProjection()})
    assert list(iter_projected_document_records(io.BytesIO(wire), rule)) == [{"selected": selected}]
    mode, projected = project_detection_input(io.BytesIO(wire), rule)
    assert mode == "record" and projected == {"selected": selected}


def test_grammar_only_jsonl_successor_accepts_large_integer() -> None:
    from polylogue.core.json_envelope import jsonl_has_record_successor

    giant = b"9" * 65537
    assert jsonl_has_record_successor(io.BytesIO(b'{"ignored":' + giant + b"}\n{}"))
    assert not jsonl_has_record_successor(io.BytesIO(b'{"ignored":' + giant + b"x}\n{}"))


@pytest.mark.parametrize("token", [b"\xed\xa0\x80" + b"\\udc00", b"\\ud800" + b"\xed\xb0\x80"])
def test_stream_projection_preserves_mixed_surrogate_spelling(token: bytes) -> None:
    from polylogue.core.json import decode_provider_utf8
    from polylogue.sources.detection_projection import project_detection_input

    wire = b'{"selected":"' + token + b'"}'
    expected = json.loads(decode_provider_utf8(wire))
    rule = DetectorProjection(fields={"selected": DetectorProjection()})
    assert list(iter_projected_document_records(io.BytesIO(wire), rule)) == [expected]
    assert project_detection_input(io.BytesIO(wire), rule) == ("record", expected)
