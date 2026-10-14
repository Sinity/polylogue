"""Shared payload serialization and row conversion retain their wire contract."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from polylogue.coordination.payloads import CoordinationProvenancePayload
from polylogue.core.json import JSONDocument
from polylogue.core.payload_model import SurfacePayloadModel, model_json_document, serialize_surface_payload


@pytest.mark.parametrize("exclude_none", [False, True])
def test_coordination_payload_preserves_json_bytes_and_null_policy(exclude_none: bool) -> None:
    payload = CoordinationProvenancePayload(source="synthetic", confidence=1.0, freshness="current")
    fields = ['  "source": "synthetic"', '  "command": []']
    if not exclude_none:
        fields.append('  "path": null')
    fields.extend(['  "confidence": 1.0', '  "freshness": "current"'])
    if not exclude_none:
        fields.append('  "note": null')
    expected = "{\n" + ",\n".join(fields) + "\n}"
    assert payload.to_json(exclude_none=exclude_none) == expected
    assert serialize_surface_payload(payload, exclude_none=exclude_none) == expected
    document: JSONDocument = {"source": "synthetic", "command": [], "confidence": 1.0, "freshness": "current"}
    if not exclude_none:
        document.update(path=None, note=None)
    assert model_json_document(payload, exclude_none=exclude_none) == document


def test_row_conversion_ignores_unrelated_attributes_and_applies_overrides() -> None:
    class Payload(SurfacePayloadModel):
        model_name: str
        count: int
        absent: str | None = None

    row = SimpleNamespace(model_name="synthetic", count=2, unrelated="not a payload field")
    payload = Payload._from_row_generic(row, count=3)
    assert payload.model_dump() == {"model_name": "synthetic", "count": 3, "absent": None}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Payload._from_row_generic(row, unrelated="explicit unknown field")
    with pytest.raises(ValidationError, match="frozen_instance"):
        payload.count = 4


def test_surface_catalog_does_not_reexport_shared_payload_owners() -> None:
    from polylogue.surfaces import payloads

    for name in ("SurfacePayloadModel", "serialize_surface_payload", "model_json_document"):
        assert not hasattr(payloads, name)
        assert name not in payloads.__all__
