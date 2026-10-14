"""Required fields and publication metadata follow their actual owners."""

from __future__ import annotations

import json

import pytest

from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.classification import classify_schema_constructs
from polylogue.schemas.synthetic.models import SchemaRecord


def test_required_fields_are_drawn_despite_optional_marginal_frequency() -> None:
    schema: SchemaRecord = {
        "type": "object",
        "properties": {"value": {"type": "integer", "x-polylogue-frequency": 0.1}},
        "required": ["value"],
    }
    assert all(item.state == "supported" for item in classify_schema_constructs(schema))
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral")
    try:
        for raw in corpus.generate(count=16, messages_per_session=range(1, 2), seed=0):
            value = json.loads(raw)
            assert isinstance(value["value"], int)
    finally:
        corpus._relation_solver.close()


def test_required_fields_without_a_declared_property_schema_refuse() -> None:
    schema: SchemaRecord = {"type": "object", "properties": {"other": {"type": "integer"}}, "required": ["missing"]}
    assert (
        next(item for item in classify_schema_constructs(schema) if item.construct == "required").state == "unsupported"
    )
    corpus = SyntheticCorpus(schema, WireFormat(encoding="json"), "neutral")
    try:
        with pytest.raises(ValueError, match="required fields"):
            corpus.generate(count=1, messages_per_session=range(1, 2), seed=0)
    finally:
        corpus._relation_solver.close()


def test_known_publication_counters_are_metadata_and_unknown_extensions_refuse() -> None:
    schema: SchemaRecord = {
        "type": "integer",
        "x-polylogue-publication-omitted-structure-witness-count": 0,
        "x-polylogue-source-evidence-unretained-shape-observation-lower-bound": 0,
    }
    assert all(item.state == "supported" for item in classify_schema_constructs(schema))
    schema["x-unrecognized-constraint"] = True
    assert (
        next(item for item in classify_schema_constructs(schema) if item.construct == "x-unrecognized-constraint").state
        == "unsupported"
    )
