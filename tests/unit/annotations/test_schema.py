"""Tests for the annotation schema registry (polylogue-rxdo.7)."""

from __future__ import annotations

import pytest

from polylogue.annotations.schema import (
    AnnotationField,
    AnnotationSchema,
    AnnotationSchemaError,
    AnnotationSchemaRegistry,
    validate_annotation_row,
    validate_annotation_value,
)


def _schema(**overrides: object) -> AnnotationSchema:
    defaults: dict[str, object] = {
        "schema_id": "test.delegation-tone",
        "version": 1,
        "title": "Test delegation tone",
        "fields": (
            AnnotationField(name="score", value_type="integer", minimum=1, maximum=5),
            AnnotationField(name="status", value_type="enum", enum_values=("approved", "rejected")),
            AnnotationField(name="note", value_type="string", required=False),
            AnnotationField(name="abstain", value_type="boolean", required=False),
        ),
        "target_ref_kinds": ("session", "message"),
        "abstain_field": "abstain",
        "evidence_policy": "required",
    }
    defaults.update(overrides)
    return AnnotationSchema(**defaults)  # type: ignore[arg-type]


class TestAnnotationFieldValidation:
    def test_string_field_accepts_str_rejects_other_types(self) -> None:
        field = AnnotationField(name="note", value_type="string", required=False)
        assert field.validate_value("hello") is None
        assert field.validate_value(5) is not None

    def test_boolean_field_rejects_non_bool(self) -> None:
        field = AnnotationField(name="flag", value_type="boolean", required=False)
        assert field.validate_value(True) is None
        assert field.validate_value(1) is not None

    def test_enum_field_rejects_unknown_value(self) -> None:
        field = AnnotationField(name="status", value_type="enum", enum_values=("a", "b"))
        assert field.validate_value("a") is None
        assert field.validate_value("c") is not None

    def test_integer_field_rejects_float_and_bool(self) -> None:
        field = AnnotationField(name="score", value_type="integer")
        assert field.validate_value(3) is None
        assert field.validate_value(3.5) is not None
        assert field.validate_value(True) is not None

    def test_number_field_accepts_int_and_float(self) -> None:
        field = AnnotationField(name="confidence", value_type="number", minimum=0.0, maximum=1.0)
        assert field.validate_value(0.5) is None
        assert field.validate_value(1) is None
        assert field.validate_value(1.5) is not None

    def test_numeric_bounds_enforced(self) -> None:
        field = AnnotationField(name="score", value_type="integer", minimum=1, maximum=5)
        assert field.validate_value(0) is not None
        assert field.validate_value(6) is not None
        assert field.validate_value(1) is None
        assert field.validate_value(5) is None

    def test_enum_field_without_values_rejected_at_construction(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            AnnotationField(name="status", value_type="enum", enum_values=())

    def test_non_enum_field_with_enum_values_rejected(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            AnnotationField(name="status", value_type="string", enum_values=("a",))

    def test_bounds_on_non_numeric_field_rejected(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            AnnotationField(name="note", value_type="string", minimum=0)

    def test_invalid_field_name_rejected(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            AnnotationField(name="Score", value_type="integer")
        with pytest.raises(AnnotationSchemaError):
            AnnotationField(name="_score", value_type="integer")


class TestAnnotationSchemaConstruction:
    def test_valid_schema_constructs(self) -> None:
        schema = _schema()
        assert schema.qualified_id == "test.delegation-tone@v1"

    def test_duplicate_field_names_rejected(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(
                fields=(
                    AnnotationField(name="score", value_type="integer"),
                    AnnotationField(name="score", value_type="string"),
                )
            )

    def test_unknown_target_ref_kind_rejected(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(target_ref_kinds=("not-a-real-kind",))

    def test_empty_fields_rejected(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(fields=())

    def test_version_must_be_positive(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(version=0)

    def test_invalid_schema_id_rejected(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(schema_id="Not Valid")

    def test_abstain_field_must_be_declared(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(abstain_field="does_not_exist")

    def test_abstain_field_must_be_boolean(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(abstain_field="status")

    def test_abstain_field_must_not_be_required(self) -> None:
        with pytest.raises(AnnotationSchemaError):
            _schema(
                fields=(
                    AnnotationField(name="score", value_type="integer"),
                    AnnotationField(name="abstain", value_type="boolean", required=True),
                ),
                abstain_field="abstain",
            )

    def test_accepts_target_kind(self) -> None:
        schema = _schema()
        assert schema.accepts_target_kind("session:abc")
        assert schema.accepts_target_kind("message:abc")
        assert not schema.accepts_target_kind("block:abc")
        assert not schema.accepts_target_kind("not-a-ref")


class TestValidateAnnotationValue:
    def test_valid_row_has_no_errors(self) -> None:
        schema = _schema()
        errors = validate_annotation_value(schema, {"score": 4, "status": "approved"})
        assert errors == []

    def test_missing_required_field_reported(self) -> None:
        schema = _schema()
        errors = validate_annotation_value(schema, {"score": 4})
        assert any("status" in error for error in errors)

    def test_unknown_field_reported(self) -> None:
        schema = _schema()
        errors = validate_annotation_value(schema, {"score": 4, "status": "approved", "bogus": 1})
        assert any("bogus" in error for error in errors)

    def test_type_error_reported(self) -> None:
        schema = _schema()
        errors = validate_annotation_value(schema, {"score": "not-a-number", "status": "approved"})
        assert any("score" in error for error in errors)

    def test_abstain_true_exempts_other_required_fields(self) -> None:
        schema = _schema()
        errors = validate_annotation_value(schema, {"abstain": True})
        assert errors == []

    def test_abstain_false_does_not_exempt_required_fields(self) -> None:
        schema = _schema()
        errors = validate_annotation_value(schema, {"abstain": False})
        assert any("score" in error for error in errors)
        assert any("status" in error for error in errors)

    def test_abstain_true_still_type_checks_present_fields(self) -> None:
        schema = _schema()
        errors = validate_annotation_value(schema, {"abstain": True, "score": "bad"})
        assert any("score" in error for error in errors)


class TestValidateAnnotationRow:
    def test_wrong_target_kind_reported(self) -> None:
        schema = _schema()
        errors = validate_annotation_row(
            schema,
            target_ref="block:abc:0",
            value={"score": 4, "status": "approved"},
            evidence_refs=["session:abc"],
        )
        assert any("target_ref_kinds" in error for error in errors)

    def test_missing_evidence_reported_when_required(self) -> None:
        schema = _schema()
        errors = validate_annotation_row(
            schema,
            target_ref="session:abc",
            value={"score": 4, "status": "approved"},
            evidence_refs=(),
        )
        assert any("evidence" in error for error in errors)

    def test_evidence_optional_policy_does_not_require_refs(self) -> None:
        schema = _schema(evidence_policy="optional")
        errors = validate_annotation_row(
            schema,
            target_ref="session:abc",
            value={"score": 4, "status": "approved"},
            evidence_refs=(),
        )
        assert errors == []

    def test_fully_valid_row_has_no_errors(self) -> None:
        schema = _schema()
        errors = validate_annotation_row(
            schema,
            target_ref="session:abc",
            value={"score": 4, "status": "approved"},
            evidence_refs=["session:abc"],
        )
        assert errors == []


class TestAnnotationSchemaRegistry:
    def test_register_and_get_latest_version(self) -> None:
        registry = AnnotationSchemaRegistry()
        v1 = _schema(version=1)
        v2 = _schema(version=2)
        registry.register(v1)
        registry.register(v2)
        assert registry.get("test.delegation-tone") is v2
        assert registry.get("test.delegation-tone", version=1) is v1

    def test_get_missing_schema_raises_key_error(self) -> None:
        registry = AnnotationSchemaRegistry()
        with pytest.raises(KeyError):
            registry.get("does.not.exist")

    def test_get_missing_version_raises_key_error(self) -> None:
        registry = AnnotationSchemaRegistry()
        registry.register(_schema(version=1))
        with pytest.raises(KeyError):
            registry.get("test.delegation-tone", version=2)

    def test_reregistering_identical_schema_is_idempotent(self) -> None:
        registry = AnnotationSchemaRegistry()
        schema = _schema()
        registry.register(schema)
        registry.register(schema)  # no raise
        assert registry.list() == (schema,)

    def test_reregistering_drifted_schema_raises(self) -> None:
        registry = AnnotationSchemaRegistry()
        registry.register(_schema(title="Original"))
        with pytest.raises(AnnotationSchemaError):
            registry.register(_schema(title="Changed"))

    def test_list_sorted_by_schema_id_and_version(self) -> None:
        registry = AnnotationSchemaRegistry()
        b1 = _schema(schema_id="test.b", version=1)
        a2 = _schema(schema_id="test.a", version=2)
        a1 = _schema(schema_id="test.a", version=1)
        registry.register(b1)
        registry.register(a2)
        registry.register(a1)
        assert registry.list() == (a1, a2, b1)
