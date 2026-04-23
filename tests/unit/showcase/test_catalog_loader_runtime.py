# mypy: disable-error-code="union-attr"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.scenarios import AssertionSpec, CorpusSpec, ScenarioMetadata, polylogue_execution
from polylogue.showcase.catalog_loader import (
    _each_line_valid_json,
    _is_integer,
    _is_valid_json_array,
    _json_only_fields,
    _load_assertion,
    _load_custom_validator,
    _load_exercise,
    _search_exit_ok,
    load_exercise_catalog,
)


def test_is_integer_validates_integer_output() -> None:
    assert _is_integer("42\n", 0) is None
    assert _is_integer("", 0) == "output is empty"
    assert _is_integer("forty-two", 0) == "expected integer, got: 'forty-two'"


def test_is_valid_json_array_checks_shape() -> None:
    assert _is_valid_json_array("[1, 2]", 0) is None
    assert _is_valid_json_array('{"a": 1}', 0) == "expected JSON array, got dict"
    assert _is_valid_json_array("{", 0).startswith("invalid JSON:")


def test_each_line_valid_json_accepts_blank_lines_and_reports_line_number() -> None:
    assert _each_line_valid_json('{"ok": true}\n\n{"id": 1}\n', 0) is None
    assert _each_line_valid_json('{"ok": true}\nnot-json\n', 0).startswith("line 2 invalid JSON:")


def test_json_only_fields_validator_checks_allowed_keys() -> None:
    validator = _json_only_fields("id", "title")

    assert validator('[{"id": 1}, {"title": "ok"}, "skip"]', 0) is None
    assert validator('[{"id": 1, "extra": true}]', 0) == "item 0 has unexpected keys: {'extra'}"
    assert validator("{", 0).startswith("invalid JSON:")
    assert validator('{"id": 1}', 0) == "expected JSON array, got dict"


@pytest.mark.parametrize(
    ("exit_code", "expected"), [(0, None), (2, None), (1, "unexpected exit code 1 (expected 0 or 2)")]
)
def test_search_exit_ok_accepts_only_search_exit_codes(exit_code: int, expected: str | None) -> None:
    assert _search_exit_ok("", exit_code) == expected


def test_load_custom_validator_supports_json_only_fields() -> None:
    validator = _load_custom_validator({"kind": "json_only_fields", "allowed": ["id", "title"]})

    assert validator('[{"id": 1}]', 0) is None
    assert validator('[{"extra": true}]', 0) == "item 0 has unexpected keys: {'extra'}"


def test_load_custom_validator_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported showcase custom validator"):
        _load_custom_validator({"kind": "unsupported"})


def test_load_assertion_supports_custom_validator() -> None:
    assertion = _load_assertion(
        {
            "exit_code": 2,
            "stdout_contains": ["ok"],
            "stdout_not_contains": ["boom"],
            "stdout_is_valid_json": True,
            "stdout_min_lines": 3,
            "custom": {"kind": "is_integer"},
        }
    )

    assert assertion == AssertionSpec(
        exit_code=2,
        stdout_contains=("ok",),
        stdout_not_contains=("boom",),
        stdout_is_valid_json=True,
        stdout_min_lines=3,
        custom=assertion.custom,
    )
    assert assertion.custom is not None
    assert assertion.custom("12", 0) is None
    assert assertion.custom("not-an-int", 0) == "expected integer, got: 'not-an-int'"


def test_load_assertion_defaults_when_payload_is_none() -> None:
    assert _load_assertion(None) == AssertionSpec()


def test_load_exercise_requires_execution_payload() -> None:
    with pytest.raises(ValueError, match="must declare execution payloads"):
        _load_exercise(
            {
                "name": "missing-execution",
                "group": "structural",
                "description": "missing execution",
            }
        )


def test_load_exercise_reads_metadata_assertion_and_corpus_specs() -> None:
    payload = {
        "name": "json-stats",
        "group": "query-read",
        "description": "Show stats as JSON",
        "execution": polylogue_execution("stats", "--json").to_payload(),
        "corpus_specs": [
            CorpusSpec.for_provider("chatgpt", count=2, messages_min=4, messages_max=4).to_payload(),
        ],
        "assertion": {"stdout_is_valid_json": True},
        "tier": 2,
        "env": "seeded",
        "output_ext": ".json",
        "origin": "authored.showcase-catalog",
        "path_targets": ["archive-query-loop"],
        "artifact_targets": ["archive_readiness"],
        "operation_targets": ["project-archive-readiness"],
        "tags": ["json-contract", "query"],
    }

    exercise = _load_exercise(payload)

    assert exercise.name == "json-stats"
    assert exercise.args == ["stats", "--json"]
    assert exercise.corpus_specs[0].provider == "chatgpt"
    assert exercise.assertion.stdout_is_valid_json is True
    assert exercise.tier == 2
    assert exercise.env == "seeded"
    assert exercise.output_ext == ".json"
    assert exercise.origin == "authored.showcase-catalog"
    assert exercise.path_targets == ("archive-query-loop",)
    assert exercise.artifact_targets == ("archive_readiness",)
    assert exercise.operation_targets == ("project-archive-readiness",)
    assert exercise.tags == ("json-contract", "query")


def test_load_exercise_catalog_reads_builtin_catalog() -> None:
    catalog = load_exercise_catalog()

    assert "structural" in catalog.groups
    assert any(exercise.name == "help-main" for exercise in catalog.exercises)


def test_load_exercise_catalog_requires_object_payload(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_exercise_catalog(catalog_path)


def test_load_exercise_catalog_loads_custom_file(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    payload = {
        "groups": ["query-read"],
        "exercises": [
            {
                "name": "custom",
                "group": "query-read",
                "description": "custom exercise",
                "execution": polylogue_execution("stats").to_payload(),
                "metadata": ScenarioMetadata(origin="authored.custom").to_payload(),
            }
        ],
    }
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    catalog = load_exercise_catalog(catalog_path)

    assert catalog.groups == ("query-read",)
    assert catalog.exercises[0].name == "custom"
