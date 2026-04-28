"""Serialized showcase exercise catalog loader."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polylogue.products.authored_payloads import (
    PayloadDict,
    payload_bool,
    payload_float,
    payload_int,
    payload_items,
    payload_mapping,
    payload_string_tuple,
)
from polylogue.scenarios import (
    AssertionSpec,
    CorpusSpec,
    ExecutionSpec,
    ScenarioMetadata,
)
from polylogue.showcase.exercise_models import Exercise

_CATALOG_PATH = Path(__file__).with_name("exercise_catalog.json")


def _is_integer(output: str, _exit_code: int) -> str | None:
    """Validate output is a single integer."""
    stripped = output.strip()
    if not stripped:
        return "output is empty"
    try:
        int(stripped)
    except ValueError:
        return f"expected integer, got: {stripped!r}"
    return None


def _is_valid_json_array(output: str, _exit_code: int) -> str | None:
    """Validate output is a valid JSON array."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    if not isinstance(data, list):
        return f"expected JSON array, got {type(data).__name__}"
    return None


def _each_line_valid_json(output: str, _exit_code: int) -> str | None:
    """Validate each non-empty line is valid JSON (for streaming JSONL output)."""
    for line_number, line in enumerate(output.strip().splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
        except json.JSONDecodeError as exc:
            return f"line {line_number} invalid JSON: {exc}"
    return None


def _json_only_fields(*allowed: str) -> Callable[[str, int], str | None]:
    """Validate JSON array entries contain only the specified keys."""
    allowed_set = set(allowed)

    def check(output: str, _exit_code: int) -> str | None:
        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            return f"invalid JSON: {exc}"
        if not isinstance(data, list):
            return f"expected JSON array, got {type(data).__name__}"
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            extra = set(item.keys()) - allowed_set
            if extra:
                return f"item {index} has unexpected keys: {extra}"
        return None

    return check


def _search_exit_ok(_output: str, exit_code: int) -> str | None:
    """Search exercises accept exit 0 (results found) or 2 (no results)."""
    if exit_code not in (0, 2):
        return f"unexpected exit code {exit_code} (expected 0 or 2)"
    return None


_CUSTOM_VALIDATORS = {
    "each_line_valid_json": _each_line_valid_json,
    "is_integer": _is_integer,
    "is_valid_json_array": _is_valid_json_array,
    "search_exit_ok": _search_exit_ok,
}


@dataclass(frozen=True)
class ExerciseCatalog:
    groups: tuple[str, ...]
    exercises: tuple[Exercise, ...]


def _load_custom_validator(payload: PayloadDict) -> Callable[[str, int], str | None]:
    kind = str(payload["kind"])
    if kind == "json_only_fields":
        return _json_only_fields(*payload_string_tuple(payload.get("allowed")))
    validator = _CUSTOM_VALIDATORS.get(kind)
    if validator is None:
        raise ValueError(f"Unsupported showcase custom validator: {kind}")
    return validator


def _load_assertion(payload: PayloadDict | None) -> AssertionSpec:
    if payload is None:
        return AssertionSpec()
    custom_payload = payload_mapping(payload.get("custom"))
    custom = None
    if custom_payload is not None:
        custom = _load_custom_validator(dict(custom_payload))
    return AssertionSpec(
        exit_code=payload_int(payload.get("exit_code"), "exit_code") if payload.get("exit_code") is not None else 0,
        stdout_contains=payload_string_tuple(payload.get("stdout_contains")),
        stdout_not_contains=payload_string_tuple(payload.get("stdout_not_contains")),
        stdout_is_valid_json=payload_bool(payload.get("stdout_is_valid_json")),
        stdout_min_lines=payload_int(payload.get("stdout_min_lines"), "stdout_min_lines")
        if payload.get("stdout_min_lines") is not None
        else None,
        custom=custom,
    )


def _load_exercise(payload: PayloadDict) -> Exercise:
    metadata = ScenarioMetadata.from_payload(payload)
    execution_payload = payload_mapping(payload.get("execution"))
    if execution_payload is None:
        raise ValueError("Serialized showcase exercises must declare execution payloads")
    corpus_payloads = payload.get("corpus_specs")
    corpus_specs: list[CorpusSpec] = []
    if isinstance(corpus_payloads, list):
        for spec_payload in corpus_payloads:
            parsed_spec_payload = payload_mapping(spec_payload)
            if parsed_spec_payload is not None:
                corpus_specs.append(CorpusSpec.from_payload(parsed_spec_payload))
    assertion_payload = payload_mapping(payload.get("assertion"))
    return Exercise(
        name=str(payload["name"]),
        group=str(payload["group"]),
        description=str(payload["description"]),
        execution=ExecutionSpec.from_payload(execution_payload),
        corpus_specs=tuple(corpus_specs),
        assertion=_load_assertion(dict(assertion_payload) if assertion_payload is not None else None),
        needs_data=payload_bool(payload.get("needs_data")),
        writes=payload_bool(payload.get("writes")),
        depends_on=str(payload["depends_on"]) if payload.get("depends_on") is not None else None,
        output_ext=str(payload.get("output_ext", ".txt")),
        tier=payload_int(payload.get("tier"), "tier") or 1,
        env=str(payload.get("env", "any")),
        timeout_s=payload_float(payload.get("timeout_s"), "timeout_s") or 120.0,
        vhs_capture=payload_bool(payload.get("vhs_capture")),
        artifact_class=str(payload.get("artifact_class", "text")),
        capture_steps=payload_string_tuple(payload.get("capture_steps")),
        origin=metadata.origin,
        path_targets=metadata.path_targets,
        artifact_targets=metadata.artifact_targets,
        operation_targets=metadata.operation_targets,
        tags=metadata.tags,
    )


def load_exercise_catalog(path: Path | None = None) -> ExerciseCatalog:
    """Load the serialized exercise catalog."""
    catalog_path = path or _CATALOG_PATH
    payload = payload_mapping(json.loads(catalog_path.read_text(encoding="utf-8")))
    if payload is None:
        raise ValueError("Showcase catalog payload must be a JSON object")
    groups = tuple(str(group) for group in payload_items(payload.get("groups")))
    exercises_list: list[Exercise] = []
    for exercise in payload_items(payload.get("exercises")):
        exercise_payload = payload_mapping(exercise)
        if exercise_payload is not None:
            exercises_list.append(_load_exercise(dict(exercise_payload)))
    exercises = tuple(exercises_list)
    return ExerciseCatalog(groups=groups, exercises=exercises)


__all__ = [
    "ExerciseCatalog",
    "load_exercise_catalog",
]
