"""Shared success-criteria models for executable scenarios."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from polylogue.insights.authored_payloads import PayloadDict, payload_float, payload_int, payload_items


class AssertionClass(Enum):
    """Classifies what kind of evidence an AssertionSpec provides.

    The classification is auto-derived from which assertion fields are
    populated, but callers may override it explicitly when the auto-derived
    class does not match intent (e.g. a lane that runs pytest tests owns
    its semantic checks in the test files, not in the AssertionSpec).
    """

    SMOKE_PROCESS = "smoke/process"
    SEMANTIC_OUTPUT = "semantic-output"
    RUNTIME_BUDGET = "runtime-budget"
    LIVE_OBSERVABILITY = "live-observability"
    METADATA_SPEC = "metadata/spec"


def _classify_assertion(
    stdout_contains: tuple[str, ...],
    stdout_not_contains: tuple[str, ...],
    stdout_is_valid_json: bool,
    custom: object,
    benchmark_warn_pct: object,
    benchmark_fail_pct: object,
) -> AssertionClass:
    """Auto-classify an assertion spec from which fields are populated."""
    if benchmark_warn_pct is not None or benchmark_fail_pct is not None:
        return AssertionClass.RUNTIME_BUDGET
    if stdout_contains or stdout_not_contains or stdout_is_valid_json or custom is not None:
        return AssertionClass.SEMANTIC_OUTPUT
    return AssertionClass.SMOKE_PROCESS


@dataclass(frozen=True)
class AssertionSpec:
    """Expected success criteria for an authored executable scenario.

    The default AssertionSpec(exit_code=0) only checks that the process exits
    cleanly - it is a smoke/process check, not semantic evidence.  Semantic
    assertions require explicit stdout_contains, stdout_not_contains,
    stdout_is_valid_json, or custom fields.
    """

    exit_code: int | None = 0
    stdout_contains: tuple[str, ...] = ()
    stdout_not_contains: tuple[str, ...] = ()
    stdout_is_valid_json: bool = False
    stdout_min_lines: int | None = None
    benchmark_warn_pct: float | None = None
    benchmark_fail_pct: float | None = None
    custom: Callable[[str, int], str | None] | None = None
    classification_override: AssertionClass | None = None

    classification: AssertionClass = field(init=False)

    def __post_init__(self) -> None:
        """Auto-classify from populated fields unless explicitly overridden."""
        klass = self.classification_override or _classify_assertion(
            self.stdout_contains,
            self.stdout_not_contains,
            self.stdout_is_valid_json,
            self.custom,
            self.benchmark_warn_pct,
            self.benchmark_fail_pct,
        )
        object.__setattr__(self, "classification", klass)

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {}
        if self.exit_code != 0:
            payload["exit_code"] = self.exit_code
        if self.stdout_contains:
            payload["stdout_contains"] = list(self.stdout_contains)
        if self.stdout_not_contains:
            payload["stdout_not_contains"] = list(self.stdout_not_contains)
        if self.stdout_is_valid_json:
            payload["stdout_is_valid_json"] = True
        if self.stdout_min_lines is not None:
            payload["stdout_min_lines"] = self.stdout_min_lines
        if self.benchmark_warn_pct is not None:
            payload["benchmark_warn_pct"] = self.benchmark_warn_pct
        if self.benchmark_fail_pct is not None:
            payload["benchmark_fail_pct"] = self.benchmark_fail_pct
        if self.classification_override is not None:
            payload["classification"] = self.classification.value
        return payload

    def validate(self) -> list[str]:
        """Return validation warnings if classification doesn't match assertion content.

        A lane classified as SEMANTIC_OUTPUT should have at least one
        stdout assertion or point to pytest tests.  This method flags
        lanes that claim semantic evidence but only check exit codes.
        """
        if self.classification != AssertionClass.SEMANTIC_OUTPUT:
            return []
        has_semantic = bool(
            self.stdout_contains
            or self.stdout_not_contains
            or self.stdout_is_valid_json
            or self.stdout_min_lines is not None
            or self.custom is not None
        )
        if not has_semantic:
            return [
                "SEMANTIC_OUTPUT classification requires stdout assertions "
                "(stdout_contains, stdout_not_contains, stdout_is_valid_json, "
                "stdout_min_lines, or custom). Either add semantic assertions "
                "or override classification to SMOKE_PROCESS."
            ]
        return []

    @classmethod
    def from_payload(cls, payload: Mapping[str, object] | None) -> AssertionSpec:
        if payload is None:
            return cls()
        return cls(
            exit_code=payload_int(payload.get("exit_code"), "exit_code") if payload.get("exit_code") is not None else 0,
            stdout_contains=tuple(str(item) for item in payload_items(payload.get("stdout_contains"))),
            stdout_not_contains=tuple(str(item) for item in payload_items(payload.get("stdout_not_contains"))),
            stdout_is_valid_json=bool(payload.get("stdout_is_valid_json", False)),
            stdout_min_lines=payload_int(payload.get("stdout_min_lines"), "stdout_min_lines"),
            benchmark_warn_pct=payload_float(payload.get("benchmark_warn_pct"), "benchmark_warn_pct"),
            benchmark_fail_pct=payload_float(payload.get("benchmark_fail_pct"), "benchmark_fail_pct"),
            classification_override=_parse_classification(payload.get("classification")),
        )

    def validate_process(self, output: str, exit_code: int) -> str | None:
        if self.exit_code is not None and exit_code != self.exit_code:
            return f"exit code {exit_code}, expected {self.exit_code}"

        for needle in self.stdout_contains:
            if needle not in output:
                return f"output missing {needle!r}"

        for needle in self.stdout_not_contains:
            if needle in output:
                return f"output unexpectedly contains {needle!r}"

        if self.stdout_is_valid_json:
            try:
                json.loads(output)
            except json.JSONDecodeError as exc:
                return f"invalid JSON: {exc}"

        if self.stdout_min_lines is not None:
            line_count = len(output.strip().splitlines())
            if line_count < self.stdout_min_lines:
                return f"only {line_count} lines, expected >= {self.stdout_min_lines}"

        if self.custom:
            return self.custom(output, exit_code)

        return None

    def resolved_benchmark_warn_pct(self, default: float = 0.0) -> float:
        return default if self.benchmark_warn_pct is None else self.benchmark_warn_pct

    def resolved_benchmark_fail_pct(self, default: float = 0.0) -> float:
        return default if self.benchmark_fail_pct is None else self.benchmark_fail_pct


def _parse_classification(value: object) -> AssertionClass | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return AssertionClass(value)
        except ValueError:
            return None
    return None


def classification_label(klass: AssertionClass) -> str:
    """Human-readable label for assertion classification."""
    return klass.value


__all__ = ["AssertionClass", "AssertionSpec", "classification_label"]
