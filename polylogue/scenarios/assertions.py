"""Shared success-criteria models for executable scenarios."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .payloads import PayloadDict, payload_float, payload_int, payload_items


@dataclass(frozen=True)
class AssertionSpec:
    """Expected success criteria for an authored executable scenario."""

    exit_code: int | None = 0
    stdout_contains: tuple[str, ...] = ()
    stdout_not_contains: tuple[str, ...] = ()
    stdout_is_valid_json: bool = False
    stdout_min_lines: int | None = None
    benchmark_warn_pct: float | None = None
    benchmark_fail_pct: float | None = None
    custom: Callable[[str, int], str | None] | None = None

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
        return payload

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


__all__ = ["AssertionSpec"]
