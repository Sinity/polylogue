"""Typed import and maintenance operation contracts.

Shared by CLI ingest, daemon HTTP, live watcher, and MCP maintenance
so adapters don't drift into separate write/control semantics.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from polylogue.operations.specs import OperationKind


@dataclass(frozen=True, slots=True)
class RawFailureSample:
    """One bounded failure record for an import operation."""

    raw_id: str
    source_path: str | None = None
    error_class: str | None = None
    redacted_text: str | None = None
    attempt_count: int = 1


def bounded_failure_samples(
    failures: dict[str, str],
    *,
    source_path: str | None = None,
    max_samples: int = 50,
) -> list[RawFailureSample]:
    """Convert a raw_id→error dict into bounded, typed failure samples."""
    samples: list[RawFailureSample] = []
    for raw_id, error_text in failures.items():
        if len(samples) >= max_samples:
            break
        error_class: str | None = None
        redacted: str | None = None
        if error_text:
            # Derive a short error class from the first line
            first_line = error_text.split("\n")[0].strip()
            error_class = first_line[:120] + "..." if len(first_line) > 120 else first_line
            redacted = error_text[:500] if len(error_text) > 500 else error_text
        samples.append(
            RawFailureSample(
                raw_id=raw_id,
                source_path=source_path,
                error_class=error_class,
                redacted_text=redacted,
            )
        )
    return samples


@dataclass(frozen=True, slots=True)
class ImportOperation:
    """Result envelope for an import/maintenance operation.

    Returned by CLI ingest, daemon HTTP POST /api/ingest, live watcher
    batches, and MCP maintenance tools.
    """

    operation_id: str
    kind: OperationKind = OperationKind.IMPORT
    status: str = "accepted"
    path: str | None = None
    message: str = ""
    error: str | None = None
    raw_failure_samples: list[RawFailureSample] = field(default_factory=list)

    @classmethod
    def pending(cls, *, operation_id: str, path: str | None = None, message: str = "") -> ImportOperation:
        return cls(operation_id=operation_id, kind=OperationKind.IMPORT, status="pending", path=path, message=message)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImportOperation:
        kind_raw = data.get("kind", "import")
        kind = OperationKind.IMPORT
        if isinstance(kind_raw, str):
            with contextlib.suppress(ValueError):
                kind = OperationKind(kind_raw)
        samples_raw = data.get("raw_failure_samples", [])
        samples: list[RawFailureSample] = []
        if isinstance(samples_raw, list):
            samples = [
                RawFailureSample(
                    raw_id=str(s.get("raw_id", "")),
                    source_path=s.get("source_path"),
                    error_class=s.get("error_class"),
                    redacted_text=s.get("redacted_text"),
                    attempt_count=int(s.get("attempt_count", 1)),
                )
                for s in samples_raw
                if isinstance(s, dict)
            ]
        return cls(
            operation_id=str(data.get("operation_id", "")),
            kind=kind,
            status=str(data.get("status", "unknown")),
            path=data.get("path"),
            message=str(data.get("message", "")),
            error=data.get("error"),
            raw_failure_samples=samples,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": self.status not in ("failed", "error"),
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "status": self.status,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.message:
            result["message"] = self.message
        if self.error is not None:
            result["error"] = self.error
        if self.raw_failure_samples:
            result["raw_failure_samples"] = [
                {
                    "raw_id": s.raw_id,
                    "source_path": s.source_path,
                    "error_class": s.error_class,
                    "redacted_text": s.redacted_text,
                    "attempt_count": s.attempt_count,
                }
                for s in self.raw_failure_samples
            ]
        return result
