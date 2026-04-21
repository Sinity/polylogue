"""Durable regression-case records for verification probe failures."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from polylogue.lib.json import JSONDocument, loads, require_json_document

SCHEMA_VERSION = 1
DEFAULT_CAPTURE_KEYS: tuple[str, ...] = (
    "probe",
    "paths",
    "provenance",
    "result",
    "run_payload",
    "db_stats",
    "raw_fanout",
    "source_files",
    "source_inputs",
    "sample",
    "budgets",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "regression"


def _stable_digest(payload: JSONDocument) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()[:12]


def _string_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value for value in values if value)


def _selected_summary(summary: JSONDocument, capture_keys: tuple[str, ...]) -> JSONDocument:
    selected: JSONDocument = {}
    for key in capture_keys:
        if key in summary:
            selected[key] = summary[key]
    if not selected:
        raise ValueError("regression case summary did not contain any recognized capture keys")
    return selected


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """A minimized, replayable record derived from a failed verification probe."""

    case_id: str
    name: str
    source: str
    created_at: str
    summary: JSONDocument
    provenance: JSONDocument = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_probe_summary(
        cls,
        *,
        name: str,
        summary: JSONDocument,
        tags: Iterable[str] = (),
        notes: Iterable[str] = (),
        created_at: str | None = None,
        capture_keys: tuple[str, ...] = DEFAULT_CAPTURE_KEYS,
    ) -> RegressionCase:
        """Create a regression case from a ``devtools pipeline-probe`` summary."""
        if "probe" not in summary or "result" not in summary:
            raise ValueError("pipeline probe regression cases require `probe` and `result` summary keys")
        selected = _selected_summary(summary, capture_keys)
        provenance = require_json_document(summary.get("provenance", {}), context="probe provenance")
        case_id = f"{_slug(name)}-{_stable_digest(selected)}"
        return cls(
            case_id=case_id,
            name=name,
            source="pipeline-probe",
            created_at=created_at or _utc_now(),
            summary=selected,
            provenance=provenance,
            tags=_string_tuple(tags),
            notes=_string_tuple(notes),
        )

    @classmethod
    def from_payload(cls, payload: JSONDocument) -> RegressionCase:
        """Hydrate a stored regression case payload."""
        raw_schema_version = payload.get("schema_version")
        schema_version = raw_schema_version if isinstance(raw_schema_version, int) else 0
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported regression case schema_version: {schema_version}")
        tags_raw = payload.get("tags", [])
        notes_raw = payload.get("notes", [])
        tags = tuple(str(value) for value in tags_raw) if isinstance(tags_raw, list) else ()
        notes = tuple(str(value) for value in notes_raw) if isinstance(notes_raw, list) else ()
        return cls(
            case_id=str(payload["case_id"]),
            name=str(payload["name"]),
            source=str(payload["source"]),
            created_at=str(payload["created_at"]),
            summary=require_json_document(payload["summary"], context="regression case summary"),
            provenance=require_json_document(payload.get("provenance", {}), context="regression case provenance"),
            tags=tags,
            notes=notes,
            schema_version=schema_version,
        )

    @classmethod
    def read(cls, path: Path) -> RegressionCase:
        """Read a regression case JSON file."""
        return cls.from_payload(require_json_document(loads(path.read_text(encoding="utf-8"))))

    def to_payload(self) -> JSONDocument:
        """Render the regression case as a stable JSON document."""
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "name": self.name,
            "source": self.source,
            "created_at": self.created_at,
            "tags": list(self.tags),
            "notes": list(self.notes),
            "provenance": self.provenance,
            "summary": self.summary,
        }

    def to_json_text(self) -> str:
        """Render pretty, deterministic JSON for durable storage."""
        return json.dumps(self.to_payload(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def write(self, directory: Path) -> Path:
        """Write the case under ``directory`` and return the JSON path."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.case_id}.json"
        path.write_text(self.to_json_text(), encoding="utf-8")
        return path


@dataclass(frozen=True, slots=True)
class RegressionCaseStore:
    """Repository-local store for captured regression cases."""

    root: Path

    def write(self, case: RegressionCase) -> Path:
        return case.write(self.root)

    def read(self, case_id: str) -> RegressionCase:
        return RegressionCase.read(self.root / f"{case_id}.json")

    def list_paths(self) -> tuple[Path, ...]:
        if not self.root.exists():
            return ()
        return tuple(sorted(self.root.glob("*.json")))


def regression_case_path_payload(case: RegressionCase, path: Path) -> JSONDocument:
    """Render a small machine payload after capture."""
    payload = case.to_payload()
    payload["path"] = str(path)
    return payload


def json_input_document(raw: str | bytes) -> JSONDocument:
    """Parse command input as a JSON document."""
    return require_json_document(loads(raw), context="regression capture input")


__all__ = [
    "DEFAULT_CAPTURE_KEYS",
    "RegressionCase",
    "RegressionCaseStore",
    "json_input_document",
    "regression_case_path_payload",
]
