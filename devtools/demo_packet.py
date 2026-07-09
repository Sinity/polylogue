"""Demo Finding Packet contract (polylogue-212.7).

Background
----------

The 212 demo portfolio (polylogue-212) converts from "a shelf of named demo
scripts" into a PORTFOLIO CONTRACT: every demo is an executable ``PROMPT.md``
handed to a coding agent, and every run of that prompt emits an identical
Demo Finding Packet — the same file shapes regardless of which demo produced
them, so the portfolio is machine-checkable rather than hand-curated.

A packet directory contains:

- ``PROMPT.md`` — the executable prompt (product primitives only; shell/
  python is glue, per the 212 compositionality rule).
- ``finding.yaml`` — the five-part PROVENANCE STANZA (archive cursor,
  measure/query version, commit SHA, sample-frame predicate, run date) plus
  the finding's structural claim.
- ``report.md`` — fixed section order: claim, corpus, method, findings,
  specimens, counterexamples, limits, reproduce.
- ``evidence.ndjson`` — one row per cited ref.
- ``queries.ndjson`` — one row per query: text + lowered spec.
- ``annotations.ndjson`` — optional; external-agent annotations, if any.
- ``checks.json`` — pass/fail + unsupported-claim list + coverage notes.
- ``run.log`` — the raw execution transcript/output.

Provenance stanza note
-----------------------

The five-field shape (``archive_cursor``, ``measure_version``, ``commit_sha``,
``sample_frame_predicate``, ``run_date``) is documented authoritatively by
polylogue-3tl.4, which will also land the publishing pipeline (``devtools
render findings``, ``docs/findings/<slug>/``, the provenance-refusal gate,
living-page changelog semantics) and OWN this schema going forward. 3tl.4 is
not yet implemented. This module inlines the same five fields as a
provisional, self-contained copy so 212.7 does not block on 3tl.4's full
publishing lane -- when 3tl.4 lands, its schema should absorb this one
(matching field names deliberately) rather than diverge.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

PROVENANCE_STANZA_FIELDS: tuple[str, ...] = (
    "archive_cursor",
    "measure_version",
    "commit_sha",
    "sample_frame_predicate",
    "run_date",
)

REPORT_SECTION_ORDER: tuple[str, ...] = (
    "claim",
    "corpus",
    "method",
    "findings",
    "specimens",
    "counterexamples",
    "limits",
    "reproduce",
)

PACKET_FILENAMES: tuple[str, ...] = (
    "PROMPT.md",
    "finding.yaml",
    "report.md",
    "evidence.ndjson",
    "queries.ndjson",
    "checks.json",
    "run.log",
)
#: Optional -- present only when an external-agent annotation loop ran.
OPTIONAL_PACKET_FILENAMES: tuple[str, ...] = ("annotations.ndjson",)


class DemoPacketValidationError(ValueError):
    """Raised when a packet directory or registry entry fails the contract."""


@dataclass(frozen=True)
class PacketValidationResult:
    """Outcome of validating one packet directory."""

    packet_dir: Path
    ok: bool
    missing_files: tuple[str, ...] = ()
    missing_stanza_fields: tuple[str, ...] = ()
    malformed_sections: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "packet_dir": str(self.packet_dir),
            "ok": self.ok,
            "missing_files": list(self.missing_files),
            "missing_stanza_fields": list(self.missing_stanza_fields),
            "malformed_sections": list(self.malformed_sections),
            "errors": list(self.errors),
        }


def _parse_minimal_yaml_mapping(text: str) -> dict[str, str]:
    """Parse a flat ``key: value`` mapping without a YAML dependency.

    ``finding.yaml`` at the level this contract checks is a flat provenance
    stanza plus a few scalar fields -- not a full YAML document. This avoids
    adding a YAML parsing dependency for a shape this simple; a nested/complex
    finding.yaml is out of scope for this validator (which only checks the
    stanza fields are present and non-empty).
    """

    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value.strip().strip("\"'")
    return result


def _validate_ndjson(path: Path) -> str | None:
    """Return an error string if ``path`` is not valid line-delimited JSON, else None."""

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            return f"{path.name}:{line_no}: invalid JSON ({exc})"
    return None


def validate_packet(packet_dir: Path) -> PacketValidationResult:
    """Validate a packet directory against the Demo Finding Packet contract.

    Read-only; never mutates ``packet_dir``.
    """

    missing_files = tuple(name for name in PACKET_FILENAMES if not (packet_dir / name).exists())
    errors: list[str] = []
    missing_stanza_fields: tuple[str, ...] = ()
    malformed_sections: list[str] = []

    finding_path = packet_dir / "finding.yaml"
    if finding_path.exists():
        stanza = _parse_minimal_yaml_mapping(finding_path.read_text(encoding="utf-8"))
        missing_stanza_fields = tuple(name for name in PROVENANCE_STANZA_FIELDS if not stanza.get(name))

    report_path = packet_dir / "report.md"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        for section in REPORT_SECTION_ORDER:
            if f"## {section}" not in report_text.lower() and f"# {section}" not in report_text.lower():
                malformed_sections.append(section)

    checks_path = packet_dir / "checks.json"
    if checks_path.exists():
        try:
            checks_payload = json.loads(checks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"checks.json is not valid JSON: {exc}")
        else:
            for required_key in ("pass", "unsupported_claims", "coverage_notes"):
                if required_key not in checks_payload:
                    errors.append(f"checks.json missing required key: {required_key}")

    for ndjson_name in ("evidence.ndjson", "queries.ndjson"):
        ndjson_path = packet_dir / ndjson_name
        if ndjson_path.exists():
            error = _validate_ndjson(ndjson_path)
            if error is not None:
                errors.append(error)

    ok = not (missing_files or missing_stanza_fields or malformed_sections or errors)
    return PacketValidationResult(
        packet_dir=packet_dir,
        ok=ok,
        missing_files=missing_files,
        missing_stanza_fields=missing_stanza_fields,
        malformed_sections=tuple(malformed_sections),
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class DemoRegistryEntry:
    """One row in the demo registry manifest."""

    slug: str
    prompt_path: str
    packet_dir: str
    mode: str  # "public" | "private"
    required_primitives: tuple[str, ...] = ()

    @staticmethod
    def from_dict(payload: dict[str, object]) -> DemoRegistryEntry:
        try:
            raw_primitives = payload.get("required_primitives", ())
            primitives = raw_primitives if isinstance(raw_primitives, list | tuple) else ()
            return DemoRegistryEntry(
                slug=str(payload["slug"]),
                prompt_path=str(payload["prompt_path"]),
                packet_dir=str(payload["packet_dir"]),
                mode=str(payload["mode"]),
                required_primitives=tuple(str(item) for item in primitives),
            )
        except KeyError as exc:
            raise DemoPacketValidationError(f"registry entry missing required key: {exc}") from None


def load_demo_registry(registry_path: Path) -> tuple[DemoRegistryEntry, ...]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise DemoPacketValidationError(f"{registry_path}: expected a JSON array of registry entries")
    return tuple(DemoRegistryEntry.from_dict(entry) for entry in payload)


@dataclass(frozen=True)
class RegistryLintResult:
    """Outcome of linting every entry in a demo registry manifest."""

    registry_path: Path
    ok: bool
    entry_results: tuple[tuple[str, PacketValidationResult | None], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "registry_path": str(self.registry_path),
            "ok": self.ok,
            "entries": [
                {
                    "slug": slug,
                    "packet_missing": result is None,
                    **({"validation": result.to_dict()} if result is not None else {}),
                }
                for slug, result in self.entry_results
            ],
        }


def lint_demo_registry(registry_path: Path, *, repo_root: Path) -> RegistryLintResult:
    """Validate every registered demo's packet exists and conforms.

    A registry entry whose ``packet_dir`` does not exist at all is reported
    distinctly from one whose packet exists but fails validation -- catching
    a missing packet is the registry lint's specific job (212.7 AC clause 3).
    """

    entries = load_demo_registry(registry_path)
    results: list[tuple[str, PacketValidationResult | None]] = []
    for entry in entries:
        packet_dir = repo_root / entry.packet_dir
        if not packet_dir.is_dir():
            results.append((entry.slug, None))
            continue
        results.append((entry.slug, validate_packet(packet_dir)))

    ok = all(result is not None and result.ok for _, result in results)
    return RegistryLintResult(registry_path=registry_path, ok=ok, entry_results=tuple(results))


def iter_registry_slugs(entries: Iterable[DemoRegistryEntry]) -> tuple[str, ...]:
    return tuple(entry.slug for entry in entries)


__all__ = [
    "OPTIONAL_PACKET_FILENAMES",
    "PACKET_FILENAMES",
    "PROVENANCE_STANZA_FIELDS",
    "REPORT_SECTION_ORDER",
    "DemoPacketValidationError",
    "DemoRegistryEntry",
    "PacketValidationResult",
    "RegistryLintResult",
    "iter_registry_slugs",
    "lint_demo_registry",
    "load_demo_registry",
    "validate_packet",
]
