"""Demo Packet v2 validation and registry policy (polylogue-212.12).

The original Demo Finding Packet established a uniform directory shape. V2
adds an epistemic contract: a demo is not conforming unless it predeclares one
primary construct and a receipt-backed claim, provides an independently
checkable oracle, baseline, negative and missing-evidence controls, explicit
falsifier, content-bound receipts, and bounded non-claims.

The normative schema lives at ``docs/schemas/demo-packet-v2.schema.json``.
This module intentionally keeps the existing human packet files too: JSON is
the gate, Markdown/NDJSON are the inspectable publication surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

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
    "non-claims",
    "reproduce",
)

PACKET_FILENAMES: tuple[str, ...] = (
    "PROMPT.md",
    "packet.json",
    "finding.yaml",
    "report.md",
    "evidence.ndjson",
    "queries.ndjson",
    "checks.json",
    "NON-CLAIMS.md",
    "run.log",
)
#: Optional -- present only when an external-agent annotation loop ran.
OPTIONAL_PACKET_FILENAMES: tuple[str, ...] = ("annotations.ndjson",)
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "demo-packet-v2.schema.json"
DEFAULT_DEMO_ROOT = Path(".agent/demos")


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
    schema_errors: tuple[str, ...] = ()
    receipt_errors: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "packet_dir": str(self.packet_dir),
            "ok": self.ok,
            "missing_files": list(self.missing_files),
            "missing_stanza_fields": list(self.missing_stanza_fields),
            "malformed_sections": list(self.malformed_sections),
            "schema_errors": list(self.schema_errors),
            "receipt_errors": list(self.receipt_errors),
            "errors": list(self.errors),
        }


def _parse_minimal_yaml_mapping(text: str) -> dict[str, str]:
    """Parse the packet's flat provenance stanza without a full YAML loader."""

    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value.strip().strip("\"'")
    return result


def _validate_ndjson(path: Path) -> str | None:
    """Return an error string if *path* is not valid line-delimited JSON."""

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            return f"{path.name}:{line_no}: invalid JSON ({exc})"
    return None


def _json_pointer(error: object) -> str:
    absolute_path = getattr(error, "absolute_path", ())
    parts = [str(part).replace("~", "~0").replace("/", "~1") for part in absolute_path]
    return "/" + "/".join(parts) if parts else "/"


def _validate_schema(payload: object, *, schema_path: Path) -> tuple[str, ...]:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"schema unavailable at {schema_path}: {exc}",)

    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        return ("jsonschema is required for the Demo Packet v2 policy gate",)

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda item: (list(item.absolute_path), item.message))
    return tuple(f"packet.json{_json_pointer(error)}: {error.message}" for error in errors)


def _receipt_artifact_path(packet_dir: Path, raw_path: object) -> tuple[Path | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "receipt artifact_path must be a non-empty string"
    posix = PurePosixPath(raw_path)
    if posix.is_absolute() or ".." in posix.parts:
        return None, f"receipt artifact_path escapes packet directory: {raw_path!r}"
    candidate = (packet_dir / Path(*posix.parts)).resolve()
    packet_root = packet_dir.resolve()
    if not candidate.is_relative_to(packet_root):
        return None, f"receipt artifact_path escapes packet directory: {raw_path!r}"
    return candidate, None


def _iter_referenced_receipts(value: object, *, at_root: bool = True) -> Iterable[str]:
    """Yield every receipt ref used by an epistemic field in *value*.

    The top-level ``receipts`` member declares the resolver table and is not a
    use site.  Nested members named ``receipts`` are citations and must resolve
    to that table.  The recursive form keeps future packet sections honest
    without requiring a second list of allowed citation locations.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "receipts" and not at_root and isinstance(child, list):
                yield from (ref for ref in child if isinstance(ref, str))
                continue
            yield from _iter_referenced_receipts(child, at_root=False)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_referenced_receipts(child, at_root=False)


def _validate_receipts(packet_dir: Path, payload: object) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        return ()
    raw_receipts = payload.get("receipts")
    if not isinstance(raw_receipts, list):
        return ()  # The schema reports this more precisely.

    errors: list[str] = []
    seen_refs: set[str] = set()
    for index, raw_receipt in enumerate(raw_receipts):
        if not isinstance(raw_receipt, Mapping):
            continue
        ref = raw_receipt.get("ref")
        if isinstance(ref, str):
            if ref in seen_refs:
                errors.append(f"receipt[{index}] duplicates ref {ref!r}")
            seen_refs.add(ref)
            kind = raw_receipt.get("kind")
            expected_kind = ref.partition(":")[0]
            if isinstance(kind, str) and kind != expected_kind:
                errors.append(f"receipt[{index}] kind {kind!r} does not match ref prefix {expected_kind!r}")
        artifact, artifact_error = _receipt_artifact_path(packet_dir, raw_receipt.get("artifact_path"))
        if artifact_error is not None:
            errors.append(f"receipt[{index}]: {artifact_error}")
            continue
        assert artifact is not None
        if not artifact.is_file():
            errors.append(f"receipt[{index}] artifact missing: {artifact.relative_to(packet_dir.resolve())}")
            continue
        try:
            artifact_bytes = artifact.read_bytes()
            content = artifact_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"receipt[{index}] artifact unreadable: {exc}")
            continue
        if isinstance(ref, str) and ref not in content:
            errors.append(
                f"receipt[{index}] ref {ref!r} is not present in {artifact.relative_to(packet_dir.resolve())}"
            )
        expected_sha = raw_receipt.get("sha256")
        if isinstance(expected_sha, str):
            actual_sha = hashlib.sha256(artifact_bytes).hexdigest()
            if actual_sha != expected_sha:
                errors.append(
                    f"receipt[{index}] sha256 mismatch for {artifact.name}: expected {expected_sha}, got {actual_sha}"
                )

    undeclared = sorted(set(_iter_referenced_receipts(payload)) - seen_refs)
    errors.extend(f"referenced receipt is not declared in packet.receipts: {ref!r}" for ref in undeclared)
    return tuple(errors)


def _canonical_report_heading(section: str) -> str:
    return f"## {section[0].upper()}{section[1:]}"


def _report_headings(report_text: str) -> tuple[str, ...]:
    """Return Markdown headings outside fenced code, preserving exact bytes."""

    headings: list[str] = []
    fence: str | None = None
    for line in report_text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is None and line.startswith("#"):
            headings.append(line)
    return tuple(headings)


def _validate_report_sections(report_text: str) -> tuple[str, ...]:
    """Require each canonical level-two heading exactly once and in order."""

    headings = _report_headings(report_text)
    malformed: list[str] = []
    positions: list[int] = []
    for section in REPORT_SECTION_ORDER:
        canonical = _canonical_report_heading(section)
        matches = [index for index, heading in enumerate(headings) if heading == canonical]
        if len(matches) != 1:
            malformed.append(section)
        else:
            positions.append(matches[0])
    if len(positions) == len(REPORT_SECTION_ORDER) and positions != sorted(positions):
        malformed.append("section-order")
    return tuple(malformed)


def _duplicate_strings(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _validate_semantic_consistency(payload: object) -> tuple[str, ...]:
    """Check cross-field invariants that JSON Schema cannot state clearly."""

    if not isinstance(payload, Mapping):
        return ()
    errors: list[str] = []

    falsifier = payload.get("falsifier")
    if isinstance(falsifier, Mapping):
        triggered = falsifier.get("triggered")
        result = falsifier.get("result")
        if triggered is True and result != "fail":
            errors.append("falsifier.triggered=true requires falsifier.result='fail'")
        if result == "fail" and triggered is not True:
            errors.append("falsifier.result='fail' requires falsifier.triggered=true")
        if result == "pass" and triggered is not False:
            errors.append("falsifier.result='pass' requires falsifier.triggered=false")

    controls = payload.get("controls")
    if isinstance(controls, Mapping):
        control_ids: list[object] = []
        for lane in ("negative", "missing_evidence"):
            raw_controls = controls.get(lane)
            if isinstance(raw_controls, list):
                control_ids.extend(control.get("id") for control in raw_controls if isinstance(control, Mapping))
        for duplicate in _duplicate_strings(control_ids):
            errors.append(f"control id is duplicated across packet controls: {duplicate!r}")

    results = payload.get("results")
    if isinstance(results, Mapping):
        measurements = results.get("measurements")
        if isinstance(measurements, list):
            names = [measurement.get("name") for measurement in measurements if isinstance(measurement, Mapping)]
            for duplicate in _duplicate_strings(names):
                errors.append(f"measurement name is duplicated: {duplicate!r}")

    return tuple(errors)


def validate_packet(packet_dir: Path, *, schema_path: Path = DEFAULT_SCHEMA_PATH) -> PacketValidationResult:
    """Validate *packet_dir* against both the human and v2 machine contracts.

    The function is read-only. Receipt resolution is intentionally bounded to
    committed artifacts inside the packet directory; generation-time code may
    additionally resolve the same refs against a live deterministic archive.
    """

    missing_files = tuple(name for name in PACKET_FILENAMES if not (packet_dir / name).exists())
    errors: list[str] = []
    missing_stanza_fields: tuple[str, ...] = ()
    malformed_sections: list[str] = []
    schema_errors: tuple[str, ...] = ()
    receipt_errors: tuple[str, ...] = ()
    packet_payload: object = None

    packet_path = packet_dir / "packet.json"
    if packet_path.exists():
        try:
            packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"packet.json is not valid JSON: {exc}")
        else:
            schema_errors = _validate_schema(packet_payload, schema_path=schema_path)
            receipt_errors = _validate_receipts(packet_dir, packet_payload)
            errors.extend(_validate_semantic_consistency(packet_payload))

    finding_path = packet_dir / "finding.yaml"
    if finding_path.exists():
        stanza = _parse_minimal_yaml_mapping(finding_path.read_text(encoding="utf-8"))
        missing_stanza_fields = tuple(name for name in PROVENANCE_STANZA_FIELDS if not stanza.get(name))

    report_path = packet_dir / "report.md"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        malformed_sections.extend(_validate_report_sections(report_text))

    nonclaims_path = packet_dir / "NON-CLAIMS.md"
    if nonclaims_path.exists() and not nonclaims_path.read_text(encoding="utf-8").strip():
        errors.append("NON-CLAIMS.md must not be empty")

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

    ok = not (missing_files or missing_stanza_fields or malformed_sections or schema_errors or receipt_errors or errors)
    return PacketValidationResult(
        packet_dir=packet_dir,
        ok=ok,
        missing_files=missing_files,
        missing_stanza_fields=missing_stanza_fields,
        malformed_sections=tuple(malformed_sections),
        schema_errors=schema_errors,
        receipt_errors=receipt_errors,
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class DemoRegistryEntry:
    """One row in the demo registry manifest."""

    slug: str
    prompt_path: str
    packet_dir: str
    mode: str  # public | private | fixture | anti-demo
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
    registry_errors: tuple[str, ...] = ()
    unregistered_packet_dirs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "registry_path": str(self.registry_path),
            "ok": self.ok,
            "registry_errors": list(self.registry_errors),
            "unregistered_packet_dirs": list(self.unregistered_packet_dirs),
            "entries": [
                {
                    "slug": slug,
                    "packet_missing": result is None,
                    **({"validation": result.to_dict()} if result is not None else {}),
                }
                for slug, result in self.entry_results
            ],
        }


def _registered_packet_paths(entries: Iterable[DemoRegistryEntry], *, repo_root: Path) -> set[Path]:
    return {(repo_root / entry.packet_dir).resolve() for entry in entries}


def _discover_v2_packet_dirs(*, repo_root: Path) -> set[Path]:
    demo_root = repo_root / DEFAULT_DEMO_ROOT
    if not demo_root.is_dir():
        return set()
    return {path.parent.resolve() for path in demo_root.rglob("packet.json") if path.is_file()}


def lint_demo_registry(
    registry_path: Path,
    *,
    repo_root: Path,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> RegistryLintResult:
    """Validate every registered packet and reject unregistered v2 packets."""

    entries = load_demo_registry(registry_path)
    results: list[tuple[str, PacketValidationResult | None]] = []
    registry_errors: list[str] = []
    slugs: set[str] = set()
    packet_paths: set[Path] = set()

    for entry in entries:
        if entry.slug in slugs:
            registry_errors.append(f"duplicate registry slug: {entry.slug}")
        slugs.add(entry.slug)
        packet_dir = (repo_root / entry.packet_dir).resolve()
        if packet_dir in packet_paths:
            registry_errors.append(f"duplicate registry packet_dir: {entry.packet_dir}")
        packet_paths.add(packet_dir)
        prompt_path = (repo_root / entry.prompt_path).resolve()
        if not prompt_path.is_file():
            registry_errors.append(f"{entry.slug}: prompt missing at {entry.prompt_path}")
        if not packet_dir.is_dir():
            results.append((entry.slug, None))
            continue
        results.append((entry.slug, validate_packet(packet_dir, schema_path=schema_path)))

    discovered = _discover_v2_packet_dirs(repo_root=repo_root)
    registered = _registered_packet_paths(entries, repo_root=repo_root)
    unregistered = tuple(sorted(str(path.relative_to(repo_root.resolve())) for path in discovered - registered))
    ok = not registry_errors and not unregistered and all(result is not None and result.ok for _, result in results)
    return RegistryLintResult(
        registry_path=registry_path,
        ok=ok,
        entry_results=tuple(results),
        registry_errors=tuple(registry_errors),
        unregistered_packet_dirs=unregistered,
    )


def iter_registry_slugs(entries: Iterable[DemoRegistryEntry]) -> tuple[str, ...]:
    return tuple(entry.slug for entry in entries)


__all__ = [
    "DEFAULT_SCHEMA_PATH",
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
