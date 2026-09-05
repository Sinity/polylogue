"""``devtools gate population-coverage``: the real source inventory is declared and witnessed.

Two halves, both read-only and neither able to create a fixture or a parser:

* **Declarations**: every executable ``OriginSpec`` has a capability-matrix
  entry whose witness fixtures exist on disk, and every non-executable origin
  carries an unsupported receipt. Runs without an archive.
* **Inventory**: every origin, detector route (``detected_provider``), and
  artifact kind observed in an archive's ``source.db`` maps to a declared
  parser route, a declared artifact rule, or a typed unsupported exclusion.
  A construct nothing declares is reported as typed unsupported evidence and
  fails the gate.

Ordinary value variation inside a declared construct is not a construct; a
new origin token, detector route, or artifact kind is.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.artifact_taxonomy.models import ArtifactKind
from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.core.sources import origin_from_provider
from polylogue.sources.origin_specs import ORIGIN_SPECS, OriginSpec
from polylogue.storage.introspection import table_exists
from tests.infra.origin_capability_matrix import CapabilityManifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

COVERED = "covered"
UNSUPPORTED_DECLARED = "unsupported_declared"
UNCOVERED = "uncovered"

#: Artifact kinds a session-bearing parser route consumes directly.
_SESSION_BEARING_KINDS: frozenset[str] = frozenset(
    {
        ArtifactKind.SESSION_DOCUMENT.value,
        ArtifactKind.SESSION_RECORD_STREAM.value,
        ArtifactKind.AGENT_TRANSCRIPT.value,
        ArtifactKind.COORDINATOR_SESSION_STREAM.value,
    }
)


@dataclass(frozen=True, slots=True)
class CoverageConstruct:
    """One construct of the population and how it is covered."""

    family: str
    key: str
    status: str
    route: str
    witness: str
    count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "key": self.key,
            "status": self.status,
            "route": self.route,
            "witness": self.witness,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class PopulationCoverageReport:
    archive_root: str | None
    inventory_evaluated: bool
    constructs: tuple[CoverageConstruct, ...]

    @property
    def uncovered(self) -> tuple[CoverageConstruct, ...]:
        return tuple(construct for construct in self.constructs if construct.status == UNCOVERED)

    @property
    def ok(self) -> bool:
        return not self.uncovered

    def to_dict(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for construct in self.constructs:
            counts[construct.status] = counts.get(construct.status, 0) + 1
        return {
            "ok": self.ok,
            "archive_root": self.archive_root,
            "inventory_evaluated": self.inventory_evaluated,
            "summary": counts,
            "constructs": [construct.to_dict() for construct in self.constructs],
        }


def _spec_by_origin(specs: Sequence[OriginSpec]) -> dict[str, OriginSpec]:
    return {spec.origin.value: spec for spec in specs}


def _matrix_witness(manifest: CapabilityManifest, origin: str) -> tuple[str, str] | None:
    """Return ``(status, witness)`` for an origin from the capability matrix."""
    for entry in manifest.entries:
        if entry.origin.value != origin:
            continue
        if entry.unsupported is not None:
            return UNSUPPORTED_DECLARED, f"matrix unsupported: {entry.unsupported.reason}"
        present = [witness.fixture_path for witness in entry.witnesses if (REPO_ROOT / witness.fixture_path).is_file()]
        if not present:
            return None
        return COVERED, ";".join(present)
    return None


def declaration_constructs(
    *, specs: Sequence[OriginSpec] = ORIGIN_SPECS, manifest: CapabilityManifest | None = None
) -> tuple[CoverageConstruct, ...]:
    """Every declared origin has a witness (executable) or an unsupported receipt."""
    manifest = manifest if manifest is not None else load_manifest()
    out: list[CoverageConstruct] = []
    for spec in specs:
        origin = spec.origin.value
        witness = _matrix_witness(manifest, origin)
        route = ";".join(spec.parser_paths) or f"lifecycle:{spec.lifecycle}"
        if spec.lifecycle == "executable":
            if witness is None or witness[0] != COVERED:
                out.append(CoverageConstruct("origin-declaration", origin, UNCOVERED, route, "no matrix witness"))
            else:
                out.append(CoverageConstruct("origin-declaration", origin, COVERED, route, witness[1]))
        elif witness is None:
            out.append(CoverageConstruct("origin-declaration", origin, UNCOVERED, route, "no unsupported receipt"))
        else:
            out.append(CoverageConstruct("origin-declaration", origin, UNSUPPORTED_DECLARED, route, witness[1]))
    return tuple(out)


def inventory_constructs(
    source_db: Path,
    *,
    specs: Sequence[OriginSpec] = ORIGIN_SPECS,
    manifest: CapabilityManifest | None = None,
) -> tuple[CoverageConstruct, ...]:
    """Classify every origin, detector route, and artifact kind in ``source_db``."""
    manifest = manifest if manifest is not None else load_manifest()
    by_origin = _spec_by_origin(specs)
    executable_wires: dict[str, str] = {
        provider.value: spec.origin.value
        for spec in specs
        if spec.lifecycle == "executable"
        for provider in spec.provider_wires
    }
    out: list[CoverageConstruct] = []
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        for origin, count in conn.execute("SELECT origin, COUNT(*) FROM raw_sessions GROUP BY origin"):
            origin = str(origin)
            spec = by_origin.get(origin)
            if spec is None:
                out.append(CoverageConstruct("origin", origin, UNCOVERED, "no OriginSpec", "none", int(count)))
                continue
            witness = _matrix_witness(manifest, origin)
            if spec.lifecycle == "executable" and witness is not None and witness[0] == COVERED:
                out.append(
                    CoverageConstruct("origin", origin, COVERED, ";".join(spec.parser_paths), witness[1], int(count))
                )
            elif witness is not None and witness[0] == UNSUPPORTED_DECLARED:
                out.append(
                    CoverageConstruct(
                        "origin", origin, UNSUPPORTED_DECLARED, f"lifecycle:{spec.lifecycle}", witness[1], int(count)
                    )
                )
            else:
                out.append(
                    CoverageConstruct(
                        "origin", origin, UNCOVERED, f"lifecycle:{spec.lifecycle}", "no matrix witness", int(count)
                    )
                )

        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(raw_sessions)")}
        if "detected_provider" in columns:
            for origin, provider, count in conn.execute(
                """
                SELECT origin, detected_provider, COUNT(*) FROM raw_sessions
                WHERE detected_provider IS NOT NULL GROUP BY origin, detected_provider
                """
            ):
                key = f"{origin}/{provider}"
                declared_origin = executable_wires.get(str(provider))
                wire = Provider.from_string(str(provider))
                mapped = origin_from_provider(wire).value if wire is not Provider.UNKNOWN else None
                if declared_origin is None or mapped != str(origin):
                    out.append(
                        CoverageConstruct(
                            "detector-route", key, UNCOVERED, "no executable provider wire", "none", int(count)
                        )
                    )
                else:
                    spec = by_origin[declared_origin]
                    out.append(
                        CoverageConstruct(
                            "detector-route",
                            key,
                            COVERED,
                            ";".join(binding.predicate_path for binding in spec.detector_bindings)
                            or ";".join(spec.parser_paths),
                            ";".join(spec.coverage_refs),
                            int(count),
                        )
                    )

        if table_exists(conn, "raw_artifacts"):
            for origin, kind, support, count in conn.execute(
                "SELECT origin, artifact_kind, support_status, COUNT(*) FROM raw_artifacts GROUP BY 1, 2, 3"
            ):
                out.append(_artifact_construct(by_origin, manifest, str(origin), str(kind), str(support), int(count)))
    finally:
        conn.close()
    return tuple(out)


def _artifact_construct(
    by_origin: dict[str, OriginSpec],
    manifest: CapabilityManifest,
    origin: str,
    kind: str,
    support: str,
    count: int,
) -> CoverageConstruct:
    key = f"{origin}/{kind}/{support}"
    spec = by_origin.get(origin)
    known_kind = kind in {member.value for member in ArtifactKind}
    if spec is None or not known_kind or kind == ArtifactKind.UNKNOWN.value:
        return CoverageConstruct("artifact-kind", key, UNCOVERED, "no artifact declaration", "none", count)
    if support == ArtifactSupportStatus.UNSUPPORTED_PARSEABLE.value:
        return CoverageConstruct(
            "artifact-kind", key, UNSUPPORTED_DECLARED, "artifact taxonomy: unsupported_parseable", "taxonomy", count
        )
    for rule in spec.artifact_rules:
        if rule.kind == kind:
            route = rule.parser_path or f"parse_policy:{rule.parse_policy}"
            return CoverageConstruct("artifact-kind", key, COVERED, route, rule.coverage_role, count)
    if kind in _SESSION_BEARING_KINDS and spec.lifecycle == "executable":
        witness = _matrix_witness(manifest, origin)
        if witness is not None and witness[0] == COVERED:
            return CoverageConstruct("artifact-kind", key, COVERED, ";".join(spec.parser_paths), witness[1], count)
    if kind == ArtifactKind.HOOK_EVENT.value:
        return CoverageConstruct("artifact-kind", key, COVERED, "raw hook event capture", "hook_event", count)
    return CoverageConstruct("artifact-kind", key, UNCOVERED, "no artifact rule for origin", "none", count)


def evaluate_population_coverage(
    archive_root: Path | None,
    *,
    specs: Sequence[OriginSpec] = ORIGIN_SPECS,
    manifest: CapabilityManifest | None = None,
) -> PopulationCoverageReport:
    manifest = manifest if manifest is not None else load_manifest()
    constructs = list(declaration_constructs(specs=specs, manifest=manifest))
    source_db = archive_root / "source.db" if archive_root is not None else None
    evaluated = source_db is not None and source_db.is_file()
    if evaluated and source_db is not None:
        constructs.extend(inventory_constructs(source_db, specs=specs, manifest=manifest))
    return PopulationCoverageReport(
        archive_root=str(archive_root) if archive_root is not None else None,
        inventory_evaluated=evaluated,
        constructs=tuple(constructs),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify every origin, detector route, and artifact kind in the source inventory is declared."
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="evaluate the source inventory at this archive root; declarations alone are checked without it",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = evaluate_population_coverage(args.archive_root)
    if args.as_json:
        print(json.dumps(report.to_dict(), sort_keys=True))
    else:
        print(f"Population coverage: {'PASS' if report.ok else 'FAIL'}")
        print(
            f"Inventory: {'evaluated at ' + str(report.archive_root) if report.inventory_evaluated else 'not evaluated (no source.db)'}"
        )
        summary: dict[str, int] = {}
        for construct in report.constructs:
            summary[construct.status] = summary.get(construct.status, 0) + 1
        for status, n in sorted(summary.items()):
            print(f"  {status}: {n}")
        for construct in report.uncovered:
            print(f"  UNCOVERED {construct.family} {construct.key} ({construct.count:,}): {construct.route}")
    return 0 if report.ok else 1


__all__ = [
    "COVERED",
    "UNCOVERED",
    "UNSUPPORTED_DECLARED",
    "CoverageConstruct",
    "PopulationCoverageReport",
    "declaration_constructs",
    "evaluate_population_coverage",
    "inventory_constructs",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
