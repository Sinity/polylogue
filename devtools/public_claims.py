"""Verify the generated public-claims projection and its coverage markers."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from devtools.command_catalog import control_plane_command
from polylogue.insights.measurement.public_claims import (
    CapabilityClaimInput,
    EvidenceIntegrityStatus,
    EvidenceIntegrityVerdict,
    MappingEvidenceIntegrityProvider,
    PublicClaimPresetName,
    PublicClaimProjection,
    build_public_claims_payload,
    project_public_claims,
    render_public_claims_json,
    render_public_claims_markdown,
)
from polylogue.scenarios.corpus import claim_vs_evidence_findings
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_findings_as_assertions
from polylogue.storage.sqlite.finding_provenance import list_public_finding_inputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "generated" / "public-claims"
DEFAULT_COMPATIBILITY_PATH = ROOT / "docs" / "public-claims.yaml"
COMPATIBILITY_SCHEMA = "polylogue.public-claims-compatibility-view.v1"
VERDICT_EXPORT_SCHEMA = "polylogue.evidence-integrity-verdicts.v1"
PUBLIC_SURFACES = (
    Path("README.md"),
    Path("docs/demos.md"),
    Path("docs/findings/claim-vs-evidence.md"),
)
RETIRED_PHRASES = ("your AI memory",)
_MARKER_RE = re.compile(r"<!--\s*public-claim:([a-z0-9][a-z0-9._-]*)\s*-->")
_FORBIDDEN_PUBLIC_PATH_RE = re.compile(r"(?:^|[\s`'\"])(?:/home/|/realm/|[A-Za-z]:\\)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class ClaimProblem:
    """One bounded verification problem."""

    claim_id: str | None
    message: str

    def to_payload(self) -> dict[str, str | None]:
        return {"claim_id": self.claim_id, "message": self.message}


REPOSITORY_CAPABILITY_CLAIMS: tuple[CapabilityClaimInput, ...] = (
    CapabilityClaimInput(
        claim_key="category.local-evidence-system",
        publication="Polylogue is a local evidence system for AI work.",
        scope="The current local archive, query, evidence, judgment, and context surfaces.",
        caveat="This is a product-category capability statement, not a measured performance or prevalence claim.",
        public_evidence_refs=(
            "file:README.md",
            "file:docs/architecture.md",
            "file:docs/proof-artifacts.md",
        ),
        presets=tuple(PublicClaimPresetName),
    ),
)


def load_integrity_verdicts(path: Path | None) -> MappingEvidenceIntegrityProvider:
    """Load a bounded 37t.14 receipt export; absence means uncomputed/unresolved."""

    if path is None:
        return MappingEvidenceIntegrityProvider({})
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != VERDICT_EXPORT_SCHEMA:
        raise ValueError(f"integrity verdict export must use schema {VERDICT_EXPORT_SCHEMA!r}")
    raw_verdicts = document.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise ValueError("integrity verdict export verdicts must be a list")

    verdicts: dict[str, EvidenceIntegrityVerdict] = {}
    for raw in raw_verdicts:
        if not isinstance(raw, dict):
            raise ValueError("every integrity verdict must be a mapping")
        finding_ref = _required_string(raw.get("finding_ref"), field="finding_ref")
        if finding_ref in verdicts:
            raise ValueError(f"duplicate integrity verdict for {finding_ref!r}")
        try:
            status = EvidenceIntegrityStatus(_required_string(raw.get("status"), field="status"))
        except ValueError as exc:
            raise ValueError(f"unknown integrity status for {finding_ref!r}") from exc
        verdicts[finding_ref] = EvidenceIntegrityVerdict(
            finding_ref=finding_ref,
            status=status,
            public_evidence_refs=_string_tuple(raw.get("public_evidence_refs")),
            reason_codes=_string_tuple(raw.get("reason_codes")),
            blind_spot_codes=_string_tuple(raw.get("blind_spot_codes")),
            as_of_epoch=_optional_string(raw.get("as_of_epoch")),
            frame_ref=_optional_string(raw.get("frame_ref")),
            definition_ref=_optional_string(raw.get("definition_ref")),
            public_remediation_refs=_string_tuple(raw.get("public_remediation_refs")),
        )
    return MappingEvidenceIntegrityProvider(verdicts)


def build_repository_projection(
    *,
    archive_root: Path | None = None,
    verdicts_path: Path | None = None,
) -> tuple[PublicClaimProjection, ...]:
    """Build one projection from a live user.db or the deterministic seed population."""

    integrity = load_integrity_verdicts(verdicts_path)
    if archive_root is not None:
        user_db_path = archive_root / "user.db"
        if not user_db_path.exists():
            raise ValueError(f"archive user tier does not exist: {user_db_path}")
        conn = sqlite3.connect(user_db_path)
        conn.row_factory = sqlite3.Row
        try:
            findings = list_public_finding_inputs(conn)
        finally:
            conn.close()
    else:
        with tempfile.TemporaryDirectory(prefix="polylogue-public-claims-") as temp_dir:
            user_db_path = Path(temp_dir) / "user.db"
            initialize_archive_database(user_db_path, ArchiveTier.USER)
            conn = sqlite3.connect(user_db_path)
            conn.row_factory = sqlite3.Row
            try:
                upsert_findings_as_assertions(conn, claim_vs_evidence_findings(), now_ms=1_720_080_953_667)
                conn.commit()
                findings = list_public_finding_inputs(conn)
            finally:
                conn.close()

    return project_public_claims(
        findings,
        REPOSITORY_CAPABILITY_CLAIMS,
        integrity=integrity,
    )


def rendered_artifacts(
    claims: Sequence[PublicClaimProjection],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compatibility_path: Path = DEFAULT_COMPATIBILITY_PATH,
) -> dict[Path, str]:
    """Return every Markdown/JSON preset plus the generated YAML compatibility view."""

    generated_note = (
        f"<!-- Generated by `{control_plane_command('render public-claims')}`. "
        "Edit FINDING assertions/capability declarations or supply a 37t.14 verdict receipt instead. -->\n\n"
    )
    artifacts: dict[Path, str] = {}
    for preset in PublicClaimPresetName:
        artifacts[output_dir / f"{preset.value}.md"] = generated_note + render_public_claims_markdown(claims, preset)
        artifacts[output_dir / f"{preset.value}.json"] = render_public_claims_json(claims, preset)

    compatibility = build_public_claims_payload(claims, PublicClaimPresetName.VERIFIED_EXPORT)
    compatibility_document = {
        "schema": COMPATIBILITY_SCHEMA,
        "generated_by": control_plane_command("render public-claims"),
        "authority": compatibility["authority"],
        "preset": compatibility["preset"],
        "claim_count": compatibility["claim_count"],
        "publishable_claim_keys": compatibility["publishable_claim_keys"],
        "claims": compatibility["claims"],
    }
    artifacts[compatibility_path] = yaml.safe_dump(
        compatibility_document,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )
    return artifacts


def build_report(
    *,
    root: Path = ROOT,
    claims: Sequence[PublicClaimProjection] | None = None,
    output_dir: Path | None = None,
    compatibility_path: Path | None = None,
) -> dict[str, Any]:
    """Return a machine-readable generated-view, parity, and coverage report."""

    problems: list[ClaimProblem] = []
    resolved_claims = tuple(claims) if claims is not None else build_repository_projection()
    resolved_output_dir = output_dir or root / "docs" / "generated" / "public-claims"
    resolved_compatibility_path = compatibility_path or root / "docs" / "public-claims.yaml"
    expected = rendered_artifacts(
        resolved_claims,
        output_dir=resolved_output_dir,
        compatibility_path=resolved_compatibility_path,
    )

    for path, rendered in expected.items():
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != rendered:
            problems.append(
                ClaimProblem(None, f"generated public-claims artifact is out of sync: {_display_path(path, root)}")
            )
        if _FORBIDDEN_PUBLIC_PATH_RE.search(current):
            problems.append(
                ClaimProblem(
                    None,
                    f"checked-in generated artifact contains an absolute private path: {_display_path(path, root)}",
                )
            )
        if _FORBIDDEN_PUBLIC_PATH_RE.search(rendered):
            problems.append(
                ClaimProblem(None, f"generated artifact contains an absolute private path: {_display_path(path, root)}")
            )

    problems.extend(_preset_parity_problems(resolved_claims))
    problems.extend(_coverage_problems(root, resolved_claims))
    problems.extend(_retired_phrase_problems(root))
    problems.extend(_public_ref_problems(resolved_claims))

    status_counts = Counter(claim.status.value for claim in resolved_claims)
    integrity_counts = Counter(
        claim.integrity_status.value for claim in resolved_claims if claim.integrity_status is not None
    )
    return {
        "ok": not problems,
        "schema": COMPATIBILITY_SCHEMA,
        "claim_count": len(resolved_claims),
        "status_counts": dict(sorted(status_counts.items())),
        "integrity_status_counts": dict(sorted(integrity_counts.items())),
        "artifact_count": len(expected),
        "surface_count": len(PUBLIC_SURFACES),
        "public_surfaces": [path.as_posix() for path in PUBLIC_SURFACES],
        "claim_keys": [claim.claim_key for claim in resolved_claims],
        "problems": [problem.to_payload() for problem in problems],
    }


def _preset_parity_problems(claims: Sequence[PublicClaimProjection]) -> list[ClaimProblem]:
    statuses_by_key: dict[str, set[str]] = defaultdict(set)
    for preset in PublicClaimPresetName:
        payload = build_public_claims_payload(claims, preset)
        raw_claims = payload.get("claims")
        if not isinstance(raw_claims, list):
            return [ClaimProblem(None, f"preset {preset.value!r} did not render a claims list")]
        for raw in raw_claims:
            if isinstance(raw, dict) and isinstance(raw.get("claim_key"), str) and isinstance(raw.get("status"), str):
                statuses_by_key[raw["claim_key"]].add(raw["status"])
    return [
        ClaimProblem(claim_key, f"status differs across presets: {sorted(statuses)!r}")
        for claim_key, statuses in sorted(statuses_by_key.items())
        if len(statuses) != 1
    ]


def _coverage_problems(root: Path, claims: Sequence[PublicClaimProjection]) -> list[ClaimProblem]:
    known = {claim.claim_key for claim in claims}
    covered: set[str] = set()
    problems: list[ClaimProblem] = []
    for relative_path in PUBLIC_SURFACES:
        path = root / relative_path
        if not path.exists():
            problems.append(ClaimProblem(None, f"public surface does not exist: {relative_path.as_posix()}"))
            continue
        text = path.read_text(encoding="utf-8")
        for marker in _MARKER_RE.findall(text):
            if marker not in known:
                problems.append(
                    ClaimProblem(marker, f"public surface marker has no projected claim: {relative_path.as_posix()}")
                )
            else:
                covered.add(marker)
    for claim_key in sorted(known - covered):
        problems.append(ClaimProblem(claim_key, "projected claim is not covered by a public-surface marker"))
    return problems


def _retired_phrase_problems(root: Path) -> list[ClaimProblem]:
    problems: list[ClaimProblem] = []
    for relative_path in PUBLIC_SURFACES:
        path = root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").casefold()
        for phrase in RETIRED_PHRASES:
            if phrase.casefold() in text:
                problems.append(ClaimProblem(None, f"retired phrase {phrase!r} appears in {relative_path.as_posix()}"))
    return problems


def _public_ref_problems(claims: Sequence[PublicClaimProjection]) -> list[ClaimProblem]:
    problems: list[ClaimProblem] = []
    for claim in claims:
        if not claim.public_evidence_refs and claim.privacy_review != "held_private":
            problems.append(ClaimProblem(claim.claim_key, "claim publishes no evidence refs"))
        for ref in (*claim.public_evidence_refs, *claim.public_remediation_refs):
            if _public_ref_is_unsafe(ref):
                problems.append(ClaimProblem(claim.claim_key, f"unsafe public ref: {ref}"))
    return problems


def _public_ref_is_unsafe(ref: str) -> bool:
    _kind, separator, object_id = ref.partition(":")
    if not separator or not object_id:
        return True
    path_text = object_id.split("#", maxsplit=1)[0]
    if ref.startswith("file:"):
        return (
            path_text.startswith("~")
            or PurePosixPath(path_text).is_absolute()
            or PureWindowsPath(path_text).is_absolute()
            or ".." in PurePosixPath(path_text).parts
            or ".." in PureWindowsPath(path_text).parts
        )
    return False


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _required_string(value, field="optional verdict field")


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError("verdict list fields must contain strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _print_human(report: Mapping[str, Any]) -> None:
    print(f"public claims: {'ok' if report['ok'] else 'FAIL'}")
    print(f"claims: {report['claim_count']}")
    print(f"generated artifacts: {report['artifact_count']}")
    print(f"public surfaces: {report['surface_count']}")
    for problem in report["problems"]:
        prefix = f"{problem['claim_id']}: " if problem["claim_id"] else ""
        print(f"  - {prefix}{problem['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = build_report()
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
        report = {
            "ok": False,
            "claim_count": 0,
            "artifact_count": 0,
            "surface_count": len(PUBLIC_SURFACES),
            "problems": [ClaimProblem(None, str(exc)).to_payload()],
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
