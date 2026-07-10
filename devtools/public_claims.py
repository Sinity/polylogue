"""Validate the public claims ledger and its evidence bindings."""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "public-claims.yaml"
SUPPORTED_SCHEMA = "external-claims-ledger/v2"
_CLAIM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


@dataclass(frozen=True, slots=True)
class ClaimProblem:
    """One bounded validation problem."""

    claim_id: str | None
    message: str

    def to_payload(self) -> dict[str, str | None]:
        return {"claim_id": self.claim_id, "message": self.message}


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_bead_ids(root: Path) -> set[str]:
    path = root / ".beads" / "issues.jsonl"
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Beads JSONL at {path}:{line_number}: {exc}") from exc
            bead_id = payload.get("id")
            if isinstance(bead_id, str):
                ids.add(bead_id)
    return ids


def _as_string_list(value: object, *, field: str, claim_id: str | None, problems: list[ClaimProblem]) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        problems.append(ClaimProblem(claim_id, f"{field} must be a non-empty list of strings"))
        return []
    return [item.strip() for item in value]


def _existing_evidence_paths(root: Path, evidence: list[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for raw in evidence:
        path_text = raw.split("#", maxsplit=1)[0]
        matches = [Path(match) for match in glob.glob(str(root / path_text), recursive=True)]
        if matches and any(path.exists() for path in matches):
            present.append(raw)
        else:
            missing.append(raw)
    return present, missing


def _expanded_public_surfaces(root: Path, patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(str(root / pattern), recursive=True):
            path = Path(match)
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def build_report(*, ledger_path: Path = DEFAULT_LEDGER, root: Path = ROOT) -> dict[str, Any]:
    """Return a machine-readable claims-ledger validation report."""

    problems: list[ClaimProblem] = []
    if not ledger_path.exists():
        return {
            "ok": False,
            "ledger": str(ledger_path),
            "schema": None,
            "claim_count": 0,
            "surface_count": 0,
            "problems": [ClaimProblem(None, "claims ledger does not exist").to_payload()],
        }

    try:
        document = _load_yaml(ledger_path)
    except (OSError, yaml.YAMLError) as exc:
        return {
            "ok": False,
            "ledger": str(ledger_path),
            "schema": None,
            "claim_count": 0,
            "surface_count": 0,
            "problems": [ClaimProblem(None, f"could not load claims ledger: {exc}").to_payload()],
        }

    if not isinstance(document, dict):
        document = {}
        problems.append(ClaimProblem(None, "ledger root must be a mapping"))

    schema = document.get("schema")
    if schema != SUPPORTED_SCHEMA:
        problems.append(ClaimProblem(None, f"schema must be {SUPPORTED_SCHEMA!r}, got {schema!r}"))

    statuses = _as_string_list(document.get("statuses"), field="statuses", claim_id=None, problems=problems)
    evidence_classes = _as_string_list(
        document.get("evidence_classes"),
        field="evidence_classes",
        claim_id=None,
        problems=problems,
    )
    public_surface_patterns = _as_string_list(
        document.get("public_surfaces"),
        field="public_surfaces",
        claim_id=None,
        problems=problems,
    )
    retired_phrases_raw = document.get("retired_phrases", [])
    if not isinstance(retired_phrases_raw, list) or not all(isinstance(item, str) for item in retired_phrases_raw):
        problems.append(ClaimProblem(None, "retired_phrases must be a list of strings"))
        retired_phrases: list[str] = []
    else:
        retired_phrases = [item.strip() for item in retired_phrases_raw if item.strip()]

    claims_raw = document.get("claims")
    if not isinstance(claims_raw, list) or not claims_raw:
        problems.append(ClaimProblem(None, "claims must be a non-empty list"))
        claims: list[dict[str, Any]] = []
    else:
        claims = [item for item in claims_raw if isinstance(item, dict)]
        if len(claims) != len(claims_raw):
            problems.append(ClaimProblem(None, "every claims entry must be a mapping"))

    try:
        bead_ids = _load_bead_ids(root)
    except ValueError as exc:
        bead_ids = set()
        problems.append(ClaimProblem(None, str(exc)))

    seen_ids: set[str] = set()
    status_counts: dict[str, int] = dict.fromkeys(statuses, 0)
    evidence_path_count = 0
    proof_command_count = 0

    for claim in claims:
        claim_id_raw = claim.get("id")
        claim_id = claim_id_raw if isinstance(claim_id_raw, str) else None
        if claim_id is None or not _CLAIM_ID_RE.fullmatch(claim_id):
            problems.append(ClaimProblem(claim_id, "id must match ^[a-z0-9][a-z0-9._-]*$"))
            continue
        if claim_id in seen_ids:
            problems.append(ClaimProblem(claim_id, "duplicate claim id"))
        seen_ids.add(claim_id)

        status = claim.get("status")
        if not isinstance(status, str) or status not in statuses:
            problems.append(ClaimProblem(claim_id, f"unknown status {status!r}"))
        else:
            status_counts[status] = status_counts.get(status, 0) + 1

        evidence_class = claim.get("evidence_class")
        if not isinstance(evidence_class, str) or evidence_class not in evidence_classes:
            problems.append(ClaimProblem(claim_id, f"unknown evidence_class {evidence_class!r}"))

        for required_field in ("publication", "scope", "caveat"):
            value = claim.get(required_field)
            if not isinstance(value, str) or not value.strip():
                problems.append(ClaimProblem(claim_id, f"{required_field} must be a non-empty string"))

        publication = claim.get("publication")
        if isinstance(publication, str) and _NUMBER_RE.search(publication):
            scope = claim.get("scope")
            caveat = claim.get("caveat")
            if not isinstance(scope, str) or len(scope.strip()) < 12:
                problems.append(ClaimProblem(claim_id, "quantitative publication requires a specific scope"))
            if not isinstance(caveat, str) or len(caveat.strip()) < 12:
                problems.append(ClaimProblem(claim_id, "quantitative publication requires a substantive caveat"))

        evidence = _as_string_list(claim.get("evidence"), field="evidence", claim_id=claim_id, problems=problems)
        evidence_path_count += len(evidence)
        _, missing = _existing_evidence_paths(root, evidence)
        for path in missing:
            problems.append(ClaimProblem(claim_id, f"evidence path does not exist: {path}"))

        owner_beads = claim.get("owner_beads", [])
        if not isinstance(owner_beads, list) or not all(isinstance(item, str) and item for item in owner_beads):
            problems.append(ClaimProblem(claim_id, "owner_beads must be a list of Bead ids"))
        elif bead_ids:
            for bead_id in owner_beads:
                if bead_id not in bead_ids:
                    problems.append(ClaimProblem(claim_id, f"owner Bead does not exist: {bead_id}"))

        proof_commands = claim.get("proof_commands", [])
        if not isinstance(proof_commands, list) or not all(
            isinstance(item, str) and item.strip() for item in proof_commands
        ):
            problems.append(ClaimProblem(claim_id, "proof_commands must be a list of non-empty strings"))
        else:
            proof_command_count += len(proof_commands)
            if status == "proven" and not proof_commands:
                problems.append(ClaimProblem(claim_id, "proven claim must name at least one proof command"))

        if status == "proven" and evidence_class in {"architecture_decision", "hypothesis"}:
            problems.append(
                ClaimProblem(claim_id, "proven claim cannot use an architecture-decision or hypothesis evidence class")
            )
        if status == "aspirational" and evidence_class not in {"architecture_decision", "hypothesis"}:
            problems.append(ClaimProblem(claim_id, "aspirational claim must use architecture_decision or hypothesis"))

    public_surfaces = _expanded_public_surfaces(root, public_surface_patterns)
    if not public_surfaces:
        problems.append(ClaimProblem(None, "public_surfaces patterns matched no files"))
    for phrase in retired_phrases:
        needle = phrase.casefold()
        for surface_path in public_surfaces:
            try:
                text = surface_path.read_text(encoding="utf-8")
            except OSError as exc:
                problems.append(ClaimProblem(None, f"could not read public surface {surface_path}: {exc}"))
                continue
            if needle in text.casefold():
                problems.append(
                    ClaimProblem(None, f"retired phrase {phrase!r} appears in {surface_path.relative_to(root)}")
                )

    return {
        "ok": not problems,
        "ledger": str(ledger_path.relative_to(root) if ledger_path.is_relative_to(root) else ledger_path),
        "schema": schema,
        "claim_count": len(claims),
        "status_counts": status_counts,
        "evidence_path_count": evidence_path_count,
        "proof_command_count": proof_command_count,
        "surface_count": len(public_surfaces),
        "public_surfaces": [str(path.relative_to(root)) for path in public_surfaces],
        "problems": [problem.to_payload() for problem in problems],
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"public claims: {'ok' if report['ok'] else 'FAIL'}")
    print(f"ledger: {report['ledger']}")
    print(f"claims: {report['claim_count']}")
    print(f"evidence paths: {report['evidence_path_count']}")
    print(f"proof commands: {report['proof_command_count']}")
    print(f"public surfaces: {report['surface_count']}")
    for problem in report["problems"]:
        prefix = f"{problem['claim_id']}: " if problem["claim_id"] else ""
        print(f"  - {prefix}{problem['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(ledger_path=args.ledger)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
