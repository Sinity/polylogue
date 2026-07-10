"""Validate the externally-presentable release gate definition (#1827)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GATE_DOC = ROOT / "docs" / "plans" / "release-readiness-gate.md"


@dataclass(frozen=True, slots=True)
class GateCommand:
    argv: tuple[str, ...]
    required: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "required": self.required, "reason": self.reason}


REQUIRED_COMMANDS: tuple[GateCommand, ...] = (
    GateCommand(("devtools", "release", "readiness"), True, "release gate definition"),
    GateCommand(("devtools", "verify", "--quick"), True, "static/generated baseline"),
    GateCommand(("devtools", "verify", "public-claims"), True, "public claim/evidence ledger"),
    GateCommand(("devtools", "verify", "--lab"), True, "lab baseline"),
    GateCommand(("devtools", "release", "build-package"), True, "wheel/sdist/Nix package smoke"),
    GateCommand(("devtools", "render", "pages"), True, "documentation site build"),
    GateCommand(("devtools", "verify doc-commands"), True, "README/docs command examples"),
)

FOCUSED_COMMANDS: tuple[GateCommand, ...] = (
    GateCommand(
        ("devtools", "test", "tests/unit/cli/test_query_verbs_runtime.py"),
        False,
        "command/read surface changed",
    ),
    GateCommand(
        ("devtools", "test", "tests/unit/storage/test_blackboard_facade.py"),
        False,
        "user assertion/KV surface changed",
    ),
    GateCommand(
        (
            "devtools",
            "test",
            "tests/unit/cli/test_demo_command.py",
            "tests/unit/demo/test_demo_seed_verify.py",
            "tests/visual",
        ),
        False,
        "demo or reader visual surfaces changed",
    ),
    GateCommand(("nix", "flake", "check"), False, "packaging, Nix, or dependency metadata changed"),
)

REQUIRED_DOC_HEADINGS: tuple[str, ...] = (
    "## Gate Rule",
    "## Required Local Commands",
    "## Automated Gate Matrix",
    "## Manual Release Review",
    "## Current Status",
    "## Release PR Body Requirements",
)

REQUIRED_RELEASE_BODY_HEADINGS: tuple[str, ...] = ("Release gate:", "Verification:")

REQUIRED_RELEASE_BODY_FIELDS: tuple[str, ...] = (
    "- Command floor:",
    "- Machine output:",
    "- README/demo:",
    "- Import/demo fixture:",
    "- Session digest/context:",
    "- Web/API scope:",
    "- Packaging:",
    "- Known caveats scoped out:",
)

STATUS_SATISFIED_HEADING = "Satisfied:"
STATUS_BLOCKING_HEADING = "Still blocking external release claims:"
RETIRED_ISSUE_REFS = ("#1839",)
_STATUS_CAVEAT_RE = re.compile(
    r"\b(do not advertise|scoped out|unless|if the release claims|if claimed|caveat)\b", re.I
)
_ISSUE_REF_RE = re.compile(r"#\d+")


def _status_list(text: str, *, heading: str) -> list[str]:
    """Extract top-level bullet lines under a release-status heading."""

    if heading not in text:
        return []
    after_heading = text.split(heading, 1)[1]
    bullets: list[str] = []
    for line in after_heading.splitlines():
        if line.startswith("## ") or (line in {STATUS_SATISFIED_HEADING, STATUS_BLOCKING_HEADING} and line != heading):
            break
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif bullets and line.startswith("  "):
            bullets[-1] = f"{bullets[-1]} {line.strip()}"
    return bullets


def _issue_refs(lines: list[str]) -> set[str]:
    refs: set[str] = set()
    for line in lines:
        refs.update(_ISSUE_REF_RE.findall(line))
    return refs


def _catalog_command_name(argv: tuple[str, ...], commands: set[str]) -> str:
    """Return the devtools command path before command-local arguments."""

    command_parts: list[str] = []
    for part in argv[1:]:
        if part.startswith("-"):
            break
        command_parts.append(part)
    for end in range(len(command_parts), 0, -1):
        candidate = " ".join(command_parts[:end])
        if candidate in commands:
            return candidate
    return " ".join(command_parts)


def _read_text_file(path: Path, *, label: str, errors: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"could not read {label}: {path}: {exc}")
        return None


def _validate_release_body(text: str, *, errors: list[str]) -> dict[str, Any]:
    missing_headings = [heading for heading in REQUIRED_RELEASE_BODY_HEADINGS if heading not in text]
    missing_fields = [field for field in REQUIRED_RELEASE_BODY_FIELDS if field not in text]
    for heading in missing_headings:
        errors.append(f"release PR body missing heading: {heading}")
    for field in missing_fields:
        errors.append(f"release PR body missing field: {field}")
    return {
        "checked": True,
        "missing_headings": missing_headings,
        "missing_fields": missing_fields,
    }


def build_report(
    *,
    gate_doc: Path = GATE_DOC,
    release_body_file: Path | None = None,
    release_body_text: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable release-gate definition report."""
    from devtools.command_catalog import COMMANDS

    errors: list[str] = []
    text = gate_doc.read_text(encoding="utf-8") if gate_doc.exists() else ""
    satisfied = _status_list(text, heading=STATUS_SATISFIED_HEADING)
    blocking = _status_list(text, heading=STATUS_BLOCKING_HEADING)
    if not text:
        errors.append(f"missing gate document: {gate_doc}")

    for heading in REQUIRED_DOC_HEADINGS:
        if heading not in text:
            errors.append(f"gate document missing heading: {heading}")
    for heading in REQUIRED_RELEASE_BODY_HEADINGS:
        if heading not in text:
            errors.append(f"release PR template missing heading: {heading}")
    for field in REQUIRED_RELEASE_BODY_FIELDS:
        if field not in text:
            errors.append(f"release PR template missing field: {field}")

    for command in (*REQUIRED_COMMANDS, *FOCUSED_COMMANDS):
        command_text = " ".join(command.argv)
        if command.required and command_text not in text:
            errors.append(f"required command missing from gate document: {command_text}")
        if command.argv[0] == "devtools" and _catalog_command_name(command.argv, set(COMMANDS)) not in COMMANDS:
            errors.append(f"unknown devtools command in release gate: {command_text}")

    if STATUS_SATISFIED_HEADING not in text:
        errors.append("gate document missing satisfied release-status list")
    if STATUS_BLOCKING_HEADING not in text:
        errors.append("gate document missing blocking release-status list")
    for retired_ref in RETIRED_ISSUE_REFS:
        if retired_ref in text:
            errors.append(f"release gate references retired issue: {retired_ref}")

    satisfied_refs = _issue_refs(satisfied)
    for line in blocking:
        overlapping_refs = sorted(satisfied_refs.intersection(_ISSUE_REF_RE.findall(line)))
        if overlapping_refs and _STATUS_CAVEAT_RE.search(line) is None:
            errors.append(
                "blocking release-status line also cites satisfied issue(s) "
                f"{', '.join(overlapping_refs)} without scoped-out/caveat wording"
            )

    release_body_report: dict[str, Any] = {"checked": False}
    if release_body_text is not None and release_body_file is not None:
        errors.append("pass either release_body_text or release_body_file, not both")
    elif release_body_text is not None:
        release_body_report = _validate_release_body(release_body_text, errors=errors)
    elif release_body_file is not None:
        body_text = _read_text_file(release_body_file, label="release PR body", errors=errors)
        if body_text is not None:
            release_body_report = _validate_release_body(body_text, errors=errors)
            release_body_report["path"] = str(release_body_file)

    return {
        "ok": not errors,
        "gate_doc": str(gate_doc.relative_to(ROOT) if gate_doc.is_relative_to(ROOT) else gate_doc),
        "required_commands": [command.to_dict() for command in REQUIRED_COMMANDS],
        "focused_commands": [command.to_dict() for command in FOCUSED_COMMANDS],
        "required_release_body_headings": list(REQUIRED_RELEASE_BODY_HEADINGS),
        "required_release_body_fields": list(REQUIRED_RELEASE_BODY_FIELDS),
        "release_body": release_body_report,
        "release_status": {
            "satisfied": satisfied,
            "blocking_external_claims": blocking,
        },
        "errors": errors,
    }


def _print_human(report: dict[str, Any]) -> None:
    status = "ok" if report["ok"] else "FAIL"
    print(f"release readiness: {status}")
    print(f"gate doc: {report['gate_doc']}")
    print("required commands:")
    for command in report["required_commands"]:
        print(f"  {' '.join(command['argv'])}  # {command['reason']}")
    print("focused commands:")
    for command in report["focused_commands"]:
        print(f"  {' '.join(command['argv'])}  # {command['reason']}")
    print("release status:")
    print(f"  satisfied: {len(report['release_status']['satisfied'])}")
    print(f"  blocking external claims: {len(report['release_status']['blocking_external_claims'])}")
    body = report.get("release_body")
    if isinstance(body, dict) and body.get("checked"):
        print("release PR body: checked")
        if body.get("missing_headings"):
            print(f"  missing headings: {len(body['missing_headings'])}")
        if body.get("missing_fields"):
            print(f"  missing fields: {len(body['missing_fields'])}")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the gate-definition report as JSON.")
    parser.add_argument(
        "--release-body-file",
        type=Path,
        help="Validate an actual release PR body against the gate evidence template.",
    )
    args = parser.parse_args(argv)

    report = build_report(release_body_file=args.release_body_file)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
