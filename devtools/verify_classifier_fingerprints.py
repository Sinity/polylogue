"""Verify classifier/parser decision-boundary drift is declared.

Background
----------

``devtools lab policy schema-versioning`` (``verify_schema_upgrade_lane.py``)
enforces the durable-vs-derived schema-evolution policy boundary, but it is
keyed entirely to ``INDEX_SCHEMA_VERSION``: a version integer. It has no
visibility into ``polylogue/sources/**`` or
``polylogue/archive/artifact_taxonomy/**``, where the actual acceptance
decision for "is this raw payload a session?" is made.

A parser/classifier can change what it accepts for *identical input bytes*
without touching the index schema version at all. PR #3428 (``ab8a92c1a``,
"require positive conversation evidence before session classification")
is the concrete case this lint exists to catch retroactively: it tightened
``looks_like_record_entry`` and ``looks_like_code`` so that a payload the
classifier used to admit is now refused (or vice versa), while
``INDEX_SCHEMA_VERSION`` stayed unchanged and ``lifecycle.py`` gained no new
declaration. ``devtools lab policy schema-versioning`` ran green on that PR --
correctly, per its own narrow contract, and uselessly for this defect. Rows
already indexed under the old classification stayed silently stale with no
signal that a reparse was ever needed (see ``polylogue-gucv``, ``-9ykn``,
``-zqph``).

What this lint checks
----------------------

1. Discover every module-level function under ``polylogue/sources/`` or
   ``polylogue/archive/artifact_taxonomy/`` whose name matches
   ``looks_like*`` or ``classify_artifact*`` -- the naming convention this
   codebase already uses for a payload-shape/acceptance decision (grepped
   directly; see the functions this module's own tests pin).

2. Fingerprint each one: a SHA-256 hash of its AST (leading docstring
   excluded, line/column attributes excluded by construction), so
   reformatting, comment edits, and docstring rewrites do not trip the gate,
   but any change to the function's actual logic does.

3. Compare against the committed manifest
   (``docs/plans/classifier-fingerprints.json``). A function whose current
   fingerprint does not match its manifest entry is *undeclared drift*: the
   classification boundary moved and nothing says whether that was safe.
   Undeclared drift fails the gate. It resolves one of two ways, both
   recorded as manifest metadata:

   * ``semantic_reparse_version`` -- the change ships alongside an
     ``INDEX_SCHEMA_VERSION`` bump whose ``lifecycle.py`` declaration
     includes ``DerivedDeltaClass.SEMANTIC_REPARSE`` for that version. The
     manifest entry names the version; this lint cross-checks it is really
     declared that way.
   * ``acknowledged_safe`` -- an explicit, reviewed statement (a reason plus
     a bead/issue reference) that the change is safe without a reparse, e.g.
     because it only *tightens* acceptance for a shape that was never validly
     admitted in the first place. This is the escape hatch the parent bead's
     acceptance criteria call for when a reparse is judged unnecessary --
     it must still be an explicit, attributable decision, not silence.

4. A function that disappears from source without its manifest entry being
   removed (an orphaned entry), or a newly added classifier function with no
   manifest entry at all, both fail -- the manifest must track exactly the
   live set of classification-relevant functions.

Scope and false-positive discipline
------------------------------------

Only functions matching the ``looks_like*`` / ``classify_artifact*`` naming
convention under the two classification directories are in scope. An
unrelated refactor elsewhere in ``polylogue/sources/`` (a parser's message
mapping, cost accounting, attachment handling, ...) never touches a function
this lint fingerprints, so it never fires. Renaming, reformatting, or
re-documenting a classifier function without changing its AST shape also
does not fire (docstrings are stripped before hashing; ``ast.dump`` already
omits source positions). The known residual false-positive source is a
behaviour-preserving *local* refactor inside a classifier function itself
(e.g. renaming a local variable) -- accepted deliberately, because these
functions are small, rarely touched, and the cost of one extra
``--ack``/``--semantic-reparse`` run is far lower than another silent
phantom-classification incident.

Wired into ``devtools verify`` directly (like the schema-versioning lane) --
static, archive-independent, sub-second.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from devtools import repo_root as _get_root
from polylogue.storage.sqlite.lifecycle import INDEX_DELTA_DECLARATIONS, DerivedDeltaClass

ROOT = _get_root()
CLASSIFICATION_ROOTS: tuple[Path, ...] = (
    ROOT / "polylogue" / "sources",
    ROOT / "polylogue" / "archive" / "artifact_taxonomy",
)
MANIFEST_PATH = ROOT / "docs" / "plans" / "classifier-fingerprints.json"

_NAME_PATTERN = re.compile(r"^(looks_like\w*|classify_artifact\w*)$")
_REF_PATTERN = re.compile(r"^(polylogue-[a-z0-9][a-z0-9.]*|#\d+)$")
_MIN_REASON_LEN = 20


@dataclass(frozen=True, slots=True)
class ClassifierFunction:
    qualname: str
    path: Path
    lineno: int
    fingerprint: str


def _fingerprint_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Hash the AST of a function, excluding its leading docstring."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    replacement_cls = ast.FunctionDef if isinstance(node, ast.FunctionDef) else ast.AsyncFunctionDef
    replacement = replacement_cls(
        name="_",
        args=node.args,
        body=body or [ast.Pass()],
        decorator_list=node.decorator_list,
        returns=None,
    )
    dump = ast.dump(replacement, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def collect_classifier_functions(roots: tuple[Path, ...] = CLASSIFICATION_ROOTS) -> dict[str, ClassifierFunction]:
    """Return every in-scope classification function, keyed by qualname."""
    found: dict[str, ClassifierFunction] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _NAME_PATTERN.match(node.name):
                    rel = path.relative_to(ROOT).as_posix()
                    qualname = f"{rel}:{node.name}"
                    found[qualname] = ClassifierFunction(
                        qualname=qualname,
                        path=path,
                        lineno=node.lineno,
                        fingerprint=_fingerprint_function(node),
                    )
    return found


CoveredByKind = Literal["semantic_reparse_version", "acknowledged_safe"]


@dataclass(frozen=True, slots=True)
class CoveredBy:
    kind: CoveredByKind
    reason: str
    ref: str
    version: int | None = None


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    fingerprint: str
    covered_by: CoveredBy


class ManifestJSON(TypedDict):
    functions: dict[str, dict[str, object]]


def _covered_by_from_dict(data: dict[str, object]) -> CoveredBy:
    kind_raw = data["kind"]
    if kind_raw not in ("semantic_reparse_version", "acknowledged_safe"):
        raise ValueError(f"unknown covered_by kind in manifest: {kind_raw!r}")
    kind: CoveredByKind = kind_raw
    version_raw = data.get("version")
    return CoveredBy(
        kind=kind,
        reason=str(data["reason"]),
        ref=str(data["ref"]),
        version=int(version_raw) if isinstance(version_raw, int) else None,
    )


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    raw: ManifestJSON = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, ManifestEntry] = {}
    for qualname, data in raw.get("functions", {}).items():
        covered_by_raw = data["covered_by"]
        assert isinstance(covered_by_raw, dict)
        entries[qualname] = ManifestEntry(
            fingerprint=str(data["fingerprint"]),
            covered_by=_covered_by_from_dict(covered_by_raw),
        )
    return entries


def save_manifest(entries: dict[str, ManifestEntry], path: Path = MANIFEST_PATH) -> None:
    payload: ManifestJSON = {
        "functions": {
            qualname: {
                "fingerprint": entry.fingerprint,
                "covered_by": {
                    "kind": entry.covered_by.kind,
                    "reason": entry.covered_by.reason,
                    "ref": entry.covered_by.ref,
                    **({"version": entry.covered_by.version} if entry.covered_by.version is not None else {}),
                },
            }
            for qualname, entry in sorted(entries.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class DriftReport:
    missing: tuple[str, ...]
    orphaned: tuple[str, ...]
    drifted: tuple[str, ...]
    invalid_covered_by: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.orphaned and not self.drifted and not self.invalid_covered_by


def _valid_semantic_reparse_version(version: int) -> bool:
    for declaration in INDEX_DELTA_DECLARATIONS:
        if declaration.version == version:
            return DerivedDeltaClass.SEMANTIC_REPARSE in declaration.classes
    return False


def _validate_covered_by(covered_by: CoveredBy) -> str | None:
    if len(covered_by.reason.strip()) < _MIN_REASON_LEN:
        return f"reason too short (must explain why no reparse is required, >= {_MIN_REASON_LEN} chars)"
    if not _REF_PATTERN.match(covered_by.ref):
        return "ref must be a bead id (polylogue-xxxx) or issue number (#N)"
    if covered_by.kind == "semantic_reparse_version":
        if covered_by.version is None:
            return "semantic_reparse_version covered_by requires a version"
        if not _valid_semantic_reparse_version(covered_by.version):
            return (
                f"version {covered_by.version} has no SEMANTIC_REPARSE declaration in "
                "polylogue/storage/sqlite/lifecycle.py INDEX_DELTA_DECLARATIONS"
            )
    elif covered_by.kind == "acknowledged_safe":
        pass
    else:
        return f"unknown covered_by kind: {covered_by.kind!r}"
    return None


def compute_drift_report(
    current: dict[str, ClassifierFunction] | None = None,
    manifest: dict[str, ManifestEntry] | None = None,
) -> DriftReport:
    if current is None:
        current = collect_classifier_functions()
    if manifest is None:
        manifest = load_manifest()

    missing = tuple(sorted(qualname for qualname in current if qualname not in manifest))
    orphaned = tuple(sorted(qualname for qualname in manifest if qualname not in current))
    drifted = tuple(
        sorted(
            qualname
            for qualname in current
            if qualname in manifest and manifest[qualname].fingerprint != current[qualname].fingerprint
        )
    )
    invalid_covered_by: list[tuple[str, str]] = []
    for qualname, entry in sorted(manifest.items()):
        if qualname in orphaned or qualname in drifted:
            # An orphaned/drifted entry's covered_by is stale by construction;
            # report the structural problem once, not a redundant validation.
            continue
        problem = _validate_covered_by(entry.covered_by)
        if problem is not None:
            invalid_covered_by.append((qualname, problem))

    return DriftReport(
        missing=missing,
        orphaned=orphaned,
        drifted=drifted,
        invalid_covered_by=tuple(invalid_covered_by),
    )


def _format_report(report: DriftReport) -> str:
    lines = [
        f"classification functions missing a manifest entry: {len(report.missing)}",
        f"stale manifest entries (function no longer exists): {len(report.orphaned)}",
        f"classification functions with undeclared fingerprint drift: {len(report.drifted)}",
        f"manifest entries with an invalid covered_by declaration: {len(report.invalid_covered_by)}",
    ]
    if report.missing:
        lines.append("")
        lines.append("New classification functions with no manifest entry:")
        for qualname in report.missing:
            lines.append(f"  {qualname}")
        lines.append(
            "  Fix: devtools lab policy classifier-fingerprints --ack <qualname> --reason '...' --ref <bead-or-issue>"
        )
    if report.orphaned:
        lines.append("")
        lines.append("Manifest entries whose function no longer exists (remove them):")
        for qualname in report.orphaned:
            lines.append(f"  {qualname}")
        lines.append(f"  Fix: edit {MANIFEST_PATH.relative_to(ROOT)} and delete the stale entry.")
    if report.drifted:
        lines.append("")
        lines.append(
            "Classification functions whose logic changed without a declared reparse "
            "or acknowledgment (a parser/classifier decision boundary moved for identical "
            "input bytes -- see devtools/verify_classifier_fingerprints.py's module docstring):"
        )
        for qualname in report.drifted:
            lines.append(f"  {qualname}")
        lines.append(
            "  Fix: either bump INDEX_SCHEMA_VERSION with a declared SEMANTIC_REPARSE delta "
            "and run `devtools lab policy classifier-fingerprints --ack <qualname> "
            "--semantic-reparse <N> --reason '...' --ref <bead-or-issue>`, or record an explicit "
            "safety acknowledgment with `devtools lab policy classifier-fingerprints --ack <qualname> "
            "--reason '...' --ref <bead-or-issue>`."
        )
    if report.invalid_covered_by:
        lines.append("")
        lines.append("Manifest entries with an invalid covered_by declaration:")
        for qualname, problem in report.invalid_covered_by:
            lines.append(f"  {qualname}: {problem}")
    if report.ok:
        lines.append("")
        lines.append("Classifier fingerprint policy intact.")
    return "\n".join(lines)


def _cmd_ack(qualname: str, *, reason: str, ref: str, semantic_reparse_version: int | None) -> int:
    current = collect_classifier_functions()
    if qualname not in current:
        print(f"error: {qualname!r} is not a currently discovered classification function", file=sys.stderr)
        return 2
    manifest = load_manifest()
    covered_by = CoveredBy(
        kind="semantic_reparse_version" if semantic_reparse_version is not None else "acknowledged_safe",
        reason=reason,
        ref=ref,
        version=semantic_reparse_version,
    )
    problem = _validate_covered_by(covered_by)
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    manifest[qualname] = ManifestEntry(fingerprint=current[qualname].fingerprint, covered_by=covered_by)
    save_manifest(manifest)
    print(f"recorded {qualname} ({covered_by.kind})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--ack",
        metavar="QUALNAME",
        help="record an acknowledged_safe (or, with --semantic-reparse, semantic_reparse_version) "
        "manifest entry for a classification function at its current fingerprint",
    )
    parser.add_argument("--reason", help="justification text for --ack (required with --ack)")
    parser.add_argument("--ref", help="bead id (polylogue-xxxx) or issue number (#N) for --ack (required with --ack)")
    parser.add_argument(
        "--semantic-reparse",
        dest="semantic_reparse_version",
        type=int,
        metavar="VERSION",
        help="with --ack: record semantic_reparse_version covered_by instead of acknowledged_safe",
    )
    args = parser.parse_args(argv)

    if args.ack:
        if not args.reason or not args.ref:
            parser.error("--ack requires --reason and --ref")
        return _cmd_ack(
            args.ack,
            reason=args.reason,
            ref=args.ref,
            semantic_reparse_version=args.semantic_reparse_version,
        )

    report = compute_drift_report()

    if args.json:
        print(
            json.dumps(
                {
                    "missing": list(report.missing),
                    "orphaned": list(report.orphaned),
                    "drifted": list(report.drifted),
                    "invalid_covered_by": [
                        {"qualname": qualname, "problem": problem} for qualname, problem in report.invalid_covered_by
                    ],
                    "ok": report.ok,
                },
                indent=2,
            )
        )
    else:
        print(_format_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
