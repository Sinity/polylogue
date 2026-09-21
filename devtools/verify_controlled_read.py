"""Census every direct ``ArchiveStore.open_existing`` against the read boundary.

Gate classification: **blocking architectural boundary check**.

The rule this gate makes executable
-----------------------------------

Opening the archive directly is one of exactly three things, and the
classification is total:

``read_boundary``
    A read-only open that *installs the read control plane* -- admission,
    a pinned tier snapshot, a cancellation/progress guard, and a receipt.
    There are two such owners and read parity is an equivalence between them
    (commit 6fcfe44d8), not convergence on one route: the in-process read
    executor ``polylogue/archive/query/execution_control.py``, and the
    operation kernel's pinned snapshot ``operation_context.open_operation_read``.
    Which owner is allowed is *policy* (``read_boundary_owners`` in the
    declaration), not something a census row may claim for itself.

``writer``
    An explicit ``read_only=False`` open that takes the writer lease to apply
    a mutation, build a generation, or seed a throwaway archive.  It is not a
    read, so the read boundary does not govern it -- the write-owner rule
    does.  Each one names its writer role and its reason.

``violation``
    Anything else: a read-only or default open outside the two declared
    owners.  Such an open reads the archive with no admission, no snapshot
    pin, no cancellation and no receipt, and the answer it returns cannot say
    which archive revision produced it.

Why a gate and not a list
-------------------------

The count was the problem.  ``polylogue-vclez`` claimed "six production
``ArchiveStore.open_existing()`` reads outside the controlled boundary" and
nothing said *which* of the call sites those were, so the claim could not be
audited and the work it gated could not be given disjoint ownership.  A prose
list drifts the moment someone adds a call site -- the population grew from
about 25 to 26 while that claim sat unaudited.  A declaration the gate checks
cannot: a new open is either declared with a stated reason and reviewed, or
the gate names it by ``file:line``.

The load-bearing check is not "is this open read-only".  It is that a site
*declared* as a writer must actually pass ``read_only=False``: a site that
quietly loses its explicit write mode becomes an uncontrolled read while its
declaration still says it is a writer.

Known limits, stated so nobody mistakes the gate for more than it is: the
observation is AST over ``polylogue/``, matching the literal attribute call
``ArchiveStore.open_existing(...)``.  An open reached through ``getattr``, an
alias bound to another name, or a stored callable is not resolved.  The gate
also does not judge whether a licensed writer open *should* be a daemon-owned
write -- that is the write-owner rule, and the sites carrying that debt record
it in their ``debt`` field.

Usage:
  devtools gate controlled-read
  devtools gate controlled-read --json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from devtools import repo_root as _get_root
from devtools.required_gate import evidence_gate_result

#: The declaration this gate reads.  Policy (which modules may own the read
#: boundary) and census (every observed open, with its reason) live together
#: because both are deliberate edits reviewed as one change.
DECLARATION_PATH = "docs/plans/controlled-read-census.yaml"

#: The package the census covers.
CENSUS_PACKAGE = "polylogue"

#: Every classification a census entry may claim, with what claiming it asserts.
CLASSIFICATION_VOCABULARY: dict[str, str] = {
    "read_boundary": (
        "a read-only open that installs the read control plane (admission, pinned snapshot, "
        "cancellation guard, receipt); only a declared read_boundary_owners entry may claim it"
    ),
    "writer": (
        "an explicit read_only=False open that takes the writer lease; it is not a read, "
        "so the write-owner rule governs it and the gate requires the explicit write mode"
    ),
}

#: Every writer role a ``writer`` entry may claim.  The vocabulary exists so a
#: new writer has to be placed in an existing family or argue for a new one,
#: rather than arriving as a free-text singleton nobody compares.
WRITER_ROLE_VOCABULARY: dict[str, str] = {
    "daemon_write": "inside the daemon's own admitted write path or a declared daemon operation handler",
    "facade_mutation": "a Python-API facade mutation entry point that takes the writer lease in-process",
    "ingest_write": "the acquisition/ingest writer that lands parsed sessions and raw evidence",
    "generation_lifecycle": "index-generation bootstrap, creation, or promotion",
    "durable_probe": "a durable change-train probe against a throwaway temporary archive",
    "fixture_seeding": "demo or scenario seeding of a throwaway archive, not a route over an operator archive",
}

MODULE_SCOPE = "<module>"


@dataclass(frozen=True)
class ObservedSite:
    """Every ``ArchiveStore.open_existing`` call inside one function."""

    site: str
    path: str
    lines: tuple[int, ...]
    #: The literal ``read_only=`` argument of each call, in ``lines`` order.
    #: ``None`` means the keyword was absent (the read-only default) and
    #: ``"<expr>"`` means it was a non-literal the gate will not evaluate.
    read_only: tuple[object, ...]

    def explicit_writer(self) -> bool:
        return bool(self.read_only) and all(value is False for value in self.read_only)

    def any_read_open(self) -> bool:
        return any(value is not False for value in self.read_only)


@dataclass(frozen=True)
class CensusEntry:
    """One declared archive open."""

    site: str
    classification: str
    role: str | None
    reason: str
    debt: str | None


@dataclass(frozen=True)
class Declaration:
    """The checked-in policy plus census."""

    package: str
    owners: tuple[str, ...]
    entries: Mapping[str, CensusEntry] = field(default_factory=dict)


def _qualified(scope: Sequence[str]) -> str:
    return ".".join(scope) if scope else MODULE_SCOPE


class _OpenExistingVisitor(ast.NodeVisitor):
    """Record every ``ArchiveStore.open_existing`` call with its enclosing scope."""

    def __init__(self, relative: str, sink: dict[str, tuple[str, list[int], list[object]]]) -> None:
        self._relative = relative
        self._sink = sink
        self._scope: list[str] = []
        super().__init__()

    def _scoped(self, node: ast.AST) -> None:
        self._scope.append(getattr(node, "name", MODULE_SCOPE))
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "open_existing"
            and isinstance(func.value, ast.Name)
            and func.value.id == "ArchiveStore"
        ):
            read_only: object = None
            for keyword in node.keywords:
                if keyword.arg != "read_only":
                    continue
                read_only = keyword.value.value if isinstance(keyword.value, ast.Constant) else "<expr>"
            key = f"{self._relative}::{_qualified(self._scope)}"
            entry = self._sink.setdefault(key, (self._relative, [], []))
            entry[1].append(node.lineno)
            entry[2].append(read_only)
        self.generic_visit(node)


def observe_sites(repo_root: Path, *, package: str = CENSUS_PACKAGE) -> dict[str, ObservedSite]:
    """Return every observed direct archive open, keyed by ``path::function``."""

    observed: dict[str, tuple[str, list[int], list[object]]] = {}
    for path in sorted((repo_root / package).rglob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _OpenExistingVisitor(relative, observed).visit(tree)
    return {
        key: ObservedSite(key, relative, tuple(lines), tuple(read_only))
        for key, (relative, lines, read_only) in observed.items()
    }


def load_declaration(path: Path) -> Declaration:
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    package = str(document.get("package") or CENSUS_PACKAGE)
    owners = tuple(str(owner) for owner in document.get("read_boundary_owners") or ())
    entries: dict[str, CensusEntry] = {}
    for raw in document.get("sites") or ():
        site = str(raw.get("site") or "")
        entries[site] = CensusEntry(
            site=site,
            classification=str(raw.get("classification") or ""),
            role=str(raw["role"]) if raw.get("role") else None,
            reason=str(raw.get("reason") or "").strip(),
            debt=str(raw["debt"]).strip() if raw.get("debt") else None,
        )
    return Declaration(package=package, owners=owners, entries=entries)


def _owns_boundary(site: str, owners: Sequence[str]) -> bool:
    """Whether ``site`` is covered by a declared read-boundary owner.

    An owner names either a whole module (``path``) or one function inside one
    (``path::function``).  Naming the module is how a module whose *entire*
    job is the read control plane declares itself; naming the function is what
    keeps an accidental sibling open in a mixed module from inheriting the
    exemption.
    """

    path = site.split("::", 1)[0]
    return any(owner in (site, path) for owner in owners)


def collect_violations(
    *,
    repo_root: Path,
    declaration_path: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, ObservedSite], Declaration]:
    declaration = load_declaration(declaration_path or repo_root / DECLARATION_PATH)
    observed = observe_sites(repo_root, package=declaration.package)
    violations: list[dict[str, object]] = []

    for owner in declaration.owners:
        path = owner.split("::", 1)[0]
        if not (repo_root / path).is_file():
            violations.append(
                {"site": owner, "rule": "controlled_read_owner_missing", "detail": "declared owner path does not exist"}
            )

    for site in sorted(declaration.entries):
        if site not in observed:
            violations.append(
                {
                    "site": site,
                    "rule": "controlled_read_census_stale",
                    "detail": "declared but no longer opens the archive; remove the declaration "
                    "rather than leaving the census wider than the code",
                }
            )

    for site, observation in sorted(observed.items()):
        entry = declaration.entries.get(site)
        location = f"{observation.path}:{observation.lines[0]}"
        if entry is None:
            violations.append(
                {
                    "site": site,
                    "file": location,
                    "lines": list(observation.lines),
                    "rule": "controlled_read_site_undeclared",
                    "detail": f"declare it in {DECLARATION_PATH} with a classification and a reason",
                }
            )
            continue
        if entry.classification not in CLASSIFICATION_VOCABULARY:
            violations.append(
                {
                    "site": site,
                    "file": location,
                    "rule": "controlled_read_unknown_classification",
                    "declared": entry.classification,
                    "allowed": sorted(CLASSIFICATION_VOCABULARY),
                }
            )
            continue
        if not entry.reason:
            violations.append({"site": site, "file": location, "rule": "controlled_read_reason_missing"})
        if entry.classification == "read_boundary":
            if not _owns_boundary(site, declaration.owners):
                violations.append(
                    {
                        "site": site,
                        "file": location,
                        "rule": "controlled_read_boundary_not_an_owner",
                        "detail": "only a declared read_boundary_owners entry may claim the read boundary",
                    }
                )
            if observation.explicit_writer():
                violations.append(
                    {
                        "site": site,
                        "file": location,
                        "rule": "controlled_read_boundary_is_a_writer",
                        "detail": "a read-boundary open must not take the writer lease",
                    }
                )
            continue
        # ``writer``: the explicit write mode is the whole licence.  A site
        # that silently loses it becomes an uncontrolled read while its
        # declaration still reads as a writer.
        if entry.role not in WRITER_ROLE_VOCABULARY:
            violations.append(
                {
                    "site": site,
                    "file": location,
                    "rule": "controlled_read_unknown_writer_role",
                    "declared": entry.role,
                    "allowed": sorted(WRITER_ROLE_VOCABULARY),
                }
            )
        if observation.any_read_open():
            violations.append(
                {
                    "site": site,
                    "file": location,
                    "lines": list(observation.lines),
                    "rule": "controlled_read_writer_is_not_explicit",
                    "detail": "declared a writer but opens without read_only=False, so it is an uncontrolled read",
                }
            )

    boundary_sites = [
        site
        for site, entry in declaration.entries.items()
        if entry.classification == "read_boundary" and site in observed
    ]
    if not boundary_sites:
        # Without this the census is trivially clean on a tree where the read
        # boundary has been deleted outright.
        violations.append(
            {
                "site": DECLARATION_PATH,
                "rule": "controlled_read_boundary_absent",
                "detail": "no declared read-boundary owner opens the archive; the controlled reader is gone",
            }
        )
    return violations, observed, declaration


def _format_violation(violation: Mapping[str, object]) -> str:
    location = str(violation.get("file") or "")
    prefix = f"  {location}: " if location else "  "
    detail = violation.get("detail") or violation.get("declared") or ""
    suffix = f" ({detail})" if detail else ""
    return f"{prefix}{violation['site']}: {violation['rule']}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    parser.add_argument("--census", action="store_true", help="Print the full classification, one line per site.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = _get_root()
    declaration_path = repo_root / DECLARATION_PATH
    if not declaration_path.is_file():
        gate = evidence_gate_result(
            gate="controlled-read",
            executable=sys.executable,
            executable_available=True,
            required_count=1,
            inspected_count=0,
            missing_count=1,
            details=(str(declaration_path),),
        )
        if args.json:
            print(json.dumps({"ok": False, "required_gate": gate.to_payload()}, indent=2))
        else:
            print(f"error: {declaration_path} not found", file=sys.stderr)
        return 1

    violations, observed, declaration = collect_violations(repo_root=repo_root)
    classified = {site: declaration.entries[site].classification for site in observed if site in declaration.entries}
    boundary = sorted(site for site, kind in classified.items() if kind == "read_boundary")
    writers = sorted(site for site, kind in classified.items() if kind == "writer")
    debts = sorted(site for site in observed if (entry := declaration.entries.get(site)) is not None and entry.debt)
    gate = evidence_gate_result(
        gate="controlled-read",
        executable=sys.executable,
        executable_available=True,
        required_count=len(observed),
        inspected_count=len(classified),
        missing_count=len(observed) - len(classified),
        semantic_violation_count=len(violations),
        details=tuple(str(item.get("site")) for item in violations[:8]),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "violations": violations,
                    "count": len(violations),
                    "observed": len(observed),
                    "read_boundary": boundary,
                    "writers": writers,
                    "debt": debts,
                    "required_gate": gate.to_payload(),
                },
                indent=2,
            )
        )
        return 1 if violations or not gate.ok else 0

    if args.census:
        for site, observation in sorted(observed.items()):
            entry = declaration.entries.get(site)
            kind = entry.classification if entry is not None else "VIOLATION"
            role = f"/{entry.role}" if entry is not None and entry.role else ""
            for line in observation.lines:
                print(f"{observation.path}:{line} {kind}{role} {site.split('::', 1)[1]}")
    for violation in violations:
        print(_format_violation(violation))
    if not violations:
        print(
            f"controlled-read: {len(observed)} direct archive open(s) -- "
            f"{len(boundary)} read-boundary owner(s), {len(writers)} licensed writer(s), 0 uncontrolled read(s)."
        )
        if debts:
            print(f"  recorded debt on {len(debts)} site(s): {', '.join(debts)}")
    return 1 if violations or not gate.ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
