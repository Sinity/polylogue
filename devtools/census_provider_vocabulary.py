"""Census: provider/origin vocabulary leak surface (polylogue-9e5.8 Step 0).

Background: the provider->origin retirement program (``polylogue-9e5.8``) was
rejected twice by paired adversarial falsification (2026-07-10) because its
manual ``rg "Provider\\."`` census only counted ``Provider.<MEMBER>`` enum
literal usage (Axis 1) and missed that ``provider``/``providers`` as a bare
*identifier* -- a parameter name, a dataclass/Pydantic/TypedDict field, a
dict/set string-literal key, a CLI flag, an HTTP route -- is an independent,
much larger leak axis (Axis 2/3) concentrated in one connected call graph
running from ``protocols.py`` down through ``api/*.py``,
``storage/repository/**``, and ``storage/sqlite/**``. A lexical grep for the
enum token stays green while missing all of that.

This is a scripted, AST-level replacement for that manual census (the bead's
own Step 0): it enumerates, per run:

- ``param``   -- function/method parameters literally named ``provider``/``providers``.
- ``field``   -- dataclass/Pydantic/TypedDict class-body fields, same names.
- ``key``     -- dict or set/frozenset string-literal elements ``"provider"``/``"providers"``
                 (covers filter-key sets like ``daemon/http.py``'s ``_SCOPE_FILTER_KEYS``).
- ``literal`` -- CLI-option-shaped (``--...provider...``) or HTTP-route-shaped
                 (``/...provider...``) string literals.

Explicitly deprecated Click option declarations are compatibility aliases,
not active public-contract vocabulary. Their option spelling is therefore
excluded from the literal category while the alias remains implemented and
visible to users as deprecated.

Enum-literal usage (Axis 1, ``Provider.<MEMBER>``) is intentionally out of
scope here -- the 2026-07-09 census already tiered it correctly (see the
Appendix in polylogue-9e5.8's design field) and it is not the axis the
falsification rounds found missing.

Scanning excludes ``polylogue/sources``, ``polylogue/schemas``,
``polylogue/pipeline`` (Tier-A, wire-boundary-legitimate by design -- see
CLAUDE.md's Provider/Origin/Source vocabulary section) and
``polylogue/browser_capture`` (wire-boundary-adjacent for the same reason:
it identifies which website a payload came from or an outbound reply posts
to, per polylogue-9e5.8's Axis-2 exclusion #4), plus all test files.

A handful of remaining *mixed-purpose* files carry one genuinely
false-positive site alongside real leak sites (billing-vendor terminology,
not archive-origin identity) -- those are pre-approved in the allowlist at
``docs/plans/provider-vocabulary-exclusions.yaml`` with a one-line rationale,
the same shape ``devtools/verify_degrade_loudly.py`` uses for its allowlist.

This tool is deliberately a *census*, not a hard gate on the (currently
large, expected) backlog of real findings -- Step 3 of polylogue-9e5.8's plan
is the actual coordinated flip that retires them. The only thing ``--check``
enforces is allowlist hygiene: no stale entries (an allowlist entry that no
longer matches any current site), and a non-zero total site count (a census
that ever reports zero everywhere is far more likely a broken scanner than a
fully retired vocabulary -- see the bead's own "not vacuous" acceptance
criterion). Re-running this after each later step in the plan should show
the unallowlisted count trending toward zero, which is the durable,
independently-verifiable substrate the falsification rounds asked for.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard repo dep
    yaml = None  # type: ignore[assignment]

from devtools import repo_root as _get_root

ROOT = _get_root()

# Tier-A / wire-boundary-legitimate top-level packages (polylogue-9e5.8
# Appendix A + Axis-2 exclusion #4): sources/schemas/pipeline speak
# provider-wire vocabulary by design at the parse boundary; browser_capture
# identifies which website a payload came from / an outbound reply posts to,
# "Tier-A-like for the same reason sources/parsers are". Never flip, so never
# flagged.
EXCLUDED_TOP_PACKAGES: tuple[str, ...] = ("sources", "schemas", "pipeline", "browser_capture")

ALLOWLIST_PATH = ROOT / "docs" / "plans" / "provider-vocabulary-exclusions.yaml"

_TOKENS = frozenset({"provider", "providers"})
_CLI_FLAG_RE = re.compile(r"^--[\w-]*provider[\w-]*$", re.IGNORECASE)
_ROUTE_RE = re.compile(r"^/[\w/{}\-]*provider[\w/{}\-]*$", re.IGNORECASE)

_CATEGORY_LABELS: dict[str, str] = {
    "param": 'function/method parameter literally named "provider"/"providers"',
    "field": 'dataclass/Pydantic/TypedDict field literally named "provider"/"providers"',
    "key": 'dict or set/frozenset string-literal element "provider"/"providers"',
    "literal": 'CLI-option or HTTP-route string literal containing "provider"',
}


@dataclass(frozen=True, slots=True)
class Site:
    path: str
    lineno: int
    category: str
    qualname: str
    identifier: str
    occurrence: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.path, self.category, self.identifier, self.occurrence)


class _Visitor(ast.NodeVisitor):
    """Walk a module tracking enclosing scope and per-(category, identifier)
    occurrence counters for provider/providers identifier and literal sites."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self._stack: list[str] = ["<module>"]
        self._occurrence: dict[tuple[str, str], int] = {}
        self.sites: list[Site] = []

    def _qualname(self) -> str:
        return ".".join(self._stack)

    def _record(self, lineno: int, category: str, identifier: str) -> None:
        occ_key = (category, identifier)
        occurrence = self._occurrence.get(occ_key, 0)
        self._occurrence[occ_key] = occurrence + 1
        self.sites.append(
            Site(
                path=self.relpath,
                lineno=lineno,
                category=category,
                qualname=self._qualname(),
                identifier=identifier,
                occurrence=occurrence,
            )
        )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = node.args
        all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        for arg in all_args:
            if arg.arg in _TOKENS:
                self._record(node.lineno, "param", arg.arg)
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id in _TOKENS:
                self._record(stmt.lineno, "field", stmt.target.id)
        self.generic_visit(node)
        self._stack.pop()

    def visit_Dict(self, node: ast.Dict) -> None:
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value in _TOKENS:
                self._record(node.lineno, "key", k.value)
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and elt.value in _TOKENS:
                self._record(node.lineno, "key", elt.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_deprecated_click_option(node):
            # A one-release Click compatibility alias still needs to be
            # accepted, but it is not an active public-contract token. Visit
            # its keyword values in case they contain independently relevant
            # expressions, while intentionally omitting its positional flag
            # spellings from the literal census.
            self.visit(node.func)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            value = node.value
            if _CLI_FLAG_RE.match(value) or _ROUTE_RE.match(value):
                self._record(node.lineno, "literal", value)
        self.generic_visit(node)


def _is_deprecated_click_option(node: ast.Call) -> bool:
    """Whether ``node`` declares an explicitly deprecated ``click.option``."""

    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "option"
        and isinstance(func.value, ast.Name)
        and func.value.id == "click"
    ):
        return False
    for keyword in node.keywords:
        if keyword.arg != "deprecated":
            continue
        if isinstance(keyword.value, ast.Constant):
            return bool(keyword.value.value)
        return True
    return False


def _scan_file(path: Path, *, root: Path) -> list[Site]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    relpath = str(path.relative_to(root))
    visitor = _Visitor(relpath)
    visitor.visit(tree)
    return visitor.sites


def _scan_repo(*, root: Path) -> list[Site]:
    base = root / "polylogue"
    if not base.exists():
        return []
    sites: list[Site] = []
    for path in sorted(base.rglob("*.py")):
        relative_parts = path.relative_to(base).parts
        if relative_parts and relative_parts[0] in EXCLUDED_TOP_PACKAGES:
            continue
        if "test" in path.parts:
            continue
        sites.extend(_scan_file(path, root=root))
    return sites


@dataclass(frozen=True, slots=True)
class AllowlistEntry:
    path: str
    category: str
    identifier: str
    occurrence: int
    reason: str

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.path, self.category, self.identifier, self.occurrence)


def _load_allowlist(allowlist_path: Path) -> list[AllowlistEntry]:
    if not allowlist_path.exists():
        return []
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the provider-vocabulary census allowlist")
    data = yaml.safe_load(allowlist_path.read_text(encoding="utf-8")) or {}
    entries = data.get("entries", []) or []
    out: list[AllowlistEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.append(
            AllowlistEntry(
                path=str(entry.get("path", "")),
                category=str(entry.get("category", "")),
                identifier=str(entry.get("identifier", "")),
                occurrence=int(entry.get("occurrence", 0)),
                reason=str(entry.get("reason", "")),
            )
        )
    return out


def _format_report(
    *,
    sites: list[Site],
    allowlist: list[AllowlistEntry],
    unallowlisted: list[Site],
    stale: list[AllowlistEntry],
) -> str:
    lines = [f"provider/origin vocabulary sites scanned: {len(sites)}"]
    for category, label in _CATEGORY_LABELS.items():
        count = sum(1 for s in sites if s.category == category)
        lines.append(f"  {category:<8} ({label}): {count}")
    lines.append(f"allowlisted: {len(allowlist)}")
    lines.append(f"unallowlisted (retirement candidates + not-yet-triaged findings): {len(unallowlisted)}")
    lines.append(f"stale allowlist entries: {len(stale)}")
    if unallowlisted:
        lines.append("")
        lines.append("Sites (path:line category identifier [in qualname]):")
        for site in sorted(unallowlisted, key=lambda s: (s.category, s.path, s.lineno)):
            lines.append(f"  {site.path}:{site.lineno} {site.category} {site.identifier!r} in {site.qualname}()")
    if stale:
        lines.append("")
        lines.append("Allowlist entries that no longer match a current site (remove them):")
        for entry in sorted(stale, key=lambda e: (e.path, e.category, e.identifier)):
            lines.append(f"  {entry.path} :: {entry.category} {entry.identifier!r} [occurrence {entry.occurrence}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to scan.")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Allowlist YAML path (defaults to <root>/docs/plans/provider-vocabulary-exclusions.yaml).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Exit non-zero on allowlist hygiene failures only (stale entries, or a "
            "vacuous zero-site scan) -- does NOT fail on the large, expected backlog "
            "of unallowlisted retirement candidates. Use for CI; omit for a plain census."
        ),
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    default_allowlist = root / "docs" / "plans" / "provider-vocabulary-exclusions.yaml"
    allowlist_path: Path = (args.allowlist or default_allowlist).resolve()

    sites = _scan_repo(root=root)
    allowlist = _load_allowlist(allowlist_path)
    allowed_keys = {entry.key for entry in allowlist}
    site_keys = {site.key for site in sites}

    unallowlisted = [site for site in sites if site.key not in allowed_keys]
    stale = [entry for entry in allowlist if entry.key not in site_keys]

    ok = True
    if args.check:
        ok = not stale and len(sites) > 0

    if args.json:
        payload = {
            "sites_scanned": len(sites),
            "counts_by_category": {
                category: sum(1 for s in sites if s.category == category) for category in _CATEGORY_LABELS
            },
            "allowlisted": len(allowlist),
            "unallowlisted": [
                {
                    "path": site.path,
                    "line": site.lineno,
                    "category": site.category,
                    "identifier": site.identifier,
                    "qualname": site.qualname,
                    "occurrence": site.occurrence,
                }
                for site in sorted(unallowlisted, key=lambda s: (s.category, s.path, s.lineno))
            ],
            "stale_allowlist_entries": [
                {
                    "path": entry.path,
                    "category": entry.category,
                    "identifier": entry.identifier,
                    "occurrence": entry.occurrence,
                }
                for entry in sorted(stale, key=lambda e: (e.path, e.category, e.identifier))
            ],
            "ok": ok,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            _format_report(
                sites=sites,
                allowlist=allowlist,
                unallowlisted=unallowlisted,
                stale=stale,
            )
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
