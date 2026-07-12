"""Lint: broad except-handlers in derived-read/status/probe code must signal.

Background (polylogue-cpf.4, the "degrade loudly" doctrine under
``polylogue-cpf``): a deep read (2026-07-05) found the daemon/storage/
insights/coordination packages systematically degrade *silently* on
derived-read, fallback, and freshness-probe paths — a caught exception is
swallowed and a plain default (``None``/``0``/``[]``/``False``) is returned,
which is indistinguishable from the query genuinely finding nothing. For a
system-of-record that is a construct-validity hole: a reader cannot tell
"no data" from "the probe/query failed".

This lint scans ``polylogue/daemon``, ``polylogue/storage``,
``polylogue/insights``, and ``polylogue/coordination`` (excluding tests) for
``except`` handlers that catch a broad exception type (``Exception``,
``BaseException``, or any ``*.Error`` such as ``sqlite3.Error``) whose body:

1. Never calls anything with "log" in its name (covers ``logger.warning``,
   ``self._logger.exception``, etc.) — the doctrine's "log loudly" minimum, and
2. Never re-raises.

Narrower excepts (``ValueError``, ``TypeError``, ``JSONDecodeError``, ...) are
not flagged: in this codebase they are overwhelmingly routine defensive value
coercion on a single optionally-malformed field (a documented fallback for
*one field*, not a derived-read health/readiness signal), not the
system-of-record degradation the doctrine targets.

A violation is not automatically a bug: many broad excepts already return a
typed signal instead of logging (``HealthAlert(severity=ERROR, message=f"...:
{exc}")`` in ``daemon/health.py``, ``{"available": False, "error": str(exc)}``
dicts, ``_repair_result(..., success=False, detail=f"...: {exc}")``). This
lint cannot see through a return value to tell whether it structurally
encodes the failure, so those sites are pre-approved in the allowlist at
``docs/plans/degrade-loudly-allowlist.yaml`` with a one-line rationale, the
same shape as ``docs/plans/test-clock-allowlist.yaml``.

New broad excepts must either add a log call (cheapest fix), return a typed
signal *and* add an allowlist entry explaining what the signal is, or
genuinely re-raise. The allowlist is keyed by (path, enclosing function
qualname, sorted exception names, occurrence index within that function) —
not line number — so it survives unrelated line-shift churn elsewhere in the
file; only an edit to the flagged function's except-handler shape invalidates
an entry. Stale allowlist entries (no longer matching any current violation)
are also rejected, so the allowlist can't quietly grow unbounded.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard repo dep
    yaml = None  # type: ignore[assignment]

from devtools import repo_root as _get_root

ROOT = _get_root()
TARGET_DIRS = (
    "polylogue/daemon",
    "polylogue/storage",
    "polylogue/insights",
    "polylogue/coordination",
)
ALLOWLIST_PATH = ROOT / "docs" / "plans" / "degrade-loudly-allowlist.yaml"

_BROAD_NAMES = {"Exception", "BaseException", "Error"}
_LOG_HINT = "log"


@dataclass(frozen=True, slots=True)
class Site:
    path: str
    function: str
    exceptions: tuple[str, ...]
    occurrence: int
    lineno: int

    @property
    def key(self) -> tuple[str, str, tuple[str, ...], int]:
        return (self.path, self.function, self.exceptions, self.occurrence)


def _exception_names(handler: ast.ExceptHandler) -> tuple[str, ...]:
    node = handler.type
    if node is None:
        return ("<bare>",)
    names: list[str] = []
    elts = node.elts if isinstance(node, ast.Tuple) else [node]
    for elt in elts:
        if isinstance(elt, ast.Name):
            names.append(elt.id)
        elif isinstance(elt, ast.Attribute):
            names.append(elt.attr)
    return tuple(sorted(names))


def _body_logs_or_raises(body: list[ast.stmt]) -> bool:
    holder = ast.Module(body=body, type_ignores=[])
    for node in ast.walk(holder):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and _LOG_HINT in func.value.id.lower()
            ):
                return True
            if isinstance(func, ast.Name) and _LOG_HINT in func.id.lower():
                return True
    return False


class _FunctionScopeVisitor(ast.NodeVisitor):
    """Walk a module tracking enclosing function qualnames and per-function
    occurrence counters for broad, unsignalled except-handlers."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self._stack: list[str] = ["<module>"]
        self._occurrence: dict[str, int] = {}
        self.sites: list[Site] = []

    def _qualname(self) -> str:
        return ".".join(self._stack)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        names = _exception_names(node)
        if set(names) & _BROAD_NAMES and not _body_logs_or_raises(node.body):
            qualname = self._qualname()
            occ_key = f"{qualname}::{names}"
            occurrence = self._occurrence.get(occ_key, 0)
            self._occurrence[occ_key] = occurrence + 1
            self.sites.append(
                Site(
                    path=self.relpath,
                    function=qualname,
                    exceptions=names,
                    occurrence=occurrence,
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)


def _scan_file(path: Path, *, root: Path) -> list[Site]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    relpath = str(path.relative_to(root))
    visitor = _FunctionScopeVisitor(relpath)
    visitor.visit(tree)
    return visitor.sites


def _scan_repo(*, root: Path) -> list[Site]:
    sites: list[Site] = []
    for target in TARGET_DIRS:
        base = root / target
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "test" in path.parts:
                continue
            sites.extend(_scan_file(path, root=root))
    return sites


@dataclass(frozen=True, slots=True)
class AllowlistEntry:
    path: str
    function: str
    exceptions: tuple[str, ...]
    occurrence: int
    reason: str

    @property
    def key(self) -> tuple[str, str, tuple[str, ...], int]:
        return (self.path, self.function, self.exceptions, self.occurrence)


def _load_allowlist(allowlist_path: Path) -> list[AllowlistEntry]:
    if not allowlist_path.exists():
        return []
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the degrade-loudly allowlist")
    data = yaml.safe_load(allowlist_path.read_text(encoding="utf-8")) or {}
    entries = data.get("entries", []) or []
    out: list[AllowlistEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        exceptions = entry.get("exceptions", [])
        out.append(
            AllowlistEntry(
                path=str(entry.get("path", "")),
                function=str(entry.get("function", "")),
                exceptions=tuple(sorted(str(e) for e in exceptions)),
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
    root: Path,
    allowlist_path: Path,
) -> str:
    lines = [
        f"broad except-handlers scanned across {len(TARGET_DIRS)} package(s): {len(sites)}",
        f"allowlisted: {len(allowlist)}",
        f"unallowlisted (new silent soft-fails): {len(unallowlisted)}",
        f"stale allowlist entries: {len(stale)}",
    ]
    if unallowlisted:
        lines.append("")
        lines.append("Broad except-handlers with no log call and no re-raise, outside the allowlist:")
        for site in sorted(unallowlisted, key=lambda s: (s.path, s.lineno)):
            lines.append(
                f"  {site.path}:{site.lineno} in {site.function}() except {list(site.exceptions)} "
                "— add a log call (or re-raise), or add an allowlist entry at "
                f"{allowlist_path.relative_to(root) if allowlist_path.is_relative_to(root) else allowlist_path} "
                "explaining the existing signal (polylogue-cpf.4)."
            )
    if stale:
        lines.append("")
        lines.append("Allowlist entries that no longer match a current violation (remove them):")
        for entry in sorted(stale, key=lambda e: (e.path, e.function)):
            lines.append(
                f"  {entry.path} :: {entry.function}() except {list(entry.exceptions)} [occurrence {entry.occurrence}]"
            )
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
        help="Allowlist YAML path (defaults to <root>/docs/plans/degrade-loudly-allowlist.yaml).",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    default_allowlist = root / "docs" / "plans" / "degrade-loudly-allowlist.yaml"
    allowlist_path: Path = (args.allowlist or default_allowlist).resolve()

    sites = _scan_repo(root=root)
    allowlist = _load_allowlist(allowlist_path)
    allowed_keys = {entry.key for entry in allowlist}
    site_keys = {site.key for site in sites}

    unallowlisted = [site for site in sites if site.key not in allowed_keys]
    stale = [entry for entry in allowlist if entry.key not in site_keys]

    ok = not unallowlisted and not stale

    if args.json:
        payload = {
            "sites_scanned": len(sites),
            "allowlisted": len(allowlist),
            "violations": [
                {
                    "path": site.path,
                    "line": site.lineno,
                    "function": site.function,
                    "exceptions": list(site.exceptions),
                }
                for site in sorted(unallowlisted, key=lambda s: (s.path, s.lineno))
            ],
            "stale_allowlist_entries": [
                {
                    "path": entry.path,
                    "function": entry.function,
                    "exceptions": list(entry.exceptions),
                    "occurrence": entry.occurrence,
                }
                for entry in sorted(stale, key=lambda e: (e.path, e.function))
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
                root=root,
                allowlist_path=allowlist_path,
            )
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
