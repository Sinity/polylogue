"""Lint: new hashlib/core.hashing call sites must register in the hash-boundary registry.

Background (polylogue-okpn, follow-up to the polylogue-9e5.6 census at
``docs/audits/2026-07-09-hash-boundary-census.md``): that audit hand-classified
every digest producer call site in ``polylogue/`` (42 direct ``hashlib.*``
calls + 23 ``core.hashing`` helper calls at the time) into content-hash/
integrity-drift checks, identifier-generation-only sites, and a couple of
other one-off uses (HMAC signatures, redaction digests, re-hashing an
already-hashed value). The audit's own "Register-or-fail lint" section
deferred building the mechanical gate to this follow-up, flagging three open
design questions:

1. **Registry format** -- a dedicated machine-readable YAML that the lint
   diffs against, not a lint that regexes the prose census markdown table.
   This follows the same convention as
   ``docs/plans/degrade-loudly-allowlist.yaml`` (AST-discovered sites keyed
   by path/function/occurrence, not line number, with a human-written
   rationale) and ``docs/plans/docs-coverage-baseline.yaml`` (a grandfathered
   baseline of pre-existing gaps at gate introduction, tagged as tracked debt
   rather than silently re-legitimized).
2. **Classification granularity** -- pure ID-generation sites (a hash used
   only to derive a stable identifier / uniqueness key, never compared for
   drift) get the lighter ``identifier`` tag rather than being forced through
   the same shape a ``content-hash`` drift-check site needs. A residual
   ``other`` tag covers the handful of sites that are neither (HMAC
   signatures, redaction digests, re-hashing an already-hashed value). A
   ``baseline-unclassified`` tag grandfathers real call sites that exist in
   the codebase today but were not part of the 65-site 2026-07-09 census and
   have not yet been individually re-audited -- this is tracked debt, not a
   rubber stamp; see the registry file's own header.
3. **Cost** -- this is a static AST scan of ``polylogue/`` plus a YAML diff,
   the same cost class as ``verify_degrade_loudly.py`` (~15s warm gate
   budget), so it is wired into ``devtools verify --quick`` alongside it.

Scan scope: every call to ``hashlib.sha256``/``sha1``/``md5``/``blake2b``/
``blake2s``/``sha3_*``/``sha512``/``sha224``/``sha384`` anywhere in
``polylogue/`` *except* their four definitional uses inside
``polylogue/core/hashing.py`` itself (those calls ARE the implementation of
``hash_text``/``hash_payload``/``hash_bytes``/``hash_file`` -- registering
them separately from the helpers they define would double-count the same
producer), plus every call to ``hash_text``/``hash_text_short``/
``hash_payload``/``hash_bytes``/``hash_file`` anywhere else in ``polylogue/``
(the helper-call sites). Tests are excluded (matching the audit's own scope).

The registry is keyed by ``(path, function qualname, call name, occurrence
index within that function+call-name)`` -- not line number -- so it survives
unrelated line-shift churn, exactly like the degrade-loudly allowlist. A new
call site with no matching registry entry fails the lint with an actionable
message naming the exact site and the registry file to edit. A registry entry
matching no current call site also fails (stale-entry check), so the
registry cannot silently accumulate dead rows.
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
TARGET_DIR = "polylogue"
REGISTRY_PATH = ROOT / "docs" / "plans" / "hash-boundary-registry.yaml"

# The four current core.hashing helpers, plus hash_bytes (added after the
# 2026-07-09 census). Calls to these OUTSIDE core/hashing.py are helper-call
# sites; calls to hashlib.* INSIDE core/hashing.py are the helpers' own
# implementation and are excluded from the direct-hashlib scan below.
_HELPER_NAMES = {"hash_text", "hash_text_short", "hash_payload", "hash_bytes", "hash_file"}
_HASHLIB_ALGOS = {
    "sha256",
    "sha1",
    "md5",
    "sha512",
    "sha224",
    "sha384",
    "blake2b",
    "blake2s",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "shake_128",
    "shake_256",
}
_HASHING_DEFINITION_FILE = "polylogue/core/hashing.py"

_ALLOWED_CLASSIFICATIONS = {"content-hash", "identifier", "other", "baseline-unclassified"}


@dataclass(frozen=True, slots=True)
class Site:
    path: str
    function: str
    call: str
    occurrence: int
    lineno: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.path, self.function, self.call, self.occurrence)


class _FunctionScopeVisitor(ast.NodeVisitor):
    """Walk a module tracking enclosing function qualnames and per-function
    occurrence counters for hashlib/core.hashing call sites."""

    def __init__(self, relpath: str, *, is_hashing_definition_file: bool) -> None:
        self.relpath = relpath
        self._is_hashing_definition_file = is_hashing_definition_file
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

    def _record(self, call_name: str, lineno: int) -> None:
        qualname = self._qualname()
        occ_key = f"{qualname}::{call_name}"
        occurrence = self._occurrence.get(occ_key, 0)
        self._occurrence[occ_key] = occurrence + 1
        self.sites.append(
            Site(path=self.relpath, function=qualname, call=call_name, occurrence=occurrence, lineno=lineno)
        )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _HASHLIB_ALGOS:
            value = func.value
            if isinstance(value, ast.Name) and value.id == "hashlib" and not self._is_hashing_definition_file:
                self._record(f"hashlib.{func.attr}", node.lineno)
        elif isinstance(func, ast.Name) and func.id in _HELPER_NAMES and not self._is_hashing_definition_file:
            self._record(func.id, node.lineno)
        self.generic_visit(node)


def _scan_file(path: Path, *, root: Path) -> list[Site]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    relpath = str(path.relative_to(root))
    visitor = _FunctionScopeVisitor(relpath, is_hashing_definition_file=relpath == _HASHING_DEFINITION_FILE)
    visitor.visit(tree)
    return visitor.sites


def _scan_repo(*, root: Path) -> list[Site]:
    sites: list[Site] = []
    base = root / TARGET_DIR
    if not base.exists():
        return sites
    for path in sorted(base.rglob("*.py")):
        if "test" in path.parts:
            continue
        sites.extend(_scan_file(path, root=root))
    return sites


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    path: str
    function: str
    call: str
    occurrence: int
    classification: str
    note: str

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.path, self.function, self.call, self.occurrence)


@dataclass(frozen=True, slots=True)
class RegistryError:
    message: str


def _load_registry(registry_path: Path) -> tuple[list[RegistryEntry], list[RegistryError]]:
    if not registry_path.exists():
        return [], []
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the hash-boundary registry")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    raw_entries = data.get("entries", []) or []
    entries: list[RegistryEntry] = []
    errors: list[RegistryError] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            errors.append(RegistryError(f"registry entry is not a mapping: {raw!r}"))
            continue
        classification = str(raw.get("classification", ""))
        if classification not in _ALLOWED_CLASSIFICATIONS:
            errors.append(
                RegistryError(
                    f"{raw.get('path')}::{raw.get('function')}::{raw.get('call')} has unknown "
                    f"classification {classification!r} (must be one of {sorted(_ALLOWED_CLASSIFICATIONS)})"
                )
            )
            continue
        entries.append(
            RegistryEntry(
                path=str(raw.get("path", "")),
                function=str(raw.get("function", "")),
                call=str(raw.get("call", "")),
                occurrence=int(raw.get("occurrence", 0)),
                classification=classification,
                note=str(raw.get("note", "")),
            )
        )
    return entries, errors


def _format_report(
    *,
    sites: list[Site],
    registry: list[RegistryEntry],
    unregistered: list[Site],
    stale: list[RegistryEntry],
    malformed: list[RegistryError],
    root: Path,
    registry_path: Path,
) -> str:
    rel_registry = registry_path.relative_to(root) if registry_path.is_relative_to(root) else registry_path
    lines = [
        f"hash producer call sites scanned in {TARGET_DIR}/: {len(sites)}",
        f"registered: {len(registry)}",
        f"unregistered (new ungoverned hash sites): {len(unregistered)}",
        f"stale registry entries: {len(stale)}",
        f"malformed registry entries: {len(malformed)}",
    ]
    if unregistered:
        lines.append("")
        lines.append("Hash call sites with no registry entry:")
        for site in sorted(unregistered, key=lambda s: (s.path, s.lineno)):
            lines.append(
                f"  {site.path}:{site.lineno} in {site.function}() call={site.call} "
                f"(occurrence {site.occurrence}) -- add an entry to {rel_registry} with "
                "classification (content-hash | identifier | other) and a one-line note "
                "on producer + consumer (polylogue-okpn)."
            )
    if stale:
        lines.append("")
        lines.append("Registry entries that no longer match a current call site (remove them):")
        for entry in sorted(stale, key=lambda e: (e.path, e.function)):
            lines.append(f"  {entry.path} :: {entry.function}() call={entry.call} [occurrence {entry.occurrence}]")
    if malformed:
        lines.append("")
        lines.append("Malformed registry entries:")
        for error in malformed:
            lines.append(f"  {error.message}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to scan.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry YAML path (defaults to <root>/docs/plans/hash-boundary-registry.yaml).",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    default_registry = root / "docs" / "plans" / "hash-boundary-registry.yaml"
    registry_path: Path = (args.registry or default_registry).resolve()

    sites = _scan_repo(root=root)
    registry, malformed = _load_registry(registry_path)
    registered_keys = {entry.key for entry in registry}
    site_keys = {site.key for site in sites}

    unregistered = [site for site in sites if site.key not in registered_keys]
    stale = [entry for entry in registry if entry.key not in site_keys]

    ok = not unregistered and not stale and not malformed

    if args.json:
        payload = {
            "sites_scanned": len(sites),
            "registered": len(registry),
            "violations": [
                {"path": s.path, "line": s.lineno, "function": s.function, "call": s.call}
                for s in sorted(unregistered, key=lambda s: (s.path, s.lineno))
            ],
            "stale_registry_entries": [
                {"path": e.path, "function": e.function, "call": e.call, "occurrence": e.occurrence}
                for e in sorted(stale, key=lambda e: (e.path, e.function))
            ],
            "malformed_registry_entries": [e.message for e in malformed],
            "ok": ok,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            _format_report(
                sites=sites,
                registry=registry,
                unregistered=unregistered,
                stale=stale,
                malformed=malformed,
                root=root,
                registry_path=registry_path,
            )
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
