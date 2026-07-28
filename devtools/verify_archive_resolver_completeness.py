"""Inventory call sites of ``polylogue.paths._roots``'s duplicate archive resolvers.

Background
----------

polylogue-ovme.2.1 (AC4's remaining ask, from the polylogue-ovme.2 migration
epic) originally found four resolvers in ``polylogue/paths/_roots.py`` that
duplicated or bypassed
:class:`polylogue.storage.archive_identity.ArchiveLocation`'s pointer/tier
resolution instead of delegating to it:

* ``active_index_db_path()`` -- re-reads ``.index-active-pointer`` itself
  (43 call sites at the time this lint was added).
* ``resolve_active_index_db_path()`` -- the same duplicate pointer-read logic
  under a different name (7 call sites); included a real divergent-target
  bug found during polylogue-ovme.2 (it read its own module-level
  ``archive_root()`` rather than any caller-side override). **Fully migrated
  and removed by polylogue-l2cd**: every call site now uses
  ``polylogue.storage.archive_identity.resolve_active_index_path`` directly,
  and the function itself is deleted from ``_roots.py``.
* ``sibling_index_db()`` -- the literal sibling-derivation-from-anchor-parent
  anti-pattern (16 call sites). **Fully migrated and removed by
  polylogue-l2cd**: every call site now resolves through
  ``polylogue.storage.archive_identity.ArchiveLocation`` directly (or, where
  the anchor was provably already the resolved active index path, the
  now-dead derivation was simply dropped), and the function itself is
  deleted from ``_roots.py``.
* ``archive_file_set_root_for_paths()`` -- derives an archive root from
  ``db_anchor.parent`` when the anchor names ``index.db`` (27 call sites).

These span nearly every read-path surface (cli, mcp, api, daemon, insights),
well beyond the storage/status/maintenance/transition boundaries
polylogue-ovme.2 named. A full one-pass migration of all 93 call sites was
judged too large and risky for one session (polylogue-ovme.2.1's own
scoping note); polylogue-l2cd tracks migrating them resolver-by-resolver,
smallest call-count first. This lint gives the completeness/visibility half
of AC4 without forcing the whole migration in one pass: it inventories every
current call site against a recorded baseline and fails when a call site
appears somewhere NOT already in that baseline -- so the anti-pattern's
footprint cannot grow silently even before it is retired file-by-file.
Shrinking the baseline (migrating a call site away) is always allowed and
encouraged; growing it is not. Once a resolver's call-site count reaches
zero, delete both the resolver function from ``_roots.py`` and its key from
``BASELINE_CALL_SITES``/``RESOLVER_FUNCTIONS`` (derived from the dict's keys)
here, as done for ``resolve_active_index_db_path`` above.

This is intentionally the same shape as
``devtools/verify_campaign_archive_boundaries.py`` (polylogue-ovme.3): a
narrow, grep-based, non-AST completeness scan wired into ``devtools verify
--lab`` as an architectural policy gate, not a per-edit check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root

ROOT = _get_root()

#: The module that legitimately defines and calls these functions internally;
#: never scanned as a "call site" (it IS the resolver).
DEFINING_MODULE = ROOT / "polylogue" / "paths" / "_roots.py"

#: This lint's own module and its unit test: both name every resolver
#: function followed by "(" in prose/seeded fixtures, which would otherwise
#: self-match as spurious "call sites".
_SELF_MODULES: frozenset[Path] = frozenset(
    {
        ROOT / "devtools" / "verify_archive_resolver_completeness.py",
        ROOT / "tests" / "unit" / "devtools" / "test_verify_archive_resolver_completeness.py",
    }
)

#: Directories scanned for call sites. Broad by design -- these resolvers'
#: whole problem is that they leak across nearly every surface package.
SCAN_ROOTS: tuple[Path, ...] = (ROOT / "polylogue", ROOT / "devtools", ROOT / "tests")

#: Baseline call-site inventory as of polylogue-ovme.2.1, one entry per
#: resolver function, values as repo-relative POSIX paths. A file appearing
#: here is a KNOWN, already-audited call site -- not yet migrated, but not
#: new growth either. Removing a path here (because a call site was migrated
#: to ArchiveLocation) is always safe; the lint only fails on paths calling a
#: resolver that are NOT in this set.
BASELINE_CALL_SITES: dict[str, frozenset[str]] = {
    "active_index_db_path": frozenset(
        {
            "devtools/daemon_workload_probe.py",
            "devtools/pipeline_probe/engine.py",
            "polylogue/cli/commands/status.py",
            "polylogue/cli/shell_completion_values.py",
            "polylogue/coordination/envelope.py",
            "polylogue/daemon/cli.py",
            "polylogue/daemon/embedding_backlog.py",
            "polylogue/daemon/events.py",
            "polylogue/daemon/fts_identity_convergence.py",
            "polylogue/daemon/fts_startup.py",
            "polylogue/daemon/healthz.py",
            "polylogue/daemon/http.py",
            "polylogue/daemon/lifecycle.py",
            "polylogue/daemon/lineage_startup.py",
            "polylogue/daemon/provenance.py",
            "polylogue/daemon/similarity.py",
            "polylogue/daemon/status_snapshot.py",
            "polylogue/insights/correlation_view.py",
            "polylogue/maintenance/preview.py",
            "polylogue/readiness/__init__.py",
            "polylogue/storage/message_type_backfill.py",
            "polylogue/storage/repair.py",
            "polylogue/storage/session_timestamp_backfill.py",
            "tests/unit/cli/test_materialize_incident_evidence_command.py",
            "tests/unit/cli/test_reconcile_work_effects_command.py",
            "tests/unit/core/test_paths.py",
            "tests/unit/daemon/test_provenance_endpoint.py",
            "tests/unit/daemon/test_similarity_endpoint.py",
        }
    ),
    "archive_file_set_root_for_paths": frozenset(
        {
            "polylogue/api/archive.py",
            "polylogue/archive/query/plan.py",
            "polylogue/archive/query/search_hits.py",
            "polylogue/archive/query/spec.py",
            "polylogue/cli/archive_query.py",
            "polylogue/cli/click_app.py",
            "polylogue/cli/commands/maintenance/_archive_read.py",
            "polylogue/cli/commands/maintenance/_embeddings.py",
            "polylogue/cli/commands/maintenance/_embeddings_rescue.py",
            "polylogue/cli/commands/status.py",
            "polylogue/cli/read_views/chronicle.py",
            "polylogue/cli/read_views/standard.py",
            "polylogue/cli/select.py",
            "polylogue/cli/verb_cardinality.py",
            "polylogue/maintenance/planner.py",
            "polylogue/mcp/archive_support.py",
            "polylogue/storage/raw_reconciler.py",
            "polylogue/storage/repair.py",
            "tests/unit/core/test_config.py",
        }
    ),
}

RESOLVER_FUNCTIONS: tuple[str, ...] = tuple(BASELINE_CALL_SITES)


def _call_pattern(name: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(name) + r"\(")


@dataclass(frozen=True, slots=True)
class ResolverCallSite:
    function: str
    path: Path
    lineno: int
    line: str


@dataclass(frozen=True, slots=True)
class UnbaselinedCallSite:
    function: str
    path: Path


def _iter_python_files(roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def find_call_sites(
    *,
    roots: tuple[Path, ...] = SCAN_ROOTS,
    functions: tuple[str, ...] = RESOLVER_FUNCTIONS,
    defining_module: Path = DEFINING_MODULE,
) -> list[ResolverCallSite]:
    """Scan ``roots`` for call sites of ``functions``, excluding ``defining_module``."""
    patterns = {name: _call_pattern(name) for name in functions}
    hits: list[ResolverCallSite] = []
    for path in _iter_python_files(roots):
        resolved = path.resolve()
        if resolved == defining_module.resolve() or resolved in {module.resolve() for module in _SELF_MODULES}:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    hits.append(ResolverCallSite(function=name, path=path, lineno=lineno, line=line.strip()))
    return hits


def unbaselined_call_sites(
    hits: list[ResolverCallSite],
    *,
    baseline: dict[str, frozenset[str]] = BASELINE_CALL_SITES,
    repo_root: Path = ROOT,
) -> list[UnbaselinedCallSite]:
    """Return call sites whose (function, file) pair is not already in the baseline."""
    seen: set[tuple[str, str]] = set()
    unbaselined: list[UnbaselinedCallSite] = []
    for hit in hits:
        try:
            relative = hit.path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative = hit.path.as_posix()
        key = (hit.function, relative)
        if relative in baseline.get(hit.function, frozenset()) or key in seen:
            continue
        seen.add(key)
        unbaselined.append(UnbaselinedCallSite(function=hit.function, path=hit.path))
    return unbaselined


def _format_report(*, hits: list[ResolverCallSite], unbaselined: list[UnbaselinedCallSite]) -> str:
    per_function: dict[str, int] = dict.fromkeys(RESOLVER_FUNCTIONS, 0)
    for hit in hits:
        per_function[hit.function] += 1
    lines = ["archive-resolver call-site inventory (polylogue-ovme.2.1 AC4):"]
    for name in RESOLVER_FUNCTIONS:
        lines.append(f"  {name}: {per_function[name]} call site(s)")
    lines.append("")
    lines.append(f"unbaselined (new, unaudited) call sites: {len(unbaselined)}")
    if unbaselined:
        lines.append("")
        for entry in unbaselined:
            lines.append(f"  {entry.path.relative_to(ROOT)}: {entry.function}")
        lines.append("")
        lines.append(
            "Policy violation: a new call site of a duplicate archive-path resolver "
            "(active_index_db_path/archive_file_set_root_for_paths) was introduced outside the recorded "
            "baseline. Route the new read through ArchiveLocation instead, or -- if "
            "the existing resolver is genuinely still the right tool -- add the file "
            "to BASELINE_CALL_SITES in devtools/verify_archive_resolver_completeness.py "
            "with a rationale in the accompanying PR."
        )
    else:
        lines.append("")
        lines.append("No growth beyond the recorded baseline.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    hits = find_call_sites()
    unbaselined = unbaselined_call_sites(hits)
    ok = not unbaselined

    if args.json:
        per_function: dict[str, int] = dict.fromkeys(RESOLVER_FUNCTIONS, 0)
        for hit in hits:
            per_function[hit.function] += 1
        payload = {
            "call_site_counts": per_function,
            "unbaselined_call_sites": [
                {"path": str(entry.path.relative_to(ROOT)), "function": entry.function} for entry in unbaselined
            ],
            "ok": ok,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(hits=hits, unbaselined=unbaselined))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
