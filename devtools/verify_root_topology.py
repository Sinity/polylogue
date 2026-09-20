"""Verify no non-kernel module sits at the ``polylogue/`` package root.

The package root is the one directory every surface imports through, so a
module parked there is reachable from everywhere and owned by nobody. It is
also the path of least resistance: a helper that does not obviously belong to
``storage``, ``analysis``, ``sources`` or a surface lands at the root and
stays. ``devtools/verify_topology.py`` used to refuse that, and was deleted
with the per-file placement-judgment machinery it was entangled with
(3d65277de, #3653). Nothing has refused it since (polylogue-8a060).

This check is deliberately the smallest thing that restores the refusal: a
flat inventory of ``polylogue/*.py`` compared against the owner-tagged
allowlist below. It carries no per-file target, reason, or placement
projection — reintroducing those columns would rebuild what #3653 deleted.

The *orphan* class (a module under ``polylogue/`` that no production
entrypoint reaches) is not checked here: the ``consumer-reachability`` gate
already refuses a newly added module with no production consumer, fail-closed
per diff, which is the only way an orphan enters the tree. The two remaining
classes from the deleted checker -- "declared but missing from the tree" and
"declared twice" -- depended on ``topology-target.yaml``, which is also gone;
with no declaration file there is nothing to declare twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "polylogue"

#: Every module allowed to sit at ``polylogue/``, tagged with the concern that
#: owns it. A module belongs here only when every surface needs it before any
#: subpackage is chosen; anything else belongs to the subpackage that owns its
#: concern. This list is the whole declaration: adding an entry is the
#: deliberate act the check exists to force.
ROOT_KERNEL: dict[str, str] = {
    "__init__.py": "package root: the archive-facing public API",
    "__main__.py": "package root: the CLI entry point",
    "_sqlite_compat.py": "runtime: bundled-SQLite capability swap, imported before any tier opens",
    "assets.py": "runtime: packaged asset locations",
    "config.py": "runtime: filesystem/env configuration precedence",
    "daemon_client.py": "runtime: the stdlib UDS client every thin surface talks through",
    "logging.py": "observability: structured logging configuration",
    "logging_fields.py": "observability: the structured-log field allowlist, the PII boundary as data",
    "runtime.py": "runtime: interpreter identity and extension-safety contract",
    "services.py": "runtime: explicit service scope for config/backend/repository access",
    "version.py": "runtime: distribution version resolution",
}


def root_modules(package_root: Path = PACKAGE_ROOT) -> list[str]:
    """Flat sorted inventory of the ``.py`` files at the package root."""
    return sorted(path.name for path in package_root.glob("*.py"))


def strays(package_root: Path = PACKAGE_ROOT) -> list[str]:
    """Root modules with no owner tag in :data:`ROOT_KERNEL`."""
    return [name for name in root_modules(package_root) if name not in ROOT_KERNEL]


def stale_allowlist_entries(package_root: Path = PACKAGE_ROOT) -> list[str]:
    """Allowlist entries whose module no longer exists.

    Without this the allowlist only ever grows, and a name kept after its
    module moved silently re-admits a future file under the same name.
    """
    present = set(root_modules(package_root))
    return sorted(name for name in ROOT_KERNEL if name not in present)


def _format_report(stray: list[str], stale: list[str]) -> str:
    if not stray and not stale:
        return f"root-topology: {len(ROOT_KERNEL)} owner-tagged kernel modules at polylogue/, no strays"
    lines: list[str] = []
    if stray:
        lines += [
            f"root-topology: {len(stray)} module{'' if len(stray) == 1 else 's'} at polylogue/ with no declared owner.",
            "",
            "The package root is imported by every surface, so a module here is",
            "reachable from everywhere and owned by nobody. Move it into the",
            "subpackage that owns its concern, or add it to ROOT_KERNEL in",
            "devtools/verify_root_topology.py with the concern that owns it:",
            "",
        ]
        lines += [f"  polylogue/{name}" for name in stray]
    if stale:
        if lines:
            lines.append("")
        lines += [
            f"root-topology: {len(stale)} ROOT_KERNEL entr{'y names a module' if len(stale) == 1 else 'ies name modules'}"
            " that no longer exists at polylogue/. Remove the entry:",
            "",
        ]
        lines += [f"  {name}" for name in stale]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    stray = strays()
    stale = stale_allowlist_entries()

    if args.json:
        payload = {
            "inventory": root_modules(),
            "strays": stray,
            "stale_allowlist_entries": stale,
            "ok": not stray and not stale,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(stray, stale))

    return 0 if not stray and not stale else 1


if __name__ == "__main__":
    sys.exit(main())
