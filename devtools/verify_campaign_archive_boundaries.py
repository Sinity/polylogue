"""Verify devtools synthetic benchmark/scale campaigns route through ArchiveLocation.

Background
----------

The canonical "phantom benchmark.db" regression (polylogue-ovme,
polylogue-ovme.3): campaign constructors generated a real synthetic
archive under an output directory, then handed a root-shaped or
filename-shaped sentinel path (``archive_dir / "benchmark.db"``) to
benchmark runners instead of the archive's real, ArchiveLocation-resolved
active ``index.db``. Some consumers (``SQLiteBackend``) canonicalize a
non-``index.db`` filename before opening it; others
(``polylogue.storage.sqlite.connection.open_connection``) do not -- so the
same sentinel silently produced two divergent SQLite files, and some
benchmark runners quietly measured the empty phantom instead of the real
generated archive.

This lint is a narrow, devtools-campaign-scoped completeness check that
catches a regression of that exact shape recurring in the campaign entry
points this bead fixed. It is intentionally **not** a repo-wide boundary
audit over storage/diagnostics/daemon/maintenance/transitions -- that
broader sweep is polylogue-ovme.2's migration surface (storage, status,
maintenance, b5l transitions) and building it here, before that migration
lands, would either produce false positives against not-yet-migrated
production code or require editing files ovme.2 owns concurrently. See
``docs/internals.md`` / the ovme epic notes for the full boundary-audit
scope; this lint covers only the devtools campaign slice of it.

What this lint checks
----------------------

1. None of the campaign entry-point modules may contain the literal
   ``"benchmark.db"`` sentinel string anywhere in source (the concrete
   filename of the historical bug). Reintroducing it in these files is
   almost certainly the same regression recurring.
2. None of the campaign entry-point modules may construct a sibling tier
   path via ad hoc ``Path`` arithmetic (``<name> / "index.db"``,
   ``<name> / "source.db"``, etc.) -- tier-path derivation belongs solely
   to :class:`polylogue.storage.archive_identity.ArchiveLocation`'s
   resolver; a campaign module deriving one directly is the same class of
   ambiguous-path bug with a different filename.
3. The known campaign entry points (``generate_archive``,
   ``run_full_campaign``, ``devtools.run_campaign._run``) must each
   reference ``CampaignArchiveLocation`` somewhere in their module -- a
   purely negative "no bad string" scan can be defeated by simply deleting
   the ownership plumbing while leaving no forbidden string behind, so this
   positive check is required too.

Wired into ``devtools verify --lab`` (a lab/architectural policy concern,
not a per-edit gate).
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

# Campaign entry-point modules this lint owns. Intentionally narrow --
# see the module docstring for why this does not cover storage/diagnostics/
# daemon/maintenance/transitions (polylogue-ovme.2's surface).
CAMPAIGN_MODULES: tuple[Path, ...] = (
    ROOT / "devtools" / "large_archive_generator.py",
    ROOT / "devtools" / "benchmark_campaigns.py",
    ROOT / "devtools" / "run_campaign.py",
    ROOT / "devtools" / "synthetic_benchmark_runtime.py",
)

# Modules exempted from the sibling-derivation/sentinel scan because they
# themselves ARE the sanctioned resolver, or are the ownership wrapper that
# legitimately names "index.db" once in its own docstrings/property.
RESOLVER_MODULES: frozenset[str] = frozenset({"archive_identity.py", "campaign_archive_location.py"})

_FORBIDDEN_SENTINEL_PATTERN = re.compile(r'"benchmark\.db"')

# Ad hoc sibling tier-path derivation: `<expr> / "index.db"` (or any other
# known tier filename), constructed directly rather than obtained from
# ArchiveLocation/CampaignArchiveLocation. A bare string search is
# deliberately used (not just AST) so it also catches f-string/format
# variants of the same shape.
_TIER_FILENAMES: tuple[str, ...] = ("source.db", "index.db", "embeddings.db", "user.db", "ops.db")
_SIBLING_DERIVATION_PATTERN = re.compile(
    r'/\s*["\'](' + "|".join(re.escape(name) for name in _TIER_FILENAMES) + r')["\']'
)

# Entry points that must positively reference CampaignArchiveLocation
# (module path, symbol name) -- a purely negative scan can be defeated by
# deleting the ownership plumbing while leaving no forbidden string behind.
_REQUIRED_OWNERSHIP_REFERENCE: tuple[tuple[Path, str], ...] = (
    (ROOT / "devtools" / "large_archive_generator.py", "generate_archive"),
    (ROOT / "devtools" / "benchmark_campaigns.py", "run_full_campaign"),
    (ROOT / "devtools" / "run_campaign.py", "_run"),
)


@dataclass(frozen=True, slots=True)
class SentinelHit:
    path: Path
    lineno: int
    line: str


@dataclass(frozen=True, slots=True)
class SiblingDerivationHit:
    path: Path
    lineno: int
    line: str
    tier_filename: str


@dataclass(frozen=True, slots=True)
class MissingOwnershipReference:
    path: Path
    symbol: str


def _scan_modules(modules: tuple[Path, ...]) -> tuple[list[SentinelHit], list[SiblingDerivationHit]]:
    sentinel_hits: list[SentinelHit] = []
    sibling_hits: list[SiblingDerivationHit] = []
    for path in modules:
        if not path.exists() or path.name in RESOLVER_MODULES:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _FORBIDDEN_SENTINEL_PATTERN.search(line):
                sentinel_hits.append(SentinelHit(path=path, lineno=lineno, line=line.strip()))
            match = _SIBLING_DERIVATION_PATTERN.search(line)
            if match:
                sibling_hits.append(
                    SiblingDerivationHit(path=path, lineno=lineno, line=line.strip(), tier_filename=match.group(1))
                )
    return sentinel_hits, sibling_hits


def _missing_ownership_references(
    entries: tuple[tuple[Path, str], ...] = _REQUIRED_OWNERSHIP_REFERENCE,
) -> list[MissingOwnershipReference]:
    missing: list[MissingOwnershipReference] = []
    for path, symbol in entries:
        if not path.exists():
            missing.append(MissingOwnershipReference(path=path, symbol=symbol))
            continue
        source = path.read_text(encoding="utf-8")
        if f"def {symbol}" not in source:
            missing.append(MissingOwnershipReference(path=path, symbol=symbol))
            continue
        if "CampaignArchiveLocation" not in source:
            missing.append(MissingOwnershipReference(path=path, symbol=symbol))
    return missing


def _format_report(
    *,
    sentinel_hits: list[SentinelHit],
    sibling_hits: list[SiblingDerivationHit],
    missing_refs: list[MissingOwnershipReference],
) -> str:
    lines = [
        f"forbidden benchmark.db sentinel references found: {len(sentinel_hits)}",
        f"ad hoc tier-path sibling derivations found: {len(sibling_hits)}",
        f"campaign entry points missing CampaignArchiveLocation reference: {len(missing_refs)}",
    ]
    if sentinel_hits:
        lines.append("")
        lines.append("Forbidden benchmark.db sentinel references:")
        for sentinel in sentinel_hits:
            lines.append(f"  {sentinel.path.relative_to(ROOT)}:{sentinel.lineno}: {sentinel.line}")
        lines.append("")
        lines.append(
            "Policy violation: campaigns must resolve the real active index via "
            "CampaignArchiveLocation, never invent a 'benchmark.db' sentinel."
        )
    if sibling_hits:
        lines.append("")
        lines.append("Ad hoc tier-path sibling derivations:")
        for sibling in sibling_hits:
            lines.append(f"  {sibling.path.relative_to(ROOT)}:{sibling.lineno}: {sibling.line}")
        lines.append("")
        lines.append(
            "Policy violation: tier-path derivation belongs to ArchiveLocation's resolver alone; "
            "a campaign module deriving a sibling path directly reintroduces ambiguous archive typing."
        )
    if missing_refs:
        lines.append("")
        lines.append("Campaign entry points missing ownership plumbing:")
        for ref in missing_refs:
            lines.append(f"  {ref.path.relative_to(ROOT) if ref.path.exists() else ref.path}: {ref.symbol}")
        lines.append("")
        lines.append(
            "Policy violation: campaign, ownership, generation, and target tier must stay stable "
            "through every reopen -- each entry point must route through CampaignArchiveLocation."
        )
    if not sentinel_hits and not sibling_hits and not missing_refs:
        lines.append("")
        lines.append("Campaign archive-location boundary intact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    sentinel_hits, sibling_hits = _scan_modules(CAMPAIGN_MODULES)
    missing_refs = _missing_ownership_references()

    ok = not sentinel_hits and not sibling_hits and not missing_refs

    if args.json:
        payload = {
            "sentinel_hits": [
                {"path": str(sentinel.path.relative_to(ROOT)), "line": sentinel.lineno, "text": sentinel.line}
                for sentinel in sentinel_hits
            ],
            "sibling_derivation_hits": [
                {
                    "path": str(sibling.path.relative_to(ROOT)),
                    "line": sibling.lineno,
                    "text": sibling.line,
                    "tier_filename": sibling.tier_filename,
                }
                for sibling in sibling_hits
            ],
            "missing_ownership_references": [
                {"path": str(ref.path.relative_to(ROOT) if ref.path.exists() else ref.path), "symbol": ref.symbol}
                for ref in missing_refs
            ],
            "ok": ok,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(sentinel_hits=sentinel_hits, sibling_hits=sibling_hits, missing_refs=missing_refs))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
