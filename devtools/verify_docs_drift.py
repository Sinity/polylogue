"""Verify checkable factual claims in the Reference-docs table against current source.

Background
----------

``CLAUDE.md``'s "Reference docs" table points readers at fifteen hand-written
markdown files (architecture, internals, testing, CLI/MCP reference, ...).
Unlike the generated surfaces (``docs/cli-reference.md``'s body, the topology
projection, the devtools catalog), these docs are free text: nothing forces
them to track a renamed table, a moved file, or a bumped schema version. Drift
accumulates silently until an agent or operator hits a stale path/table/flag
mid-task (polylogue-9e5.13).

This lint does not attempt full natural-language fact-checking. It mechanically
checks three narrow, high-confidence claim shapes:

1. **File paths** — every backtick-quoted, extension-bearing path referenced in
   a doc must exist on disk (tried as-is, then under ``polylogue/`` for paths
   given relative to the package root).
2. **Schema versions** — every "<Tier> schema version N" mention must not
   exceed the tier's current ``*_SCHEMA_VERSION`` constant. (A doc whose
   history stops below the current constant is an incomplete-but-not-false
   history; this lint does not flag that — only a claim of a version that does
   not exist yet.)
3. **Watchlist table names** — a short, explicit list of table names known to
   have been renamed or removed (e.g. ``artifact_observations`` -> the DDL now
   defines ``raw_artifacts``) must not appear in doc prose describing the
   *current* schema. New renames get added to ``_RENAMED_TABLE_WATCHLIST`` as
   they're discovered rather than the lint attempting a fully generic
   table/column sweep (which would false-positive heavily on code identifiers,
   config keys, and CLI flags that happen to be backtick-quoted snake_case).

Wired into ``devtools verify --lab`` alongside the other policy checks, since
this is a repo-hygiene boundary check rather than a per-edit gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The doc set from CLAUDE.md's "Reference docs" table (paths relative to repo root).
REFERENCE_DOCS: tuple[str, ...] = (
    "docs/architecture.md",
    "docs/internals.md",
    "docs/architecture-spine.md",
    "CONTRIBUTING.md",
    "TESTING.md",
    "docs/devtools.md",
    "docs/cloud-agents.md",
    "docs/provider-origin-identity.md",
    "docs/search.md",
    "docs/data-model.md",
    "docs/daemon.md",
    "docs/daemon-threat-model.md",
    "docs/cost-model.md",
    "docs/cli-reference.md",
    "docs/mcp-reference.md",
)

# Renamed/removed table names known from source-history archaeology. Each
# watchlist entry fires only when the OLD name appears; it does not require
# the doc to use the new name verbatim (a doc may legitimately say "there is
# no separate X table; the name is a historical alias for Y").
_RENAMED_TABLE_WATCHLIST: dict[str, str] = {
    "artifact_observations": "renamed to `raw_artifacts` (see polylogue/storage/sqlite/archive_tiers/source.py)",
}

# Backtick spans that look like a checkable repo-relative file path: contains
# a "/" and ends with a common source/doc extension. Deliberately excludes
# bare module dotted-names, SQL fragments, and shell snippets.
_PATH_CANDIDATE = re.compile(
    r"`([A-Za-z0-9_./{}\-]+/[A-Za-z0-9_.\-]+\.(?:py|md|sql|json|ya?ml|toml|sh|txt|cfg|ini))"
    r"(?::[A-Za-z_][A-Za-z0-9_]*\(\))?`"
)

# "<Tier> schema version N" claims, tier names matching the *_SCHEMA_VERSION constants.
_SCHEMA_VERSION_CLAIM = re.compile(
    r"\b(Source|Index|Embeddings|User|Ops)\s+schema\s+version\s+(\d+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MissingPath:
    doc: str
    line: int
    quoted: str
    tried: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SchemaVersionOverclaim:
    doc: str
    line: int
    tier: str
    claimed: int
    current: int


@dataclass(frozen=True, slots=True)
class WatchlistHit:
    doc: str
    line: int
    term: str
    note: str


@dataclass(frozen=True, slots=True)
class DriftReport:
    missing_paths: tuple[MissingPath, ...]
    schema_overclaims: tuple[SchemaVersionOverclaim, ...]
    watchlist_hits: tuple[WatchlistHit, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing_paths or self.schema_overclaims or self.watchlist_hits)


def _current_schema_versions() -> dict[str, int]:
    from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
    from polylogue.storage.sqlite.archive_tiers.ops import OPS_SCHEMA_VERSION
    from polylogue.storage.sqlite.archive_tiers.source import SOURCE_SCHEMA_VERSION
    from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION

    return {
        "source": SOURCE_SCHEMA_VERSION,
        "index": INDEX_SCHEMA_VERSION,
        "embeddings": EMBEDDINGS_SCHEMA_VERSION,
        "user": USER_SCHEMA_VERSION,
        "ops": OPS_SCHEMA_VERSION,
    }


def _resolve_path(candidate: str) -> tuple[bool, tuple[str, ...]]:
    """Return (found, tried) for a doc-referenced path candidate.

    Tried as-is relative to the repo root, then with a ``polylogue/`` prefix
    (many internals.md snippets give paths relative to the package root
    without the leading ``polylogue/`` segment).
    """
    # Expand a single brace-group like "{source,user}" into each alternative;
    # docs use this shorthand for the migrations directory family.
    brace_match = re.search(r"\{([a-z_]+(?:,[a-z_]+)+)\}", candidate)
    variants = (
        [candidate.replace(brace_match.group(0), alt) for alt in brace_match.group(1).split(",")]
        if brace_match
        else [candidate]
    )

    tried: list[str] = []
    for variant in variants:
        for prefix in ("", "polylogue/", "tests/"):
            tried_path = f"{prefix}{variant}"
            tried.append(tried_path)
            if (_REPO_ROOT / tried_path).exists():
                return True, tuple(tried)
    return False, tuple(tried)


# Phrases that mark a path as a deliberate historical reference (a doc
# explaining that something used to exist and was retired/removed/superseded),
# not a live claim that the path exists today.
_HISTORICAL_MARKERS = ("retired", "superseded", "no longer exists", "was removed", "has been removed")


def _check_paths(doc: str, text: str) -> list[MissingPath]:
    missing: list[MissingPath] = []
    lines = text.splitlines()
    # Markdown hard-wraps prose across lines, so a "was retired" clause about a
    # path mentioned earlier in the same paragraph can land a few lines below
    # it. Check each paragraph (blank-line-delimited block) as a whole rather
    # than line-by-line for the historical-reference exemption.
    paragraph_start = 0
    for idx, line in enumerate(lines):
        is_blank = line.strip() == ""
        if is_blank or idx == len(lines) - 1:
            end = idx if is_blank else idx + 1
            paragraph = lines[paragraph_start:end]
            paragraph_text = "\n".join(paragraph).lower()
            is_historical = any(marker in paragraph_text for marker in _HISTORICAL_MARKERS)
            if not is_historical:
                for offset, para_line in enumerate(paragraph):
                    lineno = paragraph_start + offset + 1
                    for match in _PATH_CANDIDATE.finditer(para_line):
                        candidate = match.group(1)
                        # Runtime/generated output (gitignored under .cache/, .local/)
                        # only exists after a command has run; not a stale doc claim.
                        if candidate.startswith((".cache/", ".local/")):
                            continue
                        found, tried = _resolve_path(candidate)
                        if not found:
                            missing.append(MissingPath(doc=doc, line=lineno, quoted=candidate, tried=tried))
            paragraph_start = idx + 1
    return missing


def _check_schema_versions(doc: str, text: str, current: dict[str, int]) -> list[SchemaVersionOverclaim]:
    overclaims: list[SchemaVersionOverclaim] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _SCHEMA_VERSION_CLAIM.finditer(line):
            tier = match.group(1).lower()
            claimed = int(match.group(2))
            current_version = current.get(tier)
            if current_version is not None and claimed > current_version:
                overclaims.append(
                    SchemaVersionOverclaim(doc=doc, line=lineno, tier=tier, claimed=claimed, current=current_version)
                )
    return overclaims


def _check_watchlist(doc: str, text: str) -> list[WatchlistHit]:
    hits: list[WatchlistHit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for term, note in _RENAMED_TABLE_WATCHLIST.items():
            if term in line:
                # A line explicitly explaining the alias/rename (contains "renamed"
                # or "historical alias") is documenting the fact, not asserting it.
                if "historical alias" in line or "renamed" in line:
                    continue
                hits.append(WatchlistHit(doc=doc, line=lineno, term=term, note=note))
    return hits


def collect_drift(docs: tuple[str, ...] | None = None) -> DriftReport:
    # Resolved at call time (not bound as a default-argument value) so a test
    # or caller that monkeypatches the module-level REFERENCE_DOCS/_REPO_ROOT
    # observes the change.
    if docs is None:
        docs = REFERENCE_DOCS
    current = _current_schema_versions()
    missing_paths: list[MissingPath] = []
    schema_overclaims: list[SchemaVersionOverclaim] = []
    watchlist_hits: list[WatchlistHit] = []

    for doc in docs:
        doc_path = _REPO_ROOT / doc
        if not doc_path.exists():
            missing_paths.append(MissingPath(doc=doc, line=0, quoted=doc, tried=(doc,)))
            continue
        text = doc_path.read_text(encoding="utf-8")
        missing_paths.extend(_check_paths(doc, text))
        schema_overclaims.extend(_check_schema_versions(doc, text, current))
        watchlist_hits.extend(_check_watchlist(doc, text))

    return DriftReport(
        missing_paths=tuple(missing_paths),
        schema_overclaims=tuple(schema_overclaims),
        watchlist_hits=tuple(watchlist_hits),
    )


def _format_report(report: DriftReport) -> str:
    if report.ok:
        return "docs drift: zero unhandled drift across the reference-docs sweep."

    lines: list[str] = []
    if report.missing_paths:
        lines.append(f"Missing referenced paths: {len(report.missing_paths)}")
        for missing in report.missing_paths:
            lines.append(f"  {missing.doc}:{missing.line}: `{missing.quoted}` (tried: {', '.join(missing.tried)})")
        lines.append("")
    if report.schema_overclaims:
        lines.append(f"Schema version overclaims: {len(report.schema_overclaims)}")
        for overclaim in report.schema_overclaims:
            lines.append(
                f"  {overclaim.doc}:{overclaim.line}: {overclaim.tier} schema version {overclaim.claimed} "
                f"claimed, current constant is {overclaim.current}"
            )
        lines.append("")
    if report.watchlist_hits:
        lines.append(f"Renamed/removed table names still referenced: {len(report.watchlist_hits)}")
        for hit in report.watchlist_hits:
            lines.append(f"  {hit.doc}:{hit.line}: `{hit.term}` -- {hit.note}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = collect_drift()

    if args.json:
        payload = {
            "missing_paths": [
                {"doc": m.doc, "line": m.line, "quoted": m.quoted, "tried": list(m.tried)} for m in report.missing_paths
            ],
            "schema_overclaims": [
                {"doc": s.doc, "line": s.line, "tier": s.tier, "claimed": s.claimed, "current": s.current}
                for s in report.schema_overclaims
            ],
            "watchlist_hits": [
                {"doc": w.doc, "line": w.line, "term": w.term, "note": w.note} for w in report.watchlist_hits
            ],
            "ok": report.ok,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
