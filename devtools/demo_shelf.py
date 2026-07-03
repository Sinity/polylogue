"""Refresh or verify a curated current demo shelf."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(".agent/demos")
DEFAULT_MANIFEST = "MANIFEST.readable.json"
DEFAULT_SUMMARY_INDEX = "SUMMARY_INDEX.json"
DEFAULT_README = "README.md"
DEFAULT_CATALOG = "CURATED_CATALOG.md"
DEMO_SET_CONTRACT = "current-curated-demo-set"
READABLE_SUFFIXES = frozenset({".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml"})
SUMMARY_COVERAGE_FIELDS = frozenset({"claim", "non_claim", "proof_fields", "caveat_fields"})


@dataclass(frozen=True, slots=True)
class ShelfRender:
    manifest_path: Path
    summary_index_path: Path
    readme_path: Path
    catalog_path: Path
    manifest_text: str
    summary_index_text: str
    readme_text: str
    catalog_text: str
    file_count: int
    readable_count: int
    summary_count: int
    summary_coverage: dict[str, list[str]]
    unsummarized_demos: list[str]


def _is_readable_artifact(path: Path) -> bool:
    return path.suffix.lower() in READABLE_SUFFIXES


def _first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^#\s+(.+)$", line)
            if match:
                return match.group(1).strip()
    except OSError:
        return ""
    return ""


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _summary_timestamp(payload: dict[str, Any]) -> object:
    for key in ("updated_at", "generated_at", "created_at"):
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _summary_record(root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    demo = path.parent.relative_to(root).as_posix()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "demo": demo,
            "summary_path": rel,
            "parse_error": str(exc),
            "coverage": {
                "claim": False,
                "non_claim": False,
                "proof_fields": False,
                "caveat_fields": False,
            },
        }
    if not isinstance(payload, dict):
        return {
            "demo": demo,
            "summary_path": rel,
            "parse_error": "summary root is not a JSON object",
            "coverage": {
                "claim": False,
                "non_claim": False,
                "proof_fields": False,
                "caveat_fields": False,
            },
        }

    proof_fields = sorted(key for key in payload if "proof" in key.lower())
    caveat_fields = sorted(key for key in payload if "caveat" in key.lower())
    claim = payload.get("claim")
    non_claim = payload.get("non_claim")
    record: dict[str, Any] = {
        "demo": demo,
        "summary_path": rel,
        "artifact": payload.get("artifact", demo),
        "claim": claim if _nonempty_string(claim) else None,
        "non_claim": non_claim if _nonempty_string(non_claim) else None,
        "proof_fields": proof_fields,
        "caveat_fields": caveat_fields,
        "archive_root": payload.get("archive_root"),
        "index_schema_version": payload.get("index_schema_version", payload.get("index_schema")),
        "timestamp": _summary_timestamp(payload),
        "coverage": {
            "claim": _nonempty_string(claim),
            "non_claim": _nonempty_string(non_claim),
            "proof_fields": bool(proof_fields),
            "caveat_fields": bool(caveat_fields),
        },
    }
    return record


def _summary_records(root: Path, generated: set[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*summary.json")):
        if not path.is_file() or path.resolve() in generated:
            continue
        records.append(_summary_record(root, path))
    return records


def _summary_coverage(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "without_claim": [record["summary_path"] for record in records if not record["coverage"]["claim"]],
        "without_non_claim": [record["summary_path"] for record in records if not record["coverage"]["non_claim"]],
        "without_proof_fields": [
            record["summary_path"] for record in records if not record["coverage"]["proof_fields"]
        ],
        "without_caveat_fields": [
            record["summary_path"] for record in records if not record["coverage"]["caveat_fields"]
        ],
    }


def _schema_version_mismatches(records: list[dict[str, Any]], required: int | None) -> list[dict[str, Any]]:
    if required is None:
        return []
    mismatches: list[dict[str, Any]] = []
    for record in records:
        observed = record.get("index_schema_version")
        if observed is None:
            continue
        try:
            observed_int = int(observed)
        except (TypeError, ValueError):
            observed_int = None
        if observed_int != required:
            mismatches.append(
                {
                    "summary_path": record["summary_path"],
                    "observed": observed,
                    "required": required,
                }
            )
    return mismatches


def _unsummarized_demos(demo_records: list[dict[str, object]], summary_records: list[dict[str, Any]]) -> list[str]:
    summarized_top_level = {str(record["demo"]).split("/", maxsplit=1)[0] for record in summary_records}
    return [str(record["id"]) for record in demo_records if str(record["id"]) not in summarized_top_level]


def _build_summary_index(
    root: Path,
    summary_records: list[dict[str, Any]],
    unsummarized_demos: list[str],
) -> tuple[str, int, dict[str, list[str]]]:
    coverage = _summary_coverage(summary_records)
    coverage = {
        **coverage,
        "unsummarized_demos": unsummarized_demos,
    }
    summary_index = {
        "root": str(root),
        "summary_count": len(summary_records),
        "coverage": coverage,
        "records": summary_records,
    }
    return json.dumps(summary_index, indent=2) + "\n", len(summary_records), coverage


def _demo_records(root: Path, files: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for entry in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        readme = entry / "README.md"
        if not readme.exists():
            readme = entry / "current" / "README.md"
        analysis = entry / "ANALYSIS.md"
        if not analysis.exists():
            analysis = entry / "current" / "ANALYSIS.md"
        title = _first_heading(readme) or _first_heading(analysis) or entry.name.replace("-", " ").title()
        entry_files = [record for record in files if str(record["path"]).startswith(f"{entry.name}/")]
        readable_count = sum(1 for record in entry_files if record["readable"])
        records.append(
            {
                "id": entry.name,
                "title": title,
                "readme": readme.relative_to(root).as_posix() if readme.exists() else None,
                "analysis": analysis.relative_to(root).as_posix() if analysis.exists() else None,
                "file_count": len(entry_files),
                "readable_count": readable_count,
            }
        )
    return records


def _build_readme(records: list[dict[str, object]]) -> str:
    lines = [
        "# Polylogue Current Demo Shelf",
        "",
        "Generated by `devtools workspace demo-shelf`.",
        "",
        "This shelf contains the best current Polylogue demos for the active devloop.",
        "It is not append-only. Replace, consolidate, or move stale demos out when a",
        "better demo supersedes them.",
        "",
        "## Current Entries",
        "",
    ]
    if records:
        for record in records:
            lines.append(f"- `{record['id']}` — {record['title']}")
            if record["readme"]:
                lines.append(f"  - readme: `{record['readme']}`")
            if record["analysis"]:
                lines.append(f"  - analysis: `{record['analysis']}`")
            lines.append(f"  - files: {record['file_count']} ({record['readable_count']} readable)")
    else:
        lines.append("No demo entries found.")
    lines.extend(["", "Retired demo material belongs under `.agent/archive/retired-demos/`, not here.", ""])
    return "\n".join(lines)


def _build_catalog(records: list[dict[str, object]]) -> str:
    lines = [
        "# Polylogue Demo Catalog",
        "",
        "Generated by `devtools workspace demo-shelf`.",
        "",
    ]
    if records:
        for record in records:
            lines.append(f"## {record['title']}")
            lines.append("")
            lines.append(f"- id: `{record['id']}`")
            if record["readme"]:
                lines.append(f"- readme: `{record['readme']}`")
            if record["analysis"]:
                lines.append(f"- analysis: `{record['analysis']}`")
            lines.append(f"- files: {record['file_count']} ({record['readable_count']} readable)")
            lines.append("")
    else:
        lines.append("No demo entries found.")
        lines.append("")
    return "\n".join(lines)


def _parse_required_summary_coverage(value: str) -> set[str]:
    if not value:
        return set()
    fields = {item.strip() for item in value.split(",") if item.strip()}
    unknown = fields - SUMMARY_COVERAGE_FIELDS
    if unknown:
        expected = ", ".join(sorted(SUMMARY_COVERAGE_FIELDS))
        raise argparse.ArgumentTypeError(
            f"unknown coverage field(s): {', '.join(sorted(unknown))}; expected {expected}"
        )
    return fields


def _coverage_failures(summary_coverage: dict[str, list[str]], required: set[str]) -> dict[str, list[str]]:
    return {
        field: summary_coverage.get(f"without_{field}", [])
        for field in sorted(required)
        if summary_coverage.get(f"without_{field}", [])
    }


def render_demo_shelf(
    root: Path,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
    bundle_name: str | None = None,
    summary_index_name: str = DEFAULT_SUMMARY_INDEX,
    readme_name: str = DEFAULT_README,
    catalog_name: str = DEFAULT_CATALOG,
) -> ShelfRender:
    """Build deterministic demo shelf indexes.

    ``bundle_name`` is accepted for CLI compatibility only. Declarative read
    packages own portable bundle generation; this helper deliberately does not
    concatenate readable artifact bodies.
    """

    root = root.expanduser().resolve()
    manifest_path = root / manifest_name
    summary_index_path = root / summary_index_name
    readme_path = root / readme_name
    catalog_path = root / catalog_name
    generated = {manifest_path.resolve(), summary_index_path.resolve(), readme_path.resolve(), catalog_path.resolve()}
    if bundle_name:
        generated.add((root / bundle_name).resolve())
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in generated:
            continue
        rel = path.relative_to(root).as_posix()
        files.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "readable": _is_readable_artifact(path),
            }
        )
    manifest = {
        "contract": DEMO_SET_CONTRACT,
        "root": str(root),
        "packaging": (
            "Use devtools workspace read-package for portable readable bundles; this helper only writes indexes."
        ),
        "curation_policy": (
            "This is not append-only. Keep the best current demos here; replace, consolidate, or move stale demos out."
        ),
        "file_count": len(files),
        "readable_count": sum(1 for item in files if item["readable"]),
        "files": files,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    summary_records = _summary_records(root, generated)
    records = _demo_records(root, files)
    unsummarized_demos = _unsummarized_demos(records, summary_records)
    summary_index_text, summary_count, summary_coverage = _build_summary_index(
        root,
        summary_records,
        unsummarized_demos,
    )
    readme_text = _build_readme(records)
    catalog_text = _build_catalog(records)
    return ShelfRender(
        manifest_path=manifest_path,
        summary_index_path=summary_index_path,
        readme_path=readme_path,
        catalog_path=catalog_path,
        manifest_text=manifest_text,
        summary_index_text=summary_index_text,
        readme_text=readme_text,
        catalog_text=catalog_text,
        file_count=len(files),
        readable_count=sum(1 for item in files if item["readable"]),
        summary_count=summary_count,
        summary_coverage=summary_coverage,
        unsummarized_demos=unsummarized_demos,
    )


def _changed(path: Path, expected: str) -> bool:
    try:
        return path.read_text() != expected
    except FileNotFoundError:
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools workspace demo-shelf",
        description="Refresh or verify curated current demo shelf indexes.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"Demo shelf root (default: {DEFAULT_ROOT})")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"Manifest filename (default: {DEFAULT_MANIFEST})")
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "Deprecated compatibility option; readable bundles are generated by "
            "devtools workspace read-package, not this helper."
        ),
    )
    parser.add_argument(
        "--summary-index",
        default=DEFAULT_SUMMARY_INDEX,
        help=f"Summary index filename (default: {DEFAULT_SUMMARY_INDEX})",
    )
    parser.add_argument(
        "--require-summary-coverage",
        type=_parse_required_summary_coverage,
        default=set(),
        metavar="FIELDS",
        help="In check mode, fail when summaries lack comma-separated fields: claim, non_claim, proof_fields, caveat_fields.",
    )
    parser.add_argument(
        "--require-index-schema-version",
        type=int,
        default=None,
        metavar="VERSION",
        help="In check mode, fail when a summary declares a different index_schema_version.",
    )
    parser.add_argument("--check", action="store_true", help="Verify files are current without writing.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    rendered = render_demo_shelf(
        args.root,
        manifest_name=args.manifest,
        bundle_name=args.bundle,
        summary_index_name=args.summary_index,
    )
    changed = {
        "manifest": _changed(rendered.manifest_path, rendered.manifest_text),
        "summary_index": _changed(rendered.summary_index_path, rendered.summary_index_text),
        "readme": _changed(rendered.readme_path, rendered.readme_text),
        "catalog": _changed(rendered.catalog_path, rendered.catalog_text),
    }
    coverage_failures = _coverage_failures(rendered.summary_coverage, args.require_summary_coverage)
    summary_index_payload = json.loads(rendered.summary_index_text)
    schema_mismatches = _schema_version_mismatches(
        summary_index_payload["records"],
        args.require_index_schema_version,
    )
    ok = not any(changed.values()) and not coverage_failures and not schema_mismatches
    if not args.check:
        rendered.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.manifest_path.write_text(rendered.manifest_text)
        rendered.summary_index_path.write_text(rendered.summary_index_text)
        rendered.readme_path.write_text(rendered.readme_text)
        rendered.catalog_path.write_text(rendered.catalog_text)
        ok = True
        changed = {"manifest": False, "summary_index": False, "readme": False, "catalog": False}
        coverage_failures = {}
        schema_mismatches = []

    payload = {
        "ok": ok,
        "root": str(args.root.expanduser().resolve()),
        "manifest": str(rendered.manifest_path),
        "summary_index": str(rendered.summary_index_path),
        "file_count": rendered.file_count,
        "readable_count": rendered.readable_count,
        "summary_count": rendered.summary_count,
        "unsummarized_demos": rendered.unsummarized_demos,
        "required_summary_coverage": sorted(args.require_summary_coverage),
        "summary_coverage_failures": coverage_failures,
        "required_index_schema_version": args.require_index_schema_version,
        "summary_schema_mismatches": schema_mismatches,
        "changed": changed,
        "mode": "check" if args.check else "write",
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif ok:
        print(
            f"demo shelf {'current' if args.check else 'refreshed'}: "
            f"{rendered.file_count} files, {rendered.readable_count} readable, "
            f"{rendered.summary_count} summaries, "
            f"{len(rendered.unsummarized_demos)} unsummarized; portable bundles: read-package"
        )
    else:
        reasons = [name for name, value in changed.items() if value]
        reasons.extend(f"summary coverage {field}" for field in coverage_failures)
        if schema_mismatches:
            reasons.append("summary schema version")
        print("demo shelf drift:", ", ".join(reasons), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
