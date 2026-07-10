"""Refresh or verify a curated current demo shelf."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
MAX_EXCLUSION_SAMPLES = 20


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
    input_closure_mode: str
    display_root: str
    excluded_input_count: int
    exclusion_reason_counts: dict[str, int]
    exclusion_samples: list[dict[str, str]]


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


def _summary_records(root: Path, included_paths: set[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(included_paths):
        if path.name.endswith("summary.json"):
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


def _demo_records(
    root: Path,
    files: list[dict[str, object]],
    included_paths: set[Path],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    demo_ids = sorted(
        {
            str(record["path"]).split("/", maxsplit=1)[0]
            for record in files
            if "/" in str(record["path"]) and not str(record["path"]).startswith(".")
        }
    )
    for demo_id in demo_ids:
        entry = root / demo_id
        readme_candidates = (entry / "README.md", entry / "current" / "README.md")
        analysis_candidates = (entry / "ANALYSIS.md", entry / "current" / "ANALYSIS.md")
        readme = next((path for path in readme_candidates if path.resolve() in included_paths), None)
        analysis = next((path for path in analysis_candidates if path.resolve() in included_paths), None)
        title = (
            (_first_heading(readme) if readme is not None else "")
            or (_first_heading(analysis) if analysis is not None else "")
            or demo_id.replace("-", " ").title()
        )
        entry_files = [record for record in files if str(record["path"]).startswith(f"{demo_id}/")]
        readable_count = sum(1 for record in entry_files if record["readable"])
        records.append(
            {
                "id": demo_id,
                "title": title,
                "readme": readme.relative_to(root).as_posix() if readme is not None else None,
                "analysis": analysis.relative_to(root).as_posix() if analysis is not None else None,
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


def _repository_context(root: Path) -> tuple[Path, str] | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    repository_root = Path(result.stdout.strip()).resolve()
    try:
        relative_root = root.relative_to(repository_root).as_posix()
    except ValueError:
        return None
    return repository_root, relative_root or "."


def _select_input_files(
    root: Path,
    *,
    generated: set[Path],
    output_root: Path,
    private_projection: bool,
) -> tuple[str, str, list[Path], list[dict[str, str]]]:
    repository_context = _repository_context(root)
    relative_root = repository_context[1] if repository_context is not None else None
    display_root = relative_root if relative_root is not None else str(root)
    all_files = [
        path.resolve()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.resolve() not in generated
        and (output_root == root or not path.resolve().is_relative_to(output_root))
    ]
    if private_projection or repository_context is None:
        mode = "filesystem-private" if private_projection else "filesystem"
        return mode, display_root, all_files, []

    repository_root, relative_root = repository_context
    result = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "-z", "--", relative_root],
        check=True,
        capture_output=True,
    )
    tracked = {
        (repository_root / raw.decode("utf-8", errors="surrogateescape")).resolve()
        for raw in result.stdout.split(b"\0")
        if raw
    }
    included = [path for path in all_files if path in tracked]
    excluded = [
        {"path": path.relative_to(root).as_posix(), "reason": "not_git_tracked"}
        for path in all_files
        if path not in tracked
    ]
    return "git-tracked", display_root, included, excluded


def render_demo_shelf(
    root: Path,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
    bundle_name: str | None = None,
    summary_index_name: str = DEFAULT_SUMMARY_INDEX,
    readme_name: str = DEFAULT_README,
    catalog_name: str = DEFAULT_CATALOG,
    private_output: Path | None = None,
) -> ShelfRender:
    """Build deterministic indexes from tracked inputs or an explicit private projection."""
    root = root.expanduser().resolve()
    output_root = private_output.expanduser().resolve() if private_output is not None else root
    if private_output is not None and output_root.is_relative_to(root):
        raise ValueError("private output must be outside the committed shelf root")
    manifest_path = output_root / manifest_name
    summary_index_path = output_root / summary_index_name
    readme_path = output_root / readme_name
    catalog_path = output_root / catalog_name
    generated = {
        path.resolve()
        for path in (
            manifest_path,
            summary_index_path,
            readme_path,
            catalog_path,
            root / manifest_name,
            root / summary_index_name,
            root / readme_name,
            root / catalog_name,
        )
    }
    if bundle_name:
        generated.update({(output_root / bundle_name).resolve(), (root / bundle_name).resolve()})
    mode, display_root, input_files, excluded_inputs = _select_input_files(
        root,
        generated=generated,
        output_root=output_root,
        private_projection=private_output is not None,
    )
    reason_counts: dict[str, int] = {}
    for excluded in excluded_inputs:
        reason = excluded["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "readable": _is_readable_artifact(path),
        }
        for path in input_files
    ]
    manifest = {
        "contract": DEMO_SET_CONTRACT,
        "root": display_root,
        "input_closure": {
            "mode": mode,
            "included_count": len(files),
            "excluded_count": len(excluded_inputs),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "exclusion_samples": "bounded samples are emitted in the command JSON payload",
        },
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
    included_paths = set(input_files)
    summary_records = _summary_records(root, included_paths)
    records = _demo_records(root, files, included_paths)
    unsummarized_demos = _unsummarized_demos(records, summary_records)
    summary_index_text, summary_count, summary_coverage = _build_summary_index(
        Path(display_root),
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
        input_closure_mode=mode,
        display_root=display_root,
        excluded_input_count=len(excluded_inputs),
        exclusion_reason_counts=dict(sorted(reason_counts.items())),
        exclusion_samples=excluded_inputs[:MAX_EXCLUSION_SAMPLES],
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
    parser.add_argument(
        "--private-output",
        type=Path,
        default=None,
        help="Write a full-filesystem private projection to this separate, untracked directory.",
    )
    parser.add_argument("--check", action="store_true", help="Verify files are current without writing.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    rendered = render_demo_shelf(
        args.root,
        manifest_name=args.manifest,
        bundle_name=args.bundle,
        summary_index_name=args.summary_index,
        private_output=args.private_output,
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
    undeclared_inputs = rendered.input_closure_mode == "git-tracked" and rendered.excluded_input_count > 0
    ok = not any(changed.values()) and not coverage_failures and not schema_mismatches and not undeclared_inputs
    if not args.check and not undeclared_inputs:
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
        "input_closure": {
            "mode": rendered.input_closure_mode,
            "included_count": rendered.file_count,
            "excluded_count": rendered.excluded_input_count,
            "exclusion_reason_counts": rendered.exclusion_reason_counts,
        },
        "undeclared_inputs": undeclared_inputs,
        "exclusion_samples": rendered.exclusion_samples,
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
        if undeclared_inputs:
            reasons.append("undeclared shelf inputs (use --private-output for a private projection)")
        print("demo shelf drift:", ", ".join(reasons), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
