"""Refresh or verify a local demo shelf manifest and readable bundle."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/realm/inbox/demos_polylogue")
DEFAULT_MANIFEST = "MANIFEST.readable.json"
DEFAULT_BUNDLE = "CONCATENATED_READABLE.md"
DEFAULT_SUMMARY_INDEX = "SUMMARY_INDEX.json"
READABLE_SUFFIXES = frozenset({".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml"})
SUMMARY_COVERAGE_FIELDS = frozenset({"claim", "non_claim", "proof_fields", "caveat_fields"})


@dataclass(frozen=True, slots=True)
class ShelfRender:
    manifest_path: Path
    bundle_path: Path
    summary_index_path: Path
    manifest_text: str
    bundle_text: str
    summary_index_text: str
    file_count: int
    readable_count: int
    summary_count: int
    summary_coverage: dict[str, list[str]]


def _is_readable_artifact(path: Path) -> bool:
    return path.suffix.lower() in READABLE_SUFFIXES


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


def _build_summary_index(root: Path, generated: set[Path]) -> tuple[str, int, dict[str, list[str]]]:
    records = []
    for path in sorted(root.rglob("*summary.json")):
        if not path.is_file() or path.resolve() in generated:
            continue
        records.append(_summary_record(root, path))
    coverage = {
        "without_claim": [record["summary_path"] for record in records if not record["coverage"]["claim"]],
        "without_non_claim": [record["summary_path"] for record in records if not record["coverage"]["non_claim"]],
        "without_proof_fields": [
            record["summary_path"] for record in records if not record["coverage"]["proof_fields"]
        ],
        "without_caveat_fields": [
            record["summary_path"] for record in records if not record["coverage"]["caveat_fields"]
        ],
    }
    summary_index = {
        "root": str(root),
        "summary_count": len(records),
        "coverage": coverage,
        "records": records,
    }
    return json.dumps(summary_index, indent=2) + "\n", len(records), coverage


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
    bundle_name: str = DEFAULT_BUNDLE,
    summary_index_name: str = DEFAULT_SUMMARY_INDEX,
) -> ShelfRender:
    """Build the deterministic manifest and concatenated readable bundle."""

    root = root.expanduser().resolve()
    manifest_path = root / manifest_name
    bundle_path = root / bundle_name
    summary_index_path = root / summary_index_name
    generated = {manifest_path.resolve(), bundle_path.resolve(), summary_index_path.resolve()}
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
        "root": str(root),
        "bundle": str(bundle_path),
        "file_count": len(files),
        "readable_count": sum(1 for item in files if item["readable"]),
        "files": files,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    parts = [
        "# Demo Shelf - Concatenated Readable Artifacts",
        "",
        f"Generated from `{manifest_path}`.",
        "",
    ]
    for item in files:
        if not item["readable"]:
            continue
        artifact_path = root / str(item["path"])
        try:
            text = artifact_path.read_text(errors="replace").rstrip()
        except OSError as exc:
            text = f"[unreadable: {exc}]"
        parts.extend([f"## {item['path']}", "", text, ""])
    bundle_text = "\n".join(parts).rstrip() + "\n"
    summary_index_text, summary_count, summary_coverage = _build_summary_index(root, generated)
    return ShelfRender(
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        summary_index_path=summary_index_path,
        manifest_text=manifest_text,
        bundle_text=bundle_text,
        summary_index_text=summary_index_text,
        file_count=len(files),
        readable_count=sum(1 for item in files if item["readable"]),
        summary_count=summary_count,
        summary_coverage=summary_coverage,
    )


def _changed(path: Path, expected: str) -> bool:
    try:
        return path.read_text() != expected
    except FileNotFoundError:
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools workspace demo-shelf",
        description="Refresh or verify a demo shelf readable manifest and concatenated bundle.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"Demo shelf root (default: {DEFAULT_ROOT})")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"Manifest filename (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, help=f"Bundle filename (default: {DEFAULT_BUNDLE})")
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
        "bundle": _changed(rendered.bundle_path, rendered.bundle_text),
        "summary_index": _changed(rendered.summary_index_path, rendered.summary_index_text),
    }
    coverage_failures = _coverage_failures(rendered.summary_coverage, args.require_summary_coverage)
    ok = not any(changed.values()) and not coverage_failures
    if not args.check:
        rendered.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.manifest_path.write_text(rendered.manifest_text)
        rendered.bundle_path.write_text(rendered.bundle_text)
        rendered.summary_index_path.write_text(rendered.summary_index_text)
        ok = True
        changed = {"manifest": False, "bundle": False, "summary_index": False}
        coverage_failures = {}

    payload = {
        "ok": ok,
        "root": str(args.root.expanduser().resolve()),
        "manifest": str(rendered.manifest_path),
        "bundle": str(rendered.bundle_path),
        "summary_index": str(rendered.summary_index_path),
        "file_count": rendered.file_count,
        "readable_count": rendered.readable_count,
        "summary_count": rendered.summary_count,
        "required_summary_coverage": sorted(args.require_summary_coverage),
        "summary_coverage_failures": coverage_failures,
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
            f"{rendered.summary_count} summaries"
        )
    else:
        reasons = [name for name, value in changed.items() if value]
        reasons.extend(f"summary coverage {field}" for field in coverage_failures)
        print("demo shelf drift:", ", ".join(reasons), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
