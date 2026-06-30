"""Refresh or verify a local demo shelf manifest and readable bundle."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/realm/inbox/demos_polylogue")
DEFAULT_MANIFEST = "MANIFEST.readable.json"
DEFAULT_BUNDLE = "CONCATENATED_READABLE.md"
READABLE_SUFFIXES = frozenset({".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml"})


@dataclass(frozen=True, slots=True)
class ShelfRender:
    manifest_path: Path
    bundle_path: Path
    manifest_text: str
    bundle_text: str
    file_count: int
    readable_count: int


def _is_readable_artifact(path: Path) -> bool:
    return path.suffix.lower() in READABLE_SUFFIXES


def render_demo_shelf(
    root: Path,
    *,
    manifest_name: str = DEFAULT_MANIFEST,
    bundle_name: str = DEFAULT_BUNDLE,
) -> ShelfRender:
    """Build the deterministic manifest and concatenated readable bundle."""

    root = root.expanduser().resolve()
    manifest_path = root / manifest_name
    bundle_path = root / bundle_name
    generated = {manifest_path.resolve(), bundle_path.resolve()}
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
    return ShelfRender(
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        manifest_text=manifest_text,
        bundle_text=bundle_text,
        file_count=len(files),
        readable_count=sum(1 for item in files if item["readable"]),
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
    parser.add_argument("--check", action="store_true", help="Verify files are current without writing.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    rendered = render_demo_shelf(args.root, manifest_name=args.manifest, bundle_name=args.bundle)
    changed = {
        "manifest": _changed(rendered.manifest_path, rendered.manifest_text),
        "bundle": _changed(rendered.bundle_path, rendered.bundle_text),
    }
    ok = not any(changed.values())
    if not args.check:
        rendered.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.manifest_path.write_text(rendered.manifest_text)
        rendered.bundle_path.write_text(rendered.bundle_text)
        ok = True
        changed = {"manifest": False, "bundle": False}

    payload = {
        "ok": ok,
        "root": str(args.root.expanduser().resolve()),
        "manifest": str(rendered.manifest_path),
        "bundle": str(rendered.bundle_path),
        "file_count": rendered.file_count,
        "readable_count": rendered.readable_count,
        "changed": changed,
        "mode": "check" if args.check else "write",
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif ok:
        print(
            f"demo shelf {'current' if args.check else 'refreshed'}: "
            f"{rendered.file_count} files, {rendered.readable_count} readable"
        )
    else:
        print("demo shelf drift:", ", ".join(name for name, value in changed.items() if value), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
