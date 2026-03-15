"""QA artifact snapshot/index command."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import click

from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.machine_errors import emit_success
from polylogue.cli.types import AppEnv
from polylogue.paths import safe_path_component


def _iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_qa_sources(base_dir: Path) -> list[Path]:
    candidates = [base_dir / "qa_outputs", base_dir / "qa_archive"]
    return [path for path in candidates if path.exists() and path.is_dir()]


def _write_index(snapshot_dir: Path, entries: list[dict[str, object]]) -> None:
    lines = [
        "# QA Snapshot Index",
        "",
        f"- Snapshot: `{snapshot_dir.name}`",
        f"- Created: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Files: `{len(entries)}`",
        "",
        "| File | Size (bytes) | SHA256 |",
        "| --- | ---: | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['relative_path']}` | {entry['size_bytes']} | `{entry['sha256']}` |"
        )
    (snapshot_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_latest_symlink(output_root: Path, snapshot_dir: Path) -> None:
    latest = output_root / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(snapshot_dir.name)
    except OSError:
        # Best-effort convenience pointer only.
        pass


@click.command("qa")
@click.option(
    "--source",
    "sources",
    multiple=True,
    type=click.Path(path_type=Path),
    help="QA source directory (repeatable). Defaults: ./qa_outputs and ./qa_archive if present.",
)
@click.option(
    "--name",
    default="snapshot",
    show_default=True,
    help="Snapshot label (included in output directory name).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Snapshot root directory (default: <archive_root>/qa/snapshots).",
)
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable summary JSON")
@click.option(
    "--from-showcase",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Import a showcase output directory as a QA snapshot (reads showcase-manifest.json).",
)
@click.pass_obj
def qa_command(
    env: AppEnv,
    sources: tuple[Path, ...],
    name: str,
    output_root: Path | None,
    json_output: bool,
    from_showcase: Path | None,
) -> None:
    """Snapshot and index QA artifacts to a reproducible archive path."""
    config = load_effective_config(env)

    # --from-showcase: import a showcase output directory as a QA snapshot
    if from_showcase is not None:
        _do_from_showcase(env, from_showcase, name, output_root, json_output, config)
        return

    source_dirs = list(sources) if sources else _default_qa_sources(Path.cwd())
    if not source_dirs:
        fail("qa", "No QA sources found. Provide --source or create qa_outputs/qa_archive.")

    missing = [path for path in source_dirs if not path.exists() or not path.is_dir()]
    if missing:
        fail("qa", f"QA source path(s) missing or not directories: {', '.join(str(p) for p in missing)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = safe_path_component(name, fallback="snapshot")
    root = output_root or (config.archive_root / "qa" / "snapshots")
    snapshot_dir = root / f"{stamp}-{label}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, object]] = []
    for source_dir in source_dirs:
        for src_file in _iter_files(source_dir):
            rel = Path(source_dir.name) / src_file.relative_to(source_dir)
            dst_file = snapshot_dir / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            entries.append(
                {
                    "relative_path": str(rel),
                    "source_path": str(src_file),
                    "size_bytes": dst_file.stat().st_size,
                    "sha256": _sha256(dst_file),
                }
            )

    manifest = {
        "snapshot": snapshot_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dirs": [str(path) for path in source_dirs],
        "entry_count": len(entries),
        "entries": entries,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_index(snapshot_dir, entries)
    _update_latest_symlink(root, snapshot_dir)

    if json_output:
        emit_success({
            "snapshot_dir": str(snapshot_dir),
            "entry_count": len(entries),
            "sources": [str(path) for path in source_dirs],
        })
        return

    env.ui.console.print(f"QA snapshot created: {snapshot_dir}")
    env.ui.console.print(f"Files captured: {len(entries)}")
    env.ui.console.print("Artifacts:")
    env.ui.console.print(f"  - {snapshot_dir / 'manifest.json'}")
    env.ui.console.print(f"  - {snapshot_dir / 'INDEX.md'}")


def _do_from_showcase(
    env: AppEnv,
    showcase_dir: Path,
    name: str,
    output_root: Path | None,
    json_output: bool,
    config: object,
) -> None:
    """Import a showcase output directory as a QA snapshot."""
    manifest_path = showcase_dir / "showcase-manifest.json"
    showcase_manifest: dict | None = None
    if manifest_path.exists():
        showcase_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = safe_path_component(name, fallback="showcase")
    root = output_root or (config.archive_root / "qa" / "snapshots")  # type: ignore[union-attr]
    snapshot_dir = root / f"{stamp}-{label}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, object]] = []
    for src_file in _iter_files(showcase_dir):
        rel = src_file.relative_to(showcase_dir)
        dst_file = snapshot_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        entries.append(
            {
                "relative_path": str(rel),
                "source_path": str(src_file),
                "size_bytes": dst_file.stat().st_size,
                "sha256": _sha256(dst_file),
            }
        )

    manifest: dict[str, object] = {
        "snapshot": snapshot_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dirs": [str(showcase_dir)],
        "entry_count": len(entries),
        "entries": entries,
        "source_type": "showcase",
    }
    if showcase_manifest is not None:
        manifest["showcase_manifest"] = showcase_manifest

    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_index(snapshot_dir, entries)
    _update_latest_symlink(root, snapshot_dir)

    if json_output:
        emit_success({
            "snapshot_dir": str(snapshot_dir),
            "entry_count": len(entries),
            "sources": [str(showcase_dir)],
            "source_type": "showcase",
        })
        return

    env.ui.console.print(f"QA snapshot created from showcase: {snapshot_dir}")
    env.ui.console.print(f"Files captured: {len(entries)}")
    if showcase_manifest:
        env.ui.console.print(
            f"Showcase artifacts: {showcase_manifest.get('entry_count', '?')} entries"
        )
    env.ui.console.print("Artifacts:")
    env.ui.console.print(f"  - {snapshot_dir / 'manifest.json'}")
    env.ui.console.print(f"  - {snapshot_dir / 'INDEX.md'}")


__all__ = ["qa_command"]
