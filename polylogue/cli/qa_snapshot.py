"""QA snapshot archival helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from polylogue.cli.machine_errors import emit_success
from polylogue.cli.qa_requests import QASnapshotPlan
from polylogue.cli.types import AppEnv
from polylogue.paths import safe_path_component


def iter_snapshot_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_snapshot_index(snapshot_dir: Path, entries: list[dict[str, object]]) -> None:
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
        lines.append(f"| `{entry['relative_path']}` | {entry['size_bytes']} | `{entry['sha256']}` |")
    (snapshot_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_latest_symlink(output_root: Path, snapshot_dir: Path) -> None:
    latest = output_root / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(snapshot_dir.name)
    except OSError:
        pass


def snapshot_results(
    source_dir: Path,
    *,
    label: str,
    output_root: Path,
    json_output: bool,
    env: AppEnv,
) -> None:
    """Archive a QA output directory into a timestamped snapshot."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = safe_path_component(label, fallback="snapshot")
    snapshot_dir = output_root / f"{stamp}-{safe_label}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, object]] = []
    for src_file in iter_snapshot_files(source_dir):
        rel = src_file.relative_to(source_dir)
        dst_file = snapshot_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        entries.append(
            {
                "relative_path": str(rel),
                "source_path": str(src_file),
                "size_bytes": dst_file.stat().st_size,
                "sha256": sha256_file(dst_file),
            }
        )

    manifest: dict[str, object] = {
        "snapshot": snapshot_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dirs": [str(source_dir)],
        "entry_count": len(entries),
        "entries": entries,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_snapshot_index(snapshot_dir, entries)
    update_latest_symlink(output_root, snapshot_dir)

    if json_output:
        emit_success(
            {
                "snapshot_dir": str(snapshot_dir),
                "entry_count": len(entries),
                "sources": [str(source_dir)],
            }
        )
    else:
        env.ui.console.print(f"QA snapshot created: {snapshot_dir}")
        env.ui.console.print(f"Files captured: {len(entries)}")


def execute_snapshot_plan(
    plan: QASnapshotPlan,
    *,
    fallback_source_dir: Path | None,
    output_root: Path,
    json_output: bool,
    env: AppEnv,
) -> bool:
    """Execute a normalized snapshot plan if a source directory is available."""
    source_dir = plan.resolve_source_dir(fallback_source_dir)
    if source_dir is None:
        return False
    snapshot_results(
        source_dir,
        label=plan.label,
        output_root=output_root,
        json_output=json_output,
        env=env,
    )
    return True


__all__ = ["execute_snapshot_plan", "snapshot_results"]
