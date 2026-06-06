"""Paths command — print canonical archive paths and bind-mount detection."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import click


@click.command("paths")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
def paths_command(output_format: str) -> None:
    """Print canonical archive paths and filesystem topology.

    Reports the resolved archive root, database path, config file path,
    blob store root, and whether any bind mounts are detected.
    """
    from polylogue.paths import (
        active_index_db_path,
        archive_root,
        blob_store_root,
        config_home,
        data_home,
    )

    archive = archive_root().resolve()
    active_db = active_index_db_path().resolve()
    active_archive = active_db.parent.resolve()
    db = (active_archive / "index.db").resolve()
    source_db = (active_archive / "source.db").resolve()
    embeddings_db = (active_archive / "embeddings.db").resolve()
    ops_db = (active_archive / "ops.db").resolve()
    user_db = (active_archive / "user.db").resolve()
    tier_paths = {
        "source": source_db,
        "index": db,
        "embeddings": embeddings_db,
        "ops": ops_db,
        "user": user_db,
    }
    present_tiers = [name for name, path in tier_paths.items() if path.exists()]
    missing_tiers = [name for name, path in tier_paths.items() if not path.exists()]
    archive_ready = source_db.exists() and db.exists()
    final_shape_ready = not missing_tiers
    archive_layout_blockers = _archive_layout_blockers(
        present_tiers=present_tiers,
        missing_tiers=missing_tiers,
    )
    archive_layout_ready = not archive_layout_blockers
    storage_layout = _storage_layout(
        present_tiers=present_tiers,
        final_shape_ready=final_shape_ready,
    )
    active_database_role = _active_database_role(active_db, tier_paths=tier_paths)
    blob = blob_store_root().resolve()
    config_dir = config_home().resolve()
    toml_path = (config_dir / "polylogue.toml").resolve()

    # ── Bind-mount / same-device-inode detection ──────────────────
    bind_mounts: list[dict[str, object]] = []
    with contextlib.suppress(Exception):
        _detect_bind_mounts(archive, bind_mounts)

    # ── JSON output ────────────────────────────────────────────────
    if output_format == "json":
        payload: dict[str, object] = {
            "archive_root": str(archive),
            "active_archive_root": str(active_archive),
            "active_archive_root_matches_configured": active_archive == archive,
            "database_path": str(db),
            "database_exists": db.exists(),
            "database_size_bytes": db.stat().st_size if db.exists() else None,
            "active_database_path": str(active_db),
            "active_index_database_path": str(active_archive / "index.db"),
            "active_database_role": active_database_role,
            "active_database_exists": active_db.exists(),
            "active_database_size_bytes": active_db.stat().st_size if active_db.exists() else None,
            "storage_layout": storage_layout,
            "archive_ready": archive_ready,
            "final_shape_ready": final_shape_ready,
            "archive_layout_ready": archive_layout_ready,
            "archive_layout_blockers": archive_layout_blockers,
            "present_tiers": present_tiers,
            "missing_tiers": missing_tiers,
            "source_database_path": str(source_db),
            "source_database_exists": source_db.exists(),
            "source_database_size_bytes": source_db.stat().st_size if source_db.exists() else None,
            "index_database_path": str(db),
            "index_database_exists": db.exists(),
            "index_database_size_bytes": db.stat().st_size if db.exists() else None,
            "embeddings_database_path": str(embeddings_db),
            "embeddings_database_exists": embeddings_db.exists(),
            "embeddings_database_size_bytes": embeddings_db.stat().st_size if embeddings_db.exists() else None,
            "ops_database_path": str(ops_db),
            "ops_database_exists": ops_db.exists(),
            "ops_database_size_bytes": ops_db.stat().st_size if ops_db.exists() else None,
            "user_database_path": str(user_db),
            "user_database_exists": user_db.exists(),
            "user_database_size_bytes": user_db.stat().st_size if user_db.exists() else None,
            "config_file_path": str(toml_path),
            "config_file_exists": toml_path.exists(),
            "blob_store_root": str(blob),
            "blob_store_exists": blob.exists(),
            "data_home": str(data_home().resolve()),
            "bind_mounts": bind_mounts,
        }
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    # ── Text output ───────────────────────────────────────────────
    _print_line("Archive root", str(archive))
    if active_archive != archive:
        _print_line("Active archive root", str(active_archive), extra="derived from active index")
    layout_extra = f"active={active_database_role}"
    if missing_tiers and present_tiers:
        layout_extra += f"; missing={','.join(missing_tiers)}"
    _print_line("Storage layout", storage_layout, extra=layout_extra)
    layout_status_extra = "ready" if archive_layout_ready else f"blocked={','.join(archive_layout_blockers)}"
    _print_line("Archive layout", "ready" if archive_layout_ready else "not ready", extra=layout_status_extra)
    _print_line("Source DB", str(source_db), extra=_size_fmt(source_db) if source_db.exists() else "not found")
    _print_line("Index DB", str(db), extra=_size_fmt(db) if db.exists() else "not found")
    _print_line(
        "Embeddings DB",
        str(embeddings_db),
        extra=_size_fmt(embeddings_db) if embeddings_db.exists() else "not found",
    )
    _print_line("Ops DB", str(ops_db), extra=_size_fmt(ops_db) if ops_db.exists() else "not found")
    _print_line("User DB", str(user_db), extra=_size_fmt(user_db) if user_db.exists() else "not found")
    if active_db != db:
        _print_line("Active DB", str(active_db), extra=_size_fmt(active_db) if active_db.exists() else "not found")
    _print_line("Config file", str(toml_path), extra="exists" if toml_path.exists() else "not found")
    _print_line("Blob store", str(blob), extra=f"present ({_size_fmt(blob)})" if blob.exists() else "not found")
    _print_line("Data home", str(data_home().resolve()))
    click.echo()

    if bind_mounts:
        click.echo("Bind mounts detected:")
        for bm in bind_mounts:
            src = bm.get("source", "?")
            tgt = bm.get("target", "?")
            fs = bm.get("filesystem", "?")
            click.echo(f"  {tgt}  <-  {src}  ({fs})")
        click.echo()


# ── helpers ────────────────────────────────────────────────────────


def _detect_bind_mounts(archive: Path, out: list[dict[str, object]]) -> None:
    """Detect bind mounts that make the archive visible at multiple paths.

    Walks up the directory tree looking for mount points. If the
    archive is under a bind-mounted filesystem, records the source and
    target.
    """
    # Strategy: read /proc/self/mountinfo and look for entries where the
    # mount target is an ancestor of (or equal to) the archive path.
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text()
    except OSError:
        return

    archive_dev = archive.stat().st_dev if archive.exists() else None

    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        # Fields: id parent_id dev root mount_point options ...
        mount_point = fields[4]
        mount_source = fields[3] if fields[3] != "/" else fields[8]
        fs_type = fields[7] if len(fields) > 7 else "?"

        # Check if the archive is under this mount point.
        mp_path = Path(mount_point).resolve()
        try:
            archive.relative_to(mp_path)
        except ValueError:
            continue

        # Skip the trivial self-mount (/).
        if mount_point == "/":
            continue

        src_path = _parse_mount_source(mount_source)

        out.append(
            {
                "target": str(mp_path),
                "source": src_path if src_path else mount_source,
                "filesystem": fs_type,
                "shared_device": bool(
                    archive_dev is not None and mp_path.exists() and mp_path.stat().st_dev == archive_dev
                ),
            }
        )

    # Also check for same-device-different-path via stat.
    # If archive_root() = /realm/data/captures/polylogue and
    # default (= ~/.local/share/polylogue) resolves to a different path
    # but same device/inode, flag it as a bind mount.
    from polylogue.paths import data_home as _data_home

    default_root = _data_home().resolve()
    if default_root != archive and default_root.exists() and archive.exists():
        try:
            if (
                default_root.stat().st_dev == archive.stat().st_dev
                and default_root.stat().st_ino == archive.stat().st_ino
                and not any(
                    str(archive) == str(bm.get("target")) or str(default_root) == str(bm.get("target")) for bm in out
                )
            ):
                out.append(
                    {
                        "target": str(default_root),
                        "source": str(archive),
                        "filesystem": "bind (same inode)",
                        "shared_device": True,
                    }
                )
        except OSError:
            pass


def _parse_mount_source(raw: str) -> str | None:
    """Try to extract a real path from a mount source field."""
    if raw.startswith("/"):
        return raw
    return None


def _storage_layout(*, present_tiers: list[str], final_shape_ready: bool) -> str:
    if final_shape_ready:
        return "archive_complete"
    if present_tiers:
        return "archive_partial"
    return "archive_missing"


def _archive_layout_blockers(
    *,
    present_tiers: list[str],
    missing_tiers: list[str],
) -> list[str]:
    blockers: list[str] = []
    if not present_tiers:
        blockers.append("no_archive_tiers_present")
    if missing_tiers:
        blockers.append("missing_archive_tiers")
    for tier in ("source", "embeddings", "user"):
        if tier in missing_tiers:
            blockers.append(f"missing_backup_required_tier:{tier}")
    return blockers


def _active_database_role(active_db: Path, *, tier_paths: dict[str, Path]) -> str:
    for name, path in tier_paths.items():
        if active_db == path:
            return name
    return "unknown"


def _size_fmt(p: Path) -> str:
    """Human-readable byte size, directory-aware."""
    if p.is_dir():
        try:
            total = sum(f.stat().st_size for f in p.iterdir() if f.is_file())
            return _size_fmt_bytes(total)
        except OSError:
            return "? bytes"
    if p.is_file():
        try:
            return _size_fmt_bytes(p.stat().st_size)
        except OSError:
            return "? bytes"
    return ""


_SZ_UNITS = ("bytes", "KiB", "MiB", "GiB", "TiB")


def _size_fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} bytes"
    size = float(n)
    for unit in _SZ_UNITS[1:]:
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} {_SZ_UNITS[-1]}"


def _print_line(label: str, value: str, *, extra: str | None = None) -> None:
    """Print a left-aligned label and right-aligned value with optional extra."""
    if extra:
        click.echo(f"{label:20s} {value}\n{'':20s} ({extra})")
    else:
        click.echo(f"{label:20s} {value}")


__all__ = ["paths_command"]
