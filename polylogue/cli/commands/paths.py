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
        archive_root,
        blob_store_root,
        config_home,
        data_home,
        db_path,
    )

    archive = archive_root().resolve()
    db = db_path().resolve()
    blob = blob_store_root().resolve()
    config_dir = config_home().resolve()
    toml_path = (config_dir / "polylogue.toml").resolve()

    # ── Bind-mount / same-device-inode detection ──────────────────
    bind_mounts: list[dict[str, object]] = []
    with contextlib.suppress(Exception):
        _detect_bind_mounts(archive, bind_mounts)

    # ── Non-canonical files at the archive root ───────────────────
    non_canonical: list[dict[str, object]] = []
    if archive.exists() and archive.is_dir():
        for entry in sorted(archive.iterdir()):
            if entry.name in ("polylogue.db", "polylogue.db-wal", "polylogue.db-shm"):
                continue
            if entry.name.startswith("."):
                continue
            stat = entry.stat()
            non_canonical.append(
                {
                    "name": entry.name,
                    "kind": _entry_kind(entry),
                    "size_bytes": stat.st_size,
                }
            )

    # ── JSON output ────────────────────────────────────────────────
    if output_format == "json":
        payload: dict[str, object] = {
            "archive_root": str(archive),
            "database_path": str(db),
            "database_exists": db.exists(),
            "database_size_bytes": db.stat().st_size if db.exists() else None,
            "config_file_path": str(toml_path),
            "config_file_exists": toml_path.exists(),
            "blob_store_root": str(blob),
            "blob_store_exists": blob.exists(),
            "data_home": str(data_home().resolve()),
            "bind_mounts": bind_mounts,
            "non_canonical_files": non_canonical,
        }
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    # ── Text output ───────────────────────────────────────────────
    _print_line("Archive root", str(archive))
    _print_line("Database", str(db), extra=_size_fmt(db) if db.exists() else "not found")
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

    if non_canonical:
        click.echo("Non-canonical files at archive root:")
        for nc in non_canonical:
            kind = nc.get("kind", "?")
            name = nc.get("name", "?")
            raw_size = nc.get("size_bytes", 0)
            size_bytes = raw_size if isinstance(raw_size, int) else 0
            click.echo(f"  {name}  ({kind}, {_size_fmt_bytes(size_bytes)})")
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


def _entry_kind(entry: Path) -> str:
    if entry.is_symlink():
        return "symlink"
    if entry.is_dir():
        return "directory"
    if entry.is_file():
        return "file"
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
