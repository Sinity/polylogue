"""Paths command — print canonical archive paths and bind-mount detection."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import click

from polylogue.storage import archive_layout
from polylogue.storage.archive_readiness import (
    active_rebuild_index_attempts,
    raw_materialization_readiness_snapshot,
    raw_materialization_ready,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


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
    tier_versions = _tier_version_status(tier_paths)
    present_tiers = [name for name, path in tier_paths.items() if path.exists()]
    missing_tiers = [name for name, path in tier_paths.items() if not path.exists()]
    archive_schema_ready = all(
        tier_versions[name]["version_status"] == "ok" for name in ("source", "index", "embeddings", "ops", "user")
    )
    active_rebuild_attempts = active_rebuild_index_attempts(ops_db)
    raw_materialization_readiness = _raw_materialization_readiness(active_archive)
    archive_materialization_ready = (
        source_db.exists()
        and db.exists()
        and archive_schema_ready
        and not active_rebuild_attempts
        and raw_materialization_ready(raw_materialization_readiness)
    )
    archive_ready = (
        source_db.exists()
        and db.exists()
        and archive_schema_ready
        and not active_rebuild_attempts
        and archive_materialization_ready
    )
    final_shape_ready = not missing_tiers
    missing_backup_required = [tier for tier in archive_layout.BACKUP_REQUIRED_TIERS if tier in missing_tiers]
    layout_blockers = archive_layout.archive_layout_blockers(
        present_count=len(present_tiers),
        final_shape_ready=final_shape_ready,
        missing_backup_required=missing_backup_required,
    )
    archive_layout_ready = not layout_blockers
    storage_layout = archive_layout.classify_storage_layout(
        present_count=len(present_tiers),
        final_shape_ready=final_shape_ready,
    )
    active_database_role = archive_layout.active_tier_role(active_db, tier_paths)
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
            "archive_materialization_ready": archive_materialization_ready,
            "raw_materialization_readiness": raw_materialization_readiness,
            "active_rebuild_index_attempts": active_rebuild_attempts,
            "final_shape_ready": final_shape_ready,
            "archive_schema_ready": archive_schema_ready,
            "archive_layout_ready": archive_layout_ready,
            "archive_layout_blockers": layout_blockers,
            "archive_tier_versions": tier_versions,
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
    layout_status_extra = "ready" if archive_layout_ready else f"blocked={','.join(layout_blockers)}"
    _print_line("Archive layout", "ready" if archive_layout_ready else "not ready", extra=layout_status_extra)
    schema_extra = "ready" if archive_schema_ready else _schema_blocker_text(tier_versions)
    _print_line("Archive schema", "ready" if archive_schema_ready else "not ready", extra=schema_extra)
    if active_rebuild_attempts:
        _print_line("Archive materialization", "rebuilding", extra=f"attempts={len(active_rebuild_attempts)}")
    else:
        _print_line("Archive materialization", "ready" if archive_materialization_ready else "not ready")
    _print_line("Source DB", str(source_db), extra=_tier_extra("source", source_db, tier_versions))
    _print_line("Index DB", str(db), extra=_tier_extra("index", db, tier_versions))
    _print_line(
        "Embeddings DB",
        str(embeddings_db),
        extra=_tier_extra("embeddings", embeddings_db, tier_versions),
    )
    _print_line("Ops DB", str(ops_db), extra=_tier_extra("ops", ops_db, tier_versions))
    _print_line("User DB", str(user_db), extra=_tier_extra("user", user_db, tier_versions))
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

_TIER_ENUM_BY_NAME = {
    "source": ArchiveTier.SOURCE,
    "index": ArchiveTier.INDEX,
    "embeddings": ArchiveTier.EMBEDDINGS,
    "ops": ArchiveTier.OPS,
    "user": ArchiveTier.USER,
}


def _tier_version_status(tier_paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    status: dict[str, dict[str, object]] = {}
    for name, path in tier_paths.items():
        expected = ARCHIVE_VERSION_BY_TIER[_TIER_ENUM_BY_NAME[name]]
        user_version = _read_user_version(path) if path.exists() else None
        if not path.exists():
            version_status = "missing"
        elif user_version == expected:
            version_status = "ok"
        elif user_version is None:
            version_status = "invalid"
        else:
            version_status = "mismatch"
        status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "user_version": user_version,
            "expected_user_version": expected,
            "version_status": version_status,
        }
    return status


def _read_user_version(path: Path) -> int | None:
    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            row = conn.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0] or 0) if row is not None else 0


def _tier_extra(name: str, path: Path, tier_versions: dict[str, dict[str, object]]) -> str:
    if not path.exists():
        return "not found"
    info = tier_versions[name]
    version = info["user_version"]
    expected = info["expected_user_version"]
    version_status = info["version_status"]
    return f"{_size_fmt(path)}; v{version}/{expected} {version_status}"


def _schema_blocker_text(tier_versions: dict[str, dict[str, object]]) -> str:
    blockers = [
        f"{name}:{info['version_status']}({info['user_version']}/{info['expected_user_version']})"
        for name, info in tier_versions.items()
        if info["version_status"] != "ok"
    ]
    return "blocked=" + ",".join(blockers)


def _raw_materialization_readiness(active_archive: Path) -> dict[str, object]:
    readiness = raw_materialization_readiness_snapshot(active_archive)
    return _merge_raw_materialization_debt(readiness, active_archive)


def _merge_raw_materialization_debt(readiness: dict[str, object], active_archive: Path) -> dict[str, object]:
    """Fold the unified raw-materialization debt classifier into readiness.

    The structural snapshot is intentionally cheap and local to raw_id joins.
    Archive debt is the richer shared classifier for actionable/blocking raw
    materialization rows. `paths` reports archive readiness, so it must compose
    both instead of treating the richer debt projection as an optional operator
    command.
    """
    try:
        from polylogue.operations import archive_debt as archive_debt_ops

        payload = archive_debt_ops.archive_debt_list(
            archive_root=active_archive,
            kinds=("raw-materialization",),
            exact_fts=False,
        )
    except Exception as exc:
        # The docstring contract is that paths composes the debt classifier;
        # when it fails, readiness must say so instead of implying clean debt.
        degraded = dict(readiness)
        degraded["debt_classifier_error"] = f"{type(exc).__name__}: {exc}"
        return degraded
    rows = getattr(payload, "rows", ())
    totals = getattr(payload, "totals", None)
    if totals is None or not rows:
        return readiness
    enriched = dict(readiness)
    enriched["available"] = True
    enriched["classification"] = _combined_classification(str(enriched.get("classification") or "not_run"))
    for key in (
        "total",
        "critical",
        "warning",
        "info",
        "actionable",
        "blocked",
        "classified",
        "affected_total",
        "affected_critical",
        "affected_warning",
        "affected_info",
        "affected_actionable",
        "affected_blocked",
        "affected_open",
        "affected_classified",
    ):
        enriched[key] = _max_payload_int(enriched.get(key), getattr(totals, key, 0))
    existing_categories = _string_int_mapping(enriched.get("category_counts"))
    debt_categories: dict[str, int] = {}
    source_families = _string_int_mapping(enriched.get("source_family_counts"))
    for row in rows:
        category = getattr(row, "category", None)
        affected_count = getattr(row, "affected_count", None)
        count = _max_payload_int(affected_count, 1)
        if category:
            debt_categories[str(category)] = debt_categories.get(str(category), 0) + count
        source_family = getattr(row, "source_family", None)
        if source_family:
            family = str(source_family)
            source_families[family] = _max_payload_int(source_families.get(family), 0) + count
    existing_categories.update(debt_categories)
    enriched["category_counts"] = existing_categories
    enriched["source_family_counts"] = source_families
    return enriched


def _combined_classification(current: str) -> str:
    if current and current != "not_run":
        return f"{current}+archive_debt"
    return "archive_debt"


def _max_payload_int(left: Any, right: Any) -> int:
    return max(_payload_int(left), _payload_int(right))


def _payload_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _string_int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _payload_int(item) for key, item in value.items()}


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
