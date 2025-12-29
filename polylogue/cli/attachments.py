from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

from ..commands import CommandEnv
from .context import resolve_output_roots
from ..schema import stamp_payload
from ..util import parse_input_time_to_epoch


def _parse_provider_filter(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    values = {chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()}
    return values or None


def _iter_indexed_attachments(env: CommandEnv) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        query = """
            SELECT
                a.provider,
                a.conversation_id,
                a.branch_id,
                a.message_id,
                a.attachment_name,
                a.attachment_path,
                a.size_bytes,
                a.content_hash,
                a.mime_type,
                a.text_bytes,
                a.ocr_used,
                m.timestamp AS message_timestamp
            FROM attachments a
            LEFT JOIN messages m
              ON m.provider = a.provider
             AND m.conversation_id = a.conversation_id
             AND m.branch_id = a.branch_id
             AND m.message_id = a.message_id
        """
        rows_raw = env.conversations.database.query(query)
    except Exception:
        return rows
    for row in rows_raw:
        rows.append(dict(row))
    return rows


def _iter_attachment_files(root: Path) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for att_dir in root.rglob("attachments"):
        if att_dir.is_dir():
            for candidate in att_dir.iterdir():
                if candidate.is_file():
                    files.append(candidate)
    return files


def _collect_roots(args_dir, env: CommandEnv) -> List[Path]:
    if args_dir:
        root = Path(args_dir).expanduser()
        return [root]
    return [path for path in resolve_output_roots(env.config) if path.exists()]


def run_attachments_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    sub = getattr(args, "attachments_cmd", None)
    if sub == "stats":
        _run_attachment_stats(args, env)
    elif sub == "extract":
        _run_attachment_extract(args, env)
    else:
        env.ui.console.print(f"[red]Unknown attachments subcommand: {sub}")
        raise SystemExit(1)


def _run_attachment_stats(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    use_index = bool(getattr(args, "from_index", False))
    provider_filter = _parse_provider_filter(getattr(args, "provider", None))
    since_epoch = parse_input_time_to_epoch(getattr(args, "since", None))
    until_epoch = parse_input_time_to_epoch(getattr(args, "until", None))
    clean_orphans = bool(getattr(args, "clean_orphans", False))
    dry_run = bool(getattr(args, "dry_run", False))
    json_lines = bool(getattr(args, "json_lines", False))
    json_mode = bool(getattr(args, "json", False) or json_lines)
    if (since_epoch is not None or until_epoch is not None) and not use_index:
        ui.console.print("[red]--since/--until require --from-index for attachment stats.")
        raise SystemExit(1)
    if clean_orphans and not use_index:
        ui.console.print("[red]--clean-orphans requires --from-index.")
        raise SystemExit(1)

    orphan_summary: Dict[str, object] = {"requested": clean_orphans, "count": 0, "removed": 0, "bytes": 0, "errors": 0}
    if clean_orphans:
        roots = _collect_roots(getattr(args, "dir", None), env)
        if not roots:
            ui.console.print("[red]No output roots found for orphan cleanup.")
            raise SystemExit(1)

        referenced_rows = _iter_indexed_attachments(env)
        if provider_filter:
            referenced_rows = [r for r in referenced_rows if str(r.get("provider") or "").lower() in provider_filter]
        referenced = _referenced_attachment_paths(referenced_rows)

        disk_files: List[Path] = []
        for root in roots:
            disk_files.extend(_iter_attachment_files(root))
        if provider_filter:
            filtered_disk: List[Path] = []
            for path in disk_files:
                parts = {part.lower() for part in path.parts}
                if any(provider in parts for provider in provider_filter):
                    filtered_disk.append(path)
            disk_files = filtered_disk

        orphan_files: List[Path] = []
        orphan_bytes = 0
        for file in disk_files:
            try:
                resolved = file.resolve()
            except Exception:
                resolved = file
            if resolved in referenced:
                continue
            orphan_files.append(file)
            try:
                orphan_bytes += file.stat().st_size
            except OSError:
                pass

        orphan_summary["count"] = len(orphan_files)
        orphan_summary["bytes"] = orphan_bytes
        if dry_run:
            if not json_mode:
                for path in orphan_files[:50]:
                    ui.console.print(f"[yellow][dry-run] Would remove orphan: {path}")
                if len(orphan_files) > 50:
                    ui.console.print(f"[yellow][dry-run] ... and {len(orphan_files) - 50} more")
        else:
            removed = 0
            errors = 0
            for path in orphan_files:
                try:
                    path.unlink()
                    removed += 1
                    parent = path.parent
                    if parent.name == "attachments":
                        try:
                            if not any(parent.iterdir()):
                                parent.rmdir()
                        except Exception:
                            pass
                except Exception:
                    errors += 1
            orphan_summary["removed"] = removed
            orphan_summary["errors"] = errors

    if use_index:
        rows = _iter_indexed_attachments(env)
        if not rows:
            ui.console.print("[yellow]No indexed attachments found.")
            return
        if provider_filter:
            rows = [r for r in rows if str(r.get("provider") or "").lower() in provider_filter]
        ext_filter = getattr(args, "ext", None)
        if ext_filter:
            ext_lower = ext_filter.lower()
            rows = [r for r in rows if str(r.get("attachment_name", "")).lower().endswith(ext_lower)]
        missing_ts = 0
        filtered_out_by_time = 0
        if since_epoch is not None or until_epoch is not None:
            filtered_rows: List[Dict[str, object]] = []
            for row in rows:
                ts = row.get("message_timestamp")
                ts_epoch = parse_input_time_to_epoch(ts) if ts else None
                if ts_epoch is None:
                    missing_ts += 1
                    continue
                if since_epoch is not None and ts_epoch < since_epoch:
                    filtered_out_by_time += 1
                    continue
                if until_epoch is not None and ts_epoch > until_epoch:
                    filtered_out_by_time += 1
                    continue
                filtered_rows.append(row)
            rows = filtered_rows

        total_bytes = sum(int(r.get("size_bytes") or 0) for r in rows)
        total_text_bytes = sum(int(r.get("text_bytes") or 0) for r in rows)
        ocr_used = sum(1 for r in rows if r.get("ocr_used"))
        hashes: Dict[str, Tuple[int, str]] = {}
        hashed_bytes = 0
        if getattr(args, "hash", False):
            for r in rows:
                digest = r.get("content_hash")
                size_val = int(r.get("size_bytes") or 0)
                if digest and digest not in hashes:
                    hashes[digest] = (size_val, r.get("attachment_path") or "")
                    hashed_bytes += size_val
        limit = getattr(args, "limit", 10)
        top_rows = sorted(
            rows,
            key=lambda r: (int(r.get("size_bytes") or 0), str(r.get("attachment_name") or "")),
            reverse=True,
        )
        if limit > 0:
            top_rows = top_rows[:limit]
        if json_lines:
            for row in rows:
                print(json.dumps(stamp_payload(row, include_versions=False), separators=(",", ":")))
            return
        if json_mode:
            payload = stamp_payload(
                {
                    "count": len(rows),
                    "bytes": total_bytes,
                    "textBytes": total_text_bytes,
                    "ocrUsed": ocr_used,
                    "uniqueBytes": hashed_bytes if getattr(args, "hash", False) else None,
                    "providers": sorted(provider_filter) if provider_filter else None,
                    "since": getattr(args, "since", None),
                    "until": getattr(args, "until", None),
                    "missingTimestamps": missing_ts,
                    "filteredOutByTime": filtered_out_by_time,
                    "orphans": orphan_summary,
                    "top": top_rows,
                }
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        csv_target = getattr(args, "csv", None)
        if csv_target:
            target = Path(csv_target)
            fieldnames = [
                "attachment_name",
                "size_bytes",
                "text_bytes",
                "ocr_used",
                "attachment_path",
                "provider",
                "conversation_id",
                "branch_id",
                "message_id",
            ]
            if str(target) == "-":
                writer = csv.DictWriter(env.ui.console.file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(top_rows)
            else:
                target = target.expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(top_rows)
                ui.console.print(f"[green]Wrote attachment rows to {target}")
            return
        lines = [
            f"Indexed attachments: {len(rows)} (~{total_bytes / (1024 * 1024):.2f} MiB) text={total_text_bytes} bytes",
            f"OCR used on {ocr_used} attachment(s)",
        ]
        if provider_filter:
            lines.append(f"Providers: {', '.join(sorted(provider_filter))}")
        if since_epoch is not None or until_epoch is not None:
            lines.append(f"Time window: {getattr(args, 'since', None) or '-'} .. {getattr(args, 'until', None) or '-'}")
            if missing_ts:
                lines.append(f"Missing timestamps: {missing_ts}")
            if filtered_out_by_time:
                lines.append(f"Filtered out by time: {filtered_out_by_time}")
        if getattr(args, "hash", False):
            lines.append(f"Unique (by hash): {len(hashes)} (~{hashed_bytes / (1024 * 1024):.2f} MiB)")
        if clean_orphans:
            lines.append(
                f"Orphans: {orphan_summary['count']} (~{int(orphan_summary['bytes']) / (1024 * 1024):.2f} MiB) removed={orphan_summary['removed']} errors={orphan_summary['errors']}"
            )
        if top_rows:
            lines.append("Top attachments:")
            for row in top_rows:
                lines.append(
                    f"  {row.get('attachment_name')} {row.get('size_bytes')} bytes text={row.get('text_bytes')} ({row.get('attachment_path')})"
                )
        ui.summary("Attachment Stats (index)", lines)
        return

    roots = _collect_roots(getattr(args, "dir", None))
    if not roots:
        ui.console.print("[red]No output roots found for attachment scan.")
        raise SystemExit(1)

    files: List[Path] = []
    for root in roots:
        files.extend(_iter_attachment_files(root))

    if provider_filter:
        filtered_files: List[Path] = []
        for path in files:
            parts = {part.lower() for part in path.parts}
            if any(provider in parts for provider in provider_filter):
                filtered_files.append(path)
        files = filtered_files

    ext_filter = getattr(args, "ext", None)
    if ext_filter:
        ext_lower = ext_filter.lower()
        files = [p for p in files if p.suffix.lower() == ext_lower]

    total_bytes = 0
    hashes: Dict[str, Tuple[int, Path]] = {}
    hashed_bytes = 0
    rows = []
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total_bytes += size
        row = {"path": str(path), "name": path.name, "size": size}
        rows.append(row)
        if getattr(args, "hash", False):
            digest = _hash_file(path)
            if digest not in hashes:
                hashes[digest] = (size, path)
                hashed_bytes += size
    sort_field = getattr(args, "sort", "size")
    reverse = sort_field == "size"
    rows_sorted = sorted(rows, key=lambda r: (r[sort_field], r["name"]) if sort_field == "size" else (r["name"], r["size"]), reverse=reverse)
    limit = getattr(args, "limit", 10)
    if limit > 0:
        rows_sorted = rows_sorted[:limit]

    json_lines = bool(getattr(args, "json_lines", False))
    json_mode = bool(getattr(args, "json", False) or json_lines)
    if json_lines:
        for row in rows:
            print(json.dumps(stamp_payload(row, include_versions=False), separators=(",", ":")))
        return
    if json_mode:
        payload = stamp_payload(
            {
                "roots": [str(r) for r in roots],
                "count": len(files),
                "bytes": total_bytes,
                "uniqueBytes": hashed_bytes if getattr(args, "hash", False) else None,
                "providers": sorted(provider_filter) if provider_filter else None,
                "orphans": orphan_summary if clean_orphans else None,
                "top": rows_sorted,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    csv_target = getattr(args, "csv", None)
    if csv_target:
        target = Path(csv_target)
        fieldnames = ["name", "size", "path"]
        if str(target) == "-":
            writer = csv.DictWriter(env.ui.console.file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)
        else:
            target = target.expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_sorted)
            ui.console.print(f"[green]Wrote attachment rows to {target}")
        return

    lines = [f"Roots: {', '.join(str(r) for r in roots)}"]
    if provider_filter:
        lines.append(f"Providers: {', '.join(sorted(provider_filter))}")
    lines.append(f"Attachments: {len(files)} (~{total_bytes / (1024 * 1024):.2f} MiB)")
    if getattr(args, "hash", False):
        lines.append(f"Unique (by hash): {len(hashes)} (~{hashed_bytes / (1024 * 1024):.2f} MiB)")
    if rows_sorted:
        lines.append("Top attachments:")
        for row in rows_sorted:
            lines.append(f"  {row['name']} {row['size']} bytes ({row['path']})")
    ui.summary("Attachment Stats", lines)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _referenced_attachment_paths(rows: List[Dict[str, object]]) -> Set[Path]:
    referenced: Set[Path] = set()
    for row in rows:
        value = row.get("attachment_path")
        if not value:
            continue
        try:
            candidate = Path(str(value)).expanduser()
        except Exception:
            continue
        if not candidate.is_absolute():
            continue
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        referenced.add(resolved)
    return referenced


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for idx in range(1, 10000):
        variant = dest_dir / f"{stem}-{idx}{suffix}"
        if not variant.exists():
            return variant
    raise SystemExit(f"Too many filename collisions for {filename} in {dest_dir}")


def _run_attachment_extract(args: SimpleNamespace, env: CommandEnv) -> None:
    ui = env.ui
    roots = _collect_roots(getattr(args, "dir", None))
    if not roots:
        ui.console.print("[red]No output roots found for extraction.")
        raise SystemExit(1)
    ext = getattr(args, "ext", None)
    if not ext:
        ui.console.print("[red]--ext is required for extraction")
        raise SystemExit(1)
    ext_lower = ext.lower()
    destination = Path(args.out).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    limit = max(0, getattr(args, "limit", 0))
    overwrite = bool(getattr(args, "overwrite", False))
    json_lines = bool(getattr(args, "json_lines", False))
    json_mode = bool(getattr(args, "json", False) or json_lines)

    copied = 0
    errors = 0
    emitted: List[Dict[str, object]] = []
    for root in roots:
        for att_dir in root.rglob("attachments"):
            if not att_dir.is_dir():
                continue
            for file in att_dir.iterdir():
                if not file.is_file() or file.suffix.lower() != ext_lower:
                    continue
                target = destination / file.name
                if target.exists() and not overwrite:
                    target = _unique_destination(destination, file.name)
                try:
                    shutil.copy2(file, target)
                    copied += 1
                    if json_lines:
                        emitted.append(
                            {
                                "source": str(file),
                                "dest": str(target),
                                "sizeBytes": file.stat().st_size if file.exists() else None,
                            }
                        )
                except Exception:
                    errors += 1
                if limit and copied >= limit:
                    break
            if limit and copied >= limit:
                break
        if limit and copied >= limit:
            break

    if json_lines:
        for row in emitted:
            print(json.dumps(stamp_payload(row), separators=(",", ":")))
        return

    if json_mode:
        payload = stamp_payload(
            {
                "roots": [str(r) for r in roots],
                "copied": copied,
                "errors": errors,
                "destination": str(destination),
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    lines = [f"Roots: {', '.join(str(r) for r in roots)}", f"Copied: {copied}", f"Errors: {errors}", f"Destination: {destination}"]
    ui.summary("Attachment Extract", lines)


__all__ = ["run_attachments_cli"]
