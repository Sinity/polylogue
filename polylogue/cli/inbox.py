"""Inbox coverage/quarantine helpers."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..commands import CommandEnv
from ..config import CONFIG
from ..local_sync import _detect_export_provider, _discover_export_targets
from ..schema import stamp_payload


def _parse_providers(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return values or None


def _dir_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def run_inbox_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    ui = env.ui
    provider_filter = _parse_providers(getattr(args, "providers", None))
    quarantine_enabled = bool(getattr(args, "quarantine", False))
    override_root: Optional[Path] = getattr(args, "dir", None)

    roots: List[Tuple[Optional[str], Path]] = []
    if override_root:
        roots.append((None, Path(override_root).expanduser()))
    else:
        roots.append(("chatgpt", CONFIG.exports.chatgpt))
        roots.append(("claude", CONFIG.exports.claude))

    entries: List[Dict[str, object]] = []
    quarantined: List[str] = []
    missing_roots = 0

    for provider, root in roots:
        if not root.exists():
            missing_roots += 1
            continue
        candidates = _discover_export_targets(root, provider=provider)
        for cand in candidates:
            detected = _detect_export_provider(cand)
            entry_provider = detected or provider or "unknown"
            if provider_filter and entry_provider != "unknown" and entry_provider.lower() not in provider_filter:
                continue
            size_bytes = _dir_size(cand)
            try:
                mtime = cand.stat().st_mtime
                modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                modified_at = None
            status = "pending"
            target_path = cand
            if quarantine_enabled and detected is None:
                quarantine_dir = getattr(args, "quarantine_dir", None)
                qdir = Path(quarantine_dir).expanduser() if quarantine_dir else root / "quarantine"
                qdir.mkdir(parents=True, exist_ok=True)
                dest = qdir / cand.name
                shutil.move(str(cand), dest)
                status = "quarantined"
                target_path = dest
                quarantined.append(str(dest))
            entries.append(
                {
                    "provider": entry_provider,
                    "path": str(target_path),
                    "sourcePath": str(cand),
                    "sizeBytes": size_bytes,
                    "modifiedAt": modified_at,
                    "status": status,
                }
            )

    if getattr(args, "json", False):
        payload = stamp_payload(
            {
                "entries": entries,
                "quarantined": quarantined,
                "missingRoots": missing_roots,
            }
        )
        print(json.dumps(payload, indent=2))
        return

    if missing_roots == len(roots):
        ui.console.print("[red]No inbox roots found (set exports.chatgpt/exports.claude or pass --dir).")
        raise SystemExit(1)

    summary: Dict[str, int] = {}
    for entry in entries:
        prov = str(entry.get("provider") or "unknown")
        summary[prov] = summary.get(prov, 0) + 1

    lines = []
    for prov, count in sorted(summary.items()):
        lines.append(f"{prov}: {count} pending")
    if quarantined:
        lines.append(f"Quarantined: {len(quarantined)} item(s)")
    if not entries and not quarantined:
        lines.append("No inbox items found.")
    ui.summary("Inbox", lines)


__all__ = ["run_inbox_cli"]
