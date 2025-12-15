"""Inbox coverage/quarantine helpers."""

from __future__ import annotations

import fnmatch
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from ..commands import CommandEnv
from ..config import CONFIG
from ..local_sync import _detect_export_provider
from ..schema import stamp_payload


def _load_ignore_patterns(base_dir: Path) -> List[str]:
    path = base_dir / ".polylogueignore"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    patterns: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _is_ignored(path: Path, base_dir: Path, patterns: List[str]) -> bool:
    if not patterns:
        return False
    try:
        rel = path.relative_to(base_dir)
        candidate = rel.as_posix()
    except ValueError:
        candidate = path.name
    return any(fnmatch.fnmatch(candidate, pat) for pat in patterns)


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


_MALFORMED_ERROR_KINDS = {
    "not-a-zip",
    "missing-conversations-json",
    "invalid-conversations-json",
    "invalid-jsonl",
}


def _inspect_inbox_candidate(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort inbox inspection: returns (provider, error_kind)."""
    suffix = path.suffix.lower()
    conv_path: Optional[Path] = None
    raw: Optional[object] = None

    if path.is_dir():
        candidate = path / "conversations.json"
        if candidate.exists():
            conv_path = candidate
    elif path.is_file() and path.name.lower() == "conversations.json":
        conv_path = path
    elif path.is_file() and suffix == ".zip":
        if not zipfile.is_zipfile(path):
            return None, "not-a-zip"
        try:
            with zipfile.ZipFile(path) as zf:
                if "conversations.json" not in zf.namelist():
                    return None, "missing-conversations-json"
                with zf.open("conversations.json") as fh:
                    raw = json.loads(fh.read().decode("utf-8"))
        except Exception:
            return None, "invalid-conversations-json"
    elif path.is_file() and suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as handle:
                first = None
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        first = stripped
                        break
        except Exception:
            return None, "invalid-jsonl"
        if not first:
            return None, "invalid-jsonl"
        try:
            obj = json.loads(first)
        except Exception:
            return None, "invalid-jsonl"
        if isinstance(obj, dict):
            entry_type = (obj.get("type") or "").strip().lower()
            if entry_type in {"session_meta", "response_item", "request", "response"}:
                return "codex", None
            if entry_type in {"user", "assistant", "system"} and isinstance(obj.get("message"), dict):
                return "claude-code", None
        return None, "unknown-jsonl"

    if raw is None and conv_path and conv_path.exists():
        try:
            raw = json.loads(conv_path.read_text(encoding="utf-8"))
        except Exception:
            return None, "invalid-conversations-json"

    if raw is None:
        provider = _detect_export_provider(path)
        return provider, None if provider else "unknown-format"

    provider = _detect_export_provider(path)
    if provider:
        return provider, None
    return None, "unknown-format"


def run_inbox_cli(args: SimpleNamespace, env: CommandEnv) -> None:
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
    ignored_by_rule = 0
    malformed_by_reason: Dict[str, int] = {}
    malformed = 0
    unrecognized_by_reason: Dict[str, int] = {}
    unrecognized = 0
    totals = {"pending": 0, "quarantined": 0}

    for provider, root in roots:
        if not root.exists():
            missing_roots += 1
            continue
        ignore_patterns = _load_ignore_patterns(root)
        candidates: set[Path] = set()
        try:
            for zip_path in root.rglob("*.zip"):
                candidates.add(zip_path)
        except OSError:
            pass
        try:
            for jsonl_path in root.rglob("*.jsonl"):
                candidates.add(jsonl_path)
        except OSError:
            pass
        try:
            for conv_file in root.rglob("conversations.json"):
                candidates.add(conv_file.parent)
        except OSError:
            pass

        for cand in sorted(candidates):
            if _is_ignored(cand, root, ignore_patterns):
                ignored_by_rule += 1
                continue

            detected, error_kind = _inspect_inbox_candidate(cand)
            entry_provider = detected or provider or "unknown"
            entry_provider_lower = entry_provider.lower()
            if provider_filter and entry_provider_lower != "unknown" and entry_provider_lower not in provider_filter:
                continue
            if error_kind:
                if error_kind in _MALFORMED_ERROR_KINDS:
                    malformed += 1
                    malformed_by_reason[error_kind] = malformed_by_reason.get(error_kind, 0) + 1
                else:
                    unrecognized += 1
                    unrecognized_by_reason[error_kind] = unrecognized_by_reason.get(error_kind, 0) + 1

            size_bytes = _dir_size(cand)
            try:
                mtime = cand.stat().st_mtime
                modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                modified_at = None
            status = "pending"
            target_path = cand
            if quarantine_enabled and (detected is None or error_kind):
                quarantine_dir = getattr(args, "quarantine_dir", None)
                qdir = Path(quarantine_dir).expanduser() if quarantine_dir else root / "quarantine"
                qdir.mkdir(parents=True, exist_ok=True)
                dest = qdir / cand.name
                shutil.move(str(cand), dest)
                status = "quarantined"
                target_path = dest
                quarantined.append(str(dest))
                totals["quarantined"] += 1
            entries.append(
                {
                    "provider": entry_provider,
                    "detectedProvider": detected,
                    "path": str(target_path),
                    "sourcePath": str(cand),
                    "sizeBytes": size_bytes,
                    "modifiedAt": modified_at,
                    "status": status,
                    "errorKind": error_kind,
                }
            )
            if status == "pending":
                totals["pending"] += 1

    if getattr(args, "json", False):
        payload = stamp_payload(
            {
                "entries": entries,
                "quarantined": quarantined,
                "ignoredByRule": ignored_by_rule,
                "malformed": malformed,
                "malformedByReason": dict(sorted(malformed_by_reason.items())),
                "unrecognized": unrecognized,
                "unrecognizedByReason": dict(sorted(unrecognized_by_reason.items())),
                "missingRoots": missing_roots,
                "totals": totals,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
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
    if ignored_by_rule:
        lines.append(f"Skipped by .polylogueignore: {ignored_by_rule} item(s)")
    if malformed:
        lines.append(f"Malformed exports/JSONL: {malformed} item(s)")
    if unrecognized:
        lines.append(f"Unrecognized inbox items: {unrecognized} item(s)")
    if not entries and not quarantined:
        lines.append("No inbox items found.")
    else:
        lines.append(f"Pending total: {totals['pending']}")
    ui.summary("Inbox", lines)


__all__ = ["run_inbox_cli"]
