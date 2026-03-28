from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import CONFIG
from .db import open_connection
from .util import STATE_PATH, STATE_HOME, get_conversation_state


@dataclass
class DoctorIssue:
    provider: str
    path: Path
    message: str
    severity: str = "error"


@dataclass
class DoctorReport:
    checked: Dict[str, int]
    issues: List[DoctorIssue]


def _check_jsonl(path: Path, max_lines: int = 3) -> Optional[str]:
    try:
        with path.open(encoding="utf-8") as handle:
            for _ in range(max_lines):
                line = handle.readline()
                if not line.strip():
                    continue
                json.loads(line)
                break
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"
    except Exception as exc:  # pragma: no cover
        return str(exc)
    return None


def inspect_codex(base_dir: Path, limit: Optional[int]) -> DoctorReport:
    issues: List[DoctorIssue] = []
    checked = 0
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        issues.append(DoctorIssue("codex", base_dir, "Codex sessions directory missing", "warning"))
        return DoctorReport({"codex": 0}, issues)
    for path in sorted(base_dir.rglob("*.jsonl")):
        if limit is not None and checked >= limit:
            break
        checked += 1
        error = _check_jsonl(path)
        if error:
            issues.append(DoctorIssue("codex", path, error))
    return DoctorReport({"codex": checked}, issues)


def inspect_claude_code(base_dir: Path, limit: Optional[int]) -> DoctorReport:
    issues: List[DoctorIssue] = []
    checked = 0
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        issues.append(DoctorIssue("claude-code", base_dir, "Claude Code projects directory missing", "warning"))
        return DoctorReport({"claude-code": 0}, issues)
    for path in sorted(base_dir.rglob("*.jsonl")):
        if limit is not None and checked >= limit:
            break
        checked += 1
        error = _check_jsonl(path)
        if error:
            issues.append(DoctorIssue("claude-code", path, error))
    return DoctorReport({"claude-code": checked}, issues)


def _provider_output_root(provider: str) -> Optional[Path]:
    defaults = CONFIG.defaults.output_dirs
    mapping = {
        "render": defaults.render,
        "drive-sync": defaults.sync_drive,
        "codex": defaults.sync_codex,
        "claude-code": defaults.sync_claude_code,
        "chatgpt": defaults.import_chatgpt,
        "claude.ai": defaults.import_claude,
    }
    return mapping.get(provider)


def prune_state_entries() -> Tuple[int, List[DoctorIssue]]:
    issues: List[DoctorIssue] = []
    if not STATE_PATH.exists():
        return 0, issues
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(DoctorIssue("state", STATE_PATH, f"Invalid JSON: {exc}", "error"))
        return 0, issues
    except Exception as exc:  # pragma: no cover
        issues.append(DoctorIssue("state", STATE_PATH, str(exc), "error"))
        return 0, issues

    conversations = data.get("conversations")
    if not isinstance(conversations, dict):
        return 0, issues

    removed = 0
    for provider, conv_map in list(conversations.items()):
        provider_root = _provider_output_root(provider)
        if not isinstance(conv_map, dict):
            continue
        for conversation_id, entry in list(conv_map.items()):
            if not isinstance(entry, dict):
                conv_map.pop(conversation_id, None)
                removed += 1
                continue
            output_path = entry.get("outputPath")
            if not output_path:
                continue
            conv_path = Path(output_path)
            if conv_path.exists():
                continue
            conv_map.pop(conversation_id, None)
            removed += 1
            attachment_dir = entry.get("attachmentsDir")
            if attachment_dir:
                attachment_dir = Path(attachment_dir)
                allowed = False
                attachment_parent = conv_path.parent
                candidates: List[Path] = []
                if attachment_parent:
                    candidates.append(attachment_parent.resolve())
                if provider_root:
                    try:
                        candidates.append(provider_root.resolve())
                    except Exception:
                        pass
                try:
                    attachment_real = attachment_dir.resolve()
                except Exception:
                    attachment_real = attachment_dir
                for base in candidates:
                    try:
                        attachment_real.relative_to(base)
                        allowed = True
                        break
                    except ValueError:
                        continue
                if not allowed and attachment_dir.exists():
                    issues.append(
                        DoctorIssue(
                            "state",
                            attachment_dir,
                            "Skipped removing attachment path outside managed directories",
                            "warning",
                        )
                    )
                    continue
                if attachment_dir.exists():
                    try:
                        if attachment_dir.is_dir():
                            shutil.rmtree(attachment_dir)
                        else:
                            attachment_dir.unlink()
                    except Exception:
                        issues.append(DoctorIssue("state", attachment_dir, "Failed to remove stale attachment path", "warning"))
        if not conv_map:
            conversations.pop(provider, None)

    if removed:
        try:
            STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            issues.append(DoctorIssue("state", STATE_PATH, f"Failed to write state: {exc}", "error"))
        else:
            issues.append(DoctorIssue("state", STATE_PATH, f"Removed {removed} stale entries", "info"))
    return removed, issues


def prune_database_entries() -> Tuple[int, List[DoctorIssue]]:
    issues: List[DoctorIssue] = []
    to_delete: List[Tuple[str, str]] = []
    try:
        with open_connection() as conn:
            rows = conn.execute(
                "SELECT provider, conversation_id, slug FROM conversations"
            ).fetchall()
            for row in rows:
                provider = row["provider"]
                conversation_id = row["conversation_id"]
                slug = row["slug"]
                candidate_paths: List[Path] = []
                state_entry = get_conversation_state(provider, conversation_id)
                output_path = None
                if isinstance(state_entry, dict):
                    raw_output = state_entry.get("outputPath")
                    if isinstance(raw_output, str):
                        output_path = Path(raw_output)
                        candidate_paths.append(output_path)
                root = _provider_output_root(provider)
                if root is not None:
                    candidate_paths.append(root / slug / "conversation.md")
                exists = False
                for path in candidate_paths:
                    try:
                        if path.exists():
                            exists = True
                            break
                    except Exception:
                        continue
                if not exists:
                    to_delete.append((provider, conversation_id))
            for provider, conversation_id in to_delete:
                conn.execute(
                    "DELETE FROM conversations WHERE provider = ? AND conversation_id = ?",
                    (provider, conversation_id),
                )
            if to_delete:
                conn.commit()
    except Exception as exc:
        issues.append(DoctorIssue("database", STATE_HOME / "polylogue.db", str(exc), "error"))
        return 0, issues

    if to_delete:
        issues.append(DoctorIssue("database", STATE_HOME / "polylogue.db", f"Removed {len(to_delete)} stale entries", "info"))
    return len(to_delete), issues


def run_doctor(
    *,
    codex_dir: Path,
    claude_code_dir: Path,
    limit: Optional[int] = None,
) -> DoctorReport:
    codex_report = inspect_codex(codex_dir, limit)
    claude_report = inspect_claude_code(claude_code_dir, limit)
    issues = codex_report.issues + claude_report.issues
    counts = {**codex_report.checked, **claude_report.checked}

    removed_state, state_issues = prune_state_entries()
    if removed_state:
        counts["state"] = removed_state
    issues.extend(state_issues)

    removed_db, db_issues = prune_database_entries()
    if removed_db:
        counts["database"] = removed_db
    issues.extend(db_issues)

    return DoctorReport(checked=counts, issues=issues)
