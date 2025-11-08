from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .archive import Archive
from .config import CONFIG
from .persistence.database import ConversationDatabase
from .persistence.state import ConversationStateRepository
from .util import STATE_HOME, RUNS_PATH, load_runs
from .services.conversation_service import ConversationService, create_conversation_service


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


_PROVIDER_ALIASES = {
    "drive-sync": "drive",
    "claude.ai": "claude",
}


def _drive_failure_issues() -> List[DoctorIssue]:
    data = load_runs()
    aggregates: Dict[str, Dict[str, Any]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        provider = entry.get("provider")
        if not provider:
            continue
        requests = int(entry.get("driveRequests", 0) or 0)
        failures = int(entry.get("driveFailures", 0) or 0)
        if requests == 0 and failures == 0:
            continue
        bucket = aggregates.setdefault(provider, {"requests": 0, "failures": 0, "last_error": None})
        bucket["requests"] += requests
        bucket["failures"] += failures
        last_error = entry.get("driveLastError")
        if isinstance(last_error, str) and last_error.strip():
            bucket["last_error"] = last_error.strip()

    issues: List[DoctorIssue] = []
    for provider, stats in aggregates.items():
        failures = stats["failures"]
        requests = stats["requests"] or 1
        if failures <= 0:
            continue
        rate = failures / requests
        severity = "error" if rate >= 0.5 else "warning"
        message = f"Drive retries/failures: {failures} out of {requests} requests ({rate:.0%})."
        if stats.get("last_error"):
            message += f" Last error: {stats['last_error']}"
        issues.append(DoctorIssue(provider, RUNS_PATH, message, severity))
    return issues


def _resolve_provider_root(archive: Archive, provider: str) -> Optional[Path]:
    canonical = _PROVIDER_ALIASES.get(provider, provider)
    try:
        return archive.provider_root(canonical)
    except Exception:
        return None


def prune_state_entries(
    state_repo: ConversationStateRepository,
    archive: Archive,
) -> Tuple[int, List[DoctorIssue]]:
    issues: List[DoctorIssue] = []
    state = state_repo.load()
    conversations = state.get("conversations")
    if not isinstance(conversations, dict):
        return 0, issues

    removed = 0
    for provider, conv_map in list(conversations.items()):
        if not isinstance(conv_map, dict):
            continue
        provider_root = _resolve_provider_root(archive, provider)
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
                candidates: List[Path] = []
                if conv_path.parent:
                    candidates.append(conv_path.parent.resolve())
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
                        issues.append(
                            DoctorIssue(
                                "state",
                                attachment_dir,
                                "Failed to remove stale attachment path",
                                "warning",
                            )
                        )
        if not conv_map:
            conversations.pop(provider, None)

    if removed:
        try:
            state_repo.store.save(state)
        except Exception as exc:  # pragma: no cover
            issues.append(DoctorIssue("state", state_repo.store.path, f"Failed to write state: {exc}", "error"))
        else:
            issues.append(DoctorIssue("state", state_repo.store.path, f"Removed {removed} stale entries", "info"))
    return removed, issues


def prune_database_entries(
    database: ConversationDatabase,
    state_repo: ConversationStateRepository,
    archive: Archive,
) -> Tuple[int, List[DoctorIssue]]:
    issues: List[DoctorIssue] = []
    to_delete: List[Tuple[str, str]] = []
    try:
        rows = database.query("SELECT provider, conversation_id, slug FROM conversations")
        for row in rows:
            provider = row["provider"]
            conversation_id = row["conversation_id"]
            slug = row["slug"]
            candidate_paths: List[Path] = []
            state_entry = state_repo.get(provider, conversation_id)
            if isinstance(state_entry, dict):
                raw_output = state_entry.get("outputPath")
                if isinstance(raw_output, str):
                    candidate_paths.append(Path(raw_output))
            provider_root = _resolve_provider_root(archive, provider)
            if provider_root is not None:
                candidate_paths.append(provider_root / slug / "conversation.md")
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
            database.execute(
                "DELETE FROM conversations WHERE provider = ? AND conversation_id = ?",
                (provider, conversation_id),
            )
    except Exception as exc:
        db_path = database.path or STATE_HOME / "polylogue.db"
        issues.append(DoctorIssue("database", db_path, str(exc), "error"))
        return 0, issues

    if to_delete:
        db_path = database.path or STATE_HOME / "polylogue.db"
        issues.append(DoctorIssue("database", db_path, f"Removed {len(to_delete)} stale entries", "info"))
    return len(to_delete), issues


def run_doctor(
    *,
    codex_dir: Path,
    claude_code_dir: Path,
    limit: Optional[int] = None,
    service: Optional[ConversationService] = None,
    archive: Optional[Archive] = None,
) -> DoctorReport:
    codex_report = inspect_codex(codex_dir, limit)
    claude_report = inspect_claude_code(claude_code_dir, limit)
    issues = codex_report.issues + claude_report.issues
    counts = {**codex_report.checked, **claude_report.checked}

    service = service or create_conversation_service()
    state_repo = service.state_repo
    database = service.database
    archive = archive or Archive(CONFIG)

    removed_state, state_issues = prune_state_entries(state_repo, archive)
    if removed_state:
        counts["state"] = removed_state
    issues.extend(state_issues)

    removed_db, db_issues = prune_database_entries(database, state_repo, archive)
    if removed_db:
        counts["database"] = removed_db
    issues.extend(db_issues)

    issues.extend(_drive_failure_issues())

    return DoctorReport(checked=counts, issues=issues)
