from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import os

from .archive import Archive
from .config import CONFIG, CONFIG_ENV, CONFIG_PATH, DEFAULT_PATHS
from .db import default_db_path
from .drive_client import DEFAULT_CREDENTIALS, DEFAULT_TOKEN
from .paths import STATE_HOME
from .persistence.database import ConversationDatabase
from .persistence.state import ConversationStateRepository
from .util import load_runs
from .services.conversation_service import ConversationService, create_conversation_service
from .index_health import verify_qdrant_collection, verify_sqlite_indexes


@dataclass
class DoctorIssue:
    provider: str
    path: Path
    message: str
    severity: str = "error"
    hint: Optional[str] = None


@dataclass
class DoctorReport:
    checked: Dict[str, int]
    issues: List[DoctorIssue]
    credential_path: Path = DEFAULT_CREDENTIALS
    token_path: Path = DEFAULT_TOKEN
    credential_env: Optional[str] = None
    token_env: Optional[str] = None
    credentials_present: bool = False
    token_present: bool = False


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


ProgressCallback = Callable[[int, Optional[int]], None]


def _iter_jsonl_paths(
    base_dir: Path,
    *,
    limit: Optional[int],
    progress: Optional[ProgressCallback],
) -> Tuple[Iterable[Path], Optional[int]]:
    total = limit if limit is not None else None
    if progress:
        progress(0, total)
    return base_dir.rglob("*.jsonl"), total


def inspect_codex(base_dir: Path, limit: Optional[int], progress: Optional[ProgressCallback] = None) -> DoctorReport:
    issues: List[DoctorIssue] = []
    checked = 0
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        issues.append(DoctorIssue("codex", base_dir, "Codex sessions directory missing", "warning"))
        return DoctorReport({"codex": 0}, issues)
    paths, total = _iter_jsonl_paths(base_dir, limit=limit, progress=progress)
    for path in paths:
        if limit is not None and checked >= limit:
            break
        checked += 1
        error = _check_jsonl(path)
        if error:
            issues.append(DoctorIssue("codex", path, error))
        if progress:
            progress(checked, total)
    if progress and (total is None or (total and checked < total)):
        progress(checked, checked)
    return DoctorReport({"codex": checked}, issues)


def inspect_claude_code(
    base_dir: Path,
    limit: Optional[int],
    progress: Optional[ProgressCallback] = None,
) -> DoctorReport:
    issues: List[DoctorIssue] = []
    checked = 0
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        issues.append(DoctorIssue("claude-code", base_dir, "Claude Code projects directory missing", "warning"))
        return DoctorReport({"claude-code": 0}, issues)
    paths, total = _iter_jsonl_paths(base_dir, limit=limit, progress=progress)
    for path in paths:
        if limit is not None and checked >= limit:
            break
        checked += 1
        error = _check_jsonl(path)
        if error:
            issues.append(DoctorIssue("claude-code", path, error))
        if progress:
            progress(checked, total)
    if progress and (total is None or (total and checked < total)):
        progress(checked, checked)
    issues.extend(_claude_code_retention_issues())
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
        issues.append(DoctorIssue(provider, STATE_HOME / "polylogue.db", message, severity))
    return issues


def _claude_code_retention_issues() -> List[DoctorIssue]:
    issues: List[DoctorIssue] = []
    config_root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    settings_path = config_root / "claude" / "settings.json"
    if not settings_path.exists():
        issues.append(
            DoctorIssue(
                "claude-code",
                settings_path,
                "Claude Code settings.json missing; default cleanup keeps history for 30 days.",
                "warning",
                hint="Add cleanupPeriodDays to settings.json to disable auto-deletion (e.g., 99999).",
            )
        )
        return issues
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(
            DoctorIssue(
                "claude-code",
                settings_path,
                f"Claude Code settings.json unreadable: {exc}",
                "warning",
                hint="Fix settings.json and ensure cleanupPeriodDays is set to a large value.",
            )
        )
        return issues
    cleanup_days = data.get("cleanupPeriodDays")
    if cleanup_days is None:
        issues.append(
            DoctorIssue(
                "claude-code",
                settings_path,
                "cleanupPeriodDays not set; default cleanup keeps history for 30 days.",
                "warning",
                hint="Set cleanupPeriodDays to a large value (e.g., 99999) to disable auto-deletion.",
            )
        )
        return issues
    try:
        cleanup_int = int(cleanup_days)
    except (TypeError, ValueError):
        issues.append(
            DoctorIssue(
                "claude-code",
                settings_path,
                f"cleanupPeriodDays is not an integer: {cleanup_days!r}.",
                "warning",
                hint="Set cleanupPeriodDays to a large integer (e.g., 99999).",
            )
        )
        return issues
    if cleanup_int < 90:
        issues.append(
            DoctorIssue(
                "claude-code",
                settings_path,
                f"cleanupPeriodDays is set to {cleanup_int} days; history may be pruned.",
                "warning",
                hint="Increase cleanupPeriodDays to retain local history (e.g., 99999).",
            )
        )
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
    removed = 0
    for provider in state_repo.providers():
        provider_root = _resolve_provider_root(archive, provider)
        for conversation_id, entry in state_repo.provider_items(provider):
            output_path = entry.get("outputPath") if isinstance(entry, dict) else None
            if not output_path:
                continue
            conv_path = Path(output_path)
            if conv_path.exists():
                continue
            state_repo.remove(provider, conversation_id)
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
    if removed:
        issues.append(DoctorIssue("state", state_repo.path, f"Removed {removed} stale entries", "info"))
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


def _dependency_issues() -> List[DoctorIssue]:
    return []


def _config_issues() -> List[DoctorIssue]:
    issues: List[DoctorIssue] = []
    config_target = CONFIG_PATH if CONFIG_PATH else (DEFAULT_PATHS[0] if DEFAULT_PATHS else STATE_HOME)
    if CONFIG_PATH is None or not CONFIG_PATH.exists():
        candidates = ", ".join(str(path) for path in DEFAULT_PATHS)
        hint = f"Copy docs/polylogue.config.sample.jsonc to one of [{candidates}] or set ${CONFIG_ENV}."
        issues.append(
            DoctorIssue(
                "config",
                Path(config_target),
                "Polylogue config not found.",
                "warning",
                hint=hint,
            )
        )
    return issues


def _credential_issues() -> List[DoctorIssue]:
    issues: List[DoctorIssue] = []
    if not DEFAULT_CREDENTIALS.exists():
        issues.append(
            DoctorIssue(
                "drive",
                DEFAULT_CREDENTIALS,
                "Google Drive credentials missing.",
                "warning",
                hint="Run `polylogue sync drive` and follow the OAuth prompt to save credentials.json.",
            )
        )
    if not DEFAULT_TOKEN.exists():
        issues.append(
            DoctorIssue(
                "drive",
                DEFAULT_TOKEN,
                "Drive OAuth token missing; next sync will request authorization.",
                "info",
                hint="Allow the OAuth consent flow during the next Drive sync.",
            )
        )
    return issues


def run_doctor(
    *,
    codex_dir: Path,
    claude_code_dir: Path,
    limit: Optional[int] = None,
    service: Optional[ConversationService] = None,
    archive: Optional[Archive] = None,
    progress: Optional[Callable[[str, int, Optional[int]], None]] = None,
) -> DoctorReport:
    credential_env = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")
    credential_path = DEFAULT_CREDENTIALS
    token_path = DEFAULT_TOKEN

    def _notify(provider: str, checked: int, total: Optional[int]) -> None:
        if progress:
            progress(provider, checked, total)

    codex_report = inspect_codex(
        codex_dir,
        limit,
        progress=(lambda checked, total: _notify("codex", checked, total)) if progress else None,
    )
    claude_report = inspect_claude_code(
        claude_code_dir,
        limit,
        progress=(lambda checked, total: _notify("claude-code", checked, total)) if progress else None,
    )
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

    issues.extend(_dependency_issues())
    issues.extend(_config_issues())
    issues.extend(_credential_issues())
    issues.extend(_drive_failure_issues())
    try:
        sqlite_notes = verify_sqlite_indexes(default_db_path())
        if sqlite_notes:
            counts["indexes"] = counts.get("indexes", 0) + len(sqlite_notes)
            for note in sqlite_notes:
                issues.append(DoctorIssue("index", default_db_path(), note, "info"))
    except Exception as exc:
        issues.append(DoctorIssue("index", default_db_path(), str(exc), "error"))
    try:
        qdrant_notes = verify_qdrant_collection()
        if qdrant_notes:
            for note in qdrant_notes:
                issues.append(DoctorIssue("qdrant", Path("qdrant"), note, "info"))
    except RuntimeError as exc:
        if os.environ.get("POLYLOGUE_INDEX_BACKEND", "sqlite").strip().lower() == "qdrant":
            issues.append(DoctorIssue("qdrant", Path("qdrant"), str(exc), "error"))

    return DoctorReport(
        checked=counts,
        issues=issues,
        credential_path=credential_path,
        token_path=token_path,
        credential_env=credential_env,
        token_env=token_env,
        credentials_present=credential_path.exists(),
        token_present=token_path.exists(),
    )
