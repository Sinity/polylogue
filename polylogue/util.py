import datetime
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .db import open_connection, record_run
from .paths import CACHE_HOME, CONFIG_HOME, DATA_HOME, STATE_HOME
from .persistence.state import ConversationStateRepository

try:  # pragma: no cover - optional dependency
    import pyperclip  # type: ignore
    from pyperclip import PyperclipException  # type: ignore
except ImportError:  # pragma: no cover - clipboard support optional
    pyperclip = None

    class PyperclipException(Exception):
        pass


CODEX_SESSIONS_ROOT = Path(
    os.environ.get("POLYLOGUE_CODEX_SESSIONS", str(DATA_HOME / "codex" / "sessions"))
).expanduser()


CLAUDE_CODE_PROJECT_ROOT = Path(
    os.environ.get("POLYLOGUE_CLAUDE_CODE_PROJECTS", str(DATA_HOME / "claude" / "projects"))
).expanduser()


def colorize(text: str, color: str) -> str:
    colors = {
        "reset": "\033[0m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "grey": "\033[90m",
    }
    if os.environ.get("NO_COLOR") or os.environ.get("POLYLOGUE_NO_COLOR"):
        return text
    stream = sys.stdout if sys.stdout.isatty() else sys.stderr
    if not stream.isatty():
        return text
    prefix = colors.get(color, "")
    return f"{prefix}{text}{colors['reset']}" if prefix else text


def sanitize_filename(filename: str) -> str:
    sanitized = "".join(c for c in filename if ord(c) >= 32)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = sanitized.strip(". ")
    max_len = 200
    encoded = sanitized.encode("utf-8")
    if len(encoded) > max_len:
        sanitized = encoded[:max_len].decode("utf-8", errors="ignore")
    if not sanitized:
        sanitized = "_unnamed_"
    return sanitized


def path_order_key(path: Path) -> Tuple[float, str]:
    try:
        stat_result = path.stat()
        return (stat_result.st_mtime, str(path))
    except OSError:
        return (0.0, str(path))


def parse_rfc3339_to_epoch(ts: Optional[str]) -> Optional[float]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def parse_input_time_to_epoch(s: Optional[Union[str, float, int, datetime.date, datetime.datetime]]) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        try:
            return float(s)
        except Exception:
            return None
    if isinstance(s, datetime.datetime):
        dt = s
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    if isinstance(s, datetime.date):
        dt = datetime.datetime.combine(s, datetime.time.min, tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    if not isinstance(s, str):
        return None
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            dt = datetime.datetime.fromisoformat(s)
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


# Simple cache for discovered Drive IDs
_STATE_REPO: Optional[ConversationStateRepository] = None


def _state_repository() -> ConversationStateRepository:
    global _STATE_REPO
    if _STATE_REPO is None:
        _STATE_REPO = ConversationStateRepository()
    return _STATE_REPO


def _get_meta_value(key: str) -> Optional[str]:
    with open_connection(None) as conn:
        row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _set_meta_value(key: str, value: str) -> None:
    with open_connection(None) as conn:
        conn.execute(
            """
            INSERT INTO state_meta(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        conn.commit()


def load_runs(limit: Optional[int] = None) -> list[dict]:
    sql = "SELECT * FROM runs ORDER BY id DESC"
    params: Tuple[Any, ...] = ()
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params = (limit,)
    with open_connection(None) as conn:
        rows = conn.execute(sql, params).fetchall()
    results: list[dict] = []
    for row in rows:
        payload = {
            "timestamp": row["timestamp"],
            "cmd": row["cmd"],
            "count": row["count"],
            "attachments": row["attachments"],
            "attachment_bytes": row["attachment_bytes"],
            "tokens": row["tokens"],
            "skipped": row["skipped"],
            "pruned": row["pruned"],
            "diffs": row["diffs"],
            "duration": row["duration"],
            "out": row["out"],
            "provider": row["provider"],
            "branch_id": row["branch_id"],
        }
        metadata = row["metadata_json"]
        if metadata:
            try:
                payload.update(json.loads(metadata))
            except Exception:
                pass
        results.append(payload)
    results.reverse()  # restore chronological order
    return results


class DiffTracker:
    """Manage pre/post snapshots when diff output is requested."""

    def __init__(self, path: Path, enabled: bool):
        self._enabled = enabled
        self._snapshot = snapshot_for_diff(path) if enabled else None

    def finalize(self, new_path: Path) -> Optional[Path]:
        if not self._enabled or self._snapshot is None:
            return None
        try:
            return write_delta_diff(self._snapshot, new_path)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._snapshot is None:
            return
        try:
            if self._snapshot.exists():
                self._snapshot.unlink()
        except Exception:
            pass
        finally:
            self._snapshot = None


class RunAccumulator:
    """Collect numeric statistics across processed conversations."""

    def __init__(self) -> None:
        self._totals: Dict[str, int] = defaultdict(int)

    def add_stats(self, attachments: int, stats: Dict[str, Any]) -> None:
        self._totals["attachments"] += int(attachments)
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self._totals[key] += int(value)

    def increment(self, key: str, value: int = 1) -> None:
        self._totals[key] += value

    def totals(self) -> Dict[str, int]:
        totals = dict(self._totals)
        totals.setdefault("attachments", 0)
        totals.setdefault("skipped", 0)
        totals.setdefault("diffs", 0)
        return totals


def get_cached_folder_id(name: str) -> Optional[str]:
    return _get_meta_value(f"drive.folder.{name}")


def set_cached_folder_id(name: str, folder_id: str) -> None:
    _set_meta_value(f"drive.folder.{name}", folder_id)

def assign_conversation_slug(
    provider: str,
    conversation_id: str,
    title: Optional[str],
    *,
    id_hint: Optional[str] = None,
) -> str:
    repo = _state_repository()
    existing = repo.get(provider, conversation_id)
    if existing and isinstance(existing.get("slug"), str):
        return str(existing["slug"])

    base_source = title or id_hint or conversation_id or "conversation"
    base = slugify_title(base_source)
    if not base:
        fallback = sanitize_filename(base_source)
        base = fallback.replace(" ", "-") or (conversation_id[:8] if conversation_id else "conversation")

    with open_connection(repo.database.resolve_path()) as conn:
        rows = conn.execute(
            "SELECT slug FROM conversations WHERE provider = ? AND slug IS NOT NULL",
            (provider,),
        ).fetchall()
    existing_slugs = {row["slug"] for row in rows if row["slug"]}
    slug_candidate = base
    counter = 1
    while slug_candidate in existing_slugs:
        slug_candidate = f"{base}-{counter}"
        counter += 1

    repo.upsert(provider, conversation_id, {"slug": slug_candidate})
    return slug_candidate


def slugify_title(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    slug = slug.strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug


def conversation_is_current(
    provider: str,
    conversation_id: str,
    *,
    updated_at: Optional[str],
    content_hash: Optional[str],
    output_path: Optional[Path],
    collapse_threshold: Optional[int] = None,
    attachment_policy: Optional[Dict[str, Any]] = None,
    html: Optional[bool] = None,
    dirty: Optional[bool] = None,
    entry: Optional[Dict[str, Any]] = None,
) -> bool:
    if not entry:
        return False
    if dirty:
        return False
    if entry.get("dirty"):
        return False
    if updated_at and entry.get("lastUpdated") and entry["lastUpdated"] != updated_at:
        return False
    if content_hash and entry.get("contentHash") and entry["contentHash"] != content_hash:
        return False
    if output_path and not output_path.exists():
        return False
    if collapse_threshold is not None:
        if entry.get("collapseThreshold") != collapse_threshold:
            return False
    if attachment_policy is not None:
        stored_policy = entry.get("attachmentPolicy")
        if stored_policy != attachment_policy:
            return False
    if html is not None:
        stored_html = bool(entry.get("html"))
        if stored_html != bool(html):
            return False
        if html:
            html_path = entry.get("htmlPath")
            if html_path and not Path(html_path).exists():
                return False
    attachments_dir = entry.get("attachmentsDir")
    if attachments_dir:
        try:
            if not Path(attachments_dir).exists():
                return False
        except Exception:
            return False
    return True


def _run_log_enabled() -> bool:
    flag = os.environ.get("POLYLOGUE_RUN_LOG")
    if flag is None:
        return True
    return str(flag).strip().lower() not in {"0", "false", "off", "no"}


def _emit_structured_run_log(entry: Dict[str, Any]) -> None:
    if not _run_log_enabled():  # pragma: no cover - optional emission
        return
    try:
        serialized = json.dumps(entry, default=str, ensure_ascii=False)
    except Exception:
        serialized = json.dumps({k: str(v) for k, v in entry.items()}, ensure_ascii=False)
    stream = sys.stderr
    stream.write(serialized + "\n")
    stream.flush()


def add_run(record: Dict[str, Any]) -> None:
    payload = dict(record)
    timestamp = payload.get("timestamp") or current_utc_timestamp()
    cmd = payload.get("cmd") or "unknown"
    count = int(payload.get("count") or 0)
    attachments = int(payload.get("attachments") or 0)
    attachment_bytes = int(payload.get("attachment_bytes") or 0)
    tokens = int(payload.get("tokens") or 0)
    skipped = int(payload.get("skipped") or 0)
    pruned = int(payload.get("pruned") or 0)
    diffs = int(payload.get("diffs") or 0)
    duration = payload.get("duration")
    out = payload.get("out")
    provider = payload.get("provider")
    branch_id = payload.get("branch_id")
    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "timestamp",
            "cmd",
            "count",
            "attachments",
            "attachment_bytes",
            "tokens",
            "skipped",
            "pruned",
            "diffs",
            "duration",
            "out",
            "provider",
            "branch_id",
        }
    }
    with open_connection(None) as conn:
        record_run(
            conn,
            timestamp=timestamp,
            cmd=cmd,
            count=count,
            attachments=attachments,
            attachment_bytes=attachment_bytes,
            tokens=tokens,
            skipped=skipped,
            pruned=pruned,
            diffs=diffs,
            duration=float(duration) if isinstance(duration, (int, float)) else None,
            out=str(out) if out is not None else None,
            provider=str(provider) if provider is not None else None,
            branch_id=str(branch_id) if branch_id is not None else None,
            metadata=metadata or None,
        )
        conn.commit()

    log_entry = {
        "event": "polylogue_run",
        "timestamp": timestamp,
        "cmd": cmd,
        "provider": provider,
        "count": count,
        "attachments": attachments,
        "attachment_bytes": attachment_bytes,
        "tokens": tokens,
        "skipped": skipped,
        "pruned": pruned,
        "diffs": diffs,
        "duration": float(duration) if isinstance(duration, (int, float)) else None,
        "out": str(out) if out is not None else None,
        "retries": metadata.get("driveRetries") or metadata.get("retries") or 0,
        "failures": metadata.get("driveFailures") or metadata.get("failures") or 0,
        "last_error": metadata.get("driveLastError") or metadata.get("lastError"),
        "metadata": metadata or None,
    }
    # Remove None values for terser logs
    log_entry = {key: value for key, value in log_entry.items() if value not in (None, "")}
    _emit_structured_run_log(log_entry)


def write_delta_diff(old_path: Path, new_path: Path, *, suffix: str = ".diff.txt") -> Optional[Path]:
    if not old_path.exists() or not new_path.exists():
        return None
    diff_path = new_path.with_suffix(new_path.suffix + suffix)
    result = subprocess.run(
        ["delta", str(old_path), str(new_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        output = (result.stdout or "").strip()
    else:
        output = ""

    if not output:
        try:
            with old_path.open(encoding="utf-8") as old_f, new_path.open(encoding="utf-8") as new_f:
                diff = difflib.unified_diff(
                    old_f.readlines(),
                    new_f.readlines(),
                    fromfile=str(old_path),
                    tofile=str(new_path),
                )
                output = "".join(diff)
        except Exception:
            output = ""

    if not output.strip():
        if diff_path.exists():
            try:
                diff_path.unlink()
            except OSError:
                pass
        return None

    diff_path.write_text(output, encoding="utf-8")
    return diff_path


def snapshot_for_diff(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix)
    try:
        tmp.write(path.read_bytes())
    finally:
        tmp.close()
    return Path(tmp.name)


def current_utc_timestamp() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_clipboard_text() -> Optional[str]:
    if pyperclip is None:
        return None
    try:
        text = pyperclip.paste()  # type: ignore[attr-defined]
    except PyperclipException:  # pragma: no cover - clipboard backend unavailable
        return None
    if not text:
        return None
    return text


def write_clipboard_text(text: str) -> bool:
    if pyperclip is None:
        return False
    try:
        pyperclip.copy(text)  # type: ignore[attr-defined]
        return True
    except PyperclipException:  # pragma: no cover - clipboard backend unavailable
        return False
