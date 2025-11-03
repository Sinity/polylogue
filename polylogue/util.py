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

try:  # pragma: no cover - optional dependency
    import pyperclip  # type: ignore
    from pyperclip import PyperclipException  # type: ignore
except ImportError:  # pragma: no cover - clipboard support optional
    pyperclip = None

    class PyperclipException(Exception):
        pass


def _xdg_path(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser()
    return fallback


CONFIG_ROOT = _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")
DATA_ROOT = _xdg_path("XDG_DATA_HOME", Path.home() / ".local/share")
CACHE_ROOT = _xdg_path("XDG_CACHE_HOME", Path.home() / ".cache")
STATE_ROOT = _xdg_path("XDG_STATE_HOME", Path.home() / ".local/state")


CONFIG_HOME = CONFIG_ROOT / "polylogue"
DATA_HOME = DATA_ROOT / "polylogue"
CACHE_HOME = CACHE_ROOT / "polylogue"


STATE_HOME = STATE_ROOT / "polylogue"


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
STATE_PATH = STATE_HOME / "state.json"
RUNS_PATH = STATE_HOME / "runs.json"


class StateStore:
    """Lightweight JSON state persistence with injectable storage for tests."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        try:
            if self.path.exists():
                return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def save(self, state: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def mutate(self, mutator: Callable[[dict], None]) -> dict:
        state = self.load()
        mutator(state)
        self.save(state)
        return state


STATE_STORE = StateStore(STATE_PATH)


def get_state_store() -> StateStore:
    global STATE_STORE
    if isinstance(STATE_STORE, StateStore) and STATE_STORE.path != STATE_PATH:
        STATE_STORE = StateStore(STATE_PATH)
    return STATE_STORE


def configure_state_store(store: StateStore) -> None:
    global STATE_STORE
    STATE_STORE = store


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
    state = get_state_store().load()
    folders = state.get("folders") if isinstance(state.get("folders"), dict) else None
    if isinstance(folders, dict):
        value = folders.get(name)
        if isinstance(value, str):
            return value
    return None


def set_cached_folder_id(name: str, folder_id: str) -> None:
    def _mutator(state: dict) -> None:
        folders = state.get("folders")
        if not isinstance(folders, dict):
            folders = {}
            state["folders"] = folders
        folders[name] = folder_id

    get_state_store().mutate(_mutator)


def get_conversation_state(provider: str, conversation_id: str) -> Optional[Dict[str, Any]]:
    state = get_state_store().load()
    conversations = state.get("conversations")
    if not isinstance(conversations, dict):
        return None
    provider_map = conversations.get(provider)
    if not isinstance(provider_map, dict):
        return None
    entry = provider_map.get(conversation_id)
    return entry if isinstance(entry, dict) else None


def assign_conversation_slug(
    provider: str,
    conversation_id: str,
    title: Optional[str],
    *,
    id_hint: Optional[str] = None,
) -> str:
    chosen: Dict[str, Optional[str]] = {"value": None}

    def _mutator(state: dict) -> None:
        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            conversations = {}
            state["conversations"] = conversations
        provider_map = conversations.get(provider)
        if not isinstance(provider_map, dict):
            provider_map = {}
            conversations[provider] = provider_map

        entry = provider_map.get(conversation_id)
        if isinstance(entry, dict) and entry.get("slug"):
            chosen["value"] = entry["slug"]
            return

        base_source = title or id_hint or conversation_id or "conversation"
        base = slugify_title(base_source)
        if not base:
            fallback = sanitize_filename(base_source)
            base = fallback.replace(" ", "-") or (conversation_id[:8] if conversation_id else "conversation")

        existing_slugs = {
            data.get("slug")
            for data in provider_map.values()
            if isinstance(data, dict) and data.get("slug")
        }
        slug_candidate = base
        counter = 1
        while slug_candidate in existing_slugs:
            slug_candidate = f"{base}-{counter}"
            counter += 1

        provider_map[conversation_id] = {"slug": slug_candidate}
        chosen["value"] = slug_candidate

    get_state_store().mutate(_mutator)
    return chosen["value"] or "conversation"


def update_conversation_state(
    provider: str,
    conversation_id: str,
    **updates: Any,
) -> None:
    def _mutator(state: dict) -> None:
        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            conversations = {}
            state["conversations"] = conversations
        provider_map = conversations.get(provider)
        if not isinstance(provider_map, dict):
            provider_map = {}
            conversations[provider] = provider_map
        entry = provider_map.get(conversation_id)
        if not isinstance(entry, dict):
            entry = {}
        for key, value in updates.items():
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
        provider_map[conversation_id] = entry

    get_state_store().mutate(_mutator)


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
) -> bool:
    entry = get_conversation_state(provider, conversation_id)
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


def add_run(record: Dict[str, Any]) -> None:
    payload = dict(record)
    payload.setdefault("timestamp", current_utc_timestamp())
    try:
        if RUNS_PATH.exists():
            runs = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
            if not isinstance(runs, list):
                runs = []
        else:
            runs = []
        runs.append(payload)
        # Keep last 200
        runs = runs[-200:]
        RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUNS_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    except Exception:
        pass


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
