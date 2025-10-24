import datetime
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:  # pragma: no cover - optional dependency
    import pyperclip  # type: ignore
    from pyperclip import PyperclipException  # type: ignore
except Exception:  # pragma: no cover
    pyperclip = None  # type: ignore

    class PyperclipException(Exception):
        pass


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
    return f"{colors.get(color, '')}{text}{colors['reset']}" if sys.stderr.isatty() else text


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
STATE_HOME = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "polylogue"
STATE_PATH = STATE_HOME / "state.json"
RUNS_PATH = STATE_HOME / "runs.json"


def _load_state() -> dict:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_cached_folder_id(name: str) -> Optional[str]:
    st = _load_state()
    return (st.get("folders") or {}).get(name)


def set_cached_folder_id(name: str, folder_id: str) -> None:
    st = _load_state()
    st.setdefault("folders", {})[name] = folder_id
    _save_state(st)


def get_conversation_state(provider: str, conversation_id: str) -> Optional[Dict[str, Any]]:
    st = _load_state()
    convs = st.get("conversations") if isinstance(st.get("conversations"), dict) else {}
    provider_map = convs.get(provider) if isinstance(convs.get(provider), dict) else {}
    entry = provider_map.get(conversation_id)
    if isinstance(entry, dict):
        return entry
    return None


def assign_conversation_slug(
    provider: str,
    conversation_id: str,
    title: Optional[str],
    *,
    id_hint: Optional[str] = None,
) -> str:
    st = _load_state()
    convs = st.setdefault("conversations", {})
    if not isinstance(convs, dict):
        convs = {}
        st["conversations"] = convs
    provider_map = convs.setdefault(provider, {})
    if not isinstance(provider_map, dict):
        provider_map = {}
        convs[provider] = provider_map

    entry = provider_map.get(conversation_id)
    if isinstance(entry, dict) and entry.get("slug"):
        return entry["slug"]

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
    slug = base
    counter = 1
    while slug in existing_slugs:
        slug = f"{base}-{counter}"
        counter += 1

    provider_map[conversation_id] = {"slug": slug}
    _save_state(st)
    return slug


def update_conversation_state(
    provider: str,
    conversation_id: str,
    **updates: Any,
) -> None:
    st = _load_state()
    convs = st.setdefault("conversations", {})
    if not isinstance(convs, dict):
        convs = {}
        st["conversations"] = convs
    provider_map = convs.setdefault(provider, {})
    if not isinstance(provider_map, dict):
        provider_map = {}
        convs[provider] = provider_map
    entry = provider_map.get(conversation_id)
    if not isinstance(entry, dict):
        entry = {}
    entry.update({k: v for k, v in updates.items() if v is not None})
    provider_map[conversation_id] = entry
    _save_state(st)


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
    try:
        result = subprocess.run(
            ["delta", str(old_path), str(new_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "").strip()
    except FileNotFoundError:
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
