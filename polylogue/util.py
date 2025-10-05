import json
import os
import re
import sys
import datetime
import difflib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Any, Dict


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
    if os.environ.get("NO_COLOR") or os.environ.get("POLYLOGUE_NO_COLOR") or os.environ.get("GMD_NO_COLOR"):
        return text
    return f"{colors.get(color, '')}{text}{colors['reset']}" if sys.stderr.isatty() else text


def sanitize_filename(filename: str) -> str:
    sanitized = "".join(c for c in filename if ord(c) >= 32)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", sanitized)
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


def parse_input_time_to_epoch(s: Optional[str]) -> Optional[float]:
    if not s:
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
STATE_PATH = Path.home() / ".polylogue_state.json"
RUNS_PATH = Path.home() / ".polylogue_runs.json"


def _load_state() -> dict:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(state: dict) -> None:
    try:
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


def add_run(record: Dict[str, Any]) -> None:
    payload = dict(record)
    payload.setdefault("timestamp", datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
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
