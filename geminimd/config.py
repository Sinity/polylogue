import os
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None


DEFAULTS: Dict[str, Any] = {
    "folder_name": "AI Studio",
    "credentials": str(Path("credentials.json").resolve()),
    "collapse_threshold": 25,
    "out_dir_render": str(Path("gmd_out").resolve()),
    "out_dir_sync": str(Path("gemini_synced").resolve()),
    "remote_links": False,
}


def find_conf_path() -> Path:
    # Project-local config preferred: search from CWD upwards
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        cand = p / ".gmdrc"
        if cand.exists():
            return cand
    return Path.cwd() / ".gmdrc"  # default location to create


def default_conf_text() -> str:
    return (
        "# gmd configuration (TOML). Project-local; checked into your repo if desired.\n"
        "# Lines starting with # are comments.\n"
        "# Drive folder name to sync (ID is auto-discovered and cached).\n"
        f"folder_name = \"{DEFAULTS['folder_name']}\"\n\n"
        "# Path to Google OAuth credentials.json\n"
        f"credentials = \"{DEFAULTS['credentials']}\"\n\n"
        "# Fold model responses longer than this many lines (0 disables).\n"
        f"collapse_threshold = {DEFAULTS['collapse_threshold']}\n\n"
        "# Default output directories.\n"
        f"out_dir_render = \"{DEFAULTS['out_dir_render']}\"\n"
        f"out_dir_sync = \"{DEFAULTS['out_dir_sync']}\"\n\n"
        "# If true, never download attachments; link to Drive URLs.\n"
        f"remote_links = {str(DEFAULTS['remote_links']).lower()}\n"
    )


def load_config() -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    conf_path = find_conf_path()
    # Load TOML if available
    try:
        if conf_path.exists() and tomllib is not None:
            data = tomllib.loads(conf_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update(data)
    except Exception:
        pass
    # Env overrides
    env_map = {
        "GMD_FOLDER_NAME": "folder_name",
        "GMD_CREDENTIALS": "credentials",
        "GMD_COLLAPSE_THRESHOLD": "collapse_threshold",
        "GMD_OUT_DIR_RENDER": "out_dir_render",
        "GMD_OUT_DIR_SYNC": "out_dir_sync",
        "GMD_REMOTE_LINKS": "remote_links",
    }
    for env_key, cfg_key in env_map.items():
        if env_key in os.environ:
            val = os.environ[env_key]
            if cfg_key == "collapse_threshold":
                try:
                    cfg[cfg_key] = int(val)
                except ValueError:
                    pass
            elif cfg_key == "remote_links":
                cfg[cfg_key] = val.lower() in ("1", "true", "yes", "on")
            else:
                cfg[cfg_key] = val
    return cfg

