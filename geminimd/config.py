import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "folder_name": "AI Studio",
    "credentials": str(Path("credentials.json").resolve()),
    "collapse_threshold": 10,
    "out_dir_render": str(Path("gmd_out").resolve()),
    "out_dir_sync": str(Path("gemini_synced").resolve()),
    "remote_links": False,
}


CONF_PATH = Path.home() / ".gmdrc"


def load_config() -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    # Load file if present (JSON)
    try:
        if CONF_PATH.exists():
            data = json.loads(CONF_PATH.read_text(encoding="utf-8"))
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

