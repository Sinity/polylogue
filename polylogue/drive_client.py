from __future__ import annotations

import json
import os
import shutil
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from .drive import (
    download_file,
    download_to_path,
    find_folder_id,
    get_drive_service,
    get_file_meta,
    list_children,
)
from .util import parse_rfc3339_to_epoch, read_clipboard_text
from .ui import UI

GDRIVE_INSTRUCTIONS = "https://developers.google.com/drive/api/quickstart/python"
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "polylogue"
DEFAULT_CREDENTIALS = CONFIG_DIR / "credentials.json"
DEFAULT_TOKEN = CONFIG_DIR / "token.json"
DEFAULT_FOLDER_NAME = "AI Studio"


class DriveClient:
    """Wrapper that manages credentials and Drive service access."""

    def __init__(self, ui: UI):
        self.ui = ui
        self._credentials_path: Path = DEFAULT_CREDENTIALS
        self._service = None

    @property
    def credentials_path(self) -> Path:
        return self._credentials_path

    def ensure_credentials(self) -> Path:
        cred_path = self._credentials_path
        if cred_path.exists():
            return cred_path
        if self.ui.plain:
            raise SystemExit(
                "Missing credentials.json. Download a Google OAuth client secret and place it next to polylogue.py."
            )
        self.ui.banner("Google Drive access needs credentials", "Download OAuth client for Desktop app")
        clipboard_checked = False
        while True:
            if not clipboard_checked:
                clipboard_checked = True
                clip_text = read_clipboard_text()
                if clip_text:
                    try:
                        parsed = json.loads(clip_text)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict) and (set(parsed.keys()) & {"installed", "web"}):
                        if self.ui.confirm("Use OAuth client JSON from clipboard?", default=True):
                            try:
                                cred_path.parent.mkdir(parents=True, exist_ok=True)
                                cred_path.write_text(clip_text, encoding="utf-8")
                                self.ui.banner("Saved credentials", f"Captured from clipboard → {cred_path}")
                                return cred_path
                            except Exception as exc:
                                self.ui.banner("Failed to write credentials", str(exc))
            choice = self.ui.choose(
                "Provide credentials",
                ["Paste path to credentials.json", "Open setup guide", "Cancel"],
            )
            if choice is None or choice == "Cancel":
                raise SystemExit("Drive features require credentials.json")
            if choice == "Open setup guide":
                webbrowser.open(GDRIVE_INSTRUCTIONS)
                continue
            value = self.ui.input("Path to OAuth client JSON", default=str(Path.cwd()))
            if not value:
                continue
            src = Path(value).expanduser()
            if src.is_dir():
                potential = src / "credentials.json"
                if potential.exists():
                    src = potential
            if not src.exists():
                self.ui.banner("File not found", str(src))
                continue
            try:
                cred_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, cred_path)
                self.ui.banner("Saved credentials", f"Copied to {cred_path}")
            except shutil.SameFileError:
                self.ui.banner("Using credentials", str(cred_path))
            except Exception as exc:
                self.ui.banner("Failed to copy credentials", str(exc))
                continue
            return cred_path

    def service(self):
        if self._service is None:
            cred_path = self.ensure_credentials()
            svc = get_drive_service(cred_path, verbose=not self.ui.plain)
            if not svc:
                raise SystemExit("Drive auth failed")
            self._service = svc
        return self._service

    # Convenience wrappers -------------------------------------------------
    def resolve_folder_id(self, folder_name: Optional[str], folder_id: Optional[str]) -> str:
        if folder_id:
            return folder_id
        name = folder_name or DEFAULT_FOLDER_NAME
        svc = self.service()
        resolved = find_folder_id(svc, name)
        if not resolved:
            raise SystemExit(f"Folder not found: {name}")
        return resolved

    def list_chats(
        self,
        folder_name: Optional[str],
        folder_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        svc = self.service()
        fid = self.resolve_folder_id(folder_name, folder_id)
        children = list_children(svc, fid)
        chats = [
            c
            for c in children
            if ("." not in (c.get("name") or ""))
            and not c.get("mimeType", "").startswith("application/vnd.google-apps.")
        ]
        return chats

    def download_chat_bytes(self, file_id: str) -> Optional[bytes]:
        svc = self.service()
        return download_file(svc, file_id)

    def attachment_meta(self, file_id: str) -> Optional[Dict[str, Any]]:
        svc = self.service()
        return get_file_meta(svc, file_id)

    def download_attachment(self, file_id: str, path: Path) -> bool:
        svc = self.service()
        ok = download_to_path(svc, file_id, path)
        if not ok:
            return False
        return True

    def touch_mtime(self, path: Path, iso_time: Optional[str]) -> None:
        mtime = parse_rfc3339_to_epoch(iso_time)
        if not mtime:
            return
        try:
            path.touch(exist_ok=True)
            path.stat()  # ensure exists
            import os

            os.utime(path, (mtime, mtime))
        except Exception:
            pass
