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
from .paths import CONFIG_HOME
from .util import parse_rfc3339_to_epoch, read_clipboard_text
from .ui import UI

GDRIVE_INSTRUCTIONS = "https://developers.google.com/drive/api/quickstart/python"
CONFIG_DIR = CONFIG_HOME
DEFAULT_CREDENTIALS = CONFIG_DIR / "credentials.json"
DEFAULT_TOKEN = CONFIG_DIR / "token.json"
DEFAULT_FOLDER_NAME = "AI Studio"
DRIVE_CREDENTIAL_ENV = "POLYLOGUE_DRIVE_CREDENTIALS"


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
        env_copy = self._copy_credentials_from_env()
        if env_copy:
            return env_copy
        if self.ui.plain:
            raise SystemExit(
                f"Missing credentials.json. Set ${DRIVE_CREDENTIAL_ENV} or download a Google OAuth client secret "
                f"and place it at {cred_path}."
            )
        return self._prompt_for_credentials()

    def _copy_credentials_from_env(self) -> Optional[Path]:
        env_credential = os.environ.get(DRIVE_CREDENTIAL_ENV)
        if not env_credential:
            return None
        src = Path(env_credential).expanduser()
        if not src.exists():
            raise SystemExit(
                f"Credential path from ${DRIVE_CREDENTIAL_ENV} not found: {src}"
            )
        return self._copy_credentials_from_path(src)

    def _copy_credentials_from_path(self, src: Path) -> Path:
        cred_path = self._credentials_path
        try:
            cred_path.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != cred_path.resolve():
                shutil.copyfile(src, cred_path)
        except shutil.SameFileError:
            pass
        except Exception as exc:
            raise RuntimeError(f"Failed to copy credentials: {exc}") from exc
        return cred_path

    def _prompt_for_credentials(self) -> Path:
        self.ui.banner("Google Drive access needs credentials", "Download OAuth client for Desktop app")
        clipboard_checked = False
        while True:
            if not clipboard_checked:
                clipboard_checked = True
                clip_path = self._try_clipboard_credentials()
                if clip_path:
                    return clip_path
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
                cred_path = self._copy_credentials_from_path(src)
            except RuntimeError as exc:
                self.ui.banner("Failed to copy credentials", str(exc))
                continue
            self.ui.banner("Saved credentials", f"Copied to {cred_path}")
            return cred_path

    def _try_clipboard_credentials(self) -> Optional[Path]:
        clip_text = read_clipboard_text()
        if not clip_text:
            return None
        try:
            parsed = json.loads(clip_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not (set(parsed.keys()) & {"installed", "web"}):
            return None
        if not self.ui.confirm("Use OAuth client JSON from clipboard?", default=True):
            return None
        try:
            self._credentials_path.parent.mkdir(parents=True, exist_ok=True)
            self._credentials_path.write_text(clip_text, encoding="utf-8")
            self.ui.banner("Saved credentials", f"Captured from clipboard → {self._credentials_path}")
            return self._credentials_path
        except Exception as exc:  # pragma: no cover - I/O failures are rare
            self.ui.banner("Failed to write credentials", str(exc))
            return None

    def service(self):
        if self._service is None:
            cred_path = self.ensure_credentials()
            try:
                svc = get_drive_service(cred_path, verbose=not self.ui.plain)
            except RuntimeError as exc:
                message = str(exc)
                if self.ui.plain:
                    raise SystemExit(message) from exc
                self.ui.banner("Drive dependencies missing", message)
                raise SystemExit(message) from exc
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
        resolved = find_folder_id(svc, name, notifier=self._notify_retry)
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
        children = list_children(svc, fid, notifier=self._notify_retry)
        chats = [
            c
            for c in children
            if not c.get("mimeType", "").startswith("application/vnd.google-apps.")
        ]
        return chats

    def download_chat_bytes(self, file_id: str) -> Optional[bytes]:
        svc = self.service()
        return download_file(svc, file_id, operation="chat", notifier=self._notify_retry)

    def attachment_meta(self, file_id: str) -> Optional[Dict[str, Any]]:
        svc = self.service()
        return get_file_meta(svc, file_id, notifier=self._notify_retry)

    def download_attachment(self, file_id: str, path: Path) -> bool:
        svc = self.service()
        ok = download_to_path(svc, file_id, path, operation="attachment", notifier=self._notify_retry)
        if not ok:
            return False
        return True

    def touch_mtime(self, path: Path, iso_time: Optional[str]) -> None:
        mtime = parse_rfc3339_to_epoch(iso_time)
        if mtime is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
            os.utime(path, (mtime, mtime))
        except Exception:
            pass

    def _notify_retry(self, *, operation: str, attempt: int, total: int, error: Exception, delay: float) -> None:
        if self.ui.plain:
            return
        message = f"[yellow]Drive {operation} retry {attempt}/{total}: {error} (waiting {delay:.1f}s)"
        self.ui.console.print(message)
