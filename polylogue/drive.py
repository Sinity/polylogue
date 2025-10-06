import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import time

from .util import colorize, get_cached_folder_id, set_cached_folder_id

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    HAS_GOOGLE = True
except Exception:
    HAS_GOOGLE = False


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"

def _retry(fn, *, retries: int = 3, base_delay: float = 0.5):
    # Allow env overrides for tuning
    try:
        retries = int(os.environ.get("POLYLOGUE_RETRIES", retries))
    except Exception:
        pass
    try:
        base_delay = float(os.environ.get("POLYLOGUE_RETRY_BASE", base_delay))
    except Exception:
        pass
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # HttpError or transport errors
            last_err = e
            time.sleep(base_delay * (2 ** i))
    if last_err:
        raise last_err
    return None


def get_drive_service(credentials_path: Path, verbose: bool = False):
    if not HAS_GOOGLE:
        print(colorize("Google API libraries not available.", "red"))
        return None
    creds = None
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")
    token_path = Path(token_env) if token_env else (credentials_path.parent / TOKEN_FILE)
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            if verbose:
                print(colorize(f"Loaded token from {token_path}", "magenta"))
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds:
            if not credentials_path.is_file():
                print(colorize(f"Missing credentials at {credentials_path}", "red"))
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            mode = os.environ.get("POLYLOGUE_AUTH_MODE", "console").lower()
            try:
                if mode == "local":
                    creds = flow.run_local_server(port=0)
                else:
                    creds = flow.run_console()
            except Exception:
                # fallback to console
                creds = flow.run_console()
            try:
                token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
            except Exception:
                pass
    try:
        return build("drive", "v3", credentials=creds)
    except HttpError:
        return None


def list_children(service, folder_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = _retry(lambda: service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, createdTime, size)",
            pageSize=1000,
            pageToken=page_token,
        ).execute())
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def find_folder_id(service, folder_name: str) -> Optional[str]:
    cached = get_cached_folder_id(folder_name)
    if cached:
        try:
            # Validate cached id exists
            meta = _retry(lambda: service.files().get(fileId=cached, fields="id").execute())
            if meta and meta.get("id"):
                return cached
        except Exception:
            pass
    # Exact name search, pick most recently modified
    try:
        resp = _retry(lambda: service.files().list(
            q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false",
            fields="files(id, name, modifiedTime)",
            pageSize=50,
        ).execute())
        files = resp.get("files", [])
        if not files:
            return None
        files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
        sel = files[0]["id"]
        set_cached_folder_id(folder_name, sel)
        return sel
    except HttpError:
        return None


def get_file_meta(service, file_id: str, fields: str = "id, name, mimeType, modifiedTime, createdTime, size") -> Optional[Dict[str, Any]]:
    try:
        return _retry(lambda: service.files().get(fileId=file_id, fields=fields).execute())
    except HttpError:
        return None


def download_file(service, file_id: str) -> Optional[bytes]:
    try:
        req = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            def _next():
                return downloader.next_chunk()
            _, done = _retry(_next)
        return buf.getvalue()
    except HttpError:
        return None


def download_to_path(service, file_id: str, target_path: Path) -> bool:
    data = download_file(service, file_id)
    if data is None:
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)
    return True
