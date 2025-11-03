from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .util import colorize, get_cached_folder_id, set_cached_folder_id

try:
    import requests
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    HAS_GOOGLE = True
    GOOGLE_IMPORT_ERROR = None
except Exception as exc:
    HAS_GOOGLE = False
    GOOGLE_IMPORT_ERROR = exc


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
DRIVE_BASE = "https://www.googleapis.com/drive/v3"


class DriveApiError(RuntimeError):
    def __init__(self, message: str, status: int, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


def require_google():
    if HAS_GOOGLE:
        return
    raise RuntimeError(
        "Google Drive support requires google-auth and requests dependencies."
    ) from GOOGLE_IMPORT_ERROR


def _retry(fn, *, retries: int = 3, base_delay: float = 0.5):
    # Allow env overrides for tuning
    try:
        override = int(os.environ.get("POLYLOGUE_RETRIES", retries))
        if override >= 0:
            retries = override
    except Exception:
        pass
    try:
        base_delay = float(os.environ.get("POLYLOGUE_RETRY_BASE", base_delay))
    except Exception:
        pass
    base_delay = max(0.0, base_delay)
    retries = max(1, retries)
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** i))
    if last_err:
        raise last_err
    return None


def _raise_for_status(resp: requests.Response) -> None:
    if resp.status_code < 400:
        return
    message = f"HTTP {resp.status_code}"
    payload: Optional[Dict[str, Any]] = None
    try:
        payload = resp.json()
        msg = payload.get("error", {}).get("message")
        if isinstance(msg, str) and msg.strip():
            message = msg
    except Exception:
        text = resp.text.strip()
        if text:
            message = text
    raise DriveApiError(message, resp.status_code, payload)


def _authorized_session(creds: Credentials) -> AuthorizedSession:
    return AuthorizedSession(creds)


def _run_console_flow(flow: InstalledAppFlow, *, verbose: bool) -> Credentials:
    """Console-mode auth fallback when InstalledAppFlow.run_console is unavailable."""
    run_console = getattr(flow, "run_console", None)
    if callable(run_console):
        return run_console()

    auth_url, _ = flow.authorization_url(prompt="consent")
    prompt = (
        "Open this link in your browser to authorize Polylogue:\n"
        f"  {auth_url}\n"
    )
    print(colorize(prompt, "magenta") if verbose else prompt)

    try:
        code = input(
            colorize("Enter the authorization code: ", "cyan")
            if verbose
            else "Enter the authorization code: "
        ).strip()
    except EOFError as exc:
        raise RuntimeError("Authorization cancelled") from exc

    if not code:
        raise RuntimeError("Empty authorization code supplied.")

    flow.fetch_token(code=code)
    return flow.credentials


def _drive_get_json(session: AuthorizedSession, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{DRIVE_BASE}/{path}"

    def call():
        resp = session.get(url, params=params, timeout=60)
        try:
            _raise_for_status(resp)
            return resp.json()
        finally:
            resp.close()

    return _retry(call)


def get_drive_service(credentials_path: Path, verbose: bool = False):
    require_google()
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
                    creds = _run_console_flow(flow, verbose=verbose)
            except Exception:
                creds = _run_console_flow(flow, verbose=verbose)
            try:
                token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(token_path, "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
            except Exception:
                pass
    if not creds:
        return None
    try:
        return _authorized_session(creds)
    except Exception:
        return None


def list_children(session, folder_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, createdTime, size)",
        "pageSize": 1000,
    }
    while True:
        if page_token:
            params["pageToken"] = page_token
        elif "pageToken" in params:
            del params["pageToken"]
        resp = _drive_get_json(session, "files", params)
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def find_folder_id(session, folder_name: str) -> Optional[str]:
    cached = get_cached_folder_id(folder_name)
    if cached:
        try:
            meta = _drive_get_json(session, f"files/{cached}", {"fields": "id"})
            if meta and meta.get("id"):
                return cached
        except Exception:
            pass
    try:
        escaped_name = folder_name.replace("'", "\\'")
        resp = _drive_get_json(
            session,
            "files",
            {
                "q": f"mimeType='application/vnd.google-apps.folder' and name='{escaped_name}' and trashed=false",
                "fields": "files(id, name, modifiedTime)",
                "pageSize": 50,
            },
        )
        files = resp.get("files", [])
        if not files:
            return None
        files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
        sel = files[0]["id"]
        set_cached_folder_id(folder_name, sel)
        return sel
    except DriveApiError:
        return None


def get_file_meta(session, file_id: str, fields: str = "id, name, mimeType, modifiedTime, createdTime, size") -> Optional[Dict[str, Any]]:
    try:
        return _drive_get_json(session, f"files/{file_id}", {"fields": fields})
    except DriveApiError:
        return None


def download_file(session, file_id: str) -> Optional[bytes]:
    url = f"{DRIVE_BASE}/files/{file_id}"

    def fetch():
        resp = session.get(url, params={"alt": "media"}, timeout=120, stream=True)
        try:
            _raise_for_status(resp)
            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    buf.write(chunk)
            return buf.getvalue()
        finally:
            resp.close()

    try:
        return _retry(fetch)
    except DriveApiError:
        return None


def download_to_path(session, file_id: str, target_path: Path) -> bool:
    data = download_file(session, file_id)
    if data is None:
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)
    return True
