from __future__ import annotations

import io
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

from .util import colorize, get_cached_folder_id, set_cached_folder_id

try:
    import httpx
    from google.auth.transport.requests import Request
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


@dataclass
class DriveMetrics:
    requests: int = 0
    retries: int = 0
    failures: int = 0
    last_error: Optional[str] = None
    operations: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"requests": 0, "retries": 0, "failures": 0})
    )


_DRIVE_METRICS = DriveMetrics()


def _record_drive_completion(operation: str, attempts: int, error: Optional[Exception] = None) -> None:
    global _DRIVE_METRICS
    _DRIVE_METRICS.requests += 1
    op_stats = _DRIVE_METRICS.operations[operation]
    op_stats["requests"] += 1
    if attempts > 1:
        retries = attempts - 1
        _DRIVE_METRICS.retries += retries
        op_stats["retries"] += retries
    if error is not None:
        _DRIVE_METRICS.failures += 1
        op_stats["failures"] += 1
        _DRIVE_METRICS.last_error = str(error)


def snapshot_drive_metrics(*, reset: bool = False) -> Dict[str, Any]:
    global _DRIVE_METRICS
    snapshot = {
        "requests": _DRIVE_METRICS.requests,
        "retries": _DRIVE_METRICS.retries,
        "failures": _DRIVE_METRICS.failures,
        "lastError": _DRIVE_METRICS.last_error,
        "operations": {k: dict(v) for k, v in _DRIVE_METRICS.operations.items()},
    }
    if reset:
        _DRIVE_METRICS = DriveMetrics()
    return snapshot


class DriveApiError(RuntimeError):
    def __init__(self, message: str, status: int, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


def require_google():
    if HAS_GOOGLE:
        return
    raise RuntimeError(
        "Google Drive support requires google-auth and httpx dependencies."
    ) from GOOGLE_IMPORT_ERROR


def _retry(
    fn,
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    operation: str = "request",
    notifier=None,
):
    base_delay = max(0.0, base_delay)
    retries = max(1, retries)
    last_err = None
    attempts = 0
    for i in range(retries):
        attempts += 1
        try:
            result = fn()
            _record_drive_completion(operation, attempts)
            return result
        except Exception as e:
            last_err = e
            delay = base_delay * (2 ** i)
            if notifier:
                try:
                    notifier(operation=operation, attempt=attempts, total=retries, error=e, delay=delay)
                except Exception:
                    pass
            if i == retries - 1:
                _record_drive_completion(operation, attempts, error=e)
                raise
            time.sleep(delay)
    if last_err:
        raise last_err
    return None


def _raise_for_status(resp: httpx.Response) -> None:
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


class AuthorizedClient:
    """httpx.Client wrapper that automatically adds OAuth2 authorization headers."""

    def __init__(self, credentials: Credentials):
        self.credentials = credentials
        self.client = httpx.Client()

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers, refreshing token if needed."""
        if not self.credentials.valid:
            if self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
        return {"Authorization": f"Bearer {self.credentials.token}"}

    def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request with authorization."""
        headers = self._get_headers()
        if "headers" in kwargs:
            kwargs["headers"].update(headers)
        else:
            kwargs["headers"] = headers
        return self.client.get(url, **kwargs)

    def close(self) -> None:
        """Close the underlying httpx client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _authorized_session(creds: Credentials) -> AuthorizedClient:
    return AuthorizedClient(creds)


def _run_console_flow(flow: InstalledAppFlow, *, verbose: bool) -> Credentials:
    """Console-mode auth fallback when InstalledAppFlow.run_console is unavailable."""
    run_console = getattr(flow, "run_console", None)
    if callable(run_console):
        return run_console()

    redirect_uri = getattr(flow, "redirect_uri", None)
    if not redirect_uri:
        client_config = getattr(flow, "client_config", {}) or {}
        redirect_candidates = client_config.get("redirect_uris") or []
        redirect_uri = redirect_candidates[0] if redirect_candidates else "urn:ietf:wg:oauth:2.0:oob"
        flow.redirect_uri = redirect_uri
    session = getattr(flow, "oauth2session", None)
    if session is not None and getattr(session, "redirect_uri", None) != flow.redirect_uri:
        session.redirect_uri = flow.redirect_uri

    auth_url, _ = flow.authorization_url(prompt="consent", redirect_uri=flow.redirect_uri)
    prompt = (
        "Open this link in your browser to authorize Polylogue:\n"
        f"  {auth_url}\n"
    )
    print(colorize(prompt, "magenta") if verbose else prompt)
    if not sys.stdin.isatty():
        raise RuntimeError("Authorization requires an interactive terminal; rerun with a TTY.")

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


def _drive_get_json(session: AuthorizedClient, path: str, params: Dict[str, Any], *, notifier=None) -> Dict[str, Any]:
    url = f"{DRIVE_BASE}/{path}"

    def call():
        resp = session.get(url, params=params, timeout=60)
        _raise_for_status(resp)
        return resp.json()

    return _retry(call, operation="metadata", notifier=notifier)


def get_drive_service(credentials_path: Path, token_path: Optional[Path] = None, verbose: bool = False):
    require_google()
    creds = None
    token_path = token_path or credentials_path.parent / TOKEN_FILE
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


def list_children(session, folder_id: str, *, notifier=None) -> List[Dict[str, Any]]:
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
        resp = _drive_get_json(session, "files", params, notifier=notifier)
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def find_folder_id(session, folder_name: str, *, notifier=None) -> Optional[str]:
    cached = get_cached_folder_id(folder_name)
    if cached:
        try:
            meta = _drive_get_json(session, f"files/{cached}", {"fields": "id"}, notifier=notifier)
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


def get_file_meta(
    session,
    file_id: str,
    fields: str = "id, name, mimeType, modifiedTime, createdTime, size",
    *,
    notifier=None,
) -> Optional[Dict[str, Any]]:
    try:
        return _drive_get_json(session, f"files/{file_id}", {"fields": fields}, notifier=notifier)
    except DriveApiError:
        return None


def download_file(session, file_id: str, *, operation: str = "download", notifier=None) -> Optional[bytes]:
    url = f"{DRIVE_BASE}/files/{file_id}"

    def fetch():
        with session.get(url, params={"alt": "media"}, timeout=120) as resp:
            _raise_for_status(resp)
            buf = io.BytesIO()
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                buf.write(chunk)
            return buf.getvalue()

    try:
        return _retry(fetch, operation=operation, notifier=notifier)
    except DriveApiError:
        return None


def download_to_path(session, file_id: str, target_path: Path, *, operation: str = "download", notifier=None) -> bool:
    data = download_file(session, file_id, operation=operation, notifier=notifier)
    if data is None:
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)
    return True
