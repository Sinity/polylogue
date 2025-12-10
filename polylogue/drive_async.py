"""Async Google Drive client using httpx for parallel downloads."""

from __future__ import annotations

import asyncio
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
    import aiofiles
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
        "Google Drive support requires google-auth, httpx, and aiofiles dependencies."
    ) from GOOGLE_IMPORT_ERROR


async def _retry_async(
    fn,
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    operation: str = "request",
    notifier=None,
):
    """Async retry wrapper with exponential backoff."""
    base_delay = max(0.0, base_delay)
    retries = max(1, retries)
    last_err = None
    attempts = 0
    for i in range(retries):
        attempts += 1
        try:
            result = await fn()
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
            await asyncio.sleep(delay)
    if last_err:
        raise last_err
    return None


def _raise_for_status(resp: httpx.Response) -> None:
    """Raise DriveApiError for non-2xx responses."""
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


class AsyncDriveClient:
    """Async Google Drive client with parallel download support."""

    def __init__(self, credentials: Credentials, max_concurrent: int = 10):
        """Initialize async Drive client.

        Args:
            credentials: Google OAuth2 credentials
            max_concurrent: Maximum number of concurrent downloads (default: 10)
        """
        self.credentials = credentials
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers, refreshing token if needed."""
        if not self.credentials.valid:
            if self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
        return {"Authorization": f"Bearer {self.credentials.token}"}

    async def _drive_get_json(
        self,
        client: httpx.AsyncClient,
        path: str,
        params: Dict[str, Any],
        *,
        notifier=None,
    ) -> Dict[str, Any]:
        """Make async GET request to Drive API and return JSON."""
        url = f"{DRIVE_BASE}{path}"

        async def call():
            headers = await self._get_headers()
            resp = await client.get(url, params=params, headers=headers, timeout=60.0)
            _raise_for_status(resp)
            return resp.json()

        return await _retry_async(call, operation=f"get:{path}", notifier=notifier)

    async def list_children(
        self,
        client: httpx.AsyncClient,
        folder_id: str,
        *,
        notifier=None,
    ) -> List[Dict[str, Any]]:
        """List all files in a Drive folder."""
        children = []
        page_token = None
        while True:
            params = {
                "q": f"'{folder_id}' in parents",
                "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime)",
                "pageSize": 1000,
            }
            if page_token:
                params["pageToken"] = page_token
            data = await self._drive_get_json(client, "/files", params, notifier=notifier)
            children.extend(data.get("files", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return children

    async def find_folder_id(
        self,
        client: httpx.AsyncClient,
        folder_name: str,
        *,
        notifier=None,
    ) -> Optional[str]:
        """Find folder ID by name."""
        cached = get_cached_folder_id(folder_name)
        if cached:
            return cached

        params = {
            "q": f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            "fields": "files(id,name)",
            "pageSize": 10,
        }
        data = await self._drive_get_json(client, "/files", params, notifier=notifier)
        files = data.get("files", [])
        if not files:
            return None

        folder_id = files[0]["id"]
        set_cached_folder_id(folder_name, folder_id)
        return folder_id

    async def get_file_meta(
        self,
        client: httpx.AsyncClient,
        file_id: str,
        *,
        operation: str = "metadata",
        notifier=None,
    ) -> Dict[str, Any]:
        """Get file metadata."""
        params = {"fields": "id,name,mimeType,size,modifiedTime,parents"}
        return await self._drive_get_json(client, f"/files/{file_id}", params, notifier=notifier)

    async def download_file(
        self,
        client: httpx.AsyncClient,
        file_id: str,
        *,
        operation: str = "download",
        notifier=None,
    ) -> Optional[bytes]:
        """Download file content as bytes with rate limiting."""
        async with self._semaphore:
            url = f"{DRIVE_BASE}/files/{file_id}"

            async def fetch():
                headers = await self._get_headers()
                resp = await client.get(
                    url,
                    params={"alt": "media"},
                    headers=headers,
                    timeout=120.0,
                )
                _raise_for_status(resp)
                return resp.content

            return await _retry_async(fetch, operation=operation, notifier=notifier)

    async def download_to_path(
        self,
        client: httpx.AsyncClient,
        file_id: str,
        target_path: Path,
        *,
        operation: str = "download",
        notifier=None,
    ) -> bool:
        """Download file to disk using aiofiles."""
        async with self._semaphore:
            url = f"{DRIVE_BASE}/files/{file_id}"

            async def fetch_and_write():
                headers = await self._get_headers()
                async with client.stream(
                    "GET",
                    url,
                    params={"alt": "media"},
                    headers=headers,
                    timeout=120.0,
                ) as resp:
                    _raise_for_status(resp)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(target_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=8192):
                            await f.write(chunk)
                return True

            return await _retry_async(fetch_and_write, operation=operation, notifier=notifier)

    async def download_batch(
        self,
        file_ids: List[str],
        *,
        operation: str = "batch_download",
        notifier=None,
    ) -> List[Optional[bytes]]:
        """Download multiple files in parallel.

        Args:
            file_ids: List of Drive file IDs to download
            operation: Operation name for metrics
            notifier: Optional callback for progress updates

        Returns:
            List of file contents (bytes), same order as file_ids
        """
        async with httpx.AsyncClient() as client:
            tasks = [
                self.download_file(client, file_id, operation=operation, notifier=notifier)
                for file_id in file_ids
            ]
            return await asyncio.gather(*tasks)

    async def download_batch_to_paths(
        self,
        downloads: List[tuple[str, Path]],
        *,
        operation: str = "batch_download",
        notifier=None,
    ) -> List[bool]:
        """Download multiple files to disk in parallel.

        Args:
            downloads: List of (file_id, target_path) tuples
            operation: Operation name for metrics
            notifier: Optional callback for progress updates

        Returns:
            List of booleans indicating success for each download
        """
        async with httpx.AsyncClient() as client:
            tasks = [
                self.download_to_path(client, file_id, path, operation=operation, notifier=notifier)
                for file_id, path in downloads
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)


def get_drive_service(credentials_path: Path, token_path: Optional[Path] = None, verbose: bool = False):
    """Get Drive credentials (sync, for compatibility)."""
    require_google()

    if token_path and token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    else:
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0) if not verbose else flow.run_console()

        if token_path:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, "w") as f:
                f.write(creds.to_json())

    return creds


def create_async_client(
    credentials_path: Path,
    token_path: Optional[Path] = None,
    max_concurrent: int = 10,
    verbose: bool = False,
) -> AsyncDriveClient:
    """Create async Drive client.

    Args:
        credentials_path: Path to credentials.json
        token_path: Path to token.json (optional)
        max_concurrent: Maximum concurrent downloads (default: 10)
        verbose: Enable verbose output

    Returns:
        AsyncDriveClient instance
    """
    creds = get_drive_service(credentials_path, token_path, verbose)
    return AsyncDriveClient(creds, max_concurrent=max_concurrent)
