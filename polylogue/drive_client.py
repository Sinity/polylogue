from __future__ import annotations

import importlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .paths import CONFIG_HOME


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DEFAULT_CREDENTIALS_NAME = "credentials.json"
DEFAULT_TOKEN_NAME = "token.json"
DEFAULT_DRIVE_RETRIES = 3
DEFAULT_DRIVE_RETRY_BASE = 0.5
ENV_DRIVE_RETRIES = "POLYLOGUE_DRIVE_RETRIES"
ENV_DRIVE_RETRY_BASE = "POLYLOGUE_DRIVE_RETRY_BASE"


class DriveError(RuntimeError):
    pass


class DriveAuthError(DriveError):
    pass


class DriveNotFoundError(DriveError):
    pass


@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: Optional[str]
    size_bytes: Optional[int]


def default_credentials_path() -> Path:
    return CONFIG_HOME / DEFAULT_CREDENTIALS_NAME


def default_token_path() -> Path:
    return CONFIG_HOME / DEFAULT_TOKEN_NAME


def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise DriveAuthError(
            "Drive dependencies are not available. "
            "Install google-api-python-client + google-auth-oauthlib "
            "or run Polylogue from a Nix build/dev shell."
        ) from exc


def _parse_modified_time(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _parse_size(raw: Optional[str | int]) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _looks_like_id(value: str) -> bool:
    if not value or " " in value:
        return False
    return all(ch.isalnum() or ch in "-_" for ch in value)


def _resolve_credentials_path(ui: Optional[object]) -> Path:
    env_path = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    if env_path:
        return Path(env_path).expanduser()
    default_path = default_credentials_path()
    if default_path.exists():
        return default_path
    if ui is not None and not getattr(ui, "plain", True):
        prompt = f"Path to Google OAuth client JSON (default {default_path}):"
        response = ui.input(prompt, default=str(default_path))
        if response:
            candidate = Path(response).expanduser()
            if candidate.exists():
                default_path.parent.mkdir(parents=True, exist_ok=True)
                if candidate != default_path:
                    shutil.copy(candidate, default_path)
                return default_path
    raise DriveAuthError(
        f"Drive credentials not found. Set POLYLOGUE_CREDENTIAL_PATH or place a client JSON at {default_path}."
    )


def _resolve_token_path() -> Path:
    env_path = os.environ.get("POLYLOGUE_TOKEN_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return default_token_path()


def _resolve_retries(value: Optional[int]) -> int:
    if value is not None:
        return max(0, int(value))
    env_value = os.environ.get(ENV_DRIVE_RETRIES)
    if env_value:
        try:
            return max(0, int(env_value))
        except ValueError:
            pass
    return DEFAULT_DRIVE_RETRIES


def _resolve_retry_base(value: Optional[float]) -> float:
    if value is not None:
        return max(0.0, float(value))
    env_value = os.environ.get(ENV_DRIVE_RETRY_BASE)
    if env_value:
        try:
            return max(0.0, float(env_value))
        except ValueError:
            pass
    return DEFAULT_DRIVE_RETRY_BASE


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (DriveAuthError, DriveNotFoundError)):
        return False
    return True


class DriveClient:
    def __init__(
        self,
        *,
        ui: Optional[object] = None,
        credentials_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
        retries: Optional[int] = None,
        retry_base: Optional[float] = None,
    ) -> None:
        self._ui = ui
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._service = None
        self._meta_cache: Dict[str, DriveFile] = {}
        self._retries = _resolve_retries(retries)
        self._retry_base = _resolve_retry_base(retry_base)

    def _call_with_retry(self, func, *args, **kwargs):
        from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type
        
        retryer = Retrying(
            stop=stop_after_attempt(max(self._retries, 0) + 1),
            wait=wait_exponential(multiplier=self._retry_base, min=self._retry_base, max=10),
            retry=retry_if_exception_type(Exception) & ~retry_if_exception_type((DriveAuthError, DriveNotFoundError)),
            reraise=True,
        )
        return retryer(func, *args, **kwargs)

    def _load_credentials(self):
        Request = _import_module("google.auth.transport.requests").Request
        Credentials = _import_module("google.oauth2.credentials").Credentials
        InstalledAppFlow = _import_module("google_auth_oauthlib.flow").InstalledAppFlow

        token_path = self._token_path or _resolve_token_path()
        creds = None
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception:
                creds = None
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if creds and creds.valid:
            self._persist_token(creds, token_path)
            return creds
        if creds is None and token_path.exists() and (self._ui is None or getattr(self._ui, "plain", True)):
            raise DriveAuthError(
                f"Drive token at {token_path} is invalid or expired. "
                "Delete it and re-run with --interactive to re-authorize."
            )

        credentials_path = self._credentials_path or _resolve_credentials_path(self._ui)
        if not credentials_path.exists():
            raise DriveAuthError(f"Drive credentials not found: {credentials_path}")
        if self._ui is None or getattr(self._ui, "plain", True):
            raise DriveAuthError(
                "Drive authorization required but no interactive UI is available. "
                "Run with --interactive or set POLYLOGUE_TOKEN_PATH with a valid token."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        try:
            creds = flow.run_local_server(open_browser=False, port=0)
        except Exception:
            auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
            self._ui.console.print("Open this URL in your browser to authorize Drive access:")
            self._ui.console.print(auth_url)
            code = self._ui.input("Paste the authorization code")
            if not code:
                raise DriveAuthError("Drive authorization cancelled.")
            try:
                flow.fetch_token(code=code)
            except Exception as exc:
                raise DriveAuthError(f"Drive authorization failed: {exc}") from exc
            creds = flow.credentials
        self._persist_token(creds, token_path)
        return creds

    def _persist_token(self, creds, token_path: Path) -> None:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    def _service_handle(self):
        if self._service is not None:
            return self._service
        build = _import_module("googleapiclient.discovery").build

        creds = self._load_credentials()
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def resolve_folder_id(self, folder_ref: str) -> str:
        service = self._service_handle()
        if _looks_like_id(folder_ref):
            try:
                file_meta = self._call_with_retry(
                    lambda: service.files()
                    .get(fileId=folder_ref, fields="id,name,mimeType")
                    .execute()
                )
                if file_meta and file_meta.get("mimeType") == FOLDER_MIME_TYPE:
                    return file_meta["id"]
            except Exception:
                pass
        escaped = folder_ref.replace("'", "\\'")
        query = f"name = '{escaped}' and mimeType = '{FOLDER_MIME_TYPE}' and trashed = false"
        response = self._call_with_retry(
            lambda: service.files().list(q=query, fields="files(id,name)").execute()
        )
        matches = response.get("files", [])
        if not matches:
            raise DriveNotFoundError(f"Folder not found: {folder_ref}")
        return matches[0]["id"]

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        service = self._service_handle()
        page_token = None
        query = f"'{folder_id}' in parents and trashed = false"
        fields = "nextPageToken, files(id,name,mimeType,modifiedTime,size)"
        while True:
            response = self._call_with_retry(
                lambda: service.files()
                .list(q=query, fields=fields, pageToken=page_token, pageSize=1000)
                .execute()
            )
            for item in response.get("files", []):
                name = item.get("name") or ""
                if not name.lower().endswith((".json", ".jsonl")):
                    continue
                file_obj = DriveFile(
                    file_id=item.get("id", ""),
                    name=name,
                    mime_type=item.get("mimeType") or "",
                    modified_time=item.get("modifiedTime"),
                    size_bytes=_parse_size(item.get("size")),
                )
                if file_obj.file_id:
                    self._meta_cache[file_obj.file_id] = file_obj
                    yield file_obj
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def download_bytes(self, file_id: str) -> bytes:
        MediaIoBaseDownload = _import_module("googleapiclient.http").MediaIoBaseDownload

        def _download():
            service = self._service_handle()
            request = service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buffer.getvalue()

        return self._call_with_retry(_download)

    def get_metadata(self, file_id: str) -> DriveFile:
        if file_id in self._meta_cache:
            return self._meta_cache[file_id]
        service = self._service_handle()
        meta = self._call_with_retry(
            lambda: service.files()
            .get(fileId=file_id, fields="id,name,mimeType,modifiedTime,size")
            .execute()
        )
        file_obj = DriveFile(
            file_id=meta.get("id", file_id),
            name=meta.get("name") or file_id,
            mime_type=meta.get("mimeType") or "",
            modified_time=meta.get("modifiedTime"),
            size_bytes=_parse_size(meta.get("size")),
        )
        self._meta_cache[file_id] = file_obj
        return file_obj

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        MediaIoBaseDownload = _import_module("googleapiclient.http").MediaIoBaseDownload

        meta = self.get_metadata(file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        needs_download = True
        if dest.exists():
            needs_download = False
            try:
                stat = dest.stat()
            except OSError:
                needs_download = True
            else:
                if meta.size_bytes is not None and stat.st_size != meta.size_bytes:
                    needs_download = True
                modified_timestamp = _parse_modified_time(meta.modified_time)
                if modified_timestamp is not None and abs(stat.st_mtime - modified_timestamp) > 1:
                    needs_download = True
        if needs_download:
            def _download_once():
                tmp_path = None
                try:
                    service = self._service_handle()
                    request = service.files().get_media(fileId=file_id)
                    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
                        tmp_path = Path(handle.name)
                        downloader = MediaIoBaseDownload(handle, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                    tmp_path.replace(dest)
                except Exception:
                    if tmp_path is not None:
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                    raise

            self._call_with_retry(_download_once)
        modified_timestamp = _parse_modified_time(meta.modified_time)
        if modified_timestamp is not None:
            os.utime(dest, (modified_timestamp, modified_timestamp))
        return meta

    def download_json_payload(self, file_id: str, *, name: str) -> object:
        import ijson
        raw = self.download_bytes(file_id)
        handle = io.BytesIO(raw)
        
        if name.lower().endswith(".jsonl"):
            items = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return items

        # Return the whole object but use ijson for potentially better memory handling
        # (though for standard json.load it won't matter much unless we refactor 
        # higher up to handle generators, which we will do in source_ingest).
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            text = handle.read().decode("utf-8", errors="replace")
            return json.loads(text)


__all__ = [
    "DriveClient",
    "DriveFile",
    "DriveError",
    "DriveAuthError",
    "DriveNotFoundError",
    "default_credentials_path",
    "default_token_path",
]
