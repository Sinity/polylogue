from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from tenacity import (
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..paths import DRIVE_CREDENTIALS_PATH, DRIVE_TOKEN_PATH

T = TypeVar("T")

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
GEMINI_PROMPT_MIME_TYPE = "application/vnd.google-makersuite.prompt"
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
    modified_time: str | None
    size_bytes: int | None


def default_credentials_path(config: object | None = None) -> Path:
    """Get default credentials path, optionally from DriveConfig."""
    if config is not None and hasattr(config, "credentials_path"):
        cred_path = getattr(config, "credentials_path", None)
        if cred_path:
            return Path(cred_path)
    return DRIVE_CREDENTIALS_PATH


def default_token_path(config: object | None = None) -> Path:
    """Get default token path, optionally from DriveConfig."""
    if config is not None and hasattr(config, "token_path"):
        token_path = getattr(config, "token_path", None)
        if token_path:
            return Path(token_path)
    return DRIVE_TOKEN_PATH


def _import_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise DriveAuthError(
            "Drive dependencies are not available. "
            "Install google-api-python-client + google-auth-oauthlib "
            "or run Polylogue from a Nix build/dev shell."
        ) from exc


def _parse_modified_time(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _parse_size(raw: str | int | None) -> int | None:
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


def _resolve_credentials_path(ui: object | None, config: object | None = None) -> Path:
    """Resolve credentials path from config, environment, or defaults."""
    # First check config
    if config is not None and hasattr(config, "credentials_path"):
        cred_path = getattr(config, "credentials_path", None)
        if cred_path:
            return Path(cred_path)

    # Then environment variable
    env_path = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    if env_path:
        return Path(env_path).expanduser()

    # Then default path
    default_path = default_credentials_path(config)
    if default_path.exists():
        return default_path

    # Interactive prompt if UI available
    if ui is not None and not getattr(ui, "plain", True):
        prompt = f"Path to Google OAuth client JSON (default {default_path}):"
        response = getattr(ui, "input", lambda p, default: None)(prompt, default=str(default_path))
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


def _resolve_token_path(config: object | None = None) -> Path:
    """Resolve token path from config, environment, or defaults."""
    # First check config
    if config is not None and hasattr(config, "token_path"):
        token_path = getattr(config, "token_path", None)
        if token_path:
            return Path(token_path)

    # Then environment variable
    env_path = os.environ.get("POLYLOGUE_TOKEN_PATH")
    if env_path:
        return Path(env_path).expanduser()

    # Then default
    return default_token_path(config)


def _resolve_retries(value: int | None, config: object | None = None) -> int:
    """Resolve retry count from explicit value, config, environment, or default."""
    if value is not None:
        return max(0, int(value))

    # Check config
    if config is not None and hasattr(config, "retry_count"):
        return max(0, int(config.retry_count))

    # Check environment
    env_value = os.environ.get(ENV_DRIVE_RETRIES)
    if env_value:
        try:
            return max(0, int(env_value))
        except ValueError:
            pass

    return DEFAULT_DRIVE_RETRIES


def _resolve_retry_base(value: float | None) -> float:
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
    return not isinstance(exc, (DriveAuthError, DriveNotFoundError))


class DriveClient:
    def __init__(
        self,
        *,
        ui: object | None = None,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        retries: int | None = None,
        retry_base: float | None = None,
        config: object | None = None,
    ) -> None:
        self._ui = ui
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._config = config
        self._service = None
        self._meta_cache: dict[str, DriveFile] = {}
        self._retries = _resolve_retries(retries, config)
        self._retry_base = _resolve_retry_base(retry_base)

    def _call_with_retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        from tenacity import Retrying, retry_if_not_exception_type

        retryer = Retrying(
            stop=stop_after_attempt(max(self._retries, 0) + 1),
            wait=wait_exponential(multiplier=self._retry_base, min=self._retry_base, max=10),
            retry=retry_if_exception_type(Exception)
            & retry_if_not_exception_type((DriveAuthError, DriveNotFoundError)),
            reraise=True,
        )
        return retryer(func, *args, **kwargs)

    def _load_credentials(self) -> Any:
        request_cls = _import_module("google.auth.transport.requests").Request
        credentials_cls = _import_module("google.oauth2.credentials").Credentials
        installed_app_flow_cls = _import_module("google_auth_oauthlib.flow").InstalledAppFlow

        token_path = self._token_path or _resolve_token_path(self._config)
        creds = None
        if token_path.exists():
            try:
                creds = credentials_cls.from_authorized_user_file(str(token_path), SCOPES)
            except (OSError, ValueError):
                # Token file corrupt or invalid
                creds = None
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(request_cls())
            except Exception as exc:
                # Token refresh failed - expose the error to the user instead of silently
                # falling back to re-authentication, so they know what went wrong
                raise DriveAuthError(
                    f"Failed to refresh OAuth token: {exc}. "
                    "Try re-authenticating with 'polylogue auth'."
                ) from exc
        if creds and creds.valid:
            self._persist_token(creds, token_path)
            return creds
        if creds and not creds.valid and not creds.refresh_token:
            raise DriveAuthError(
                f"Drive token at {token_path} is invalid and cannot be refreshed "
                "(no refresh token). Delete it and re-run with --interactive to re-authorize."
            )
        if creds is None and token_path.exists() and (self._ui is None or getattr(self._ui, "plain", True)):
            raise DriveAuthError(
                f"Drive token at {token_path} is invalid or expired. "
                "Delete it and re-run with --interactive to re-authorize."
            )

        credentials_path = self._credentials_path or _resolve_credentials_path(self._ui, self._config)
        if not credentials_path.exists():
            raise DriveAuthError(f"Drive credentials not found: {credentials_path}")
        if self._ui is None or getattr(self._ui, "plain", True):
            raise DriveAuthError(
                "Drive authorization required but no interactive UI is available. "
                "Run with --interactive or set POLYLOGUE_TOKEN_PATH with a valid token."
            )
        flow = installed_app_flow_cls.from_client_secrets_file(str(credentials_path), SCOPES)
        try:
            creds = flow.run_local_server(open_browser=False, port=0)
        except OSError as exc:
            # Local server auth failed - try manual flow
            logger.info("Local server auth unavailable (%s). Using manual flow.", exc)
            auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
            console = getattr(self._ui, "console", None)
            if console:
                getattr(console, "print", print)("Open this URL in your browser to authorize Drive access:")
                getattr(console, "print", print)(auth_url)
            code = getattr(self._ui, "input", lambda x: None)("Paste the authorization code")
            if not code:
                raise DriveAuthError("Drive authorization cancelled.") from None
            try:
                flow.fetch_token(code=code)
            except Exception as exc:
                raise DriveAuthError(f"Drive authorization failed: {exc}") from exc
            creds = flow.credentials
        except Exception as exc:
            # Other unexpected errors - try manual flow
            # Keep broad exception here as google-auth can raise various errors
            logger.warning("Unexpected auth error: %s. Falling back to manual flow.", exc)
            auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
            console = getattr(self._ui, "console", None)
            if console:
                getattr(console, "print", print)("Open this URL in your browser to authorize Drive access:")
                getattr(console, "print", print)(auth_url)
            code = getattr(self._ui, "input", lambda x: None)("Paste the authorization code")
            if not code:
                raise DriveAuthError("Drive authorization cancelled.") from None
            try:
                flow.fetch_token(code=code)
            except Exception as exc:
                raise DriveAuthError(f"Drive authorization failed: {exc}") from exc
            creds = flow.credentials
        self._persist_token(creds, token_path)
        return creds

    def _persist_token(self, creds: Any, token_path: Path) -> None:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    def _service_handle(self) -> Any:
        if self._service is not None:
            return self._service
        build = _import_module("googleapiclient.discovery").build

        creds = self._load_credentials()
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def resolve_folder_id(self, folder_ref: str) -> str:
        http_error_cls = _import_module("googleapiclient.errors").HttpError
        service = self._service_handle()
        if _looks_like_id(folder_ref):
            try:
                file_meta: dict[str, Any] = self._call_with_retry(
                    lambda: service.files().get(fileId=folder_ref, fields="id,name,mimeType").execute()
                )
                if file_meta and file_meta.get("mimeType") == FOLDER_MIME_TYPE:
                    return str(file_meta["id"])
            except Exception as exc:
                # File not found or permission denied - try by name
                # Keep broad exception here as google-api errors vary
                if isinstance(exc, http_error_cls):
                    if exc.resp.status not in (404, 403):
                        logger.warning("Unexpected Drive API error resolving %s: %s", folder_ref, exc)
                else:
                    logger.warning("Error resolving folder ID %s: %s", folder_ref, exc)
                pass
        escaped = folder_ref.replace("'", "\\'")
        query = f"name = '{escaped}' and mimeType = '{FOLDER_MIME_TYPE}' and trashed = false"
        response: dict[str, Any] = self._call_with_retry(lambda: service.files().list(q=query, fields="files(id,name)").execute())
        matches = response.get("files", [])
        if not matches:
            raise DriveNotFoundError(f"Folder not found: {folder_ref}")
        return str(matches[0]["id"])

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        service = self._service_handle()
        page_token: str | None = None
        query = f"'{folder_id}' in parents and trashed = false"
        fields = "nextPageToken, files(id,name,mimeType,modifiedTime,size)"
        while True:
            response: dict[str, Any] = self._call_with_retry(
                lambda t=page_token: service.files().list(q=query, fields=fields, pageToken=t, pageSize=1000).execute()
            )
            for item in response.get("files", []):
                name = item.get("name") or ""
                mime_type = item.get("mimeType") or ""
                # Include .json/.jsonl files OR Gemini AI Studio prompt files
                is_json_file = name.lower().endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson"))
                is_gemini_prompt = mime_type == GEMINI_PROMPT_MIME_TYPE
                if not (is_json_file or is_gemini_prompt):
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
        media_io_base_download_cls = _import_module("googleapiclient.http").MediaIoBaseDownload

        def _download() -> bytes:
            service = self._service_handle()
            request = service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = media_io_base_download_cls(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buffer.getvalue()

        return self._call_with_retry(_download)

    def get_metadata(self, file_id: str) -> DriveFile:
        if file_id in self._meta_cache:
            return self._meta_cache[file_id]
        service = self._service_handle()
        meta: dict[str, Any] = self._call_with_retry(
            lambda: service.files().get(fileId=file_id, fields="id,name,mimeType,modifiedTime,size").execute()
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
        media_io_base_download_cls = _import_module("googleapiclient.http").MediaIoBaseDownload

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

            def _download_once() -> None:
                tmp_path: Path | None = None
                try:
                    service = self._service_handle()
                    request = service.files().get_media(fileId=file_id)
                    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
                        tmp_path = Path(handle.name)
                        downloader = media_io_base_download_cls(handle, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                    tmp_path.replace(dest)
                except Exception as e:
                    if tmp_path is not None:
                        with contextlib.suppress(OSError):
                            tmp_path.unlink()
                    raise e from None

            self._call_with_retry(_download_once)
        modified_timestamp = _parse_modified_time(meta.modified_time)
        if modified_timestamp is not None:
            os.utime(dest, (modified_timestamp, modified_timestamp))
        return meta

    def download_json_payload(self, file_id: str, *, name: str) -> object:
        raw = self.download_bytes(file_id)
        handle = io.BytesIO(raw)

        # Treat all newline-delimited JSON formats consistently
        name_lower = name.lower()
        is_ndjson = (
            name_lower.endswith(".jsonl")
            or name_lower.endswith(".jsonl.txt")
            or name_lower.endswith(".ndjson")
        )
        if is_ndjson:
            items = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON line in Drive file %s: %s", name, exc)
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
    "_parse_modified_time",
]
