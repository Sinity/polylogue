"""Mock objects for Google Drive API testing."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import ParamSpec, Protocol, TypeVar

from polylogue.core.json import json_document, json_document_list
from polylogue.sources.drive.gateway import DriveListFilesResponse, DrivePayloadRecord
from polylogue.sources.drive.types import DriveError, DriveNotFoundError

P = ParamSpec("P")
T = TypeVar("T")


class BinaryWritable(Protocol):
    """Minimal writable handle contract used by Drive downloads."""

    def write(self, data: bytes) -> object: ...


@dataclass
class MockCredentials:
    """Mock Google OAuth credentials."""

    token: str = "mock_access_token"
    refresh_token: str = "mock_refresh_token"
    token_uri: str = "https://oauth2.googleapis.com/token"
    client_id: str = "mock_client_id.apps.googleusercontent.com"
    client_secret: str = "mock_client_secret"
    scopes: list[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/drive.readonly"])
    expiry: datetime | None = None
    valid: bool = True
    expired: bool = False

    def refresh(self, request: object) -> None:
        """Mock refresh method."""
        del request
        if not self.refresh_token:
            raise Exception("Refresh token not found")
        self.token = "refreshed_access_token"
        self.expired = False
        self.valid = True

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(
            {
                "token": self.token,
                "refresh_token": self.refresh_token,
                "token_uri": self.token_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scopes": self.scopes,
            }
        )


@dataclass
class MockDriveFile:
    """Mock Google Drive file metadata."""

    file_id: str
    name: str
    mime_type: str
    modified_time: str | None = None
    size: int | None = None
    parents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to API response format."""
        result: dict[str, object] = {
            "id": self.file_id,
            "name": self.name,
            "mimeType": self.mime_type,
        }
        if self.modified_time:
            result["modifiedTime"] = self.modified_time
        if self.size is not None:
            result["size"] = str(self.size)
        if self.parents:
            result["parents"] = self.parents
        return result


@dataclass
class MockListResponse:
    """Mock Google Drive files().list() response."""

    files: list[MockDriveFile]
    next_page_token: str | None = None

    def execute(self) -> dict[str, object]:
        """Execute the request and return response."""
        result: dict[str, object] = {"files": [f.to_dict() for f in self.files]}
        if self.next_page_token:
            result["nextPageToken"] = self.next_page_token
        return result


@dataclass
class MockGetResponse:
    """Mock Google Drive files().get() response."""

    file: MockDriveFile

    def execute(self) -> dict[str, object]:
        """Execute the request and return file metadata."""
        return self.file.to_dict()


@dataclass
class MockGetMediaResponse:
    """Mock Google Drive files().get_media() response."""

    content: bytes | str

    def execute(self) -> bytes:
        """Execute the request and return file content."""
        if isinstance(self.content, str):
            return self.content.encode("utf-8")
        return self.content


class MockMediaIoBaseDownload:
    """Mock Google's MediaIoBaseDownload for chunked file downloads.

    Simulates the MediaIoBaseDownload class from googleapiclient.http.
    Writes content to the provided file handle and tracks download progress.

    Usage:
        request = service.files().get_media(fileId="file123")
        downloader = MockMediaIoBaseDownload(file_handle, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    """

    def __init__(self, fd: BinaryWritable, request: MockGetMediaResponse, chunksize: int = 1024 * 1024):
        """Initialize the mock downloader.

        Args:
            fd: File-like object to write downloaded content to
            request: MockGetMediaResponse containing the file content
            chunksize: Size of chunks to download (for simulation, we complete in one chunk)
        """
        self._fd = fd
        self._request = request
        self._chunksize = chunksize
        self._done = False
        self._progress = 0.0

    def next_chunk(self) -> tuple[MockDownloadStatus | None, bool]:
        """Download the next chunk of the file.

        Returns:
            Tuple of (status, done) where status contains progress info
            and done is True when download is complete.
        """
        if self._done:
            return None, True

        # Get content from the request
        content = self._request.execute()

        # Write all content at once (simulating completed download)
        self._fd.write(content)
        self._done = True
        self._progress = 1.0

        # Return a status-like object and done flag
        status = MockDownloadStatus(progress=self._progress, total_size=len(content))
        return status, True


@dataclass
class MockDownloadStatus:
    """Mock download status returned by next_chunk()."""

    progress: float = 1.0
    total_size: int = 0

    def progress_percent(self) -> float:
        """Return progress as percentage (0.0 to 1.0)."""
        return self.progress


class MockFilesResource:
    """Mock Google Drive files() resource."""

    def __init__(
        self, files: dict[str, MockDriveFile] | None = None, file_content: dict[str, bytes | str] | None = None
    ):
        """Initialize with file metadata and content.

        Args:
            files: Dict mapping file_id → MockDriveFile
            file_content: Dict mapping file_id → file content (bytes or str)
        """
        self.files = files or {}
        self.file_content = file_content or {}
        self._list_filters: dict[str, object] = {}

    def _parse_query(self, q: str) -> dict[str, str]:
        """Parse Drive API query string into filters.

        Handles queries like:
          name='Google AI Studio'
          name='folder' and mimeType='application/vnd.google-apps.folder'
          'parent_id' in parents and trashed = false
        """
        import re

        filters: dict[str, str] = {}

        # Extract name='value' pattern
        name_match = re.search(r"name\s*=\s*['\"]([^'\"]+)['\"]", q)
        if name_match:
            filters["name"] = name_match.group(1)

        # Extract mimeType='value' pattern
        mime_match = re.search(r"mimeType\s*=\s*['\"]([^'\"]+)['\"]", q)
        if mime_match:
            filters["mimeType"] = mime_match.group(1)

        # Extract 'parent_id' in parents pattern
        parent_match = re.search(r"['\"]([^'\"]+)['\"]\s+in\s+parents", q)
        if parent_match:
            filters["parent_id"] = parent_match.group(1)

        return filters

    def list(
        self,
        q: str | None = None,
        spaces: str = "drive",
        fields: str = "files(id,name,mimeType,modifiedTime,size,parents)",
        pageSize: int = 100,
        pageToken: str | None = None,
    ) -> MockListResponse:
        """Mock files().list() method."""
        self._list_filters = {"q": q, "spaces": spaces, "fields": fields, "pageSize": pageSize, "pageToken": pageToken}

        matching_files = list(self.files.values())

        if q:
            # Parse Drive API queries - handle AND clauses and various filters
            # Example queries:
            #   name='Google AI Studio'
            #   name='folder' and mimeType='application/vnd.google-apps.folder'
            #   'parent_id' in parents and trashed = false
            filters = self._parse_query(q)

            if "name" in filters:
                matching_files = [f for f in matching_files if f.name == filters["name"]]
            if "mimeType" in filters:
                matching_files = [f for f in matching_files if f.mime_type == filters["mimeType"]]
            if "parent_id" in filters:
                matching_files = [f for f in matching_files if filters["parent_id"] in f.parents]

        # Pagination simulation
        page_size = pageSize or 100
        start_idx = int(pageToken) if pageToken else 0

        page_files = matching_files[start_idx : start_idx + page_size]
        next_token = None
        if start_idx + page_size < len(matching_files):
            next_token = str(start_idx + page_size)

        return MockListResponse(files=page_files, next_page_token=next_token)

    def get(self, fileId: str, fields: str = "id,name,mimeType,modifiedTime,size") -> MockGetResponse:
        """Mock files().get() method."""
        if fileId not in self.files:
            raise DriveNotFoundError(f"File not found: {fileId}")
        return MockGetResponse(file=self.files[fileId])

    def get_media(self, fileId: str) -> MockGetMediaResponse:
        """Mock files().get_media() method."""
        if fileId not in self.file_content:
            raise DriveNotFoundError(f"File content not found: {fileId}")
        return MockGetMediaResponse(content=self.file_content[fileId])


class MockDriveService:
    """Mock Google Drive service."""

    _http: object | None = None

    def __init__(
        self, files_data: dict[str, MockDriveFile] | None = None, file_content: dict[str, bytes | str] | None = None
    ):
        """Initialize mock Drive service.

        Args:
            files_data: Dict mapping file_id → MockDriveFile
            file_content: Dict mapping file_id → file content
        """
        self._files_resource = MockFilesResource(files=files_data, file_content=file_content)

    def files(self) -> MockFilesResource:
        """Return mock files() resource."""
        return self._files_resource


class FakeDriveServiceGateway:
    """Lightweight DriveServiceGateway substitute for DriveSourceClient tests.

    Wraps MockDriveService and exposes the same interface as DriveServiceGateway
    without requiring a real auth manager or network access.
    """

    def __init__(
        self,
        mock_service: MockDriveService | None = None,
        file_content: dict[str, bytes | str] | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self._mock_service = mock_service or MockDriveService(file_content=file_content)
        self._service = self._mock_service
        self._download_error = download_error

    def call_with_retry(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)

    def _service_handle(self) -> MockDriveService:
        return self._mock_service

    def get_file(self, file_id: str, fields: str) -> DrivePayloadRecord:
        payload = self._mock_service.files().get(fileId=file_id, fields=fields).execute()
        return json_document(payload)

    def list_files(
        self,
        *,
        q: str,
        fields: str,
        page_token: str | None,
        page_size: int,
    ) -> DriveListFilesResponse:
        payload = json_document(
            self._mock_service.files().list(q=q, fields=fields, pageToken=page_token, pageSize=page_size).execute()
        )
        response: DriveListFilesResponse = {}
        if "files" in payload:
            response["files"] = json_document_list(payload["files"])
        next_page_token = payload.get("nextPageToken")
        if isinstance(next_page_token, str):
            response["nextPageToken"] = next_page_token
        return response

    def download_file(self, file_id: str, handle: BinaryWritable) -> None:
        if self._download_error is not None:
            raise self._download_error
        content = self._mock_service.files().get_media(fileId=file_id).execute()
        handle.write(content)

    def _download_request(
        self,
        request: MockGetMediaResponse,
        handle: BinaryWritable,
        downloader_cls: type[MockMediaIoBaseDownload],
        *,
        file_id: str,
    ) -> None:
        downloader = downloader_cls(handle, request)
        done = False
        max_chunks = 10_000
        chunks = 0
        while not done:
            _, done = downloader.next_chunk()
            chunks += 1
            if chunks >= max_chunks:
                raise DriveError(f"Download exceeded {max_chunks} chunks for file {file_id}")


def mock_drive_file(
    file_id: str = "file123",
    name: str = "test.txt",
    mime_type: str = "text/plain",
    modified_time: str = "2024-01-01T12:00:00Z",
    size: int = 1024,
    parents: list[str] | None = None,
) -> MockDriveFile:
    """Factory function to create MockDriveFile with defaults.

    Args:
        file_id: File ID
        name: File name
        mime_type: MIME type
        modified_time: Modified timestamp (ISO format)
        size: File size in bytes
        parents: List of parent folder IDs

    Returns:
        MockDriveFile instance
    """
    return MockDriveFile(
        file_id=file_id,
        name=name,
        mime_type=mime_type,
        modified_time=modified_time,
        size=size,
        parents=parents or [],
    )
