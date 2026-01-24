"""Mock objects for Google Drive API testing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockCredentials:
    """Mock Google OAuth credentials."""

    token: str = "mock_access_token"
    refresh_token: str = "mock_refresh_token"
    token_uri: str = "https://oauth2.googleapis.com/token"
    client_id: str = "mock_client_id.apps.googleusercontent.com"
    client_secret: str = "mock_client_secret"
    scopes: list[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/drive.readonly"])
    expiry: Any = None
    valid: bool = True
    expired: bool = False

    def refresh(self, request: Any) -> None:
        """Mock refresh method."""
        if not self.refresh_token:
            raise Exception("Refresh token not found")
        self.token = "refreshed_access_token"
        self.expired = False
        self.valid = True

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "token": self.token,
            "refresh_token": self.refresh_token,
            "token_uri": self.token_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scopes": self.scopes,
        })


@dataclass
class MockDriveFile:
    """Mock Google Drive file metadata."""

    file_id: str
    name: str
    mime_type: str
    modified_time: str | None = None
    size: int | None = None
    parents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to API response format."""
        result = {
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

    def execute(self) -> dict[str, Any]:
        """Execute the request and return response."""
        result = {"files": [f.to_dict() for f in self.files]}
        if self.next_page_token:
            result["nextPageToken"] = self.next_page_token
        return result


@dataclass
class MockGetResponse:
    """Mock Google Drive files().get() response."""

    file: MockDriveFile

    def execute(self) -> dict[str, Any]:
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

    def __init__(self, fd: Any, request: MockGetMediaResponse, chunksize: int = 1024 * 1024):
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

    def next_chunk(self) -> tuple[Any, bool]:
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

    def __init__(self, files: dict[str, MockDriveFile] | None = None, file_content: dict[str, bytes | str] | None = None):
        """Initialize with file metadata and content.

        Args:
            files: Dict mapping file_id → MockDriveFile
            file_content: Dict mapping file_id → file content (bytes or str)
        """
        self.files = files or {}
        self.file_content = file_content or {}
        self._list_filters: dict[str, Any] = {}

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

        # Simple query parsing (just handle "name=" and "in parents")
        matching_files = list(self.files.values())

        if q:
            # Parse simple queries like "name='folder'" or "'parent_id' in parents"
            if "name=" in q:
                name_query = q.split("name=")[1].strip(" '\"")
                matching_files = [f for f in matching_files if f.name == name_query]
            elif "in parents" in q:
                parent_id = q.split("'")[1]
                matching_files = [f for f in matching_files if parent_id in f.parents]

        # Pagination simulation
        page_size = pageSize or 100
        if pageToken:
            start_idx = int(pageToken)
        else:
            start_idx = 0

        page_files = matching_files[start_idx : start_idx + page_size]
        next_token = None
        if start_idx + page_size < len(matching_files):
            next_token = str(start_idx + page_size)

        return MockListResponse(files=page_files, next_page_token=next_token)

    def get(self, fileId: str, fields: str = "id,name,mimeType,modifiedTime,size") -> MockGetResponse:
        """Mock files().get() method."""
        if fileId not in self.files:
            raise Exception(f"File not found: {fileId}")
        return MockGetResponse(file=self.files[fileId])

    def get_media(self, fileId: str) -> MockGetMediaResponse:
        """Mock files().get_media() method."""
        if fileId not in self.file_content:
            raise Exception(f"File content not found: {fileId}")
        return MockGetMediaResponse(content=self.file_content[fileId])


class MockDriveService:
    """Mock Google Drive service."""

    def __init__(self, files_data: dict[str, MockDriveFile] | None = None, file_content: dict[str, bytes | str] | None = None):
        """Initialize mock Drive service.

        Args:
            files_data: Dict mapping file_id → MockDriveFile
            file_content: Dict mapping file_id → file content
        """
        self._files_resource = MockFilesResource(files=files_data, file_content=file_content)

    def files(self) -> MockFilesResource:
        """Return mock files() resource."""
        return self._files_resource


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
