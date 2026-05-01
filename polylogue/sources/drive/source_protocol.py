from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from polylogue.core.json import JSONValue

from .types import DriveFile


class DriveSourceAPI(Protocol):
    """Minimal interface that Drive ingestion code depends on."""

    def resolve_folder_id(self, folder_ref: str) -> str: ...
    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]: ...
    def download_json_payload(self, file_id: str, *, name: str) -> JSONValue: ...
    def download_to_path(self, file_id: str, dest: Path) -> DriveFile: ...
    def download_bytes(self, file_id: str) -> bytes: ...
