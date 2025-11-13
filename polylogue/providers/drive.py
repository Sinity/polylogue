from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from typing import Callable

from ..drive_client import DriveClient
from ..ui import UI


class DriveProviderSession:
    """Bridge DriveClient to the provider registry."""

    name = "drive"
    title = "Google Drive"

    def __init__(self, ui: UI, client_factory: Callable[[UI], DriveClient] = DriveClient):
        self._client = client_factory(ui)

    @property
    def ui(self) -> UI:
        return self._client.ui

    def __getattr__(self, item: str):
        # Delegate to underlying DriveClient (download/list/etc.)
        return getattr(self._client, item)

    @property
    def client(self) -> DriveClient:
        return self._client

    # Explicit wrappers for type clarity
    def list_chats(self, folder_name: Optional[str], folder_id: Optional[str]) -> List[Dict[str, object]]:
        return self._client.list_chats(folder_name, folder_id)

    def download_chat_bytes(self, file_id: str) -> Optional[bytes]:
        return self._client.download_chat_bytes(file_id)

    def download_attachment(self, file_id: str, path: Path) -> bool:
        return self._client.download_attachment(file_id, path)
