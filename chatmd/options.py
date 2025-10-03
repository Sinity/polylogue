from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RenderOptions:
    inputs: List[Path]
    output_dir: Path
    collapse_threshold: int
    download_attachments: bool
    dry_run: bool
    force: bool
    html: bool = False
    html_theme: str = "light"


@dataclass
class SyncOptions:
    folder_name: str
    folder_id: Optional[str]
    output_dir: Path
    collapse_threshold: int
    download_attachments: bool
    dry_run: bool
    force: bool
    prune: bool
    since: Optional[str]
    until: Optional[str]
    name_filter: Optional[str]
    selected_ids: Optional[List[str]] = None
    html: bool = False
    html_theme: str = "light"


@dataclass
class ListOptions:
    folder_name: str
    folder_id: Optional[str]
    since: Optional[str]
    until: Optional[str]
    name_filter: Optional[str]


@dataclass
class StatusOptions:
    pass


@dataclass
class RenderFile:
    output: Path
    attachments: int
    stats: Dict[str, Any]
    html: Optional[Path] = None


@dataclass
class RenderResult:
    count: int
    output_dir: Path
    files: List[RenderFile]
    total_stats: Dict[str, Any]


@dataclass
class SyncItem:
    id: Optional[str]
    name: Optional[str]
    output: Path
    attachments: int
    stats: Dict[str, Any]
    html: Optional[Path] = None


@dataclass
class SyncResult:
    count: int
    output_dir: Path
    folder_name: str
    folder_id: Optional[str]
    items: List[SyncItem]
    total_stats: Dict[str, Any]


@dataclass
class ListResult:
    folder_name: str
    folder_id: Optional[str]
    files: List[dict]


@dataclass
class StatusResult:
    credentials_present: bool
    token_present: bool
    state_path: Path
    runs_path: Path
    recent_runs: List[dict]
