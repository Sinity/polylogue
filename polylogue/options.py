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
    diff: bool = False


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
    diff: bool = False


@dataclass
class ListOptions:
    folder_name: str
    folder_id: Optional[str]
    since: Optional[str]
    until: Optional[str]
    name_filter: Optional[str]


@dataclass
class StatusOptions:
    json: bool = False
    watch: bool = False
    interval: float = 5.0


@dataclass
class RenderFile:
    output: Path
    slug: str
    attachments: int
    stats: Dict[str, Any]
    html: Optional[Path] = None
    diff: Optional[Path] = None


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
    slug: str
    attachments: int
    stats: Dict[str, Any]
    html: Optional[Path] = None
    diff: Optional[Path] = None


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
    run_summary: Dict[str, Any]
    provider_summary: Dict[str, Any]
    runs: List[dict]


@dataclass
class BranchExploreOptions:
    provider: Optional[str]
    slug: Optional[str]
    conversation_id: Optional[str]
    min_branches: int
    canonical_branch: Optional[str] = None


@dataclass
class BranchExploreResult:
    conversations: List[Any]


@dataclass
class SearchOptions:
    query: str
    limit: int
    provider: Optional[str]
    slug: Optional[str]
    conversation_id: Optional[str]
    branch_id: Optional[str]
    model: Optional[str]
    since: Optional[str]
    until: Optional[str]
    has_attachments: Optional[bool]


@dataclass
class SearchHit:
    provider: str
    conversation_id: str
    slug: str
    title: Optional[str]
    branch_id: str
    message_id: str
    position: int
    timestamp: Optional[str]
    attachment_count: int
    score: float
    snippet: str
    body: str
    conversation_path: Optional[Path]
    branch_path: Optional[Path]
    model: Optional[str]


@dataclass
class SearchResult:
    hits: List[SearchHit]
