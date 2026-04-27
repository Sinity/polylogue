"""Search helpers for static-site generation."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import IO

from polylogue.site.models import ConversationIndex, SearchDocument, SiteConfig


class SearchBuildStatus(str, Enum):
    """Materialization state for static-site search assets."""

    DISABLED = "disabled"
    JSON_INDEX_WRITTEN = "json_index_written"
    BUILT = "built"
    FAILED = "failed"
    PENDING = "pending"

    def __str__(self) -> str:
        return self.value


def build_search_document(conversation: ConversationIndex) -> SearchDocument:
    """Build the canonical search payload for one indexed conversation."""
    return SearchDocument.from_conversation(conversation)


class SearchIndexWriter:
    """Incremental JSON search-index writer for site scans."""

    def __init__(self, output_dir: Path, config: SiteConfig) -> None:
        self._enabled = config.enable_search and not config.uses_pagefind
        self._path = output_dir / "search-index.json"
        self._handle: IO[str] | None = None
        self._wrote_entry = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def open(self) -> None:
        """Open the JSON array stream when this build writes a search index."""
        if not self._enabled:
            return
        self._handle = self._path.open("w", encoding="utf-8")
        self._handle.write("[")

    def append(self, document: SearchDocument) -> None:
        """Append one document to the open JSON array."""
        if self._handle is None:
            return
        if self._wrote_entry:
            self._handle.write(",")
        json.dump(document.to_payload(), self._handle)
        self._wrote_entry = True

    def finish(self) -> None:
        """Close the JSON array stream after a successful scan."""
        if self._handle is None:
            return
        self._handle.write("]")
        self._handle.close()
        self._handle = None

    def abort(self) -> None:
        """Discard a partial search index after a failed scan."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._path.unlink(missing_ok=True)


def render_search_markup(config: SiteConfig) -> str:
    """Render the search UI snippet for index pages."""
    if not config.enable_search:
        return ""
    if config.uses_pagefind:
        return """
        <div id="search" style="margin-bottom: 2rem;"></div>
        <link href="/_pagefind/pagefind-ui.css" rel="stylesheet">
        <script src="/_pagefind/pagefind-ui.js"></script>
        <script>
            window.addEventListener('DOMContentLoaded', function() {
                new PagefindUI({ element: "#search", showSubResults: true });
            });
        </script>
"""
    return """
        <div class="search-panel" style="margin-bottom: 2rem;">
            <input id="search-input" type="search" placeholder="Search conversations..." style="width: 100%; padding: 0.85rem 1rem; border-radius: 10px; border: 1px solid var(--border); background: var(--bg-secondary); color: var(--text-primary);" />
            <p id="search-status" style="margin-top: 0.75rem; color: var(--text-secondary);"></p>
            <ul id="search-results" class="conversation-list" style="margin-top: 1rem; display: none;"></ul>
        </div>
        <script>
            window.addEventListener('DOMContentLoaded', async function() {
                const input = document.getElementById('search-input');
                const status = document.getElementById('search-status');
                const results = document.getElementById('search-results');
                const archiveList = document.querySelector('.conversation-list');
                let docs = [];
                try {
                    const response = await fetch('/search-index.json');
                    docs = response.ok ? await response.json() : [];
                } catch (error) {
                    status.textContent = 'Search index unavailable.';
                    return;
                }

                function renderResults(query) {
                    const term = query.trim().toLowerCase();
                    if (!term) {
                        results.style.display = 'none';
                        results.innerHTML = '';
                        archiveList.style.display = '';
                        status.textContent = '';
                        return;
                    }

                    archiveList.style.display = 'none';
                    const matches = docs.filter((doc) => {
                        return [doc.title, doc.provider, doc.preview]
                            .filter(Boolean)
                            .join(' ')
                            .toLowerCase()
                            .includes(term);
                    }).slice(0, 50);

                    const esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
                    results.innerHTML = matches.map((doc) => `
                        <li class="conversation-card">
                            <a href="${esc(doc.path)}" class="conversation-link">
                                <h2 class="conversation-title">${esc(doc.title)}</h2>
                                <div class="conversation-meta">
                                    <span class="badge">${esc(doc.provider)}</span>
                                    <span>${esc(doc.preview || '')}</span>
                                </div>
                            </a>
                        </li>
                    `).join('');
                    results.style.display = matches.length ? '' : 'none';
                    status.textContent = matches.length
                        ? `${matches.length} result(s)`
                        : 'No conversations matched.';
                }

                input.addEventListener('input', (event) => renderResults(event.target.value));
            });
        </script>
"""


def _pagefind_config(output_dir: Path) -> Mapping[str, str]:
    return {
        "site": str(output_dir),
        "output_subdir": "_pagefind",
    }


def generate_pagefind_config(output_dir: Path) -> SearchBuildStatus:
    """Generate pagefind config and opportunistically build the index."""
    (output_dir / "pagefind.json").write_text(
        json.dumps(_pagefind_config(output_dir), indent=2),
        encoding="utf-8",
    )

    pagefind_bin = shutil.which("pagefind")
    if pagefind_bin is None:
        return SearchBuildStatus.PENDING
    try:
        subprocess.run(
            [pagefind_bin, "--site", str(output_dir)],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    except subprocess.CalledProcessError:
        return SearchBuildStatus.FAILED
    except FileNotFoundError:
        return SearchBuildStatus.PENDING
    return SearchBuildStatus.BUILT


__all__ = [
    "SearchBuildStatus",
    "SearchIndexWriter",
    "build_search_document",
    "generate_pagefind_config",
    "render_search_markup",
]
