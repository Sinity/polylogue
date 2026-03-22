"""Search helpers for static-site generation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from polylogue.site.models import ConversationIndex, SiteConfig


def build_search_document(conversation: ConversationIndex) -> dict[str, str]:
    """Build a JSON-search entry for a site index conversation."""
    return {
        "id": conversation.id,
        "title": conversation.title,
        "provider": conversation.provider,
        "preview": conversation.preview,
        "path": conversation.path,
    }


def render_search_markup(config: SiteConfig) -> str:
    """Render the search UI snippet for index pages."""
    if not config.enable_search:
        return ""
    if config.search_provider == "pagefind":
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

                    results.innerHTML = matches.map((doc) => `
                        <li class="conversation-card">
                            <a href="${doc.path}" class="conversation-link">
                                <h2 class="conversation-title">${doc.title}</h2>
                                <div class="conversation-meta">
                                    <span class="badge">${doc.provider}</span>
                                    <span>${doc.preview || ''}</span>
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


def generate_pagefind_config(output_dir: Path) -> str:
    """Generate pagefind config and opportunistically build the index."""
    config = {
        "site": str(output_dir),
        "output_subdir": "_pagefind",
    }
    (output_dir / "pagefind.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    pagefind_bin = shutil.which("pagefind")
    if pagefind_bin:
        try:
            subprocess.run(
                [pagefind_bin, "--site", str(output_dir)],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            return "built"
        except subprocess.CalledProcessError:
            return "failed"
        except FileNotFoundError:
            return "pending"
    return "pending"

