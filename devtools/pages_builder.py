"""Documentation-site builder for GitHub Pages."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib
from jinja2 import DictLoader, Environment, select_autoescape

from devtools import repo_root as _get_root
from devtools.pages_style import PAGES_STYLE
from devtools.pages_templates import PAGES_TEMPLATES
from polylogue.paths import active_index_db_path
from polylogue.rendering.renderers.html import PygmentsHighlighter
from polylogue.rendering.renderers.html_sanitizer import sanitize_html

ROOT = _get_root()


@dataclass
class NavItem:
    label: str
    path: str


@dataclass
class NavSection:
    title: str
    items: list[NavItem] = field(default_factory=list)


@dataclass
class PageEntry:
    path: str
    title: str
    source: str = ""
    template: str = "doc.html"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PagesConfig:
    nav: list[NavSection] = field(default_factory=list)
    pages: list[PageEntry] = field(default_factory=list)
    site_name: str = "Polylogue"
    site_url: str = "/polylogue/"


def _parse_nav(raw: list[dict[str, Any]]) -> list[NavSection]:
    sections: list[NavSection] = []
    for sec in raw:
        items = [NavItem(label=item["label"], path=item["path"]) for item in sec.get("items", [])]
        sections.append(NavSection(title=sec["title"], items=items))
    return sections


def _parse_pages(raw: list[dict[str, Any]]) -> list[PageEntry]:
    return [
        PageEntry(
            path=entry["path"],
            title=entry["title"],
            source=entry.get("source", ""),
            template=entry.get("template", "doc.html"),
            data=entry.get("data", {}),
        )
        for entry in raw
    ]


def load_pages_config(config_path: Path) -> PagesConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    return PagesConfig(
        nav=_parse_nav(raw.get("nav", [])),
        pages=_parse_pages(raw.get("pages", [])),
        site_name=raw.get("site_name", "Polylogue"),
        site_url=raw.get("site_url", "/polylogue/"),
    )


def _build_env() -> Environment:
    env = Environment(
        loader=DictLoader(PAGES_TEMPLATES),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["sanitize_html"] = sanitize_html
    return env


def _render_markdown(content: str) -> str:
    from markdown_it import MarkdownIt

    md: MarkdownIt = MarkdownIt("gfm-like", {"highlight": None})
    result: str = md.render(content)
    return result


def _read_source(source: str) -> str:
    path = ROOT / source
    if not path.exists():
        return f"<!-- Source not found: {source} -->"
    return path.read_text()


def _find_prev_next(page: PageEntry, all_pages: list[PageEntry]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    idx = next((i for i, p in enumerate(all_pages) if p.path == page.path), -1)
    prev_entry = all_pages[idx - 1] if idx > 0 else None
    next_entry = all_pages[idx + 1] if idx >= 0 and idx < len(all_pages) - 1 else None
    prev_dict: dict[str, str] | None = {"label": prev_entry.title, "path": prev_entry.path} if prev_entry else None
    next_dict: dict[str, str] | None = {"label": next_entry.title, "path": next_entry.path} if next_entry else None
    return prev_dict, next_dict


def _site_archive_stats() -> dict[str, Any]:
    """Return cheap archive counts for the site hero without invoking the CLI."""
    path = active_index_db_path()
    if not path.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        try:
            if not _table_exists(conn, "sessions"):
                return {}
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_sessions,
                    COALESCE(SUM(message_count), 0) AS total_messages,
                    COUNT(DISTINCT origin) AS provider_count
                FROM sessions
                """
            ).fetchone()
            if row is None:
                return {}
            total_sessions = int(row[0] or 0)
            total_messages = int(row[1] or 0)
            provider_count = int(row[2] or 0)
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "provider_count": provider_count,
    }


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def build_site(config_path: Path | None = None, output_dir: Path | None = None) -> Path:
    if config_path is None:
        config_path = ROOT / "pages.toml"
    if output_dir is None:
        output_dir = ROOT / ".cache" / "site"

    config = load_pages_config(config_path)
    env = _build_env()
    highlighter = PygmentsHighlighter()

    nav_data: list[dict[str, Any]] = [
        {
            "title": sec.title,
            "entries": [{"label": item.label, "path": item.path} for item in sec.items],
        }
        for sec in config.nav
    ]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    stats_raw = _site_archive_stats()

    stats = {
        "session_count": f"{stats_raw.get('total_sessions', 0):,}",
        "message_count": f"{stats_raw.get('total_messages', 0) / 1_000_000:.1f}M",
        "provider_count": str(stats_raw.get("provider_count", 0)),
    }

    for page in config.pages:
        out_path = output_dir / page.path.lstrip("/")
        if out_path.suffix != ".html":
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / "index.html"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        content_html = ""
        if page.source:
            raw = _read_source(page.source)
            if page.source.endswith(".md"):
                content_html = _render_markdown(raw)
            else:
                content_html = f"<pre><code>{sanitize_html(raw)}</code></pre>"

        data = dict(page.data)
        template_data: dict[str, Any] = {
            "title": page.title,
            "style": PAGES_STYLE + "\n" + highlighter.get_css(),
            "nav": nav_data,
            "current_path": page.path,
            "content": content_html,
            "stats": stats,
            "updated_at": "",
            **data,
        }

        if page.template in ("doc.html",):
            all_doc_pages = [p for p in config.pages if p.template == "doc.html"]
            prev_doc, next_doc = _find_prev_next(page, all_doc_pages)
            template_data["prev"] = prev_doc or False
            template_data["next"] = next_doc or False

        template = env.get_template(page.template)
        html: str = template.render(**template_data)
        out_path.write_text(html)

    return output_dir


def build_site_with_pagefind(config_path: Path | None = None, output_dir: Path | None = None) -> Path:
    site_dir = build_site(config_path=config_path, output_dir=output_dir)
    try:
        subprocess.run(
            ["pagefind", "--site", str(site_dir)],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: pagefind not available, skipping search index", file=sys.stderr)
    return site_dir


__all__ = [
    "build_site",
    "build_site_with_pagefind",
    "load_pages_config",
    "PagesConfig",
    "NavSection",
    "NavItem",
    "PageEntry",
]
