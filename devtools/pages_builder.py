"""Documentation-site builder for GitHub Pages."""

from __future__ import annotations

import os
import posixpath
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import tomllib
from jinja2 import DictLoader, Environment, select_autoescape

from devtools import repo_root as _get_root
from devtools.pages_style import PAGES_STYLE
from devtools.pages_templates import PAGES_TEMPLATES
from polylogue.paths import active_index_db_path
from polylogue.rendering.renderers.html import PygmentsHighlighter
from polylogue.rendering.renderers.html_sanitizer import sanitize_html

ROOT = _get_root()
GITHUB_BLOB_BASE = "https://github.com/Sinity/polylogue/blob/master/"


@dataclass(frozen=True)
class BrokenSiteLink:
    page: str
    href: str
    target: str


def _default_config_path() -> Path:
    return ROOT / "docs" / "site" / "pages.toml"


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


def _page_output_rel(page_path: str) -> str:
    rel = page_path.lstrip("/")
    if not rel:
        return "index.html"
    if rel.endswith("/"):
        return f"{rel}index.html"
    if rel.endswith(".html"):
        return rel
    return f"{rel}/index.html"


def _relative_href_from_output(current_output_rel: str, target_output_rel: str) -> str:
    start = posixpath.dirname(current_output_rel) or "."
    href = posixpath.relpath(target_output_rel, start=start)
    if href == ".":
        return "./"
    if href == "index.html":
        return "./"
    if href.endswith("/index.html"):
        href = href[: -len("index.html")]
    return href


def _href_between_pages(current_page_path: str, target_page_path: str) -> str:
    return _relative_href_from_output(_page_output_rel(current_page_path), _page_output_rel(target_page_path))


def _source_to_page_map(pages: list[PageEntry]) -> dict[str, str]:
    return {Path(page.source).as_posix(): page.path for page in pages if page.source}


def _source_relative_path(source: str, href_path: str) -> str | None:
    if href_path.startswith("/"):
        rel = posixpath.normpath(href_path.lstrip("/"))
        return None if rel.startswith("../") else rel
    source_dir = posixpath.dirname(source)
    rel = posixpath.normpath(posixpath.join(source_dir, href_path))
    if rel.startswith("../"):
        return None
    return rel


def _rewrite_markdown_href(
    href: str,
    *,
    source: str,
    current_page_path: str,
    source_to_page: dict[str, str],
) -> str:
    parsed = urlparse(href)
    if parsed.scheme or parsed.netloc or href.startswith("#"):
        return href
    if not parsed.path:
        return href
    target_source = _source_relative_path(source, parsed.path)
    if target_source is None:
        return href
    target_page_path = source_to_page.get(target_source)
    if target_page_path:
        rewritten = _href_between_pages(current_page_path, target_page_path)
    elif (ROOT / target_source).exists():
        rewritten = f"{GITHUB_BLOB_BASE}{target_source}"
    else:
        return href
    if parsed.query:
        rewritten = f"{rewritten}?{parsed.query}"
    if parsed.fragment:
        rewritten = f"{rewritten}#{parsed.fragment}"
    return rewritten


def _render_markdown(
    content: str,
    *,
    source: str,
    current_page_path: str,
    source_to_page: dict[str, str],
) -> str:
    from markdown_it import MarkdownIt

    md: MarkdownIt = MarkdownIt("gfm-like", {"highlight": None, "linkify": False})
    md.disable("linkify")

    renderer: Any = md.renderer
    default_link_open = renderer.rules.get("link_open")

    def _link_open(tokens: Any, idx: int, options: Any, env: Any) -> str:
        token = tokens[idx]
        href = token.attrGet("href")
        if href:
            token.attrSet(
                "href",
                _rewrite_markdown_href(
                    href,
                    source=source,
                    current_page_path=current_page_path,
                    source_to_page=source_to_page,
                ),
            )
        if default_link_open is not None:
            return str(default_link_open(tokens, idx, options, env))
        return str(renderer.renderToken(tokens, idx, options, env))

    renderer.rules["link_open"] = _link_open
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
    prev_dict: dict[str, str] | None = (
        {"label": prev_entry.title, "path": prev_entry.path, "href": _href_between_pages(page.path, prev_entry.path)}
        if prev_entry
        else None
    )
    next_dict: dict[str, str] | None = (
        {"label": next_entry.title, "path": next_entry.path, "href": _href_between_pages(page.path, next_entry.path)}
        if next_entry
        else None
    )
    return prev_dict, next_dict


def _nav_data_for_page(config: PagesConfig, page: PageEntry) -> list[dict[str, Any]]:
    return [
        {
            "title": sec.title,
            "entries": [
                {
                    "label": item.label,
                    "path": item.path,
                    "href": _href_between_pages(page.path, item.path),
                }
                for item in sec.items
            ],
        }
        for sec in config.nav
    ]


def _home_links_for_page(page: PageEntry, page_paths: set[str]) -> list[dict[str, str]]:
    candidates = [
        ("Run the demos", "/demos/"),
        ("Inspect the proof", "/proof/"),
        ("Get started", "/docs/getting-started/"),
        ("Polylogue on Sinex", "/architecture/sinex/"),
    ]
    return [
        {"label": label, "href": _href_between_pages(page.path, target), "suffix": "->"}
        for label, target in candidates
        if target in page_paths
    ]


def _validate_site_config(config: PagesConfig) -> None:
    page_paths = {page.path for page in config.pages}
    missing_nav = [
        f"{section.title}: {item.label} -> {item.path}"
        for section in config.nav
        for item in section.items
        if item.path not in page_paths
    ]
    if missing_nav:
        joined = "\n".join(f"- {item}" for item in missing_nav)
        raise ValueError(f"Navigation entries point at unbuilt pages:\n{joined}")


def _site_archive_stats() -> dict[str, Any]:
    """Return cheap archive counts for the site hero without invoking the CLI.

    The hero counts are read from the live archive and are therefore volatile:
    in CI no archive exists (stats are empty), while a local dev archive — and
    especially one mid re-ingest — changes second to second. Embedding those
    counts in the built HTML made ``render pages --check`` non-deterministic
    (a fresh build never matched a `.cache/site` built moments earlier),
    perma-failing the pre-push gate during any ingest. The site build is
    therefore reproducible by default: live counts are read only when
    ``POLYLOGUE_PAGES_LIVE_STATS`` is set (e.g. for a deliberate local preview).
    """
    if not os.environ.get("POLYLOGUE_PAGES_LIVE_STATS"):
        return {}
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
        config_path = _default_config_path()
    if output_dir is None:
        output_dir = ROOT / ".cache" / "site"

    config = load_pages_config(config_path)
    _validate_site_config(config)
    env = _build_env()
    highlighter = PygmentsHighlighter()
    source_to_page = _source_to_page_map(config.pages)
    page_paths = {page.path for page in config.pages}

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
                content_html = _render_markdown(
                    raw,
                    source=page.source,
                    current_page_path=page.path,
                    source_to_page=source_to_page,
                )
            else:
                content_html = f"<pre><code>{sanitize_html(raw)}</code></pre>"

        data = dict(page.data)
        template_data: dict[str, Any] = {
            "title": page.title,
            "style": PAGES_STYLE + "\n" + highlighter.get_css(),
            "nav": _nav_data_for_page(config, page),
            "current_path": page.path,
            "content": content_html,
            "stats": stats,
            "site_root": _href_between_pages(page.path, "/index.html"),
            "search_href": _href_between_pages(page.path, "/docs/search/") if "/docs/search/" in page_paths else "",
            "home_links": _home_links_for_page(page, page_paths),
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


class _SiteLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "a" and attr_map.get("href"):
            self.links.append(attr_map["href"] or "")
        elif tag in {"img", "script"} and attr_map.get("src"):
            self.links.append(attr_map["src"] or "")
        elif tag == "link" and attr_map.get("href"):
            self.links.append(attr_map["href"] or "")


def _local_link_target(page_path: Path, site_dir: Path, href: str) -> Path | None:
    parsed = urlparse(href)
    if parsed.scheme or parsed.netloc or href.startswith("#"):
        return None
    if parsed.path in {"", "#"}:
        return None
    path = parsed.path
    target = site_dir / path.lstrip("/") if path.startswith("/") else page_path.parent / path
    target = target.resolve()
    try:
        target.relative_to(site_dir.resolve())
    except ValueError:
        return None
    if target.is_dir() or path.endswith("/"):
        return target / "index.html"
    if target.suffix:
        return target
    return target / "index.html"


def validate_site_links(site_dir: Path) -> list[BrokenSiteLink]:
    broken: list[BrokenSiteLink] = []
    for page in sorted(site_dir.rglob("*.html")):
        parser = _SiteLinkParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for href in parser.links:
            target = _local_link_target(page, site_dir, href)
            if target is not None and not target.exists():
                broken.append(
                    BrokenSiteLink(
                        page=page.relative_to(site_dir).as_posix(),
                        href=href,
                        target=target.relative_to(site_dir).as_posix(),
                    )
                )
    return broken


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
