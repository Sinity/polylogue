"""Behavioral tests for the GitHub Pages site renderer (#1301).

The renderer is implemented under ``devtools/pages_builder.py`` and is
driven by ``devtools render pages``. Until #1301, no test exercised its
output — a regression that produced broken HTML, missing navigation, or
stale anchors would ship unnoticed. The tests below cover:

* end-to-end page generation from a minimal ``pages.toml`` plus inline
  source files,
* navigation rendering and link integrity across emitted pages,
* asset references (stylesheet inlining, page-relative anchors),
* frontmatter / per-page data passed through to templates,
* error / fallback paths for malformed inputs (missing source file,
  unknown template, mistyped TOML section).

The tests work directly against ``build_site`` / ``load_pages_config``
without invoking the ``devtools`` CLI wrapper, so they stay fast and
deterministic and do not depend on an operator archive.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from devtools import pages_builder
from devtools.pages_builder import (
    PagesConfig,
    build_site,
    load_pages_config,
    validate_site_links,
)

_MINIMAL_CONFIG = """
site_name = "Polylogue"
site_url = "/polylogue/"

[[nav]]
title = "Docs"
[[nav.items]]
label = "Getting Started"
path = "/docs/getting-started/"
[[nav.items]]
label = "Search"
path = "/docs/search/"

[[pages]]
path = "/index.html"
title = "Home"
template = "home.html"

[[pages]]
path = "/beads/"
title = "Roadmap"
template = "beads.html"

[[pages]]
path = "/demos/"
title = "Demos"
source = "docs/demos.md"

[[pages]]
path = "/docs/"
title = "Documentation"
source = "docs/README.md"

[[pages]]
path = "/docs/getting-started/"
title = "Getting Started"
source = "docs/getting-started.md"

[[pages]]
path = "/docs/search/"
title = "Search"
source = "docs/search.md"
"""


@pytest.fixture
def synthetic_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Build a synthetic site against an isolated repo root.

    Writes a tiny ``pages.toml`` plus two markdown sources into ``tmp_path``
    and re-points ``pages_builder.ROOT`` so the renderer reads them from
    there instead of the real repo. Returns the output directory.
    """
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    (fake_root / "README.md").write_text("# Repo README\n", encoding="utf-8")
    docs = fake_root / "docs"
    docs.mkdir()
    (docs / "getting-started.md").write_text(
        "# Getting Started\n\n"
        "Welcome to the synthetic site.\n\n"
        "[Search details](search.md#search-details) and [repo README](../README.md).\n\n"
        "Plain file names such as CLAUDE.md should not become external links.\n",
        encoding="utf-8",
    )
    (docs / "search.md").write_text(
        "# Search\n\nLook things up here.\n\n## Search Details\n\nUse a query.\n",
        encoding="utf-8",
    )
    (docs / "demos.md").write_text("# Demos\n\nRun a proof.\n", encoding="utf-8")
    (docs / "README.md").write_text("# Documentation\n\nChoose a guide.\n", encoding="utf-8")
    config_path = fake_root / "pages.toml"
    config_path.write_text(_MINIMAL_CONFIG, encoding="utf-8")
    monkeypatch.setattr(pages_builder, "ROOT", fake_root)

    out = tmp_path / "out"
    build_site(config_path=config_path, output_dir=out)
    return out


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.anchors: list[str] = []
        self.images: list[str] = []
        self.title_chars: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "a":
            href = attr_map.get("href")
            if href:
                self.hrefs.append(href)
        elif tag == "img":
            src = attr_map.get("src")
            if src:
                self.images.append(src)
        elif tag == "h1":
            name = attr_map.get("id")
            if name:
                self.anchors.append(name)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_chars.append(data)


def _collect_links(html: str) -> _LinkCollector:
    collector = _LinkCollector()
    collector.feed(html)
    return collector


# ---------------------------------------------------------------------------
# Page generation
# ---------------------------------------------------------------------------


def test_build_site_emits_expected_pages(synthetic_site: Path) -> None:
    """Each page entry produces an HTML file at the configured path."""
    assert (synthetic_site / "index.html").is_file()
    assert (synthetic_site / "docs" / "getting-started" / "index.html").is_file()
    assert (synthetic_site / "docs" / "search" / "index.html").is_file()


def test_doc_pages_render_markdown_body(synthetic_site: Path) -> None:
    """Markdown sources are rendered into the page body."""
    html = _read(synthetic_site / "docs" / "getting-started" / "index.html")
    assert '<h1 id="getting-started">Getting Started</h1>' in html
    assert "Welcome to the synthetic site" in html


def test_doc_pages_carry_page_title(synthetic_site: Path) -> None:
    """Per-page title flows into the document <title>."""
    html = _read(synthetic_site / "docs" / "search" / "index.html")
    collector = _collect_links(html)
    rendered_title = "".join(collector.title_chars).strip()
    assert "Search" in rendered_title
    assert "Polylogue" in rendered_title


def test_markdown_headings_receive_linkable_ids(synthetic_site: Path) -> None:
    html = _read(synthetic_site / "docs" / "search" / "index.html")
    assert '<h1 id="search">Search</h1>' in html
    assert '<h2 id="search-details">Search Details</h2>' in html


# ---------------------------------------------------------------------------
# Navigation & link integrity
# ---------------------------------------------------------------------------


def test_navigation_section_rendered_on_every_page(synthetic_site: Path) -> None:
    """Each emitted page contains every nav entry from pages.toml.

    Items are rendered as ``<a href="...">\\n   Label\\n</a>``, so we
    normalize whitespace before substring-checking. Links are relative to
    the current HTML file so the same artifact can be deployed under
    ``/main/``, ``/pr/N/``, ``/latest/``, or served from disk.
    """
    for rel in (
        "index.html",
        "docs/getting-started/index.html",
        "docs/search/index.html",
    ):
        html = _read(synthetic_site / rel)
        normalized = re.sub(r"\s+", " ", html)
        assert ">Docs<" in normalized, rel
        assert 'href="/polylogue' not in normalized, rel
        assert "> Getting Started <" in normalized or ">Getting Started<" in normalized, rel
        assert "> Search <" in normalized or ">Search<" in normalized, rel


def test_navigation_links_resolve_to_built_pages(synthetic_site: Path) -> None:
    """Every nav-sidebar link maps to a built page in the output.

    We extract only sidebar links (not body links from rendered markdown /
    home-page hero) and require each one to resolve to an emitted file from
    the current page location.
    """
    page = synthetic_site / "docs" / "getting-started" / "index.html"
    html = _read(page)
    nav_block_match = re.search(r'<nav class="site-nav"[^>]*>(.*?)</nav>', html, re.DOTALL)
    assert nav_block_match, "site-nav block not found"
    nav_hrefs = re.findall(r'href="([^"]+)"', nav_block_match.group(1))
    assert nav_hrefs, "expected at least one sidebar link"
    for href in nav_hrefs:
        target = (page.parent / href).resolve()
        if href.endswith("/") or target.is_dir():
            target = target / "index.html"
        assert target.exists(), f"nav href {href!r} points at missing file {target}"


def test_generated_local_links_resolve(synthetic_site: Path) -> None:
    """The site-level link verifier catches local broken links."""
    assert validate_site_links(synthetic_site) == []


def test_markdown_links_rewritten_for_site_and_repo_files(synthetic_site: Path) -> None:
    """Markdown links point at built site pages or stable source blobs."""
    html = _read(synthetic_site / "docs" / "getting-started" / "index.html")
    hrefs = _collect_links(html).hrefs
    assert "../search/#search-details" in hrefs
    assert "https://github.com/Sinity/polylogue/blob/master/README.md" in hrefs
    assert "http://CLAUDE.md" not in hrefs


def test_no_unresolved_template_placeholders(synthetic_site: Path) -> None:
    """Rendered pages never contain raw ``{{`` Jinja markers — those indicate
    template variables that were not substituted.
    """
    for path in synthetic_site.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert "{{" not in text, f"unresolved Jinja in {path}: {text[:200]!r}"
        assert "{%" not in text, f"unresolved Jinja block in {path}: {text[:200]!r}"


# ---------------------------------------------------------------------------
# Asset references
# ---------------------------------------------------------------------------


def test_inline_stylesheet_embedded(synthetic_site: Path) -> None:
    """The renderer inlines a stylesheet via ``<style>`` rather than
    linking to an external CSS file, so the built site is self-contained.
    """
    html = _read(synthetic_site / "index.html")
    assert "<style>" in html
    # PAGES_STYLE has CSS variables for theming; spot-check one.
    assert re.search(r":root|--bg|body\s*\{", html), "expected inline CSS body"


def test_external_github_link_present(synthetic_site: Path) -> None:
    """Footer carries an absolute external link; not all links are relative."""
    html = _read(synthetic_site / "index.html")
    hrefs = _collect_links(html).hrefs
    assert any(h.startswith("https://github.com/") for h in hrefs)


# ---------------------------------------------------------------------------
# Frontmatter / per-page data
# ---------------------------------------------------------------------------


def test_home_page_does_not_publish_operator_archive_counts(synthetic_site: Path) -> None:
    """The public landing page describes capability, not a private archive."""
    html = _read(synthetic_site / "index.html")
    assert "sessions</span>" not in html
    assert "messages</span>" not in html
    assert "origins</span>" not in html


def test_per_page_data_parsed_and_passed_to_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``[pages.data]`` entries reach the template namespace.

    Verifies both directions:

    * ``load_pages_config`` carries the ``[pages.data]`` mapping onto
      ``PageEntry.data``.
    * ``build_site`` passes that mapping into the Jinja template render
      context. We capture the captured ``template_data`` via a spy on
      ``Environment.get_template`` rather than asserting on rendered HTML,
      because the production ``verifiability_catalog`` template does not
      currently render every supplied key — but the renderer must still
      pass them through, so later template changes can consume them.
    """
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(pages_builder, "ROOT", fake_root)
    config_path = fake_root / "pages.toml"
    config_path.write_text(
        """
site_name = "Polylogue"
site_url = "/polylogue/"

[[pages]]
path = "/architecture/contributing/"
title = "Contributing"
template = "doc.html"
[pages.data]
source_command = "render cli-reference"
custom_marker = "frontmatter-roundtrip-ok"
""",
        encoding="utf-8",
    )

    cfg = load_pages_config(config_path)
    page_entry = next(p for p in cfg.pages if p.path == "/architecture/contributing/")
    assert page_entry.data == {
        "source_command": "render cli-reference",
        "custom_marker": "frontmatter-roundtrip-ok",
    }

    # Spy on Template.render to capture the render context.
    captured: dict[str, object] = {}
    from jinja2 import Template

    real_render = Template.render

    def _spy(self: Template, *args: object, **kwargs: object) -> str:
        if self.name == "doc.html":
            captured.update(kwargs)
        return real_render(self, *args, **kwargs)

    monkeypatch.setattr(Template, "render", _spy)

    out = tmp_path / "out"
    build_site(config_path=config_path, output_dir=out)

    assert captured.get("source_command") == "render cli-reference"
    assert captured.get("custom_marker") == "frontmatter-roundtrip-ok"
    assert captured.get("title") == "Contributing"
    assert captured.get("current_path") == "/architecture/contributing/"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_load_pages_config_round_trip(tmp_path: Path) -> None:
    """The TOML loader produces a typed ``PagesConfig`` with nav/pages."""
    config_path = tmp_path / "pages.toml"
    config_path.write_text(_MINIMAL_CONFIG, encoding="utf-8")
    cfg = load_pages_config(config_path)
    assert isinstance(cfg, PagesConfig)
    assert cfg.site_name == "Polylogue"
    assert [s.title for s in cfg.nav] == ["Docs"]
    assert [i.label for i in cfg.nav[0].items] == ["Getting Started", "Search"]
    assert [p.path for p in cfg.pages] == [
        "/index.html",
        "/beads/",
        "/demos/",
        "/docs/",
        "/docs/getting-started/",
        "/docs/search/",
    ]


# ---------------------------------------------------------------------------
# Error / fallback paths
# ---------------------------------------------------------------------------


def test_missing_source_file_produces_placeholder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a page's ``source`` does not exist on disk, the renderer
    emits a placeholder comment instead of raising. This keeps the site
    buildable when a docs file is renamed or deleted before pages.toml is
    updated — drift surfaces as a visible HTML comment.
    """
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(pages_builder, "ROOT", fake_root)
    config_path = fake_root / "pages.toml"
    config_path.write_text(
        """
site_name = "Polylogue"
site_url = "/polylogue/"

[[pages]]
path = "/index.html"
title = "Home"
template = "home.html"

[[pages]]
path = "/docs/ghost/"
title = "Ghost"
source = "docs/does-not-exist.md"
""",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    build_site(config_path=config_path, output_dir=out)
    html = _read(out / "docs" / "ghost" / "index.html")
    assert "Source not found: docs/does-not-exist.md" in html


def test_unknown_template_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuring a page with a template not in ``PAGES_TEMPLATES``
    raises a Jinja ``TemplateNotFound``. Silent fallback would hide
    typos like ``hom.html`` until a reader noticed the missing page.
    """
    from jinja2 import TemplateNotFound

    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(pages_builder, "ROOT", fake_root)
    config_path = fake_root / "pages.toml"
    config_path.write_text(
        """
site_name = "Polylogue"
site_url = "/polylogue/"

[[pages]]
path = "/oops/"
title = "Oops"
template = "this-template-does-not-exist.html"
""",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    with pytest.raises(TemplateNotFound):
        build_site(config_path=config_path, output_dir=out)


def test_malformed_toml_raises(tmp_path: Path) -> None:
    """A pages.toml that is not valid TOML surfaces as ``TOMLDecodeError``
    from the loader. We do not swallow it — bad config must fail loudly.
    """
    import tomllib

    config_path = tmp_path / "pages.toml"
    config_path.write_text("this = is = not = toml\n", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_pages_config(config_path)


def test_build_site_recreates_output_directory(
    synthetic_site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second build wipes and recreates the output directory so stale
    files from a previous build cannot linger.
    """
    stale = synthetic_site / "stale-file.txt"
    stale.write_text("leftover", encoding="utf-8")
    assert stale.exists()

    # Rebuild with the same fake root layout the fixture set up.
    fake_root = pages_builder.ROOT
    config_path = fake_root / "pages.toml"
    build_site(config_path=config_path, output_dir=synthetic_site)
    assert not stale.exists(), "second build should wipe stale files"
    assert (synthetic_site / "index.html").is_file()
