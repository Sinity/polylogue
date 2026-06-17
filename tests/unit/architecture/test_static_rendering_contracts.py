"""Static-rendering contracts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_read_export_html_template_is_retained() -> None:
    """HTML output remains available through the read/export renderer path."""

    assert (ROOT / "polylogue" / "rendering" / "templates" / "session.html").is_file()
