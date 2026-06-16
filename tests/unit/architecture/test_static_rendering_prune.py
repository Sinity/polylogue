"""Regression checks for #1848 static-rendering surface pruning."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_legacy_static_product_template_files_are_removed() -> None:
    """The old standalone static-site template files are not a product surface."""

    legacy_templates = ROOT / "polylogue" / "templates"
    assert not (legacy_templates / "index.html").exists()
    assert not (legacy_templates / "modern.html").exists()
    assert not (legacy_templates / "style.css").exists()


def test_read_export_html_template_is_retained() -> None:
    """HTML output remains available through the read/export renderer path."""

    assert (ROOT / "polylogue" / "rendering" / "templates" / "session.html").is_file()


def test_legacy_qa_report_aggregator_is_removed() -> None:
    """QA report behavior lives in owned payload, Markdown, and summary modules."""

    assert not (ROOT / "polylogue" / "showcase" / "qa_report.py").exists()
