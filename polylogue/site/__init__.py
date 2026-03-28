"""Static site generation for polylogue archives.

This package provides tools to generate a complete static HTML site
from a polylogue conversation archive, including:
- Root index with recent conversations
- Per-provider and per-date index pages
- Dashboard with archive statistics
- Client-side search (pagefind or lunr.js)
"""

from polylogue.site.builder import SiteBuilder, SiteConfig

__all__ = ["SiteBuilder", "SiteConfig"]
