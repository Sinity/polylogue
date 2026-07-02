"""Polylogue hook adapter — AI coding agent session lifecycle capture.

This package ships the ``polylogue-hook`` console script as a pure-stdlib
Python implementation so it can be installed without pulling the full
``polylogue`` runtime closure. The bash equivalent under
``contrib/polylogue-hook`` in the main repository remains supported for
operators who prefer a shell-only install path.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.2.0"  # x-release-please-version
