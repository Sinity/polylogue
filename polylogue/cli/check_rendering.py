"""Rendering helpers for the check command."""

from __future__ import annotations

from polylogue.cli.check_rendering_json import emit_json_output
from polylogue.cli.check_rendering_plain import render_plain_output

__all__ = ["emit_json_output", "render_plain_output"]
