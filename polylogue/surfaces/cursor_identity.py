"""Ranked-search request identity, isolated from the heavy payload models.

Kept dependency-free (stdlib only) so the daemon-served CLI fast path can mint
a cursor identity without importing ``polylogue.surfaces.payloads`` and its
pydantic/archive-model import cost.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

__all__ = ["search_cursor_request_identity"]


def search_cursor_request_identity(arguments: Mapping[str, object]) -> str:
    """Return a stable identity for the logical ranked-search request."""
    presentation_only = {
        "cursor",
        "offset",
        "limit",
        "output",
        "output_destination",
        "out_path",
        "output_format",
        "fields",
        "verbose",
        "plain",
        "no_daemon",
        "explain_query",
        "why",
        "output_as_json",
        "diagnose",
        "stream",
        "print_url",
    }
    canonical = {key: value for key, value in arguments.items() if key not in presentation_only}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
