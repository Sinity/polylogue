"""Walk a ChatGPT export and report which keys it states, by scope.

Used by ``tests/unit/sources/test_chatgpt_field_ledger.py`` to hold the
parser's declared field register (``CHATGPT_READ_KEYS`` /
``CHATGPT_EXCLUDED_KEYS``) against the repository's committed synthetic
exports.  The walker knows only the export's shape -- top-level record,
``mapping`` node, message envelope, and the two free-form metadata bags -- and
deliberately knows nothing about which keys are interesting, so a new upstream
field appears here the moment a fixture carries it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def census_export_keys(conversation: Mapping[str, object]) -> dict[str, set[str]]:
    """Return ``scope -> keys stated`` for one ChatGPT conversation record."""

    observed: dict[str, set[str]] = {
        "conversation": set(conversation),
        "node": set(),
        "message": set(),
        "message.metadata": set(),
        "author": set(),
        "author.metadata": set(),
    }
    mapping = conversation.get("mapping")
    nodes = mapping.values() if isinstance(mapping, Mapping) else ()
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        observed["node"] |= set(node)
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        observed["message"] |= set(message)
        metadata = message.get("metadata")
        if isinstance(metadata, Mapping):
            observed["message.metadata"] |= set(metadata)
        author = message.get("author")
        if isinstance(author, Mapping):
            observed["author"] |= set(author)
            author_metadata = author.get("metadata")
            if isinstance(author_metadata, Mapping):
                observed["author.metadata"] |= set(author_metadata)
    return observed


def census_export_file(path: Path) -> dict[str, set[str]]:
    """Census one committed export file, conversation-list or single record."""

    payload = json.loads(path.read_text())
    records = payload if isinstance(payload, list) else [payload]
    observed: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for scope, keys in census_export_keys(record).items():
            observed.setdefault(scope, set()).update(keys)
    return observed


__all__ = ["census_export_file", "census_export_keys"]
