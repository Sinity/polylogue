"""Incremental document encoding over an owned, repeatable message source."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping


def message_document_chunks(
    metadata: Mapping[str, object],
    messages: Iterable[Mapping[str, object]] | None,
    output_format: str,
    *,
    sort_keys: bool = False,
    compact: bool = False,
    arrays: Mapping[str, Iterable[object]] | None = None,
) -> Iterator[str]:
    """Keep one semantic message resident while encoding its complete document."""
    relations = dict(arrays or {})
    if messages is not None:
        relations["messages"] = messages
    keys = list(metadata)
    for key in relations:
        if key not in keys:
            keys.append(key)
    if sort_keys:
        keys.sort()
    if output_format == "yaml":
        import yaml

        for key in keys:
            if key not in relations:
                yield yaml.safe_dump({key: metadata[key]}, allow_unicode=True, sort_keys=False)
                continue
            seen = False
            for message in relations[key]:
                if not seen:
                    yield f"{key}:\n"
                    seen = True
                yield yaml.safe_dump([message], allow_unicode=True, sort_keys=False)
            if not seen:
                yield f"{key}: []\n"
        return
    encoder = json.JSONEncoder(
        indent=None if compact else 2, sort_keys=sort_keys, separators=(",", ":") if compact else None
    )
    yield "{" if compact else "{\n"
    for key_index, key in enumerate(keys):
        if key_index:
            yield "," if compact else ",\n"
        yield ("" if compact else "  ") + json.dumps(key) + (":" if compact else ": ")
        if key not in relations:
            # Indentation depends only on this field, not on preceding messages.
            chunks = encoder.iterencode(metadata[key])
            for chunk in chunks:
                yield chunk if compact else chunk.replace("\n", "\n  ")
            continue
        yield "["
        seen = False
        for message_index, message in enumerate(relations[key]):
            seen = True
            if message_index:
                yield ","
            if not compact:
                yield "\n    "
            for chunk in encoder.iterencode(message):
                yield chunk if compact else chunk.replace("\n", "\n    ")
        yield "\n  ]" if seen and not compact else "]"
    yield "}" if compact else "\n}"


def stripped_chunks(chunks: Iterable[str], *, leading: bool = True, characters: str | None = None) -> Iterator[str]:
    """Trim document edges without retaining an arbitrarily long whitespace tail."""
    from tempfile import TemporaryFile

    started = not leading
    with TemporaryFile(mode="w+t", encoding="utf-8", newline="") as pending:
        for chunk in chunks:
            if not started:
                chunk = chunk.lstrip(characters)
                if not chunk:
                    continue
                started = True
            body = chunk.rstrip(characters)
            if body:
                pending.seek(0)
                while held := pending.read(65536):
                    yield held
                pending.seek(0)
                pending.truncate()
                yield body
            pending.write(chunk[len(body) :])
