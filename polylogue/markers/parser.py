"""Streaming-safe marker grammar."""

from __future__ import annotations

import re

from polylogue.markers.models import MarkerMatch
from polylogue.markers.registry import MARKER_REGISTRY, MarkerRegistry, marker_spec

_LINE = re.compile(r"^(?P<indent>[ \t]*)::(?P<kind>[a-z][a-z0-9_-]*)(?:\((?P<args>[^)]*)\))?:[ \t]*(?P<body>.*)$")
_INLINE = re.compile(r"\[\[(?P<kind>[a-z][a-z0-9_-]*):[ \t]*(?P<body>[^\]]*?)\]\]")
_MALFORMED = re.compile(r"^[ \t]*::")


def _args(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    result: dict[str, str] = {}
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            result[str(len(result))] = item.strip()
        else:
            result[key.strip()] = value.strip()
    return result


def parse_markers(text: str, *, registry: MarkerRegistry = MARKER_REGISTRY) -> tuple[MarkerMatch, ...]:
    """Extract declared and malformed markers, preserving offsets and raw text."""
    matches: list[MarkerMatch] = []
    offset = 0
    fenced = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            offset += len(line)
            continue
        if not fenced and not line.lstrip().startswith(r"\::"):
            line_match = _LINE.match(line.rstrip("\r\n"))
            if line_match:
                kind = line_match.group("kind")
                registered = marker_spec(registry, kind) is not None
                matches.append(
                    MarkerMatch(
                        kind if registered else "malformed",
                        line_match.group("body"),
                        _args(line_match.group("args")) if registered else {"unregistered_kind": kind},
                        line.rstrip("\r\n"),
                        offset,
                        offset + len(line.rstrip("\r\n")),
                        malformed=not registered,
                    )
                )
            elif _MALFORMED.match(line):
                matches.append(
                    MarkerMatch(
                        "malformed",
                        line.strip(),
                        {},
                        line.rstrip("\r\n"),
                        offset,
                        offset + len(line.rstrip("\r\n")),
                        malformed=True,
                    )
                )
            for inline in _INLINE.finditer(line):
                if inline.group("kind") not in registry:
                    continue
                matches.append(
                    MarkerMatch(
                        inline.group("kind"),
                        inline.group("body"),
                        {},
                        inline.group(0),
                        offset + inline.start(),
                        offset + inline.end(),
                        inline=True,
                    )
                )
        offset += len(line)
    return tuple(matches)


class MarkerStreamParser:
    """Buffer incomplete final lines so a split marker is parsed once."""

    def __init__(self, *, registry: MarkerRegistry = MARKER_REGISTRY) -> None:
        self.registry = registry
        self._buffer = ""

    def feed(self, chunk: str) -> tuple[MarkerMatch, ...]:
        self._buffer += chunk
        complete, sep, remainder = self._buffer.rpartition("\n")
        if not sep:
            return ()
        self._buffer = remainder
        return parse_markers(complete + "\n", registry=self.registry)

    def finish(self) -> tuple[MarkerMatch, ...]:
        result = parse_markers(self._buffer, registry=self.registry) if self._buffer else ()
        self._buffer = ""
        return result
