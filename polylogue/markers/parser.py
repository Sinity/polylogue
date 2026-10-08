r"""Streaming-safe marker grammar.

The sigil is ``::`` for line markers (``::kind(args): body``) and
``[[kind: body]]`` for inline markers.  Three constraints keep the sigil from
claiming ordinary prose, and each is load-bearing rather than stylistic:

- a line marker must be line-anchored, because a live read-only corpus scan
  found ``::kind:`` occurring mid-prose inside captured tool data, which rules
  out treating an unanchored prefix as structure;
- fenced Markdown blocks are skipped, so quoted or illustrative text is never
  parsed as a marker;
- prose that legitimately begins with ``::`` escapes it as ``\::``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from io import StringIO

from polylogue.markers.models import MarkerMatch
from polylogue.markers.registry import MARKER_REGISTRY, MarkerRegistry, marker_spec

_LINE = re.compile(r"^(?P<indent>[ \t]*)::(?P<kind>[a-z][a-z0-9_-]*)(?:\((?P<args>[^)]*)\))?:[ \t]*(?P<body>.*)$")
_INLINE = re.compile(r"\[\[(?P<kind>[a-z][a-z0-9_-]*):[ \t]*(?P<body>[^\]]*?)\]\]")
_INLINE_OPEN = re.compile(r"\[\[(?P<kind>[a-z][a-z0-9_-]*):[ \t]*")
_MALFORMED = re.compile(r"^[ \t]*::")
_FENCE = re.compile(r"^[ ]{0,3}(?P<delimiter>`{3,}|~{3,})")


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
    return tuple(iter_parse_markers(text, registry=registry))


def iter_parse_markers(text: str, *, registry: MarkerRegistry = MARKER_REGISTRY) -> Iterator[MarkerMatch]:
    """Yield markers with memory bounded by one physical line."""
    offset = 0
    fence: tuple[str, int] | None = None
    for line in StringIO(text, newline=""):
        fence_match = _FENCE.match(line.rstrip("\r\n"))
        if fence_match:
            delimiter = fence_match.group("delimiter")
            if fence is None:
                fence = (delimiter[0], len(delimiter))
            elif (
                delimiter[0] == fence[0]
                and len(delimiter) >= fence[1]
                and not line.rstrip("\r\n")[fence_match.end() :].strip()
            ):
                fence = None
            offset += len(line)
            continue
        if fence is None and not line.lstrip().startswith(r"\::"):
            line_match = _LINE.match(line.rstrip("\r\n"))
            if line_match:
                kind = line_match.group("kind")
                registered = marker_spec(registry, kind) is not None
                yield MarkerMatch(
                    kind if registered else "malformed",
                    line_match.group("body"),
                    _args(line_match.group("args")) if registered else {"unregistered_kind": kind},
                    line.rstrip("\r\n"),
                    offset,
                    offset + len(line.rstrip("\r\n")),
                    malformed=not registered,
                )
            elif _MALFORMED.match(line):
                yield MarkerMatch(
                    "malformed",
                    line.strip(),
                    {},
                    line.rstrip("\r\n"),
                    offset,
                    offset + len(line.rstrip("\r\n")),
                    malformed=True,
                )
            for inline in _INLINE.finditer(line):
                kind = inline.group("kind")
                yield MarkerMatch(
                    kind if kind in registry else "malformed",
                    inline.group("body"),
                    {} if kind in registry else {"unregistered_kind": kind},
                    inline.group(0),
                    offset + inline.start(),
                    offset + inline.end(),
                    inline=True,
                    malformed=kind not in registry,
                )
            raw_line = line.rstrip("\r\n")
            accepted_spans = iter((inline.start(), inline.end()) for inline in _INLINE.finditer(line))
            accepted_span = next(accepted_spans, None)
            for inline in _INLINE_OPEN.finditer(raw_line):
                while accepted_span is not None and accepted_span[1] <= inline.start():
                    accepted_span = next(accepted_spans, None)
                if accepted_span is not None and accepted_span[0] <= inline.start() < accepted_span[1]:
                    # Already covered by an accepted inline span; a second,
                    # overlapping malformed marker would contradict it.
                    continue
                body_start = inline.end()
                next_open = raw_line.find("[[", body_start)
                close = raw_line.find("]]", body_start)
                if close >= 0 and (next_open < 0 or close < next_open):
                    continue
                end = next_open if next_open >= 0 else len(raw_line)
                yield MarkerMatch(
                    "malformed",
                    raw_line[body_start:end],
                    {"unregistered_kind": inline.group("kind")},
                    raw_line[inline.start() : end],
                    offset + inline.start(),
                    offset + end,
                    inline=True,
                    malformed=True,
                )
        offset += len(line)


class MarkerStreamParser:
    """Buffer incomplete final lines so a split marker is parsed once."""

    def __init__(self, *, registry: MarkerRegistry = MARKER_REGISTRY) -> None:
        self.registry = registry
        self._buffer = ""
        self._consumed = 0

    def _absolute(self, matches: tuple[MarkerMatch, ...], base: int) -> tuple[MarkerMatch, ...]:
        return tuple(
            MarkerMatch(
                m.kind,
                m.body,
                m.arguments,
                m.raw_text,
                m.start + base,
                m.end + base,
                inline=m.inline,
                malformed=m.malformed,
            )
            for m in matches
        )

    def feed(self, chunk: str) -> tuple[MarkerMatch, ...]:
        self._buffer += chunk
        complete, sep, remainder = self._buffer.rpartition("\n")
        if not sep:
            return ()
        self._buffer = remainder
        emitted = parse_markers(complete + "\n", registry=self.registry)
        result = self._absolute(emitted, self._consumed)
        self._consumed += len(complete) + 1
        return result

    def finish(self) -> tuple[MarkerMatch, ...]:
        result = (
            self._absolute(parse_markers(self._buffer, registry=self.registry), self._consumed) if self._buffer else ()
        )
        self._consumed += len(self._buffer)
        self._buffer = ""
        return result
