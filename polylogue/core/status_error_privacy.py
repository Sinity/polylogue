"""Privacy projection for unstructured status failure diagnostics."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

# Match only at the start of a scheme token. Failed searches do not retry at
# every character of a long token. A local path consumes its tail before URL
# recognition can inspect a substring of that path.
_URL = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>\\]+")
# In unstructured prose these separators can introduce a local path as well
# as belong to a URI. Without producer-owned URL evidence, conceal that suffix.
_URL_LOCAL_SUFFIX = re.compile(r"[,;|:=()\[\]{}](?=/|\\|[A-Za-z]:[/\\])")
_DIAGNOSTIC_START = " \t\"'([{=:;"
_DIAGNOSTIC_END = " \t\"')]}=:;"


def redact_status_error(value: object, *, relative_path_spans: Sequence[tuple[int, int]] = ()) -> str:
    """Conceal ambiguous slash prose while retaining standalone network URLs.

    A slash in arbitrary exception prose carries no evidence that it is
    relative, even after a letter. Relative exemptions require exact spans
    declared by the diagnostic producer. URL exemptions require a parsed
    scheme and authority outside an already-started local path.
    """
    if not isinstance(value, str):
        if relative_path_spans:
            raise ValueError("relative path spans require a diagnostic string")
        return ""
    relative_spans = sorted(relative_path_spans)
    previous_end = 0
    for start, end in relative_spans:
        path = value[start:end]
        if (
            not (0 <= start < end <= len(value))
            or not path
            or PurePosixPath(path).is_absolute()
            or PureWindowsPath(path).anchor
        ):
            raise ValueError("relative path declaration must name an exact relative span")
        if start < previous_end:
            raise ValueError("relative diagnostic spans must not overlap")
        if any(character in path for character in "\n\r\"'") or "://" in path:
            raise ValueError("relative path declaration must contain a path, not diagnostic structure")
        if start and value[start - 1] not in _DIAGNOSTIC_START:
            raise ValueError("relative path declaration must begin at a diagnostic boundary")
        if end < len(value) and value[end] not in _DIAGNOSTIC_END:
            raise ValueError("relative path declaration must end at a diagnostic boundary")
        previous_end = end
    relative_ends = dict(relative_spans)
    relative_starts = [start for start, _ in relative_spans]
    parts: list[str] = []
    position = 0
    quote: str | None = None
    escaped = False
    while position < len(value):
        relative_end = relative_ends.get(position)
        if relative_end is not None:
            parts.append(value[position:relative_end])
            position = relative_end
            continue
        character = value[position]
        # A drive path with repeated forward separators (C://...) also
        # parses as a one-letter URL scheme. Its filesystem ownership wins.
        drive_path = (
            character.isascii()
            and character.isalpha()
            and value[position + 1 : position + 2] == ":"
            and value[position + 2 : position + 3] in {"/", "\\"}
        )
        match = None if drive_path else _URL.match(value, position)
        if match is not None:
            url = None
            try:
                parsed = urlsplit(match.group())
                if parsed.scheme and parsed.netloc and parsed.hostname and parsed.scheme != "file":
                    url = parsed
            except ValueError:
                pass
            authority_start = value.index("://", position, match.end()) + 3
            local_suffix = _URL_LOCAL_SUFFIX.search(value, authority_start, min(len(value), match.end() + 1))
            # A validated bracketed IP authority owns its closing bracket.
            # An invalid whole token can still have an independently valid
            # network prefix before a diagnostic bracket and local-path tail.
            if url is not None and local_suffix is not None:
                authority_end = authority_start + len(url.netloc)
                if local_suffix.start() < authority_end and local_suffix.group() == "]":
                    local_suffix = _URL_LOCAL_SUFFIX.search(value, authority_end, min(len(value), match.end() + 1))
            url_end = local_suffix.start() if local_suffix is not None else match.end()
            if local_suffix is not None:
                url = None
                try:
                    parsed = urlsplit(value[position:url_end])
                    if parsed.scheme and parsed.netloc and parsed.hostname and parsed.scheme != "file":
                        url = parsed
                except ValueError:
                    pass
            if url is not None:
                parts.append(value[position:url_end])
                position = url_end
                continue
        if character in {"/", "\\"} or drive_path:
            # Unquoted text has no filename terminator. Inside a quoted
            # diagnostic, only its enclosing (unescaped) quote terminates it.
            end = position + 1
            tail_escaped = False
            while end < len(value):
                tail_character = value[end]
                if tail_character in "\r\n" or (quote is not None and tail_character == quote and not tail_escaped):
                    break
                tail_escaped = tail_character == "\\" and not tail_escaped
                end += 1
            parts.append("[redacted]")
            # Producer-owned relative evidence remains valid even alongside
            # an ambiguous local-path tail. No inferred URL is exempted here.
            span_index = bisect_right(relative_starts, position)
            while span_index < len(relative_spans) and relative_spans[span_index][0] < end:
                relative_start, relative_end = relative_spans[span_index]
                parts.append(value[relative_start:relative_end])
                if relative_end < end:
                    parts.append("[redacted]")
                span_index += 1
            position = end
            escaped = False
            continue
        parts.append(character)
        if character in "\r\n" or character == quote and not escaped:
            quote = None
        elif quote is None and character in "\"'" and (position == 0 or value[position - 1] in _DIAGNOSTIC_START):
            quote = character
        escaped = character == "\\" and not escaped
        position += 1
    return "".join(parts)
