"""Anonymous spellings of measured, public Codex call and result grammar.

Only fixed syntax tokens and anonymous run widths leave the source reducer.
Paths, command bodies and source identifiers are never representative values.
Large operands remain numeric runs until the existing wire serializer draws them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping

from polylogue.sources.parsers.codex import (
    _CODEX_CHUNK_ID_PREFIX_RE,
    _CODEX_COMMAND_LINE_PREFIX_RE,
    _CODEX_EXEC_ENVELOPE_OUTCOME_RE,
    _CODEX_FREEFORM_OUTCOME_RE,
    _skip_js_string_or_comment,
)

PUBLIC_TOOL_NAMES = frozenset({"apply_patch", "patch", "exec", "exec_command", "shell_command"})
_WORDS = PUBLIC_TOOL_NAMES | {"await", "tools", "functions", "cmd", "command", "input"}
_PATCH_PREFIXES = (
    "*** Begin Patch",
    "*** End Patch",
    "*** Add File: ",
    "*** Update File: ",
    "*** Delete File: ",
    "*** Move to: ",
    "*** Move File: ",
    "@@",
    "+",
    "-",
    " ",
)
_PUNCTUATION = frozenset("(){}[]:;,.=\"'` \t\r\n")
_ESCAPES = frozenset({"\\n", "\\r", "\\t", "\\\\", '\\"', "\\'", "\\`"})
_RESULT_LITERALS = frozenset(
    {
        "Command: ",
        "Chunk ID: ",
        "Wall time: ",
        " seconds\n",
        "Process exited with code ",
        "Process completed with exit code ",
        "Process running with session ID ",
        "Exit code: ",
        "\nWall time: ",
        "Output:\n",
        "\nOutput:\n",
        "Total output lines: ",
        "-",
    }
)
_INPUT_LITERALS = _WORDS | set(_PATCH_PREFIXES) | _PUNCTUATION | _ESCAPES | {"//", "/*", "*/"}
_LITERALS = _RESULT_LITERALS | _INPUT_LITERALS
_RUNS = re.compile(r"[^\x00-\x7f]+|[^\n\r\x80-\U0010ffff]+|[\n\r]")
_CODE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|[(){}\[\]:;,.= \t\r\n]|[^A-Za-z_$(){}\[\]:;,.= \t\r\n\"'`]+")
_LINES = re.compile(r"\\n|\n")
_STRING_ESCAPES = re.compile(r"\\(?:[nrt\\\"'`]|u[0-9a-fA-F]{4})")
_OBJECT_KEY_END = re.compile(r"\s*:")


def _anonymous(parts: list[list[object]], value: str, start: int, stop: int) -> None:
    # Character ordering inside an anonymous hole is private. Retain counts,
    # not one recipe row per alternating character or an operand-sized copy.
    ascii_width = non_ascii_width = 0

    def flush() -> None:
        for kind, width in (("ascii", ascii_width), ("non_ascii", non_ascii_width)):
            if not width:
                continue
            if parts and parts[-1][0] == kind:
                prior = parts[-1][1]
                assert isinstance(prior, int)
                parts[-1][1] = prior + width
            else:
                parts.append([kind, width])

    for match in _RUNS.finditer(value, start, stop):
        width = match.end() - match.start()
        token = value[match.start()] if width == 1 else ""
        if token in {"\n", "\r"}:
            flush()
            ascii_width = non_ascii_width = 0
            parts.append(["literal", token])
        elif ord(value[match.start()]) < 128:
            ascii_width += width
        else:
            non_ascii_width += width
    flush()


def _patch_line(parts: list[list[object]], value: str, start: int, stop: int) -> None:
    prefix = next((prefix for prefix in _PATCH_PREFIXES if value.startswith(prefix, start, stop)), "")
    if prefix:
        parts.append(["literal", prefix])
    _anonymous(parts, value, start + len(prefix), stop)


def _patch(parts: list[list[object]], value: str, start: int, stop: int) -> None:
    for match in _LINES.finditer(value, start, stop):
        _patch_line(parts, value, start, match.start())
        parts.append(["literal", match.group()])
        start = match.end()
    _patch_line(parts, value, start, stop)


def _quoted(parts: list[list[object]], value: str, start: int, stop: int, *, patch_argument: bool) -> None:
    quote = value[start]
    escape = stop - 2
    while escape > start and value[escape] == "\\":
        escape -= 1
    closed = stop > start + 1 and value[stop - 1] == quote and (stop - 2 - escape) % 2 == 0
    parts.append(["literal", quote])
    end = stop - 1 if closed else stop
    word = value[start + 1 : end] if end - start - 1 <= max(map(len, _WORDS)) else ""
    if closed and word in {"cmd", "command", "input", "patch"} and _OBJECT_KEY_END.match(value, stop):
        parts.append(["literal", word])
    elif patch_argument and value.startswith("*** Begin Patch", start + 1, end):
        _patch(parts, value, start + 1, end)
    else:
        position = start + 1
        for match in _STRING_ESCAPES.finditer(value, position, end):
            _anonymous(parts, value, position, match.start())
            escape_token = match.group()
            if escape_token in _ESCAPES:
                parts.append(["literal", escape_token])
            else:
                _anonymous(parts, value, match.start(), match.end())
            position = match.end()
        _anonymous(parts, value, position, end)
    if closed:
        parts.append(["literal", quote])


def measured_tool_syntax(tool: Mapping[str, object]) -> str:
    """Return a coupled public name/field/syntax observation, never raw content."""
    name = tool.get("name")
    if not isinstance(name, str) or name not in PUBLIC_TOOL_NAMES:
        return ""
    result: dict[str, object] = {"name": name}
    for field in ("arguments", "input"):
        value = tool.get(field)
        if not isinstance(value, str):
            continue
        parts: list[list[object]] = []
        if name in {"apply_patch", "patch"}:
            _patch(parts, value, 0, len(value))
        elif name == "exec":
            position = 0
            recent: list[str] = []
            patch_argument = False
            static_call = False
            while position < len(value):
                if value[position] == "/":
                    end = _skip_js_string_or_comment(value, position)
                    if end is not None:
                        if value.startswith(("//", "/*"), position):
                            parts.append(["literal", value[position : position + 2]])
                            closed_comment = value.startswith("/*", position) and value.startswith("*/", end - 2)
                            _anonymous(parts, value, position + 2, end - 2 if closed_comment else end)
                            if closed_comment:
                                parts.append(["literal", "*/"])
                        else:
                            _anonymous(parts, value, position, end)
                        position = end
                        continue
                if value[position] in {'"', "'", "`"}:
                    end = _skip_js_string_or_comment(value, position)
                    assert end is not None
                    _quoted(parts, value, position, end, patch_argument=patch_argument)
                    patch_argument = False
                    recent.append("")
                    recent = recent[-3:]
                    position = end
                    continue
                match = _CODE.match(value, position)
                if (
                    match is not None
                    and match.end() - position <= max(map(len, _INPUT_LITERALS))
                    and match.group() in _INPUT_LITERALS
                ):
                    token = match.group()
                    parts.append(["literal", token])
                    if not token.isspace():
                        known_call = (
                            token == "("
                            and len(recent) == 3
                            and recent[0] in {"tools", "functions"}
                            and recent[1] == "."
                            and recent[2] in PUBLIC_TOOL_NAMES
                        )
                        static_call = static_call or known_call
                        patch_argument = known_call and recent[2] in {"apply_patch", "patch"}
                        recent.append(token)
                        recent = recent[-3:]
                    position = match.end()
                else:
                    end = match.end() if match is not None else position + 1
                    _anonymous(parts, value, position, end)
                    patch_argument = False
                    recent.append("")
                    recent = recent[-3:]
                    position = end
            if not static_call:
                parts = []
        if parts:
            result[field] = parts
    for field in ("arguments", "input"):
        if field in result:
            for _ in validated_syntax_parts(result[field]):
                pass
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def validated_syntax_parts(parts: object) -> Iterator[tuple[str, str | int]]:
    if not isinstance(parts, list):
        raise ValueError("tool syntax must contain anonymous runs")
    for part in parts:
        if not isinstance(part, list) or len(part) != 2 or not isinstance(part[0], str):
            raise ValueError("tool syntax run is malformed")
        kind, value = part
        if kind == "literal" and isinstance(value, str) and value in _LITERALS:
            yield "literal", value
        elif kind in {"ascii", "non_ascii", "digits", "zeros"} and type(value) is int and value > 0:
            yield kind, value
        else:
            raise ValueError("tool syntax contains a private or unsupported token")


# Reuse the production header grammar at an explicit offset without copying
# the potentially huge result body merely to remove a prefix.
_CHUNK_PREFIX = re.compile(_CODEX_CHUNK_ID_PREFIX_RE.pattern.removeprefix(r"\A"))
_EXEC_HEADER = re.compile(_CODEX_EXEC_ENVELOPE_OUTCOME_RE.pattern.removeprefix(r"\A"))
_FREEFORM_HEADER = re.compile(_CODEX_FREEFORM_OUTCOME_RE.pattern.removeprefix(r"\A"))
_RESULT_TOKEN = re.compile(
    r"[0-9]+|" + "|".join(re.escape(token) for token in sorted(_RESULT_LITERALS | {".", "\n"}, key=len, reverse=True))
)
OUTCOME_WRAPPERS = ("", "metadata.", "result.", "output.")
OUTCOME_FIELDS = ("exit_code", "is_error", "timed_out")


def codex_tool_class(kind: str, name: object) -> str:
    """Pair the public call family with a public or anonymous name class."""
    call = {"function_call_output": "function_call", "custom_tool_call_output": "custom_tool_call"}.get(kind, kind)
    if call not in {"function_call", "custom_tool_call"}:
        raise ValueError("Codex tool class has no declared call family")
    return call + ":" + (name if isinstance(name, str) and name in PUBLIC_TOOL_NAMES else "")


def measured_result_syntax(value: object) -> dict[str, object]:
    """Keep fixed result grammar and verdict classes without private values."""
    result: dict[str, object] = {}
    if isinstance(value, Mapping):
        outcomes: dict[str, str] = {}
        for prefix in OUTCOME_WRAPPERS:
            wrapper = value if not prefix else value.get(prefix[:-1])
            for field in OUTCOME_FIELDS:
                current = wrapper.get(field) if isinstance(wrapper, Mapping) else None
                label = "absent" if not isinstance(wrapper, Mapping) or field not in wrapper else "other"
                if field == "exit_code" and type(current) is int:
                    label = "zero" if current == 0 else "nonzero"
                elif field != "exit_code" and type(current) is bool:
                    label = "true" if current else "false"
                outcomes[prefix + field] = label
        result["outcomes"] = outcomes
    elif isinstance(value, str):
        parts: list[list[object]] = []
        position = 0
        command = _CODEX_COMMAND_LINE_PREFIX_RE.match(value)
        if command is not None:
            parts.append(["literal", "Command: "])
            _anonymous(parts, value, len("Command: "), command.end())
            position = command.end()
        # Chunk IDs are fixed protocol fields with anonymous hexadecimal
        # spellings; neutral digits remain in that same lexical support.
        chunk = _CHUNK_PREFIX.match(value, position)
        if chunk is not None:
            parts.extend(
                [
                    ["literal", "Chunk ID: "],
                    ["digits", chunk.end() - position - len("Chunk ID: ") - 1],
                    ["literal", "\n"],
                ]
            )
            position = chunk.end()
        match = _EXEC_HEADER.match(value, position) or _FREEFORM_HEADER.match(value, position)
        if match is None:
            return result
        exit_spans = [
            match.span(name) for name in match.re.groupindex if name.startswith("exit_code") and match.start(name) >= 0
        ]
        previous = position
        for token in _RESULT_TOKEN.finditer(value, position, match.end()):
            if token.start() != previous:
                _anonymous(parts, value, previous, token.start())
            if value[token.start()].isdigit():
                zero = any(start <= token.start() and token.end() <= stop for start, stop in exit_spans) and all(
                    value[index] == "0" for index in range(token.start(), token.end())
                )
                parts.append(["zeros" if zero else "digits", token.end() - token.start()])
            else:
                parts.append(["literal", token.group()])
            previous = token.end()
        _anonymous(parts, value, previous, len(value))
        for _ in validated_syntax_parts(parts):
            pass
        result["output_syntax"] = parts
    return result
