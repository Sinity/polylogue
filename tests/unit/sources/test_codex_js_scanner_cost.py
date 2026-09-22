"""The Code Mode JS scan is not per-character (polylogue-s8x8s residual (c)).

``_scan_code_mode_child_calls`` walks a Codex ``code`` program looking for
``tools.*``/``functions.*`` calls. It used to advance one character at a time,
asking ``_skip_js_string_or_comment`` at every position, so a program's inert
regions -- whitespace, arithmetic, punctuation, and the interior of every
string literal -- each cost a Python loop iteration that could only decline.

These are *cost* assertions, measured in interpreted line events inside the
scanner family. A behavioural test passes whether the scan is O(1) or
O(program) per inert character; only a step count separates them. The
equivalence tests beside them pin what the cost work must not change.

Anti-vacuity: restoring either per-character loop makes a cost test red, and
the equivalence tests go red if a jump ever skips a token.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import polylogue.sources.parsers.codex as codex

#: The scan family whose interpreted steps are being counted.
_SCANNERS = frozenset(
    {
        "_scan_code_mode_child_calls",
        "_balanced_js_call_argument",
        "_first_js_argument",
        "_skip_js_string_or_comment",
    }
)


def _scanner_line_events(run: Callable[[], object]) -> int:
    """Interpreted line events executed inside the scan family while ``run`` ran."""
    count = 0

    def tracer(frame: object, event: str, arg: object) -> object:
        nonlocal count
        if frame.f_code.co_name not in _SCANNERS:  # type: ignore[attr-defined]
            return None
        if event == "line":
            count += 1
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        run()
    finally:
        sys.settrace(previous)
    return count


#: Arithmetic, spaces and semicolons only. Nothing here can begin a string, a
#: comment, a regex or an identifier, so a scanner that understands its own
#: input has no reason to stop anywhere inside it.
_INERT = "1 + 2 - 3 ; "

_CALL = 'const r = await tools.web.search({"query": "hi"});'


def _program(inert_repeats: int) -> str:
    return _INERT * inert_repeats + _CALL


def test_inert_program_regions_cost_no_interpreted_steps() -> None:
    small = _program(4)
    large = _program(4 + 2000)

    small_events = _scanner_line_events(lambda: codex._scan_code_mode_child_calls(small))
    large_events = _scanner_line_events(lambda: codex._scan_code_mode_child_calls(large))

    added_characters = len(large) - len(small)
    added_events = large_events - small_events
    # A per-character scan pays roughly a dozen line events per added
    # character. One event per twenty added characters leaves that mutation
    # red by more than two orders of magnitude while tolerating the handful
    # of regex-search iterations the jump itself costs.
    assert added_events < added_characters / 20, (
        f"inert scan cost grew by {added_events} line events over {added_characters} added inert characters"
    )


def test_long_string_literal_body_costs_no_interpreted_steps() -> None:
    """Skipping a string literal is a ``find``, not a loop over its body."""
    short = 'x = "' + "a" * 8 + '";'
    long = 'x = "' + "a" * 20_000 + '";'

    short_events = _scanner_line_events(lambda: codex._skip_js_string_or_comment(short, 4))
    long_events = _scanner_line_events(lambda: codex._skip_js_string_or_comment(long, 4))

    added_characters = len(long) - len(short)
    added_events = long_events - short_events
    assert added_events < added_characters / 100, (
        f"string-literal skip cost grew by {added_events} line events over {added_characters} added literal characters"
    )


def test_escaped_literal_body_still_costs_per_escape_not_per_character() -> None:
    """An escape-dense literal is the worst case, and it is still sublinear."""
    body = "a" * 200
    source = 'x = "' + (body + "\\n") * 50 + '";'

    events = _scanner_line_events(lambda: codex._skip_js_string_or_comment(source, 4))

    assert events < len(source) / 20, f"escaped-literal skip cost {events} line events for {len(source)} characters"


def test_scan_finds_the_same_call_inside_a_large_inert_program() -> None:
    """The jump may not skip a token: same calls, tiny or huge inert regions."""
    tiny = [
        (call.tool_path, call.tool_name, call.registry_type, call.argument, call.parse_state)
        for call in codex._scan_code_mode_child_calls(_program(1))
    ]
    huge = [
        (call.tool_path, call.tool_name, call.registry_type, call.argument, call.parse_state)
        for call in codex._scan_code_mode_child_calls(_program(2000))
    ]

    assert tiny == huge
    assert tiny == [(("tools", "web", "search"), "web", "web", {"query": "hi"}, "parsed")]


def test_calls_separated_only_by_inert_text_are_all_found() -> None:
    source = _INERT.join(
        [
            'await tools.web.search({"query": "a"});',
            'await functions.image.generate({"prompt": "b"});',
            'await tools.mcp.repo.search({"q": "c"});',
        ]
    )

    found = [call.registry_type for call in codex._scan_code_mode_child_calls(source)]

    assert found == ["web", "image", "mcp"]


def test_string_literal_skip_boundaries() -> None:
    """Escapes, unterminated literals and a trailing backslash keep their meaning."""
    cases = [
        ('"ab"', 4),
        ('"a\\"b"', 6),
        ("'a\\\\'", 5),
        ("`a`", 3),
        ('"unterminated', 13),
        ('"trailing\\', 10),
        ('""', 2),
    ]
    for source, expected_end in cases:
        assert codex._skip_js_string_or_comment(source, 0) == expected_end, source


def test_a_call_inside_a_string_literal_is_not_scanned_as_a_call() -> None:
    """The literal skip is what suppresses it, so the jump must not bypass it."""
    source = 'const s = "await tools.web.search({})"; await tools.image.generate({});'

    found = [call.registry_type for call in codex._scan_code_mode_child_calls(source)]

    assert found == ["image"]
