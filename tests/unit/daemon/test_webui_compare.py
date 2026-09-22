"""``GET /w/compare`` renders a comparison, not a stringified envelope.

The compare envelope carries both whole session payloads under ``left`` and
``right``, the full pairing over them under ``pairs``, and ``metadata_diff``.
The generic typed-data page projects any non-list value through ``str``, which
on those keys is an escaped Python ``repr`` of every message -- unreadable, and
repeated once per key that holds it. Measured on the fixture below before the
change: 20 205 bytes of HTML and 480 copies of the message-text fragment for
twelve short messages, growing without bound in the transcripts behind it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.compare import build_compare_envelope
from polylogue.daemon.webui import _scalar_projection, render_compare_page
from polylogue.daemon.workspace_routes import MessageWindow
from tests.infra.daemon_http_harness import make_daemon_handler

_TEXT_FRAGMENT = "lorem ipsum dolor sit amet "


def _session(session_id: str, count: int, *, repeats: int = 20) -> dict[str, object]:
    return {
        "session_id": session_id,
        "title": f"Session {session_id}",
        "origin": "codex-session",
        "message_count": count,
        "total": count,
        "messages": [
            {
                "id": f"{session_id}:m{index}",
                "role": "user" if index % 2 == 0 else "assistant",
                "text": _TEXT_FRAGMENT * repeats,
            }
            for index in range(count)
        ],
    }


def _bundle() -> Any:
    return SimpleNamespace(entrypoint=lambda: SimpleNamespace(stylesheets=(), script="reader.js"))


def _render(count: int, *, repeats: int = 20) -> str:
    envelope = build_compare_envelope(
        _session("a", count, repeats=repeats),
        _session("b", count, repeats=repeats),
        "a",
        "b",
        "prompt",
        window=MessageWindow(),
    )
    return render_compare_page(_bundle(), payload=envelope, empty="nothing to compare")


def test_compare_page_emits_no_python_repr() -> None:
    """Anti-vacuity: route compare back through ``render_typed_data_page``
    and this finds the escaped ``{&#x27;session_id&#x27;: ...`` dump."""
    assert "{&#x27;" not in _render(6)


def test_compare_page_never_carries_a_whole_message() -> None:
    """Every message reaches the page through the bounded preview only.

    Anti-vacuity: the stringified envelope embeds each message's complete
    ``text`` verbatim, so this finds the 540-character body below.
    """
    body = _render(6)
    assert _TEXT_FRAGMENT in body, "the preview itself must still be rendered"
    assert _TEXT_FRAGMENT * 20 not in body


def test_compare_page_size_is_flat_in_message_length() -> None:
    """Page size follows the window, not the length of the transcripts.

    Anti-vacuity: the stringified envelope holds four copies of each
    message's full text, so a tenfold longer message grows the page roughly
    tenfold and this goes red. Message *count* is held fixed here on purpose
    -- both implementations grow with it, so varying it separates nothing.
    """
    short = len(_render(6, repeats=20))
    long = len(_render(6, repeats=200))
    assert long < short * 1.5


def test_compare_page_keeps_the_window_and_alignment() -> None:
    """The pairing covers exactly the served window, so the page says so."""
    body = _render(6)
    assert "limit" in body.lower()
    assert "sequential" in body or "anchor" in body
    assert 'class="compare-pairs"' in body
    assert 'class="compare-metadata"' in body


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"a": 1, "b": 2}, "2 fields"),
        ({"a": 1}, "1 field"),
        ([1, 2, 3], "3 entries"),
        ([1], "1 entry"),
        ("plain", "plain"),
        (7, "7"),
    ],
)
def test_scalar_projection_counts_structures(value: object, expected: str) -> None:
    """Scalars stay verbatim; a structure gets a bound, never a ``repr``."""
    assert _scalar_projection(value) == expected


def test_compare_route_uses_the_compare_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production dispatch, not just the renderer, must select it.

    Anti-vacuity: restore the single ``_serve_webui_secondary`` tail and the
    compare mode lands in ``generic``.
    """
    from polylogue.daemon.http import DaemonAPIHandler

    handler = make_daemon_handler("GET", "/w/compare?left=a&right=b")
    seen: dict[str, object] = {}
    monkeypatch.setattr("polylogue.daemon.http._web_reader_archive_root", lambda: None)
    monkeypatch.setattr(
        DaemonAPIHandler,
        "_serve_webui_compare",
        lambda _self, payload: seen.__setitem__("compare", payload),
    )
    monkeypatch.setattr(
        DaemonAPIHandler,
        "_serve_webui_secondary",
        lambda _self, **kwargs: seen.__setitem__("generic", kwargs),
    )

    cast(Any, handler)._serve_webui_workspace("compare", {"left": ["a"], "right": ["b"]})
    assert "compare" in seen
    assert "generic" not in seen

    seen.clear()
    cast(Any, handler)._serve_webui_workspace("stack", {"ids": ["a,b"]})
    assert "generic" in seen
    assert "compare" not in seen
