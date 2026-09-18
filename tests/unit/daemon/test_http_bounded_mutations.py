"""Bounded mutating waits and honest page totals on the daemon HTTP surface.

Covers polylogue-8r4zq (a mutating route waited on its writer future with no
timeout) and polylogue-q54dt (paging walks published the page size as the
archive total).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from io import BytesIO
from typing import Any, cast

import pytest


class _Headers:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = values or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key, default)


def _mutating_handler(kernel: object, *, deadline_ms: str = "300") -> tuple[Any, list[tuple[HTTPStatus, object]]]:
    """A handler wired to a real compute kernel, inside the write gate."""
    from polylogue.daemon.http import DaemonAPIHandler

    sent: list[tuple[HTTPStatus, object]] = []

    class _Server:
        execution_kernel = kernel

    class _RecordingHandler(DaemonAPIHandler):
        def __init__(self) -> None:
            self.server = cast("Any", _Server())
            self.path = "/api/test-mutation"
            self.command = "POST"
            self.requestline = "POST /api/test-mutation HTTP/1.1"
            self.client_address = ("127.0.0.1", 12345)
            self.rfile = BytesIO(b"")
            self.wfile = BytesIO()
            self.headers = cast("Any", _Headers({"X-Polylogue-Deadline-Ms": deadline_ms}))
            # The write gate is what marks a route mutating.
            self._write_gate_depth = 1

        def _send_json(
            self, status: HTTPStatus, payload: object, *, extra_headers: Mapping[str, str] | None = None
        ) -> None:
            sent.append((status, payload))

    return _RecordingHandler(), sent


@pytest.mark.uses_real_clock("asserts a real-thread wait actually returns within its bound")
def test_mutating_route_wait_is_bounded_by_the_request_deadline() -> None:
    """polylogue-8r4zq: a blocked mutation answers, typed, within its bound.

    Input: the route body blocks the way a control mutation queued behind a
    convergence-pass lease hold blocks -- it never returns on its own. At the
    reported head ``submitted.future.result()`` carried no timeout, so the
    client had no observable bound at all.

    Anti-vacuity: restoring the untimed ``submitted.future.result()`` makes
    this test hang past its own wall-clock assertion rather than produce a
    503 -- there is no bound to observe. The elapsed assertion also fails if
    the wait silently falls back to the 60s default instead of honouring the
    declared deadline.
    """
    from polylogue.daemon.execution import BoundedComputeAdapter
    from polylogue.daemon.http import daemon_safe_handler

    release = threading.Event()
    kernel = BoundedComputeAdapter(max_workers=2, queue_units=2)
    handler, sent = _mutating_handler(kernel, deadline_ms="300")

    async def _blocked(_poly: object) -> object:  # pragma: no cover - never completes in time
        release.wait(30)
        return {"ok": True}

    def _route(self: Any) -> None:
        self._send_json(HTTPStatus.OK, self._sync_run(_blocked))

    guarded = daemon_safe_handler(_route)

    started = time.monotonic()
    try:
        guarded(handler)
    finally:
        release.set()
        kernel.executor.shutdown(wait=False)

    elapsed = time.monotonic() - started
    assert elapsed < 10.0, f"the mutating wait was not bounded: {elapsed:.1f}s"
    assert len(sent) == 1
    status, payload = sent[0]
    assert status is HTTPStatus.SERVICE_UNAVAILABLE, payload
    assert isinstance(payload, dict)
    # Indeterminate, not cancelled: the submitted write may still land.
    assert payload["error"] == "mutation_indeterminate"


def test_declared_deadline_outside_the_band_falls_back_to_the_route_default() -> None:
    """A header can shorten the bound; it can never remove it.

    Anti-vacuity: accepting the header value unchecked lets ``0`` or a
    negative value reach ``future.result(timeout=...)``, and this assertion on
    the default budget turns red.
    """
    from polylogue.daemon.http import _MUTATION_WAIT_TIMEOUT_S

    for raw in ("0", "-5", "", "not-a-number", "9999999"):
        handler, _ = _mutating_handler(object(), deadline_ms=raw)
        assert handler._mutation_wait_budget_s() == _MUTATION_WAIT_TIMEOUT_S, raw

    handler, _ = _mutating_handler(object(), deadline_ms="250")
    assert handler._mutation_wait_budget_s() == pytest.approx(0.25)


def test_truncated_paste_browser_page_does_not_publish_the_page_size_as_total() -> None:
    """polylogue-q54dt: a full page reports an unknown total, not its own size.

    At the reported head the walk's running counter stopped incrementing the
    moment the page filled, so ``GET /api/paste-browser?limit=200`` over an
    archive with far more matches answered ``items 200, total 200``.

    Anti-vacuity: restoring ``total=total_messages_seen`` unconditionally
    turns this red -- the payload would carry ``total == len(items)`` and
    claim it exact.
    """
    from polylogue.daemon.webui_data import PasteBrowserEntry, build_paste_browser_payload

    entries = [
        PasteBrowserEntry(
            session_id=f"origin:s{index}",
            session_title="t",
            origin="codex-session",
            message_id=f"m{index}",
            message_anchor=f"#m{index}",
            role="user",
            timestamp=None,
            word_count=0,
            snippet="",
            paste_spans=[],
            has_diff=False,
        )
        for index in range(2)
    ]

    truncated = build_paste_browser_payload(entries, total=None, total_is_exact=False, matched_so_far=2)
    assert truncated["total"] is None
    assert truncated["total_is_exact"] is False
    assert truncated["total_lower_bound"] == 2

    complete = build_paste_browser_payload(entries, total=2, total_is_exact=True, matched_so_far=2)
    assert complete["total"] == 2
    assert complete["total_is_exact"] is True


async def test_paste_browser_walk_reports_unknown_total_when_the_page_fills() -> None:
    """The production walk, not just its payload builder, refuses the fake total.

    Three paste-bearing messages exist and the page holds one, so the walk
    breaks early and cannot know the denominator.

    Anti-vacuity: deleting the ``page_truncated`` flag from
    ``_do_paste_browser`` restores ``total == 1`` for a three-match archive
    and turns this red.
    """
    from types import SimpleNamespace

    from polylogue.daemon.http import DaemonAPIHandler

    def _message(index: int) -> object:
        return SimpleNamespace(
            id=f"m{index}",
            has_paste=True,
            text="@@ -1,2 +1,2 @@\n-a\n+b\n",
            role="user",
            timestamp=None,
            word_count=3,
        )

    class _Filter:
        async def list_summaries(self) -> list[object]:
            return [SimpleNamespace(id="codex-session:s1", display_title="s1", origin="codex-session")]

    class _Poly:
        def filter(self) -> _Filter:
            return _Filter()

        async def get_session(self, _sid: str) -> object:
            return SimpleNamespace(messages=[_message(0), _message(1), _message(2)])

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    payload = await handler._do_paste_browser(cast("Any", _Poly()), limit=1, offset=0)

    assert isinstance(payload, dict)
    assert len(payload["items"]) == 1
    assert payload["total"] is None, payload
    assert payload["total_is_exact"] is False
