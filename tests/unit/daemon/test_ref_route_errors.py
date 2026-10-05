"""HTTP ref reads retain distinct invalid and stale continuation refusals."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.query.transaction import QueryContinuationInvalidError, QueryContinuationStaleError
from polylogue.daemon.route_families.read_query import _handle_ref_resolve
from polylogue.operations.ref_resolution import RefResolutionPlan


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (QueryContinuationInvalidError("invalid token"), HTTPStatus.BAD_REQUEST),
        (QueryContinuationStaleError(issued_epoch="old", current_epoch="current"), HTTPStatus.CONFLICT),
    ],
)
def test_ref_http_preserves_typed_continuation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: ValueError, status: HTTPStatus
) -> None:
    from polylogue.archive.query import transaction
    from polylogue.daemon import http
    from polylogue.operations import ref_resolution

    @contextmanager
    def refuse(*args: object, **kwargs: object) -> Iterator[object]:
        raise error
        yield

    def read(archive: Any) -> Any:
        raise AssertionError("ref work must not run after frame refusal")

    monkeypatch.setattr(http, "_web_reader_archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        ref_resolution, "plan_ref_resolution", lambda *a, **k: RefResolutionPlan("session:neutral", read=read)
    )
    monkeypatch.setattr(transaction, "archive_read_context", refuse)
    replies: list[tuple[HTTPStatus, dict[str, object]]] = []

    class Handler:
        def _get_param(self, params: dict[str, list[str]], name: str, default: str | None = None) -> str | None:
            return params[name][0] if name in params else default

        def _send_json(self, response_status: HTTPStatus, payload: dict[str, object]) -> None:
            replies.append((response_status, payload))

    _handle_ref_resolve(Handler(), {"ref": ["session:neutral"]})
    assert len(replies) == 1
    assert replies[0][0] == status
    assert isinstance(error, (QueryContinuationInvalidError, QueryContinuationStaleError))
    assert replies[0][1]["error"] == error.code
