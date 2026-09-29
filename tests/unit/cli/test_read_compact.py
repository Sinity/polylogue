"""``read --view compact`` reaches the compaction operation, not context-image."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.surfaces.compaction import compact_sessions


def test_read_view_compact_dispatches_read_compact_with_the_token_budget() -> None:
    """The CLI view lowers to ``read.compact`` and renders the returned pack.

    Anti-vacuity: before the view existed, ``--view compact`` was a usage
    error; and if ``--max-tokens`` still diverted every non-dialogue view into
    the context-image path, ``read.context-image`` would be dispatched instead
    and the budget would not reach the compaction projection.
    """

    message = {"id": "m1", "text": "hello", "material_origin": "human_authored"}
    pack = compact_sessions([{"id": "s1", "messages": [message]}])
    dispatched: list[object] = []

    def _dispatch(config: object, request: object, **kwargs: object) -> SimpleNamespace:
        dispatched.append(request)
        return SimpleNamespace(value={"view": "compact", "payload": pack.model_dump(mode="json")})

    with patch("polylogue.cli.operation_kernel.dispatch", side_effect=_dispatch):
        result = CliRunner().invoke(
            cli,
            ["--plain", "repo:polylogue", "read", "--view", "compact", "--max-tokens", "500", "--format", "json"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert [request.operation for request in dispatched] == ["read.compact"]  # type: ignore[attr-defined]
    assert dispatched[0].payload["projection"] == {"max_tokens": 500}  # type: ignore[attr-defined]
    assert json.loads(result.output)["pack_ref"] == pack.pack_ref
