from __future__ import annotations

import json
from typing import Any, cast

import pytest

from devtools.render_api_operation_parity import build_parity_payload, render_parity_output


def test_api_operation_parity_renderer_emits_stable_machine_readable_authority() -> None:
    payload = cast(dict[str, Any], build_parity_payload())
    authority = cast(dict[str, Any], payload["authority"])
    operations = cast(list[dict[str, Any]], payload["operations"])
    exclusions = cast(list[dict[str, Any]], payload["exclusions"])
    assert payload["schema_version"] == 1
    assert authority["drift_owner"] == "polylogue-s1kr"
    assert any(row["operation_id"] == "api.lifecycle.construct" for row in operations)
    assert any(row["binding"] == "select_pending_embedding_session_window" for row in exclusions)
    assert json.loads(render_parity_output()) == payload


def test_renderer_fails_closed_for_unclassified_live_facade_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.api import Polylogue
    from polylogue.api.operation_parity import validate_live_facade

    async def unclassified(self: Polylogue) -> None:
        return None

    monkeypatch.setattr(Polylogue, "unclassified", unclassified, raising=False)
    with pytest.raises(ValueError, match="unclassified live callables"):
        validate_live_facade()
