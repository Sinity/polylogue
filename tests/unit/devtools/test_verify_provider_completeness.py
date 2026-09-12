from __future__ import annotations

import json

import pytest

from devtools import verify_provider_completeness


def test_provider_completeness_json_is_origin_filterable_and_checkable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert verify_provider_completeness.main(["--origin", "codex-session", "--check", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "provider-package-completeness"
    assert payload["totals"]["total"] == 1
    assert payload["rows"][0]["origin"] == "codex-session"
    assert payload["rows"][0]["status"] == "complete"


def test_provider_completeness_human_output_includes_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_provider_completeness.main(["--origin", "grok-export"]) == 0

    output = capsys.readouterr().out
    assert "provider completeness:" in output
    assert "provider-package:grok-export/export-json@v1: complete" in output
