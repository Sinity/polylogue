from __future__ import annotations

import json

from pytest import CaptureFixture

from devtools import provider_completeness


def test_provider_completeness_command_emits_json_for_filtered_origin(capsys: CaptureFixture[str]) -> None:
    exit_code = provider_completeness.main(["--origin", "codex-session", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "provider-package-completeness"
    assert [row["origin"] for row in payload["rows"]] == ["codex-session"]


def test_provider_completeness_check_ignores_proposed_incomplete_rows() -> None:
    exit_code = provider_completeness.main(["--origin", "unknown-export", "--check"])

    assert exit_code == 0
