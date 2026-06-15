from __future__ import annotations

import json

import pytest

from devtools import release_readiness


def test_release_readiness_report_validates_committed_gate() -> None:
    report = release_readiness.build_report()

    assert report["ok"] is True
    assert report["gate_doc"] == "docs/plans/release-readiness-gate.md"
    assert ["devtools", "verify", "--quick"] in [command["argv"] for command in report["required_commands"]]
    assert ["devtools", "build-package"] in [command["argv"] for command in report["required_commands"]]
    assert "- Known caveats scoped out:" in report["required_release_body_fields"]
    assert report["errors"] == []


def test_release_readiness_json_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert release_readiness.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["required_commands"][0]["argv"] == ["devtools", "release-readiness"]
