from __future__ import annotations

import pytest

import devtools.reindex_canary as reindex_canary
from polylogue.scenarios import ExecutionSpec


def test_devtools_reindex_canary_delegates_to_product_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_invoke(execution: ExecutionSpec) -> object:
        captured["execution"] = execution

        class Result:
            exit_code = 0
            stdout = "canary output\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(reindex_canary, "invoke_polylogue_cli", fake_invoke)

    assert (
        reindex_canary.main(
            [
                "--archive-root",
                "/tmp/isolated-archive",
                "--schema-inference-receipt",
                "/tmp/schema-inference-gate-receipt.json",
                "--report",
                "/tmp/canary.json",
                "--no-promote",
                "--json",
            ]
        )
        == 0
    )
    execution = captured["execution"]
    assert isinstance(execution, ExecutionSpec)
    assert execution.command == (
        "polylogue",
        "--plain",
        "ops",
        "maintenance",
        "reindex-canary",
        "--archive-root",
        "/tmp/isolated-archive",
        "--schema-inference-receipt",
        "/tmp/schema-inference-gate-receipt.json",
        "--report",
        "/tmp/canary.json",
        "--no-promote",
        "--output-format",
        "json",
    )
    assert capsys.readouterr().out == "canary output\n"
