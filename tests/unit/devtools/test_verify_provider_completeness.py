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
    assert verify_provider_completeness.main(["--origin", "codex-session"]) == 0

    output = capsys.readouterr().out
    assert "provider completeness:" in output
    assert "provider-package:codex-session/session-jsonl@v1: complete" in output


def test_a_withdrawn_schema_package_reports_partial_and_names_the_missing_item(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """polylogue-1khzy: grok-export has no schema package, and must say so.

    ``polylogue/schemas/providers/grok`` was deleted by d7d3ddae5 (#5224)
    under polylogue-n61h5: its committed element schemas carried
    ``$id: polylogue://schemas/claude-ai/v2/session_document`` and Claude.ai's
    document shape (``uuid``, ``name``, ``summary``), so the package described
    the wrong subject. A withdrawn package is the correct state until one is
    inferred from real Grok sources; this test pins that the report says
    ``partial`` and names ``schema_package``, rather than reporting an absent
    package as complete -- which is the failure mode n61h5 was filed about.

    Anti-vacuity: make the row read ``complete`` -- by treating an absent or
    undeclared schema owner path as satisfied in
    ``polylogue/sources/provider_completeness.py``, or by re-pointing
    ``schema_paths`` at a package that is not Grok's -- and this goes red.
    """
    assert verify_provider_completeness.main(["--origin", "grok-export"]) == 0

    output = capsys.readouterr().out
    assert "provider-package:grok-export/export-json@v1: partial" in output
    assert "schema_package is missing" in output
    assert "provider-package:grok-export/export-json@v1: complete" not in output
