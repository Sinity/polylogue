from __future__ import annotations

import json

import pytest

from devtools import affected_obligations


def test_main_json_routes_explicit_paths_without_git_diff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert affected_obligations.main(["--json", "--path", "docs/verification-catalog.md"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["changed_paths"] == ["docs/verification-catalog.md"]
    assert payload["change_subjects"][0]["kind"] == "generated_surface"
    assert isinstance(payload["affected_obligations"], list)
    assert [check["command"] for check in payload["pr_gates"]] == [
        ["devtools", "verify", "--quick"],
        ["devtools", "verify"],
    ]


def test_main_human_discovers_paths_from_refs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        affected_obligations,
        "changed_paths_between_refs",
        lambda base_ref, head_ref: ("polylogue/sources/parsers/codex.py",),
    )
    monkeypatch.setattr(affected_obligations, "obligation_ids_for_ref", lambda ref: ())

    assert affected_obligations.main(["--base-ref", "origin/master", "--head-ref", "HEAD"]) == 0

    rendered = capsys.readouterr().out
    assert "Refs: origin/master..HEAD" in rendered
    assert "parser:polylogue/sources/parsers/codex.py" in rendered
    assert "provider.capability.codex" in rendered
