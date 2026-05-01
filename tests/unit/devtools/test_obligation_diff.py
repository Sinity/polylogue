from __future__ import annotations

import json

import pytest

from devtools import obligation_diff


def test_obligation_diff_accepts_ref_flags_and_markdown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        obligation_diff,
        "changed_paths_between_refs",
        lambda base_ref, head_ref: ("docs/plans/layering.yaml",),
    )
    monkeypatch.setattr(obligation_diff, "obligation_ids_for_ref", lambda ref: ())

    assert obligation_diff.main(["--base-ref", "origin/master", "--head-ref", "HEAD", "--markdown"]) == 0

    rendered = capsys.readouterr().out
    assert "## Proof Obligations" in rendered
    assert "`origin/master..HEAD`" in rendered


def test_obligation_diff_json_reports_changed_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        obligation_diff,
        "changed_paths_between_refs",
        lambda base_ref, head_ref: ("docs/plans/layering.yaml",),
    )
    monkeypatch.setattr(obligation_diff, "obligation_ids_for_ref", lambda ref: ())

    assert obligation_diff.main(["--base-ref", "origin/master", "--head-ref", "HEAD", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["changed_paths"] == ["docs/plans/layering.yaml"]
    assert payload["change_subjects"][0]["kind"] == "architecture"
