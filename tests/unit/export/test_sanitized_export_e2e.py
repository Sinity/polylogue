"""Adversarial no-leak end-to-end test for sanitized export (#2381).

Builds session profiles that carry a private absolute path, a ``$HOME`` path,
and a secret-looking value in several fields, runs the real export path
(projection → redaction → write → fail-closed gate → publish), then reads the
written ``dataset.jsonl`` + ``redaction-manifest.json`` back as text and asserts
no planted leak survives anywhere. This extends the spirit of the demo
``absolute_path_leaks == ()`` assertion.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.api.archive import _archive_sanitized_export_row
from polylogue.archive.session.models import SessionProfile
from polylogue.export.sanitize import SanitizedExportRequest, produce_sanitized_export

PLANTED_ABS_PATH = "/home/someone/secret/project/notes.md"
PLANTED_HOME = "/home/someone"
PLANTED_REPO_PATH = "/home/someone/code/private-repo"
PLANTED_SECRET = "sk-ABCD1234EFGH5678IJKLmnop"


def _profile(session_id: str, *, title: str | None) -> SessionProfile:
    return SessionProfile(
        session_id=session_id,
        origin="claude-code-session",
        title=title,
        created_at=None,
        updated_at=None,
        message_count=5,
        substantive_count=5,
        tool_use_count=2,
        thinking_count=0,
        attachment_count=0,
        word_count=120,
        total_cost_usd=1.25,
        total_duration_ms=0,
        tool_categories={},
        # Highest-leak fields — must NOT reach the dataset at all.
        repo_paths=(PLANTED_REPO_PATH,),
        cwd_paths=(PLANTED_REPO_PATH,),
        branch_names=("main",),
        file_paths_touched=(PLANTED_ABS_PATH,),
        languages_detected=("python",),
        # repo_names is exported but still redacted field-wise.
        repo_names=("polylogue", PLANTED_REPO_PATH),
        work_events=(),
        phases=(),
        first_message_at=None,
        last_message_at=None,
        wall_duration_ms=4200,
        cost_is_estimated=True,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_cache_read_tokens=0,
        total_cache_write_tokens=0,
    )


def _rows() -> list[dict[str, object]]:
    profiles = [
        _profile("s-1", title=f"debugging {PLANTED_ABS_PATH}"),
        _profile("s-2", title=f"leaked token {PLANTED_SECRET}"),
    ]
    return [_archive_sanitized_export_row(profile) for profile in profiles]


def test_projection_excludes_high_leak_path_fields() -> None:
    row = _archive_sanitized_export_row(_profile("s-1", title="t"))
    # The path-bearing profile fields are not even projected into the row.
    assert "repo_paths" not in row
    assert "cwd_paths" not in row
    assert "file_paths_touched" not in row
    assert "branch_names" not in row


def test_full_export_no_planted_leak_survives(tmp_path: Path) -> None:
    out_dir = tmp_path / "shareable"
    request = SanitizedExportRequest(output_path=out_dir)
    result = produce_sanitized_export(
        rows=_rows(),
        scope={"query": f"secret {PLANTED_SECRET}", "matched_session_count": 2},
        request=request,
        home=PLANTED_HOME,
    )

    assert result.verify_ok is True
    assert result.row_count == 2

    dataset_text = result.dataset_path.read_text(encoding="utf-8")
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    readme_text = result.readme_path.read_text(encoding="utf-8")

    for planted in (PLANTED_ABS_PATH, PLANTED_REPO_PATH, PLANTED_HOME, PLANTED_SECRET):
        for name, text in (
            ("dataset.jsonl", dataset_text),
            ("redaction-manifest.json", manifest_text),
            ("README.md", readme_text),
        ):
            assert planted not in text, f"{planted!r} leaked into {name}"


def test_full_export_with_postmortem_payload_is_gated(tmp_path: Path) -> None:
    out_dir = tmp_path / "shareable-pm"
    request = SanitizedExportRequest(output_path=out_dir, with_postmortem=True)
    # A postmortem payload that itself carries a leaked repo path + secret.
    postmortem: dict[str, object] = {
        "repos_touched": [{"repo": PLANTED_REPO_PATH, "session_count": 1}],
        "note": f"key {PLANTED_SECRET}",
    }
    result = produce_sanitized_export(
        rows=_rows(),
        scope={},
        request=request,
        postmortem=postmortem,
        home=PLANTED_HOME,
    )
    assert result.verify_ok is True
    assert result.postmortem_path is not None
    pm_text = result.postmortem_path.read_text(encoding="utf-8")
    assert PLANTED_REPO_PATH not in pm_text
    assert PLANTED_SECRET not in pm_text
