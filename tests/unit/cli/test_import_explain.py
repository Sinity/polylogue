from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from polylogue.cli.click_app import cli
from polylogue.core.enums import Provider
from polylogue.sources import decoder_zip as decoder_zip_module
from polylogue.sources import import_explain as import_explain_module
from polylogue.sources.decoder_zip import ZipEntryValidator
from polylogue.sources.import_explain import explain_import_path


def test_explain_import_path_reports_codex_parser_and_counts(tmp_path: Path) -> None:
    source = Path("tests/data/codex_event_stream/text_only_stream.jsonl")
    target = tmp_path / "session.jsonl"
    shutil.copy2(source, target)

    payload = explain_import_path(target, source_name="codex")

    assert payload.mode == "import-explain"
    assert payload.produced.sessions == 1
    assert payload.produced.messages >= 1
    assert payload.entries[0].detected_provider == "codex"
    assert payload.entries[0].detected_origin == "codex-session"
    assert payload.entries[0].artifact_kind == "session_record_stream"
    assert payload.entries[0].parser_mode == "grouped_records"
    assert payload.entries[0].produced.session_refs


def test_explain_import_path_treats_jsonl_text_json_wrappers_as_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "aggregate.jsonl.txt.json"
    target.write_text(
        "\n".join(
            (
                '{"type":"user","sessionId":"first-session","uuid":"u1","message":{"role":"user","content":"one"}}',
                '{"type":"user","sessionId":"second-session","uuid":"u2","message":{"role":"user","content":"two"}}',
            )
        ),
        encoding="utf-8",
    )

    payload = explain_import_path(target, source_name="claude-code")

    assert payload.produced.sessions == 2
    assert payload.entries[0].detected_origin == "claude-code-session"
    assert payload.entries[0].parser_mode == "grouped_records"
    assert payload.entries[0].produced.session_refs == (
        "session:claude-code:first-session",
        "session:claude-code:second-session",
    )


def test_explain_import_path_reports_malformed_json_as_skip(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("{not json", encoding="utf-8")

    payload = explain_import_path(target)

    assert payload.produced.sessions == 0
    assert payload.skipped
    assert payload.skipped[0].reason.startswith("decode failure:")
    assert payload.entries[0].skipped[0].source_path == str(target.resolve())


def test_import_explain_cli_emits_finite_json(tmp_path: Path) -> None:
    source = Path("tests/data/codex_event_stream/text_only_stream.jsonl")
    target = tmp_path / "session.jsonl"
    shutil.copy2(source, target)

    result = CliRunner().invoke(cli, ["--plain", "import", str(target), "--explain", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "import-explain"
    assert payload["produced"]["sessions"] == 1
    assert payload["entries"][0]["detected_origin"] == "codex-session"
    assert "raw_bytes" not in result.output


def test_import_explain_cli_ndjson_emits_entries(tmp_path: Path) -> None:
    source = Path("tests/data/codex_event_stream/text_only_stream.jsonl")
    target = tmp_path / "session.jsonl"
    shutil.copy2(source, target)

    result = CliRunner().invoke(cli, ["--plain", "import", str(target), "--explain", "--format", "ndjson"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["detected_provider"] == "codex"


def test_import_explain_zip_propagates_member_decode_skip(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/broken.json", "{not json")

    payload = explain_import_path(archive)

    assert payload.produced.sessions == 0
    assert payload.entries[0].skipped
    assert payload.skipped
    skipped_path = payload.skipped[0].source_path
    assert skipped_path is not None
    assert skipped_path.endswith("broken.zip:nested/broken.json")
    assert payload.skipped[0].reason.startswith("decode failure:")


def test_import_explain_zip_rejects_oversized_member_before_read(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    archive = tmp_path / "oversized.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("big.json", b"{}")
    monkeypatch.setattr(import_explain_module, "MAX_UNCOMPRESSED_SIZE", 1)

    payload = explain_import_path(archive)

    assert payload.produced.sessions == 0
    assert payload.skipped
    skipped_path = payload.skipped[0].source_path
    assert skipped_path is not None
    assert skipped_path.endswith("oversized.zip:big.json")
    assert "file size" in payload.skipped[0].reason


def test_import_explain_zip_rejects_aggregate_over_cap_before_read(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A zip whose entries are each individually under the per-entry cap but
    whose running total would exceed MAX_AGGREGATE_UNCOMPRESSED_SIZE must be
    reported by the --explain preview the same way a real ``import`` run
    rejects it (polylogue-it3u): before this fix, the preview only checked
    the old per-entry ratio/size limits and would wrongly claim every entry
    here "will import".
    """
    archive = tmp_path / "aggregate.zip"
    entry_bytes = b'{"a": 1}'
    entry_names = [f"entry_{i}.json" for i in range(3)]
    with zipfile.ZipFile(archive, "w") as zf:
        for name in entry_names:
            zf.writestr(name, entry_bytes)
    monkeypatch.setattr(import_explain_module, "MAX_AGGREGATE_UNCOMPRESSED_SIZE", len(entry_bytes))
    monkeypatch.setattr(decoder_zip_module, "MAX_AGGREGATE_UNCOMPRESSED_SIZE", len(entry_bytes))

    payload = explain_import_path(archive)

    aggregate_skips = [row for row in payload.skipped if "aggregate uncompressed size" in row.reason]
    assert [row.source_path for row in aggregate_skips] == [
        f"{archive}:entry_1.json",
        f"{archive}:entry_2.json",
    ]

    # Cross-check against the real decode-path validator over the exact same
    # entries: the preview's accepted/rejected split must match it exactly.
    with zipfile.ZipFile(archive) as zf:
        validator = ZipEntryValidator("chatgpt", cursor_state=None, zip_path=archive)
        accepted_names = {info.filename for info in validator.filter_entries(zf.infolist())}
    assert accepted_names == {"entry_0.json"}
    rejected_by_preview = {row.source_path.split(":", 1)[1] for row in aggregate_skips if row.source_path is not None}
    assert rejected_by_preview == set(entry_names) - accepted_names


def test_import_explain_zip_excludes_non_session_artifact_from_aggregate(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A non-session-classified entry must not count toward the preview's
    aggregate total (CodeRabbit finding on PR #3317): ``process_zip`` always
    constructs ``ZipEntryValidator`` with ``session_only=True``, which
    excludes non-session-classified entries from the running total entirely
    (they ``continue`` before the aggregate check in ``decoder_zip.py``).
    The preview must apply the identical exclusion, or it can wrongly
    predict an aggregate-cap rejection a real import would never hit.

    Uses a monkeypatched ``classify_artifact_path`` (isolating the exclusion
    LOGIC in ``_zip_entry_skip_reason`` from real ``OriginArtifactRule``
    matching, which is covered separately) matching on the bare intra-archive
    relative path -- both ``_zip_entry_skip_reason`` and
    ``ZipEntryValidator.filter_entries`` classify on that bare path, not a
    ``{zip_path}:{name}`` prefix (polylogue-dc1k: every rule's ``(?:^|/)``-anchored
    pattern only matches after start-of-string or ``/``, never after the ``:``
    a container prefix would insert).
    """
    from polylogue.archive.artifact_taxonomy.models import ArtifactClassification, ArtifactKind

    archive = tmp_path / "workflow.zip"
    session_bytes = b'{"a": 1}'
    non_session_bytes = b'{"run": "snapshot"}' * 1000
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("session.json", session_bytes)
        zf.writestr("run.json", non_session_bytes)
    # Cap sits between the session entry alone and session+non-session
    # combined -- if the non-session entry wrongly counted, this would
    # falsely reject the accepted session entry too.
    monkeypatch.setattr(
        import_explain_module,
        "MAX_AGGREGATE_UNCOMPRESSED_SIZE",
        len(session_bytes) + len(non_session_bytes) // 2,
    )

    def fake_classify(source_path: object, *, provider: object) -> ArtifactClassification | None:
        if str(source_path) == "run.json":
            return ArtifactClassification(
                provider=Provider.CLAUDE_CODE,
                kind=ArtifactKind.WORKFLOW_RUN_SNAPSHOT,
                parse_as_session=False,
                schema_eligible=False,
                default_priority=0,
                reason="non-session workflow snapshot (test fixture)",
            )
        return None

    monkeypatch.setattr(import_explain_module, "classify_artifact_path", fake_classify)

    payload = explain_import_path(archive, source_name="claude-code")

    assert not any("aggregate uncompressed size" in row.reason for row in payload.skipped)
    non_session_skips = [row for row in payload.skipped if row.source_path == f"{archive}:run.json"]
    assert len(non_session_skips) == 1
    assert non_session_skips[0].reason == "non-session workflow snapshot (test fixture)"
    # session.json is separately skipped as "metadata-oriented document" (its
    # trivial fixture bytes aren't a real session shape) -- but crucially
    # NOT for an aggregate-size reason, which is the only thing this test
    # proves: run.json's bytes never reached the running aggregate total.
    session_skips = [row for row in payload.skipped if row.source_path == f"{archive}:session.json"]
    assert len(session_skips) == 1
    assert "aggregate uncompressed size" not in session_skips[0].reason


def test_import_explain_zip_allows_archive_comfortably_under_aggregate_cap(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Multiple small entries whose sum stays well under the aggregate cap
    are all reported importable -- no regression for legitimate multi-file
    exports."""
    archive = tmp_path / "normal.zip"
    entry_bytes = b'{"a": 1}'
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(3):
            zf.writestr(f"entry_{i}.json", entry_bytes)
    monkeypatch.setattr(import_explain_module, "MAX_AGGREGATE_UNCOMPRESSED_SIZE", len(entry_bytes) * 10)

    payload = explain_import_path(archive)

    assert not any("aggregate uncompressed size" in row.reason for row in payload.skipped)
