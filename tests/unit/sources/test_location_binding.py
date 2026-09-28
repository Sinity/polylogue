"""Acquisition binds a location to its origin and refuses foreign content.

A source location admits only its own origin's material. Shape detection there
only validates: content carrying another origin's shape is a typed refusal,
never reparsed as the origin its shape suggests. Only the operator's import
inbox (and browser-capture envelopes, which declare their provider) classify.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocumentList
from polylogue.sources.dispatch import (
    ForeignOriginContentError,
    detect_provider,
    detect_provider_from_raw_bytes_evidence,
)
from polylogue.sources.live.batch_support import (
    _detect_provider_from_path_sample,
    _jsonl_provider_and_session_artifact,
)
from polylogue.sources.source_parsing import parse_one_source_path

_CODEX_ROLLOUT: JSONDocumentList = [
    {
        "type": "session_meta",
        "payload": {"id": "codex-session-1", "timestamp": "2026-01-01T10:00:00Z"},
    },
    {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Run checks."}],
        },
    },
]

_CLAUDE_CODE_TRANSCRIPT: JSONDocumentList = [
    {
        "type": "user",
        "uuid": "u1",
        "sessionId": "bad69218-73bd-490a-869a-2b3a30bf421b",
        "timestamp": "2025-06-13T17:40:00.000Z",
        "cwd": "/home/user/project",
        "message": {"role": "user", "content": "Search for ad-hoc solutions."},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "parentUuid": "u1",
        "sessionId": "bad69218-73bd-490a-869a-2b3a30bf421b",
        "timestamp": "2025-06-13T17:40:05.000Z",
        "cwd": "/home/user/project",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Looking now."}]},
    },
]


def _jsonl(records: JSONDocumentList) -> bytes:
    return ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")


def test_bound_detection_refuses_another_origins_shape() -> None:
    """A bound location validates; it never re-identifies.

    Anti-vacuity: without the ``expected`` check, the Codex rollout at a
    Claude Code location is classified as Codex and returned.
    """
    with pytest.raises(ForeignOriginContentError) as refused:
        detect_provider(_CODEX_ROLLOUT, expected=Provider.CLAUDE_CODE)
    assert refused.value.expected is Provider.CLAUDE_CODE
    assert refused.value.found is Provider.CODEX
    assert refused.value.code == "foreign_origin_content"

    assert detect_provider(_CODEX_ROLLOUT, expected=Provider.CODEX) is Provider.CODEX
    assert detect_provider(_CLAUDE_CODE_TRANSCRIPT, expected=Provider.CLAUDE_CODE) is Provider.CLAUDE_CODE


def test_unbound_locations_still_classify() -> None:
    """The import inbox has no single owning origin, so it classifies.

    Anti-vacuity: binding UNKNOWN as if it were an origin refuses every
    import.
    """
    assert detect_provider(_CODEX_ROLLOUT) is Provider.CODEX
    assert detect_provider(_CODEX_ROLLOUT, expected=Provider.UNKNOWN) is Provider.CODEX


def test_raw_bytes_detection_binds_to_the_callers_location() -> None:
    """The per-file acquisition chokepoint refuses foreign bytes.

    Anti-vacuity: returning the shape-detected provider over
    ``fallback_provider`` reparses a Codex rollout as Codex from Claude
    Code's directory.
    """
    with pytest.raises(ForeignOriginContentError):
        detect_provider_from_raw_bytes_evidence(_jsonl(_CODEX_ROLLOUT), "rollout.jsonl", Provider.CLAUDE_CODE)
    provider, _evidence = detect_provider_from_raw_bytes_evidence(
        _jsonl(_CODEX_ROLLOUT), "rollout.jsonl", Provider.UNKNOWN
    )
    assert provider is Provider.CODEX


def test_live_sniff_refuses_a_codex_rollout_in_claude_codes_directory(tmp_path: Path) -> None:
    """The daemon's JSONL sniff raises the typed refusal the batch records.

    Anti-vacuity: without binding, the sniff returns ``(CODEX, True)`` and
    the daemon parses the file as a Codex session.
    """
    project = tmp_path / "projects" / "proj"
    project.mkdir(parents=True)
    rollout = project / "c0ffee00-1111-2222-3333-444455556666.jsonl"
    rollout.write_bytes(_jsonl(_CODEX_ROLLOUT))
    transcript = project / "bad69218-73bd-490a-869a-2b3a30bf421b.jsonl"
    transcript.write_bytes(_jsonl(_CLAUDE_CODE_TRANSCRIPT))

    with pytest.raises(ForeignOriginContentError):
        _jsonl_provider_and_session_artifact(rollout, Provider.CLAUDE_CODE)
    assert _jsonl_provider_and_session_artifact(transcript, Provider.CLAUDE_CODE) == (Provider.CLAUDE_CODE, True)


def test_gemini_cli_prompt_log_with_claude_code_shape_is_refused(tmp_path: Path) -> None:
    """A Gemini CLI prompt log that happens to look like Claude Code records.

    Real Gemini CLI ``logs.json`` rows carry ``sessionId``/``type``/
    ``message`` keys that the Claude Code detector recognizes; before binding,
    they were admitted as Claude Code sessions from Gemini CLI's directory.

    Anti-vacuity: without binding the sniff returns CLAUDE_CODE.
    """
    chats = tmp_path / "tmp" / "abc" / "chats"
    chats.mkdir(parents=True)
    log = chats / "logs.jsonl"
    log.write_bytes(_jsonl(_CLAUDE_CODE_TRANSCRIPT))

    with pytest.raises(ForeignOriginContentError) as refused:
        _detect_provider_from_path_sample(log, Provider.GEMINI_CLI)
    assert refused.value.found is Provider.CLAUDE_CODE


def test_one_shot_source_route_refuses_foreign_content(tmp_path: Path) -> None:
    """The one-shot parse route shares the refusal, not a silent reparse.

    Anti-vacuity: without binding it yields one Codex session.
    """
    project = tmp_path / ".claude" / "projects" / "proj"
    project.mkdir(parents=True)
    rollout = project / "c0ffee00-1111-2222-3333-444455556666.jsonl"
    rollout.write_bytes(_jsonl(_CODEX_ROLLOUT))

    with pytest.raises(ForeignOriginContentError):
        list(
            parse_one_source_path(
                str(rollout),
                file_mtime=None,
                source_name="claude-code",
                sidecar_data={},
                capture_raw=False,
            )
        )


def test_refusal_survives_process_boundaries() -> None:
    """Parse workers are subprocesses; the typed refusal must unpickle.

    Anti-vacuity: without ``__reduce__`` the keyword-only constructor makes
    unpickling raise ``TypeError``.
    """
    error = ForeignOriginContentError(expected=Provider.CLAUDE_CODE, found=Provider.CODEX, evidence="probe")
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, ForeignOriginContentError)
    assert (restored.expected, restored.found, restored.evidence) == (Provider.CLAUDE_CODE, Provider.CODEX, "probe")


def test_provider_wires_of_one_origin_are_not_foreign() -> None:
    """``drive`` and ``gemini`` wires both name AI Studio on Drive.

    Anti-vacuity: comparing wires by identity refuses a genuine AI Studio
    prompt at the Drive location.
    """
    prompt = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "Summarize this."}]}}
    assert detect_provider(prompt) is Provider.GEMINI
    assert detect_provider(prompt, expected=Provider.DRIVE) is Provider.GEMINI


def test_json_document_sampling_does_not_swallow_the_refusal(tmp_path: Path) -> None:
    """A foreign ``.json`` array is refused, not quietly given the fallback.

    Anti-vacuity: the sampler's ``except (OSError, ValueError)`` catches the
    refusal (a ``ValueError`` subclass) and returns ``CLAUDE_CODE``.
    """
    document = tmp_path / "projects" / "proj" / "export.json"
    document.parent.mkdir(parents=True)
    document.write_text(json.dumps(_CODEX_ROLLOUT), encoding="utf-8")
    with pytest.raises(ForeignOriginContentError):
        _detect_provider_from_path_sample(document, Provider.CLAUDE_CODE, json_document=True)


def test_raw_only_paths_are_classified_by_location_before_any_probe(tmp_path: Path) -> None:
    """Declared raw-only evidence keeps its location's origin whatever it holds.

    ``history.jsonl`` is Claude Code's raw-only prompt log. Holding
    Codex-shaped rows, it is still retained Claude Code evidence and never
    refused as foreign.

    Anti-vacuity: probing content before the raw-only declaration raises
    ``ForeignOriginContentError`` here.
    """
    history = tmp_path / "history.jsonl"
    history.write_bytes(_jsonl(_CODEX_ROLLOUT))
    assert _jsonl_provider_and_session_artifact(history, Provider.CLAUDE_CODE) == (Provider.CLAUDE_CODE, False)


def test_archive_members_at_a_bound_location_are_validated() -> None:
    """ZIP members inherit their archive's location binding.

    Anti-vacuity: an unbound member sniff classifies the Codex member as
    Codex under Claude Code's root.
    """
    from io import BytesIO

    from polylogue.sources.source_acquisition_components import iter_entry_payloads

    with pytest.raises(ForeignOriginContentError):
        list(
            iter_entry_payloads(
                BytesIO(_jsonl(_CODEX_ROLLOUT)),
                stream_name="member.jsonl",
                provider_hint=Provider.CLAUDE_CODE,
                bound_provider=Provider.CLAUDE_CODE,
            )
        )
    unbound = list(
        iter_entry_payloads(BytesIO(_jsonl(_CODEX_ROLLOUT)), stream_name="member.jsonl", provider_hint=Provider.UNKNOWN)
    )
    assert {entry.provider for entry in unbound} == {Provider.CODEX}


def test_one_shot_zip_parse_refuses_per_member(tmp_path: Path) -> None:
    """A refused ZIP member does not discard the admissible member after it.

    Anti-vacuity: letting the refusal escape ``process_zip`` records the whole
    archive as failed and never parses the Claude Code member.
    """
    import zipfile

    from polylogue.sources.decoder_zip import process_zip
    from polylogue.storage.cursor_state import CursorStatePayload

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a-codex.jsonl", _jsonl(_CODEX_ROLLOUT))
        zf.writestr("b-claude.jsonl", _jsonl(_CLAUDE_CODE_TRANSCRIPT))

    cursor_state: CursorStatePayload = {"failed_count": 0, "failed_files": []}
    sessions = [
        session
        for _raw, session in process_zip(
            archive,
            provider_hint=Provider.CLAUDE_CODE,
            should_group=True,
            file_mtime=None,
            capture_raw=False,
            cursor_state=cursor_state,
            blob_root=tmp_path / "blobs",
        )
    ]
    assert [session.source_name for session in sessions] == [Provider.CLAUDE_CODE]
    assert "foreign_origin_content" in str(cursor_state["failed_files"])


def test_one_shot_acquisition_refuses_grouped_foreign_zip_members(tmp_path: Path) -> None:
    """Grouped members bypass the splitter, so they are validated before preservation.

    Anti-vacuity: without the bounded sample check the Codex member is
    preserved under a Claude Code hint.
    """
    import zipfile

    from polylogue.config import Source
    from polylogue.sources.source_acquisition import iter_source_raw_data
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.cursor_state import CursorStatePayload

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a-codex.jsonl", _jsonl(_CODEX_ROLLOUT))
        zf.writestr("b-claude.jsonl", _jsonl(_CLAUDE_CODE_TRANSCRIPT))

    cursor_state: CursorStatePayload = {"failed_count": 0, "failed_files": []}
    items = list(
        iter_source_raw_data(
            Source(name="claude-code", path=archive),
            blob_store=BlobStore(tmp_path / "blobs"),
            cursor_state=cursor_state,
        )
    )
    assert [item.source_path for item in items] == [f"{archive}:b-claude.jsonl"]
    assert "foreign_origin_content" in str(cursor_state["failed_files"])


def test_baseline_replay_agrees_with_live_zip_refusal(tmp_path: Path) -> None:
    """Replay must refuse the members live acquisition refuses.

    The production baseline replays ZIP members to predict raw rows. If it
    accepts a member the live path refuses, verification waits forever for a
    row that can never exist.

    Anti-vacuity: replaying without the location binding yields the Codex
    member as an accepted payload.
    """
    import zipfile

    from polylogue.config import Source
    from polylogue.sources.source_acquisition_components import (
        ZipEntryReadContext,
        replay_zip_entry_acquisition_payloads,
    )

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a-codex.jsonl", _jsonl(_CODEX_ROLLOUT))
    with zipfile.ZipFile(archive) as zf:
        info = zf.infolist()[0]
        context = ZipEntryReadContext(
            Source(name="claude-code", path=tmp_path),
            archive,
            info,
            None,
            Provider.CLAUDE_CODE,
            None,  # type: ignore[arg-type]
            bound_provider=Provider.CLAUDE_CODE,
        )
        with pytest.raises(ForeignOriginContentError):
            list(replay_zip_entry_acquisition_payloads(zf, context))


def test_one_shot_fact_path_refuses_foreign_document(tmp_path: Path) -> None:
    """A foreign document at a non-session fact path is refused, not skipped.

    Anti-vacuity: without the fact-path check the Codex-shaped workflow
    document is silently skipped with no recorded refusal.
    """
    from polylogue.sources.origin_specs import artifact_rule_for_path

    workflows = tmp_path / ".claude" / "projects" / "proj" / "workflows"
    workflows.mkdir(parents=True)
    document = workflows / "wf-1.json"
    document.write_text(json.dumps(_CODEX_ROLLOUT), encoding="utf-8")
    rule = artifact_rule_for_path(Provider.CLAUDE_CODE, str(document))
    assert rule is not None and rule.parse_policy == "fact"
    with pytest.raises(ForeignOriginContentError):
        list(
            parse_one_source_path(
                str(document),
                file_mtime=None,
                source_name="claude-code",
                sidecar_data={},
                capture_raw=False,
            )
        )


def test_baseline_keeps_inbox_archives_unbound(tmp_path: Path) -> None:
    """An inbox archive classifies each member in replay, as live intake does.

    Anti-vacuity: binding replay to the sniffed dominant provider excludes the
    Gemini member of a ChatGPT-dominant inbox export as foreign.
    """
    import zipfile

    from polylogue.sources.live.production_baseline import _archive_members

    chatgpt_member = {
        "id": "chatgpt-1",
        "title": "t",
        "create_time": 1700000000.0,
        "mapping": {
            "n1": {
                "id": "n1",
                "message": {
                    "id": "n1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hi"]},
                },
                "children": [],
            }
        },
    }
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.json", json.dumps(chatgpt_member))
        zf.writestr("b.json", json.dumps(chatgpt_member | {"id": "chatgpt-2"}))
        zf.writestr("gemini.json", json.dumps({"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}}))

    decisions = _archive_members(archive, "inbox")
    assert not [decision for decision in decisions if "foreign_origin_content" in decision.reason]


def test_one_choke_point_refuses_foreign_material(tmp_path: Path) -> None:
    """``refuse_foreign_material`` is the shared bind-then-validate check.

    It refuses a foreign JSONL at a bound location and passes the inbox,
    declared raw-only paths and non-JSON material.

    Anti-vacuity: without the location binding the Codex rollout passes at a
    Claude Code location.
    """
    from polylogue.sources.dispatch import refuse_foreign_material

    rollout = tmp_path / "projects" / "proj" / "c0ffee00-1111-2222-3333-444455556666.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_bytes(_jsonl(_CODEX_ROLLOUT))
    with pytest.raises(ForeignOriginContentError):
        refuse_foreign_material(rollout, Provider.CLAUDE_CODE)
    refuse_foreign_material(rollout, Provider.UNKNOWN)
    refuse_foreign_material(rollout, Provider.CODEX)
    history = tmp_path / "history.jsonl"
    history.write_bytes(_jsonl(_CODEX_ROLLOUT))
    refuse_foreign_material(history, Provider.CLAUDE_CODE)
    notes = tmp_path / "notes.md"
    notes.write_text("# notes", encoding="utf-8")
    refuse_foreign_material(notes, Provider.CLAUDE_CODE)


def test_production_baseline_excludes_refused_plain_files(tmp_path: Path) -> None:
    """The baseline expects no raw row for a file live intake refuses.

    Anti-vacuity: without the choke point in the baseline, the Codex rollout
    under a Claude Code root is recorded as ``accepted``.
    """
    from polylogue.sources.live.production_baseline import capture_production_source_baseline
    from polylogue.sources.live.watcher import WatchSource

    root = tmp_path / "projects"
    project = root / "proj"
    project.mkdir(parents=True)
    (project / "c0ffee00-1111-2222-3333-444455556666.jsonl").write_bytes(_jsonl(_CODEX_ROLLOUT))
    (project / "bad69218-73bd-490a-869a-2b3a30bf421b.jsonl").write_bytes(_jsonl(_CLAUDE_CODE_TRANSCRIPT))
    baseline = capture_production_source_baseline(
        (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),),
        operation_id="op-test",
    )
    by_name = {Path(decision.path).name: decision for decision in baseline.decisions}
    assert by_name["c0ffee00-1111-2222-3333-444455556666.jsonl"].disposition == "excluded"
    assert "foreign_origin_content" in by_name["c0ffee00-1111-2222-3333-444455556666.jsonl"].reason
    assert by_name["bad69218-73bd-490a-869a-2b3a30bf421b.jsonl"].disposition == "accepted"


def test_publisher_discards_one_refused_pending_blob(tmp_path: Path) -> None:
    """A refused blob's queued publication is dropped before any flush.

    Anti-vacuity: without ``discard_pending_hash`` the refused blob stays
    queued and a later flush reserves it with no raw row to consume it.
    """
    from polylogue.storage.blob_publication import ArchiveBlobPublisher, discard_pending_blob

    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    kept, _ = publisher.write_from_bytes(b"kept")
    refused, _ = publisher.write_from_bytes(b"refused")
    assert discard_pending_blob(publisher, refused) is True
    assert discard_pending_blob(publisher, refused) is False
    assert publisher.receipt_id(refused) is None
    assert publisher.receipt_id(kept) is not None
    assert publisher.has_pending


def test_capture_validates_the_retained_bytes_not_the_path(tmp_path: Path) -> None:
    """The captured blob is what is validated, so a check-then-copy race cannot admit.

    The capture callable swaps the file to foreign content before copying,
    as a racing writer would. The refusal must fire on the captured bytes and
    drop the queued publication.

    Anti-vacuity: validating the path before capture passes (the file is
    still Claude Code then) and retains the Codex bytes.
    """
    from polylogue.sources.bound_capture import capture_bound_source
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    source = tmp_path / "projects" / "proj" / "bad69218-73bd-490a-869a-2b3a30bf421b.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(_jsonl(_CLAUDE_CODE_TRANSCRIPT))
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")

    def racing_capture() -> tuple[str, int]:
        source.write_bytes(_jsonl(_CODEX_ROLLOUT))
        return publisher.write_from_path(source)

    with pytest.raises(ForeignOriginContentError):
        capture_bound_source(publisher, source, Provider.CLAUDE_CODE, racing_capture)
    assert not publisher.has_pending


def test_grouped_member_validation_is_byte_bounded(tmp_path: Path) -> None:
    """One oversized record is not materialized to validate a member.

    The member is a single ~2 MiB JSONL record with a foreign envelope up
    front; validation reads only a bounded byte prefix yet still refuses.

    Anti-vacuity: record-count sampling decodes the whole record before
    deciding.
    """
    import zipfile

    from polylogue.config import Source
    from polylogue.sources.source_acquisition_components import (
        ZipEntryReadContext,
        validate_bound_grouped_zip_member,
    )

    record = {
        "type": "session_meta",
        "payload": {"id": "c1", "timestamp": "2026-01-01T10:00:00Z", "pad": "x" * 2_000_000},
    }
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("member.jsonl", (json.dumps(record) + "\n").encode("utf-8"))
    with zipfile.ZipFile(archive) as zf:
        context = ZipEntryReadContext(
            Source(name="claude-code", path=tmp_path),
            archive,
            zf.infolist()[0],
            None,
            Provider.CLAUDE_CODE,
            None,  # type: ignore[arg-type]
            bound_provider=Provider.CLAUDE_CODE,
        )
        with pytest.raises(ForeignOriginContentError):
            validate_bound_grouped_zip_member(zf, context)


def test_truncated_json_document_is_validated_from_its_partial_structure() -> None:
    """A foreign ``.json`` document larger than the prefix is still refused.

    Anti-vacuity: without the partial-structure fallback for documents, the
    invalid truncated prefix decides nothing and the location's own origin is
    assumed.
    """
    from polylogue.sources.dispatch import LOCATION_VALIDATION_PREFIX_BYTES, refuse_foreign_material

    document = json.dumps(_CODEX_ROLLOUT + [{"type": "pad", "pad": "x" * 50_000}]).encode("utf-8")
    prefix = document[:LOCATION_VALIDATION_PREFIX_BYTES]
    with pytest.raises(ForeignOriginContentError):
        refuse_foreign_material("export.json", Provider.CLAUDE_CODE, prefix=prefix)
