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
from polylogue.sources.acquisition_boundary import refuse_foreign_path
from polylogue.sources.dispatch import ForeignOriginContentError, detect_provider
from polylogue.sources.live.batch_support import _jsonl_provider_and_session_artifact
from polylogue.sources.source_layout import export_drop_layout
from polylogue.sources.source_parsing import parse_one_source_path
from tests.infra.source_builders import acquired_payloads

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


def test_gemini_cli_prompt_log_with_claude_code_shape_is_refused(tmp_path: Path) -> None:
    """A Gemini CLI prompt log that happens to look like Claude Code records.

    Real Gemini CLI ``logs.json`` rows carry ``sessionId``/``type``/
    ``message`` keys that the Claude Code detector recognizes; before binding,
    they were admitted as Claude Code sessions from Gemini CLI's directory.

    Anti-vacuity: without binding the boundary admits the log.
    """
    chats = tmp_path / "tmp" / "abc" / "chats"
    chats.mkdir(parents=True)
    log = chats / "logs.jsonl"
    log.write_bytes(_jsonl(_CLAUDE_CODE_TRANSCRIPT))

    with pytest.raises(ForeignOriginContentError) as refused:
        refuse_foreign_path(log, Provider.GEMINI_CLI)
    assert refused.value.found is Provider.CLAUDE_CODE


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


def test_raw_only_paths_are_classified_by_location_before_any_probe(tmp_path: Path) -> None:
    """Declared raw-only evidence keeps its location's origin whatever it holds.

    ``history.jsonl`` is Claude Code's raw-only prompt log. Holding
    Codex-shaped rows, it is still retained Claude Code evidence and never
    refused as foreign.

    Anti-vacuity: validating content before the raw-only declaration raises
    ``ForeignOriginContentError`` here.
    """
    history = tmp_path / ".claude" / "history.jsonl"
    history.parent.mkdir()
    history.write_bytes(_jsonl(_CODEX_ROLLOUT))
    refuse_foreign_path(history, Provider.CLAUDE_CODE)
    assert _jsonl_provider_and_session_artifact(history, Provider.CLAUDE_CODE) == (Provider.CLAUDE_CODE, False, None)


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
    from polylogue.sources.source_acquisition import iter_source_acquisition_records
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.cursor_state import CursorStatePayload

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a-codex.jsonl", _jsonl(_CODEX_ROLLOUT))
        zf.writestr("b-claude.jsonl", _jsonl(_CLAUDE_CODE_TRANSCRIPT))

    cursor_state: CursorStatePayload = {"failed_count": 0, "failed_files": []}
    items = list(
        acquired_payloads(
            iter_source_acquisition_records(
                Source(name="claude-code", path=archive),
                blob_store=BlobStore(tmp_path / "blobs"),
                cursor_state=cursor_state,
            )
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
        replay_zip_entry_acquisition_revisions,
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
            list(replay_zip_entry_acquisition_revisions(zf, context))


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
        (WatchSource(name="claude-code", root=root, layout=export_drop_layout((".jsonl",))),),
        operation_id="op-test",
    )
    by_name = {Path(decision.path).name: decision for decision in baseline.decisions}
    assert by_name["c0ffee00-1111-2222-3333-444455556666.jsonl"].disposition == "excluded"
    assert "foreign_origin_content" in by_name["c0ffee00-1111-2222-3333-444455556666.jsonl"].reason
    assert by_name["bad69218-73bd-490a-869a-2b3a30bf421b.jsonl"].disposition == "accepted"


def test_publisher_discards_one_refused_pending_blob(tmp_path: Path) -> None:
    """A refused capture's queued publication is dropped before any flush.

    Anti-vacuity: without ``discard_pending_receipt`` the refused blob stays
    queued and a later flush reserves it with no raw row to consume it.
    """
    from polylogue.sources.acquisition_boundary import release_refused_capture
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    kept, _ = publisher.write_from_bytes(b"kept")
    refused, _ = publisher.write_from_bytes(b"refused")
    refused_receipt = publisher.receipt_id(refused)
    assert refused_receipt is not None
    release_refused_capture(publisher, refused, refused_receipt)
    assert publisher.discard_pending_receipt(refused_receipt) is False
    assert publisher.receipt_id(refused) is None
    assert publisher.receipt_id(kept) is not None
    assert [receipt.blob_hash for receipt, _ in publisher._pending] == [kept]

    # An adoption of the same bytes queued behind a discarded capture keeps
    # its own receipt; discarding the adoption leaves nothing to reserve.
    adopted, _ = publisher.write_from_bytes(b"adopted")
    adoption_hash, _ = publisher.adopt_published(adopted, len(b"adopted"))
    adoption_receipt = publisher.receipt_id(adoption_hash)
    assert adoption_receipt is not None
    [capture_receipt] = [receipt.publication_id for receipt, _ in publisher._pending if receipt.blob_hash == adopted]
    assert publisher.discard_pending_receipt(capture_receipt) is True
    assert publisher.receipt_id(adopted) == adoption_receipt
    assert publisher.discard_pending_receipt(adoption_receipt) is True
    assert publisher.receipt_id(adopted) is None
    assert [receipt.blob_hash for receipt, _ in publisher._pending] == [kept]
    assert not publisher._adoptions


def test_refused_unit_releases_every_capture_even_identical_ones(tmp_path: Path) -> None:
    """A refusal releases each capture of its unit, keyed by receipt.

    Two byte-identical captures share a blob hash; both must be released,
    and an unrelated capture must survive.

    Anti-vacuity: releasing by hash drops only the latest identical capture
    and leaves the first queued for publication.
    """
    from polylogue.sources.acquisition_boundary import release_captures_on_refusal
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    kept, _ = publisher.write_from_bytes(b"sibling")
    with pytest.raises(ForeignOriginContentError):
        with release_captures_on_refusal(publisher) as captures:
            for _ in range(2):
                blob_hash, _size = publisher.write_from_bytes(b"split")
                captures.append((blob_hash, publisher.receipt_id(blob_hash)))
            raise ForeignOriginContentError(expected=Provider.CHATGPT, found=Provider.CLAUDE_AI, evidence="probe")
    assert [receipt.blob_hash for receipt, _ in publisher._pending] == [kept]


def test_clearing_one_archives_member_debt_spares_a_case_sibling(tmp_path: Path) -> None:
    """A scan of ``a.zip`` clears only ``a.zip``'s member refusals.

    Anti-vacuity: selecting subjects with SQLite's default ``LIKE`` folds
    ASCII case, so ``A.zip``'s still-active refusal gap is deleted too.
    """
    from polylogue.sources.live.cursor import CursorStore

    cursor = CursorStore(tmp_path / "cursor.sqlite")
    for subject in ("/sessions/a.zip:m1", "/sessions/A.zip:m1"):
        cursor.record_convergence_debt(
            stage="live_ingest_admission", subject_type="source_path", subject_id=subject, error="refused"
        )
    cursor.clear_convergence_debt_under_prefix(
        stage="live_ingest_admission", subject_type="source_path", prefix="/sessions/a.zip:"
    )
    remaining = {debt.subject_id for debt in cursor.list_convergence_debt(stage="live_ingest_admission")}
    assert remaining == {"/sessions/A.zip:m1"}


def test_a_bound_parse_stream_releases_no_session_before_it_validates() -> None:
    """At a bound location a stream is one admission unit.

    A Codex record after admissible Claude Code records is refused when its
    bytes are read; no session parsed from the earlier records may escape
    first. Anti-vacuity: streaming sessions out of ``emit`` as they parse
    yields the Claude Code session before the refusal.
    """
    from io import BytesIO

    from polylogue.sources.cursor import _ParseContext
    from polylogue.sources.emitter import _SessionEmitter

    emitter = _SessionEmitter(
        _ParseContext(
            provider_hint=Provider.CLAUDE_CODE,
            should_group=False,
            source_path_str="member.jsonl",
            fallback_id="member",
            file_mtime=None,
            capture_raw=False,
            sidecar_data={},
            bound_provider=Provider.CLAUDE_CODE,
        )
    )
    released = []
    with pytest.raises(ForeignOriginContentError):
        for item in emitter.emit(BytesIO(_jsonl([*_CLAUDE_CODE_TRANSCRIPT, *_CODEX_ROLLOUT])), "member.jsonl"):
            released.append(item)
    assert released == []


@pytest.mark.parametrize(
    ("relative", "expected_count"), [("workflows/recovered.json", 1), ("tool-results/opaque.json", 0)]
)
def test_decoded_json_session_outranks_only_parseable_fact_paths(
    tmp_path: Path, relative: str, expected_count: int
) -> None:
    from polylogue.sources.source_parsing import has_decoded_session_evidence

    path = tmp_path / ".claude" / "projects" / "neutral" / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_CLAUDE_CODE_TRANSCRIPT))
    assert has_decoded_session_evidence(path, provider=Provider.CLAUDE_CODE)
    parsed = list(
        parse_one_source_path(str(path), file_mtime=None, source_name="claude-code", sidecar_data={}, capture_raw=False)
    )
    assert len(parsed) == expected_count
    if parsed:
        raw, session = parsed[0]
        assert raw is None
        assert session.source_name is Provider.CLAUDE_CODE
        assert [message.provider_message_id for message in session.messages] == ["u1", "a1"]
        assert [block.text for message in session.messages for block in message.blocks] == [
            "Search for ad-hoc solutions.",
            "Looking now.",
        ]
