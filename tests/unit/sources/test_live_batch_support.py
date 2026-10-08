from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sqlite3
import zipfile
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import (
    HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live import batch as live_batch
from polylogue.sources.live.batch import (
    _MAX_APPEND_PLAN_PAYLOAD_BYTES,
    RAW_RETENTION_LIMIT_PER_PATH,
    RAW_RETENTION_STAGE,
    LiveBatchProcessor,
    _ArchiveFullWriteResult,
    append_capability_receipt,
)
from polylogue.sources.live.batch_support import (
    _BROWSER_CAPTURE_PREFIX_PROBE_BYTES,
    _DEFER_APPEND,
    JsonlBoundary,
    _AppendPlan,
    _AppendResult,
    _browser_capture_prefix_probe,
    _detect_provider_from_path,
    _FullIngestResult,
    _parse_path_as_session_artifact,
    classify_pre_writer_admissions,
    encode_cursor_hash_authority,
    jsonl_complete_prefix,
    jsonl_complete_prefix_path,
    jsonl_parse_prefix_size,
    jsonl_parse_prefix_size_of_handle,
    sha256_range_from_path,
    tail_hash_from_path,
)
from polylogue.sources.live.convergence_debt import ConvergenceDebt
from polylogue.sources.live.cursor import ConvergenceDebtSettlement, CursorStore
from polylogue.sources.live.metrics import REFUSED_CORRUPT_INPUT, REFUSED_NO_SESSIONS, SETTLED_EXCLUSION_REASONS
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.source_acquisition_components import stream_preserved_zip_entry_raw_data
from polylogue.sources.source_layout import export_drop_layout
from polylogue.sources.source_parsing import has_decoded_session_evidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session
from tests.infra.raw_owner_routes import (
    ingest_append_with_owner,
    ingest_files_with_owners,
    run_ingest_files,
    seed_membership_census,
    supplied_live_owners,
)
from tests.infra.retained_replay import replay_retained_components
from tests.infra.source_builders import (
    live_zip_capture,
    make_chatgpt_node,
    make_claude_chat_message,
)

_RETIRED_FULL_INGEST_SIZE_BOUND = 8 * 1024 * 1024


def _retained_parse_by_path(
    sessions_for: Callable[[Path], list[ParsedSession]],
) -> Callable[..., Generator[ParsedSession, None, None]]:
    """Stand in for retained preparation's stream parser, keyed by the raw's source path.

    Retained preparation is the only parser on the live route, and a replay
    re-parses the same retained raw, so the stub answers per path rather
    than per call.
    """

    def parse(*_args: object, source_path: str, **_kwargs: object) -> Generator[ParsedSession, None, None]:
        yield from sessions_for(Path(source_path))

    return parse


def _codex_shaped_bytes(tag: str) -> bytes:
    """Distinct rollout bytes retained preparation admits as Codex session input."""
    return (
        json.dumps({"type": "session_meta", "payload": {"id": tag, "timestamp": "2026-06-02T00:00:00Z"}})
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": tag}]},
            }
        )
        + "\n"
    ).encode()


def _full_paths_sync(processor: LiveBatchProcessor, paths: list[Path], *, source_name: str, **kwargs: Any) -> Any:
    """Run the full-path route with captures prepared as the live route prepares them.

    ``LiveBatchProcessor._ingest_full_paths`` seals declared Codex state
    databases through the capture stage before the writer runs and discards
    them afterwards; other inputs are acquired by path inside the body.
    ``_ingest_full_paths_prepared`` then acquires on the daemon's admitted
    writer and publishes the acquired raws through the retained raw owner,
    both supplied here as the daemon supplies them.
    """
    import threading

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.provider_identity import canonical_acquisition_provider
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage, PreparedLiveSQLiteCapture
    from polylogue.sources.origin_specs import database_capability_for_provider

    provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
    capability = database_capability_for_provider(Provider.CODEX)
    state_paths = [
        path
        for path in paths
        if provider in (Provider.CODEX, Provider.UNKNOWN)
        and capability is not None
        and (member := capability.member(path.name)) is not None
        and member.disposition != "out-of-scope"
    ]
    archive_root = Path(getattr(processor._polylogue, "archive_root", processor._cursor._db_path.parent))
    captures: dict[Path, PreparedLiveSQLiteCapture | Exception] = {}
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    try:
        if state_paths:
            captures = LiveSQLiteCaptureStage(compute_adapter=compute).prepare_sqlite_paths(
                state_paths, archive_root=archive_root, cancelled=threading.Event(), fallback_provider=provider
            )

        admissions = classify_pre_writer_admissions(
            [path for path in paths if path not in captures], fallback_provider=provider
        )

        async def run() -> Any:
            async with supplied_live_owners(processor):
                return await processor._ingest_full_paths_prepared(
                    paths, source_name=source_name, pre_writer_admissions={**admissions, **captures}, **kwargs
                )

        return asyncio.run(run())
    finally:
        for capture in captures.values():
            if isinstance(capture, PreparedLiveSQLiteCapture):
                capture.discard()
        compute.shutdown(wait=True)


@pytest.mark.parametrize(
    ("payload", "prefix_size", "record_count", "incomplete"),
    [
        (b"", 0, 0, False),
        (b'{"a":1}', 7, 1, False),
        (b'{"a":1}\n{"b":', 8, 1, True),
        (b'{"text":"brace } and \\"quote\\"\\n"}\r\n{"n":2}\r\n', 45, 2, False),
    ],
)
def test_jsonl_complete_prefix_is_lexical_and_newline_bound(
    payload: bytes, prefix_size: int, record_count: int, incomplete: bool
) -> None:
    result = jsonl_complete_prefix(payload)
    assert (result.prefix_size, result.record_count, result.incomplete_tail) == (
        prefix_size,
        record_count,
        incomplete,
    )


def test_jsonl_complete_prefix_validates_only_the_tail_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restored forward scanner validates every record instead of only the tail."""
    from polylogue.sources.live import batch_support

    payload = b'{"record":0}\n' * 10_000 + b'{"partial":'
    original = batch_support._valid_jsonl_tail
    validated: list[int] = []

    def tail_only(handle: Any, start: int, end: int, **kwargs: Any) -> bool:
        validated.append(end - start)
        return original(handle, start, end, **kwargs)

    monkeypatch.setattr(batch_support, "_valid_jsonl_tail", tail_only)
    boundary = jsonl_complete_prefix(payload)
    assert boundary == JsonlBoundary(len(payload) - len(b'{"partial":'), 10_000, True, False)
    assert validated == [len(b'{"partial":')]


def test_unfinished_jsonl_tail_keeps_its_complete_prefix_retryable() -> None:
    """An ordinary mid-write snapshot must not be reported as a bad record.

    Anti-vacuity: report ``malformed_record=True`` here and ``batch.py``
    suppresses ``complete_prefix_size``, which turns an unchanged capture into
    ``TERMINAL_CORRUPT_INPUT`` and withholds ``{"ok":1}`` entirely.
    """
    payload = b'{"ok":1}\n{"partial":'

    boundary = jsonl_complete_prefix(payload)

    assert boundary.prefix_size == len(b'{"ok":1}\n')
    assert boundary.record_count == 1
    assert boundary.incomplete_tail is True
    assert boundary.malformed_record is False


def test_terminated_malformed_jsonl_record_stays_malformed() -> None:
    """The opposite direction: a blanket ``False`` must not pass either."""
    payload = b'{"ok":1}\n{"broken":}\n'

    boundary = jsonl_complete_prefix(payload)

    assert boundary.prefix_size == len(b'{"ok":1}\n')
    assert boundary.malformed_record is True


def test_jsonl_complete_prefix_keeps_malformed_record_before_terminal_blank_lines() -> None:
    """A one-line tail probe must not skip a malformed, newline-terminated record."""
    payload = b'{"accepted":1}\n{"malformed":}\n\n'

    boundary = jsonl_complete_prefix(payload)

    assert boundary == JsonlBoundary(len(b'{"accepted":1}\n'), 1, True, True)


def test_jsonl_complete_prefix_counts_records_before_terminal_blank_lines() -> None:
    """Blank JSONL separators are not admitted records."""
    payload = b'{"first":1}\n\n{"second":2}\n\n'

    boundary = jsonl_complete_prefix(payload)

    assert boundary == JsonlBoundary(len(payload), 2, False)


def test_claude_frontier_accepts_header_replacement_and_conserves_body(tmp_path: Path) -> None:
    path, plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="mutable-header",
        append=b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n',
    )
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    assert processor._record_append_cursor(plan)
    original_body = path.read_bytes().split(b"\n", 1)[1]
    replacement = b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"new"},"uuid":"header","sessionId":"mutable-header"}\n'
    second_append = b'{"type":"assistant","message":{"role":"assistant","content":"two"},"uuid":"message-2"}\n'
    path.write_bytes(replacement + original_body + second_append)

    revised = processor._append_plan(path)

    assert isinstance(revised, _AppendPlan)
    assert revised.payload == second_append
    assert revised.start_offset == len(replacement) + len(original_body)


def test_claude_frontier_rejects_body_rewrite(tmp_path: Path) -> None:
    path, plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="rewritten-body",
        append=b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n',
    )
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    assert processor._record_append_cursor(plan)
    header, body = path.read_bytes().split(b"\n", 1)
    path.write_bytes(
        header
        + b"\n"
        + body.replace(b'"one"', b'"changed"', 1)
        + b'{"type":"assistant","message":{"role":"assistant","content":"tail"},"uuid":"message-2"}\n'
    )

    assert processor._append_plan(path) is None


def test_a_foreign_appended_record_falls_back_to_the_refusing_full_route(tmp_path: Path) -> None:
    """An appended Codex record under Claude Code is not retained as an append.

    Anti-vacuity: skip validating the append delta and this plan carries the
    Codex bytes as Claude Code raw evidence; the full route, which records the
    typed foreign-origin refusal, is never taken.
    """
    path, plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="foreign-append",
        append=b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n',
    )
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    assert processor._record_append_cursor(plan)
    accepted = path.read_bytes()
    codex = b'{"type":"session_meta","payload":{"id":"codex-session-1","timestamp":"2026-01-01T10:00:00Z"}}\n'
    path.write_bytes(accepted + codex)

    assert processor._append_plan(path) is None

    own = b'{"type":"assistant","message":{"role":"assistant","content":"two"},"uuid":"message-2"}\n'
    path.write_bytes(accepted + own)
    assert isinstance(processor._append_plan(path), _AppendPlan)


def test_append_prefix_cursor_refuses_a_rewritten_accepted_prefix_after_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: trusting only the unchanged tail would advance this cursor."""
    path, plan, owner, processor = _seed_live_append_plan(tmp_path, native_id="prefix-publication")
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    rewritten = path.read_bytes().replace(b"zero", b"zeta", 1)
    path.write_bytes(rewritten)
    monkeypatch.setattr(
        "polylogue.sources.live.batch.tail_hash_from_path",
        lambda _path, _end: (cast(str, plan.accepted_tail_hash), 0),
    )

    assert processor._record_append_cursor(plan) is False


def test_append_prefix_cursor_refuses_claude_body_rewrite_after_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: recomputing the frontier must not bless a changed stable body."""
    first_append = b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n'
    path, first_plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="claude-publication",
        append=first_append,
    )
    assert ingest_append_with_owner(owner, [first_plan]).succeeded == [first_plan]
    assert processor._record_append_cursor(first_plan)
    second_append = b'{"type":"assistant","message":{"role":"assistant","content":"two"},"uuid":"message-2"}\n'
    with path.open("ab") as handle:
        handle.write(second_append)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    path.write_bytes(path.read_bytes().replace(b"one", b"bad", 1))
    monkeypatch.setattr(
        "polylogue.sources.live.batch.tail_hash_from_path",
        lambda _path, _end: (cast(str, plan.accepted_tail_hash), 0),
    )

    assert processor._record_append_cursor(plan) is False


def test_append_cursor_hands_off_claude_plan_when_source_disappears_after_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: treating disappearance as a failed proof loses an admitted append."""
    path, plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="claude-disappeared-publication",
        append=b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n',
    )
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]

    def missing(_self: Path) -> os.stat_result:
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "stat", missing)

    assert processor._record_append_cursor(plan) is True
    cursor = processor._cursor.get_record(path)
    assert cursor is not None
    assert cursor.byte_offset == plan.last_complete_newline


def test_append_cursor_retries_claude_payload_mismatch_instead_of_raising(tmp_path: Path) -> None:
    """Mutation: a changed admitted append must become a stale-cursor retry, not escape the batch."""
    path, plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="claude-payload-mismatch",
        append=b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n',
    )
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    path.write_bytes(path.read_bytes().replace(b'"one"', b'"two"', 1))

    assert processor._record_append_cursor(plan) is False


def test_append_cursor_publishes_current_claude_header_boundary_and_observation(tmp_path: Path) -> None:
    """Mutation: plan-time offsets would point inside a rewritten mutable header."""
    path, plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="claude-header-publication",
        append=b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n',
    )
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    header, body = path.read_bytes().split(b"\n", 1)
    path.write_bytes(header.replace(b'"zero"', b'"a longer mutable header"') + b"\n" + body)
    current_stat = path.stat()
    current_header_end = path.read_bytes().index(b"\n") + 1

    assert processor._record_append_cursor(plan) is True
    cursor = processor._cursor.get_record(path)
    assert cursor is not None
    assert cursor.byte_size == current_stat.st_size
    assert cursor.byte_offset == current_header_end + len(plan.payload)
    assert cursor.last_complete_newline == current_header_end + len(plan.payload)
    assert (cursor.st_dev, cursor.st_ino, cursor.mtime_ns) == (
        current_stat.st_dev,
        current_stat.st_ino,
        current_stat.st_mtime_ns,
    )


def test_append_cursor_publishes_after_shorter_claude_header_rewrite(tmp_path: Path) -> None:
    """Mutation: a shorter mutable header must not resemble body truncation."""
    path, plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="claude-shorter-header-publication",
        append=b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n',
    )
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    _header, body = path.read_bytes().split(b"\n", 1)
    replacement = (
        b'{"type":"user","message":{"role":"user","content":"x"},"sessionId":"claude-shorter-header-publication"}\n'
    )
    assert len(replacement) < plan.start_offset
    path.write_bytes(replacement + body)

    assert processor._record_append_cursor(plan) is True
    cursor = processor._cursor.get_record(path)
    assert cursor is not None
    assert cursor.byte_size == path.stat().st_size
    assert cursor.byte_offset == len(replacement) + len(body)


def test_append_cursor_counts_failed_claude_frontier_proof_bytes(tmp_path: Path) -> None:
    """Mutation: omitting a failed semantic proof understates cursor-read amplification."""
    first_append = b'{"type":"assistant","message":{"role":"assistant","content":"one"},"uuid":"message-1"}\n'
    path, first_plan, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="claude-failed-proof-metrics",
        append=first_append,
    )
    assert ingest_append_with_owner(owner, [first_plan]).succeeded == [first_plan]
    assert processor._record_append_cursor(first_plan)
    second_append = b'{"type":"assistant","message":{"role":"assistant","content":"two"},"uuid":"message-2"}\n'
    with path.open("ab") as handle:
        handle.write(second_append)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    path.write_bytes(path.read_bytes().replace(b'"one"', b'"bad"', 1))

    assert processor._record_append_cursor(plan) is False
    header_bytes = path.read_bytes().index(b"\n") + 1
    assert processor._last_append_cursor_proof_bytes == header_bytes + len(plan.payload) + plan.last_complete_newline


def test_append_cursor_retries_when_source_grows_during_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: accepting a changed observation allows a concurrent writer to race proof publication."""
    path, plan, owner, processor = _seed_live_append_plan(tmp_path, native_id="growth-during-publication")
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    original_hash = sha256_range_from_path
    grew = False

    def grow_after_first_hash(
        source_path: Path,
        *,
        start_offset: int,
        end_offset: int,
    ) -> tuple[str, int]:
        nonlocal grew
        result = original_hash(source_path, start_offset=start_offset, end_offset=end_offset)
        if not grew:
            with path.open("ab") as handle:
                handle.write(
                    b'{"type":"response_item","payload":{"type":"message","id":"later","role":"assistant",'
                    b'"content":[{"type":"output_text","text":"later"}]}}\n'
                )
            grew = True
        return result

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", grow_after_first_hash)

    assert processor._record_append_cursor(plan) is False


def test_append_prefix_cursor_refuses_parser_semantics_drift_after_planning(tmp_path: Path) -> None:
    """Mutation: publishing under a new parser version would mislabel the proof."""
    path, plan, owner, processor = _seed_live_append_plan(tmp_path, native_id="parser-publication")
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    processor._parser_fingerprint = lambda: "new-parser-semantics"

    assert processor._record_append_cursor(plan) is False


@pytest.mark.parametrize(
    ("provider", "stable_session_identity", "status"),
    [
        ("codex", False, "unsupported"),
        ("codex", True, "supported"),
        ("claude-code", False, "unsupported"),
        ("claude-code", True, "supported"),
        ("chatgpt", True, "unsupported"),
    ],
)
def test_append_capability_receipt_is_keyed_to_live_identity_contract(
    provider: str,
    stable_session_identity: bool,
    status: str,
) -> None:
    receipt = append_capability_receipt(
        provider=provider,
        package_version="v1",
        element_kind="session_record_stream",
        stable_session_identity=stable_session_identity,
    )

    assert receipt.status == status
    payload = receipt.to_dict()
    assert (payload["provider"], payload["package_version"], payload["element_kind"]) == (
        provider,
        "v1",
        "session_record_stream",
    )
    assert payload["capability_source"] == "LiveBatchProcessor.append"
    assert payload["operation"] == "append_prefix"
    if provider not in {"codex", "claude-code"}:
        assert payload["reason"] == "live append route supports only Codex and Claude Code JSONL identity contracts"
    elif not stable_session_identity:
        assert payload["reason"] == "append delta requires a stable persisted session identity"
    else:
        assert payload["reason"] is None


from polylogue.sources.sqlite_export import logical_source_context
from polylogue.storage.sqlite.agent_thread_state import read_spawn_edges, read_thread_titles
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    read_archive_raw_session_envelope,
    upsert_raw_artifact,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.cursor_authority import fixture_cursor_authority
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture
from tests.infra.prepared_membership import publish_prepared_membership_classification

_ARCHIVE_STORAGE_TIERS = ",".join(spec.tier.value for spec in ARCHIVE_TIER_SPECS.values())


def _complete_archive_storage_probe_fields() -> dict[str, object]:
    return _archive_storage_probe_fields(
        present={spec.tier for spec in ARCHIVE_TIER_SPECS.values()},
        versions={spec.tier: spec.version for spec in ARCHIVE_TIER_SPECS.values()},
    )


def _archive_storage_probe_fields(
    *,
    present: set[ArchiveTier],
    versions: dict[ArchiveTier, int | None],
) -> dict[str, object]:
    return {
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
        "archive_present_tiers": ",".join(
            spec.tier.value for spec in ARCHIVE_TIER_SPECS.values() if spec.tier in present
        ),
        "archive_missing_tiers": ",".join(
            spec.tier.value for spec in ARCHIVE_TIER_SPECS.values() if spec.tier not in present
        ),
        "archive_tier_user_versions_json": json.dumps(
            {spec.tier.value: versions.get(spec.tier) for spec in ARCHIVE_TIER_SPECS.values()},
            sort_keys=True,
        ),
    }


def _write_archive_blob(archive_root: Path, blob_hash: bytes | str, payload: bytes) -> None:
    blob_hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else blob_hash.lower()
    blob_path = archive_root / "blob" / blob_hash_hex[:2] / blob_hash_hex[2:]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)


def _cursor_hash_authority(payload: bytes) -> str:
    return encode_cursor_hash_authority(
        sha256(payload).hexdigest(),
        sha256(payload[-64 * 1024 :]).hexdigest(),
        ctime_ns=0,
    )


def _append_plan(path: Path, payload: bytes, *, payload_hash: str, native_id_hint: str | None = None) -> _AppendPlan:
    stat = path.stat()
    return _AppendPlan(
        path=path,
        canonical_source_path=str(path),
        captured_profile_key=None,
        source_name="codex",
        start_offset=0,
        last_complete_newline=stat.st_size,
        stat_size=stat.st_size,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        payload=payload,
        payload_hash=payload_hash,
        cursor_fingerprint="base",
        bytes_read=len(payload),
        native_id_hint=native_id_hint,
    )


def _append_owner(archive_root: Path) -> object:
    # Append acquisition writes the Source tier of a bootstrapped root; a test
    # that built its own tiers keeps them.
    if not (archive_root / "source.db").exists():
        run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
    cursor = CursorStore(archive_root / "append.sqlite")
    return SimpleNamespace(
        _cursor=cursor,
        _polylogue=SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=cursor._db_path)),
    )


def _raw_parse_state(archive_root: Path) -> tuple[int | None, str | None]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        row = conn.execute("SELECT parsed_at_ms, parse_error FROM raw_sessions").fetchone()
    assert row is not None
    return cast(tuple[int | None, str | None], row)


def _append_raw_parse_state(archive_root: Path) -> tuple[int | None, str | None]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        row = conn.execute(
            """SELECT parsed_at_ms, parse_error FROM raw_sessions
               WHERE source_index = -1 ORDER BY acquired_at_ms DESC, raw_id DESC LIMIT 1"""
        ).fetchone()
    assert row is not None
    return cast(tuple[int | None, str | None], row)


def _raw_revision_envelope_row(archive_root: Path, raw_id: str) -> tuple[object, ...]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        row = conn.execute(
            """
            SELECT logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
    assert row is not None
    return cast(tuple[object, ...], row)


def _codex_meta_line(native_id: str) -> bytes:
    return f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'.encode()


def _codex_record_line(text: str) -> bytes:
    return (
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"' + text.encode() + b'"}]}}\n'
    )


def _archive_codex_session(archive_root: Path, native_id: str) -> None:
    """Archive the Codex session an append delta binds to.

    The planner emits an append plan only for a delta whose session identity
    is already bound (c07c4f44b1); without one the full route re-reads the
    file instead.
    """

    def seed() -> None:
        with ArchiveStore(archive_root) as store:
            write_index_session(
                store,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=native_id,
                    title=native_id,
                    messages=[ParsedMessage(provider_message_id=f"{native_id}-0", role=Role.USER, text="seed")],
                ),
            )

    run_off_event_loop(seed)


def _seed_live_append_plan(
    archive_root: Path,
    *,
    native_id: str,
) -> tuple[Path, _AppendPlan, object, LiveBatchProcessor]:
    root = archive_root / "sessions"
    root.mkdir()
    path = root / f"{native_id}.jsonl"
    baseline = (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}",'
        '"timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        '"role":"user","content":[{"type":"input_text","text":"zero"}]}}\n'
    ).encode()
    path.write_bytes(baseline)
    index_db = archive_root / "index.db"
    bootstrap_archive_root(archive_root)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    seeded = run_ingest_files(processor, [path], emit_event=False)
    assert seeded.succeeded_file_count == 1
    append = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
        b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
    )
    with path.open("ab") as handle:
        handle.write(append)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    return path, plan, _append_owner(archive_root), processor


def test_append_debt_lock_failure_keeps_frontier_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused debt batch must leave the advanced append available to replay."""
    path, _plan, _owner, processor = _seed_live_append_plan(tmp_path, native_id="debt-lock-replay")
    cursor = processor._cursor
    before = cursor.get_record(path)
    assert before is not None
    original_record_outcomes = processor._record_convergence_outcomes

    def hold_ops_lock_then_record(
        outcomes: Iterable[tuple[Path, Iterable[ConvergenceDebt]]], settlements: Iterable[ConvergenceDebtSettlement]
    ) -> None:
        blocker = sqlite3.connect(cursor._ops_db_path, timeout=0.001)
        blocker.execute("BEGIN IMMEDIATE")
        scope_conn = cast(Any, cursor._ops_scope).conn
        assert scope_conn is not None
        scope_conn.execute("PRAGMA busy_timeout = 1")
        try:
            original_record_outcomes(outcomes, settlements)
        finally:
            blocker.rollback()
            blocker.close()

    monkeypatch.setattr(processor, "_record_convergence_outcomes", hold_ops_lock_then_record)
    with pytest.raises(RuntimeError, match="convergence debt batch was not persisted"):
        run_ingest_files(processor, [path], emit_event=False)

    after_refusal = cursor.get_record(path)
    assert after_refusal is not None
    assert after_refusal.byte_offset == before.byte_offset
    assert processor._append_plan(path) is not None

    monkeypatch.setattr(processor, "_record_convergence_outcomes", original_record_outcomes)
    replayed = run_ingest_files(processor, [path], emit_event=False)

    assert replayed.succeeded_file_count == 1
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == path.stat().st_size
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'codex-session:debt-lock-replay'"
        ).fetchone() == (2,)


def _seed_claude_live_append_plan(
    archive_root: Path,
    *,
    native_id: str,
    append: bytes,
) -> tuple[Path, _AppendPlan, object, LiveBatchProcessor]:
    root = archive_root / "claude-projects"
    root.mkdir()
    path = root / f"{native_id}.jsonl"
    baseline = (
        f'{{"parentUuid":null,"type":"user","message":{{"role":"user","content":"zero"}},'
        f'"uuid":"message-0","timestamp":"2026-06-02T00:00:00Z","sessionId":"{native_id}"}}\n'
    ).encode()
    path.write_bytes(baseline)
    index_db = archive_root / "index.db"
    bootstrap_archive_root(archive_root)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    seeded = run_ingest_files(processor, [path], emit_event=False)
    assert seeded.succeeded_file_count == 1
    with path.open("ab") as handle:
        handle.write(append)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    return path, plan, _append_owner(archive_root), processor


def test_live_append_replay_streams_retained_jsonl_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append replay must not resurrect eager blob materialization."""
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="streamed-append")
    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("append replay eagerly read raw blob")),
    )

    result = ingest_append_with_owner(owner, [plan])

    assert result.succeeded == [plan]
    assert result.failed == []


def test_live_append_acquires_with_unreadable_active_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="degraded-append")
    (tmp_path / ".index-active-pointer").write_bytes(b"\xff")
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="derived generation unavailable",
            derived_only=True,
        )
    )
    monkeypatch.setattr(
        "polylogue.sources.dispatch.parse_stream_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("source-only append must not parse")),
    )
    try:
        result = ingest_append_with_owner(owner, [plan])
    finally:
        clear_degraded()

    assert result.succeeded == [plan]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        append_row = conn.execute(
            """
            SELECT logical_source_key, revision_kind, predecessor_raw_id,
                   baseline_raw_id, append_start_offset, append_end_offset,
                   revision_authority
            FROM raw_sessions
            WHERE source_index = -1
            """
        ).fetchone()
    assert append_row is not None
    assert append_row[:2] == ("codex-session:degraded-append", "append")
    assert append_row[2] is not None
    assert append_row[3] is not None
    assert append_row[4:] == (
        plan.start_offset,
        plan.last_complete_newline,
        "byte_proven",
    )


def test_source_only_file_history_append_binds_before_artifact_classification(tmp_path: Path) -> None:
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    native_id = "source-only-history"
    append = (
        f'{{"type":"file-history-snapshot","sessionId":"{native_id}",'
        '"uuid":"history-1","snapshot":{},"timestamp":"2026-06-02T00:00:01Z"}\n'
    ).encode()
    _path, plan, owner, _processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id=native_id,
        append=append,
    )
    assert plan.native_id_hint == native_id
    assert plan.acquisition_native_id_hint is None
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="derived generation unavailable",
            derived_only=True,
        )
    )
    try:
        result = ingest_append_with_owner(owner, [plan])
    finally:
        clear_degraded()

    assert result.succeeded == [plan]
    assert result.failed == []
    assert result.deferred == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        append_row = conn.execute(
            """
            SELECT logical_source_key, revision_kind, predecessor_raw_id,
                   baseline_raw_id, append_start_offset, append_end_offset,
                   revision_authority, native_id
            FROM raw_sessions
            WHERE source_index = -1
            """
        ).fetchone()
        artifact_count = conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone()
    assert append_row is not None
    assert append_row[:2] == (f"claude-code-session:{native_id}", "append")
    assert append_row[2] is not None
    assert append_row[3] is not None
    assert append_row[4:] == (
        plan.start_offset,
        plan.last_complete_newline,
        "byte_proven",
        None,
    )
    assert artifact_count == (0,)


def test_source_only_quarantined_append_is_deferred(tmp_path: Path) -> None:
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    path = tmp_path / "quarantined-source-only.jsonl"
    payload = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
        b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
    )
    path.write_bytes(payload)
    plan = replace(
        _append_plan(path, payload, payload_hash=sha256(payload).hexdigest()),
        native_id_hint="quarantined-source-only",
        acquisition_native_id_hint="quarantined-source-only",
    )
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="derived generation unavailable",
            derived_only=True,
        )
    )
    try:
        result = ingest_append_with_owner(_append_owner(tmp_path), [plan])
    finally:
        clear_degraded()

    assert result.succeeded == []
    assert result.failed == []
    assert result.deferred == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT logical_source_key, revision_kind, revision_authority FROM raw_sessions"
        ).fetchone() == ("codex-session:quarantined-source-only", "append", "quarantined")


def test_claude_append_retry_preserves_legacy_null_acquisition_identity(tmp_path: Path) -> None:
    native_id = "claude-legacy-append"
    append = (
        f'{{"parentUuid":"message-0","type":"assistant","message":{{"role":"assistant",'
        f'"content":[{{"type":"text","text":"one"}}]}},"uuid":"message-1",'
        f'"timestamp":"2026-06-02T00:00:01Z","sessionId":"{native_id}"}}\n'
    ).encode()
    _path, plan, owner, _processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id=native_id,
        append=append,
    )
    assert plan.native_id_hint == native_id
    assert plan.acquisition_native_id_hint is None

    # Claude append rows carry no acquisition identity (native_id NULL), and
    # every plan carries its logical session: acquisition refuses a plan
    # without one. A retry must keep the NULL-identity raw it acquired.
    first = ingest_append_with_owner(owner, [plan])
    assert first.succeeded == [plan]
    assert first.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        before_retry = conn.execute("SELECT raw_id, native_id FROM raw_sessions WHERE source_index = -1").fetchall()
    assert len(before_retry) == 1
    assert before_retry[0][1] is None

    retry = ingest_append_with_owner(owner, [plan])

    assert retry.succeeded == [plan]
    assert retry.failed == []
    assert retry.deferred == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        after_retry = conn.execute(
            "SELECT raw_id, native_id, revision_authority FROM raw_sessions WHERE source_index = -1"
        ).fetchall()
    assert after_retry == [(before_retry[0][0], None, "byte_proven")]


def test_derived_only_live_append_candidate_uses_source_acquisition(tmp_path: Path) -> None:
    """The managed batch route must not plan an index-backed append while derived-only."""

    import hashlib

    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    path, _plan, _owner, processor = _seed_live_append_plan(tmp_path, native_id="degraded-managed-append")
    index_db = tmp_path / "index.db"
    index_digest_before = hashlib.sha256(index_db.read_bytes()).hexdigest()
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_count_before = int(
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE source_path = ?", (str(path),)).fetchone()[0]
        )
    pointer = tmp_path / ".index-active-pointer"
    pointer.write_bytes(b"\xff")
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="derived generation unavailable",
            derived_only=True,
        )
    )
    try:
        metrics = run_ingest_files(processor, [path], emit_event=False)
    finally:
        clear_degraded()

    assert metrics.succeeded_file_count == 1
    assert metrics.append_file_count == 0
    assert metrics.full_file_count == 1
    assert pointer.read_bytes() == b"\xff"
    assert hashlib.sha256(index_db.read_bytes()).hexdigest() == index_digest_before
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """SELECT parsed_at_ms, parse_error FROM raw_sessions
               WHERE source_path = ? ORDER BY acquired_at_ms DESC, raw_id DESC""",
            (str(path),),
        ).fetchall()
    assert len(rows) == raw_count_before + 1
    assert rows[0] == (None, None)


def test_live_full_replay_streams_retained_jsonl_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full replay must stream its older retained JSONL snapshot."""
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "streamed-full.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"streamed-full"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    assert _full_paths_sync(processor, [path], source_name="codex").succeeded == [path]
    with path.open("ab") as handle:
        handle.write(
            b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
            b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
        )
    monkeypatch.setattr(
        ArchiveBlobPublisher,
        "read_all",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full replay eagerly read raw blob")),
    )

    result = _full_paths_sync(processor, [path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as source_conn, sqlite3.connect(index_db) as index_conn:
        mtimes = [
            row[0]
            for row in source_conn.execute(
                "SELECT file_mtime_ms FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms",
                (str(path),),
            ).fetchall()
        ]
        timestamp_row = index_conn.execute(
            "SELECT created_at_ms, updated_at_ms FROM sessions WHERE native_id = 'streamed-full'"
        ).fetchone()
    assert len(mtimes) >= 2
    assert all(mtime is not None for mtime in mtimes)
    assert timestamp_row in {(mtime, mtime) for mtime in mtimes}


def test_full_ingest_acquires_but_does_not_parse_when_derived_tier_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-gbs02: a derived-only degraded reason must still acquire raw content.

    Raw acquisition only ever writes source.db; when the daemon is degraded
    ONLY because index.db/embeddings.db are behind the running code
    (``DegradedReason.derived_only=True``), acquisition must proceed --
    otherwise the daemon loses live capture data for the entire duration of
    a schema-migration/reindex window. Materialization (parse) must still be
    skipped: the raw row lands with ``parsed_at_ms IS NULL``, exactly the
    same "not yet materialized" state ordinary convergence already knows
    how to pick up once the derived tier catches up.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "degraded-full.jsonl"
    # Claude Code-shaped payloads: the watch source binds Claude Code, and a
    # bound location refuses another origin's content even on this route.
    claude_record = (
        b'{"type":"user","uuid":"u0","sessionId":"degraded-full","timestamp":"2025-06-13T17:40:00.000Z",'
        b'"cwd":"/w","message":{"role":"user","content":"zero"}}'
    )
    path.write_bytes(claude_record + b"\n")
    json_path = root / "degraded-full.json"
    json_path.write_bytes(b"[" + claude_record + b"]")
    classified_path = root / "subagents" / "worker" / "agent-degraded.meta.json"
    classified_path.parent.mkdir(parents=True)
    classified_path.write_bytes(b'{"agentType":"worker"}')
    # Source-only acquisition writes source.db and refuses outright when the
    # durable tier is absent ("source-only acquisition refused because the
    # durable source tier is missing"), so this case has to stand up a real
    # archive rather than only naming an index path.
    bootstrap_archive_root(tmp_path)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="index.db:46!=57",
            derived_only=True,
        )
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._parse_path_as_session_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not decode source-only evidence")),
    )
    try:
        result = _full_paths_sync(processor, [path, json_path, classified_path], source_name="claude-code")
    finally:
        clear_degraded()

    assert result.succeeded == [path, json_path, classified_path]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_states = conn.execute("SELECT parsed_at_ms, parse_error FROM raw_sessions ORDER BY source_path").fetchall()
        artifact_rows = conn.execute(
            "SELECT COUNT(*) FROM raw_artifacts WHERE source_path = ?", (str(classified_path),)
        ).fetchone()
    assert raw_states == [(None, None), (None, None), (None, None)]
    assert artifact_rows == (0,)


def test_source_only_full_ingest_refuses_missing_durable_source_tier(tmp_path: Path) -> None:
    """An established archive cannot silently bootstrap over source.db loss."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    (tmp_path / "source.db").unlink()
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "pending.jsonl"
    path.write_text('{"opaque":"must remain pending"}\n', encoding="utf-8")
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        result = _full_paths_sync(processor, [path], source_name="claude-code")
    finally:
        clear_degraded()

    assert result.succeeded == []
    assert result.failed == [path]
    assert result.source_payload_read_bytes == 0
    assert not (tmp_path / "source.db").exists()


def test_source_only_antigravity_metadata_stays_pending_with_mutable_companion(tmp_path: Path) -> None:
    """Cursor authority cannot cover metadata while omitting its sibling bytes."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "antigravity"
    metadata = root / "brain" / "work-session" / "plan.md.metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"summary":"plan"}', encoding="utf-8")
    companion = metadata.with_name("plan.md")
    companion.write_text("contemporaneous body", encoding="utf-8")
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="antigravity", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        metrics = run_ingest_files(processor, [metadata], emit_event=False)
    finally:
        clear_degraded()

    assert metrics.succeeded_file_count == 0
    assert metrics.failed_paths == [str(metadata)]
    cursor_record = cursor.get_record(metadata)
    assert cursor_record is not None
    assert cursor_record.excluded is False
    assert cursor_record.failure_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


def test_unreadable_state_database_stays_retryable_instead_of_excluded(tmp_path: Path) -> None:
    """A read fault on a declared database defers the file; it is never excluded.

    The Codex state recognizer cannot open the file and reads it as "not
    Codex state". Anti-vacuity: without the readability probe in
    ``classify_pre_acquisition`` the batch records a cursor exclusion, which
    no later pass revisits while the file's observation is unchanged.
    """
    bootstrap_archive_root(tmp_path)
    root = tmp_path / "codex"
    root.mkdir()
    state = root / "state_5.sqlite"
    with sqlite3.connect(state) as conn:
        conn.execute("CREATE TABLE threads(id TEXT)")
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root, layout=export_drop_layout((".sqlite",))),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    state.chmod(0)
    try:
        result = _full_paths_sync(processor, [state], source_name="codex")
        metrics = run_ingest_files(processor, [state], emit_event=False)
    finally:
        state.chmod(0o600)

    assert result.failed == []
    assert result.source_read_deferred == [state]
    assert result.succeeded == []
    assert metrics.failed_file_count == 0
    assert metrics.deferred_file_count == 1
    assert metrics.deferred_paths == (str(state),)
    assert metrics.failed_paths == [str(state)]
    record = cursor.get_record(state)
    assert record is None or (record.excluded is False and record.failure_count == 0)


@pytest.mark.parametrize("fault_stage", ["classification", "capture", "wrapped_capture"])
def test_repeated_sqlite_read_faults_defer_without_quarantining_unchanged_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_stage: str
) -> None:
    """Six native BUSY reads must not spend the cursor's five-failure budget.

    The real producer, batch caller and writer owners run on every pass. Returning
    these reads as generic failures would exclude the unchanged database before
    the successful read, leaving its raw bytes permanently unacquired.
    """
    bootstrap_archive_root(tmp_path)
    root = tmp_path / "codex"
    state = root / "state_5.sqlite"
    _write_codex_thread_state_db(state)
    cursor = CursorStore(tmp_path / "index.db")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex-state", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    target = (
        "polylogue.sources.sqlite_inspection.classify_sqlite_source"
        if fault_stage == "classification"
        else "polylogue.sources.sqlite_snapshot.snapshot_sqlite_to_blob"
    )
    import importlib

    module_name, function_name = target.rsplit(".", 1)
    original = getattr(importlib.import_module(module_name), function_name)
    reads = 0

    def transient_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        if reads <= 6:
            fault = sqlite3.OperationalError("synthetic source read busy")
            fault.sqlite_errorcode = sqlite3.SQLITE_BUSY
            if fault_stage == "wrapped_capture":
                raise OSError("synthetic snapshot adapter") from fault
            raise fault
        return original(*args, **kwargs)

    monkeypatch.setattr(target, transient_read)
    metrics = [run_ingest_files(processor, [state], emit_event=False) for _ in range(6)]
    record = cursor.get_record(state)
    assert record is not None
    assert record.failure_count == 0
    assert record.excluded is False
    assert record.next_retry_at is not None
    assert reads == 6
    assert all(item.failed_file_count == 0 and item.deferred_file_count == 1 for item in metrics)
    assert all(item.deferred_paths == (str(state),) and item.failed_paths == [str(state)] for item in metrics)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)

    with sqlite3.connect(cursor._ops_db_path) as conn:
        assert conn.execute(
            "SELECT target_id, status FROM convergence_debt WHERE stage = 'live_ingest_source_read'"
        ).fetchall() == [(str(state), "deferred")]

    recovered = run_ingest_files(processor, [state], emit_event=False)
    assert recovered.failed_file_count == 0
    assert recovered.deferred_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT source_path FROM raw_sessions").fetchall() == [(str(state),)]
    run_ingest_files(processor, [state], emit_event=False)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT source_path FROM raw_sessions").fetchall() == [(str(state),)]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert read_thread_titles(conn) == {"codex-thread": "Recover retained state"}
    record = cursor.get_record(state)
    assert record is not None and record.failure_count == 0 and record.excluded is False
    assert record.next_retry_at is None
    with sqlite3.connect(cursor._ops_db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM convergence_debt WHERE stage = 'live_ingest_source_read'"
        ).fetchone() == (0,)


def test_pre_acquisition_reports_a_retryable_read_as_its_typed_fault(tmp_path: Path) -> None:
    """Callers see one typed retryable fault, never a raw SQLite or OS error.

    Anti-vacuity: letting the probe's ``sqlite3.OperationalError`` escape
    fails ``pytest.raises(RetryableSourceReadError)``; raising for bytes that
    are not a database would fail the exclusion assertion.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.live.batch_support import RetryableSourceReadError, classify_pre_acquisition

    root = tmp_path / "codex"
    root.mkdir()
    state = root / "state_5.sqlite"
    with sqlite3.connect(state) as conn:
        conn.execute("CREATE TABLE threads(id TEXT)")
    state.chmod(0)
    try:
        with pytest.raises(RetryableSourceReadError) as fault:
            classify_pre_acquisition(
                state, fallback_provider=Provider.CODEX, source_only=False, size_bytes=state.stat().st_size
            )
    finally:
        state.chmod(0o600)
    assert fault.value.path == state
    assert isinstance(fault.value.cause, (OSError, sqlite3.Error))

    not_a_database = root / "state_6.sqlite"
    not_a_database.write_bytes(b"not a database at all" * 64)
    decision = classify_pre_acquisition(
        not_a_database,
        fallback_provider=Provider.CODEX,
        source_only=False,
        size_bytes=not_a_database.stat().st_size,
    )
    assert decision.excluded_reason is not None


def test_pre_writer_sqlite_reads_defer_then_exclude_proven_unsupported_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admission thread owes unread input until a read can classify it."""
    bootstrap_archive_root(tmp_path)
    root = tmp_path / "inbox"
    root.mkdir()
    path = root / "candidate.sqlite"
    path.write_bytes(b"synthetic unsupported database bytes" * 64)
    cursor = CursorStore(tmp_path / "index.db")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".sqlite",))),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    def busy(*_args: Any, **_kwargs: Any) -> Any:
        error = sqlite3.OperationalError("synthetic source inspection busy")
        error.sqlite_errorcode = sqlite3.SQLITE_BUSY
        raise error

    with monkeypatch.context() as fault:
        fault.setattr("polylogue.sources.sqlite_inspection.classify_sqlite_source", busy)
        metrics = [run_ingest_files(processor, [path], emit_event=False) for _ in range(6)]
    assert all(item.failed_file_count == 0 and item.deferred_file_count == 1 for item in metrics)
    record = cursor.get_record(path)
    assert record is not None and record.failure_count == 0 and record.excluded is False
    settled = run_ingest_files(processor, [path], emit_event=False)
    assert settled.failed_file_count == 0 and settled.deferred_file_count == 0
    record = cursor.get_record(path)
    assert record is not None and record.excluded is True
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


@pytest.mark.asyncio
async def test_dispatcher_keeps_unreadable_source_owed_without_cursor_authority(tmp_path: Path) -> None:
    """Acknowledging a no-cursor deferral must not advance past unread input."""
    from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
    from tests.infra.raw_owner_routes import live_owner_set

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "codex"
    state = root / "state_5.sqlite"
    _write_codex_thread_state_db(state)
    source = WatchSource(name="codex-state", root=root)
    cursor = CursorStore(tmp_path / "index.db")
    now = [0.0]
    async with live_owner_set(tmp_path) as owners:
        watcher = LiveWatcher(
            cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
            (source,),
            cursor=cursor,
            **owners.watcher_kwargs(),
        )
        adapter = FileIntakeAdapter(
            DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),
            source,
            clock=lambda: now[0],
        )
        dispatcher = FairIntakeDispatcher(
            (IntakeClassSpec(name=source.name, adapter=adapter, page_size=1),), clock=lambda: now[0]
        )
        state.chmod(0)
        try:
            deferred = 0
            for _ in range(14):
                result = await dispatcher.run_once()
                deferred += result.classes[0].deferred
                assert result.classes[0].isolated == 0
                assert result.classes[0].retried == 0
                now[0] += 10.0
            assert deferred >= 6
            assert cursor.get_record(state) is None
        finally:
            state.chmod(0o600)
        try:
            handled = 0
            for _ in range(6):
                result = await dispatcher.run_once()
                handled += result.classes[0].admitted + result.classes[0].excluded
                now[0] += 10.0
            assert handled == 1
            with sqlite3.connect(tmp_path / "source.db") as conn:
                assert conn.execute("SELECT source_path FROM raw_sessions").fetchall() == [(str(state),)]
            record = cursor.get_record(state)
            assert record is not None and record.failure_count == 0 and record.excluded is False
            assert record.next_retry_at is None
            with sqlite3.connect(cursor._ops_db_path) as conn:
                assert conn.execute(
                    "SELECT COUNT(*) FROM convergence_debt WHERE stage = 'live_ingest_source_read'"
                ).fetchone() == (0,)
        finally:
            watcher.stop()


@pytest.mark.parametrize("state_name", ["state_5.sqlite", "candidate.sqlite"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_sqlite_admission_cancellation_propagates_without_a_failure_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state_name: str, cleanup_failure: bool
) -> None:
    """Both pre-writer admission routes preserve the owner's cancellation."""
    from polylogue.core.compute import DaemonOperationCancelled

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "input"
    path = root / state_name
    _write_plain_sqlite_db(path)
    cursor = CursorStore(tmp_path / "index.db")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex-state" if state_name == "state_5.sqlite" else "inbox", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    def cancel(*_args: Any, **_kwargs: Any) -> Any:
        if cleanup_failure:
            raise BaseExceptionGroup(
                "synthetic cancellation and cleanup failure",
                [DaemonOperationCancelled("synthetic owner cancellation"), RuntimeError("synthetic capture cleanup")],
            )
        raise DaemonOperationCancelled("synthetic owner cancellation")

    monkeypatch.setattr("polylogue.sources.sqlite_inspection.classify_sqlite_source", cancel)
    with pytest.raises(BaseExceptionGroup if cleanup_failure else DaemonOperationCancelled):
        run_ingest_files(processor, [path], emit_event=False)
    assert cursor.get_record(path) is None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_antigravity_cohort_cancellation_keeps_sources_out_of_failure_cursors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_failure: bool
) -> None:
    """The cohort catch must not turn owner cancellation into missing exports."""
    from polylogue.core.compute import DaemonOperationCancelled

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "antigravity"
    path = root / "conversations" / "synthetic-cascade.pb"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic protobuf input")
    cursor = CursorStore(tmp_path / "index.db")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="antigravity", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    cancellation = DaemonOperationCancelled("synthetic owner cancellation")
    cleanup = RuntimeError("synthetic capture cleanup")
    failure = (
        BaseExceptionGroup("synthetic cancellation and cleanup failure", [cancellation, cleanup])
        if cleanup_failure
        else cancellation
    )

    def cancel(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr("polylogue.sources.source_parsing.iter_antigravity_language_server_sessions", cancel)
    with pytest.raises(BaseExceptionGroup if cleanup_failure else DaemonOperationCancelled) as caught:
        run_ingest_files(processor, [path], emit_event=False)
    assert caught.value is failure
    if isinstance(caught.value, BaseExceptionGroup):
        assert caught.value.exceptions == (cancellation, cleanup)
    assert cursor.get_record(path) is None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_dispatcher_propagates_source_cancellation_with_its_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_failure: bool
) -> None:
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
    from tests.infra.raw_owner_routes import live_owner_set

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "codex"
    state = root / "state_5.sqlite"
    _write_codex_thread_state_db(state)
    source = WatchSource(name="codex-state", root=root)
    cursor = CursorStore(tmp_path / "index.db")
    cancellation = DaemonOperationCancelled("synthetic owner cancellation")
    cleanup = RuntimeError("synthetic capture cleanup")
    failure = (
        BaseExceptionGroup("synthetic cancellation and cleanup failure", [cancellation, cleanup])
        if cleanup_failure
        else cancellation
    )

    def cancel(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr("polylogue.sources.sqlite_inspection.classify_sqlite_source", cancel)
    async with live_owner_set(tmp_path) as owners:
        watcher = LiveWatcher(
            cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
            (source,),
            cursor=cursor,
            **owners.watcher_kwargs(),
        )
        adapter = FileIntakeAdapter(DaemonIntakeContext(tmp_path, watcher, (source,)), source)
        dispatcher = FairIntakeDispatcher((IntakeClassSpec(name=source.name, adapter=adapter),))
        try:
            with pytest.raises(BaseExceptionGroup if cleanup_failure else DaemonOperationCancelled) as caught:
                await dispatcher.run_once()
            assert caught.value is failure
            if isinstance(caught.value, BaseExceptionGroup):
                assert caught.value.exceptions == (cancellation, cleanup)
            assert cursor.get_record(state) is None
            with sqlite3.connect(tmp_path / "source.db") as conn:
                assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
        finally:
            watcher.stop()


def test_jsonl_pre_acquisition_classifies_record_by_record_in_bounded_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large rollout is classified from its records, never as one JSON document.

    Routing a ``.jsonl`` stream through the whole-input projection (the
    pure-Python event parser, once per detector binding) makes the guard
    below raise; holding the stream rather than one record at a time makes
    the traced peak follow the input size.
    """
    import tracemalloc

    from polylogue.sources import detection_projection
    from polylogue.sources.live.batch_support import classify_pre_acquisition

    def session_lines(count: int) -> Iterable[str]:
        yield json.dumps({"type": "session_meta", "payload": {"id": "bounded", "timestamp": "2026-06-02T00:00:00Z"}})
        for index in range(count):
            yield json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": f"message-{index}",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "synthetic " * 1600}],
                    },
                }
            )

    warm = tmp_path / "warm.jsonl"
    warm.write_text("\n".join(session_lines(1)) + "\n", encoding="utf-8")
    rollout = tmp_path / "rollout-bounded.jsonl"
    with rollout.open("w", encoding="utf-8") as handle:
        for line in session_lines(2600):
            handle.write(line + "\n")
    size = rollout.stat().st_size

    def whole_document(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a JSONL record stream was projected as one JSON document")

    monkeypatch.setattr(detection_projection, "project_detection_input", whole_document)
    monkeypatch.setattr(detection_projection, "_document_projection", whole_document)

    # Lazy imports and compiled registries are not per-input memory.
    classify_pre_acquisition(warm, fallback_provider=Provider.CODEX, source_only=False, size_bytes=0)
    tracemalloc.start()
    try:
        decision = classify_pre_acquisition(
            rollout, fallback_provider=Provider.CODEX, source_only=False, size_bytes=size
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert decision.excluded_reason is None
    assert decision.detected_provider is Provider.CODEX
    assert decision.detection_crash is None
    assert size > 40 * 1024 * 1024
    assert peak < size // 4


def test_source_only_full_ingest_streams_admitted_zip_members_without_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production full-ingest ZIP route must retain bytes before decode."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "degraded.zip"
    member_names = ("sessions/one.jsonl", "sessions/two.json")
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr(member_names[0], b'{"opaque":"first"}\n')
        zf.writestr(member_names[1], b'{"opaque":"second"}')
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    for target in (
        "polylogue.sources.source_acquisition_components.sniff_zip_provider",
        "polylogue.sources.source_acquisition_components.iter_entry_payloads",
        "polylogue.sources.source_acquisition_components.classify_artifact",
    ):
        monkeypatch.setattr(
            target, lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not decode ZIP"))
        )
    try:
        result = _full_paths_sync(processor, [bundle], source_name="claude-code")
    finally:
        clear_degraded()

    assert result.succeeded == [bundle]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            "SELECT source_path, source_index, parsed_at_ms, parse_error FROM raw_sessions ORDER BY source_index"
        ).fetchall()
    assert rows == [
        (f"{bundle}:{member_names[0]}", 0, None, None),
        (f"{bundle}:{member_names[1]}", 1, None, None),
    ]


def test_source_only_full_ingest_bounds_oversized_ndjson_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production NDJSON route reaches streaming retention before eager decode."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "inbox"
    root.mkdir()
    source = root / "oversized.ndjson"
    # A session, not a bare header: pre-writer admission classifies the
    # record stream and excludes a known-provider non-session sidecar.
    payload = (
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "oversized-record", "padding": "x" * (_RETIRED_FULL_INGEST_SIZE_BOUND + 1024)},
            }
        ).encode()
        + b"\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "oversized-message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            }
        ).encode()
        + b"\n"
    )
    source.write_bytes(payload)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".ndjson",))),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support.json_loads",
        lambda _raw: (_ for _ in ()).throw(AssertionError("sampling must not decode an oversized physical record")),
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        result = _full_paths_sync(processor, [source], source_name="inbox")
    finally:
        clear_degraded()

    assert result.succeeded == [source]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT origin, blob_size, parsed_at_ms, parse_error FROM raw_sessions").fetchall() == [
            ("unknown-export", len(payload), None, None)
        ]


def test_source_only_zip_read_failure_remains_retryable_after_partial_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real source-only route must not exclude a transiently unreadable ZIP."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "retry.zip"
    member_names = ("sessions/one.jsonl", "sessions/two.jsonl")
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr(member_names[0], b'{"opaque":"first"}\n')
        zf.writestr(member_names[1], b'{"opaque":"second"}\n')
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    original_stream = stream_preserved_zip_entry_raw_data
    calls = 0

    def fail_after_first_copy(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("transient ZIP read failure")
        return original_stream(*args, **kwargs)

    monkeypatch.setattr(
        "polylogue.sources.live.batch.stream_preserved_zip_entry_raw_data",
        fail_after_first_copy,
    )
    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        failed = run_ingest_files(processor, [bundle], emit_event=False)

        assert failed.succeeded_file_count == 0
        assert failed.failed_file_count == 1
        failed_cursor = cursor.get_record(bundle)
        assert failed_cursor is not None
        assert failed_cursor.failure_count == 1
        assert failed_cursor.excluded is False
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)

        retried = run_ingest_files(processor, [bundle], emit_event=False)
    finally:
        clear_degraded()

    assert retried.succeeded_file_count == 1
    assert retried.failed_file_count == 0
    recovered_cursor = cursor.get_record(bundle)
    assert recovered_cursor is not None
    assert recovered_cursor.failure_count == 0
    assert recovered_cursor.excluded is False
    with sqlite3.connect(tmp_path / "source.db") as conn:
        retained = conn.execute("SELECT source_path, source_index FROM raw_sessions ORDER BY source_index").fetchall()
    assert retained == [
        (f"{bundle}:{member_names[0]}", 0),
        (f"{bundle}:{member_names[1]}", 1),
    ]


def test_source_only_zip_replay_resolves_unknown_chatgpt_member_and_keeps_duplicate_coordinates(
    tmp_path: Path,
) -> None:
    """Recovery, not acquisition, resolves UNKNOWN ZIP bytes and replays each coordinate.

    Source-only acquisition sees opaque ZIP members and records no more than
    ``unknown-export``. Retained replay resolves each member's provider and
    publishes the raw under its resolved origin, and a live re-observation
    keeps that origin: the raw ids and container coordinates are identical
    across every route.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "inbox"
    root.mkdir()
    bundle = root / "export.zip"
    payload = json.dumps(
        [
            {
                "id": "zip-chatgpt",
                "conversation_id": "zip-chatgpt",
                "title": "ZIP recovery",
                "create_time": 1_700_000_000,
                "update_time": 1_700_000_001,
                "current_node": "assistant-node",
                "mapping": {
                    "user-node": {
                        "id": "user-node",
                        "parent": None,
                        "children": ["assistant-node"],
                        "message": {
                            "id": "user-message",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["recover ZIP"]},
                            "create_time": 1_700_000_000,
                        },
                    },
                    "assistant-node": {
                        "id": "assistant-node",
                        "parent": "user-node",
                        "children": [],
                        "message": {
                            "id": "assistant-message",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["replayed"]},
                            "create_time": 1_700_000_001,
                        },
                    },
                },
            }
        ],
        sort_keys=True,
    ).encode()
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("first/conversations.json", payload)
        zf.writestr("second/conversations.json", payload)
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        result = _full_paths_sync(processor, [bundle], source_name="unknown")
    finally:
        clear_degraded()

    assert result.succeeded == [bundle]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        before_replay = conn.execute(
            "SELECT raw_id, hex(blob_hash), source_path, source_index, origin FROM raw_sessions ORDER BY source_index"
        ).fetchall()
    assert len(before_replay) == 2
    assert len({row[0] for row in before_replay}) == 2
    assert len({row[1] for row in before_replay}) == 1
    assert [row[2:] for row in before_replay] == [
        (f"{bundle}:first/conversations.json", 0, "unknown-export"),
        (f"{bundle}:second/conversations.json", 1, "unknown-export"),
    ]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT raw_id, coordinate_format, entry_ordinal, split_index "
            "FROM raw_container_coordinates ORDER BY entry_ordinal"
        ).fetchall() == [
            (before_replay[0][0], "zip-v2", 0, 0),
            (before_replay[1][0], "zip-v2", 1, 0),
        ]

    replay = replay_retained_components(tmp_path)

    # Both acquired coordinates carry identical bytes and therefore replay as
    # one logical source while the source-tier coordinate rows remain distinct.
    assert replay.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id, message_count FROM sessions").fetchall() == [("zip-chatgpt", 2)]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT origin, detected_provider FROM raw_sessions ORDER BY source_index").fetchall() == [
            ("chatgpt-export", "chatgpt"),
            ("chatgpt-export", "chatgpt"),
        ]
        assert conn.execute(
            "SELECT raw_id, coordinate_format, entry_ordinal, split_index "
            "FROM raw_container_coordinates ORDER BY entry_ordinal"
        ).fetchall() == [
            (before_replay[0][0], "zip-v2", 0, 0),
            (before_replay[1][0], "zip-v2", 1, 0),
        ]

    reobserved = _full_paths_sync(processor, [bundle], source_name="unknown")

    assert reobserved.succeeded == [bundle]
    assert reobserved.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT origin, detected_provider FROM raw_sessions ORDER BY source_index").fetchall() == [
            ("chatgpt-export", "chatgpt"),
            ("chatgpt-export", "chatgpt"),
        ]
        assert conn.execute(
            "SELECT raw_id, coordinate_format, entry_ordinal, split_index "
            "FROM raw_container_coordinates ORDER BY entry_ordinal"
        ).fetchall() == [
            (before_replay[0][0], "zip-v2", 0, 0),
            (before_replay[1][0], "zip-v2", 1, 0),
        ]


def test_zip_duplicate_member_coordinates_stay_distinct_on_the_member_route(tmp_path: Path) -> None:
    """Central-directory ordinal and within-member split remain independent."""
    root = tmp_path / "inbox"
    root.mkdir()
    bundle = root / "duplicates.zip"
    member_name = "sessions/duplicate.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"duplicate-coordinate"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"retained twice"}]}}\n'
    )
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("ignored/readme.txt", b"not admitted")
        zf.writestr(member_name, payload)
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr(member_name, payload)

    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    with live_zip_capture(tmp_path) as (publisher, zip_inputs):
        source_only_result = processor._extract_source_only_zip_member_records(
            bundle,
            blob_store=publisher,
            zip_inputs=zip_inputs,
            fallback_provider=Provider.CODEX,
            file_mtime="2026-08-13T00:00:00+00:00",
        )

    # The source-only producer is the single ZIP member route. Both physical
    # duplicate members are retained under distinct identities, each with the
    # same container and member name and its own central-directory ordinal.
    assert source_only_result is not None
    source_only_records, _source_only_bytes = source_only_result
    source_only_ids = [raw_id for raw_id, _record in source_only_records]
    assert len(source_only_ids) == 2
    assert len(set(source_only_ids)) == 2
    assert len({record.blob_hash for _raw_id, record in source_only_records}) == 1
    coordinates = [record.captured_zip_coordinate for _raw_id, record in source_only_records]
    assert all(coordinate is not None for coordinate in coordinates)
    assert len({coordinate.canonical_container for coordinate in coordinates if coordinate is not None}) == 1
    assert [coordinate.member_name for coordinate in coordinates if coordinate is not None] == [
        member_name,
        member_name,
    ]
    assert [record.source_index for _raw_id, record in source_only_records] == [1, 3]


@pytest.mark.parametrize("state_name", ["state_5.sqlite", "goals_1.sqlite", "memories_1.sqlite"])
def test_source_only_full_ingest_snapshots_declared_codex_state_without_shape_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_name: str,
) -> None:
    """A degraded source tier retains each declared future-shaped Codex state DB."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "codex"
    root.mkdir()
    state_db = root / state_name
    _write_plain_sqlite_db(state_db)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex-state", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    monkeypatch.setattr(
        "polylogue.sources.parsers.codex_state.is_in_scope_codex_sqlite_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not inspect source-only state schema")),
    )
    try:
        result = _full_paths_sync(processor, [state_db], source_name="codex-state")
    finally:
        clear_degraded()

    assert result.succeeded == [state_db]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT source_path, parsed_at_ms FROM raw_sessions").fetchall() == [(str(state_db), None)]


def test_source_only_foreign_sqlite_name_cannot_claim_codex_authority(tmp_path: Path) -> None:
    """A foreign watch source cannot turn a filename into Codex authority."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "inbox"
    state_db = root / "state_5.sqlite"
    _write_plain_sqlite_db(state_db)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".sqlite",))),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        result = _full_paths_sync(processor, [state_db], source_name="inbox")
    finally:
        clear_degraded()

    assert result.succeeded == []
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT origin FROM raw_sessions").fetchall() == []


def _write_codex_thread_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                cwd TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                source TEXT NOT NULL,
                model TEXT,
                agent_nickname TEXT,
                agent_role TEXT,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT NOT NULL,
                child_thread_id TEXT NOT NULL PRIMARY KEY,
                status TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("codex-thread", "Recover retained state", "/work", 1, 2, "cli", "gpt-5", None, None, 0),
        )
        conn.execute(
            "INSERT INTO thread_spawn_edges VALUES (?, ?, ?)",
            ("codex-thread", "codex-child", "closed"),
        )


def test_source_only_codex_state_recovery_replays_retained_thread_evidence(tmp_path: Path) -> None:
    """Source-only Codex state acquisition replays as retained thread evidence."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "codex"
    state_db = root / "state_5.sqlite"
    _write_codex_thread_state_db(state_db)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        assert _full_paths_sync(processor, [state_db], source_name="codex").succeeded == [state_db]
    finally:
        clear_degraded()

    replay = replay_retained_components(tmp_path)

    assert replay.scanned == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parsed_at_ms IS NOT NULL FROM raw_sessions").fetchone() == (1,)
        # The retained material is a logical export, and thread evidence is
        # derived from it -- no durable per-row hook material is minted.
        assert conn.execute("SELECT count(*) FROM raw_hook_events").fetchone() == (0,)
        assert conn.execute("SELECT count(*) FROM blob_refs WHERE ref_type = 'hook_payload'").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert read_thread_titles(conn) == {"codex-thread": "Recover retained state"}
        assert read_spawn_edges(conn) == {("codex-thread", "codex-child"): "closed"}


@pytest.mark.parametrize(
    ("state_name", "declared_table"),
    [("state.db", "schema_version"), ("verification_evidence.db", "meta")],
)
def test_source_only_hermes_named_sqlite_uses_consistent_backup_before_generic_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_name: str,
    declared_table: str,
) -> None:
    """A direct file copy loses an uncheckpointed WAL row; the export retains it.

    The row lives in one of the member's declared ``logical_tables``, because
    that declared product is exactly what acquisition retains.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "hermes"
    state_db = root / state_name
    state_db.parent.mkdir(parents=True)
    writer = sqlite3.connect(state_db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute(f"CREATE TABLE {declared_table} (value TEXT NOT NULL)")
    writer.commit()
    writer.execute(f"INSERT INTO {declared_table} VALUES ('must survive')")
    writer.commit()
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="hermes", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr("polylogue.sources.parsers.hermes_state.looks_like_state_db_path", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "polylogue.sources.parsers.hermes_verification.looks_like_verification_evidence_db_path",
        lambda *_a, **_k: False,
    )

    set_degraded(DegradedReason(code="schema_version_mismatch", message="index unavailable", derived_only=True))
    try:
        assert _full_paths_sync(processor, [state_db], source_name="hermes").succeeded == [state_db]
    finally:
        clear_degraded()
        writer.close()

    with sqlite3.connect(tmp_path / "source.db") as conn:
        blob_hash = str(conn.execute("SELECT hex(blob_hash) FROM raw_sessions").fetchone()[0]).lower()
    retained = BlobStore(tmp_path / "blob").blob_path(blob_hash)
    with logical_source_context(retained) as export:
        assert export.execute(f"SELECT value FROM {declared_table}").fetchall() == [("must survive",)]


def test_full_ingest_acquires_when_index_is_genuinely_semantic_distance_stale(
    tmp_path: Path,
) -> None:
    """polylogue-gbs02: acquire-only mode must survive a REAL stale index tier.

    The sibling test above proves the skip logic but leaves index.db at the
    current version, so it never exercises the open: the ordinary
    ``ArchiveStore.open_existing(read_only=False)`` writer hard-refuses an
    index tier at a semantic-reparse distance (the live archive's actual
    pre-rebuild state, index.db 46 vs current code). This test ages the
    index to that distance first — with the source-tier-only open routed via
    ``_open_archive_for_live_write`` the acquire succeeds and the stale
    index file stays byte-identical; without it, the open raises before any
    raw write and this test fails.
    """
    import hashlib
    import sqlite3 as _sqlite3

    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "degraded-stale-index.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"degraded-stale-index"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    # Bootstrap a real archive file set, then age the index tier to the
    # semantic-reparse distance (46 is the live pre-818fy generation).
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False):
        pass
    index_db = tmp_path / "index.db"
    conn = _sqlite3.connect(index_db)
    try:
        conn.execute("PRAGMA user_version = 46")
        conn.commit()
    finally:
        conn.close()
    index_digest_before = hashlib.sha256(index_db.read_bytes()).hexdigest()
    pointer = tmp_path / ".index-active-pointer"
    pointer.write_bytes(b"\xff")

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "cursors.db", ops_db_path=tmp_path / "ops.db"),
        parser_fingerprint="test-parser",
    )
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="index.db:46!=current",
            derived_only=True,
        )
    )
    try:
        metrics = run_ingest_files(processor, [path], emit_event=False)
    finally:
        clear_degraded()

    assert metrics.succeeded_file_count == 1
    assert metrics.failed_file_count == 0
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert parse_error is None
    assert hashlib.sha256(index_db.read_bytes()).hexdigest() == index_digest_before, (
        "the stale index tier must never be opened for write during acquire-only ingest"
    )
    assert pointer.read_bytes() == b"\xff"


def test_live_raw_compaction_holds_generation_lease_through_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protected set and destructive cleanup observe one unpromotable generation."""

    from polylogue.storage import raw_retention
    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="test-parser",
    )
    phases: list[str] = []

    def assert_promotion_excluded(*_args: object, **_kwargs: object) -> SimpleNamespace:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
        phases.append("authority")
        return SimpleNamespace(protected_raw_ids=frozenset(), eligible_raw_ids=frozenset())

    def assert_delete_excluded(*_args: object, **_kwargs: object) -> SimpleNamespace:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
        phases.append("delete")
        return SimpleNamespace(errors=(), residual_source_paths=())

    monkeypatch.setattr(raw_retention, "active_raw_retention_authority", assert_promotion_excluded)
    monkeypatch.setattr(raw_retention, "compact_paths_superseded_raw_snapshots", assert_delete_excluded)

    # Both destructive steps are replaced by lease probes, so no archive write
    # needs admission; inside the writer the probe would contend with the
    # coordinator's own hold instead of observing the compaction lease.
    processor._compact_superseded_raw_snapshots([path])
    assert phases == ["authority", "delete"]
    with RebuildLease(tmp_path):
        pass


def test_full_ingest_empty_jsonl_is_not_misclassified_as_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty session candidate settles as a no-session observation, not a truncation.

    Retained publication parses it, finds no session, and records the
    current-parser non-session census; it is never corrupt input or a retry.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "empty.jsonl"
    path.write_bytes(b"")
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )

    result = _full_paths_sync(processor, [path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []
    assert result.settled_exclusions == {path: REFUSED_NO_SESSIONS}
    parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parse_error is None
    assert parsed_at_ms is not None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
        assert conn.execute("SELECT status FROM raw_membership_census").fetchall() == [("non_session",)]


def test_full_ingest_unknown_export_without_sessions_records_terminal_evidence(tmp_path: Path) -> None:
    root = tmp_path / "chatgpt"
    root.mkdir()
    path = root / "export.jsonl"
    path.write_bytes(b"")
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    # Terminal classification belongs to retained preparation, so the law
    # runs the live pass: the file settles as a terminal exclusion.
    metrics = run_ingest_files(processor, [path], emit_event=False)

    assert metrics.failed_file_count == 0
    assert set(metrics.refused_bytes_by_reason or {}) <= SETTLED_EXCLUSION_REASONS
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_export_no_session", "unsupported_parseable", 0)


def test_full_ingest_unknown_weak_path_ndjson_records_terminal_evidence(tmp_path: Path) -> None:
    """NDJSON takes the same strict terminal classification route as JSONL."""

    root = tmp_path / "chatgpt"
    path = root / "analysis" / "export.ndjson"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root, layout=export_drop_layout((".jsonl", ".ndjson"))),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    # Terminal classification belongs to retained preparation, so the law
    # runs the live pass: the file settles as a terminal exclusion.
    metrics = run_ingest_files(processor, [path], emit_event=False)

    assert metrics.failed_file_count == 0
    assert set(metrics.refused_bytes_by_reason or {}) <= SETTLED_EXCLUSION_REASONS
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_export_no_session", 0)


@pytest.mark.parametrize(
    ("payload", "expected_artifact"),
    [
        (b"{", ("terminal_unknown_json_decode", "decode_failed")),
        (b"", ("terminal_unknown_json_decode", "decode_failed")),
    ],
)
def test_full_ingest_unknown_weak_path_json_retains_terminal_evidence(
    tmp_path: Path,
    payload: bytes,
    expected_artifact: tuple[str, str],
) -> None:
    """Unknown weak-path JSON reaches durable generic terminal handling."""

    root = tmp_path / "unknown"
    path = root / "analysis" / "export.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    path_artifact = classify_artifact_path(path, provider=Provider.UNKNOWN)
    assert path_artifact is not None and not path_artifact.parse_as_session
    assert not has_decoded_session_evidence(path, provider=Provider.UNKNOWN)

    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    # Terminal classification belongs to retained preparation, so the law
    # runs the live pass: the file settles as a terminal exclusion.
    metrics = run_ingest_files(processor, [path], emit_event=False)

    assert metrics.failed_file_count == 0
    assert set(metrics.refused_bytes_by_reason or {}) <= SETTLED_EXCLUSION_REASONS
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw = conn.execute(
            "SELECT raw_id, blob_size, parse_error FROM raw_sessions WHERE source_path = ?", (str(path),)
        ).fetchone()
        artifact = conn.execute(
            """
            SELECT artifact_kind, support_status
            FROM raw_artifacts
            WHERE raw_id = ?
            """,
            (raw[0],) if raw is not None else (None,),
        ).fetchone()
    # The preconditions above would take the weak path-exclusion branch if
    # the production unknown-JSON exemption were removed.
    assert raw is not None
    assert raw[1] == len(payload)
    assert isinstance(raw[2], str)
    assert artifact == expected_artifact


def test_full_ingest_unknown_weak_directory_still_excludes_strong_sidecar(tmp_path: Path) -> None:
    """A weak directory cannot override a definitive non-session filename."""

    root = tmp_path / "unknown"
    path = root / "analysis" / "sessions-index.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"mapping":{"looks":"conversational"}}', encoding="utf-8")
    path_artifact = classify_artifact_path(path, provider=Provider.UNKNOWN)
    assert path_artifact is not None and path_artifact.kind.value == "metadata_document"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "ops.db"))),
        (WatchSource(name="unknown", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="test-parser",
    )

    result = _full_paths_sync(processor, [path], source_name="unknown")

    assert result.succeeded == []
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


def test_full_ingest_unknown_malformed_jsonl_records_terminal_decode_and_stops_retrying(tmp_path: Path) -> None:
    """Complete malformed JSONL lines are terminal decode evidence, not no-session evidence."""
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "malformed.jsonl"
    path.write_bytes(b'{"broken":}\n{"also_broken":}\n')
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    # Terminal classification belongs to retained preparation, so the law
    # runs the live pass twice: the second pass finds nothing to retry.
    first = run_ingest_files(processor, [path], emit_event=False)
    second = run_ingest_files(processor, [path], emit_event=False)

    assert first.failed_file_count == 0
    assert set(first.refused_bytes_by_reason or {}) <= SETTLED_EXCLUSION_REASONS
    assert second.failed_file_count == 0
    record = processor._cursor.get_record(path)
    assert record is not None and record.failure_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed")
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0


def test_full_ingest_unknown_malformed_final_jsonl_record_records_terminal_decode(tmp_path: Path) -> None:
    """A malformed final JSONL record contributes to strict decode evidence."""
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "malformed-final.jsonl"
    path.write_bytes(b'{"only_broken":}\n')
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    # Terminal classification belongs to retained preparation, so the law
    # runs the live pass: the file settles as a terminal exclusion.
    metrics = run_ingest_files(processor, [path], emit_event=False)

    assert metrics.failed_file_count == 0
    assert set(metrics.refused_bytes_by_reason or {}) <= SETTLED_EXCLUSION_REASONS
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed")


def test_full_ingest_unknown_json_decode_records_terminal_decode_evidence(tmp_path: Path) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "export.json"
    path.write_bytes(b"{")
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    # Terminal classification belongs to retained preparation, so the law
    # runs the live pass: the file settles as a terminal exclusion.
    metrics = run_ingest_files(processor, [path], emit_event=False)

    assert metrics.failed_file_count == 0
    assert set(metrics.refused_bytes_by_reason or {}) <= SETTLED_EXCLUSION_REASONS
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed", 0)


def test_full_ingest_unknown_invalid_utf8_records_terminal_decode_evidence(tmp_path: Path) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "export.json"
    path.write_bytes(b"\xff")
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    # Terminal classification belongs to retained preparation, so the law
    # runs the live pass: the file settles as a terminal exclusion.
    metrics = run_ingest_files(processor, [path], emit_event=False)

    assert metrics.failed_file_count == 0
    assert set(metrics.refused_bytes_by_reason or {}) <= SETTLED_EXCLUSION_REASONS
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
    assert artifact == ("terminal_unknown_json_decode", "decode_failed", 0)


def test_full_ingest_unrecognized_unknown_export_settles_as_terminal_refusal(
    tmp_path: Path,
) -> None:
    """An input no provider recognizes is retained and refused once, typed.

    Retained preparation refuses the unrecognized shape before any parser
    runs. The same bytes can only be refused the same way, so the refusal
    settles: a non-session census with typed terminal evidence and the parse
    failure on the raw, never a failed census every pass would census again.
    """
    root = tmp_path / "unknown"
    root.mkdir()
    path = root / "export.json"
    path.write_bytes(b'{"unrelated": "payload"}')
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="unknown", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )

    result = run_ingest_files(processor, [path], emit_event=False)

    assert result.ingested_session_count == 0
    assert result.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact_kinds = {row[0] for row in conn.execute("SELECT artifact_kind FROM raw_artifacts")}
        retained = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
        census = conn.execute("SELECT status, detail FROM raw_membership_census").fetchall()
        (parse_error,) = conn.execute("SELECT parse_error FROM raw_sessions").fetchone()
    assert artifact_kinds == {RawFailureEvidenceKind.TERMINAL_UNKNOWN_EXPORT_NO_SESSION.value}
    assert retained == 1
    assert [status for status, _detail in census] == ["non_session"]
    assert "UnsupportedRetainedJsonShapeError" in census[0][1]
    assert isinstance(parse_error, str) and parse_error.startswith("UnsupportedRetainedJsonShapeError:")


def test_full_ingest_hot_capture_keeps_its_frontier_before_the_unterminated_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture whose file grows after it is never settled as corrupt input.

    The source finishes its first record after the capture. The retained
    capture admits no record and records no failure evidence, so nothing
    forces it terminal; its frontier stays before the unterminated tail.
    """
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "active.jsonl"
    captured = b'{"type":"session_meta"'
    path.write_bytes(captured)
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    captured_boundary_check = live_batch._stable_truncated_tail_admission

    def grow_source_after_capture(record: Any) -> Any:
        path.write_bytes(captured + b"\n")
        return captured_boundary_check(record)

    monkeypatch.setattr(live_batch, "_stable_truncated_tail_admission", grow_source_after_capture)

    result = _full_paths_sync(processor, [path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []
    assert result.raw_frontier_sizes[path] == 0
    assert result.partial_admissions == {}
    _parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parse_error is None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_full_ingest_applies_incomplete_record_guard_to_jsonl_txt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported ``.jsonl.txt`` wire suffix has JSONL tail authority too."""
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "active.jsonl.txt"
    captured = b'{"type":"session_meta"'
    path.write_bytes(captured)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "archive.sqlite"))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "archive.sqlite"),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    boundary_check = live_batch._stable_truncated_tail_admission

    def grow_source_after_capture(record: Any) -> Any:
        path.write_bytes(captured + b"\n")
        return boundary_check(record)

    monkeypatch.setattr(live_batch, "_stable_truncated_tail_admission", grow_source_after_capture)

    result = _full_paths_sync(processor, [path], source_name="codex")

    # Without JSONL tail authority the whole payload would decode as one
    # document and fail as corrupt input; with it the unterminated tail is
    # left out of the parse and the frontier stays before it.
    assert result.succeeded == [path]
    assert result.raw_frontier_sizes[path] == 0
    _parsed_at_ms, parse_error = _raw_parse_state(tmp_path)
    assert parse_error is None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_full_ingest_claude_hot_capture_keeps_its_frontier_before_the_unterminated_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Claude Code route leaves an in-progress record out like every JSONL route."""
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "claude"
    root.mkdir()
    path = root / "active.jsonl"
    captured = b'{"type":"assistant"'
    path.write_bytes(captured)
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    boundary_check = live_batch._stable_truncated_tail_admission

    def grow_source_after_capture(record: Any) -> Any:
        path.write_bytes(captured + b"\n")
        return boundary_check(record)

    monkeypatch.setattr(live_batch, "_stable_truncated_tail_admission", grow_source_after_capture)

    result = _full_paths_sync(processor, [path], source_name="claude-code")

    assert result.succeeded == [path]
    assert result.raw_frontier_sizes[path] == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_streamed_incomplete_jsonl_capture_defers_completed_source_until_authority_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streamed capture taken mid-record is recovered once its source completes.

    The first pass admits no record and keeps the cursor frontier before the
    unterminated record; the next pass reads the completed source in full.
    """
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "streaming-active.jsonl"
    captured = b'{"type":"session_meta","payload":{"id":"streaming-active"}'
    completed = (
        b'{"type":"session_meta","payload":{"id":"streaming-active"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"complete"}]}}\n'
    )
    path.write_bytes(captured)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    boundary_check = live_batch._stable_truncated_tail_admission
    source_completed = False

    def complete_source_after_capture(record: Any) -> Any:
        nonlocal source_completed
        if not source_completed:
            path.write_bytes(completed)
            source_completed = True
        return boundary_check(record)

    monkeypatch.setattr(live_batch, "_stable_truncated_tail_admission", complete_source_after_capture)

    deferred = run_ingest_files(processor, [path])

    assert deferred.full_file_count == 1
    assert deferred.failed_file_count == 0
    assert deferred.ingested_session_count == 0
    early_cursor = cursor.get_record(path)
    assert early_cursor is not None
    assert early_cursor.byte_offset == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)

    retry = run_ingest_files(processor, [path])
    assert retry.full_file_count == 1
    assert retry.succeeded_file_count == 1
    assert retry.failed_file_count == 0

    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == len(completed)
    assert final_cursor.byte_size == len(completed)
    assert final_cursor.deferred_end_offset is None
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages").fetchall() == [("message-0",)]


def test_full_ingest_settles_a_stable_unterminated_first_record_as_no_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable capture whose only record is unterminated admits no records.

    The JSONL parse prefix leaves an unterminated tail out even when it is the
    whole payload, and live intake decides what the tail is: terminal corrupt
    input only once its file is gone, a typed partial when complete records
    precede it (xf8qp). With no complete record the capture settles as a
    no-session observation whose cursor frontier stays before the tail, so
    the record is read again once its writer finishes it.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "static.jsonl"
    path.write_bytes(b'{"type":"session_meta"')
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )

    result = _full_paths_sync(processor, [path], source_name="codex")

    assert result.succeeded == [path]
    assert result.failed == []
    assert result.settled_exclusions == {path: REFUSED_NO_SESSIONS}
    assert result.partial_admissions == {}
    assert result.raw_frontier_sizes[path] == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


def test_full_ingest_heartbeats_small_file_groups_with_current_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record so these fixtures keep testing heartbeat/byte-scan mechanics,
    # not the now-refused empty shape.
    first.write_text(
        '{"type":"session_meta","payload":{"id":"first"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user",'
        '"content":[{"type":"input_text","text":"hello"}]}}\n',
        encoding="utf-8",
    )
    second.write_text(
        '{"type":"session_meta","payload":{"id":"second"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user",'
        '"content":[{"type":"input_text","text":"hello"}]}}\n',
        encoding="utf-8",
    )
    db_path = tmp_path / "archive.sqlite"
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))
    cursor = CursorStore(db_path)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    events: list[tuple[str, Path | None, int | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del stage_payload
        del force
        events.append((phase, current_path, source_payload_read_bytes))

    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )

    result = _full_paths_sync(processor, [first, second], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [first, second]
    assert ("full_file_scan", first, 0) in events
    assert ("full_file_scan", second, first.stat().st_size) in events
    assert any(
        event == ("full_archive_write", second, first.stat().st_size + second.stat().st_size) for event in events
    )


def test_large_full_ingest_uses_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "large.jsonl"
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record so this fixture keeps testing full-ingest mechanics, not the
    # now-refused empty shape.
    source.write_text(
        '{"type":"session_meta","payload":{"id":"large"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user",'
        '"content":[{"type":"input_text","text":"hello"}]}}\n',
        encoding="utf-8",
    )
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )

    result = _full_paths_sync(processor, [source], source_name="codex")

    assert result.succeeded == [source]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "large"


def test_streaming_sized_full_ingest_uses_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "large.jsonl"
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record (before the size padding) so this fixture keeps testing the
    # streaming-vs-eager routing it is named for, not the now-refused empty
    # shape.
    source.write_bytes(
        b'{"type":"session_meta","payload":{"id":"large"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n' + (b" " * (9 * 1024 * 1024))
    )
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    # Acquisition retains bytes only; parsing belongs to retained preparation.
    for parser in ("iter_parsed_payload", "iter_parsed_stream"):
        monkeypatch.setattr(
            f"polylogue.sources.prepared_jsonl.{parser}",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("streaming-sized JSONL acquisition must not parse the input")
            ),
        )

    result = _full_paths_sync(processor, [source], source_name="codex")

    assert result.succeeded == [source]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1


def test_large_weak_path_uses_streaming_route_before_decoded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A weak path cannot force an eager whole-file evidence decode."""

    root = tmp_path / "unknown"
    path = root / "analysis" / "export.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        json.dumps(
            {
                "id": "weak-large",
                "title": "weak large export",
                "create_time": 1781442866.0,
                "update_time": 1781442966.0,
                "current_node": "assistant-node",
                "mapping": {
                    "root": {"id": "root", "message": None, "parent": None, "children": ["user-node"]},
                    "user-node": {
                        "id": "user-node",
                        "parent": "root",
                        "children": ["assistant-node"],
                        "message": {
                            "id": "weak-u1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["question"]},
                            "metadata": {},
                        },
                    },
                    "assistant-node": {
                        "id": "assistant-node",
                        "parent": "user-node",
                        "children": [],
                        "message": {
                            "id": "weak-a1",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["answer"]},
                            "metadata": {},
                        },
                    },
                },
            }
        ).encode()
        + (b" " * (9 * 1024 * 1024))
    )
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    original_read_bytes = Path.read_bytes

    def refuse_source_read_bytes(candidate: Path) -> bytes:
        if candidate == path:
            raise AssertionError("source was read into one byte buffer")
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", refuse_source_read_bytes)
    phases: list[str] = []

    def heartbeat(phase: str, **_kwargs: object) -> None:
        phases.append(phase)

    result = _full_paths_sync(processor, [path], source_name="chatgpt", heartbeat=heartbeat)

    assert result.failed == []
    assert "full_blob_copy" in phases
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)


def test_threshold_crossing_strong_sidecar_is_excluded_before_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A definitive sidecar path must never reach large-JSON admission."""

    root = tmp_path / "chatgpt"
    root.mkdir()
    path = root / "sessions-index.json"
    path.write_bytes(b"{}")
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support.detect_provider_from_path_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("strong sidecar reached JSON provider detection")
        ),
    )

    result = _full_paths_sync(processor, [path], source_name="chatgpt")

    assert result.succeeded == []
    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


def test_full_ingest_writes_archive_with_route_observability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "full-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"full-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    source.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    bootstrap_archive_root(tmp_path)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stage_events: list[tuple[str, dict[str, object] | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del current_path, source_payload_read_bytes, force
        stage_events.append((phase, stage_payload))

    result = _full_paths_sync(processor, [source], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 1
    assert result.changed_session_count == 1
    assert result.raw_fingerprints[source]
    assert {
        "full.provider_parse",
        "full.source_raw_blob_ref_write",
        "full.index_parsed_write",
        "full.index.session_upsert",
        "full.index.full_replace",
        "full.index.full_replace.fts_guard_clear",
        "full.index.full_replace.messages",
        "full.index.full_replace.blocks",
    }.issubset(result.stage_timings_s)
    # The revision replay settles the session's FTS rows in the transaction
    # that writes its blocks, so a just-ingested session is searchable when
    # the route returns. Skipping that repair zeroes ``indexed``.
    with sqlite3.connect(index_db) as conn:
        blocks = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        indexed = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert blocks > 0
    assert indexed > 0
    with sqlite3.connect(source_db) as conn:
        raw_state = conn.execute("SELECT parsed_at_ms, parse_error FROM raw_sessions").fetchone()
        assert raw_state is not None
        assert raw_state[0] is not None
        assert raw_state[1] is None
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "full-v1"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None

    probe_event = next(payload for phase, payload in stage_events if phase == "full_archive_storage_probe")
    assert probe_event == {
        "storage_route": "archive_full",
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": False,
        **_complete_archive_storage_probe_fields(),
    }
    write_event = next(payload for phase, payload in stage_events if phase == "full_archive_write")
    assert write_event == {
        "storage_route": "archive_full",
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
        "storage_write_tiers": "source,index",
        "input_file_count": 1,
        "payload_available_file_count": 0,
        "payload_unavailable_file_count": 1,
        "payload_replayed_from_blob_file_count": 1,
    }
    completed_event = next(payload for phase, payload in stage_events if phase == "full_archive_write_completed")
    assert completed_event is not None
    # Acquisition completes before retained publication writes the session,
    # so this event carries only the acquisition's own counts; the session
    # and message counts are the route result's, asserted above.
    assert {
        key: completed_event[key]
        for key in (
            "storage_route",
            "written_raw_count",
            "payload_unavailable_file_count",
            "payload_replayed_from_blob_file_count",
        )
    } == {
        "storage_route": "archive_full",
        "written_raw_count": 1,
        "payload_unavailable_file_count": 1,
        "payload_replayed_from_blob_file_count": 1,
    }


def test_streaming_full_ingest_writes_archive_from_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "stream-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"stream-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"large"}]}}\n'
    )
    source.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    bootstrap_archive_root(tmp_path)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stage_events: list[tuple[str, dict[str, object] | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del current_path, source_payload_read_bytes, force
        stage_events.append((phase, stage_payload))

    result = _full_paths_sync(processor, [source], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 1
    assert result.changed_session_count == 1
    with sqlite3.connect(source_db) as conn:
        raw_row = conn.execute("SELECT raw_id, blob_size FROM raw_sessions").fetchone()
        assert raw_row[1] == len(payload)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchone()[0] == "stream-v1"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1

    probe_event = next(payload for phase, payload in stage_events if phase == "full_archive_storage_probe")
    assert probe_event == {
        "storage_route": "archive_full",
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": False,
        **_complete_archive_storage_probe_fields(),
    }
    write_event = next(payload for phase, payload in stage_events if phase == "full_archive_write")
    assert write_event == {
        "storage_route": "archive_full",
        "storage_tiers": _ARCHIVE_STORAGE_TIERS,
        "storage_write_tiers": "source,index",
        "input_file_count": 1,
        "payload_available_file_count": 0,
        "payload_unavailable_file_count": 1,
        "payload_replayed_from_blob_file_count": 1,
    }
    completed_event = next(payload for phase, payload in stage_events if phase == "full_archive_write_completed")
    assert completed_event is not None
    # Acquisition completes before retained publication writes the session,
    # so this event carries only the acquisition's own counts; the session
    # and message counts are the route result's, asserted above.
    assert {
        key: completed_event[key]
        for key in (
            "storage_route",
            "written_raw_count",
            "payload_unavailable_file_count",
            "payload_replayed_from_blob_file_count",
        )
    } == {
        "storage_route": "archive_full",
        "written_raw_count": 1,
        "payload_unavailable_file_count": 1,
        "payload_replayed_from_blob_file_count": 1,
    }
    assert raw_row[0] == result.raw_fingerprints[source]


def test_streaming_sized_browser_capture_json_uses_native_payload_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "browser-capture" / "chatgpt"
    root.mkdir(parents=True)
    source = root / "native-capture.json"
    native_payload = {
        "id": "native-large",
        "title": "Native large capture",
        "create_time": 1781442866.0,
        "update_time": 1781442966.0,
        "current_node": "assistant-node",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["user-node"]},
            "user-node": {
                "id": "user-node",
                "parent": "root",
                "children": ["assistant-node"],
                "message": {
                    "id": "native-u1",
                    "author": {"role": "user"},
                    "create_time": 1781442870.0,
                    "content": {"content_type": "text", "parts": ["Native user text"]},
                    "metadata": {},
                },
            },
            "assistant-node": {
                "id": "assistant-node",
                "parent": "user-node",
                "children": [],
                "message": {
                    "id": "native-a1",
                    "author": {"role": "assistant"},
                    "create_time": 1781442880.0,
                    "content": {"content_type": "text", "parts": ["Native answer text"]},
                    "metadata": {"model_slug": "gpt-native"},
                },
            },
        },
        "preserved_native_bytes": "x" * (_RETIRED_FULL_INGEST_SIZE_BOUND + 1024),
    }
    capture_payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:native-large",
        "provenance": {
            "source_url": "https://chatgpt.com/c/native-large",
            "page_title": "ChatGPT - Native large capture",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-native-v1",
            "capture_mode": "snapshot",
        },
        # Real receiver artifacts are key-sorted, so a large native payload
        # precedes the typed session and can push ``session.provider`` beyond
        # the ordinary 8 KiB acquisition prefix.
        "raw_provider_payload": native_payload,
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "native-large",
            "title": "DOM fallback title",
            "updated_at": "2026-04-24T00:00:01+00:00",
            "turns": [{"provider_turn_id": "dom-u1", "role": "user", "text": "DOM fallback", "ordinal": 0}],
        },
        "padding": "x" * (_RETIRED_FULL_INGEST_SIZE_BOUND + 1024),
    }
    source.write_text(json.dumps(capture_payload), encoding="utf-8")
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    bootstrap_archive_root(tmp_path)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="browser-capture", root=root.parent),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    result = _full_paths_sync(processor, [source], source_name="browser-capture")

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 2
    assert source.read_bytes().find(b'"provider": "chatgpt"') > 8192
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT origin FROM raw_sessions").fetchone() == ("chatgpt-export",)
        assert conn.execute("SELECT logical_source_key FROM raw_session_memberships").fetchone() == (
            "chatgpt-export:native-large",
        )
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id, title FROM sessions").fetchone() == (
            "native-large",
            "Native large capture",
        )
        assert (
            conn.execute(
                """
            SELECT group_concat(item, '|')
            FROM (
                SELECT messages.role || ':' || blocks.text AS item
                FROM messages
                JOIN blocks USING (message_id)
                ORDER BY messages.position, blocks.position
            )
            """
            ).fetchone()[0]
            == "user:Native user text|assistant:Native answer text"
        )
        assert conn.execute("SELECT logical_source_key, session_id FROM raw_revision_heads").fetchone() == (
            "chatgpt-export:native-large",
            "chatgpt-export:native-large",
        )


def test_generic_large_browser_capture_json_uses_prefix_detection_without_unknown_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "inbox"
    root.mkdir()
    source = root / "large-browser-capture.json"
    capture_payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:generic-large",
        "provenance": {
            "source_url": "https://chatgpt.com/c/generic-large",
            "page_title": "ChatGPT - Generic capture",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "generic-large",
            "title": "Generic inbox browser capture",
            "updated_at": "2026-04-24T00:00:01+00:00",
            "turns": [
                {"provider_turn_id": "u1", "role": "user", "text": "Generic user text", "ordinal": 0},
                {"provider_turn_id": "a1", "role": "assistant", "text": "Generic answer text", "ordinal": 1},
            ],
        },
        "padding": "x" * (_RETIRED_FULL_INGEST_SIZE_BOUND + 1024),
    }
    source.write_text(json.dumps(capture_payload), encoding="utf-8")
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    bootstrap_archive_root(tmp_path)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    result = _full_paths_sync(processor, [source], source_name="inbox")

    assert result.succeeded == [source]
    assert result.failed == []
    assert result.ingested_session_count == 1
    assert result.ingested_message_count == 2
    with sqlite3.connect(source_db) as conn:
        # Acquisition identity is artifact-scoped because one raw file may
        # contain many sessions. Parsed identity lives in index + membership.
        assert conn.execute("SELECT origin, native_id FROM raw_sessions").fetchone() == ("chatgpt-export", None)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id, title, message_count FROM sessions").fetchone() == (
            "generic-large",
            "Generic inbox browser capture",
            2,
        )


def test_large_browser_capture_prefix_planning_does_not_materialize_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large-browser-capture.json"
    target.write_text(
        json.dumps(
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "prefix-only",
                    "turns": [{"provider_turn_id": "u1", "role": "user", "text": "x"}],
                },
                "provenance": {
                    "source_url": "https://chatgpt.com/c/prefix-only",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("large browser-capture planning must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _detect_provider_from_path(target, Provider.UNKNOWN) is Provider.CHATGPT
    assert _parse_path_as_session_artifact(target, provider=Provider.CHATGPT) is True


def test_browser_capture_prefix_probe_finds_provider_past_1mib_raw_payload(tmp_path: Path) -> None:
    """polylogue-mvq8: session.provider beyond the 1MiB prefix must still detect.

    Real receiver artifacts key-sort with ``raw_provider_payload`` (an
    unbounded copy of the provider's own wire payload) sorting before
    ``session`` alphabetically. Once ``raw_provider_payload`` alone exceeds
    the 1MiB prefix-probe window, the plain byte-prefix regex never sees
    ``session.provider`` and the capture was permanently misdetected as
    ``unknown-export`` -- this reproduces that exact shape with real file
    bytes (no probe-size monkeypatching) and asserts the provider is still
    found.
    """
    target = tmp_path / "oversized-raw-payload.json"
    huge_padding = "x" * (_BROWSER_CAPTURE_PREFIX_PROBE_BYTES + 64 * 1024)
    capture_payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:past-prefix",
        "provenance": {
            "source_url": "https://chatgpt.com/c/past-prefix",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-native-v1",
        },
        # Deliberately placed before ``session`` (as the real receiver's
        # key-sorted output places it) and sized past the probe window.
        "raw_provider_payload": {"padding": huge_padding},
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "past-prefix",
            "turns": [{"provider_turn_id": "u1", "role": "user", "text": "hi"}],
        },
    }
    target.write_text(json.dumps(capture_payload), encoding="utf-8")

    # Confirm the fixture actually reproduces the bug shape: the provider
    # marker sits past the probe window, and the file exceeds it too.
    assert target.stat().st_size > _BROWSER_CAPTURE_PREFIX_PROBE_BYTES
    assert target.read_bytes().find(b'"provider": "chatgpt"') > _BROWSER_CAPTURE_PREFIX_PROBE_BYTES

    is_browser_capture, provider = _browser_capture_prefix_probe(target)
    assert is_browser_capture is True
    assert provider is Provider.CHATGPT


def test_full_ingest_storage_probe_reports_the_existing_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "bootstrap-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"bootstrap-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"boot"}]}}\n'
    )
    source.write_bytes(payload)
    db_path = tmp_path / "archive.sqlite"
    cursor = CursorStore(db_path)
    # The live route acquires into an existing archive; it refuses a missing
    # Source tier rather than bootstrapping one (#3952).
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stage_events: list[tuple[str, dict[str, object] | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        del current_path, source_payload_read_bytes, force
        stage_events.append((phase, stage_payload))

    result = _full_paths_sync(processor, [source], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [source]
    for filename in (spec.filename for spec in ARCHIVE_TIER_SPECS.values()):
        assert (tmp_path / filename).exists()
    probe_event = next(payload for phase, payload in stage_events if phase == "full_archive_storage_probe")
    assert probe_event == {
        "storage_route": "archive_full",
        "storage_write_tiers": "source,index",
        "archive_active": True,
        "archive_bootstrapped": False,
        **_archive_storage_probe_fields(
            present=set(ARCHIVE_TIER_SPECS),
            versions={tier: spec.version for tier, spec in ARCHIVE_TIER_SPECS.items()},
        ),
    }


def test_fingerprint_file_streams_in_bounded_memory(tmp_path: Path) -> None:
    """``fingerprint_file`` must not load the whole file into memory.

    Regression: the previous implementation read the entire file via
    ``Path.read_bytes()``, producing an RSS peak proportional to file size.
    This test exercises the streaming path on a multi-megabyte synthetic
    file and asserts that the working set stays bounded by ``chunk_size``.
    """
    import hashlib
    import tracemalloc

    from polylogue.sources.live.batch_support import fingerprint_file

    payload = (b"x" * 4095 + b"\n") * 4096  # ~16 MiB, all lines newline-terminated
    target = tmp_path / "huge.jsonl"
    target.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    expected_last_nl = len(payload)  # ends in newline

    tracemalloc.start()
    try:
        fp, last_nl = fingerprint_file(target, chunk_size=64 * 1024)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert fp == expected_hash
    assert last_nl == expected_last_nl
    # Peak Python-allocated memory must stay well under the file size. The
    # 1 MiB budget is generous (chunk_size is 64 KiB) but leaves room for
    # hasher state and chunk overhead without admitting a full-file read.
    assert peak < 1 * 1024 * 1024, f"fingerprint_file peak {peak} bytes is not bounded for a {len(payload)}-byte file"


def test_fingerprint_file_tracks_last_newline_across_chunk_boundary(tmp_path: Path) -> None:
    """The streaming fingerprint must locate the last newline even when it
    sits in an earlier chunk than the file tail."""
    from polylogue.sources.live.batch_support import fingerprint_file

    # 4 KiB of newline-terminated lines, followed by 4 KiB without any \n.
    head = (b"line\n") * 1000  # 5_000 bytes, ends with \n
    tail = b"y" * 5000  # no newline anywhere
    payload = head + tail
    target = tmp_path / "no-trailing-newline.jsonl"
    target.write_bytes(payload)

    _fp, last_nl = fingerprint_file(target, chunk_size=1024)
    assert last_nl == len(head), f"last_complete_newline should be at end-of-head ({len(head)}), got {last_nl}"


def test_fingerprint_file_empty_file(tmp_path: Path) -> None:
    import hashlib

    from polylogue.sources.live.batch_support import fingerprint_file

    target = tmp_path / "empty.jsonl"
    target.write_bytes(b"")

    fp, last_nl = fingerprint_file(target)
    assert fp == hashlib.sha256(b"").hexdigest()
    assert last_nl == 0


def test_large_non_jsonl_full_ingest_planning_does_not_read_whole_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large.json"
    target.write_text('{"mapping": {}}\n', encoding="utf-8")
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("large full-ingest planning must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _detect_provider_from_path(target, Provider.CHATGPT) is Provider.CHATGPT
    assert _parse_path_as_session_artifact(target, provider=Provider.CHATGPT) is True


def test_unclassified_large_non_jsonl_is_admitted_to_preparation_without_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning never reads an unclassified file to decide its session status.

    No path declaration names it non-session material, so it is admitted and
    canonical preparation publishes its typed session, non-session, or
    refusal outcome (#5557); a bounded probe cannot decide that.
    """
    target = tmp_path / "unknown.large"
    target.write_bytes(b"not-json")
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 32 * 1024 * 1024)

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("unclassified large files must not be materialized during planning")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _parse_path_as_session_artifact(target, provider=Provider.UNKNOWN) is True


def test_full_ingest_retains_sidecar_evidence_and_ingests_genuine_session(tmp_path: Path) -> None:
    """Full live acquisition keeps non-session evidence and repairs session-shaped journals."""
    root = tmp_path / ".claude"
    metadata_path = root / "projects" / "project" / "subagents" / "agent-a.meta.json"
    journal_path = root / "projects" / "project" / "subagents" / "workflows" / "wf-run-1" / "journal.jsonl"
    session_path = root / "projects" / "project" / "genuine-session.jsonl"
    metadata_path.parent.mkdir(parents=True)
    journal_path.parent.mkdir(parents=True)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_payload = b'{"agentId":"agent-a","transcriptPath":"agent-a.jsonl"}'
    journal_payload = (
        json.dumps(
            {
                "type": "user",
                "sessionId": "wf-run-1",
                "uuid": "journal-message-1",
                "message": {"role": "user", "content": "retain this workflow evidence"},
            }
        )
        + "\n"
    ).encode()
    session_payload = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"real session"},'
        b'"uuid":"real-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"real-user","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"real reply"}]},"uuid":"real-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    metadata_path.write_bytes(metadata_payload)
    journal_path.write_bytes(journal_payload)
    session_path.write_bytes(session_payload)

    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    result = run_ingest_files(processor, [metadata_path, journal_path, session_path], emit_event=False)

    # The metadata sidecar is retained evidence that settles as a terminal
    # no-session exclusion; the two session-shaped files succeed.
    assert result.succeeded_file_count == 2
    assert result.failed_file_count == 0
    assert result.refused_bytes_by_reason == {REFUSED_NO_SESSIONS: len(metadata_payload)}
    assert result.ingested_session_count == 2
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (2,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT a.source_path, a.artifact_kind, a.support_status, a.parse_as_session, r.blob_hash
            FROM raw_artifacts AS a
            JOIN raw_sessions AS r ON r.raw_id = a.raw_id
            WHERE a.parse_as_session = 0
            ORDER BY a.source_path
            """
        ).fetchall()

    assert [(Path(row[0]).name, row[1], row[2], row[3]) for row in rows] == [
        ("agent-a.meta.json", "agent_sidecar_meta", "unknown", 0),
    ]
    expected_payloads = {
        metadata_path.name: metadata_payload,
    }
    for source_path, _kind, _support_status, _parse_as_session, blob_hash in rows:
        blob_hash_hex = bytes(blob_hash).hex()
        assert (tmp_path / "blob" / blob_hash_hex[:2] / blob_hash_hex[2:]).read_bytes() == expected_payloads[
            Path(source_path).name
        ]


def test_unknown_inbox_zip_source_only_route_retains_session_and_sidecars(tmp_path: Path) -> None:
    bundle = tmp_path / "claude-export.zip"
    session_payload = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"real session"},'
        b'"uuid":"real-user","timestamp":"2025-01-01T00:00:00Z"}\n'
    )
    sidecar_payload = b"opaque tool result"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("projects/project/tool-results/dump.json", b'{"provider":"claude.ai"}')
        archive.writestr("projects/project/session.jsonl", session_payload)
        archive.writestr("projects/project/tool-results/toolu.txt", sidecar_payload)

    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="unknown", root=tmp_path),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    with live_zip_capture(tmp_path) as (publisher, zip_inputs):
        source_only = processor._extract_source_only_zip_member_records(
            bundle,
            blob_store=publisher,
            zip_inputs=zip_inputs,
            fallback_provider=Provider.UNKNOWN,
            file_mtime="2026-09-04T00:00:00+00:00",
        )
    assert source_only is not None
    source_only_records, _source_only_bytes = source_only
    assert {record.source_path.rsplit(":", 1)[-1] for _raw_id, record in source_only_records} == {
        "projects/project/tool-results/dump.json",
        "projects/project/session.jsonl",
        "projects/project/tool-results/toolu.txt",
    }


def test_unknown_zip_live_route_retains_declared_binary_and_markdown_artifacts(tmp_path: Path) -> None:
    from polylogue.sources.live.production_baseline import capture_production_source_baseline

    bundle = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("tool-results/one.bin", b"\xff\x00opaque")
        archive.writestr("brain/one.md", b"# note\n")
    processor = LiveBatchProcessor.__new__(LiveBatchProcessor)
    processor._cursor = CursorStore(tmp_path / "index.db")
    processor._zip_member_refusals_this_pass = {}
    with live_zip_capture(tmp_path) as (publisher, zip_inputs):
        extracted = processor._extract_source_only_zip_member_records(
            bundle,
            blob_store=publisher,
            zip_inputs=zip_inputs,
            fallback_provider=Provider.UNKNOWN,
            file_mtime="2026-09-04T00:00:00+00:00",
        )
        assert extracted is not None
        records, _total_bytes = extracted
    assert {record.source_path for _raw_id, record in records} == {
        f"{bundle}:tool-results/one.bin",
        f"{bundle}:brain/one.md",
    }
    baseline = capture_production_source_baseline(
        (WatchSource(name="inbox", root=tmp_path, layout=export_drop_layout((".zip",))),), operation_id="artifacts"
    )
    assert {(row.path, row.source_index, row.revision) for row in baseline.accepted} == {
        (record.source_path, record.source_index, record.blob_hash) for _raw_id, record in records
    }


def _write_plain_sqlite_db(path: Path) -> None:
    """A genuine SQLite database with no Hermes state.db/verification_evidence.db shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE unrelated_thing (id INTEGER PRIMARY KEY, value TEXT);")
        conn.commit()


def test_parse_path_as_session_artifact_still_accepts_genuine_hermes_state_db(tmp_path: Path) -> None:
    """Regression guard: the tightened check must not break the real feature."""
    target = tmp_path / "state.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, model TEXT, model_config TEXT,
                parent_session_id TEXT, started_at REAL, ended_at REAL, title TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_name TEXT, tool_calls TEXT,
                timestamp REAL NOT NULL, observed INTEGER DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0);
            """
        )
        conn.commit()

    assert _parse_path_as_session_artifact(target, provider=Provider.HERMES) is True


def test_append_plan_chunks_large_tail_without_full_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    original = _codex_meta_line("chunked-append")
    first_chunk = _codex_record_line("x" * (_MAX_APPEND_PLAN_PAYLOAD_BYTES - 256))
    second_chunk = _codex_record_line("y" * 512)
    assert len(first_chunk) < _MAX_APPEND_PLAN_PAYLOAD_BYTES < len(first_chunk) + len(second_chunk)
    appended = first_chunk + second_chunk
    path.write_bytes(original + appended)
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    _archive_codex_session(tmp_path, "chunked-append")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(original)
    assert plan.last_complete_newline == len(original) + len(first_chunk)
    assert plan.stat_size == len(original) + len(appended)
    assert plan.bytes_read == _MAX_APPEND_PLAN_PAYLOAD_BYTES
    assert plan.payload == first_chunk

    assert processor._record_append_cursor(plan) is True
    next_plan = processor._append_plan(path)
    assert isinstance(next_plan, _AppendPlan)
    assert next_plan.start_offset == len(original) + len(first_chunk)
    assert next_plan.last_complete_newline == len(original) + len(appended)
    assert next_plan.payload == second_chunk


def test_append_cursor_survives_source_disappearing_after_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = _codex_meta_line("vanishing-append")
    path.write_bytes(original + _codex_record_line("after admission"))
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    _archive_codex_session(tmp_path, "vanishing-append")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)

    def missing(_self: Path) -> os.stat_result:
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "stat", missing)
    assert processor._record_append_cursor(plan) is True
    cursor = processor._cursor.get_record(path)
    assert cursor is not None
    assert cursor.byte_offset == plan.last_complete_newline


def test_append_plan_defers_when_tail_has_no_complete_line(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    path.write_bytes(original + b'{"b":')
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )

    assert processor._append_plan(path) is _DEFER_APPEND


@pytest.mark.asyncio
async def test_full_drive_capture_retains_acquisition_mode_after_gemini_detection(tmp_path: Path) -> None:
    """A configured Drive source survives shape detection's GEMINI fallback."""
    from polylogue.api import Polylogue

    root = tmp_path / "drive"
    root.mkdir()
    path = root / "live-capture.json"
    path.write_text(
        json.dumps(
            {
                "id": "live-drive-capture",
                "title": "Live Drive capture",
                "chunkedPrompt": {
                    "chunks": [
                        {"id": "chunk-1", "role": "user", "text": "hello"},
                        {"id": "chunk-2", "role": "model", "text": "hi"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    archive = Polylogue(archive_root=tmp_path / "archive")
    run_off_event_loop(lambda: bootstrap_archive_root(archive.archive_root))
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="drive", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        metrics = await ingest_files_with_owners(processor, [path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as conn:
            capture_mode = conn.execute("SELECT capture_mode FROM raw_sessions").fetchone()

        assert metrics.full_file_count == 1
        assert capture_mode == (Provider.DRIVE.value,)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_empty_default_claude_history_cursor_settles_as_raw_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty history sidecar has no Claude transcript semantic frontier."""
    from polylogue.api import Polylogue
    from polylogue.sources.live.watcher import default_sources
    from polylogue.sources.source_layout import declared_source_layout

    home = tmp_path / "home"
    claude_home = home / ".claude"
    claude_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    source = next(item for item in default_sources() if item.name == "claude-code-history")
    assert source.layout == declared_source_layout("claude-code-history")
    path = source.root / "history.jsonl"
    path.write_bytes(b"")

    archive = Polylogue(archive_root=tmp_path / "archive")
    run_off_event_loop(lambda: bootstrap_archive_root(archive.archive_root))
    cursor = CursorStore(archive.backend.db_path)
    processor = LiveBatchProcessor(
        archive,
        (source,),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        result = await ingest_files_with_owners(processor, [path], emit_event=False)
        assert result.excluded_reasons == {"no_sessions": 1}
        assert result.stale_cursor_write_count == 0
        row = cursor.get_record(path)
        assert row is not None and row.content_fingerprint == sha256(b"").hexdigest()
        assert row.failure_count == 0

        with sqlite3.connect(archive.archive_root / "source.db") as conn:
            artifact = conn.execute(
                "SELECT artifact_kind, parse_as_session FROM raw_artifacts WHERE source_path = ?",
                (str(path),),
            ).fetchone()
        assert artifact == ("prompt_history_log", 0)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_inbox_browser_capture_json_replacement_uses_full_ingest(tmp_path: Path) -> None:
    from polylogue.api import Polylogue

    root = tmp_path / "inbox"
    root.mkdir()
    path = root / "capture.json"

    def capture(turns: list[dict[str, object]]) -> dict[str, object]:
        return {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "chatgpt:inbox-replacement",
            "provenance": {
                "source_url": "https://chatgpt.com/c/inbox-replacement",
                "page_title": "Inbox replacement",
                "captured_at": "2026-07-11T00:00:00+00:00",
                "adapter_name": "chatgpt-native-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": "inbox-replacement",
                "title": "Inbox replacement",
                "updated_at": "2026-07-11T00:00:01+00:00",
                "turns": turns,
            },
        }

    first_turn = {
        "provider_turn_id": "turn-1",
        "role": "user",
        "text": "first snapshot",
        "ordinal": 0,
    }
    replacement_turn = {
        "provider_turn_id": "turn-2",
        "role": "assistant",
        "text": "replacement snapshot",
        "ordinal": 1,
    }
    path.write_text(json.dumps(capture([first_turn])), encoding="utf-8")
    archive = Polylogue(archive_root=tmp_path / "archive")
    run_off_event_loop(lambda: bootstrap_archive_root(archive.archive_root))
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".json", ".jsonl"))),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        first = await ingest_files_with_owners(processor, [path], emit_event=False)
        path.write_text(json.dumps(capture([first_turn, replacement_turn])), encoding="utf-8")
        second = await ingest_files_with_owners(processor, [path], emit_event=False)
        assert first.full_file_count == 1
        assert second.full_file_count == 1
        assert second.append_file_count == 0
        with sqlite3.connect(archive.archive_root / "source.db") as conn:
            source_indexes = conn.execute(
                "SELECT source_index FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms",
                (str(path),),
            ).fetchall()
        assert source_indexes == [(0,), (0,)]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_browser_capture_replacement_advances_membership_head_and_acquires_attachment(tmp_path: Path) -> None:
    """A mutable receiver snapshot must retain both raws but materialize the newer capture.

    Every retained raw of the logical key governs its membership, whatever
    route acquired it: a divergent capture is ambiguity debt that keeps the
    accepted head (the final divergent pass below, and
    ``test_live_multi_session_divergence_keeps_accepted_head_as_debt``). The
    former law that excluded an unrelated quarantined census of the same key
    predates membership-only governance (5d94f28b73 injects every cohort
    member and the head into the comparison).
    """
    from polylogue.api import Polylogue

    root = tmp_path / "browser-capture"
    root.mkdir()
    path = root / "capture.json"
    asset_bytes = b"browser-capture-asset" * 37
    asset_hash = sha256(asset_bytes).digest()

    def capture(
        turns: list[dict[str, object]],
        *,
        captured_at: str = "2026-07-12T00:00:00+00:00",
    ) -> dict[str, object]:
        return {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "chatgpt:browser-replacement",
            "provenance": {
                "source_url": "https://chatgpt.com/c/browser-replacement",
                "captured_at": captured_at,
                "adapter_name": "chatgpt-native-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": "browser-replacement",
                "title": "Browser replacement",
                "updated_at": "2026-07-12T00:00:01+00:00",
                "turns": turns,
            },
        }

    first_turn = {"provider_turn_id": "turn-1", "role": "user", "text": "make an asset", "ordinal": 0}
    acquired_turn = {
        "provider_turn_id": "turn-2",
        "role": "assistant",
        "text": "asset acquired",
        "ordinal": 1,
        "attachments": [
            {
                "provider_attachment_id": "asset-1",
                "message_provider_id": "turn-2",
                "name": "deliverable.bin",
                "mime_type": "application/octet-stream",
                "inline_base64": base64.b64encode(asset_bytes).decode("ascii"),
            }
        ],
    }
    divergent_turn = {
        "provider_turn_id": "turn-divergent",
        "role": "assistant",
        "text": "older divergent snapshot",
        "ordinal": 1,
    }
    path.write_text(json.dumps(capture([first_turn])), encoding="utf-8")
    archive = Polylogue(archive_root=tmp_path / "archive")
    run_off_event_loop(lambda: bootstrap_archive_root(archive.archive_root))
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="browser-capture", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        first = await ingest_files_with_owners(processor, [path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            first_raw_id = source_conn.execute(
                "SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),)
            ).fetchone()[0]
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            assert (
                index_conn.execute(
                    "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:browser-replacement'"
                ).fetchone()[0]
                == first_raw_id
            )
        assert first.succeeded_file_count == 1

        path.write_text(json.dumps(capture([first_turn, acquired_turn])), encoding="utf-8")
        replacement = await ingest_files_with_owners(processor, [path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids = [
                str(row[0])
                for row in source_conn.execute(
                    "SELECT raw_id FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms", (str(path),)
                )
            ]
            decisions = source_conn.execute(
                """
                SELECT raw_id, decision FROM raw_session_memberships
                WHERE logical_source_key = 'chatgpt-export:browser-replacement'
                """
            ).fetchall()
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            accepted_raw_id = index_conn.execute(
                "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:browser-replacement'"
            ).fetchone()[0]
            attachment = index_conn.execute(
                "SELECT acquisition_status, byte_count, blob_hash FROM attachments WHERE display_name = 'deliverable.bin'"
            ).fetchone()

        assert first.full_file_count == replacement.full_file_count == 1
        assert len(raw_ids) == 2
        assert accepted_raw_id in raw_ids
        live_decisions = {raw_id: decision for raw_id, decision in decisions if raw_id in raw_ids}
        assert set(live_decisions.values()) == {"superseded_prefix", "applied"}
        assert live_decisions[accepted_raw_id] == "applied"
        assert attachment == ("acquired", len(asset_bytes), asset_hash)

        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_before_reverse = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
        path.write_text(
            json.dumps(capture([first_turn], captured_at="2026-07-12T00:00:02+00:00")),
            encoding="utf-8",
        )
        reverse = await ingest_files_with_owners(processor, [path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_after_reverse = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
            reverse_raw_id = (raw_ids_after_reverse - raw_ids_before_reverse).pop()
            reverse_decision = source_conn.execute(
                """
                SELECT decision FROM raw_session_memberships
                WHERE raw_id = ? AND logical_source_key = 'chatgpt-export:browser-replacement'
                """,
                (reverse_raw_id,),
            ).fetchone()[0]
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            assert (
                index_conn.execute(
                    "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:browser-replacement'"
                ).fetchone()[0]
                == accepted_raw_id
            )
        assert reverse.full_file_count == 1
        # Equivalent parser snapshots may elect either retained raw as the
        # canonical prefix representative; both are terminal receipts and
        # neither may displace the newer accepted head.
        assert reverse_decision in {"superseded_equivalent", "superseded_prefix"}

        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_before_divergence = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
        path.write_text(json.dumps(capture([first_turn, divergent_turn])), encoding="utf-8")
        divergent = await ingest_files_with_owners(processor, [path], emit_event=False)
        with sqlite3.connect(archive.archive_root / "source.db") as source_conn:
            raw_ids_after_divergence = {
                str(row[0])
                for row in source_conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path = ?", (str(path),))
            }
            divergent_raw_id = (raw_ids_after_divergence - raw_ids_before_divergence).pop()
            divergent_decision = source_conn.execute(
                """
                SELECT decision FROM raw_session_memberships
                WHERE raw_id = ? AND logical_source_key = 'chatgpt-export:browser-replacement'
                """,
                (divergent_raw_id,),
            ).fetchone()[0]
        with sqlite3.connect(archive.archive_root / "index.db") as index_conn:
            assert (
                index_conn.execute(
                    "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:browser-replacement'"
                ).fetchone()[0]
                == accepted_raw_id
            )
        assert divergent.full_file_count == 1
        assert divergent_decision == "ambiguous"
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_browser_capture_provider_timestamp_advances_reordered_native_snapshot(tmp_path: Path) -> None:
    """Provider-native snapshots may insert work before an existing context node."""
    from polylogue.api import Polylogue

    root = tmp_path / "browser-capture"
    root.mkdir()
    path = root / "capture.json"

    def capture(turns: list[dict[str, object]], *, updated_at: str) -> dict[str, object]:
        return {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": "chatgpt:provider-ordered-replacement",
            "provenance": {
                "source_url": "https://chatgpt.com/c/provider-ordered-replacement",
                "captured_at": updated_at,
                "adapter_name": "chatgpt-native-v1",
                "capture_mode": "snapshot",
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": "provider-ordered-replacement",
                "title": "Provider ordered replacement",
                "updated_at": updated_at,
                "turns": turns,
            },
            # The synthetic compact projection is refused as non-native
            # evidence (capture_retired_projection); the snapshot's ordered
            # turns are its evidence.
        }

    prompt = {"provider_turn_id": "prompt", "role": "user", "text": "do work", "ordinal": 0}
    context = {
        "provider_turn_id": "attachment-context",
        "role": "user",
        "text": "The user provided an attachment",
        "ordinal": 1,
    }
    tool = {"provider_turn_id": "tool", "role": "assistant", "text": "tool output", "ordinal": 1}
    path.write_text(json.dumps(capture([prompt, context], updated_at="2026-07-16T00:00:00Z")), encoding="utf-8")
    archive = Polylogue(archive_root=tmp_path / "archive")
    run_off_event_loop(lambda: bootstrap_archive_root(archive.archive_root))
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="browser-capture", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(archive.backend.db_path),
        parser_fingerprint="test-parser",
    )

    try:
        first = await ingest_files_with_owners(processor, [path], emit_event=False)
        path.write_text(
            json.dumps(capture([prompt, tool, context], updated_at="2026-07-16T00:01:00Z")),
            encoding="utf-8",
        )
        second = await ingest_files_with_owners(processor, [path], emit_event=False)

        with sqlite3.connect(archive.archive_root / "index.db") as conn:
            row = conn.execute(
                "SELECT message_count, title FROM sessions WHERE session_id = ?",
                ("chatgpt-export:provider-ordered-replacement",),
            ).fetchone()
        with sqlite3.connect(archive.archive_root / "source.db") as conn:
            decisions = conn.execute(
                """
                SELECT decision FROM raw_session_memberships
                WHERE logical_source_key = 'chatgpt-export:provider-ordered-replacement'
                ORDER BY acquisition_generation
                """
            ).fetchall()

        assert first.succeeded_file_count == second.succeeded_file_count == 1
        assert row == (3, "Provider ordered replacement")
        assert {decision for (decision,) in decisions} == {"superseded_prefix", "applied"}
    finally:
        await archive.close()


def test_generic_inbox_jsonl_stream_takes_the_full_route_instead_of_an_append_plan(tmp_path: Path) -> None:
    """An inbox stream has no stable session identity to bind a delta to.

    Since c07c4f44b1 the planner returns no append plan for it, so the full
    route re-reads the file; an identity-less plan would fail acquisition and
    retry on every growth.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"append-safe"}}\n'
    appended = b'{"type":"event_msg","payload":{"message":"new"}}\n'
    path.write_bytes(original + appended)
    db_path = tmp_path / "archive.sqlite"
    cursor = CursorStore(db_path)
    stat = path.stat()
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        source_name="inbox",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".jsonl",))),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    assert processor._append_plan(path) is None


def test_incomplete_append_is_requeued_not_full_ingested(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    path.write_bytes(original + b'{"b":')
    db_path = tmp_path / "archive.sqlite"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="chatgpt", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(original),
        source_name="chatgpt",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )

    metrics = run_ingest_files(processor, [path], emit_event=False)

    assert metrics.full_file_count == 0
    assert metrics.append_file_count == 0
    assert metrics.failed_paths == [str(path)]


def test_codex_append_plan_uses_append_only_session_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rollout-2026-05-16T13-50-17-5370dcbb-87a9-446b-954f-be2a1df29915.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"5370dcbb-87a9-446b-954f-be2a1df29915"}}\n'
    appended = b'{"type":"event_msg","payload":{"message":"new"}}\n'
    path.write_bytes(original + appended)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(path),
            canonical_source_path=str(path),
            source_index=-1,
            payload=original,
            acquired_at_ms=1_770_000_000_000,
        )
        blob_hash = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
    _write_archive_blob(tmp_path, cast(bytes, blob_hash), original)
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "5370dcbb-87a9-446b-954f-be2a1df29915",
                "codex-session",
                raw_id,
                "hot session",
                bytes([7]) * 32,
                1_770_000_000_000,
                1_770_000_000_000,
            ),
        )
        conn.commit()
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base-cursor",
        tail_hash=_cursor_hash_authority(original),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(original)
    # polylogue-u19l: the stored/hashed payload is now the literal live-file
    # bytes -- no synthetic session_meta header spliced in. The identity is
    # carried instead as a sidecar hint (persisted to raw_sessions.native_id
    # and used to override the parser's fallback_id on replay).
    assert plan.payload == appended
    assert plan.native_id_hint == "5370dcbb-87a9-446b-954f-be2a1df29915"


def test_codex_append_plan_reads_archive_file_set_session_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rollout-2026-05-16T13-50-17-5370dcbb-87a9-446b-954f-be2a1df29915.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"5370dcbb-87a9-446b-954f-be2a1df29915"}}\n'
    appended = b'{"type":"event_msg","payload":{"message":"new"}}\n'
    path.write_bytes(original + appended)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(path),
            canonical_source_path=str(path),
            source_index=0,
            payload=original,
            acquired_at_ms=1_770_000_000_000,
        )
        blob_hash = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
    _write_archive_blob(tmp_path, cast(bytes, blob_hash), original)
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "5370dcbb-87a9-446b-954f-be2a1df29915",
                "codex-session",
                raw_id,
                "hot session",
                bytes([7]) * 32,
                1_770_000_000_000,
                1_770_000_000_000,
            ),
        )
        conn.commit()
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base-cursor",
        tail_hash=_cursor_hash_authority(original),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )

    plan = processor._append_plan(path)

    assert isinstance(plan, _AppendPlan)
    # polylogue-u19l: literal live-file bytes, identity carried as a hint.
    assert plan.payload == appended
    assert plan.native_id_hint == "5370dcbb-87a9-446b-954f-be2a1df29915"
    assert processor._latest_raw_fingerprint(path) == raw_id
    with cursor._connect() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None


@pytest.mark.parametrize(
    ("index_origin", "source_origin"),
    [
        ("codex-session", "claude-code-session"),
        ("claude-code-session", "codex-session"),
    ],
)
def test_codex_append_identity_rejects_mixed_origins_at_same_path(
    tmp_path: Path,
    index_origin: str,
    source_origin: str,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "shared.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"codex-id"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=source_origin,
            source_path=str(path),
            canonical_source_path=str(path),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_000,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", index_origin, raw_id, "mixed origin", bytes(32)),
        )
        conn.commit()

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._append_payload_for_provider(path, "codex", b'{"type":"event_msg"}\n') is None


def test_codex_append_identity_rejects_mismatched_index_owner_before_global_fallback(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "shared.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"codex-id"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)
    with sqlite3.connect(source_db) as conn:
        wrong_owner_raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(path),
            canonical_source_path=str(path),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_000,
        )
        unrelated_codex_raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(root / "other.jsonl"),
            canonical_source_path=str(root / "other.jsonl"),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_001,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", "claude-code-session", wrong_owner_raw_id, "wrong owner", bytes(32)),
        )
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", "codex-session", unrelated_codex_raw_id, "unrelated fallback", bytes(32)),
        )
        conn.commit()

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._append_payload_for_provider(path, "codex", b'{"type":"event_msg"}\n') is None


def test_codex_append_identity_rejects_global_fallback_when_ownership_query_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "candidate.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"codex-id"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)
    with sqlite3.connect(source_db) as conn:
        unrelated_raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(root / "unrelated.jsonl"),
            canonical_source_path=str(root / "unrelated.jsonl"),
            source_index=0,
            payload=payload,
            acquired_at_ms=1_770_000_000_000,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) VALUES (?, ?, ?, ?, ?)",
            ("codex-id", "codex-session", unrelated_raw_id, "unrelated fallback", bytes(32)),
        )
        conn.commit()

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._existing_provider_session_id(path, expected_origin="codex-session") == "codex-id"

    def unavailable_ownership_view(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("source tier unavailable")

    # The global index fallback must be viable so this assertion proves that an
    # unavailable ownership view, rather than another sqlite failure, rejects
    # the append.
    monkeypatch.setattr(processor, "_archive_has_native_session", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sqlite3, "connect", unavailable_ownership_view)

    assert processor._append_payload_for_provider(path, "codex", b'{"type":"event_msg"}\n') is None
    assert "source-path ownership view unavailable" in caplog.text


def test_latest_raw_fingerprint_ignores_archive_source_row_with_missing_blob(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "missing-blob.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"missing-blob"}}\n'
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)
    blob_hash = b"a" * 32
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-missing-blob", "codex-session", "missing-blob", str(path), 0, blob_hash, len(payload), 1),
        )
        conn.commit()
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    assert processor._latest_raw_fingerprint(path) is None

    _write_archive_blob(tmp_path, blob_hash, payload)

    assert processor._latest_raw_fingerprint(path) == "raw-missing-blob"


def test_append_ingest_preserves_successes_when_other_plan_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ok_payload = (
        b'{"type":"session_meta","payload":{"id":"append-ok","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"ok"}]}}\n'
    )
    bad_payload = b"{bad json}\n"

    class Owner:
        def __init__(self) -> None:
            self._cursor = CursorStore(tmp_path / "append.sqlite")
            self._polylogue = SimpleNamespace(
                archive_root=tmp_path,
                backend=SimpleNamespace(db_path=self._cursor._db_path),
            )

    plans = [
        _AppendPlan(
            path=tmp_path / "ok.jsonl",
            canonical_source_path=str(tmp_path / "ok.jsonl"),
            captured_profile_key=None,
            source_name="codex",
            start_offset=0,
            last_complete_newline=8,
            stat_size=8,
            st_dev=1,
            st_ino=1,
            mtime_ns=1,
            payload=ok_payload,
            payload_hash="ok",
            cursor_fingerprint="base",
            bytes_read=len(ok_payload),
            native_id_hint="append-ok",
            acquisition_native_id_hint="append-ok",
        ),
        _AppendPlan(
            path=tmp_path / "bad.jsonl",
            canonical_source_path=str(tmp_path / "bad.jsonl"),
            captured_profile_key=None,
            source_name="unknown",
            start_offset=0,
            last_complete_newline=9,
            stat_size=9,
            st_dev=1,
            st_ino=2,
            mtime_ns=1,
            payload=bad_payload,
            payload_hash="bad",
            cursor_fingerprint="base",
            bytes_read=len(bad_payload),
        ),
    ]

    owner = Owner()
    result = ingest_append_with_owner(owner, plans)

    assert result.succeeded == []
    assert result.deferred == [plans[0]]
    assert result.failed == [plans[1]]
    assert result.worker_count == 1
    # The identity-less plan is refused before acquisition writes a raw; the
    # bound plan's raw is retained and awaits its quarantined authority.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute("SELECT parse_error, revision_authority FROM raw_sessions").fetchall()
        assert rows == [(None, "quarantined")]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize("protect_chain", [True, False], ids=["protected", "protection-disabled"])
def test_live_append_chain_survives_post_ingest_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protect_chain: bool,
) -> None:
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "append-v1.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"append-v1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    path.write_bytes(payload)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    bootstrap_archive_root(tmp_path)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    original_publish = ArchiveBlobPublisher.write_from_bytes
    original_writer_publish = ArchiveBlobPublisher.write_from_writer
    published_payloads: list[bytes] = []

    def counted_publish(publisher: ArchiveBlobPublisher, raw: bytes) -> tuple[str, int]:
        published_payloads.append(raw)
        return original_publish(publisher, raw)

    def counted_writer_publish(
        publisher: ArchiveBlobPublisher, write: Callable[[IO[bytes]], None], **kwargs: Any
    ) -> tuple[str, int]:
        # Full captures stream through the acquisition boundary's writer.
        captured = io.BytesIO()
        write(captured)
        published_payloads.append(captured.getvalue())

        def replay(sink: IO[bytes]) -> None:
            sink.write(captured.getvalue())

        return original_writer_publish(publisher, replay, **kwargs)

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_bytes", counted_publish)
    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_writer", counted_writer_publish)
    if not protect_chain:
        from polylogue.storage.raw_retention import RawRetentionAuthority

        def unsafe_retention_authority(conn: sqlite3.Connection, **_kwargs: object) -> RawRetentionAuthority:
            raw_ids = frozenset(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions"))
            return RawRetentionAuthority(protected_raw_ids=frozenset(), eligible_raw_ids=raw_ids)

        monkeypatch.setattr(
            "polylogue.storage.raw_retention.active_raw_retention_authority",
            unsafe_retention_authority,
        )

    append_chunks = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n',
        b'{"type":"response_item","payload":{"type":"message","id":"message-2","role":"user",'
        b'"content":[{"type":"input_text","text":"two"}]}}\n',
        b'{"type":"response_item","payload":{"type":"message","id":"message-3","role":"assistant",'
        b'"content":[{"type":"output_text","text":"three"}]}}\n',
    )
    results = [run_ingest_files(processor, [path])]
    # The live compactor intentionally considers only raws older than the
    # process-start frontier. Move that frontier beyond this synthetic chain
    # so the test actually exercises retention authority rather than passing
    # because every raw is too new to compact.
    processor._raw_compaction_min_acquired_at = "9999-01-01T00:00:00+00:00"
    for chunk in append_chunks:
        with path.open("ab") as handle:
            handle.write(chunk)
        results.append(run_ingest_files(processor, [path]))

    assert results[0].full_file_count == 1
    assert results[0].succeeded_file_count == 1
    assert all(result.append_file_count == 1 for result in results[1:])
    # polylogue-u19l: append payloads are now published as literal live-file
    # bytes -- no synthetic session_meta header spliced in ahead of them.
    assert published_payloads == [payload, *append_chunks]
    if not protect_chain:
        # Accepted index receipts make later appends independent of whether
        # the compactor currently retains their predecessor payloads.
        assert all(result.succeeded_file_count == 1 for result in results)
        assert all(result.failed_file_count == 0 for result in results)
        cursor_record = cursor.get_record(path)
        assert cursor_record is not None
        assert cursor_record.byte_offset == len(payload) + sum(len(chunk) for chunk in append_chunks)
        assert cursor_record.failure_count == 0
        with sqlite3.connect(index_db) as conn:
            assert {str(row[0]) for row in conn.execute("SELECT native_id FROM messages")} == {
                "message-0",
                "message-1",
                "message-2",
                "message-3",
            }
            assert conn.execute("SELECT COUNT(DISTINCT raw_id) FROM raw_revision_applications").fetchone() == (4,)
        return

    assert all(result.succeeded_file_count == 1 for result in results)
    assert all(result.failed_file_count == 0 for result in results)
    expected_sessions = parse_payload(
        Provider.CODEX,
        [json.loads(line) for line in path.read_bytes().splitlines()],
        path.stem,
        source_path=str(path),
    )
    assert len(expected_sessions) == 1
    with sqlite3.connect(source_db) as conn:
        raw_rows = conn.execute(
            """SELECT raw_id, revision_kind, predecessor_raw_id, baseline_raw_id,
                      parsed_at_ms, parse_error, file_mtime_ms
               FROM raw_sessions ORDER BY acquisition_generation"""
        ).fetchall()
        assert len(raw_rows) == 4
        assert [row[1] for row in raw_rows] == ["full", "append", "append", "append"]
        assert all(row[4] is not None and row[5] is None for row in raw_rows)
        raw_by_id = {str(row[0]): row for row in raw_rows}
    final_file_mtime = datetime.fromtimestamp(raw_rows[-1][6] / 1000, UTC).isoformat()
    from polylogue.sources.assembly import get_assembly_spec

    codex_assembly = get_assembly_spec(Provider.CODEX)
    assert codex_assembly is not None
    # Live intake publishes the retained-replay interpretation: provider
    # assembly titles the session from its first authored prompt.
    expected_session = codex_assembly.enrich_session(expected_sessions[0], {})
    expected_session = normalize_session_timestamps(expected_session, fallback_timestamp=final_file_mtime)
    expected_session = expected_session.model_copy(
        update={"updated_at": final_file_mtime, "updated_at_provenance": "fallback"}
    )
    expected_session_hash = bytes.fromhex(session_content_hash(expected_session))
    with sqlite3.connect(index_db) as conn:
        session_native_id, session_hash = conn.execute("SELECT native_id, content_hash FROM sessions").fetchone()
        assert session_native_id == "append-v1"
        assert conn.execute("SELECT title FROM sessions").fetchone()[0] == "zero"
        assert {str(row[0]) for row in conn.execute("SELECT native_id FROM messages")} == {
            "message-0",
            "message-1",
            "message-2",
            "message-3",
        }
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 4
        assert conn.execute(
            """SELECT b.search_text
               FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid
               ORDER BY b.message_id"""
        ).fetchall() == [("zero",), ("one",), ("two",), ("three",)]
        head_raw_id, accepted_hash = conn.execute(
            "SELECT accepted_raw_id, accepted_content_hash FROM raw_revision_heads"
        ).fetchone()
        head_raw_id = str(head_raw_id)
        assert session_hash == accepted_hash
        assert session_hash == expected_session_hash
        assert conn.execute("SELECT COUNT(DISTINCT raw_id) FROM raw_revision_applications").fetchone()[0] == 4
        assert conn.execute(
            """SELECT decision, COUNT(DISTINCT raw_id)
               FROM raw_revision_applications GROUP BY decision ORDER BY decision"""
        ).fetchall() == [("applied_append", 3), ("selected_baseline", 1)]
        receipt_decisions = {
            str(raw_id): str(decision)
            for raw_id, decision in conn.execute("SELECT DISTINCT raw_id, decision FROM raw_revision_applications")
        }
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone() is None
    chain: list[str] = []
    current_raw_id: str | None = head_raw_id
    while current_raw_id is not None:
        chain.append(current_raw_id)
        row = raw_by_id[current_raw_id]
        current_raw_id = str(row[2]) if row[2] is not None else None
    assert len(chain) == 4
    assert raw_by_id[chain[-1]][1] == "full"
    assert receipt_decisions == {
        chain[-1]: "selected_baseline",
        **dict.fromkeys(chain[:-1], "applied_append"),
    }
    cursor_record = cursor.get_record(path)
    assert cursor_record is not None
    assert cursor_record.byte_offset == len(payload) + sum(len(chunk) for chunk in append_chunks)


def test_append_ingest_proves_byte_authority_at_capture_without_reconciler(tmp_path: Path) -> None:
    """polylogue-ds4b4 item 1: the common append case must prove itself at
    capture time, never deferring to the batch ``RawAuthorityReconciler``.

    ``append_ingest.py``'s ``ingest_append_plans`` already resolves
    the byte-contiguous predecessor via ``raw_append_revision_parent`` and,
    once found, immediately classifies+applies the revision in the SAME
    ingest call (``archive.classify_raw_revision_cohort`` /
    ``apply_raw_revision_replay``) -- there is no code path where a normal,
    single-predecessor append is left ``quarantined`` for a later async pass
    to pick up. This test locks that invariant in: after one full capture
    followed by one ordinary append, (a) the append's own raw row is
    ``revision_authority='byte_proven'`` immediately, (b) the append's
    content is already visible in ``index.db`` (``sessions``/``messages``)
    before any convergence/reconciler pass has run, and (c) the heavier,
    batch-oriented ``raw_authority_blockers`` ledger (owned by
    ``RawAuthorityReconciler``, not this synchronous per-key classifier) has
    zero rows -- proving the reconciler was never invoked for this raw.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "capture-proof.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"capture-proof","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    bootstrap_archive_root(tmp_path)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    baseline_result = run_ingest_files(processor, [path])
    assert baseline_result.succeeded_file_count == 1

    append_chunk = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n'
    )
    with path.open("ab") as handle:
        handle.write(append_chunk)
    append_result = run_ingest_files(processor, [path])

    assert append_result.append_file_count == 1
    assert append_result.succeeded_file_count == 1
    assert append_result.failed_file_count == 0

    with sqlite3.connect(source_db) as conn:
        append_row = conn.execute(
            "SELECT revision_kind, revision_authority, parsed_at_ms, parse_error "
            "FROM raw_sessions WHERE revision_kind = 'append'"
        ).fetchone()
        assert append_row is not None
        assert append_row[0] == "append"
        # Proven synchronously, at capture time -- not left 'quarantined'
        # for a later reconciler pass to resolve.
        assert append_row[1] == "byte_proven"
        assert append_row[2] is not None
        assert append_row[3] is None
        census_rows = conn.execute(
            "SELECT revision_kind, parser_fingerprint, status, logical_keys_json "
            "FROM raw_sessions JOIN raw_authority_parser_census USING (raw_id) "
            "WHERE revision_kind IN ('full', 'append') ORDER BY revision_kind"
        ).fetchall()
        assert census_rows == [
            ("append", raw_authority_parser_fingerprint(), "complete", '["codex-session:capture-proof"]'),
            ("full", raw_authority_parser_fingerprint(), "complete", '["codex-session:capture-proof"]'),
        ]
        # The durable frontier blocker ledger belongs to the separate, async
        # RawAuthorityReconciler (daemon convergence / offline backfill). A
        # normal single-predecessor append must never touch it.
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers").fetchone()[0] == 0

    with sqlite3.connect(index_db) as conn:
        assert {str(row[0]) for row in conn.execute("SELECT native_id FROM messages")} == {
            "message-0",
            "message-1",
        }


def test_full_ingest_cursor_hands_off_captured_prefix_after_growth_during_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "hot.jsonl"
    captured = (
        b'{"type":"session_meta","payload":{"id":"hot-growth","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"captured","role":"user",'
        b'"content":[{"type":"input_text","text":"captured"}]}}\n'
    )
    appended_during_parse = (
        b'{"type":"response_item","payload":{"type":"message","id":"later","role":"assistant",'
        b'"content":[{"type":"output_text","text":"later"}]}}\n'
    )
    path.write_bytes(captured)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    grew_during_prefix_proof = False
    original_hash = sha256_range_from_path

    def grow_during_prefix_proof(
        source_path: Path,
        *,
        start_offset: int,
        end_offset: int,
    ) -> tuple[str, int]:
        nonlocal grew_during_prefix_proof
        result = original_hash(source_path, start_offset=start_offset, end_offset=end_offset)
        if not grew_during_prefix_proof:
            with path.open("ab") as handle:
                handle.write(appended_during_parse)
            grew_during_prefix_proof = True
        return result

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", grow_during_prefix_proof)
    # This drives the re-hash proof racing growth; a slow host must not let the
    # capture observation settle and skip that proof.
    monkeypatch.setattr(live_batch, "_SETTLED_OBSERVATION_MARGIN_NS", 1 << 62)

    first = run_ingest_files(processor, [path])

    assert first.full_file_count == 1
    assert first.succeeded_file_count == 1
    record = cursor.get_record(path)
    assert record is not None
    assert record.byte_size == len(captured)
    assert record.byte_offset == len(captured)
    assert record.last_complete_newline == len(captured)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(captured)
    assert plan.payload.endswith(appended_during_parse)

    second = run_ingest_files(processor, [path])

    assert second.append_file_count == 1
    assert second.succeeded_file_count == 1
    final_record = cursor.get_record(path)
    assert final_record is not None
    assert final_record.byte_offset == len(captured) + len(appended_during_parse)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        full_blob_size, append_start, append_blob_hash = conn.execute(
            """SELECT
                   MAX(CASE WHEN revision_kind = 'full' THEN blob_size END),
                   MAX(CASE WHEN revision_kind = 'append' THEN append_start_offset END),
                   MAX(CASE WHEN revision_kind = 'append' THEN hex(blob_hash) END)
               FROM raw_sessions"""
        ).fetchone()
    assert full_blob_size == len(captured)
    assert append_start == len(captured)
    assert isinstance(append_blob_hash, str)
    from polylogue.storage.blob_store import BlobStore

    # polylogue-u19l: append payloads are stored as literal live-file bytes --
    # no synthetic session_meta header spliced in ahead of them.
    assert BlobStore(tmp_path / "blob").read_all(append_blob_hash.lower()) == appended_during_parse
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("captured",),
            ("later",),
        ]
        assert conn.execute(
            "SELECT b.search_text FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid ORDER BY b.message_id"
        ).fetchall() == [("captured",), ("later",)]
        session_hash = conn.execute("SELECT content_hash FROM sessions").fetchone()[0]
        accepted_hash = conn.execute("SELECT accepted_content_hash FROM raw_revision_heads").fetchone()[0]
        assert session_hash == accepted_hash


def test_busy_full_prefix_proof_defers_to_archived_cursor_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "busy-hot.jsonl"
    captured = (
        b'{"type":"session_meta","payload":{"id":"busy-hot","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"captured","role":"user",'
        b'"content":[{"type":"input_text","text":"captured"}]}}\n'
    )
    path.write_bytes(captured)
    captured_stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    cursor.set(
        path,
        len(captured),
        byte_offset=len(captured),
        last_complete_newline=len(captured),
        parser_fingerprint="previous-parser",
        content_fingerprint=sha256(captured).hexdigest(),
        tail_hash=_cursor_hash_authority(captured),
        source_name="codex",
        st_dev=captured_stat.st_dev,
        st_ino=captured_stat.st_ino,
        mtime_ns=captured_stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    original_hash = sha256_range_from_path
    next_record = 0

    def grow_on_every_prefix_proof(
        source_path: Path,
        *,
        start_offset: int,
        end_offset: int,
    ) -> tuple[str, int]:
        nonlocal next_record
        result = original_hash(source_path, start_offset=start_offset, end_offset=end_offset)
        with path.open("ab") as handle:
            handle.write(
                b'{"type":"response_item","payload":{"type":"message","id":"later-'
                + str(next_record).encode()
                + b'","role":"assistant","content":[{"type":"output_text","text":"later"}]}}\n'
            )
        next_record += 1
        return result

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", grow_on_every_prefix_proof)
    # This drives the re-hash proof racing growth; a slow host must not let the
    # capture observation settle and skip that proof.
    monkeypatch.setattr(live_batch, "_SETTLED_OBSERVATION_MARGIN_NS", 1 << 62)
    first = run_ingest_files(processor, [path])

    assert first.full_file_count == 1
    assert first.succeeded_file_count == 1
    deferred = cursor.get_record(path)
    assert deferred is not None
    assert deferred.byte_offset == 0
    assert deferred.content_fingerprint is None
    assert deferred.failure_count == 0
    assert deferred.next_retry_at is not None
    assert not deferred.excluded

    for _ in range(4):
        processor._record_full_cursor(
            path,
            raw_fingerprint=sha256(captured).hexdigest(),
            raw_byte_size=len(captured),
            source_name="codex",
            captured_content_hash=sha256(captured).hexdigest(),
            captured_file_observation=(
                captured_stat.st_dev,
                captured_stat.st_ino,
                captured_stat.st_size,
                captured_stat.st_mtime_ns,
                captured_stat.st_ctime_ns,
            ),
        )
    repeatedly_deferred = cursor.get_record(path)
    assert repeatedly_deferred is not None
    assert repeatedly_deferred.failure_count == 0
    assert repeatedly_deferred.next_retry_at is not None
    assert not repeatedly_deferred.excluded

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", original_hash)
    watcher = LiveWatcher(polylogue, (WatchSource(name="codex", root=root),), cursor=cursor)
    # The deferred observation carries its own retry time; the dispatcher
    # re-offers the file on a later pass and the selection below decides.
    original_reconcile = watcher._reconcile_archived_cursor_outcome
    monkeypatch.setattr(
        watcher,
        "_reconcile_archived_cursor_outcome",
        lambda _path, *, stat, expected: live_watcher._ArchivedCursorReconciliation.UNAVAILABLE,
    )
    monkeypatch.setattr(live_watcher, "_retry_due", lambda _retry_at: True)
    for _ in range(5):
        assert not watcher._needs_work(path)
    unavailable = cursor.get_record(path)
    assert unavailable is not None
    assert unavailable.failure_count == 0
    assert not unavailable.excluded

    monkeypatch.setattr(watcher, "_reconcile_archived_cursor_outcome", original_reconcile)
    assert watcher._needs_work(path)
    reconciled = cursor.get_record(path)
    assert reconciled is not None
    assert reconciled.byte_offset == len(captured)
    assert reconciled.content_fingerprint == sha256(captured).hexdigest()

    second = run_ingest_files(processor, [path])

    assert second.full_file_count == 0
    assert second.append_file_count == 1
    assert second.succeeded_file_count == 1

    processor._defer_full_cursor_retry(path, source_name="codex", stat=path.stat())
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real message
    # record so the third ingest below still succeeds.
    replacement = (
        b'{"type":"session_meta","payload":{"id":"busy-replacement"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    path.write_bytes(replacement)

    assert watcher._needs_work(path)
    invalidated = cursor.get_record(path)
    assert invalidated is not None
    assert invalidated.byte_offset == 0
    assert invalidated.content_fingerprint is None
    assert invalidated.next_retry_at is None

    third = run_ingest_files(processor, [path])
    assert third.full_file_count == 1
    assert third.append_file_count == 0
    assert third.succeeded_file_count == 1


@pytest.mark.parametrize("replacement_mode", ["atomic", "in-place"])
def test_full_ingest_does_not_advance_cursor_across_same_size_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_mode: str,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "replaced.jsonl"
    replacement = root / "replacement.jsonl"
    payload_a = (
        b'{"type":"session_meta","payload":{"id":"atomic-replace-a"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-a","role":"user",'
        b'"content":[{"type":"input_text","text":"alpha"}]}}\n'
    )
    payload_b = (
        b'{"type":"session_meta","payload":{"id":"atomic-replace-b"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-b","role":"user",'
        b'"content":[{"type":"input_text","text":"bravo"}]}}\n'
    )
    assert len(payload_a) == len(payload_b)
    path.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    original_stat = path.stat()
    original_identity = (original_stat.st_dev, original_stat.st_ino)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    replaced = False

    def replace_after_acquisition(
        paths: list[Path],
        **kwargs: object,
    ) -> tuple[set[Path], float, dict[str, float], list[object], list[object]]:
        del kwargs
        nonlocal replaced
        if not replaced:
            if replacement_mode == "atomic":
                replacement.replace(path)
            else:
                path.write_bytes(payload_b)
                current_stat = path.stat()
                os.utime(
                    path,
                    ns=(current_stat.st_atime_ns, max(current_stat.st_mtime_ns, original_stat.st_mtime_ns) + 1_000_000),
                )
            replaced = True
        return set(paths), 0.0, {}, [], []

    monkeypatch.setattr(processor, "_converge_paths", replace_after_acquisition)

    first = run_ingest_files(processor, [path])

    assert first.succeeded_file_count == 1
    assert first.stale_cursor_write_count == 1
    assert first.stale_cursor_paths == (str(path),)
    if replacement_mode == "atomic":
        assert (path.stat().st_dev, path.stat().st_ino) != original_identity
    else:
        assert (path.stat().st_dev, path.stat().st_ino) == original_identity
    stale_cursor = cursor.get_record(path)
    assert stale_cursor is not None
    assert stale_cursor.byte_offset == 0
    assert stale_cursor.content_fingerprint is None
    assert LiveWatcher(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )._needs_work(path)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages").fetchall() == [("message-a",)]

    second = run_ingest_files(processor, [path])

    assert second.full_file_count == 1
    assert second.succeeded_file_count == 1
    assert second.stale_cursor_write_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == len(payload_b)
    assert (final_cursor.st_dev, final_cursor.st_ino) == (path.stat().st_dev, path.stat().st_ino)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [
            ("message-a",),
            ("message-b",),
        ]
        assert conn.execute("SELECT search_text FROM blocks ORDER BY search_text").fetchall() == [
            ("alpha",),
            ("bravo",),
        ]


def test_archive_cursor_reconciliation_rejects_restored_mtime_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "archive-reconcile.jsonl"
    # polylogue-9ykn: a session_meta-only stream carries no positive
    # conversational evidence and is refused -- append one real,
    # equal-length message record to each payload so this fixture keeps
    # testing the mtime-restore-reconciliation race, not the now-refused
    # empty shape (the equal-length invariant below is load-bearing for the
    # race itself, so both messages must stay identical length too).
    payload_a = (
        b'{"type":"session_meta","payload":{"id":"archive-reconcile-a"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    payload_b = (
        b'{"type":"session_meta","payload":{"id":"archive-reconcile-b"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n'
    )
    assert len(payload_a) == len(payload_b)
    path.write_bytes(payload_a)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert run_ingest_files(processor, [path]).succeeded_file_count == 1
    with sqlite3.connect(cursor._ops_db_path) as conn:
        conn.execute("DELETE FROM ingest_cursor WHERE source_path = ?", (str(path),))
        conn.commit()
    watcher = LiveWatcher(polylogue, (WatchSource(name="codex", root=root),), cursor=cursor)
    initial_stat = path.stat()
    original_hash = sha256_range_from_path

    def rewrite_after_hash(
        source_path: Path,
        *,
        start_offset: int,
        end_offset: int,
    ) -> tuple[str, int]:
        result = original_hash(source_path, start_offset=start_offset, end_offset=end_offset)
        path.write_bytes(payload_b)
        rewritten_stat = path.stat()
        os.utime(path, ns=(rewritten_stat.st_atime_ns, initial_stat.st_mtime_ns))
        return result

    monkeypatch.setattr(live_watcher, "sha256_range_from_path", rewrite_after_hash)

    assert not watcher._reconcile_archived_cursor(path, stat=initial_stat, expected=watcher._cursor.get_record(path))
    assert cursor.get_record(path) is None
    final_stat = path.stat()
    assert final_stat.st_mtime_ns == initial_stat.st_mtime_ns
    assert final_stat.st_ctime_ns != initial_stat.st_ctime_ns


def test_rejected_full_cursor_frontier_requires_reauthorization(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rejected-frontier.jsonl"
    captured = (
        b'{"type":"session_meta","payload":{"id":"rejected-frontier"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-a","role":"user",'
        b'"content":[{"type":"input_text","text":"alpha"}]}}\n'
    )
    growth = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-b","role":"assistant",'
        b'"content":[{"type":"output_text","text":"bravo"}]}}\n'
    )
    path.write_bytes(captured)
    captured_stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    obsolete_offset = len(captured) + 1
    cursor.set(
        path,
        obsolete_offset,
        byte_offset=obsolete_offset,
        last_complete_newline=obsolete_offset,
        parser_fingerprint="test-parser",
        content_fingerprint="obsolete-frontier",
        tail_hash="obsolete-tail",
        source_name="codex",
        st_dev=captured_stat.st_dev,
        st_ino=captured_stat.st_ino,
        mtime_ns=captured_stat.st_mtime_ns,
        failure_count=2,
        authority=fixture_cursor_authority(path),
    )
    with path.open("ab") as handle:
        handle.write(growth)

    processor._record_full_cursor(
        path,
        raw_fingerprint=sha256(captured).hexdigest(),
        raw_byte_size=len(captured),
        source_name="codex",
        captured_content_hash=sha256(captured).hexdigest(),
        captured_file_observation=(
            captured_stat.st_dev,
            captured_stat.st_ino,
            captured_stat.st_size,
            captured_stat.st_mtime_ns,
            captured_stat.st_ctime_ns,
        ),
    )

    assert processor._last_cursor_write_stale is True
    invalidated = cursor.get_record(path)
    assert invalidated is not None
    assert invalidated.byte_offset == 0
    assert invalidated.content_fingerprint is None
    assert invalidated.failure_count == 2
    assert LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )._needs_work(path)
    assert processor._append_plan(path, cursor=invalidated) is None


def test_cursor_invalidation_lock_exhaustion_is_observable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "locked-invalidation.jsonl"
    path.write_bytes(b'{"type":"session_meta","payload":{"id":"locked-invalidation"}}\n')
    stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        parser_fingerprint="test-parser",
        content_fingerprint="accepted-frontier",
        tail_hash="accepted-tail",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=2,
        authority=fixture_cursor_authority(path),
    )
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(cursor, "_sync_cursor_record_to_ops", lambda _record: False)

    with pytest.raises(sqlite3.OperationalError, match="failed to persist cursor invalidation"):
        processor._invalidate_cursor_for_full_retry(path, source_name="codex", stat=stat)

    unchanged = cursor.get_record(path)
    assert unchanged is not None
    assert unchanged.byte_offset == stat.st_size
    assert unchanged.content_fingerprint == "accepted-frontier"
    assert unchanged.failure_count == 2


def test_append_plan_rejects_malformed_hash_authority(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "malformed-authority.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"malformed-authority"}}\n'
    path.write_bytes(original + b'{"type":"turn_context","payload":{}}\n')
    stat = path.stat()
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="accepted-frontier",
        tail_hash=f"sha256-prefix-v1:{sha256(original).hexdigest()}:invalid:0",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    record = cursor.get_record(path)
    assert record is not None
    assert processor._append_plan(path, cursor=record) is None


@pytest.mark.parametrize("rewrite_mode", ["atomic-replacement", "in-place-prefix"])
def test_append_cursor_redetects_source_rewrite_after_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rewrite_mode: str,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "append-replaced.jsonl"
    replacement = root / "append-replacement.jsonl"
    prefix_padding = b"p" * (70 * 1024)
    baseline_a = (
        b'{"type":"session_meta","payload":{"id":"append-replace-a"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0a","role":"user",'
        b'"content":[{"type":"input_text","text":"zeroa' + prefix_padding + b'"}]}}\n'
    )
    append_a = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-aa","role":"assistant",'
        b'"content":[{"type":"output_text","text":"alpha"}]}}\n'
    )
    replacement_b = (
        b'{"type":"session_meta","payload":{"id":"append-replace-b"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0b","role":"user",'
        b'"content":[{"type":"input_text","text":"zerob' + prefix_padding + b'"}]}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-bb","role":"assistant",'
        b'"content":[{"type":"output_text","text":"bravo"}]}}\n'
    )
    assert len(baseline_a + append_a) == len(replacement_b)
    assert len(baseline_a) > 64 * 1024
    path.write_bytes(baseline_a)
    replacement.write_bytes(replacement_b)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert run_ingest_files(processor, [path]).succeeded_file_count == 1
    with path.open("ab") as handle:
        handle.write(append_a)
    pre_rewrite_stat = path.stat()
    replaced = False

    def replace_after_append(
        paths: list[Path],
        **kwargs: object,
    ) -> tuple[set[Path], float, dict[str, float], list[object], list[object]]:
        del kwargs
        nonlocal replaced
        if not replaced:
            if rewrite_mode == "atomic-replacement":
                replacement.replace(path)
            else:
                rewritten = path.read_bytes().replace(b"zeroa", b"zerob", 1)
                assert len(rewritten) == pre_rewrite_stat.st_size
                path.write_bytes(rewritten)
                current_stat = path.stat()
                os.utime(
                    path,
                    ns=(
                        current_stat.st_atime_ns,
                        pre_rewrite_stat.st_mtime_ns,
                    ),
                )
                restored_stat = path.stat()
                assert restored_stat.st_mtime_ns == pre_rewrite_stat.st_mtime_ns
                assert restored_stat.st_ctime_ns != pre_rewrite_stat.st_ctime_ns
            replaced = True
        return set(paths), 0.0, {}, [], []

    monkeypatch.setattr(processor, "_converge_paths", replace_after_append)

    appended = run_ingest_files(processor, [path])

    assert appended.append_file_count == 1
    assert appended.succeeded_file_count == 1
    stale_cursor = cursor.get_record(path)
    assert stale_cursor is not None
    watcher = LiveWatcher(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )
    assert watcher._needs_work(path)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [
            ("message-0a",),
            ("message-aa",),
        ]
        assert conn.execute("SELECT substr(search_text, 1, 5) FROM blocks ORDER BY search_text").fetchall() == [
            ("alpha",),
            ("zeroa",),
        ]

    # Exact prefix proof rejects every rewrite before publication.
    assert appended.stale_cursor_write_count == 1
    assert stale_cursor.byte_offset == 0
    assert stale_cursor.content_fingerprint is None

    retried = run_ingest_files(processor, [path])

    assert retried.full_file_count == 1
    if rewrite_mode == "in-place-prefix":
        assert retried.append_file_count == 0
        assert retried.succeeded_file_count == 1
        assert retried.failed_file_count == 0
        assert retried.stale_cursor_write_count == 0
        retried_cursor = cursor.get_record(path)
        assert retried_cursor is not None
        assert retried_cursor.byte_offset == path.stat().st_size
        return
    assert retried.succeeded_file_count == 1
    assert retried.stale_cursor_write_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == len(replacement_b)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [
            ("message-0a",),
            ("message-0b",),
            ("message-aa",),
            ("message-bb",),
        ]


def test_append_cursor_rejects_truncation_after_append_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial append handoff must fail closed when its source truncates."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "append-truncated.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"append-truncated"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    append = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n'
    )
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert run_ingest_files(processor, [path]).succeeded_file_count == 1
    with path.open("ab") as handle:
        handle.write(append)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)

    original_tail_hash = tail_hash_from_path

    def truncate_after_tail(source_path: Path, byte_size: int) -> tuple[str, int]:
        result = original_tail_hash(source_path, byte_size)
        source_path.write_bytes(source_path.read_bytes()[: plan.last_complete_newline - 1])
        return result

    monkeypatch.setattr("polylogue.sources.live.batch.tail_hash_from_path", truncate_after_tail)

    assert processor._record_append_cursor(plan) is False
    invalidated = cursor.get_record(path)
    assert invalidated is not None
    assert invalidated.byte_offset == 0
    assert invalidated.content_fingerprint is None
    assert LiveWatcher(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )._needs_work(path)


def test_rewrite_plus_growth_before_planning_fails_closed_to_full_route(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rewrite-before-plan.jsonl"
    padding = b"p" * (70 * 1024)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"rewrite-before-plan"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zeroa' + padding + b'"}]}}\n'
    )
    appended = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"alpha"}]}}\n'
    )
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    polylogue = cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db)))
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert run_ingest_files(processor, [path]).succeeded_file_count == 1
    rewritten = baseline.replace(b"zeroa", b"zerob", 1)
    assert rewritten[-64 * 1024 :] == baseline[-64 * 1024 :]
    path.write_bytes(rewritten + appended)

    second = run_ingest_files(processor, [path])

    assert second.full_file_count == 1
    assert second.append_file_count == 0
    # The full route retains the competing bytes and records a typed deferred
    # frontier carrier; source acquisition is successful even though replay
    # remains pending until a later observation can order the revisions.
    assert second.succeeded_file_count == 1
    assert second.failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY native_id").fetchall() == [("message-0",)]
        assert conn.execute("SELECT substr(search_text, 1, 5) FROM blocks ORDER BY search_text").fetchall() == [
            ("zeroa",),
        ]
    retained_cursor = cursor.get_record(path)
    assert retained_cursor is not None
    assert retained_cursor.byte_offset == len(rewritten + appended)
    assert retained_cursor.failure_count == 0
    assert retained_cursor.next_retry_at is None
    with sqlite3.connect(tmp_path / "source.db") as conn:
        retained = conn.execute(
            """
            SELECT r.blob_hash, r.blob_size
            FROM raw_sessions AS r
            JOIN raw_artifacts AS a USING (raw_id)
            WHERE r.source_path = ? AND a.artifact_kind = ?
            """,
            (str(path), RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value),
        ).fetchone()
    assert retained is not None
    assert retained[1] == len(rewritten + appended)
    assert BlobStore(tmp_path / "blob").read_all(bytes(retained[0]).hex()) == rewritten + appended
    # The fork leaves the byte head accepted and parsed; its converted
    # membership row is decided, never left pending.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT m.decision, r.revision_authority, r.parsed_at_ms IS NOT NULL
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            ORDER BY m.decision
            """
        ).fetchall() == [("applied", "byte_proven", 1), ("deferred", "quarantined", 0)]


def test_incomplete_full_jsonl_capture_retries_without_losing_split_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "split-record.jsonl"
    prefix = (
        b'{"type":"session_meta","payload":{"id":"split-record"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    split_record = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}'
    )
    split_at = len(split_record) // 2
    path.write_bytes(prefix + split_record[:split_at])
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    first = run_ingest_files(processor, [path])

    assert first.full_file_count == 1
    # Retaining and classifying the captured raw bytes is a successful source
    # write, even though the incomplete session is terminally unmaterialized.
    assert first.succeeded_file_count == 1
    assert first.failed_file_count == 0
    captured_cursor = cursor.get_record(path)
    assert captured_cursor is not None
    # The cursor records the full observed size but its append frontier stops
    # at the last proven complete record.
    assert captured_cursor.byte_offset == len(prefix)
    assert captured_cursor.deferred_end_offset == path.stat().st_size
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)
    # A stable truncated capture is a typed partial admission (xf8qp): every
    # complete record is admitted, the unterminated tail is reported, and the
    # raw carries no failure evidence.
    assert first.refused_bytes_by_reason == {"truncated_tail": split_at}
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parse_error FROM raw_sessions").fetchall() == [(None,)]
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    assert not processor._cursor_references_raw_failure_requiring_full_replay(path, captured_cursor)

    # The writer finishes the split record and its line; a partial capture
    # replays through the full route, never an append onto its prefix.
    with path.open("ab") as handle:
        handle.write(split_record[split_at:] + b"\n")
    second = run_ingest_files(processor, [path])

    assert second.full_file_count == 1
    assert second.append_file_count == 0
    assert second.succeeded_file_count == 1
    assert second.failed_file_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.failure_count == 0
    # This used to spy on ``CursorStore.record_convergence_debt`` to prove the
    # full-ingest route recorded a deferred-FTS debt row. #5027 deleted that
    # call site: pending is now ``required - valid`` derived from the durable
    # relation itself (``storage/fts/derivation.py`` enumerates every
    # ``sessions``/``blocks`` session id), so a deferred partition is pending by
    # construction and the row was redundant bookkeeping. No debt is left for a
    # daemon to retry, and the assertions below carry what actually matters --
    # the recovered tail landed as both messages and is FTS-repairable.
    assert cursor.list_convergence_debt(limit=10) == []
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("message-0",),
            ("message-1",),
        ]
        from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync

        repair_message_fts_index_sync(conn, ["codex-session:split-record"])
        assert conn.execute(
            "SELECT b.search_text FROM messages_fts AS f JOIN blocks AS b ON b.rowid = f.rowid ORDER BY b.message_id"
        ).fetchall() == [("zero",), ("one",)]


def test_deferred_full_jsonl_with_prior_session_replays_completed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferred full capture cannot resume through an append-only tail."""
    from polylogue.sources.live import batch as live_batch

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "prior-session.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"prior-session"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    completed_record = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n'
    )
    split_at = len(completed_record) // 2
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="current-parser",
    )

    seeded = run_ingest_files(processor, [path])
    assert seeded.succeeded_file_count == 1
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [("message-0",)]

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="previous-parser",
    )
    path.write_bytes(baseline + completed_record[:split_at])
    original_boundary_check = live_batch._stable_truncated_tail_admission
    completed = False

    def complete_source_after_capture(record: Any) -> Any:
        nonlocal completed
        if not completed:
            path.write_bytes(baseline + completed_record)
            completed = True
        return original_boundary_check(record)

    monkeypatch.setattr(live_batch, "_stable_truncated_tail_admission", complete_source_after_capture)

    deferred = run_ingest_files(processor, [path])

    assert deferred.full_file_count == 1
    assert deferred.succeeded_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    replayed = run_ingest_files(processor, [path])

    assert replayed.full_file_count == 1
    assert replayed.append_file_count == 0
    assert replayed.succeeded_file_count == 1
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.failure_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("message-0",),
            ("message-1",),
        ]


def test_raw_failure_cursor_guard_uses_root_source_tier_for_pointer_index(tmp_path: Path) -> None:
    """The active index generation never owns durable raw-failure evidence."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    generation = tmp_path / "generation"
    generation.mkdir()
    index_db = generation / "index.db"
    sqlite3.connect(index_db).close()
    (archive_root / ".index-active-pointer").write_text(str(index_db), encoding="utf-8")
    source_db = archive_root / "source.db"
    initialize_runtime_source_fixture(source_db)
    path = archive_root / "sessions" / "terminal.jsonl"
    path.parent.mkdir()
    path.write_bytes(b'{"type":"session_meta"')
    payload_hash = "ab" * 32
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parse_error, detection_warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "terminal-raw",
                "codex-session",
                "terminal",
                str(path),
                bytes.fromhex(payload_hash),
                path.stat().st_size,
                1_770_000_000_000,
                "captured JSONL payload ends before a complete record boundary",
                "[]",
            ),
        )
        upsert_raw_artifact(
            conn,
            "terminal-raw",
            ArchiveSourceArtifact(
                artifact_id="terminal-evidence",
                origin="codex-session",
                source_path=str(path),
                source_index=0,
                artifact_kind="terminal_corrupt_input",
                classification_reason="terminal_corrupt_input",
                support_status=ArtifactSupportStatus.DECODE_FAILED,
            ),
        )
    cursor = CursorStore(index_db)
    stat = path.stat()
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint="test-parser",
        content_fingerprint="terminal-raw",
        tail_hash=_cursor_hash_authority(path.read_bytes()),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=path.parent),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    record = cursor.get_record(path)

    assert record is not None
    assert processor._cursor_references_raw_failure_requiring_full_replay(path, record)


def test_raw_failure_cursor_guard_rejects_contradictory_or_mismatched_evidence(tmp_path: Path) -> None:
    """Append fallback requires the same source coordinate and valid support status."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "guard.jsonl"
    path.write_bytes(b'{"type":"session_meta","payload":{"id":"guard"}}\n')
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        mismatched_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=path.read_bytes(),
            source_path=str(path),
            canonical_source_path=str(path),
            source_index=1,
            acquired_at_ms=1,
        )
        contradictory_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=path.read_bytes() + b"2",
            source_path=str(path),
            canonical_source_path=str(path),
            source_index=0,
            acquired_at_ms=2,
        )
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.executemany(
            "UPDATE raw_sessions SET parse_error = ? WHERE raw_id = ?",
            [("mismatched coordinate", mismatched_raw_id), ("contradictory support", contradictory_raw_id)],
        )
        upsert_raw_artifact(
            source_conn,
            mismatched_raw_id,
            ArchiveSourceArtifact(
                artifact_id="mismatched-coordinate-evidence",
                origin="chatgpt-export",
                source_path=str(path),
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
            ),
        )
        upsert_raw_artifact(
            source_conn,
            contradictory_raw_id,
            ArchiveSourceArtifact(
                artifact_id="contradictory-support-evidence",
                origin="codex-session",
                source_path=str(path),
                source_index=0,
                artifact_kind="deferred_cas_frontier",
                classification_reason="deferred_cas_frontier",
                support_status=ArtifactSupportStatus.DECODE_FAILED,
            ),
        )
        source_conn.commit()
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    assert processor._raw_failure_requires_full_replay(path, mismatched_raw_id) is False
    assert processor._raw_failure_requires_full_replay(path, contradictory_raw_id) is False


def test_captured_incomplete_jsonl_is_rejected_after_source_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "disappearing.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"disappearing"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"complete"}]}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-1"'
    )
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    original_ingest = processor._acquire_full_records_archive

    def remove_source_after_capture(*args: Any, **kwargs: Any) -> _ArchiveFullWriteResult:
        path.unlink()
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(processor, "_acquire_full_records_archive", remove_source_after_capture)

    result = run_ingest_files(processor, [path])

    # The acquired bytes were durably retained with a terminal classification;
    # source disappearance cannot turn that completed archive write into a
    # retryable transport failure. Nothing of the file was admitted, so it is
    # a settled corrupt-input exclusion, not a success (polylogue-xf8qp).
    assert result.succeeded_file_count == 0
    assert result.failed_file_count == 0
    assert result.excluded_paths == {str(path): "corrupt_input"}
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        parse_error = conn.execute("SELECT parse_error FROM raw_sessions").fetchone()[0]
        artifact = conn.execute("SELECT artifact_kind, support_status, parse_as_session FROM raw_artifacts").fetchone()
        assert "complete record boundary" in str(parse_error)
    assert artifact == ("terminal_corrupt_input", "decode_failed", 0)


def test_append_persistence_failure_preserves_frontier_for_next_tick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "retry.jsonl"
    baseline = (
        b'{"type":"session_meta","payload":{"id":"retry"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    append = (
        b'{"type":"response_item","payload":{"type":"message","id":"message-1","role":"assistant",'
        b'"content":[{"type":"output_text","text":"one"}]}}\n'
    )
    path.write_bytes(baseline)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    assert run_ingest_files(processor, [path]).succeeded_file_count == 1
    accepted_cursor = cursor.get_record(path)
    assert accepted_cursor is not None

    def index_state() -> tuple[object, ...]:
        with sqlite3.connect(index_db) as conn:
            return (
                conn.execute(
                    "SELECT session_id, message_count, content_hash FROM sessions ORDER BY session_id"
                ).fetchall(),
                conn.execute("SELECT message_id, position, content_hash FROM messages ORDER BY message_id").fetchall(),
                conn.execute("SELECT block_id, message_id, search_text FROM blocks ORDER BY block_id").fetchall(),
                conn.execute("SELECT id, sz FROM messages_fts_docsize ORDER BY id").fetchall(),
                conn.execute(
                    """SELECT logical_source_key, accepted_raw_id, accepted_source_revision,
                              accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                              acquisition_generation, append_end_offset
                       FROM raw_revision_heads ORDER BY logical_source_key"""
                ).fetchall(),
                conn.execute(
                    """SELECT decision_id, raw_id, decision, accepted_raw_id,
                              accepted_source_revision, accepted_content_hash
                       FROM raw_revision_applications ORDER BY decision_id"""
                ).fetchall(),
            )

    accepted_index_state = index_state()
    with path.open("ab") as handle:
        handle.write(append)

    # polylogue-1r9c: record_revision_application_sync is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through archive_tier_module -- patch it there.
    original_record = archive_revision_governance.__dict__["record_revision_application_sync"]
    fail_once = True

    def injected_failure(*args: Any, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected append persistence failure")
        original_record(*args, **kwargs)

    monkeypatch.setattr(archive_revision_governance, "record_revision_application_sync", injected_failure)
    failed = run_ingest_files(processor, [path])

    assert failed.succeeded_file_count == 0
    assert failed.failed_file_count == 1
    retry_cursor = cursor.get_record(path)
    assert retry_cursor is not None
    assert (
        retry_cursor.byte_size,
        retry_cursor.byte_offset,
        retry_cursor.last_complete_newline,
        retry_cursor.parser_fingerprint,
        retry_cursor.content_fingerprint,
        retry_cursor.tail_hash,
        retry_cursor.source_name,
        retry_cursor.st_dev,
        retry_cursor.st_ino,
        retry_cursor.mtime_ns,
    ) == (
        accepted_cursor.byte_size,
        accepted_cursor.byte_offset,
        accepted_cursor.last_complete_newline,
        accepted_cursor.parser_fingerprint,
        accepted_cursor.content_fingerprint,
        accepted_cursor.tail_hash,
        accepted_cursor.source_name,
        accepted_cursor.st_dev,
        accepted_cursor.st_ino,
        accepted_cursor.mtime_ns,
    )
    assert index_state() == accepted_index_state
    with sqlite3.connect(tmp_path / "source.db") as conn:
        retained_append = conn.execute(
            """SELECT revision_kind, predecessor_raw_id, append_start_offset,
                      append_end_offset, revision_authority, parsed_at_ms, parse_error
               FROM raw_sessions WHERE source_index = -1"""
        ).fetchone()
    assert retained_append is not None
    assert retained_append[0] == "append"
    assert retained_append[1] is not None
    assert retained_append[2:5] == (len(baseline), len(baseline) + len(append), "byte_proven")
    assert retained_append[5] is None
    assert "injected append persistence failure" in str(retained_append[6])

    cursor.reset_failures(path)
    monkeypatch.setattr("polylogue.sources.live.watcher._PARSER_FINGERPRINT", "test-parser")
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
    )
    assert watcher._needs_work(path)
    cursor.mark_failed(path, authority=fixture_cursor_authority(path))
    pending_retry = cursor.get_record(path)
    assert pending_retry is not None
    assert pending_retry.failure_count == 1

    retried = run_ingest_files(processor, [path])

    assert retried.append_file_count == 1
    assert retried.succeeded_file_count == 1
    assert retried.failed_file_count == 0
    final_cursor = cursor.get_record(path)
    assert final_cursor is not None
    assert final_cursor.byte_offset == path.stat().st_size
    assert final_cursor.failure_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT native_id FROM messages ORDER BY position").fetchall() == [
            ("message-0",),
            ("message-1",),
        ]


def test_failed_parser_upgrade_preserves_accepted_parser_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "parser-upgrade.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"parser-upgrade"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor_a = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="parser-a",
    )
    assert run_ingest_files(processor_a, [path]).succeeded_file_count == 1
    accepted = cursor.get_record(path)
    assert accepted is not None
    assert accepted.parser_fingerprint == "parser-a"

    with path.open("ab") as handle:
        handle.write(
            b'{"type":"response_item","payload":{"type":"message","id":"message-1",'
            b'"role":"assistant","content":[{"type":"output_text","text":"one"}]}}\n'
        )
    bootstrap_archive_root(tmp_path)
    processor_b = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="parser-b",
    )
    # polylogue-1r9c: record_revision_application_sync is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through archive_tier_module -- patch it there.
    original_record = archive_revision_governance.__dict__["record_revision_application_sync"]
    fail_once = True

    def injected_failure(*args: Any, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected parser-upgrade persistence failure")
        original_record(*args, **kwargs)

    monkeypatch.setattr(archive_revision_governance, "record_revision_application_sync", injected_failure)

    failed = run_ingest_files(processor_b, [path])

    assert failed.full_file_count == 1
    assert failed.failed_file_count == 1
    retry = cursor.get_record(path)
    assert retry is not None
    assert retry.parser_fingerprint == "parser-a"
    assert retry.byte_offset == accepted.byte_offset
    assert retry.content_fingerprint == accepted.content_fingerprint

    cursor.reset_failures(path)
    retried = run_ingest_files(processor_b, [path])

    assert retried.full_file_count == 1
    assert retried.append_file_count == 0
    assert retried.succeeded_file_count == 1
    final = cursor.get_record(path)
    assert final is not None
    assert final.parser_fingerprint == "parser-b"
    assert final.byte_offset == path.stat().st_size


def test_append_parse_failure_retains_typed_raw_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An append parses only once it chains off an accepted full; seed one.
    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="append-bad")

    def failing_parse(*_args: object, **_kwargs: object) -> Generator[ParsedSession, None, None]:
        raise RuntimeError("injected append parse failure")
        yield  # pragma: no cover - makes this a generator like the real parser

    # Retained preparation is the only parser on the live route.
    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_stream", failing_parse)

    result = ingest_append_with_owner(owner, [plan])

    assert result.succeeded == []
    assert result.failed == [plan]
    parsed_at_ms, parse_error = _append_raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "injected append parse failure" in parse_error
    assert len(parse_error) <= 2000
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_id = str(conn.execute("SELECT raw_id FROM raw_sessions WHERE source_index = -1").fetchone()[0])
        envelope = read_archive_raw_session_envelope(conn, raw_id)
    assert envelope.parse_error == parse_error
    assert envelope.detection_warnings == (parse_error[:500],)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        # The accepted full stays; the failed append adds nothing.
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1


def test_identity_less_append_plan_is_refused_before_any_raw_write(tmp_path: Path) -> None:
    """Only a hook carrier may append without a declared session identity.

    The planner never forms an identity-less append for a declared artifact
    (it returns no plan and the full route admits the artifact, see
    ``test_full_batch_declared_artifact_is_admitted_before_pending_raw_write``),
    and acquisition refuses such a plan before writing any raw.
    """
    path = tmp_path / "subagents" / "workflows" / "wf-append" / "journal.jsonl"
    path.parent.mkdir(parents=True)
    payload = b'{"contentKey":"call-1","agentId":"agent-a"}\n'
    path.write_bytes(payload)
    plan = replace(_append_plan(path, payload, payload_hash="artifact"), source_name="claude-code")

    result = ingest_append_with_owner(_append_owner(tmp_path), [plan])

    assert result.succeeded == []
    assert result.failed == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)


def test_full_batch_declared_artifact_is_admitted_before_pending_raw_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    payload = b'{"contentKey":"call-2","agentId":"agent-b"}\n'
    source.write_bytes(payload)
    expected_mtime_ms = int(source.stat().st_mtime * 1000)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, _fallback_provider: (Provider.CLAUDE_CODE, True),
    )

    metrics = run_ingest_files(processor, [source], emit_event=False)

    # The retained artifact settles as a terminal no-session exclusion.
    assert metrics.succeeded_file_count == 0
    assert metrics.failed_file_count == 0
    assert metrics.refused_bytes_by_reason == {REFUSED_NO_SESSIONS: len(payload)}
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        raw = conn.execute(
            "SELECT raw_id, logical_source_key, revision_kind, revision_authority, file_mtime_ms FROM raw_sessions"
        ).fetchone()
        artifact = conn.execute("SELECT artifact_kind, parse_as_session, raw_id FROM raw_artifacts").fetchone()
    assert raw is not None
    assert raw[1:] == (None, "unknown", "quarantined", expected_mtime_ms)
    assert artifact == ("workflow_journal", 0, raw[0])


def test_full_batch_session_shaped_workflow_journal_reaches_parser_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        b"".join(
            b'{"contentKey":"artifact-' + str(index).encode() + b'","agentId":"workflow-agent"}\n'
            for index in range(64)
        )
        + b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"recover this journal record"},'
        b'"uuid":"journal-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":"journal-user","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"repaired reply"}]},"uuid":"journal-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )

    first = run_ingest_files(processor, [source], emit_event=False)
    second = run_ingest_files(processor, [source], emit_event=False)

    assert first.ingested_session_count == 1
    assert first.failed_file_count == 0
    assert second.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_large_full_batch_session_shaped_workflow_journal_reaches_parser_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(
        b'{"contentKey":"artifact-0","agentId":"workflow-agent","summary":"'
        + b"x" * _RETIRED_FULL_INGEST_SIZE_BOUND
        + b'"}\n'
        + b"".join(
            b'{"contentKey":"artifact-' + str(index).encode() + b'","agentId":"workflow-agent"}\n'
            for index in range(1, 32)
        )
        + b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"recover this journal record"},'
        b'"uuid":"journal-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        + b'{"parentUuid":"journal-user","type":"assistant","message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"repaired reply"}]},"uuid":"journal-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    assert source.stat().st_size > _RETIRED_FULL_INGEST_SIZE_BOUND
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )

    first = run_ingest_files(processor, [source], emit_event=False)
    second = run_ingest_files(processor, [source], emit_event=False)

    assert first.ingested_session_count == 1
    assert first.failed_file_count == 0
    assert second.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


def test_full_batch_malformed_workflow_journal_remains_typed_evidence(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    source = root / "subagents" / "workflows" / "wf-batch" / "journal.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"contentKey":"broken"\n')
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="test-parser",
    )

    metrics = run_ingest_files(processor, [source], emit_event=False)

    # The retained journal settles as a terminal corrupt-input exclusion.
    assert metrics.succeeded_file_count == 0
    assert metrics.failed_file_count == 0
    assert metrics.refused_bytes_by_reason == {REFUSED_CORRUPT_INPUT: source.stat().st_size}
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        # A filename cannot turn a complete corrupt record into artifact
        # proof: the typed evidence is the corrupt input, not a journal.
        assert conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchone() == (
            "terminal_corrupt_input",
            0,
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_append_admission_bind_failure_persists_exact_pending_envelope_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="append-admission-retry")
    original_bind = ArchiveStore.bind_raw_revision
    fail_once = True

    def fail_bind(self: ArchiveStore, raw_id: str, revision: RawRevisionEnvelope, **kwargs: Any) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected append bind failure")
        original_bind(self, raw_id, revision, **kwargs)

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", fail_bind)
    first = ingest_append_with_owner(owner, [plan])

    assert first.succeeded == []
    assert first.failed == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """
            SELECT raw_id, blob_hash, blob_size, logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions WHERE source_index = -1
            """
        ).fetchone()
    assert row is not None
    raw_id = str(row[0])
    assert row[1] is not None
    assert BlobStore(tmp_path / "blob").read_all(bytes(row[1]).hex()) == plan.payload
    assert row[2] == len(plan.payload)
    assert row[3:13] == (
        f"pending-raw:codex-session:-1:{plan.path}:{raw_id}",
        "full",
        sha256(plan.payload).hexdigest(),
        None,
        None,
        None,
        None,
        None,
        0,
        "quarantined",
    )
    # The admitted acquire stage fails as a whole before it returns a raw id
    # (561dbe2ff0), so no parse evidence is written: the bytes did not fail
    # to decode, and the pending envelope alone carries the retry.
    assert row[13] is None

    retry = ingest_append_with_owner(owner, [plan])

    assert retry.succeeded == [plan]
    assert retry.failed == []
    bound = _raw_revision_envelope_row(tmp_path, raw_id)
    assert bound[0] == "codex-session:append-admission-retry"
    assert bound[1] == "append"
    assert bound[2] is not None
    assert bound[3] is not None
    assert bound[4] is not None
    assert bound[5] is not None
    assert bound[6] == plan.stat_size - len(plan.payload)
    assert bound[7] == plan.stat_size
    assert isinstance(bound[8], int) and bound[8] >= 0
    assert bound[9] == "byte_proven"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE source_index = -1").fetchone() == (1,)


def test_public_full_blob_publication_failure_keeps_the_bound_raw_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "blob-retry.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"blob-retry"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"hello"}]}}\n' + (b" " * (9 * 1024 * 1024))
    )
    source.write_bytes(payload)
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    from polylogue.storage.derived.raw import RawObservationDerivation

    # The full route acquires the raw under a pending envelope and binds its
    # revision when retained publication runs; fail that publication once.
    original_publish = RawObservationDerivation.publish
    fail_once = True

    def fail_publish(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> Any:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.IntegrityError("injected blob bind failure")
        return original_publish(self, *args, **kwargs)

    monkeypatch.setattr(RawObservationDerivation, "publish", fail_publish)
    first = run_ingest_files(processor, [source], emit_event=False)

    assert first.full_file_count == 1
    assert first.failed_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """
            SELECT raw_id, blob_hash, blob_size, logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions
            """
        ).fetchone()
    assert row is not None
    raw_id = str(row[0])
    assert BlobStore(tmp_path / "blob").read_all(bytes(row[1]).hex()) == payload
    assert row[2] == len(payload)
    # Retained preparation commits its Source phase (the revision binding)
    # before the failed Index publication, so the retained raw keeps its
    # bound envelope and carries no parse failure; only publication retries.
    assert row[3:14] == (
        "codex-session:blob-retry",
        "full",
        sha256(payload).hexdigest(),
        None,
        None,
        raw_id,
        None,
        None,
        0,
        "byte_proven",
        None,
    )

    retry = run_ingest_files(processor, [source], emit_event=False)
    assert retry.full_file_count == 1
    assert retry.succeeded_file_count == 1
    assert retry.failed_file_count == 0

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        retained = conn.execute(
            """
            SELECT logical_source_key, revision_kind, source_revision,
                   predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority, parse_error
            FROM raw_sessions
            """
        ).fetchone()
    assert retained == (
        "codex-session:blob-retry",
        "full",
        sha256(payload).hexdigest(),
        None,
        None,
        raw_id,
        None,
        None,
        0,
        "byte_proven",
        None,
    )


def test_append_archive_lock_propagates_for_watcher_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-locked.jsonl"
    payload = b'{"type":"session_meta","payload":{"id":"append-locked"}}\n'
    path.write_bytes(payload)
    # A watcher plan carries the session identity it binds the append to.
    plan = _append_plan(path, payload, payload_hash="locked", native_id_hint="append-locked")
    owner = _append_owner(tmp_path)

    monkeypatch.setattr(
        ArchiveStore,
        "write_raw_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        ingest_append_with_owner(owner, [plan])


def test_full_archive_lock_propagates_for_watcher_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "full-locked.jsonl"
    source.write_bytes(
        b'{"type":"session_meta","payload":{"id":"full-locked"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0",'
        b'"role":"user","content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    monkeypatch.setattr(
        ArchiveStore,
        "write_raw_blob_ref",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _full_paths_sync(processor, [source], source_name="codex")


def test_append_index_failure_never_marks_raw_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="index-fail")

    def fail_index(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.IntegrityError("injected index commit failure")

    monkeypatch.setattr(ArchiveStore, "apply_raw_revision_replay", fail_index)
    result = ingest_append_with_owner(owner, [plan])

    assert result.failed == [plan]
    parsed_at_ms, parse_error = _append_raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and "injected index commit failure" in parse_error
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1


def test_append_multi_session_payload_is_rejected_before_index_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="append-multi")
    # polylogue-9ykn: a message-less ParsedSession carries no positive
    # conversational evidence and is refused before this test's own
    # "more than one session" check ever runs -- give each session one real
    # message so this fixture keeps testing the multi-session rejection.
    sessions = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="multi-1",
            messages=[ParsedMessage(provider_message_id="multi-1-0", role=Role.USER, text="hello")],
        ),
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="multi-2",
            messages=[ParsedMessage(provider_message_id="multi-2-0", role=Role.USER, text="hello")],
        ),
    ]
    # Retained preparation is the only parser on the live route.
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream", _retained_parse_by_path(lambda _path: sessions)
    )
    result = ingest_append_with_owner(owner, [plan])

    assert result.failed == [plan]
    parsed_at_ms, parse_error = _append_raw_parse_state(tmp_path)
    assert parsed_at_ms is None
    # The chain's member yields no session for the append's logical key: a
    # typed cohort refusal, recorded on the append raw it settles.
    assert isinstance(parse_error, str) and parse_error.startswith("CohortMembershipRefusalError:")
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("append-multi",)]


def test_full_parser_exception_settles_as_terminal_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parser that fails on retained bytes refuses them once, typed.

    Parsing the same immutable bytes with the same parser can only fail the
    same way, so a non-decode parser exception settles as a terminal refusal
    with raw evidence. A failed census instead would be re-censused on every
    pass without progress.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "parser-crash.jsonl"
    source.write_bytes(_codex_shaped_bytes("parser-crash"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def crash(*_args: object, **_kwargs: object) -> Generator[ParsedSession, None, None]:
        raise RuntimeError("injected deterministic parser failure")
        yield  # pragma: no cover - makes this a generator like the real parser

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_stream", crash)

    first = run_ingest_files(processor, [source], emit_event=False)
    second = run_ingest_files(processor, [source], emit_event=False)

    assert first.failed_file_count == 0
    assert second.failed_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT r.parsed_at_ms, r.parse_error, a.artifact_kind, c.status
            FROM raw_sessions AS r
            JOIN raw_artifacts AS a ON a.raw_id = r.raw_id
            JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
            WHERE r.source_path = ?
            """,
            (str(source),),
        ).fetchall()
    assert len(rows) == 1
    parsed_at_ms, parse_error, artifact_kind, census_status = rows[0]
    assert parsed_at_ms is None
    assert isinstance(parse_error, str) and parse_error.startswith("RuntimeError:")
    assert artifact_kind == RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value
    assert census_status == "non_session"
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


def test_full_multi_session_failure_retries_without_success_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "full-multi.jsonl"
    # Rollout bytes retained preparation admits; the parser stub decides the sessions.
    source.write_bytes(_codex_shaped_bytes("full-multi"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    # polylogue-9ykn: a message-less ParsedSession carries no positive
    # conversational evidence and is refused before this test's own
    # injected-second-write-failure path ever runs -- give each session one
    # real message so this fixture keeps testing that failure-handling path.
    sessions = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="full-multi-1",
            messages=[ParsedMessage(provider_message_id="full-multi-1-0", role=Role.USER, text="hello")],
        ),
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="full-multi-2",
            messages=[ParsedMessage(provider_message_id="full-multi-2-0", role=Role.USER, text="hello")],
        ),
    ]
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    # Retained preparation is the only parser on the live route; it closes
    # the parser's generator, which a bare list iterator cannot stand in for.
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream", _retained_parse_by_path(lambda _path: sessions)
    )
    # polylogue-1r9c: _write_parsed_precedence_result is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through ArchiveStore's `self.` dispatch -- patch it there.
    original_write = archive_revision_governance._write_parsed_precedence_result
    write_count = 0

    def fail_second_index(
        archive: archive_revision_governance.RawRevisionGovernanceHost,
        session: ParsedSession,
        **kwargs: object,
    ) -> object:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise sqlite3.IntegrityError("injected full second-session index failure")
        return original_write(archive, session, **cast(Any, kwargs))

    monkeypatch.setattr(archive_revision_governance, "_write_parsed_precedence_result", fail_second_index)

    # An index write failure is this file's failure, not the batch's: the
    # live batch reports it failed and retries it on the next observation.
    failed = run_ingest_files(processor, [source], emit_event=False)
    assert failed.failed_file_count == 1
    assert failed.succeeded_file_count == 0

    # A partially written multi-session raw is not success: the component
    # transaction rolls back, so no session from it is visible.
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    retry = run_ingest_files(processor, [source], emit_event=False)

    assert retry.succeeded_file_count == 1
    assert retry.failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


def test_full_ingest_skips_durably_excised_content_without_aborting_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon's full/streaming write orchestration (polylogue-27m fix round).

    Reproduces the reviewer's finding at the orchestration layer, not just the
    low-level write gate: a durably excised blob hash must be skipped (counted
    in ``_ArchiveFullWriteResult.excised_skips``) without aborting the rest of
    the batch. Reverting the ``except ContentExcisedError`` handling in
    ``LiveBatchProcessor._acquire_full_records_archive`` back to letting it fall
    through to the generic ``except Exception`` branch (or removing the
    pre-write ``is_blob_hash_excised`` gate entirely) makes this fail: either
    the whole batch call raises, or the excised content gets a fresh raw_id.

    The streaming threshold is patched down (matching the pattern used
    elsewhere in this file, e.g. ``test_full_ingest_reports_heartbeat_stage_events``)
    so both fixture files route through ``capture_bound_path`` /
    ``archive.write_raw_blob_ref`` -> ``write_source_raw_session_blob_ref``,
    the same code path a real >8MB capture takes -- not the small-payload
    ``write_raw_payload`` -> ``write_source_raw_session`` gate, which is a
    different call site (polylogue-re4a).
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import (
        deterministic_blob_hash,
        record_excised_blob_hash,
    )

    root = tmp_path / "sessions"
    root.mkdir()
    excised_payload = b'{"secret": "sk-ant-should-not-resurrect-via-streaming-batch"}\n'
    excised_source = root / "excised.jsonl"
    excised_source.write_bytes(excised_payload)
    normal_source = root / "normal.jsonl"
    # Real, parseable content -- not a bare `{}` -- because polylogue-lb39z's
    # guarded presence-guarantee fallback (#3630) can drive this fixture's
    # raw revision through a real re-parse (_parse_raw_revision_chain) that
    # the parse_stream_payload monkeypatch below does not intercept, and a
    # genuinely empty/malformed record now correctly fails to replay to any
    # session rather than being silently accepted.
    normal_source.write_bytes(b'{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n')

    # Pre-mark the excised file's exact content hash as durably excised,
    # mirroring a prior real `polylogue ops excise` apply.
    bootstrap_archive_root(tmp_path)
    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        record_excised_blob_hash(
            source_conn,
            blob_hash=deterministic_blob_hash(excised_payload),
            reason="test: reproduces reviewer finding",
            actor="user:local",
            excised_at_ms=1_000,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    # polylogue-9ykn: a message-less ParsedSession carries no positive
    # conversational evidence and is refused before this test's own
    # durably-excised-content skip path is exercised -- give it one real
    # message so "normal.jsonl" still succeeds.
    sessions = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="normal-1",
            messages=[ParsedMessage(provider_message_id="normal-1-0", role=Role.USER, text="hello")],
        )
    ]
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    # Retained preparation is the only parser on the live route.
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream", lambda *_args, **_kwargs: iter(list(sessions))
    )

    # Force both fixture files through the streaming blob-ref write path
    # (>= this threshold uses capture_bound_path + write_raw_blob_ref,
    # never populating raw_payloads) rather than the small-payload
    # write_raw_payload path -- see polylogue-re4a.

    archive_results: list[_ArchiveFullWriteResult] = []
    original_full_write = processor._acquire_full_records_archive

    def capture_full_write(*args: Any, **kwargs: Any) -> _ArchiveFullWriteResult:
        outcome = original_full_write(*args, **kwargs)
        archive_results.append(outcome)
        return outcome

    monkeypatch.setattr(processor, "_acquire_full_records_archive", capture_full_write)

    result = _full_paths_sync(processor, [excised_source, normal_source], source_name="codex")

    assert archive_results[0].excised_skips == 1
    # The non-excised file in the same batch still succeeds -- one excised
    # record must not abort the rest of the batch.
    assert normal_source in result.succeeded
    assert excised_source not in result.succeeded

    with sqlite3.connect(tmp_path / "source.db") as conn:
        # No raw_sessions row was resurrected for the excised payload.
        rows = conn.execute("SELECT source_path FROM raw_sessions").fetchall()
        assert all("excised.jsonl" not in str(row[0]) for row in rows)

        # The streaming route publishes (stages and reserves) the payload
        # before the write refuses it, and the success path's receipt
        # consumption never runs on a refusal. An orphaned reservation makes
        # the excised hash permanently GC-immune -- inspect_blob_reservation
        # reports LIVE -- while every later pass over the same unchanged file
        # accrues another receipt. The refusal handler must release it so
        # ordinary blob GC can reclaim the content the operator excised.
        #
        # Anti-vacuity: removing release_refused_publication_receipt from the
        # ContentExcisedError handler in _acquire_full_records_archive leaves
        # exactly one reservation row here for the excised hash.
        excised_hash = deterministic_blob_hash(excised_payload)
        reserved = {
            bytes(row[0]) for row in conn.execute("SELECT blob_hash FROM blob_publication_reservations").fetchall()
        }
        assert excised_hash not in reserved, "a refused excised write left its publication reservation behind"


def test_live_multi_session_divergence_keeps_accepted_head_as_debt(tmp_path: Path) -> None:
    root = tmp_path / "inbox"
    root.mkdir()
    first = root / "first.json"
    second = root / "second.json"

    def conversation(native_id: str, *texts: str) -> dict[str, object]:
        mapping: dict[str, object] = {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": [f"{native_id}-node-0"],
            }
        }
        for index, text in enumerate(texts):
            node_id = f"{native_id}-node-{index}"
            next_node = f"{native_id}-node-{index + 1}" if index + 1 < len(texts) else None
            mapping[node_id] = {
                "id": node_id,
                "parent": "root" if index == 0 else f"{native_id}-node-{index - 1}",
                "children": [] if next_node is None else [next_node],
                "message": {
                    "id": f"{native_id}-message-{index}",
                    "author": {"role": "user"},
                    "create_time": 1_780_000_000.0 + index,
                    "content": {"content_type": "text", "parts": [text]},
                    "metadata": {},
                },
            }
        return {
            "id": native_id,
            "title": native_id,
            "create_time": 1_780_000_000.0,
            "current_node": f"{native_id}-node-{len(texts) - 1}",
            "mapping": mapping,
        }

    first.write_text(
        json.dumps([conversation("shared", "base", "left"), conversation("safe-1", "one")]),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps([conversation("shared", "base", "right"), conversation("safe-2", "two")]),
        encoding="utf-8",
    )
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    # This is a real ChatGPT export bundle, not a JSONL shape authorized by a
    # monkeypatch. Keep the taxonomy assertion next to the route assertion so
    # the test cannot pass after accidentally becoming a non-session fixture.
    assert _parse_path_as_session_artifact(first, provider=Provider.CHATGPT) is True
    first_result = _full_paths_sync(processor, [first], source_name="inbox")
    assert first_result.succeeded == [first]
    assert first_result.failed == []
    # Retained preparation resolves the inbox payload's provider.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT origin FROM raw_sessions WHERE source_path = ?", (str(first),)).fetchone() == (
            "chatgpt-export",
        )
    accepted_raw_id = first_result.raw_fingerprints[first]

    second_result = _full_paths_sync(processor, [second], source_name="inbox")
    # The divergent authority remains unresolved, but this source file was
    # acquired and parsed. Its unchanged bytes must not become a retry loop.
    assert second_result.failed == []
    assert second_result.succeeded == [second]
    # Direct check of the persisted state backing that claim (this layer --
    # ``_ingest_full_paths_sync`` -- has no CursorStore row of its own; the
    # durable "not a retry loop" evidence lives in raw_sessions/raw_session_
    # memberships). ``second``'s raw must show no parse_error (what would
    # make a later pass retry it as a failure) while its logical identity's
    # membership decision is durably ambiguous/quarantined -- i.e. the
    # deferred authority debt is actually persisted for this exact path, not
    # only implied by the in-memory FullIngestResult lists above.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT r.parse_error, m.decision, m.revision_authority
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'chatgpt-export:shared'
            """,
            (str(second),),
        ).fetchone() == (None, "ambiguous", "quarantined")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_session_memberships WHERE decision = 'ambiguous'").fetchone() == (
            2,
        )
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NULL").fetchone() == (2,)
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parse_error IS NOT NULL").fetchone() == (0,)
        assert conn.execute(
            """
            SELECT r.source_path, m.decision, m.revision_authority
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE m.logical_source_key = 'chatgpt-export:shared'
            ORDER BY r.source_path
            """
        ).fetchall() == [
            (str(first), "ambiguous", "quarantined"),
            (str(second), "ambiguous", "quarantined"),
        ]
    with sqlite3.connect(index_db) as conn:
        # The first accepted branch remains queryable; the later divergence is
        # nonterminal debt and has no deletion authority.
        assert set(conn.execute("SELECT native_id FROM sessions")) == {
            ("safe-1",),
            ("safe-2",),
            ("shared",),
        }
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:shared'"
        ).fetchone() == (accepted_raw_id,)
        assert conn.execute(
            """
            SELECT b.search_text
            FROM sessions AS s
            JOIN messages AS m USING (session_id)
            JOIN blocks AS b USING (message_id)
            WHERE s.native_id = 'shared'
            ORDER BY m.position, b.position
            """
        ).fetchall() == [("base",), ("left",)]

    # Retrying the accepted file re-observes the same bytes: the same raw, the
    # same cohort, the same decision. A decided-ambiguous cohort keeps its
    # last accepted head with the conflict as debt on every member's row,
    # including the head's (#3282); only new evidence can resolve it.
    retry_result = _full_paths_sync(processor, [first], source_name="inbox")

    assert retry_result.succeeded == [first]
    assert retry_result.failed == []
    assert retry_result.raw_fingerprints[first] == accepted_raw_id
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)
        assert conn.execute(
            """
            SELECT r.source_path, m.decision, m.revision_authority,
                   r.parsed_at_ms IS NOT NULL, r.parse_error
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE m.logical_source_key = 'chatgpt-export:shared'
            ORDER BY r.source_path
            """
        ).fetchall() == [
            (str(first), "ambiguous", "quarantined", 0, None),
            (str(second), "ambiguous", "quarantined", 0, None),
        ]
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:shared'"
        ).fetchone() == (accepted_raw_id,)


def test_live_third_raw_reunifies_with_backfill_retired_siblings(tmp_path: Path) -> None:
    """polylogue-hm2f: the live incremental path must reunite retired siblings, not drop new raws forever.

    Mirrors the exact live call sequence the polylogue-52l2 guard protects
    (``bind_raw_revision`` -> ``classify_raw_revision_cohort``), then proves
    the new routing this fix adds: when that cohort comes back empty AND
    ``raw_membership_retired_full_revision_siblings`` shows this identity has
    known siblings already retired to membership governance -- exactly the
    durable state offline backfill (``sources/revision_backfill.py``,
    ``convertible_full_revision_raw_ids`` + ``replace_raw_membership_census``)
    leaves behind for a decided-ambiguous full-only cohort -- a newly
    discovered THIRD raw for the same identity must be folded into that same
    membership governance and weighed by the real content-prefix classifier
    (``classify_membership_revisions``) alongside every known sibling,
    instead of being silently dropped with only a warning log line (the
    pre-fix behavior: ``bind_raw_revision`` succeeds, but no
    ``raw_session_memberships`` row is ever written for the raw and the file
    surfaces as failed with zero evidence trail).

    raw_a=["base","left"], raw_b=["base","right"] are byte-divergent (not a
    prefix of one another) -- a genuine, decided ambiguous cohort, retired
    here exactly the way ``backfill_historical_revision_evidence`` retires
    one once ``classify_raw_revision_cohort`` returns no accepted chain.
    raw_c=["base","left","extra"] then arrives through the live incremental
    path (``LiveBatchProcessor._ingest_full_paths_sync``, the production
    entry point, not a hand-simulated call). Content-wise raw_c does not
    strictly dominate raw_b (they diverge at message index 1) so the real
    classifier still cannot order the full three-way cohort as a clean
    containment chain -- but critically that decision is reached by
    weighing raw_c against BOTH retired siblings: since this logical source
    has never had an accepted head, the presence-guarantee fallback
    (polylogue-lb39z item 5, ``_maximal_evidence_fallback``) deterministically
    materializes raw_c (the largest-frontier representative) instead of
    leaving the reunified cohort headless, with raw_a/raw_b recorded as its
    conflict debt. All three raws end up in ``raw_session_memberships`` with
    a real, decided outcome, proving reunification happened rather than
    raw_c being evaluated alone or dropped.
    """

    def conversation(native_id: str, *texts: str) -> dict[str, object]:
        mapping: dict[str, object] = {
            "root": {"id": "root", "message": None, "parent": None, "children": [f"{native_id}-node-0"]}
        }
        for index, text in enumerate(texts):
            node_id = f"{native_id}-node-{index}"
            next_node = f"{native_id}-node-{index + 1}" if index + 1 < len(texts) else None
            mapping[node_id] = {
                "id": node_id,
                "parent": "root" if index == 0 else f"{native_id}-node-{index - 1}",
                "children": [] if next_node is None else [next_node],
                "message": {
                    "id": f"{native_id}-message-{index}",
                    "author": {"role": "user"},
                    "create_time": 1_780_000_000.0 + index,
                    "content": {"content_type": "text", "parts": [text]},
                    "metadata": {},
                },
            }
        return {
            "id": native_id,
            "title": native_id,
            "create_time": 1_780_000_000.0,
            "current_node": f"{native_id}-node-{len(texts) - 1}",
            "mapping": mapping,
        }

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as store:
        raw_a = store.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps([conversation("shared", "base", "left")]).encode(),
            source_path="a.json",
            canonical_source_path="a.json",
            acquired_at_ms=1,
        )
        store.bind_raw_revision(
            raw_a,
            RawRevisionEnvelope(
                "chatgpt-export:shared", RawRevisionKind.FULL, raw_a, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )
        raw_b = store.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps([conversation("shared", "base", "right")]).encode(),
            source_path="b.json",
            canonical_source_path="b.json",
            acquired_at_ms=2,
        )
        store.bind_raw_revision(
            raw_b,
            RawRevisionEnvelope(
                "chatgpt-export:shared", RawRevisionKind.FULL, raw_b, 0, authority=RawRevisionAuthority.QUARANTINED
            ),
        )

        # Exactly the polylogue-52l2 guard-tripping sequence: no unique
        # byte-prefix chain across a and b.
        plan = store.classify_raw_revision_cohort_for_rebuild_repair("chatgpt-export:shared")
        assert plan.accepted_raw_ids == ()

        convertible = list(store.convertible_full_revision_raw_ids("chatgpt-export:shared"))
        store.commit()
    # Mirror the retained route's retirement step once a full-only cohort is
    # decided ambiguous: move every convertible full raw to membership
    # governance on a prepared Source census.
    payloads = {
        raw_a: [conversation("shared", "base", "left")],
        raw_b: [conversation("shared", "base", "right")],
    }
    seed_membership_census(
        tmp_path,
        [
            (raw_id, parse_payload(Provider.CHATGPT, payloads[raw_id], raw_id, source_path=f"{raw_id}.json"))
            for raw_id in convertible
        ],
        parser_fingerprint=raw_authority_parser_fingerprint(),
        censused_at_ms=0,
        detail=HISTORICAL_NON_PREFIX_GOVERNANCE_DETAIL,
        retire_full_revision_governance=True,
        revision_authority=RawRevisionAuthority.QUARANTINED,
    )
    with ArchiveStore.open_existing(tmp_path, read_only=True) as store:
        retired_siblings = store.raw_membership_retired_full_revision_siblings("chatgpt-export:shared")
    assert set(retired_siblings) == {raw_a, raw_b}

    # A THIRD raw for the same logical identity, discovered afterward
    # through the actual live incremental entry point.
    root = tmp_path / "inbox"
    root.mkdir()
    third = root / "third.json"
    third.write_text(json.dumps([conversation("shared", "base", "left", "extra")]), encoding="utf-8")
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )
    third_result = _full_paths_sync(processor, [third], source_name="inbox")

    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT r.source_path, m.decision
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE m.logical_source_key = 'chatgpt-export:shared'
            ORDER BY r.source_path
            """
        ).fetchall()

    # Reunification proof: raw_c (third.json) has a raw_session_memberships
    # row -- it was folded into the SAME membership cohort as raw_a/raw_b,
    # not evaluated alone and not silently dropped. Every member of the
    # cohort has a real DECIDED outcome (not NULL/pending, not simply
    # absent).
    by_path = dict(rows)
    assert set(by_path) == {"a.json", "b.json", str(third)}
    assert all(decision is not None for decision in by_path.values())

    # raw_a/raw_b are a genuine two-way divergence (shared "left"/"right"
    # message content conflicts), and raw_c neither purely contains nor is
    # contained by raw_b -- so the cohort as a whole is still an irreducible
    # conflict; no clean prefix chain exists. This logical source has never
    # had an accepted head (raw_a/raw_b were both retired straight to
    # membership governance quarantined, never byte-governed-accepted), so
    # the presence-guarantee fallback (polylogue-lb39z item 5) is free to
    # deterministically materialize the maximal-evidence representative
    # instead of leaving the reunified cohort headless: raw_c strictly
    # contains raw_a's content plus a further "extra" message, giving it the
    # largest frontier of the three, so it wins outright (no raw_id tiebreak
    # needed) and raw_a/raw_b become its recorded conflict debt. The source
    # observation itself was acquired and parsed successfully, so its cursor
    # is complete rather than retried as a transient file failure either way.
    assert by_path[str(third)] == "applied"
    assert by_path["a.json"] == "ambiguous"
    assert by_path["b.json"] == "ambiguous"
    assert third_result.failed == []
    assert third_result.succeeded == [third]
    # Direct check of the persisted state backing "cursor is complete rather
    # than retried" above: ``_ingest_full_paths_sync`` has no CursorStore row
    # of its own, so the durable non-retry evidence is raw_sessions.parse_error
    # staying NULL for third's raw regardless of its membership decision --
    # what actually stops the daemon from reprocessing this file as a failure
    # on every restart, not just the in-memory succeeded/failed lists.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(third),),
        ).fetchone() == (None,)


def test_raw_membership_decision_pending_distinguishes_null_from_ambiguous(tmp_path: Path) -> None:
    """Pins the exact narrow scoping of the polylogue-emx2 fix (de0b2df7a regression, polylogue-lvz6 triage).

    ``raw_membership_authority_complete()`` collapses three distinct
    membership-decision states into one boolean: ``decision IS NULL``
    (genuinely async-pending -- censused but not yet arbitrated by the
    raw-materialization conveyor, ``sources/revision_backfill.py``) and
    ``decision IN ('ambiguous', 'deferred')`` (arbitration already ran and
    concluded a real conflict that needs new evidence, not time, to
    resolve). The first is a conveyor hand-off; the second is a durable
    fail-closed materialization outcome. Neither is a transient source-file
    failure, so the live cursor must not re-read unchanged bytes for either
    state. ``LiveBatchProcessor._acquire_full_records_archive`` uses
    ``raw_membership_decision_pending`` (not the coarse boolean alone) to
    preserve this distinction in its durable raw-authority state while both
    paths remain cursor-idempotent.

    This is an archive-tier predicate test rather than a full watcher
    end-to-end scenario because, by construction,
    ``LiveBatchProcessor._apply_membership_sessions`` always resolves the
    raw it just censused synchronously within the same call (both of its
    current call sites pass ``allow_current_complete_raw=True``) -- so a
    raw's own decision is never observed as NULL immediately after that
    call returns. Genuinely NULL decisions persist only across the
    conveyor's own two-phase census-then-classify split
    (``census_historical_revision_evidence`` /
    ``backfill_historical_revision_evidence``), which this test reproduces
    directly against the archive tier: census without classification (NULL,
    pending) versus census with an ambiguous classification (decided,
    unresolved).
    """
    bootstrap_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="pending-vs-ambiguous",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="hello")],
    )
    projection = session_revision_projection(session)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"native_id":"pending-vs-ambiguous"}\n',
            source_path=str(tmp_path / "pending-vs-ambiguous.jsonl"),
            canonical_source_path=str(tmp_path / "pending-vs-ambiguous.jsonl"),
            acquired_at_ms=1,
        )
        archive.commit()
    seed_membership_census(tmp_path, [(raw_id, [session])], parser_fingerprint="test-parser")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        # Census complete, classification never run: decision IS NULL. This
        # is the genuinely async-pending state -- not a failure.
        assert archive.raw_membership_authority_complete(raw_id) is False
        assert archive.raw_membership_decision_pending(raw_id) is True

        # Arbitration now runs and concludes ambiguous (a decided conflict,
        # e.g. the conveyor found no unique growth chain). This is no longer
        # pending -- it must surface as a failure, not defer forever.
        publish_prepared_membership_classification(
            archive,
            "codex-session:pending-vs-ambiguous",
            MembershipClassification((), (), (raw_id,)),
            {raw_id: session},
            {raw_id: projection},
            decided_at_ms=2,
        )
        assert archive.raw_membership_authority_complete(raw_id) is False
        assert archive.raw_membership_decision_pending(raw_id) is False
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                "SELECT decision FROM raw_session_memberships WHERE raw_id = ?", (raw_id,)
            ).fetchone() == ("ambiguous",)


def test_live_membership_reprocesses_parser_drift_without_retiring_unrelated_head(tmp_path: Path) -> None:
    """A current parse of the accepted raw is authority, even after parser drift.

    This reproduces the July 16 live failure: an older browser snapshot was
    accepted under an earlier parser, then byte-equivalent current snapshots
    reparsed both the accepted raw and the new raw to the same new projection.
    The accepted index head remains the CAS witness; its old content hash must
    not be mistaken for an unrelated raw head.
    """
    root = tmp_path / "inbox"
    root.mkdir()
    snapshot = root / "snapshot.json"
    payload: list[dict[str, object]] = [
        {
            "id": "parser-drift",
            "title": "current title",
            "create_time": 1_780_000_000.0,
            "current_node": "node",
            "mapping": {
                "node": {
                    "id": "node",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "message",
                        "author": {"role": "user"},
                        "create_time": 1_780_000_000.0,
                        "content": {"content_type": "text", "parts": ["retained evidence"]},
                        "metadata": {},
                    },
                }
            },
        }
    ]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    current_session = parse_payload(Provider.CHATGPT, payload, "snapshot")[0]
    legacy_session = current_session.model_copy(update={"title": "legacy parser title"})
    legacy_projection = session_revision_projection(legacy_session)
    current_projection = session_revision_projection(current_session)
    assert legacy_projection.session_hash != current_projection.session_hash

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        legacy_raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=snapshot.read_bytes(),
            source_path=str(snapshot),
            canonical_source_path=str(snapshot),
            acquired_at_ms=1,
        )
        archive.commit()
    seed_membership_census(tmp_path, [(legacy_raw_id, [legacy_session])], parser_fingerprint="legacy-parser")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        publish_prepared_membership_classification(
            archive,
            "chatgpt-export:parser-drift",
            MembershipClassification((legacy_raw_id,), (), ()),
            {legacy_raw_id: legacy_session},
            {legacy_raw_id: legacy_projection},
            decided_at_ms=1,
        )

    # Byte-level formatting changes create a new retained raw while preserving
    # the provider session. The live route reparses the accepted raw too.
    snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".json",))),),
        cursor=CursorStore(tmp_path / "index.db"),
        parser_fingerprint="current-parser",
    )

    result = _full_paths_sync(processor, [snapshot], source_name="inbox")

    assert result.succeeded == [snapshot]
    assert result.failed == []
    with sqlite3.connect(tmp_path / "index.db") as conn:
        stored = conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = 'chatgpt-export:parser-drift'"
        ).fetchone()
        assert stored is not None
        assert stored != (legacy_projection.session_hash,)
        head = conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = 'chatgpt-export:parser-drift'"
        ).fetchone()
        assert head == stored


def test_single_session_full_terminally_supersedes_older_membership_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "bundle.jsonl"
    older = root / "older.jsonl"
    bundle.write_bytes(_codex_shaped_bytes("bundle"))
    older.write_bytes(_codex_shaped_bytes("older"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    bundle_sessions = [session("shared", "base", "new"), session("safe", "one")]
    older_sessions = [session("shared", "base")]
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream",
        _retained_parse_by_path(lambda path: bundle_sessions if path == bundle else older_sessions),
    )

    assert run_ingest_files(processor, [bundle], emit_event=False).failed_file_count == 0
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        rejected_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=older.read_bytes(),
            source_path=str(older),
            canonical_source_path=str(older.resolve()),
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            rejected_raw_id,
            RawRevisionEnvelope(
                logical_source_key="codex-session:shared",
                kind=RawRevisionKind.FULL,
                source_revision=sha256(older.read_bytes()).hexdigest(),
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
    older_result = run_ingest_files(processor, [older], emit_event=False)

    assert older_result.succeeded_file_count == 1
    assert older_result.failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT message_count FROM sessions WHERE native_id = 'shared'").fetchone() == (2,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT m.decision, r.parsed_at_ms IS NOT NULL, r.parse_error
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'codex-session:shared'
            """,
            (str(older),),
        ).fetchone() == ("superseded_prefix", 1, None)
    # Terminal supersession leaves the older full under membership authority
    # alone: it no longer rebuilds through its byte-revision chain.
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        revision_keys = archive.raw_revision_rebuild_logical_keys([rejected_raw_id])
        _membership_raws, membership_keys = archive.expand_raw_membership_selection([rejected_raw_id])
    assert revision_keys == ()
    assert "codex-session:shared" in membership_keys


@pytest.mark.parametrize(
    ("bundle_texts", "succeeds", "census_head"),
    [
        (("base",), True, False),
        (("base", "different"), False, False),
        (("base", "new", "later"), False, False),
        (("base", "new", "later"), False, True),
    ],
)
def test_bundle_replay_respects_unconvertible_single_session_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle_texts: tuple[str, ...],
    succeeds: bool,
    census_head: bool,
) -> None:
    """Pins #2718's fail-closed contract: a bundle raw discovered later must
    never silently replace an accepted head that still has live, unresolved
    byte-append evidence (the QUARANTINED append raw this test binds), even
    when the bundle's own content happens to strictly extend the head's
    content (``bundle_texts2``/``bundle_texts3``: content-prefix growth alone
    is not proof of provenance).

    polylogue-miwv (2026-07-21): #3211 ("in-cohort head-retire drift fix")
    removed ``apply_raw_membership_classification``'s byte-governance refusal
    on the mistaken premise that its branch is only reachable after a real
    membership-governance conversion -- but ``_apply_membership_sessions``
    unconditionally injects the CURRENT accepted head into the comparison
    cohort even when it has never been converted (exactly this test's byte-
    governed-head scenario), so the removed guard's absence let the older
    bundle's superset content silently move the head (message_count 2->3,
    ``accepted_raw_id`` changed) for ``bundle_texts2``/``bundle_texts3``.
    This was not caused by, and is unrelated to, the messages_fts_identity
    UNIQUE(block_id) ledger work landing the same day (polylogue-miwv's
    other commits) -- confirmed by reproducing this exact failure on the
    commit immediately preceding messages_fts_identity's introduction.
    Restored as a narrower guard (refuses only when replay is about to change
    the accepted raw AND a live raw_sessions row still chains a
    ``predecessor_source_revision`` off the existing head) so #3211's own
    interrupted-pass-drift resumption keeps working.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    current = root / "current.jsonl"
    older_bundle = root / "older-bundle.jsonl"
    current.write_bytes(_codex_shaped_bytes("current"))
    older_bundle.write_bytes(_codex_shaped_bytes("bundle"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    current_session = session("shared", "base", "new")
    bundle_sessions = [session("shared", *bundle_texts), session("safe", "one")]
    current_raw_id: list[str] = []
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream",
        _retained_parse_by_path(lambda path: [current_session] if path == current else bundle_sessions),
    )

    assert run_ingest_files(processor, [current], emit_event=False).failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        row = conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex-session:shared'"
        ).fetchone()
        assert row is not None
        current_raw_id.append(str(row[0]))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        source_revision = (
            archive._ensure_source_conn()
            .execute(
                "SELECT source_revision FROM raw_sessions WHERE raw_id = ?",
                (current_raw_id[0],),
            )
            .fetchone()
        )
        assert source_revision is not None
        append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"append":true}\n',
            source_path=str(current),
            canonical_source_path=str(current.resolve()),
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            append_raw_id,
            RawRevisionEnvelope(
                "codex-session:shared",
                RawRevisionKind.APPEND,
                "append-blocker",
                0,
                predecessor_source_revision=str(source_revision[0]),
                append_start_offset=1,
                append_end_offset=2,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        assert archive.convertible_full_revision_raw_ids("codex-session:shared") == ()
        archive.commit()
    if census_head:
        seed_membership_census(
            tmp_path, [(current_raw_id[0], [current_session])], parser_fingerprint="test-parser", censused_at_ms=2
        )
    with sqlite3.connect(index_db) as conn:
        head_before = conn.execute(
            "SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier "
            "FROM raw_revision_heads WHERE logical_source_key = 'codex-session:shared'"
        ).fetchone()
        assert head_before is not None

    result = run_ingest_files(processor, [older_bundle], emit_event=False)

    # A same-size divergent membership result is a decided authority conflict
    # with a complete source observation; attempted replacement through live
    # append evidence raises instead and must remain retryable.
    cursor_complete = succeeds or bundle_texts == ("base", "different")
    assert result.failed_file_count == (0 if cursor_complete else 1)
    assert result.succeeded_file_count == (1 if cursor_complete else 0)
    # Direct check of the persisted state backing both branches: the
    # succeeded/failed lists above are LiveBatchProcessor's own report, not
    # proof of what raw_sessions durably holds (this layer -- ``_ingest_full_
    # paths_sync`` -- has no CursorStore row of its own). A decided-ambiguous
    # cursor_complete observation is deliberately never materialized
    # (parsed_at_ms stays NULL -- fail-closed for the head), so the actual
    # "not a failure/retry loop" evidence is parse_error staying NULL. When
    # not cursor_complete, the docstring's "must remain retryable" claim
    # requires the raw to actually carry a parse_error -- what makes a future
    # daemon pass re-attempt this exact raw instead of silently treating it
    # as already resolved.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        (parse_error,) = conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(older_bundle),),
        ).fetchone()
    if cursor_complete:
        assert parse_error is None
    else:
        assert parse_error is not None
        # polylogue-5iz4: this guard's refusal is transient/retry-eligible by
        # construction (a later pass over the same durable bytes can succeed
        # once sibling evidence resolves), but a plain RuntimeError leaves the
        # retry-candidate query (storage/raw_convergence.py) nothing stable to match
        # once the message text drifts -- exactly what happened to a real
        # production session that hit this guard under #2718's original
        # wording.
        # The structured evidence row, not the diagnostic wording, is the
        # retry authorization.
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                """
                SELECT a.artifact_kind
                FROM raw_artifacts AS a
                JOIN raw_sessions AS r ON r.raw_id = a.raw_id
                WHERE r.source_path = ?
                """,
                (str(older_bundle),),
            ).fetchone() == ("deferred_cas_frontier",)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute("SELECT message_count FROM sessions WHERE native_id = 'shared'").fetchone() == (2,)
        head_after = conn.execute(
            "SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier "
            "FROM raw_revision_heads WHERE logical_source_key = 'codex-session:shared'"
        ).fetchone()
        assert head_after == ((current_raw_id[0], "semantic", 2) if succeeds else head_before)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_session_memberships WHERE raw_id = ?",
            (current_raw_id[0],),
        ).fetchone() == ((1,) if census_head else (0,))
        decisions = conn.execute(
            "SELECT decision FROM raw_session_memberships WHERE logical_source_key = 'codex-session:shared' AND raw_id != ?",
            (current_raw_id[0],),
        ).fetchall()
        if succeeds:
            assert decisions == [("superseded_prefix",)]


def test_growing_file_incident_recovery_duplicate_recovers_after_head_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-5iz4: reproduce the real production shape at growth-chain scale.

    A real Codex session (native_id ``019f49d8-...``) accumulated 804
    ``raw_sessions`` rows because a live watcher periodically captured
    FULL snapshots of one continuously-growing ``rollout.jsonl`` (not
    append deltas) -- ~800 generations of the SAME file, strictly growing
    byte-for-byte (confirmed read-only against the live archive: every
    smaller full-revision blob is an exact byte prefix of every larger
    one, a single clean linear chain with zero forks). Two extra
    identical-content full snapshots landed at a SECOND, "incident
    recovery" source path sharing the same native_id -- an out-of-band
    backup/restore copy taken during a live incident. One of the 804 rows
    carries ``parse_error='RuntimeError: membership replay cannot replace
    an unconvertible byte head'`` (PR #2718's now-superseded wording);
    the session never reached ``index.db``.

    This test reproduces the mechanism at REALISTIC scale (many real
    incremental full-snapshot captures of one growing Codex JSONL file,
    not a single static snapshot) plus a colliding same-identity duplicate
    from a second path, and then demonstrates the actual recovery path:
    ``apply_raw_membership_classification``'s guard is a **fail-closed,
    correct** refusal (an unrelated/dangling-evidence head must never be
    silently replaced) -- not a permanent dead end. Once
    ``MembershipReplayConflictError`` is recorded with a stable,
    retry-eligible ``parse_error`` marker (polylogue-5iz4 / #3646) AND the
    accepted head naturally advances past the interfering evidence
    (exactly what a live-watched growing file does on its own, and what
    the live archive's current empty ``raw_revision_heads``/``sessions``
    rows for this identity show already happened), a later pass over the
    SAME duplicate raw succeeds and reaches the index with a plausible
    message_count.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    current = root / "rollout-growing.jsonl"
    incident_recovery = root.parent / "inbox" / "incident-recovery-rollout-growing.jsonl"
    incident_recovery.parent.mkdir(parents=True, exist_ok=True)
    current.write_bytes(_codex_shaped_bytes("current"))
    incident_recovery.write_bytes(_codex_shaped_bytes("incident-recovery"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id_: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id_,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id_}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    # A growth chain sized like the real production shape (804
    # incremental full-snapshot captures of one growing Codex rollout
    # file, reduced here for test speed -- the mechanism does not depend
    # on the absolute count, only on the accepted head carrying MANY
    # messages, not a token 2-3 like the small #2718 pin).
    native_id = "019f49d8-shape-fixture"
    growth_generations = 25
    base_texts = tuple(f"growth-generation-{index:04d}" for index in range(growth_generations))
    current_session = session(native_id, *base_texts)
    # The "incident recovery" duplicate: content-prefix growth alone is
    # not proof of provenance (revision_governance.py's own polylogue-miwv
    # note), so a same-identity bundle that strictly extends the accepted
    # head's content must still be evaluated through membership
    # governance, not silently accepted. Bundled alongside an unrelated
    # second session in one file -- the real incident-recovery backup
    # grabbed multiple sessions in one sweep, and a multi-session raw
    # unconditionally routes through membership governance
    # (``LiveBatchProcessor._ingest_full_paths_sync``'s ``len(sessions) !=
    # 1`` branch), which is what makes ``raw_revision_head_raw_id``'s
    # unconditional cohort injection reachable for a single-session-per-
    # file Codex identity like this one -- exactly how the real 804-row
    # session hit it despite Codex normally writing one session per file.
    recovered_extension = session(native_id, *base_texts, "growth-generation-0025", "growth-generation-0026")
    recovered_unrelated = session("019f49d8-unrelated-safe-session", "one")

    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream",
        _retained_parse_by_path(
            lambda path: [recovered_extension, recovered_unrelated] if path == incident_recovery else [current_session]
        ),
    )

    assert run_ingest_files(processor, [current], emit_event=False).failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        head_row = conn.execute(
            "SELECT accepted_raw_id, session_id FROM raw_revision_heads WHERE logical_source_key = ?",
            (f"codex-session:{native_id}",),
        ).fetchone()
        assert head_row is not None
        accepted_raw_id, session_id = head_row
        message_count_before = conn.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    assert message_count_before == growth_generations

    # A dangling, unresolved QUARANTINED append fragment hanging off the
    # CURRENT accepted head's own source_revision -- the live-append-
    # cursor evidence the guard exists to protect (mirrors
    # test_bundle_replay_respects_unconvertible_single_session_head's
    # ``append_raw_id`` setup) -- plus an explicit prior census of the
    # accepted head (``census_head``), matching that test's reliably
    # guard-triggering combination.
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        head_source_revision = (
            archive._ensure_source_conn()
            .execute(
                "SELECT source_revision FROM raw_sessions WHERE raw_id = ?",
                (accepted_raw_id,),
            )
            .fetchone()[0]
        )
        dangling_append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"response_item","payload":{"type":"message","id":"dangling"}}\n',
            source_path=str(current),
            canonical_source_path=str(current.resolve()),
            source_index=-1,
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            dangling_append_raw_id,
            RawRevisionEnvelope(
                f"codex-session:{native_id}",
                RawRevisionKind.APPEND,
                "dangling-append-blocker",
                0,
                predecessor_source_revision=str(head_source_revision),
                append_start_offset=1,
                append_end_offset=2,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        # Deliberately no prior ``replace_raw_membership_census`` call here
        # (unlike ``test_bundle_replay_...``'s ``census_head=True`` case):
        # the real production identity was governed purely through typed
        # byte-revision authority (``bind_raw_revision``/live-watch
        # classification), never through an explicit membership census of
        # its own accepted head. Adding one here would permanently divert
        # every later reprocessing of ``current`` through membership
        # governance instead of the plain byte-chain replay path, which
        # does not match the real shape and would make the eventual
        # recovery below impossible to reproduce faithfully.

    conflict_result = run_ingest_files(processor, [incident_recovery], emit_event=False)

    # Fail-closed is correct here: the guard must refuse to silently
    # replace a head with unresolved byte-append evidence hanging off it.
    assert conflict_result.failed_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        (parse_error,) = conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(incident_recovery),),
        ).fetchone()
    assert parse_error is not None
    # The typed evidence is authoritative for new rows. The recognized prefix
    # remains a bounded compatibility bridge for this historical diagnostic.
    assert parse_error.startswith("MembershipReplayConflictError:")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT a.artifact_kind
            FROM raw_artifacts AS a
            JOIN raw_sessions AS r ON r.raw_id = a.raw_id
            WHERE r.source_path = ?
            """,
            (str(incident_recovery),),
        ).fetchone() == ("deferred_cas_frontier",)
    from polylogue.storage.derived.raw import raw_replay_error_is_retryable

    assert raw_replay_error_is_retryable(parse_error) is True
    assert raw_replay_error_is_retryable("RuntimeError: unrelated parser failure") is False
    assert raw_replay_error_is_retryable(parse_error, True) is True

    with sqlite3.connect(index_db) as conn:
        assert (
            conn.execute("SELECT message_count FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
            == message_count_before
        )

    # The interfering condition is transient by construction, not
    # permanent, per ``MembershipReplayConflictError``'s own docstring: "a
    # later pass over the same durable bytes can succeed once sibling
    # evidence resolves or the accepted head itself changes". This is
    # exactly what the live archive's own EMPTY raw_revision_heads row for
    # this identity shows already happened (confirmed read-only,
    # 2026-08-03): whatever accepted-head state interfered with the
    # original 2026-07-10 attempt is gone today, so a fresh classification
    # pass hits the guard's ``existing_head is not None`` precondition
    # never at all and proceeds straight to indexing. Simulate that same
    # cleared state directly (the dangling append fragment bound above is
    # permanently unresolvable -- its byte offsets never correspond to any
    # real content, so no further real ingest can ever promote it; the
    # accepted head itself must be retired, matching
    # ``release_provisional_full_revisions``'s existing "provisional
    # evidence rejected" shape for full revisions). The live identity had
    # neither a head nor a session row: an ungoverned session left behind
    # would be incomparable Index state that replay refuses to adopt.
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "DELETE FROM raw_revision_heads WHERE logical_source_key = ?",
            (f"codex-session:{native_id}",),
        )
        conn.commit()
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        assert archive.delete_sessions((session_id,)) == 1

    # This is the AC#2 assertion: once the accepted head no longer
    # interferes, a retry over the SAME durable incident-recovery raw
    # succeeds and reaches the index with a plausible message_count --
    # note this exercises the live-watcher's own retry path
    # (``_ingest_full_paths_sync`` again), not
    # ``storage/raw_convergence.py``'s offline ``converge_raw_materialization``:
    # that offline path reprocesses every retained typed-'full' raw for
    # this logical_source_key on every pass (including ``current``'s own
    # cohort), which re-establishes an accepted head before ever reaching
    # ``incident_recovery`` in the same pass and so cannot demonstrate
    # this recovery in isolation here -- a real gap worth a follow-up
    # bead, not one this test's fixture can respect the scope of.
    retry_result = run_ingest_files(processor, [incident_recovery], emit_event=False)
    assert retry_result.failed_file_count == 0
    assert retry_result.succeeded_file_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        (retried_parse_error,) = conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms DESC LIMIT 1",
            (str(incident_recovery),),
        ).fetchone()
    assert retried_parse_error is None

    with sqlite3.connect(index_db) as conn:
        final_count = conn.execute("SELECT message_count FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[0]
    # Plausible: the incident-recovery bundle's own extension, 2
    # generations past the pre-conflict head.
    assert final_count == growth_generations + 2


def test_single_session_full_cannot_overwrite_divergent_membership_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "bundle.jsonl"
    divergent = root / "divergent.jsonl"
    bundle.write_bytes(_codex_shaped_bytes("bundle"))
    divergent.write_bytes(_codex_shaped_bytes("divergent"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    bundle_sessions = [session("shared", "base", "left"), session("safe", "one")]
    divergent_sessions = [session("shared", "base", "right", "extra")]
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream",
        _retained_parse_by_path(lambda path: bundle_sessions if path == bundle else divergent_sessions),
    )

    assert run_ingest_files(processor, [bundle], emit_event=False).failed_file_count == 0
    divergent_result = run_ingest_files(processor, [divergent], emit_event=False)

    # Divergence remains fail-closed for the materialized head, but the source
    # bytes were acquired and parsed successfully. Treating this as a cursor
    # success prevents each daemon restart from reprocessing the same decided
    # conflict until the file actually changes.
    assert divergent_result.succeeded_file_count == 1
    assert divergent_result.failed_file_count == 0
    # Direct check of the persisted state backing "cursor success" above:
    # this layer (``_ingest_full_paths_sync``) has no CursorStore row of its
    # own, so the durable non-retry evidence is raw_sessions.parse_error
    # staying NULL for the divergent raw despite the fail-closed membership
    # decision. A regression that started marking this a parse failure
    # would make the daemon reprocess the same decided-ambiguous divergence
    # on every restart, which is exactly what this comment says must not
    # happen.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ?",
            (str(divergent),),
        ).fetchone() == (None,)
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            """
            SELECT m.position, b.search_text
            FROM messages AS m
            JOIN blocks AS b USING (message_id)
            WHERE m.session_id = 'codex-session:shared'
            ORDER BY m.position, b.position
            """
        ).fetchall() == [(0, "base"), (1, "left")]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT m.decision, r.parsed_at_ms, r.parse_error
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'codex-session:shared'
            """,
            (str(divergent),),
        ).fetchone() == ("ambiguous", None, None)


def test_single_session_full_advances_authorized_metadata_only_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    bundle = root / "bundle.jsonl"
    metadata_update = root / "metadata-update.jsonl"
    bundle.write_bytes(_codex_shaped_bytes("bundle"))
    metadata_update.write_bytes(_codex_shaped_bytes("metadata-update"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, title: str, updated_at: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            title=title,
            updated_at=updated_at,
            messages=[ParsedMessage(provider_message_id=f"{native_id}-0", role=Role.USER, text="same content")],
        )

    older = session("shared", "old title", "2026-01-01T00:00:00Z")
    newer = session("shared", "new title", "2026-01-02T00:00:00Z")
    bundle_sessions = [older, session("safe", "safe", "2026-01-01T00:00:00Z")]
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream",
        _retained_parse_by_path(lambda path: bundle_sessions if path == bundle else [newer]),
    )

    assert run_ingest_files(processor, [bundle], emit_event=False).failed_file_count == 0
    update_result = run_ingest_files(processor, [metadata_update], emit_event=False)

    assert update_result.succeeded_file_count == 1
    assert update_result.failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            """
            SELECT s.title, s.updated_at_ms, h.accepted_frontier_kind,
                   h.accepted_content_hash = s.content_hash
            FROM sessions AS s
            JOIN raw_revision_heads AS h USING (session_id)
            WHERE s.native_id = 'shared'
            """
        ).fetchone() == ("new title", 1767312000000, "semantic", 1)


def test_bundle_promotes_prior_single_full_into_membership_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    single = root / "single.jsonl"
    bundle = root / "bundle.jsonl"
    single.write_bytes(_codex_shaped_bytes("single"))
    bundle.write_bytes(_codex_shaped_bytes("bundle"))
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    def session(native_id: str, *texts: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=native_id,
            messages=[
                ParsedMessage(provider_message_id=f"{native_id}-{index}", role=Role.USER, text=text)
                for index, text in enumerate(texts)
            ],
        )

    single_session = session("shared", "base")
    bundle_sessions = [session("shared", "base", "new"), session("safe", "one")]
    monkeypatch.setattr(
        "polylogue.sources.live.batch_support._jsonl_provider_and_session_artifact",
        lambda _path, fallback_provider, **_kwargs: (fallback_provider, True, None),
    )
    monkeypatch.setattr(
        "polylogue.sources.prepared_jsonl.iter_parsed_stream",
        _retained_parse_by_path(lambda path: [single_session] if path == single else bundle_sessions),
    )

    assert run_ingest_files(processor, [single], emit_event=False).failed_file_count == 0
    bundle_result = run_ingest_files(processor, [bundle], emit_event=False)

    assert bundle_result.succeeded_file_count == 1
    assert bundle_result.failed_file_count == 0
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            """
            SELECT s.message_count, h.accepted_frontier_kind
            FROM sessions AS s
            JOIN raw_revision_heads AS h USING (session_id)
            WHERE s.native_id = 'shared'
            """
        ).fetchone() == (2, "semantic")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        single_raw_id, decision = conn.execute(
            """
            SELECT r.raw_id, m.decision
            FROM raw_session_memberships AS m
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND m.logical_source_key = 'codex-session:shared'
            """,
            (str(single),),
        ).fetchone()
    assert decision == "superseded_prefix"
    # Promotion moves the prior full into membership authority alone: it no
    # longer rebuilds through its byte-revision chain.
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert archive.raw_revision_rebuild_logical_keys([single_raw_id]) == ()
        _membership_raws, membership_keys = archive.expand_raw_membership_selection([single_raw_id])
    assert "codex-session:shared" in membership_keys


def test_append_crash_after_index_commit_repairs_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    _path, plan, owner, _processor = _seed_live_append_plan(tmp_path, native_id="crash-retry")
    # The append replay commits its Index publication, then publishes the
    # Source acknowledgement that marks the raw parsed. Crash between them.
    original_publish_source = archive_revision_governance.publish_prepared_revision_source
    crashed = False

    def crash_after_index(seal: Any, permit: Any) -> None:
        nonlocal crashed
        with sqlite3.connect(tmp_path / "index.db") as index:
            index_committed = index.execute("SELECT 1 FROM messages WHERE native_id = 'message-1'").fetchone()
        if index_committed is not None and not crashed:
            crashed = True
            raise SimulatedProcessCrash
        original_publish_source(seal, permit)

    monkeypatch.setattr(archive_revision_governance, "publish_prepared_revision_source", crash_after_index)
    with pytest.raises(SimulatedProcessCrash):
        ingest_append_with_owner(owner, [plan])

    assert _append_raw_parse_state(tmp_path) == (None, None)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2

    monkeypatch.setattr(archive_revision_governance, "publish_prepared_revision_source", original_publish_source)
    retry = ingest_append_with_owner(owner, [plan])

    assert retry.succeeded == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 2
    parsed_at_ms, parse_error = _append_raw_parse_state(tmp_path)
    assert parsed_at_ms is not None
    assert parse_error is None
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_append_ingest_bootstraps_archive_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-bootstrap.jsonl"
    payload = (
        b'{"type":"session_meta","payload":{"id":"append-bootstrap","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}\n'
    )
    path.write_bytes(payload)
    cursor = CursorStore(tmp_path / "append.sqlite")

    class Owner:
        _cursor = cursor
        _polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))

    stat = path.stat()
    plan = _AppendPlan(
        path=path,
        canonical_source_path=str(path),
        captured_profile_key=None,
        source_name="codex",
        start_offset=0,
        last_complete_newline=stat.st_size,
        stat_size=stat.st_size,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        payload=payload,
        payload_hash="payload-hash",
        cursor_fingerprint="base",
        bytes_read=len(payload),
        native_id_hint="append-bootstrap",
    )

    result = ingest_append_with_owner(Owner(), [plan])

    assert result.succeeded == []
    assert result.deferred == [plan]
    assert result.failed == []
    for filename in (spec.filename for spec in ARCHIVE_TIER_SPECS.values()):
        assert (tmp_path / filename).exists()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_live_raw_compaction_ignores_cursor_db_without_source_db(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    cursor_db = tmp_path / "live.sqlite"
    cursor = CursorStore(cursor_db)
    with cursor._connect() as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_index INTEGER NOT NULL,
                blob_size INTEGER NOT NULL,
                acquired_at TEXT NOT NULL
            );
            INSERT INTO raw_sessions
                (raw_id, source_path, source_index, blob_size, acquired_at)
            VALUES
                ('raw-old', '/tmp/old.jsonl', 0, 10, '2026-01-01T00:00:00+00:00');
            """
        )
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    _compact_on_admitted_writer(processor, [path])
    with cursor._connect() as conn:
        rows = conn.execute("SELECT raw_id FROM raw_sessions").fetchall()
    assert rows == [("raw-old",)]


@pytest.mark.asyncio
async def test_live_full_ingest_skips_convergence_without_session_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cursor-only raw observation must not rerun global workflow materializers."""
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "unchanged.json"
    path.write_text("{}", encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))),
        (WatchSource(name="sessions", root=root, layout=export_drop_layout((".json",))),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    convergence_calls: list[list[Path]] = []

    async def fake_full_ingest(
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: object | None = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
    ) -> _FullIngestResult:
        del source_name, heartbeat, attempt_id, max_pass_seconds, pass_started
        return _FullIngestResult(
            succeeded=paths,
            failed=[],
            source_payload_read_bytes=0,
            raw_fingerprints={path: "raw-unchanged"},
            changed_session_count=0,
        )

    def record_convergence(paths: list[Path]) -> tuple[set[Path], float, dict[str, float], list[object], list[object]]:
        convergence_calls.append(paths)
        return set(paths), 0.0, {}, [], []

    def fake_append_plan(
        _path: Path,
        *,
        cursor: object | None = None,
        cursor_is_known: bool = False,
        source_index: int = -1,
    ) -> None:
        del cursor, cursor_is_known, source_index

    monkeypatch.setattr(processor, "_append_plan", fake_append_plan)
    monkeypatch.setattr(processor, "_ingest_full_paths", fake_full_ingest)
    monkeypatch.setattr(processor, "_converge_paths", record_convergence)
    monkeypatch.setattr(processor, "_record_full_cursor", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(processor, "_compact_superseded_raw_snapshots", lambda _paths: None)

    metrics = await ingest_files_with_owners(processor, [path], emit_event=False)

    assert convergence_calls == []
    assert metrics.succeeded_file_count == 1
    assert metrics.changed_session_count == 0


@pytest.mark.asyncio
async def test_live_append_plans_flush_in_bounded_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"{index}.jsonl" for index in range(5)]
    for path in paths:
        path.write_text('{"type":"session_meta","payload":{"id":"bounded"}}\n', encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    groups: list[list[Path]] = []

    def fake_append_plan(
        path: Path,
        *,
        cursor: object | None = None,
        cursor_is_known: bool = False,
        source_index: int = -1,
    ) -> _AppendPlan:
        del cursor, cursor_is_known
        return _AppendPlan(
            path=path,
            canonical_source_path=str(path),
            captured_profile_key=None,
            source_name="codex",
            start_offset=0,
            last_complete_newline=10,
            stat_size=10,
            st_dev=1,
            st_ino=1,
            mtime_ns=1,
            payload=b"payload\n",
            payload_hash="tail",
            cursor_fingerprint="base",
            bytes_read=10,
            source_index=source_index,
        )

    async def fake_append_runner(_owner: object, plans: list[_AppendPlan]) -> _AppendResult:
        groups.append([plan.path for plan in plans])
        return _AppendResult(succeeded=list(plans), failed=[], worker_count=1)

    monkeypatch.setattr(processor, "_append_plan", fake_append_plan)
    monkeypatch.setattr(processor, "_append_runner", fake_append_runner)
    monkeypatch.setattr(
        processor,
        "_converge_paths",
        lambda paths, **kwargs: (paths, 0.0, {}, [], []),
    )
    monkeypatch.setattr(processor, "_record_append_cursor", lambda plan: True)
    monkeypatch.setattr(processor, "_record_convergence_outcomes", lambda outcomes, settlements: None)
    monkeypatch.setattr("polylogue.sources.live.batch._append_plan_group_ready", lambda plans: len(plans) >= 2)

    metrics = await ingest_files_with_owners(processor, paths, emit_event=False)

    assert groups == [paths[:2], paths[2:4], paths[4:]]
    assert metrics.append_file_count == 5
    assert metrics.full_file_count == 0
    with sqlite3.connect(cursor._ops_db_path) as conn:
        stage_payloads = [
            (str(row[0]), json.loads(row[1]))
            for row in conn.execute(
                """
                SELECT stage, payload_json
                FROM daemon_stage_events
                WHERE stage IN ('append_parse', 'convergence', 'cursor_update', 'completed')
                ORDER BY observed_at_ms, rowid
                """
            ).fetchall()
        ]
    route_payloads = [(stage, payload) for stage, payload in stage_payloads if payload.get("storage_route")]
    assert route_payloads
    assert {payload["storage_route"] for _, payload in route_payloads} == {"archive_append"}
    assert ("cursor_update", "archive_append") in [
        (stage, str(payload.get("storage_route"))) for stage, payload in route_payloads
    ]
    assert ("completed", "archive_append") in [
        (stage, str(payload.get("storage_route"))) for stage, payload in route_payloads
    ]


def _gemini_cli_checkpoint(padding: str) -> dict[str, Any]:
    return {
        "sessionId": "gemini-large-1",
        "projectHash": "project-hash",
        "startTime": "2026-03-16T09:40:00.000Z",
        "lastUpdated": "2026-03-16T11:01:00.000Z",
        "kind": "chat",
        "summary": "Large checkpoint",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2026-03-16T09:40:01.000Z",
                "type": "user",
                "content": ["review this transcript"],
            },
            {
                "id": "a1",
                "timestamp": "2026-03-16T09:40:02.000Z",
                "type": "gemini",
                "content": padding,
                "model": "gemini-test",
            },
        ],
    }


def test_gemini_cli_checkpoint_over_the_streaming_bound_reaches_the_archive(tmp_path: Path) -> None:
    """A Gemini CLI checkpoint is retained and parsed above the former bound."""
    root = tmp_path / "chats"
    source = root / "session-2026-03-16T09-40-5c12869b.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(_gemini_cli_checkpoint("y" * (_RETIRED_FULL_INGEST_SIZE_BOUND + 1024))),
        encoding="utf-8",
    )
    assert source.stat().st_size > _RETIRED_FULL_INGEST_SIZE_BOUND

    cursor = CursorStore(tmp_path / "index.db")
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="gemini-cli", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    result = run_ingest_files(processor, [source], emit_event=False)

    assert result.failed_file_count == 0
    assert result.excluded_file_count == 0
    assert result.ingested_session_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE origin = 'gemini-cli-session'").fetchone() == (1,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
    with sqlite3.connect(cursor._ops_db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ingest_cursor WHERE excluded = 1").fetchone() == (0,)


#: One whole-document ``.json`` session per provider that can exceed the
#: streaming bound. Keyed on the provider the acquisition route resolves; the
#: expected admission is read from the payload predicate, never restated.
_LARGE_JSON_SESSION_DOCUMENTS: dict[Provider, Any] = {
    Provider.GEMINI_CLI: _gemini_cli_checkpoint("padded reply"),
    Provider.CHATGPT: [
        {
            "id": "conv-1",
            "title": "Export conversation",
            "create_time": 1767225600.0,
            "mapping": {
                "u1": make_chatgpt_node("u1", "user", ["hello"], children=["a1"]),
                "a1": make_chatgpt_node("a1", "assistant", ["reply"], parent="u1"),
            },
        }
    ],
    Provider.CLAUDE_AI: [
        {
            "uuid": "claude-conv-1",
            "name": "Export conversation",
            "created_at": "2026-03-16T09:40:00.000000Z",
            "chat_messages": [
                make_claude_chat_message("cm1", "human", "hello"),
                make_claude_chat_message("cm2", "assistant", "reply"),
            ],
        }
    ],
    Provider.GEMINI: {
        "runSettings": {"model": "models/gemini-test"},
        "systemInstruction": {},
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "hello"},
                {"role": "model", "text": "reply"},
            ]
        },
    },
    Provider.DRIVE: {
        "runSettings": {"model": "models/gemini-test"},
        "systemInstruction": {},
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "hello"},
                {"role": "model", "text": "reply"},
            ]
        },
    },
}


@pytest.mark.parametrize("provider", sorted(_LARGE_JSON_SESSION_DOCUMENTS))
def test_json_session_admission_does_not_depend_on_file_size(
    provider: Provider,
    tmp_path: Path,
) -> None:
    """A supported JSON document remains eligible above the former size boundary."""
    document = _LARGE_JSON_SESSION_DOCUMENTS[provider]
    target = tmp_path / "chats" / "session.json"
    target.parent.mkdir(parents=True)
    payload = json.dumps(document).encode("utf-8")
    target.write_bytes(payload)

    # The witness must itself be a session, or the parity below is vacuous.
    assert _parse_path_as_session_artifact(target, provider=provider) is True

    target.write_bytes(payload + b" " * max(0, _RETIRED_FULL_INGEST_SIZE_BOUND + 1024 - len(payload)))
    assert target.stat().st_size > _RETIRED_FULL_INGEST_SIZE_BOUND
    assert _parse_path_as_session_artifact(target, provider=provider) is True


def test_codex_state_filename_alone_does_not_route_a_foreign_file_to_codex_acquisition(tmp_path: Path) -> None:
    """polylogue-bzx7h: the Codex state-db branch requires the Codex fallback provider.

    A file merely named ``state_5.sqlite`` under a non-Codex source used to be
    acquired as Codex SQLite state before generic detection ran.
    """
    bootstrap_archive_root(tmp_path)
    root = tmp_path / "inbox"
    state_db = root / "state_5.sqlite"
    _write_plain_sqlite_db(state_db)
    index_db = tmp_path / "index.db"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="inbox", root=root, layout=export_drop_layout((".sqlite",))),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    result = _full_paths_sync(processor, [state_db], source_name="inbox")

    assert result.failed == []
    with sqlite3.connect(tmp_path / "source.db") as conn:
        origins = [row[0] for row in conn.execute("SELECT origin FROM raw_sessions").fetchall()]
    assert not any(str(origin).startswith("codex") for origin in origins)


def _live_processor(tmp_path: Path, root: Path, *, source_name: str) -> tuple[LiveBatchProcessor, CursorStore]:
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name=source_name, root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    return processor, cursor


def test_json_document_provider_detection_does_not_depend_on_file_size(tmp_path: Path) -> None:
    """A generic inbox detects a provider from a document above the former size boundary."""
    root = tmp_path / "chats"
    source = root / "session-2026-03-16T09-40-5c12869b.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(_gemini_cli_checkpoint("y" * (_RETIRED_FULL_INGEST_SIZE_BOUND + 1024))),
        encoding="utf-8",
    )
    assert source.stat().st_size > _RETIRED_FULL_INGEST_SIZE_BOUND
    processor, cursor = _live_processor(tmp_path, root, source_name="inbox")

    result = run_ingest_files(processor, [source], emit_event=False)

    assert result.ingested_session_count == 1
    assert result.excluded_file_count == 0

    record = cursor.get_record(source)
    assert record is not None
    assert record.byte_offset == source.stat().st_size
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT origin FROM raw_sessions").fetchone() == ("gemini-cli-session",)


def test_hold_budget_spent_after_the_commit_still_records_the_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admitted work finishes and publishes its cursor past diagnostic thresholds."""
    from polylogue.core.write_hold import enter_write_hold, exit_write_hold

    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "hold-budget.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"hold-budget"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    processor, cursor = _live_processor(tmp_path, root, source_name="codex")

    token = enter_write_hold("watcher.live_ingest.full", 0)
    try:
        first = run_ingest_files(processor, [path], emit_event=False)
    finally:
        exit_write_hold(token)

    assert first.succeeded_file_count == 1
    assert first.ingested_session_count == 1
    record = cursor.get_record(path)
    assert record is not None
    assert record.byte_offset == path.stat().st_size
    assert record.content_fingerprint is not None
    assert not record.excluded

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE native_id = 'hold-budget'").fetchone() == (1,)


def test_a_deferred_append_records_deferred_debt_not_only_a_failed_receipt_count(
    tmp_path: Path,
) -> None:
    """A deferral must be readable as a deferral, not as a failure or a no-op.

    The attempt receipt folds deferred paths into ``failed_file_count`` and
    ``LiveBatchMetrics`` counts them nowhere, so a deferred pass was
    distinguishable from neither failure nor success (polylogue-3r36h).
    Dropping the ``live_ingest_deferred`` debt row turns this red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "deferred.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"deferred-append"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    processor, cursor = _live_processor(tmp_path, root, source_name="codex")
    assert run_ingest_files(processor, [path], emit_event=False).succeeded_file_count == 1

    # A partial trailing record carries no complete-newline frontier, so the
    # append planner defers instead of admitting it.
    with path.open("ab") as handle:
        handle.write(b'{"type":"response_item","payload":{"type":"message","id":"message-1"')

    deferred = run_ingest_files(processor, [path], emit_event=False)

    assert deferred.succeeded_file_count == 0
    with sqlite3.connect(cursor._ops_db_path) as conn:
        rows = conn.execute(
            "SELECT target_id, status FROM convergence_debt WHERE stage = 'live_ingest_deferred'"
        ).fetchall()
    assert rows == [(str(path), "deferred")]


def test_a_deferred_pass_reports_deferral_as_its_own_count_not_as_failures(
    tmp_path: Path,
) -> None:
    """A deferral is bounded backpressure; the receipt must not call it failed.

    ``failed_file_count`` is what daemon status and catch-up status show the
    operator, and it used to include every deferred path while
    ``LiveBatchMetrics`` reported the deferral nowhere at all
    (polylogue-3r36h).

    Anti-vacuity: fold ``len(deferred_paths)`` back into the receipt's
    ``failed_file_count``, or drop ``deferred_file_count`` from
    ``LiveBatchMetrics.to_payload``, and this is red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "deferred-count.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"deferred-count"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    processor, cursor = _live_processor(tmp_path, root, source_name="codex")
    assert run_ingest_files(processor, [path], emit_event=False).succeeded_file_count == 1

    with path.open("ab") as handle:
        handle.write(b'{"type":"response_item","payload":{"type":"message","id":"message-1"')

    deferred = run_ingest_files(processor, [path], emit_event=False)

    assert deferred.deferred_paths == (str(path),)
    assert deferred.deferred_file_count == 1
    assert deferred.failed_file_count == 0
    assert deferred.to_payload()["deferred_file_count"] == 1
    assert deferred.to_payload()["failed_file_count"] == 0

    with sqlite3.connect(cursor._ops_db_path) as conn:
        payloads = [
            json.loads(row[0])
            for row in conn.execute(
                "SELECT payload_json FROM daemon_stage_events WHERE stage = 'completed' ORDER BY observed_at_ms"
            ).fetchall()
        ]
    assert payloads, "no completed stage event recorded"
    final = payloads[-1]
    assert final["deferred_file_count"] == 1
    assert final.get("failed_file_count", 0) == 0


# ── Raw retention: the recurring owner's bounded retry (polylogue-6kur AC5) ──


def _seed_superseded_raw_snapshots(
    processor: LiveBatchProcessor,
    source_db: Path,
    source_path: Path,
    *,
    count: int,
    prefix: int = 0,
) -> list[str]:
    """Write ``count`` superseded append snapshots plus one surviving head.

    ``source_index = -1`` is the append lane, the one
    ``compact_paths_superseded_raw_snapshots`` compacts: it passes
    ``keep_full_snapshots=1_000_000`` on purpose, so full snapshots are never
    its subject. Acquisition times are anchored on the processor's own
    ``_raw_compaction_min_acquired_at`` floor, which is the production rule --
    retention only compacts what this watcher itself acquired.
    """
    from polylogue.core.timestamps import to_epoch_ms

    floor_ms = to_epoch_ms(processor._raw_compaction_min_acquired_at, numeric_unit="milliseconds")
    assert floor_ms is not None
    raw_ids: list[str] = []
    with closing(sqlite3.connect(source_db)) as conn:
        for index in range(count + 1):
            raw_id = f"{prefix + index:064x}"
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index,
                    blob_hash, blob_size, acquired_at_ms
                ) VALUES (?, 'codex-session', ?, ?, -1, ?, ?, ?)
                """,
                (raw_id, raw_id, str(source_path), bytes.fromhex(raw_id), 10, floor_ms + index),
            )
            raw_ids.append(raw_id)
        conn.commit()
    # The newest row is the surviving head; the rest are superseded.
    return raw_ids[:-1]


def _compact_on_admitted_writer(processor: LiveBatchProcessor, paths: list[Path]) -> None:
    """Run raw compaction on the daemon's admitted writer, as the live pass dispatches it.

    ``LiveBatchProcessor`` hands ``_compact_superseded_raw_snapshots`` to its
    writer runner; the custody authorizer refuses its Source deletions from
    any other creator.
    """
    archive_root = Path(getattr(processor._polylogue, "archive_root", processor._cursor._db_path.parent))
    asyncio.run(run_archive_fixture_write(archive_root, lambda: processor._compact_superseded_raw_snapshots(paths)))


def _retention_processor(tmp_path: Path, root: Path) -> LiveBatchProcessor:
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "ops.db"),
        parser_fingerprint="test-parser",
    )


def _retention_debt(cursor: CursorStore) -> list[Any]:
    return [
        debt
        for debt in cursor.list_convergence_debt(limit=50, stage=RAW_RETENTION_STAGE)
        if debt.subject_type == "source_path"
    ]


def _grant_full_retention_authority(monkeypatch: pytest.MonkeyPatch, raw_ids: list[str]) -> None:
    from polylogue.storage import raw_retention

    monkeypatch.setattr(
        raw_retention,
        "active_raw_retention_authority",
        lambda *_a, **_k: raw_retention.RawRetentionAuthority(
            protected_raw_ids=frozenset(),
            eligible_raw_ids=frozenset(raw_ids),
        ),
    )


def test_raw_retention_bound_is_retained_as_retryable_backlog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bounded retention pass names its remainder instead of dropping it.

    Raw retention is one of polylogue-6kur AC5's four storage domains and it is
    the one that had no bounded retry: a pass compacted at most
    ``RAW_RETENTION_LIMIT_PER_PATH`` snapshots per path and the remainder was
    left with no record that anything still owed it. It now lands in the
    ordinary ``convergence_debt`` ledger, which carries the attempt count and
    the shared exponential backoff.

    Anti-vacuity: stop populating ``residual_source_paths`` in
    ``compact_paths_superseded_raw_snapshots`` (or drop the
    ``_record_raw_retention_outcome`` call) and the debt assertions go red
    while the pass still reports the same deletions.
    """
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    processor = _retention_processor(tmp_path, root)
    superseded = _seed_superseded_raw_snapshots(processor, tmp_path / "source.db", path, count=30)
    _grant_full_retention_authority(monkeypatch, superseded)

    _compact_on_admitted_writer(processor, [path])
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
    # 31 rows seeded, 30 superseded, one bounded pass compacts 25.
    assert remaining == 31 - RAW_RETENTION_LIMIT_PER_PATH

    debt = _retention_debt(processor._cursor)
    assert [(item.subject_id, item.status) for item in debt] == [(str(path), "deferred")]
    assert f"bounded at {RAW_RETENTION_LIMIT_PER_PATH}" in (debt[0].last_error or "")
    assert debt[0].failure_count == 1


def test_raw_retention_drains_its_due_backlog_and_clears_the_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner comes back to what its last bounded pass did not reach.

    The backlog path is not in the next pass's own subjects: it is found
    through the retry-due ``raw_retention`` debt rows, which is what makes this
    a recurring owner with bounded retry rather than a one-shot best effort.

    Anti-vacuity: drop the ``_raw_retention_backlog_paths`` extension of
    ``scoped_paths`` and the remaining rows survive and the debt row stays.
    """
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    backlog_path = root / "backlog.jsonl"
    backlog_path.write_text("{}\n", encoding="utf-8")
    other_path = root / "other.jsonl"
    other_path.write_text("{}\n", encoding="utf-8")
    processor = _retention_processor(tmp_path, root)
    superseded = _seed_superseded_raw_snapshots(processor, tmp_path / "source.db", backlog_path, count=30)
    _grant_full_retention_authority(monkeypatch, superseded)

    _compact_on_admitted_writer(processor, [backlog_path])
    assert [item.subject_id for item in _retention_debt(processor._cursor)] == [str(backlog_path)]

    # The shared backoff put the row ~60 s out. Make it due, the way the clock
    # would, without touching anything else the owner reads.
    with closing(sqlite3.connect(tmp_path / "ops.db")) as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = ? WHERE stage = ?",
            ("2000-01-01T00:00:00+00:00", RAW_RETENTION_STAGE),
        )
        conn.commit()

    # A pass whose own subject is an unrelated path still drains the backlog.
    _compact_on_admitted_writer(processor, [other_path])
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
    assert remaining == 1
    assert _retention_debt(processor._cursor) == []


def test_raw_retention_refusal_is_recorded_not_only_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsafe-evidence refusal leaves retryable debt, not just a log line.

    Anti-vacuity: restore the bare ``return`` after the
    ``RawRetentionSafetyError`` warning and this reports no debt at all, which
    is exactly the state AC5 called unproven -- work owed with no owner
    recorded anywhere.
    """
    from polylogue.storage import raw_retention
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    processor = _retention_processor(tmp_path, root)
    _seed_superseded_raw_snapshots(processor, tmp_path / "source.db", path, count=2)

    def refuse(*_args: object, **_kwargs: object) -> raw_retention.RawRetentionAuthority:
        raise raw_retention.RawRetentionSafetyError("index has no raw authority")

    monkeypatch.setattr(raw_retention, "active_raw_retention_authority", refuse)

    _compact_on_admitted_writer(processor, [path])
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 3

    debt = _retention_debt(processor._cursor)
    assert [(item.subject_id, item.status) for item in debt] == [(str(path), "failed")]
    assert "index has no raw authority" in (debt[0].last_error or "")


def test_raw_retention_waits_for_inactive_generation_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate index rows cannot authorize deletion before promotion."""
    from polylogue.sources.live import cold_build
    from polylogue.storage import raw_retention
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    processor = _retention_processor(tmp_path, root)
    superseded = _seed_superseded_raw_snapshots(processor, tmp_path / "source.db", path, count=2)
    candidate = object()
    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: candidate)

    def wrong_active_authority(*_args: object, **_kwargs: object) -> raw_retention.RawRetentionAuthority:
        raise AssertionError("active index authority was inspected before candidate promotion")

    monkeypatch.setattr(raw_retention, "active_raw_retention_authority", wrong_active_authority)
    _compact_on_admitted_writer(processor, [path])
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 3
    debt = _retention_debt(processor._cursor)
    assert [(item.subject_id, item.status) for item in debt] == [(str(path), "deferred")]
    assert "until inactive index generation is promoted" in (debt[0].last_error or "")

    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: None)
    _grant_full_retention_authority(monkeypatch, superseded)
    with closing(sqlite3.connect(tmp_path / "ops.db")) as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = ? WHERE stage = ?",
            ("2000-01-01T00:00:00+00:00", RAW_RETENTION_STAGE),
        )
        conn.commit()
    _compact_on_admitted_writer(processor, [])
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    assert _retention_debt(processor._cursor) == []


def test_raw_retention_retries_promoted_backlog_after_watcher_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new watcher drains recorded raw work from before its own start time."""
    from polylogue.sources.live import cold_build
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    first = _retention_processor(tmp_path, root)
    superseded = _seed_superseded_raw_snapshots(first, tmp_path / "source.db", path, count=2)
    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: object())
    _compact_on_admitted_writer(first, [path])
    assert [(item.subject_id, item.status) for item in _retention_debt(first._cursor)] == [(str(path), "deferred")]

    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: None)
    _grant_full_retention_authority(monkeypatch, superseded)
    with closing(sqlite3.connect(tmp_path / "ops.db")) as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = ? WHERE stage = ?",
            ("2000-01-01T00:00:00+00:00", RAW_RETENTION_STAGE),
        )
        conn.commit()
    restarted = _retention_processor(tmp_path, root)
    restarted._raw_compaction_min_acquired_at = "9999-01-01T00:00:00+00:00"
    _retry_retention_on_admitted_writer(tmp_path, restarted)

    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    assert _retention_debt(restarted._cursor) == []


def _retry_retention_on_admitted_writer(archive_root: Path, processor: LiveBatchProcessor) -> None:
    """Drive ``retry_raw_retention_backlog`` through a real daemon writer coordinator.

    The custody authorizer refuses archive writes outside admission, so the
    watcher is given the coordinator it holds in the daemon rather than an
    inline stand-in that would run the writes on the event-loop thread.
    """
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    async def run() -> None:
        coordinator = DaemonWriteCoordinator(archive_root=archive_root)
        watcher = object.__new__(LiveWatcher)
        watcher._batch_processor = processor
        watcher._ingest_lock = asyncio.Lock()
        watcher._write_coordinator = coordinator
        # The retry's Source body runs on the processor's writer runner.
        previous_runner = processor._sync_runner
        processor._sync_runner = coordinator.run_sync
        try:
            await watcher.retry_raw_retention_backlog()
        finally:
            processor._sync_runner = previous_runner
            if not await coordinator.shutdown(timeout=float("inf")):
                raise RuntimeError("retention retry coordinator did not physically settle")

    asyncio.run(run())


def test_raw_retention_retry_drains_more_than_one_bounded_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One retry call keeps draining while the due backlog moves.

    Anti-vacuity: a single pass per call drains one of the three paths (the
    page is patched to one path) and leaves two retention debts behind.
    """
    from polylogue.sources.live import batch as batch_module
    from polylogue.sources.live import cold_build
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    processor = _retention_processor(tmp_path, root)
    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: object())
    superseded: list[str] = []
    paths = []
    for index in range(3):
        path = root / f"session-{index}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        paths.append(path)
        superseded.extend(
            _seed_superseded_raw_snapshots(processor, tmp_path / "source.db", path, count=1, prefix=10 * index)
        )
    _compact_on_admitted_writer(processor, paths)
    assert len(_retention_debt(processor._cursor)) == 3

    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: None)
    _grant_full_retention_authority(monkeypatch, superseded)
    monkeypatch.setattr(batch_module, "RAW_RETENTION_BACKLOG_PER_PASS", 1)
    with closing(sqlite3.connect(tmp_path / "ops.db")) as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = ? WHERE stage = ?",
            ("2000-01-01T00:00:00+00:00", RAW_RETENTION_STAGE),
        )
        conn.commit()
    _retry_retention_on_admitted_writer(tmp_path, processor)

    assert _retention_debt(processor._cursor) == []


def test_raw_retention_backlog_does_not_widen_unrelated_batch_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical retention applies only to the path carrying recorded debt."""
    from polylogue.sources.live import cold_build
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    current_path = root / "current.jsonl"
    backlog_path = root / "backlog.jsonl"
    for path in (current_path, backlog_path):
        path.write_text("{}\n", encoding="utf-8")
    first = _retention_processor(tmp_path, root)
    old_current = _seed_superseded_raw_snapshots(first, tmp_path / "source.db", current_path, count=2, prefix=100)
    old_backlog = _seed_superseded_raw_snapshots(first, tmp_path / "source.db", backlog_path, count=2, prefix=200)
    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: object())
    _compact_on_admitted_writer(first, [backlog_path])
    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root: None)
    _grant_full_retention_authority(monkeypatch, [*old_current, *old_backlog])
    with closing(sqlite3.connect(tmp_path / "ops.db")) as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = ? WHERE stage = ?",
            ("2000-01-01T00:00:00+00:00", RAW_RETENTION_STAGE),
        )
        conn.commit()
    restarted = _retention_processor(tmp_path, root)
    restarted._raw_compaction_min_acquired_at = "9999-01-01T00:00:00+00:00"
    _compact_on_admitted_writer(restarted, [current_path])
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        counts = dict(conn.execute("SELECT source_path, COUNT(*) FROM raw_sessions GROUP BY source_path"))
    assert counts == {str(current_path): 3, str(backlog_path): 1}
    assert _retention_debt(restarted._cursor) == []


def test_deferred_cursor_records_when_the_tail_cannot_be_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable tail is a recorded deferral, never an escaping OSError.

    ``_defer_incomplete_jsonl_append`` catches its own probe failure and then
    calls this helper, which reopened the same file. Anti-vacuity: let the
    ``OSError`` propagate and this raises instead of writing a cursor row, so
    the watcher pass aborts and every later pass retries the identical read.
    """
    from polylogue.sources.live import deferred_cursor as deferred_cursor_module

    path = tmp_path / "session.jsonl"
    payload = b'{"a":1}\n'
    path.write_bytes(payload)
    store = CursorStore(tmp_path / "cursors.db")
    stat = path.stat()
    store.set(
        path,
        len(payload),
        byte_offset=len(payload),
        last_complete_newline=len(payload),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        tail_hash=_cursor_hash_authority(payload),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )
    before = store.get_record(path)
    assert before is not None

    def unreadable(*_args: Any, **_kwargs: Any) -> tuple[str, int]:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(deferred_cursor_module, "tail_hash_from_path", unreadable)

    deferred_cursor_module.record_deferred_append_cursor(
        store,
        path,
        cursor=before,
        parser_fingerprint="test-parser",
        source_name="chatgpt",
        deferred_end_offset=before.deferred_end_offset,
    )

    after = store.get_record(path)
    assert after is not None
    # The prior tail evidence is preserved rather than replaced by a digest
    # nothing could read.
    assert after.tail_hash == before.tail_hash
    assert after.byte_offset == before.byte_offset


@pytest.mark.asyncio
async def test_an_ordering_held_revision_stays_retryable_when_the_unit_ends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation leaves an ordering-held revision retryable and unpublished."""
    root = tmp_path / "sessions"
    root.mkdir()
    first, second = root / "revision-1.json", root / "revision-2.json"
    for path in (first, second):
        path.write_text("{}", encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path))),
        (WatchSource(name="sessions", root=root, layout=export_drop_layout((".json",))),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    published: list[list[Path]] = []
    deferred: list[Path] = []

    async def holding_full_ingest(paths: list[Path], **_kwargs: object) -> _FullIngestResult:
        published.append(list(paths))
        return _FullIngestResult(
            succeeded=[first],
            failed=[],
            source_payload_read_bytes=0,
            raw_fingerprints={first: "raw-first"},
            ordering_held=[second],
        )

    def fake_append_plan(_path: Path, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(processor, "_append_plan", fake_append_plan)
    monkeypatch.setattr(processor, "_ingest_full_paths", holding_full_ingest)
    monkeypatch.setattr(processor, "_converge_paths", lambda paths: (set(paths), 0.0, {}, [], []))
    monkeypatch.setattr(processor, "_record_full_cursor", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(processor, "_compact_superseded_raw_snapshots", lambda _paths: None)
    monkeypatch.setattr(processor, "_defer_full_cursor_retry", lambda path, **_kwargs: deferred.append(path))
    monkeypatch.setattr(processor, "_stop_requested", lambda: bool(published))

    metrics = await ingest_files_with_owners(processor, [first, second], emit_event=False)

    assert published == [[first, second]]
    assert metrics.succeeded_file_count == 1
    assert str(second) in metrics.deferred_paths
    assert deferred == [second]


@pytest.mark.parametrize("provider", ["codex", "claude-code"])
def test_append_publication_does_not_hide_growth_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """The prefix proof cannot bless later complete records as a probed tail."""
    native_id = "publication-growth"
    if provider == "codex":
        path, plan, owner, processor = _seed_live_append_plan(tmp_path, native_id=native_id)
        later = (
            b'{"type":"response_item","payload":{"type":"message","id":"message-2",'
            b'"role":"assistant","content":[{"type":"output_text","text":"two"}]}}\n'
        )
    else:

        def message(number: int) -> bytes:
            return (
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": f"message-{number}",
                        "parentUuid": f"message-{number - 1}",
                        "sessionId": native_id,
                        "timestamp": f"2026-06-02T00:00:0{number}Z",
                        "message": {"role": "assistant", "content": f"reply {number}"},
                    }
                )
                + "\n"
            ).encode()

        path, plan, owner, processor = _seed_claude_live_append_plan(
            tmp_path,
            native_id=native_id,
            append=message(1),
        )
        later = message(2)
    assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    with path.open("ab") as handle:
        handle.write(later)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, plan.mtime_ns + 1_000_000))

    assert processor._record_append_cursor(plan) is True
    recorded = processor._cursor.get_record(path)
    assert recorded is not None
    assert recorded.byte_offset == plan.last_complete_newline
    assert recorded.byte_size == plan.stat_size
    assert recorded.byte_size < path.stat().st_size
    # Avoid a vacuous True from the parser-version invalidation branch.
    monkeypatch.setattr(live_watcher, "_PARSER_FINGERPRINT", "test-parser")
    watcher = LiveWatcher(
        cast(Any, owner)._polylogue,
        (WatchSource(name=provider, root=path.parent),),
        cursor=processor._cursor,
    )
    assert watcher._needs_work(path) is True
    following = processor._append_plan(path)
    assert isinstance(following, _AppendPlan)
    assert ingest_append_with_owner(owner, [following]).succeeded == [following]
    assert processor._record_append_cursor(following) is True
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (3,)


def test_slow_append_finishes_once_without_poisoning_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: Any,
) -> None:
    from polylogue.core.write_hold import enter_write_hold, exit_write_hold
    from polylogue.sources.live import append_ingest

    path, plan, owner, processor = _seed_live_append_plan(tmp_path, native_id="append-budget")
    before = processor._cursor.get_record(path)
    assert before is not None
    original = append_ingest._write_append_raw_payload

    def delayed_capture(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        frozen_clock.advance(31)
        return result

    monkeypatch.setattr(append_ingest, "_write_append_raw_payload", delayed_capture)
    token = enter_write_hold("watcher.live_ingest.append", 30)
    try:
        assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
    finally:
        exit_write_hold(token)
    after = processor._cursor.get_record(path)
    assert after is not None
    assert after.byte_offset == before.byte_offset
    assert after.failure_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parse_error FROM raw_sessions WHERE source_path = ? AND source_index = -1",
            (str(path),),
        ).fetchall() == [(None,)]

    assert processor._record_append_cursor(plan) is True
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (2,)


def test_append_refuses_a_malformed_middle_record(tmp_path: Path) -> None:
    from hashlib import sha256

    path, initial, owner, processor = _seed_live_append_plan(tmp_path, native_id="malformed-middle")
    before = processor._cursor.get_record(path)
    assert before is not None
    original = path.read_bytes()[: initial.start_offset]
    malformed = initial.payload + b"{definitely not json}\n" + initial.payload.replace(b"message-1", b"message-2")
    path.write_bytes(original + malformed)
    plan = processor._append_plan(path)
    assert isinstance(plan, _AppendPlan)
    result = ingest_append_with_owner(owner, [plan])
    assert result.succeeded == []
    assert result.failed == [plan]
    after = processor._cursor.get_record(path)
    assert after is not None and after.byte_offset == before.byte_offset
    with sqlite3.connect(tmp_path / "source.db") as conn:
        [(_raw_id, retained_hash, error)] = conn.execute(
            "SELECT raw_id, hex(blob_hash), parse_error FROM raw_sessions WHERE source_path = ? AND source_index = -1",
            (str(path),),
        ).fetchall()
    assert retained_hash.lower() == sha256(malformed).hexdigest()
    assert error
    # The append's own bytes fail to decode: the same terminal evidence the
    # full route's census records for corrupt input.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id = ?", (_raw_id,)).fetchall() == [
            (RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value,)
        ]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)


def test_claude_live_append_keeps_latest_relocated_directory_first(tmp_path: Path) -> None:
    def record(number: int, relocated: str | None = None) -> bytes:
        item = {
            "type": "assistant",
            "uuid": f"message-{number}",
            "cwd": "/a/original",
            "sessionId": "moved-session",
            "timestamp": f"2026-06-02T00:00:0{number}Z",
            "message": {"role": "assistant", "content": f"reply {number}"},
        }
        payload = (json.dumps(item) + "\n").encode()
        if relocated is not None:
            # Relocation is its own producer record, not a message attribute.
            moved = {
                "type": "relocated",
                "sessionId": "moved-session",
                "relocatedCwd": relocated,
                "timestamp": item["timestamp"],
            }
            payload = (json.dumps(moved) + "\n").encode() + payload
        return payload

    path, first, owner, processor = _seed_claude_live_append_plan(
        tmp_path,
        native_id="moved-session",
        append=record(1),
    )
    assert ingest_append_with_owner(owner, [first]).succeeded == [first]
    assert processor._record_append_cursor(first)
    for number, moved in ((2, "/z/moved"), (3, "/y/latest")):
        with path.open("ab") as handle:
            handle.write(record(number, moved))
        plan = processor._append_plan(path)
        assert isinstance(plan, _AppendPlan)
        assert ingest_append_with_owner(owner, [plan]).succeeded == [plan]
        assert processor._record_append_cursor(plan)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            paths = [
                row[0]
                for row in conn.execute(
                    "SELECT path FROM session_working_dirs WHERE session_id = ? ORDER BY position, path",
                    ("claude-code-session:moved-session",),
                )
            ]
        assert paths[0] == moved
        assert "/a/original" in paths


def _settled_proof_fixture(tmp_path: Path) -> tuple[Path, bytes, os.stat_result, CursorStore, LiveBatchProcessor]:
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "settled-proof.jsonl"
    captured = (
        b'{"type":"session_meta","payload":{"id":"settled-proof"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-a","role":"user",'
        b'"content":[{"type":"input_text","text":"alpha"}]}}\n'
    )
    path.write_bytes(captured)
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    return path, captured, path.stat(), cursor, processor


def _observation(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@pytest.mark.parametrize("settled", [True, False], ids=["settled", "racy"])
def test_full_cursor_reuses_a_settled_unchanged_capture_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settled: bool
) -> None:
    """An unchanged source whose last change predates the capture is not re-hashed.

    A change time within the timestamp margin of the observation is the racy
    case and is proved by reading the bytes again. Anti-vacuity: drop the
    margin comparison and the racy case skips its re-read too; drop the
    settled shortcut and the settled case reads every byte.
    """
    path, captured, captured_stat, cursor, processor = _settled_proof_fixture(tmp_path)
    hashed: list[int] = []

    def counting_hash(source_path: Path, *, start_offset: int, end_offset: int) -> tuple[str, int]:
        hashed.append(end_offset - start_offset)
        return sha256_range_from_path(source_path, start_offset=start_offset, end_offset=end_offset)

    monkeypatch.setattr("polylogue.sources.live.batch.sha256_range_from_path", counting_hash)
    margin = live_batch._SETTLED_OBSERVATION_MARGIN_NS
    observed_at_ns = captured_stat.st_ctime_ns + (margin + 1 if settled else margin // 2)

    processor._record_full_cursor(
        path,
        raw_fingerprint=sha256(captured).hexdigest(),
        raw_byte_size=len(captured),
        source_name="codex",
        captured_content_hash=sha256(captured).hexdigest(),
        captured_file_observation=_observation(captured_stat),
        captured_observed_at_ns=observed_at_ns,
    )

    assert processor._last_cursor_write_stale is False
    record = cursor.get_record(path)
    assert record is not None
    assert record.byte_offset == len(captured)
    assert record.content_fingerprint == sha256(captured).hexdigest()
    # The racy case proves the prefix before and after deriving the cursor.
    assert hashed == ([] if settled else [len(captured), len(captured)])


def test_settled_observation_does_not_hide_a_same_size_rewrite(tmp_path: Path) -> None:
    """A rewrite after capture with its mtime restored still changes the change time.

    Anti-vacuity: compare only size and mtime and the rewritten bytes are
    accepted under the captured hash.
    """
    path, captured, captured_stat, cursor, processor = _settled_proof_fixture(tmp_path)
    rewritten = captured.replace(b"alpha", b"bravo")
    assert len(rewritten) == len(captured)
    path.write_bytes(rewritten)
    os.utime(path, ns=(captured_stat.st_atime_ns, captured_stat.st_mtime_ns))
    after = path.stat()
    if after.st_ctime_ns == captured_stat.st_ctime_ns:
        pytest.skip("filesystem did not advance ctime for the rewrite within this test's resolution")

    processor._record_full_cursor(
        path,
        raw_fingerprint=sha256(captured).hexdigest(),
        raw_byte_size=len(captured),
        source_name="codex",
        captured_content_hash=sha256(captured).hexdigest(),
        captured_file_observation=_observation(captured_stat),
        captured_observed_at_ns=captured_stat.st_ctime_ns + live_batch._SETTLED_OBSERVATION_MARGIN_NS + 1,
    )

    assert processor._last_cursor_write_stale is True
    record = cursor.get_record(path)
    assert record is None or record.content_fingerprint != sha256(captured).hexdigest()


_FRONTIER_PIECES = (
    b'{"a":1}',
    b'{"b":"x y"}',
    b"",
    b"  ",
    b"\t",
    b'{"bad"',
    b"[1,2]",
    b"\x0b",
    b'{"u":"\xc3\xa9"}',
    b"\xff",
)


@pytest.mark.parametrize("window", [1, 2, 3, 7, 1 << 20])
def test_file_frontier_matches_the_bytes_frontier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: int) -> None:
    """The tail-first file routes decide every frontier exactly as the bytes route does.

    Both the path frontier and the handle parse prefix are held to the bytes
    route. Payloads mix records, blank and whitespace-only lines, CRLF, malformed
    and unterminated tails; small read windows put every boundary across a
    window edge. Anti-vacuity: take the candidate from the last physical line
    instead of the last non-blank one, and blank-tail payloads disagree.
    """
    import random

    from polylogue.sources.live import batch_support

    monkeypatch.setattr(batch_support, "_JSONL_TAIL_READ_BYTES", window)
    rng = random.Random(window)
    path = tmp_path / "frontier.jsonl"
    for _ in range(600):
        payload = b"".join(
            rng.choice(_FRONTIER_PIECES) + rng.choice((b"\n", b"\n", b"\r\n", b"")) for _ in range(rng.randint(0, 6))
        )
        path.write_bytes(payload)
        expected = jsonl_complete_prefix(payload)
        frontier = jsonl_complete_prefix_path(path)
        assert (frontier.prefix_size, frontier.incomplete_tail, frontier.malformed_record) == (
            expected.prefix_size,
            expected.incomplete_tail,
            expected.malformed_record,
        ), payload
        with path.open("rb") as handle:
            assert jsonl_parse_prefix_size_of_handle(handle) == jsonl_parse_prefix_size(expected, len(payload)), payload
            assert handle.tell() == 0


def test_file_frontier_reads_only_the_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deciding the frontier of a large file reads its last record, not the file.

    Anti-vacuity: walk every line (the predecessor) and the bytes read equal
    the file size.
    """
    from polylogue.sources.live import batch_support

    monkeypatch.setattr(batch_support, "_JSONL_TAIL_READ_BYTES", 4096)
    path = tmp_path / "large.jsonl"
    record = b'{"type":"response_item","payload":{"text":"' + b"r" * 900 + b'"}}\n'
    path.write_bytes(record * 4000 + b'{"partial":')
    read = 0
    real_open = Path.open

    class CountingHandle:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> CountingHandle:
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.close()

        def seek(self, offset: int, whence: int = 0) -> int:
            return int(self._handle.seek(offset, whence))

        def tell(self) -> int:
            return int(self._handle.tell())

        def read(self, size: int = -1) -> bytes:
            nonlocal read
            data: bytes = self._handle.read(size)
            read += len(data)
            return data

    class CountingPath(type(path)):  # type: ignore[misc]
        def open(self, *args: Any, **kwargs: Any) -> Any:
            return CountingHandle(real_open(self, *args, **kwargs))

    frontier = jsonl_complete_prefix_path(CountingPath(path))

    assert frontier.prefix_size == len(record) * 4000
    assert frontier.incomplete_tail and not frontier.malformed_record
    assert read < 4 * 4096


@pytest.mark.parametrize(
    "tail",
    [
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"1e9999",
        b"null",
        b"true",
        b"42",
        b'"scalar"',
        b'"\\ud800"',
        b'"\xed\xa0\x80"',
        b'{"wide":' + b"9" * 5000 + b"}",
        b'{"text":"' + b"x" * 200000 + b'"}',
        b'{"unfinished":',
        b"{} {}",
        b'"\xff"',
    ],
)
@pytest.mark.parametrize("ending", [b"", b"\n", b"\n \r\n"])
def test_jsonl_frontier_grammar_is_identical_for_bytes_path_and_handle(
    tmp_path: Path,
    tail: bytes,
    ending: bytes,
) -> None:
    from polylogue.sources.live.batch_support import jsonl_frontier_of_handle

    payload = b'{"first":1}\n' + tail + ending
    path = tmp_path / "grammar.jsonl"
    path.write_bytes(payload)
    boundary = jsonl_complete_prefix(payload)
    expected = (boundary.prefix_size, boundary.incomplete_tail, boundary.malformed_record)
    frontier = jsonl_complete_prefix_path(path)
    assert (frontier.prefix_size, frontier.incomplete_tail, frontier.malformed_record) == expected
    with path.open("rb") as handle:
        frontier = jsonl_frontier_of_handle(handle, len(payload))
        assert (frontier.prefix_size, frontier.incomplete_tail, frontier.malformed_record) == expected
        assert not handle.closed


def test_jsonl_prefix_view_restores_position_and_leaves_input_open() -> None:
    import io

    from polylogue.sources.live.batch_support import jsonl_parse_input_of_handle

    prefix = b'{"first":1}\n'
    handle = io.BytesIO(prefix + b'{"unfinished":')
    handle.seek(3)
    with jsonl_parse_input_of_handle(handle) as view:
        assert view.seek(0) == 0
        assert view.read() == prefix
        assert view.seek(100000) == len(prefix)
        assert view.read(1) == b""
        assert view.seek(-2, io.SEEK_END) == len(prefix) - 2
        assert view.read(10) == prefix[-2:]
        view.close()
        assert not handle.closed
    assert handle.tell() == 3
    assert not handle.closed


@pytest.mark.parametrize("failure_type", [ValueError, RuntimeError, KeyboardInterrupt])
def test_jsonl_prefix_cancellation_restores_input_and_propagates_callback_failure(
    failure_type: type[BaseException],
) -> None:
    import io

    from polylogue.sources.live.batch_support import jsonl_parse_input_of_handle

    handle = io.BytesIO(b'{"text":"' + b"x" * 200000 + b'"}')
    handle.seek(4)
    failure = failure_type("cancelled frontier")
    calls = 0

    def check_stop() -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise failure

    with pytest.raises(failure_type) as caught:
        with jsonl_parse_input_of_handle(handle, check_stop=check_stop):
            pytest.fail("cancelled input was exposed")
    assert caught.value is failure
    assert handle.tell() == 4
    assert not handle.closed


@pytest.mark.parametrize("chunk_bytes", [1, 3, 7, 1 << 20])
@pytest.mark.parametrize(
    "prefix",
    [
        b"",
        b'{"a":1}\n',
        b'{"a":1}\n{"b":2}\n',
        b'{"a":1}\n\n   \n{"b":2}\n',
        b'\n\n{"a":1}\n \t\r\n{"b":2}\n',
        b'{"long":"' + b"x" * 40 + b'"}\n{"b":2}\n',
    ],
)
def test_the_streamed_prefix_record_count_matches_the_in_memory_count(
    monkeypatch: pytest.MonkeyPatch, prefix: bytes, chunk_bytes: int
) -> None:
    """A partial admission's record count is the complete records of its prefix, blank lines excluded.

    Anti-vacuity: counting newlines counts the blank lines; carrying a line's
    content across a chunk boundary wrongly counts a record twice or not at all.
    """
    import io

    from polylogue.sources.live import batch_support

    monkeypatch.setattr(batch_support, "_JSONL_TAIL_READ_BYTES", chunk_bytes)
    tail = b'{"cut":'
    counted = batch_support.jsonl_prefix_record_count(io.BytesIO(prefix + tail), len(prefix))
    assert counted == batch_support._jsonl_record_count(prefix)


def test_partial_prefix_count_observes_owner_cancellation() -> None:
    import io

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.sources.live.batch_support import jsonl_prefix_record_count

    # The count runs under the daemon's operation owner, so a stop surfaces
    # as that owner's typed cancellation.
    with pytest.raises(DaemonOperationCancelled):
        jsonl_prefix_record_count(io.BytesIO(b'{"a":1}\n'), 8, stop=lambda: True)


def test_live_zip_crc_failure_preserves_pending_input_without_prefix_publication(tmp_path: Path) -> None:
    """A yielded member prefix cannot prove a complete corrupted ZIP group."""
    bootstrap_archive_root(tmp_path)
    root = tmp_path / "inbox"
    root.mkdir()
    bundle = root / "bad-crc.zip"
    with zipfile.ZipFile(bundle, "w") as container:
        for index in range(2):
            container.writestr(
                f"sessions/session-{index}.jsonl",
                json.dumps({"type": "session_meta", "payload": {"id": f"session-{index}"}}) + "\n",
            )
    wire = bytearray(bundle.read_bytes())
    first_header = wire.index(b"PK\x01\x02")
    second_header = wire.index(b"PK\x01\x02", first_header + 4)
    wire[second_header + 16] ^= 1  # Actual member read now fails its central CRC.
    bundle.write_bytes(wire)
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    result = _full_paths_sync(processor, [bundle], source_name="codex")

    assert result.failed == [bundle]
    assert result.excluded == {}
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT enumerated_at_ms FROM source_items").fetchall() == [(None,)]
        assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM source_items WHERE blob_hash IS NOT NULL").fetchone()[0] == 1
