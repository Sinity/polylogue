"""Production discovery decisions and generation-bound revision conservation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.live.production_baseline import (
    ProductionBaselineError,
    ProductionBaselineObservationCancelledError,
    ProductionBaselineReadUnavailableError,
    ProductionSourceBaseline,
    SourceDecision,
    _revision,
    _seal,
    capture_production_source_baseline,
    merge_pending_production_baseline,
    unretained_source_material,
)
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.source_layout import declared_source_layout, export_drop_layout
from polylogue.sources.sqlite_snapshot import sqlite_member_revision
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture


def _source_db(path: Path, rows: tuple[tuple[str, str] | tuple[str, int, str], ...]) -> Path:
    initialize_runtime_source_fixture(path)
    for row in rows:
        if len(row) == 3:
            _retain_row(path, *row)
        else:
            _retain_row(path, row[0], 0, row[1])
    return path


def _retain_row(source_db: Path, source_path: str, source_index: int, revision: str) -> None:
    """A ledger row naming a revision whose bytes this fixture does not keep."""
    conn = sqlite3.connect(source_db)
    try:
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms) "
            "VALUES (?, 'codex-session', ?, ?, ?, 0, 0)",
            (f"{source_path}#{source_index}#{revision}", source_path, source_index, bytes.fromhex(revision)),
        )
        conn.commit()
    finally:
        conn.close()


def _zip_row(row: SourceDecision) -> tuple[str, int, str]:
    assert row.source_index is not None
    assert row.revision is not None
    return row.path, row.source_index, row.revision


def test_large_source_revision_stops_between_chunks_on_cancel(tmp_path: Path) -> None:
    source = tmp_path / "large.json"
    source.write_bytes(b"x" * (3 * 1024 * 1024))
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(ProductionBaselineObservationCancelledError):
        _revision(source, cancelled=cancelled)
    assert checks == 3


def test_only_typed_io_revision_fault_is_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.sources.live import production_baseline

    root = tmp_path / "account"
    root.mkdir()
    (root / "one.json").write_bytes(b"{}")
    source = (WatchSource("account", root, layout=export_drop_layout((".json",)), required=True),)

    def io_fault(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise OSError(errno.EIO, "source read failed")

    with monkeypatch.context() as patcher:
        patcher.setattr(production_baseline, "_revision", io_fault)
        unavailable = capture_production_source_baseline(source, operation_id="io")
    with pytest.raises(ProductionBaselineReadUnavailableError):
        unavailable.verify(tmp_path / "source.db")

    def deterministic_fault(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise ValueError("invalid logical source")

    with monkeypatch.context() as patcher:
        patcher.setattr(production_baseline, "_revision", deterministic_fault)
        invalid = capture_production_source_baseline(source, operation_id="invalid")
    with pytest.raises(ProductionBaselineError) as failure:
        invalid.verify(tmp_path / "source.db")
    assert not isinstance(failure.value, ProductionBaselineReadUnavailableError)


@pytest.mark.parametrize("fault_errno", [errno.EMFILE, errno.ENFILE])
def test_descriptor_exhaustion_baseline_fault_retries_and_preserves_prior_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_errno: int
) -> None:
    from polylogue.sources.live import production_baseline

    root = tmp_path / "account"
    root.mkdir()
    first = root / "one.json"
    second = root / "two.json"
    first_bytes = b'{"session":1}'
    second_bytes = b'{"session":2}'
    first.write_bytes(first_bytes)
    second.write_bytes(second_bytes)
    source = (WatchSource("account", root, layout=export_drop_layout((".json",)), required=True),)
    previous = capture_production_source_baseline(source, operation_id="descriptor-retry")
    source_db = _source_db(
        tmp_path / "source.db",
        (
            (str(first), hashlib.sha256(first_bytes).hexdigest()),
            (str(second), hashlib.sha256(second_bytes).hexdigest()),
        ),
    )
    previous.verify(source_db)

    original_revision = production_baseline._revision

    def exhaust_descriptors(path: Path, **kwargs: object) -> tuple[str, int]:
        if path == second:
            raise OSError(fault_errno, "descriptor table exhausted")
        return original_revision(path, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(production_baseline, "_revision", exhaust_descriptors)
        unavailable = capture_production_source_baseline(source, operation_id="descriptor-retry")
    [fault] = [row for row in unavailable.decisions if row.path == str(second)]
    assert fault.disposition == "fault"
    assert fault.reason.startswith("revision_io_unavailable:")
    with pytest.raises(ProductionBaselineReadUnavailableError):
        unavailable.verify(source_db)

    recovered = capture_production_source_baseline(source, operation_id="descriptor-retry")
    merged = merge_pending_production_baseline(recovered, unavailable)
    assert {row.path for row in merged.accepted} == {str(first), str(second)}
    assert not any(row.disposition == "fault" for row in merged.decisions)
    merged.verify(source_db)


@pytest.mark.parametrize("fault", [OSError(errno.EIO, "member read failed"), ValueError("invalid member")])
def test_zip_member_fault_preserves_typed_retry_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: Exception
) -> None:
    from polylogue.sources.live import production_baseline

    root = tmp_path / "account"
    root.mkdir()
    bundle = root / "export.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", b"[]")

    def unreadable_member(*_args: object, **_kwargs: object) -> None:
        raise fault

    with monkeypatch.context() as patcher:
        patcher.setattr(production_baseline, "replay_zip_entry_acquisition_revisions", unreadable_member)
        baseline = capture_production_source_baseline(
            (WatchSource("account", root, layout=export_drop_layout((".zip",))),), operation_id="zip-fault"
        )
    member_faults = [row for row in baseline.decisions if row.disposition == "fault"]
    assert len(member_faults) == 1
    assert member_faults[0].path == f"{bundle}:conversations.json"
    with pytest.raises(ProductionBaselineError) as failure:
        baseline.verify(tmp_path / "source.db")
    assert isinstance(failure.value, ProductionBaselineReadUnavailableError) == isinstance(fault, OSError)


def test_directory_walk_io_fault_retries_and_resolves_after_walk_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live import discovery, production_baseline

    root = tmp_path / "account"
    nested = root / "nested"
    nested.mkdir(parents=True)
    member = nested / "one.json"
    member.write_bytes(b"{}")
    source = (WatchSource("account", root, layout=export_drop_layout((".json",))),)
    original_scandir = os.scandir

    def fail_nested(path: os.PathLike[str] | str) -> Any:
        if Path(path) == nested:
            raise OSError(errno.EIO, "nested directory temporarily unreadable")
        return original_scandir(path)

    def walk_with_failure(*args: Any, **kwargs: Any) -> Any:
        return discovery._source_path_steps(*args, scandir=fail_nested, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(production_baseline, "_source_path_steps", walk_with_failure)
        unavailable = capture_production_source_baseline(source, operation_id="walk-fault")
    with pytest.raises(ProductionBaselineReadUnavailableError):
        unavailable.verify(tmp_path / "source.db")

    recovered = capture_production_source_baseline(source, operation_id="walk-fault")
    merged = merge_pending_production_baseline(recovered, unavailable)
    assert not any(row.disposition == "fault" for row in merged.decisions)
    merged.verify(_source_db(tmp_path / "source.db", ((str(member), hashlib.sha256(b"{}").hexdigest()),)))


def test_recovered_zip_open_clears_its_archive_level_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.sources.live import production_baseline

    root = tmp_path / "account"
    root.mkdir()
    bundle = root / "export.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", b"[]")
    source = (WatchSource("account", root, layout=export_drop_layout((".zip",))),)

    def unreadable_archive(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EIO, "archive open temporarily failed")

    with monkeypatch.context() as patcher:
        patcher.setattr(production_baseline, "_archive_members", unreadable_archive)
        unavailable = capture_production_source_baseline(source, operation_id="zip-open")
    assert not any(row.reason == "expanded_to_members" for row in unavailable.decisions)
    with pytest.raises(ProductionBaselineReadUnavailableError):
        unavailable.verify(tmp_path / "source.db")

    recovered = capture_production_source_baseline(source, operation_id="zip-open")
    merged = merge_pending_production_baseline(recovered, unavailable)
    assert not any(row.disposition == "fault" for row in merged.decisions)
    merged.verify(
        _source_db(tmp_path / "source.db", ((f"{bundle}:conversations.json", hashlib.sha256(b"[]").hexdigest()),))
    )


def test_baseline_uses_typed_acceptance_before_cursor_and_requires_retained_revision(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    accepted = root / "session.json"
    accepted.write_bytes(b'{"session":1}')
    (root / "note.txt").write_text("excluded")
    source = WatchSource("account", root, layout=export_drop_layout((".json",)), required=True)
    baseline = capture_production_source_baseline((source,), operation_id="build-1")
    assert [(row.path, row.disposition) for row in baseline.decisions] == [
        (str(root), "excluded"),
        (str(root / "note.txt"), "excluded"),
        (str(accepted), "accepted"),
    ]
    old_revision = baseline.accepted[0].revision
    assert old_revision == hashlib.sha256(b'{"session":1}').hexdigest()
    assert baseline.prospective_material_bytes == len(b'{"session":1}')
    assert baseline.prospective_retained_allocation_bytes(4096) == 4096
    assert baseline.prospective_source_db_allocation_bytes(4096) == 4096
    accepted.write_bytes(b'{"session":2}')
    new_revision = hashlib.sha256(accepted.read_bytes()).hexdigest()
    source_db = _source_db(tmp_path / "source.db", ((str(accepted), new_revision),))
    with pytest.raises(ProductionBaselineError, match="unretained revision"):
        baseline.verify(source_db)
    assert old_revision is not None
    _retain_row(source_db, str(accepted), 0, old_revision)
    baseline.verify(source_db)


def test_baseline_records_intake_exclusions_instead_of_requiring_retention(tmp_path: Path) -> None:
    """Each pre-acquisition exclusion intake applies is the baseline's disposition too.

    Anti-vacuity: without the shared ``classify_pre_acquisition`` decision
    every one of these files is ``accepted`` and ``verify`` raises for the
    unretained revisions that intake never writes.
    """
    codex = tmp_path / "codex"
    codex.mkdir()
    rollout = codex / "rollout.jsonl"
    rollout.write_bytes(
        b'{"type":"session_meta","payload":{"id":"s","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m","role":"user",'
        b'"content":[{"type":"input_text","text":"hi"}]}}\n'
    )
    meta_only = codex / "meta-only.jsonl"
    meta_only.write_bytes(b'{"type":"session_meta","payload":{"id":"x","timestamp":"2026-06-02T00:00:00Z"}}\n')
    unverified_state = codex / "state_5.sqlite"
    with sqlite3.connect(unverified_state) as conn:
        conn.execute("CREATE TABLE threads(id TEXT)")
    gemini = tmp_path / "gemini"
    logs = gemini / "tmp" / "project" / "logs.json"
    logs.parent.mkdir(parents=True)
    logs.write_text('[{"sessionId":"a","messageId":0,"type":"user","message":"hi","timestamp":"2026"}]')
    baseline = capture_production_source_baseline(
        (
            WatchSource("codex", codex, layout=export_drop_layout((".jsonl", ".sqlite"))),
            WatchSource("gemini-cli", gemini, layout=export_drop_layout((".json",))),
        ),
        operation_id="intake-exclusions",
    )
    decisions = {row.path: (row.disposition, row.reason) for row in baseline.decisions}
    assert decisions[str(meta_only)] == ("excluded", "intake_excluded:declared artifact rule: not parsed as a session")
    assert decisions[str(unverified_state)] == (
        "excluded",
        "intake_excluded:unsupported source class",
    )
    # Gemini CLI's ``logs.json`` prompt log is declared raw-only evidence:
    # intake retains its bytes, so the baseline demands them too.
    assert decisions[str(logs)][0] == "accepted"
    assert {row.path for row in baseline.accepted} == {str(rollout), str(logs)}
    baseline.verify(
        _source_db(
            tmp_path / "source.db",
            tuple((str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in (rollout, logs)),
        )
    )


def test_external_link_is_alias_only_with_independent_source(tmp_path: Path) -> None:
    account = tmp_path / "account"
    account.mkdir()
    (account / "one.json").write_text("{}")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "account").symlink_to(account, target_is_directory=True)
    typed = WatchSource("account", account, layout=export_drop_layout((".json",)), required=True)
    alias = WatchSource("inbox", inbox, layout=export_drop_layout((".json",)))
    baseline = capture_production_source_baseline((alias, typed), operation_id="build-2")
    assert any(row.path == str(inbox / "account") and row.disposition == "alias" for row in baseline.decisions)
    assert len(baseline.accepted) == 1
    unresolved = capture_production_source_baseline((alias,), operation_id="build-3")
    assert any(row.path == str(inbox / "account") and row.disposition == "fault" for row in unresolved.decisions)


def test_broken_accepted_looking_link_is_fault(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    (root / "gone.json").symlink_to(root / "missing.json")
    source = WatchSource("account", root, layout=export_drop_layout((".json",)), required=True)
    baseline = capture_production_source_baseline((source,), operation_id="build-4")
    assert any(row.disposition == "fault" and row.path.endswith("gone.json") for row in baseline.decisions)


def test_directory_cycle_is_alias_when_canonical_target_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    (root / "one.json").write_text("{}")
    (root / "cycle").symlink_to(root, target_is_directory=True)
    baseline = capture_production_source_baseline(
        (WatchSource("account", root, layout=export_drop_layout((".json",))),), operation_id="cycle"
    )
    assert any(row.path == str(root / "cycle") and row.disposition == "alias" for row in baseline.decisions)


def test_empty_effective_source_tuple_refuses_a_baseline() -> None:
    """Removing the typed source tuple must not recreate the old zero-root bypass."""
    with pytest.raises(ProductionBaselineError, match="no effective watch sources"):
        capture_production_source_baseline((), operation_id="build-5")


def test_history_rule_and_codex_sqlite_use_their_typed_revisions(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    claude.mkdir()
    history = claude / "history.jsonl"
    history.write_text('{"display":"hello"}\n')
    codex = tmp_path / ".codex"
    codex.mkdir()
    state = codex / "state_5.sqlite"
    conn = sqlite3.connect(state)
    try:
        # The thread-state shape intake acquires; a bare ``threads`` table is
        # structurally unverified and intake excludes it.
        conn.execute("CREATE TABLE threads(id TEXT)")
        conn.execute("CREATE TABLE thread_spawn_edges(parent TEXT, child TEXT)")
        conn.execute("INSERT INTO threads VALUES ('one')")
        conn.commit()
    finally:
        conn.close()
    baseline = capture_production_source_baseline(
        (
            WatchSource("claude-code-history", claude, layout=declared_source_layout("claude-code-history")),
            WatchSource("codex-state", codex, layout=declared_source_layout("codex-state")),
        ),
        operation_id="build-6",
    )
    assert {row.path for row in baseline.accepted} == {str(history), str(state)}
    assert next(row.revision for row in baseline.accepted if row.path == str(state)) == sqlite_member_revision(state)
    from polylogue.sources.sqlite_export import logical_export_bytes
    from polylogue.sources.sqlite_snapshot import member_export_scope

    assert next(row.material_bytes for row in baseline.accepted if row.path == str(state)) == len(
        logical_export_bytes(state, scope=member_export_scope(state))
    )


def test_temporarily_unopenable_sqlite_source_remains_a_retryable_baseline_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live import production_baseline

    codex = tmp_path / ".codex"
    codex.mkdir()
    state = codex / "state_5.sqlite"
    with sqlite3.connect(state) as conn:
        conn.execute("CREATE TABLE threads(id TEXT)")
        conn.execute("CREATE TABLE thread_spawn_edges(parent TEXT, child TEXT)")
    with pytest.raises(sqlite3.OperationalError) as unavailable:
        sqlite3.connect(f"file:{tmp_path / 'temporarily-unavailable.db'}?mode=ro", uri=True)
    assert unavailable.value.sqlite_errorcode & 0xFF == sqlite3.SQLITE_CANTOPEN

    def unavailable_revision(_path: Path, **_kwargs: object) -> tuple[str, int]:
        raise unavailable.value

    monkeypatch.setattr(production_baseline, "sqlite_member_revision_and_size", unavailable_revision)
    baseline = capture_production_source_baseline(
        (WatchSource("codex-state", codex, layout=declared_source_layout("codex-state")),),
        operation_id="temporary-sqlite-open",
    )
    assert any(
        row.path == str(state) and row.reason.startswith("revision_io_unavailable:") for row in baseline.decisions
    )
    with pytest.raises(ProductionBaselineReadUnavailableError):
        baseline.verify(tmp_path / "source.db")


def test_zip_members_keep_live_coordinates_and_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    bundle = root / "export.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", b"[]")
        archive.writestr("README.txt", b"not a session")
    baseline = capture_production_source_baseline(
        (WatchSource("account", root, layout=export_drop_layout((".zip",))),), operation_id="build-7"
    )
    assert any(
        row.path == f"{bundle}:conversations.json" and row.disposition == "accepted" for row in baseline.decisions
    )
    assert any(row.path == f"{bundle}:README.txt" and row.disposition == "excluded" for row in baseline.decisions)
    assert baseline.prospective_material_bytes == len(b"[]")
    source_db = _source_db(
        tmp_path / "source.db", ((f"{bundle}:conversations.json", hashlib.sha256(b"[]").hexdigest()),)
    )
    baseline.verify(source_db)
    conn = sqlite3.connect(source_db)
    try:
        conn.execute(
            "UPDATE raw_sessions SET blob_hash = ?", (bytes.fromhex(hashlib.sha256(b"different").hexdigest()),)
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ProductionBaselineError, match="unretained revision"):
        baseline.verify(source_db)
    conn = sqlite3.connect(source_db)
    try:
        conn.execute("UPDATE raw_sessions SET blob_hash = ?", (bytes.fromhex(hashlib.sha256(b"[]").hexdigest()),))
        conn.commit()
    finally:
        conn.close()
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("later.json", b"{}")
    baseline.verify(source_db)


def test_zip_split_records_require_each_production_revision_and_coordinate(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    bundle = root / "export.zip"
    sessions = [
        {"id": name, "mapping": {"node": {"message": {"author": {"role": "user"}}}}} for name in ("first", "second")
    ]
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", json.dumps(sessions, separators=(",", ":")))
    baseline = capture_production_source_baseline(
        (WatchSource("account", root, layout=export_drop_layout((".zip",))),), operation_id="split"
    )
    members = [row for row in baseline.accepted if row.path == f"{bundle}:conversations.json"]
    assert len(members) == 2
    assert len({row.source_index for row in members}) == 2
    assert all(
        row.revision != hashlib.sha256(json.dumps(sessions, separators=(",", ":")).encode()).hexdigest()
        for row in members
    )
    source_db = _source_db(tmp_path / "source.db", tuple(_zip_row(row) for row in members[:1]))
    with pytest.raises(ProductionBaselineError, match="unretained revision"):
        baseline.verify(source_db)
    _retain_row(source_db, *_zip_row(members[1]))
    baseline.verify(source_db)


def test_zip_declared_binary_and_markdown_artifacts_follow_live_validator(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    bundle = root / "export.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("tool-results/one.bin", b"\xff\x00opaque")
        archive.writestr("brain/one.md", b"# note\n")
    baseline = capture_production_source_baseline(
        (WatchSource("account", root, layout=export_drop_layout((".zip",))),), operation_id="artifacts"
    )
    members = [row for row in baseline.accepted if row.path.startswith(f"{bundle}:")]
    assert {row.path for row in members} == {f"{bundle}:tool-results/one.bin", f"{bundle}:brain/one.md"}
    source_db = _source_db(tmp_path / "source.db", tuple(_zip_row(row) for row in members))
    baseline.verify(source_db)


def test_old_required_root_fault_only_clears_when_current_watch_observes_it(tmp_path: Path) -> None:
    root = tmp_path / "account"
    previous = capture_production_source_baseline((WatchSource("account", root, required=True),), operation_id="old")
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    unrelated = capture_production_source_baseline((WatchSource("other", other),), operation_id="new")
    assert any(row.reason == "absent_root" for row in merge_pending_production_baseline(unrelated, previous).decisions)
    recovered = capture_production_source_baseline((WatchSource("account", root, required=True),), operation_id="new")
    assert not any(
        row.reason == "absent_root" for row in merge_pending_production_baseline(recovered, previous).decisions
    )


def test_hash_phase_starts_before_each_accepted_revision_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting the phase only after the read leaves a long hash labelled as the walk."""
    from polylogue.sources.live import production_baseline as module

    root = tmp_path / "source"
    root.mkdir()
    # A real session record: intake excludes a Codex JSONL with none, and an
    # excluded file is never hashed.
    session = (
        b'{"type":"session_meta","payload":{"id":"s","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m","role":"user",'
        b'"content":[{"type":"input_text","text":"hi"}]}}\n'
    )
    (root / "one.jsonl").write_bytes(session)
    calls: list[tuple[str, dict[str, int]]] = []
    real_revision = module._revision

    def observed_revision(path: Path, **kwargs: Any) -> tuple[str, int]:
        assert calls[-1] == ("baseline_hash", {}), "the read began before the hash phase was entered"
        return real_revision(path, **kwargs)

    monkeypatch.setattr(module, "_revision", observed_revision)
    capture_production_source_baseline(
        (WatchSource("codex", root, layout=export_drop_layout((".jsonl",))),),
        operation_id="phase",
        progress=lambda phase, **counts: calls.append((phase, counts)),
    )
    assert calls[-1] == ("baseline_hash", {"revisions": 1, "hashed_bytes": len(session)})


def test_intake_exclusion_retires_an_earlier_accepted_observation(tmp_path: Path) -> None:
    """A resumed build does not carry forward a demand intake will never meet.

    Anti-vacuity: removing the ``intake_excluded`` skip in
    ``merge_pending_production_baseline`` keeps the earlier accepted row, and
    ``verify`` raises ``unretained revision(s)``.
    """
    root = tmp_path / "codex"
    root.mkdir()
    sidecar = root / "rollout-2026-06-02T00-00-00-meta.jsonl"
    sidecar.write_bytes(
        b'{"timestamp":"2026-06-02T00:00:00Z","type":"session_meta","payload":{"id":"meta",'
        b'"timestamp":"2026-06-02T00:00:00Z","cwd":"/tmp","originator":"codex_cli_rs"}}\n'
    )
    current = capture_production_source_baseline(
        (WatchSource("codex", root, layout=export_drop_layout((".jsonl",))),), operation_id="op"
    )
    [row] = [row for row in current.decisions if row.path == str(sidecar)]
    assert row.disposition == "excluded" and row.reason.startswith("intake_excluded:")

    revision, size = _revision(sidecar)
    earlier = _seal(
        "op",
        current.source_signature,
        (SourceDecision("codex", str(sidecar), "accepted", "file", revision, material_bytes=size),),
    )
    merged = merge_pending_production_baseline(current, earlier)
    assert merged.accepted == ()
    merged.verify(_source_db(tmp_path / "source.db", ()))


def test_a_foreign_origin_file_intake_refuses_is_never_demanded(tmp_path: Path) -> None:
    """Intake refuses a Codex rollout under Claude Code's root; the baseline agrees.

    The refusal comes from the acquisition boundary (the Codex record follows
    a Claude Code record, so no pre-copy sniff sees it) and is recorded with
    intake's own ``intake_excluded:`` reason, so a resumed build also retires
    an earlier observation that accepted the same bytes.

    Anti-vacuity: hashing outside the boundary records the file ``accepted``
    and ``verify`` raises for an unretained revision intake never writes;
    recording the refusal without the ``intake_excluded:`` prefix keeps the
    earlier accepted row after the merge.
    """
    root = tmp_path / "projects"
    project = root / "proj"
    project.mkdir(parents=True)
    path = project / "bad69218-73bd-490a-869a-2b3a30bf421b.jsonl"
    path.write_bytes(
        b'{"type":"user","uuid":"u1","sessionId":"bad69218-73bd-490a-869a-2b3a30bf421b",'
        b'"timestamp":"2025-06-13T17:40:00.000Z","cwd":"/p","message":{"role":"user","content":"hi"}}\n'
        b'{"type":"session_meta","payload":{"id":"s","timestamp":"2026-06-02T00:00:00Z"}}\n'
    )
    current = capture_production_source_baseline(
        (WatchSource("claude-code", root, layout=export_drop_layout((".jsonl",))),), operation_id="op"
    )
    [row] = [row for row in current.decisions if row.path == str(path)]
    assert row.disposition == "excluded"
    assert row.reason.startswith("intake_excluded:foreign_origin_content")

    revision, size = _revision(path)
    earlier = _seal(
        "op",
        current.source_signature,
        (SourceDecision("claude-code", str(path), "accepted", "file", revision, material_bytes=size),),
    )
    merged = merge_pending_production_baseline(current, earlier)
    assert merged.accepted == ()
    merged.verify(_source_db(tmp_path / "source.db", ()))


def test_a_refused_zip_member_retires_its_earlier_acceptance(tmp_path: Path) -> None:
    """A resumed build drops the demand for a member intake now refuses.

    An earlier baseline, from before location binding, accepted a Codex
    rollout inside an archive under Claude Code's root. The member's bytes
    are unchanged and intake now refuses them.

    Anti-vacuity: recording the member refusal without the
    ``intake_excluded:`` prefix, or comparing the member coordinate as a
    file path, keeps the earlier accepted row and ``verify`` raises.
    """
    import zipfile

    from polylogue.core.raw_coordinates import zip_member_source_index

    root = tmp_path / "projects"
    root.mkdir()
    member = (
        b'{"type":"session_meta","payload":{"id":"s","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m","role":"user",'
        b'"content":[{"type":"input_text","text":"hi"}]}}\n'
    )
    archive = root / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("rollout.jsonl", member)
    current = capture_production_source_baseline(
        (WatchSource("claude-code", root, layout=export_drop_layout((".zip",))),), operation_id="op"
    )
    coordinate = f"{archive}:rollout.jsonl"
    [row] = [row for row in current.decisions if row.path == coordinate]
    assert row.disposition == "excluded"
    assert row.reason.startswith("intake_excluded:foreign_origin_content")

    earlier = _seal(
        "op",
        current.source_signature,
        (
            SourceDecision(
                "claude-code",
                coordinate,
                "accepted",
                "archive_member",
                hashlib.sha256(member).hexdigest(),
                zip_member_source_index(entry_ordinal=0, split_index=0),
                len(member),
            ),
        ),
    )
    merged = merge_pending_production_baseline(current, earlier)
    assert merged.accepted == ()
    merged.verify(_source_db(tmp_path / "source.db", ()))


def test_a_rewritten_path_keeps_its_earlier_accepted_revision(tmp_path: Path) -> None:
    """A session observed earlier stays demanded after its path is rewritten to a sidecar.

    Anti-vacuity: retiring the earlier row by coordinate alone (dropping the
    ``_unchanged_revision`` check) empties ``merged.accepted`` and ``verify``
    passes without the session ever being retained.
    """
    root = tmp_path / "codex"
    root.mkdir()
    path = root / "rollout-2026-06-02T00-00-00-rewritten.jsonl"
    session = (
        b'{"type":"session_meta","payload":{"id":"s","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m","role":"user",'
        b'"content":[{"type":"input_text","text":"hi"}]}}\n'
    )
    earlier = _seal(
        "op",
        "signature",
        (
            SourceDecision(
                "codex",
                str(path),
                "accepted",
                "file",
                hashlib.sha256(session).hexdigest(),
                material_bytes=len(session),
            ),
        ),
    )
    path.write_bytes(b'{"type":"session_meta","payload":{"id":"s","timestamp":"2026-06-02T00:00:00Z"}}\n')
    current = capture_production_source_baseline(
        (WatchSource("codex", root, layout=export_drop_layout((".jsonl",))),), operation_id="op"
    )
    assert {row.path: row.disposition for row in current.decisions}[str(path)] == "excluded"

    merged = merge_pending_production_baseline(current, earlier)
    assert [row.revision for row in merged.accepted] == [hashlib.sha256(session).hexdigest()]
    with pytest.raises(ProductionBaselineError, match="unretained revision"):
        merged.verify(_source_db(tmp_path / "source.db", ()))


def test_an_unreadable_state_database_is_a_retryable_fault_not_an_exclusion(tmp_path: Path) -> None:
    """A read fault on a declared state database keeps the build retrying.

    The structural recognizer cannot open the file and reads it as "not
    Codex state", which would otherwise record a terminal intake exclusion.

    Anti-vacuity: dropping the probe in ``classify_pre_acquisition`` records
    ``intake_excluded:...`` and the database silently leaves the demand.
    """
    root = tmp_path / "codex"
    root.mkdir()
    state = root / "state_5.sqlite"
    with sqlite3.connect(state) as conn:
        conn.execute("CREATE TABLE threads(id TEXT)")
    state.chmod(0)
    try:
        baseline = capture_production_source_baseline(
            (WatchSource("codex", root, layout=export_drop_layout((".sqlite",))),), operation_id="unreadable"
        )
    finally:
        state.chmod(0o600)
    [row] = [row for row in baseline.decisions if row.path == str(state)]
    assert row.disposition == "fault"
    assert row.reason.startswith("revision_io_unavailable:")


def test_non_database_bytes_under_a_state_name_stay_an_intake_exclusion(tmp_path: Path) -> None:
    """Bytes that are not SQLite are excluded by intake for good, so they are not a fault.

    Anti-vacuity: routing the probe's non-retryable ``SQLITE_NOTADB`` to the
    ordinary fault branch records ``revision_unreadable`` and blocks promotion
    on a file intake will never retain.
    """
    root = tmp_path / "codex"
    root.mkdir()
    state = root / "state_5.sqlite"
    state.write_bytes(b"not a database, just text\n" * 64)
    baseline = capture_production_source_baseline(
        (WatchSource("codex", root, layout=export_drop_layout((".sqlite",))),), operation_id="not-a-database"
    )
    [row] = [row for row in baseline.decisions if row.path == str(state)]
    assert row.disposition == "excluded"
    assert row.reason.startswith("intake_excluded:")


def test_the_admission_scan_checkpoints_inside_one_long_line(tmp_path: Path) -> None:
    """Cancellation reaches the sidecar scan even inside one unterminated record.

    Anti-vacuity: checkpointing only between lines (or calling the
    uncheckpointed ``jsonl_session_artifact(path)``) reads the whole 8 MiB
    line first, the checkpoint runs at most once, and nothing raises.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.live.batch_support import classify_pre_acquisition

    root = tmp_path / "codex"
    root.mkdir()
    sidecar = root / "rollout-2026-06-02T00-00-00-long.jsonl"
    with sidecar.open("wb") as stream:
        stream.write(b'{"type":"session_meta","payload":{"id":"x","timestamp":"2026-06-02T00:00:00Z"}}\n')
        stream.write(b'{"type":"session_meta","payload":"' + b"x" * (8 * 1024 * 1024))

    class CancelledError(Exception):
        pass

    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise CancelledError

    with pytest.raises(CancelledError):
        classify_pre_acquisition(
            sidecar,
            fallback_provider=Provider.CODEX,
            source_only=False,
            size_bytes=sidecar.stat().st_size,
            checkpoint=checkpoint,
        )


def test_whole_zip_member_revision_is_hashed_without_buffering_the_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preserved whole member is hashed in chunks, as acquisition streams it.

    Anti-vacuity: replaying the member through an unbounded ``handle.read()``
    (the payload replay's whole-member branch) trips the guard below.
    """
    from polylogue.sources import decoders

    root = tmp_path / "account"
    root.mkdir()
    member = b'{"type":"user","sessionId":"s1","uuid":"u1","message":{"role":"user","content":"hi"}}\n' * 50
    bundle = root / "export.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("projects/p/s1.jsonl", member)
    original_open = decoders.open_zip_entry

    class ChunkOnlyReader:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> ChunkOnlyReader:
            return self

        def __exit__(self, *_exc: object) -> None:
            self._handle.close()

        def read(self, size: int = -1) -> bytes:
            assert size > 0, "whole ZIP member buffered in memory"
            return bytes(self._handle.read(size))

    monkeypatch.setattr(
        decoders,
        "open_zip_entry",
        lambda zf, info: ChunkOnlyReader(original_open(zf, info)),
    )
    baseline = capture_production_source_baseline(
        (WatchSource("claude-code", root, layout=export_drop_layout((".zip",))),), operation_id="stream"
    )
    accepted = baseline.accepted
    assert [(row.path, row.revision, row.material_bytes) for row in accepted] == [
        (f"{bundle}:projects/p/s1.jsonl", hashlib.sha256(member).hexdigest(), len(member))
    ]


_CODEX_META = b'{"type":"session_meta","payload":{"id":"grow","timestamp":"2026-06-02T00:00:00Z"}}\n'


def _codex_turn(index: int) -> bytes:
    return (
        b'{"type":"response_item","payload":{"type":"message","id":"m%d","role":"user",'
        b'"content":[{"type":"input_text","text":"turn %d"}]}}\n' % (index, index)
    )


def _baselined_growing_file(tmp_path: Path, content: bytes) -> tuple[Path, ProductionSourceBaseline]:
    """Capture a live JSONL through the production walk, as a cold build does."""
    root = tmp_path / "codex"
    root.mkdir()
    path = root / "rollout-2026-06-02T00-00-00-grow.jsonl"
    path.write_bytes(content)
    source = WatchSource("codex", root, layout=export_drop_layout((".jsonl",)))
    baseline = capture_production_source_baseline((source,), operation_id="op")
    assert [(row.path, row.revision) for row in baseline.accepted] == [
        (str(path), hashlib.sha256(content).hexdigest())
    ], "sanity: the live file is an accepted baseline revision"
    return path, baseline


def _retain_chain(
    archive_root: Path,
    path: Path,
    full: bytes,
    tails: tuple[tuple[bytes, int], ...] = (),
    *,
    tail_authority: RawRevisionAuthority = RawRevisionAuthority.BYTE_PROVEN,
) -> list[str]:
    """Retain a whole-file capture and ordered append tails as intake binds them.

    Each tail is ``(payload, append_start_offset)``; a byte-proven tail names
    the previous member as its predecessor and the capture as its baseline.
    """
    if not (archive_root / "source.db").exists():
        initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        full_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=full,
            source_path=str(path),
            canonical_source_path=str(path),
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            full_raw_id, RawRevisionEnvelope("codex-session:grow", RawRevisionKind.FULL, "revision-0", 0)
        )
        raw_ids = [full_raw_id]
        previous_revision = "revision-0"
        for generation, (payload, start) in enumerate(tails, start=1):
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=str(path),
                canonical_source_path=str(path),
                source_index=-1,
                acquired_at_ms=1 + generation,
                post_parse=True,
            )
            proven = tail_authority is RawRevisionAuthority.BYTE_PROVEN
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    "codex-session:grow",
                    RawRevisionKind.APPEND,
                    f"revision-{generation}",
                    generation,
                    predecessor_source_revision=previous_revision,
                    predecessor_raw_id=raw_ids[-1] if proven else None,
                    baseline_raw_id=full_raw_id if proven else None,
                    append_start_offset=start,
                    append_end_offset=start + len(payload),
                    authority=tail_authority,
                ),
            )
            raw_ids.append(raw_id)
            previous_revision = f"revision-{generation}"
    return raw_ids


def test_a_revision_that_grew_before_intake_is_proven_by_the_larger_retained_capture(tmp_path: Path) -> None:
    """A JSONL that grows between baseline hashing and intake is still retained.

    Intake captures the larger file, so no raw row carries the baselined
    whole-file hash. The retained capture's leading bytes are the baselined
    revision, and verification reads them to prove it.

    Anti-vacuity: drop ``_retained_as_prefix`` and verify raises
    ``unretained revision`` although every baselined byte is retained.
    """
    earlier = _CODEX_META + _codex_turn(1)
    path, baseline = _baselined_growing_file(tmp_path, earlier)
    path.write_bytes(earlier + _codex_turn(2))
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with pytest.raises(ProductionBaselineError, match="unretained revision"):
        baseline.verify(archive_root / "source.db")

    _retain_chain(archive_root, path, path.read_bytes())

    baseline.verify(archive_root / "source.db")
    assert unretained_source_material(baseline, archive_root / "source.db", 4096, 4096) == (0, 0, 0)


def test_baselined_and_earlier_accepted_revisions_are_proven_through_an_ordered_append_chain(
    tmp_path: Path,
) -> None:
    """Every demanded prefix is read back from the capture and its ordered tails.

    The earlier accepted revision ends inside the first tail and the current
    one at the end of the second, so both proofs cross member boundaries.

    Anti-vacuity: stop the chain at the capture (or drop
    ``_retained_as_prefix``) and verify raises ``unretained revision`` for
    both revisions.
    """
    capture = _CODEX_META + _codex_turn(1)
    first_tail, second_tail = _codex_turn(2), _codex_turn(3)
    whole = capture + first_tail + second_tail
    path, current = _baselined_growing_file(tmp_path, whole)
    earlier_bytes = whole[: len(capture) + 7]
    earlier = _seal(
        "op",
        "signature",
        (
            SourceDecision(
                "codex",
                str(path),
                "accepted",
                "file",
                hashlib.sha256(earlier_bytes).hexdigest(),
                material_bytes=len(earlier_bytes),
            ),
        ),
    )
    merged = merge_pending_production_baseline(current, earlier)
    assert len(merged.accepted) == 2, "sanity: the earlier revision stays demanded"

    archive_root = tmp_path / "archive"
    _retain_chain(
        archive_root,
        path,
        capture,
        ((first_tail, len(capture)), (second_tail, len(capture) + len(first_tail))),
    )

    merged.verify(archive_root / "source.db")


def _retained_blob(archive_root: Path, raw_id: str) -> Path:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        (blob_hash,) = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    finally:
        conn.close()
    return BlobStore(archive_root / "blob").blob_path(bytes(blob_hash).hex())


@pytest.mark.parametrize("defect", ["hole", "unproven_tail", "corrupt_tail", "rewritten_prefix", "missing_blob"])
def test_a_chain_that_does_not_reproduce_the_revision_leaves_it_unretained(tmp_path: Path, defect: str) -> None:
    """Metadata alone never proves retention: every defect keeps the revision demanded.

    A quarantined tail names no predecessor, so it never joins a chain.

    Anti-vacuity: accept a chain by its offsets without reading the blobs and
    the corrupt, rewritten and missing cases pass; drop the contiguity check
    and the hole case passes, since its bytes concatenate to the revision.
    """
    capture = _CODEX_META + _codex_turn(1)
    tail = _codex_turn(2)
    whole = capture + tail
    path, baseline = _baselined_growing_file(tmp_path, whole)
    archive_root = tmp_path / "archive"
    retained_capture = capture.replace(b"turn 1", b"turn 9") if defect == "rewritten_prefix" else capture
    start = len(capture) + 1 if defect == "hole" else len(capture)
    authority = RawRevisionAuthority.QUARANTINED if defect == "unproven_tail" else RawRevisionAuthority.BYTE_PROVEN
    raw_ids = _retain_chain(archive_root, path, retained_capture, ((tail, start),), tail_authority=authority)
    tail_blob = _retained_blob(archive_root, raw_ids[1])
    if defect == "corrupt_tail":
        corrupted = bytearray(tail_blob.read_bytes())
        corrupted[0] ^= 0x01
        tail_blob.chmod(0o600)
        tail_blob.write_bytes(bytes(corrupted))
    if defect == "missing_blob":
        tail_blob.unlink()

    with pytest.raises(ProductionBaselineError, match="unretained revision") as raised:
        baseline.verify(archive_root / "source.db")
    assert not isinstance(raised.value, ProductionBaselineReadUnavailableError)


def test_a_retained_blob_read_fault_is_retryable_not_unretained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient read of retained bytes cannot settle the build as unretained.

    Anti-vacuity: fold every blob ``OSError`` into "not proven" and verify
    raises the permanent ``unretained revision`` refusal instead.
    """
    from polylogue.sources.live import production_baseline

    earlier = _CODEX_META + _codex_turn(1)
    path, baseline = _baselined_growing_file(tmp_path, earlier)
    path.write_bytes(earlier + _codex_turn(2))
    archive_root = tmp_path / "archive"
    _retain_chain(archive_root, path, path.read_bytes())

    class FaultingBlob:
        def open(self, _mode: str) -> Any:
            raise OSError(errno.EIO, "blob read failed")

    class FaultingBlobStore:
        def __init__(self, _root: Path) -> None:
            pass

        def blob_path(self, _blob_hash: str) -> FaultingBlob:
            return FaultingBlob()

    monkeypatch.setattr(production_baseline, "BlobStore", FaultingBlobStore)
    with pytest.raises(ProductionBaselineReadUnavailableError, match="retained blob"):
        baseline.verify(archive_root / "source.db")


def _write_codex_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY, title TEXT, cwd TEXT, created_at_ms INTEGER,
                updated_at_ms INTEGER, source TEXT, model TEXT, agent_nickname TEXT,
                agent_role TEXT, archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT, status TEXT);
            INSERT INTO threads (id, title, cwd, created_at_ms, updated_at_ms, source, archived)
            VALUES ('thread-1', 'A staged thread', '/repo', 1000, 2000, 'cli', 0);
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_explicit_sqlite_import_retains_its_original_coordinate_outside_watch_roots(
    workspace_env: dict[str, Path],
) -> None:
    """The retired watched-inbox snapshot producer cannot create a second raw identity."""
    from polylogue.operations.import_staging import import_staging_root
    from polylogue.operations.ingest_inputs import discover_ingest_input_spool, retain_input_page
    from polylogue.sources.live.watcher import daemon_watch_sources
    from polylogue.sources.source_staging import stage_source_input
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    archive_root = workspace_env["archive_root"]
    original = workspace_env["home_dir"] / ".codex" / "state_5.sqlite"
    _write_codex_state_db(original)
    staged = stage_source_input(original, import_staging_root(archive_root), check_stop=lambda: None)
    assert all(not staged.is_relative_to(source.root) for source in daemon_watch_sources())
    spool = discover_ingest_input_spool(staged, source_path=str(original), check_stop=lambda: None)
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    try:
        [retained] = retain_input_page(spool, after_coordinate=None, publisher=publisher, check_stop=lambda: None)
        assert retained.coordinate == "input:0"
        assert retained.source_path == str(original)
        assert retained.captured_identity is not None
        assert retained.captured_identity.canonical_source_path == str(original.resolve())
    finally:
        publisher.discard_pending()
        spool.unlink(missing_ok=True)


def test_the_default_codex_state_source_baselines_only_its_declared_jsonl_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install-level Codex sidecars are demanded; other root JSONL is not claimed.

    Anti-vacuity (polylogue-ez5b9, 11.F069): drop the two sidecar entries
    from the declared ``codex-state`` layout and neither is accepted; widen
    the layout to the suffix, or to any depth, and ``other.jsonl`` or the
    nested log joins them.
    """
    from polylogue.sources.live.watcher import default_sources

    monkeypatch.setenv("HOME", str(tmp_path))
    codex = tmp_path / ".codex"
    rollout = codex / "sessions" / "2026" / "06" / "02" / "rollout-2026-06-02T00-00-00-grow.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_bytes(_CODEX_META + _codex_turn(1))
    index = codex / "session_index.jsonl"
    index.write_text('{"id":"grow","thread_name":"A curated title"}\n')
    history = codex / "history.jsonl"
    history.write_text('{"session_id":"grow","ts":1,"text":"first prompt"}\n')
    (codex / "other.jsonl").write_text('{"session_id":"grow","ts":1,"text":"stray"}\n')
    (codex / "log").mkdir()
    (codex / "log" / "history.jsonl").write_text('{"session_id":"grow","ts":1,"text":"nested"}\n')
    sources = tuple(source for source in default_sources() if source.name in {"codex", "codex-state"})

    baseline = capture_production_source_baseline(sources, operation_id="codex")

    assert {(row.source, row.path) for row in baseline.accepted} == {
        ("codex", str(rollout)),
        ("codex-state", str(index)),
        ("codex-state", str(history)),
    }
