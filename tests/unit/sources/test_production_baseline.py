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

from polylogue.sources.live.production_baseline import (
    ProductionBaselineError,
    ProductionBaselineObservationCancelledError,
    ProductionBaselineReadUnavailableError,
    SourceDecision,
    _revision,
    _seal,
    capture_production_source_baseline,
    merge_pending_production_baseline,
)
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.sqlite_snapshot import sqlite_member_revision


def _source_db(path: Path, rows: tuple[tuple[str, str] | tuple[str, int, str], ...]) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE raw_sessions(source_path TEXT, source_index INTEGER, blob_hash BLOB)")
        conn.executemany(
            "INSERT INTO raw_sessions VALUES (?, ?, ?)",
            [
                (row[0], row[1], bytes.fromhex(row[2])) if len(row) == 3 else (row[0], 0, bytes.fromhex(row[1]))
                for row in rows
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return path


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
    source = (WatchSource("account", root, suffixes=(".json",), required=True),)

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
        patcher.setattr(production_baseline, "replay_zip_entry_acquisition_payloads", unreadable_member)
        baseline = capture_production_source_baseline(
            (WatchSource("account", root, suffixes=(".zip",)),), operation_id="zip-fault"
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
    source = (WatchSource("account", root, suffixes=(".json",)),)
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
    source = (WatchSource("account", root, suffixes=(".zip",)),)

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
    source = WatchSource("account", root, suffixes=(".json",), required=True)
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
    conn = sqlite3.connect(source_db)
    try:
        conn.execute("INSERT INTO raw_sessions VALUES (?, ?, ?)", (str(accepted), 0, bytes.fromhex(old_revision)))
        conn.commit()
    finally:
        conn.close()
    baseline.verify(source_db)


def test_baseline_records_intake_exclusions_instead_of_requiring_retention(tmp_path: Path) -> None:
    """Each pre-acquisition exclusion intake applies is the baseline's disposition too.

    Anti-vacuity: without the shared ``classify_pre_acquisition`` decision
    every one of these files is ``accepted`` and ``verify`` raises for three
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
            WatchSource("codex", codex, suffixes=(".jsonl", ".sqlite")),
            WatchSource("gemini-cli", gemini, suffixes=(".json",)),
        ),
        operation_id="intake-exclusions",
    )
    decisions = {row.path: (row.disposition, row.reason) for row in baseline.decisions}
    assert decisions[str(meta_only)] == ("excluded", "intake_excluded:declared artifact rule: not parsed as a session")
    assert decisions[str(unverified_state)] == (
        "excluded",
        "intake_excluded:unsupported source class",
    )
    assert decisions[str(logs)] == ("excluded", "intake_excluded:path rule classifies this as non-session evidence")
    assert [row.path for row in baseline.accepted] == [str(rollout)]
    baseline.verify(
        _source_db(tmp_path / "source.db", ((str(rollout), hashlib.sha256(rollout.read_bytes()).hexdigest()),))
    )


def test_external_link_is_alias_only_with_independent_source(tmp_path: Path) -> None:
    account = tmp_path / "account"
    account.mkdir()
    (account / "one.json").write_text("{}")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "account").symlink_to(account, target_is_directory=True)
    typed = WatchSource("account", account, suffixes=(".json",), required=True)
    alias = WatchSource("inbox", inbox, suffixes=(".json",))
    baseline = capture_production_source_baseline((alias, typed), operation_id="build-2")
    assert any(row.path == str(inbox / "account") and row.disposition == "alias" for row in baseline.decisions)
    assert len(baseline.accepted) == 1
    unresolved = capture_production_source_baseline((alias,), operation_id="build-3")
    assert any(row.path == str(inbox / "account") and row.disposition == "fault" for row in unresolved.decisions)


def test_broken_accepted_looking_link_is_fault(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    (root / "gone.json").symlink_to(root / "missing.json")
    source = WatchSource("account", root, suffixes=(".json",), required=True)
    baseline = capture_production_source_baseline((source,), operation_id="build-4")
    assert any(row.disposition == "fault" and row.path.endswith("gone.json") for row in baseline.decisions)


def test_directory_cycle_is_alias_when_canonical_target_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    (root / "one.json").write_text("{}")
    (root / "cycle").symlink_to(root, target_is_directory=True)
    baseline = capture_production_source_baseline(
        (WatchSource("account", root, suffixes=(".json",)),), operation_id="cycle"
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
            WatchSource("claude-code-history", claude, suffixes=()),
            WatchSource("codex-state", codex, suffixes=(".sqlite",), allow_path_scoped_artifacts=False),
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

    def unavailable_revision(_path: Path) -> tuple[str, int]:
        raise unavailable.value

    monkeypatch.setattr(production_baseline, "sqlite_member_revision_and_size", unavailable_revision)
    baseline = capture_production_source_baseline(
        (WatchSource("codex-state", codex, suffixes=(".sqlite",), allow_path_scoped_artifacts=False),),
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
        (WatchSource("account", root, suffixes=(".zip",)),), operation_id="build-7"
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
        (WatchSource("account", root, suffixes=(".zip",)),), operation_id="split"
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
    conn = sqlite3.connect(source_db)
    try:
        second_path, second_index, second_revision = _zip_row(members[1])
        conn.execute(
            "INSERT INTO raw_sessions VALUES (?, ?, ?)",
            (second_path, second_index, bytes.fromhex(second_revision)),
        )
        conn.commit()
    finally:
        conn.close()
    baseline.verify(source_db)


def test_zip_declared_binary_and_markdown_artifacts_follow_live_validator(tmp_path: Path) -> None:
    root = tmp_path / "account"
    root.mkdir()
    bundle = root / "export.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("tool-results/one.bin", b"\xff\x00opaque")
        archive.writestr("brain/one.md", b"# note\n")
    baseline = capture_production_source_baseline(
        (WatchSource("account", root, suffixes=(".zip",)),), operation_id="artifacts"
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
        (WatchSource("codex", root, suffixes=(".jsonl",)),),
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
    current = capture_production_source_baseline((WatchSource("codex", root, suffixes=(".jsonl",)),), operation_id="op")
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
    current = capture_production_source_baseline((WatchSource("codex", root, suffixes=(".jsonl",)),), operation_id="op")
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
            (WatchSource("codex", root, suffixes=(".sqlite",)),), operation_id="unreadable"
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
        (WatchSource("codex", root, suffixes=(".sqlite",)),), operation_id="not-a-database"
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
