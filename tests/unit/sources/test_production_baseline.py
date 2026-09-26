"""Production discovery decisions and generation-bound revision conservation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from polylogue.sources.live.production_baseline import (
    ProductionBaselineError,
    SourceDecision,
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
        conn.execute("CREATE TABLE threads(id TEXT)")
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
