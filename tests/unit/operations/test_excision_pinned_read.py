"""Excision previews stay coherent across archive publication changes."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs
from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor, PlanStaleError
from polylogue.operations.operation_context import open_operation_read
from polylogue.security.excision import plan_session_excision
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _seed_session(root: Path) -> str:
    initialize_active_archive_root(root)
    session_id = "codex-session:pinned-preview"
    raw_id = "raw-pinned-preview"
    with closing(sqlite3.connect(root / "source.db")) as connection, connection:
        connection.execute(
            "INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms) "
            "VALUES (?, 'codex-session', 'pinned-preview', ?, zeroblob(32), 0, 1000)",
            (raw_id, str(root / "original.jsonl")),
        )
    with closing(sqlite3.connect(root / "index.db")) as connection, connection:
        connection.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES ('pinned-preview', 'codex-session', ?, 'Pinned preview', zeroblob(32), 1000, 2000)",
            (raw_id,),
        )
    return session_id


def test_excision_preview_uses_its_pinned_view_and_reprepare_observes_republication(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    session_id = _seed_session(root)
    args = SessionExcisionArgs(
        archive_root=root,
        session_id=session_id,
        reason="fixture",
        actor="user:test",
        cascade_lineage=False,
    )
    actuator = SessionExcisionActuator()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        executor = OperationExecutor.for_archive_root(root)
        binding = runtime_operation_binding(actuator)
        capabilities = frozenset(
            capability for policy in binding.spec.target_authority for capability in policy.required_capabilities
        )
        principal = MutationPrincipal("user:test", capabilities, "cli", "write")
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
        authorization = executor.authorize_bound(binding, preview, principal)

    with open_operation_read(root) as pinned:
        original = plan_session_excision(pinned.archive, session_id)
        original_path = original.targets[0].raw_targets[0].source_path

        source_copy = root / "source-replacement.db"
        shutil.copyfile(root / "source.db", source_copy)
        with closing(sqlite3.connect(source_copy)) as connection, connection:
            connection.execute(
                "UPDATE raw_sessions SET source_path = ? WHERE raw_id = ?",
                (str(root / "replacement.jsonl"), original.targets[0].raw_targets[0].raw_id),
            )
        source_copy.replace(root / "source.db")

        generation = root / ".index-generations" / "replacement"
        generation.mkdir(parents=True)
        replacement_index = generation / "index.db"
        shutil.copyfile(ArchiveLocation.resolve(root).active_index_path, replacement_index)
        with closing(sqlite3.connect(replacement_index)) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        (root / ".index-active-pointer").write_text(str(replacement_index), encoding="utf-8")

        still_pinned = plan_session_excision(pinned.archive, session_id)
        assert still_pinned.found is True
        assert still_pinned.targets[0].raw_targets[0].source_path == original_path

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        executor = OperationExecutor.for_archive_root(root)
        try:
            executor.execute_bound(binding, preview, authorization, args)
        except PlanStaleError:
            pass
        else:
            raise AssertionError("mutation reprepare must reject the plan after active index replacement")
        assert (
            archive._conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
            == 0
        )
        with closing(sqlite3.connect(root / "source.db")) as connection:
            assert (
                connection.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = 'raw-pinned-preview'").fetchone()[
                    0
                ]
                == 1
            )
        with closing(sqlite3.connect(root / "source.db")) as connection:
            assert connection.execute("SELECT COUNT(*) FROM excised_content").fetchone()[0] == 0
