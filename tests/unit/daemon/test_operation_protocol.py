"""Declared operation protocol, exercised through a real daemon and the CLI route.

Each test drives the production stack (``running_daemon_operations`` or
``cli_daemon_archive``): the declaration fields a client relies on -- the
authorization a request must present and the recovery a lost outcome calls
for -- and the equivalence of the three routes that write sessions.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from tests.infra.daemon_operations import cli_daemon_archive, running_daemon_operations


def _write_codex_session(path: Path, session_id: str, texts: tuple[str, ...]) -> Path:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-01-01T00:00:00Z"}}
    ]
    for index, text in enumerate(texts):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{session_id}-message-{index}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": [{"type": "input_text" if index % 2 == 0 else "output_text", "text": text}],
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _two_sessions(directory: Path) -> tuple[Path, Path]:
    return (
        _write_codex_session(directory / "first.jsonl", "protocol-first", ("question one", "answer one")),
        _write_codex_session(
            directory / "second.jsonl", "protocol-second", ("question two", "answer two", "follow-up")
        ),
    )


def _normalized_material(archive_root: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        return {
            "sessions": sorted(
                conn.execute(
                    "SELECT session_id, origin, title, title_source, content_hash, message_count FROM sessions"
                )
            ),
            "messages": sorted(
                conn.execute("SELECT message_id, session_id, role, material_origin, content_hash FROM messages")
            ),
        }


def _session_ids(archive_root: Path) -> list[str]:
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        return sorted(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions"))


def test_cli_import_daemon_ingest_and_from_empty_build_write_identical_material(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One session writer: from-empty build, daemon ``ingest`` and ``polylogue import`` agree.

    The one-shot build hands files to the live batch writer; the declared
    ``ingest`` operation admits them through a frozen source generation; the
    CLI stages them and submits that same operation. All three must write the
    same normalized sessions and messages, and re-offering ingested files
    changes nothing.

    Anti-vacuity: let the ``ingest`` cohort parse retained raws without the
    provider's session assembly (``parse_retained_raw_sessions`` instead of
    ``_parse_assembled_retained_raw``) and its sessions keep the bare
    native id as title while the from-empty build's carry the assembled one.
    """
    import asyncio

    from polylogue.config import Source
    from polylogue.operations.canonical_archive_ingest import ingest_one_shot_archive
    from polylogue.storage.blob_store import reset_blob_store

    first, second = _two_sessions(tmp_path / "capture-files")

    from_empty_root = one_shot_workspace_env["archive_root"]
    asyncio.run(
        ingest_one_shot_archive(
            from_empty_root, [Source(name="codex", path=first), Source(name="codex", path=second)], parse_workers=1
        )
    )
    expected = _normalized_material(from_empty_root)
    assert len(expected["sessions"]) == 2
    assert len(expected["messages"]) == 5

    daemon_root = tmp_path_factory.mktemp("daemon-incremental")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(daemon_root))
    reset_blob_store()
    with running_daemon_operations(daemon_root, session_derivation=True) as stack:
        root = str(stack.archive_root)
        for request_id, path in (("first", first), ("second", second), ("reoffer", first)):
            envelope = stack.client.operation_to_completion(
                "ingest", {"path": str(path)}, archive_root=root, request_id=f"differential-{request_id}"
            )
            assert envelope is not None and envelope["outcome"] == "completed", envelope
    daemon_material = _normalized_material(daemon_root)
    assert daemon_material == expected, (daemon_material, expected)

    cli_root = tmp_path_factory.mktemp("cli-import")
    reset_blob_store()
    with cli_daemon_archive(cli_root, monkeypatch, session_derivation=True):
        from polylogue.cli.click_app import cli

        for path in (first, second):
            result = CliRunner().invoke(
                cli, ["import", str(path), "--wait", "--timeout", "120"], catch_exceptions=False
            )
            assert result.exit_code == 0, result.output
    cli_material = _normalized_material(cli_root)
    assert cli_material == expected, (cli_material, expected)


def test_reimporting_an_excised_file_is_a_typed_permanent_refusal(tmp_path: Path) -> None:
    """Re-offering excised bytes through ``ingest`` is refused, settled and effect-free.

    Every input of the request is excised, so the publication flush refuses
    its bytes and the request accepts nothing: it settles ``failed`` with the
    non-retryable ``ContentExcisedError``, never ``indeterminate``, and the
    excised bytes gain no reservation and no retained source item that would
    keep them from blob GC (polylogue-u6jyu).

    Anti-vacuity: let the ingest input route reserve excised bytes again (drop
    the excision read in ``BlobPublicationReservationStore.reserve_many``) and
    the request completes with a new source item retaining the excised file.
    """
    first, _second = _two_sessions(tmp_path / "capture-files")
    with running_daemon_operations(tmp_path / "archive", session_derivation=True) as stack:
        root = str(stack.archive_root)
        imported = stack.client.operation_to_completion("ingest", {"path": str(first)}, archive_root=root)
        assert imported is not None and imported["outcome"] == "completed", imported
        (session_id,) = _session_ids(stack.archive_root)
        excised = stack.client.operation_to_completion(
            "mutation.session.excision",
            {"session_id": session_id, "reason": "test", "actor": "user:local", "confirm": True},
            archive_root=root,
        )
        assert excised is not None and excised["outcome"] == "completed", excised
        file_hash = bytes.fromhex(hashlib.sha256(first.read_bytes()).hexdigest())
        retained_before = _blob_retention(stack.archive_root, file_hash)

        again = stack.client.operation_to_completion(
            "ingest", {"path": str(first)}, archive_root=root, request_id="reimport-excised"
        )

        assert again is not None and again["outcome"] == "failed", again
        assert again["error"]["code"] == "ContentExcisedError", again
        assert again["error"]["retryable"] is False, again
        assert again["accepted_reference"] is None, again
        assert _session_ids(stack.archive_root) == []
        assert _blob_retention(stack.archive_root, file_hash) == retained_before


def _blob_retention(archive_root: Path, blob_hash: bytes) -> tuple[int, int]:
    """Source items and publication reservations that keep ``blob_hash`` from GC."""
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        items = conn.execute("SELECT COUNT(*) FROM source_items WHERE blob_hash = ?", (blob_hash,)).fetchone()[0]
        reservations = conn.execute(
            "SELECT COUNT(*) FROM blob_publication_reservations WHERE blob_hash = ?", (blob_hash,)
        ).fetchone()[0]
    return int(items), int(reservations)


def test_confirmation_bound_mutation_refuses_an_unconfirmed_request(tmp_path: Path) -> None:
    """A ``confirmation`` declaration is enforced by the daemon, not the client.

    Anti-vacuity: drop ``authorization=DaemonAuthorization.CONFIRMATION`` from
    the ``mutation.session.excision`` declaration and the unconfirmed request
    excises the session.
    """
    from polylogue.operations.daemon_protocol import DaemonAuthorization, daemon_operation_spec

    spec = daemon_operation_spec("mutation.session.excision")
    assert spec is not None and spec.authorization is DaemonAuthorization.CONFIRMATION
    first, _second = _two_sessions(tmp_path / "capture-files")
    with running_daemon_operations(tmp_path / "archive", session_derivation=True) as stack:
        root = str(stack.archive_root)
        imported = stack.client.operation_to_completion("ingest", {"path": str(first)}, archive_root=root)
        assert imported is not None and imported["outcome"] == "completed", imported
        (session_id,) = _session_ids(stack.archive_root)

        unconfirmed = stack.client.operation_to_completion(
            "mutation.session.excision",
            {"session_id": session_id, "reason": "test", "actor": "user:local"},
            archive_root=root,
        )

        assert unconfirmed is not None and unconfirmed["outcome"] != "completed", unconfirmed
        assert _session_ids(stack.archive_root) == [session_id]


def test_accepted_ingest_waits_out_audit_continuity_contention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A concurrent continuity transition at the accepted load is waited out.

    After acceptance, the ingest loads its own audited execution under a
    settled audit read. Another request's continuity transition can hold the
    archive's continuity lock at that instant; the accepted ingest must still
    complete. Anti-vacuity: take that read without ``wait_for_lock`` again and
    the held lock refuses it with ``AuditContinuityPendingError``, so the
    accepted ingest ends ``indeterminate`` with stop reason ``refused``.
    """
    import threading

    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.storage.sqlite.audit_continuity import _coordinator_lock

    first, _second = _two_sessions(tmp_path / "capture-files")
    contended: list[str] = []
    original_phase = DaemonOperationRuntime.compute_phase

    async def contended_phase(self: DaemonOperationRuntime, work: object) -> object:
        if getattr(work, "__name__", "") == "load_started":
            lock = _coordinator_lock(self.archive_root)
            held = threading.Event()
            release = threading.Event()

            def hold() -> None:
                with lock:
                    held.set()
                    release.wait(timeout=5.0)

            holder = threading.Thread(target=hold, name="continuity-contender", daemon=True)
            holder.start()
            assert held.wait(timeout=5.0)
            contended.append("load_started")
            threading.Timer(0.3, release.set).start()
        return await original_phase(self, work)  # type: ignore[arg-type]

    monkeypatch.setattr(DaemonOperationRuntime, "compute_phase", contended_phase)
    with running_daemon_operations(tmp_path / "archive", session_derivation=True) as stack:
        envelope = stack.client.operation_to_completion(
            "ingest", {"path": str(first)}, archive_root=str(stack.archive_root), request_id="contended-ingest"
        )

    assert contended == ["load_started"]
    assert envelope is not None and envelope["outcome"] == "completed", envelope


def test_live_request_recovery_preserves_another_ingests_preaccept_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A work-event request cannot reclaim the live ingest's committed header.

    Pause ingest after its preparation header commits, outside compute and
    writer admission. A second real staged request then runs recovery before
    acquiring its work event. Moving the global preparation sweep back into
    request recovery deletes the header and fails this ingest's next page.
    """
    import asyncio
    import threading
    from collections.abc import Callable
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import closing
    from typing import Any

    from polylogue.daemon.operation_runtime import DaemonOperationRuntime

    prepared = threading.Event()
    release = threading.Event()
    original_phase = DaemonOperationRuntime.compute_phase

    async def pause_page(self: DaemonOperationRuntime, work: Callable[[], Any]) -> Any:
        if getattr(work, "__name__", "") == "compute_page" and not prepared.is_set():
            prepared.set()
            await asyncio.to_thread(release.wait)
        return await original_phase(self, work)

    first, second = _two_sessions(tmp_path / "capture-files")
    with running_daemon_operations(tmp_path / "archive", session_derivation=True) as stack:
        root = str(stack.archive_root)
        imported = stack.client.operation_to_completion("ingest", {"path": str(first)}, archive_root=root)
        assert imported is not None and imported["outcome"] == "completed", imported
        (session_id,) = _session_ids(stack.archive_root)
        monkeypatch.setattr(DaemonOperationRuntime, "compute_phase", pause_page)
        with ThreadPoolExecutor(max_workers=1) as requests:
            ingest = requests.submit(
                stack.client.operation_to_completion,
                "ingest",
                {"path": str(second)},
                archive_root=root,
                request_id="paused-preaccept-ingest",
            )
            try:
                assert prepared.wait(timeout=30), "ingest never committed its preparation header"
                with closing(sqlite3.connect(f"file:{stack.archive_root / 'source.db'}?mode=ro", uri=True)) as source:
                    original = source.execute(
                        "SELECT source_generation_id, publisher_id FROM prepared_source_manifests p "
                        "WHERE NOT EXISTS (SELECT 1 FROM source_generations g "
                        "WHERE g.source_generation_id=p.source_generation_id)"
                    ).fetchall()
                assert len(original) == 1
                recorded = stack.client.operation_to_completion(
                    "mutation.facade.record_work_event",
                    {
                        "session_id": session_id,
                        "event_id": "concurrent-preaccept-event",
                        "event_type": "decision",
                        "summary": "retain the active input preparation",
                        "payload": {},
                    },
                    archive_root=root,
                    request_id="concurrent-preaccept-work-event",
                )
                assert recorded is not None and recorded["outcome"] == "completed", recorded
                with closing(sqlite3.connect(f"file:{stack.archive_root / 'source.db'}?mode=ro", uri=True)) as source:
                    remaining = source.execute(
                        "SELECT source_generation_id, publisher_id FROM prepared_source_manifests p "
                        "WHERE NOT EXISTS (SELECT 1 FROM source_generations g "
                        "WHERE g.source_generation_id=p.source_generation_id)"
                    ).fetchall()
                assert remaining == original
            finally:
                release.set()
            completed = ingest.result(timeout=120)
            assert completed is not None and completed["outcome"] == "completed", completed
        assert len(_session_ids(stack.archive_root)) == 2


def test_cli_delete_refuses_a_selection_that_drifted_after_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's delete consumes its exact authorization, so drift refuses the whole plan.

    One of two authorized sessions disappears between ``authorize`` and
    ``execute``. The CLI must report a refusal and the surviving session must
    remain: the authorization named a selection that no longer exists.

    Anti-vacuity: execute the delete from a fresh read of the selection instead
    of the authorized plan and the surviving session is deleted.
    """
    import polylogue.cli.archive_query as archive_query
    from polylogue.cli.shared.types import AppEnv
    from polylogue.config import Config
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import authorized_session_removal, write_lease

    first, second = _two_sessions(tmp_path / "capture-files")
    with cli_daemon_archive(tmp_path / "archive", monkeypatch, session_derivation=True) as stack:
        root = str(stack.archive_root)
        for path in (first, second):
            imported = stack.client.operation_to_completion("ingest", {"path": str(path)}, archive_root=root)
            assert imported is not None and imported["outcome"] == "completed", imported
        drifted, surviving = _session_ids(stack.archive_root)
        submit = archive_query._submit_mutation_operation

        def submit_then_drift(config: object, operation: str, payload: dict[str, object]) -> dict[str, object]:
            result = submit(config, operation, payload)  # type: ignore[arg-type]
            if operation == "mutation.session.delete.authorize":
                # The drift is another authorized removal of one target. An
                # unauthorized disappearance is itself refused, because the
                # audited preview still references the session (2e9aae50de).
                with (
                    write_lease("test.delete.drift", archive_root=stack.archive_root),
                    authorized_session_removal(
                        archive_root=stack.archive_root, plan_hash="test-drift-plan", session_ids=(drifted,)
                    ),
                    ArchiveStore.open_existing(stack.archive_root, read_only=False) as archive,
                ):
                    archive.delete_sessions((drifted,))
            return result

        monkeypatch.setattr(archive_query, "_submit_mutation_operation", submit_then_drift)
        config = Config(
            archive_root=stack.archive_root,
            render_root=tmp_path / "render",
            sources=[],
            db_path=stack.archive_root / "index.db",
        )
        monkeypatch.setattr(archive_query, "load_effective_config", lambda _env: config)

        with pytest.raises(click.ClickException):
            archive_query.execute_delete_by_session_ids(AppEnv(), [drifted, surviving], force=True, dry_run=False)

        assert _session_ids(stack.archive_root) == [surviving]


def test_cli_delete_of_a_zero_match_selection_submits_no_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delete whose selection matched nothing authorizes and executes nothing.

    The selection is resolved by the daemon's resident preview (c4e75fe32d):
    the CLI no longer reads IDs itself, so the preview is how it learns the
    selection is empty. The verb then refuses with its typed empty-selection
    usage error before any authorization, and the archive is unchanged.

    Anti-vacuity: drop the verb's ``count == 0`` branch after the preview and
    the CLI goes on to submit ``mutation.session.delete.authorize`` for an
    empty selection.
    """
    import polylogue.cli.archive_query as archive_query
    from polylogue.cli.click_app import cli

    first, _second = _two_sessions(tmp_path / "capture-files")
    with cli_daemon_archive(tmp_path / "archive", monkeypatch, session_derivation=True) as stack:
        root = str(stack.archive_root)
        imported = stack.client.operation_to_completion("ingest", {"path": str(first)}, archive_root=root)
        assert imported is not None and imported["outcome"] == "completed", imported
        before = _session_ids(stack.archive_root)
        submit = archive_query._submit_mutation_operation
        submitted: list[str] = []

        def recording(config: object, operation: str, payload: dict[str, object]) -> dict[str, object]:
            submitted.append(operation)
            return submit(config, operation, payload)  # type: ignore[arg-type]

        monkeypatch.setattr(archive_query, "_submit_mutation_operation", recording)
        result = CliRunner().invoke(cli, ["find", "origin:chatgpt-export", "then", "delete", "--yes"])

        from polylogue.cli.contextual_errors import EmptySelectionError

        assert result.exit_code == EmptySelectionError.exit_code, (result.output, repr(result.exception))
        assert submitted == ["mutation.session.delete.preview"]
        assert _session_ids(stack.archive_root) == before


def test_indeterminate_cli_write_reports_its_declared_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lost write outcome is typed ``operation_indeterminate`` with the declared recovery.

    Anti-vacuity: fold ``OperationIndeterminateRefusal`` back into the generic
    ``ClickException`` branch of ``run_machine_entry`` and the code reads
    ``runtime_error`` -- an ordinary failure a client may retry.
    """
    import polylogue.cli.archive_query as archive_query
    from polylogue.cli.click_app import cli
    from polylogue.cli.machine_main import run_machine_entry
    from polylogue.cli.operation_kernel import OperationIndeterminateError
    from polylogue.operations.daemon_protocol import daemon_operation_spec

    def lost_outcome(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
        raise OperationIndeterminateError(f"{operation} requires receipt recovery", request_id="lost-request")

    monkeypatch.setattr(archive_query, "_submit_mutation_operation", lost_outcome)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    argv = ["polylogue", "setting", "set", "subscription_tier", "max", "--format", "json"]
    monkeypatch.setattr(sys, "argv", argv)
    capsys.readouterr()

    with pytest.raises(SystemExit) as exit_info:
        run_machine_entry(cli, argv[1:])

    payload = json.loads(capsys.readouterr().out)
    spec = daemon_operation_spec("mutation.user.setting.set")
    assert spec is not None
    assert exit_info.value.code != 0
    assert payload["code"] == "operation_indeterminate", payload
    assert payload["details"] == {
        "operation": "mutation.user.setting.set",
        "recovery": spec.recovery.value,
        "request_id": "lost-request",
    }
