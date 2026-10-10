"""Completed retained pages survive interruption without borrowing stale custody."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

import polylogue
from devtools.isolated_environment import isolated_home_environment
from polylogue.daemon import cli
from polylogue.operations import index_reconvergence_startup as startup
from polylogue.operations.raw_observation_owner import RawObservationArchiveWork
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.empty_managed_index import mutate_fixture_database
from tests.infra.populated_managed_index import make_populated_stale_index


def _run_startup() -> None:
    asyncio.run(
        cli.run_daemon_services(
            sources=(),
            enable_watch=False,
            enable_browser_capture=False,
            browser_capture_host="127.0.0.1",
            browser_capture_port=8765,
        )
    )


def _inactive(root: Path) -> IndexGeneration:
    store = IndexGenerationStore.for_archive_root(root, repair_anchor=False)
    candidates = [store.load(path.parent.name) for path in store.generations_root.glob("gen-*/generation.json")]
    return next(candidate for candidate in candidates if candidate.state == "inactive")


def _finish(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[tuple[str, ...], ...]:
    offered: list[tuple[str, ...]] = []
    select = RawObservationArchiveWork.sidecar_owner_selector

    def observe(selected: tuple[str, ...]) -> Callable[[PreparedSessionSourceRead], Sequence[str]]:
        offered.append(selected)
        return select(selected)

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        raise PreflightReachedError

    monkeypatch.setattr(RawObservationArchiveWork, "sidecar_owner_selector", staticmethod(observe))
    monkeypatch.setattr(cli, "_check_schema_version_fast", preflight)
    with pytest.raises(PreflightReachedError):
        _run_startup()
    assert _inactive_count(root) == 0
    return tuple(offered)


def _inactive_count(root: Path) -> int:
    store = IndexGenerationStore.for_archive_root(root, repair_anchor=False)
    return sum(
        store.load(path.parent.name).state == "inactive"
        for path in store.generations_root.glob("gen-*/generation.json")
    )


@pytest.mark.parametrize("phase", ["uncommitted", "committed"])
def test_abrupt_next_page_exit_resumes_the_same_durable_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    root = tmp_path / "archive"
    old, _raw, _sessions = make_populated_stale_index(
        root,
        tmp_path / "external" / "session.jsonl",
        independent_raws=2,
    )
    env = isolated_home_environment(os.environ, home=tmp_path / "home")
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, runpy; sys.path[:] = {sys.path!r}; "
            f"import polylogue; assert polylogue.__file__ == {polylogue.__file__!r}; "
            f"sys.argv = ['tests.infra.retained_startup_crash', {phase!r}]; "
            "runpy.run_module('tests.infra.retained_startup_crash', run_name='__main__')",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == (71 if phase == "uncommitted" else 72), result.stderr
    candidate = _inactive(root)
    assert candidate.reconstruction_raw_id
    assert ArchiveLocation.resolve(root).active_index_path.resolve() == old
    with closing(open_readonly_connection(Path(candidate.index_path))) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE title='uncommitted damage'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == (1 if phase == "uncommitted" else 2)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    monkeypatch.setattr(startup, "_RAW_PAGE", 1)
    offered = _finish(root, monkeypatch)
    assert offered and all(candidate.reconstruction_raw_id not in scope for scope in offered)
    active = ArchiveLocation.resolve(root).active_index_path.resolve()
    assert active == Path(candidate.index_path)
    with closing(open_readonly_connection(active)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
        assert conn.execute("PRAGMA foreign_key_check").fetchone() is None


@pytest.mark.parametrize(
    "changed",
    [
        "recipe",
        "acquisition",
        "new-earlier-raw",
        "interpretation",
        "pruned-journal",
        "regressed-journal",
        "assertions",
        "destination",
    ],
)
def test_completed_page_replays_or_restarts_when_its_binding_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    root = tmp_path / "archive"
    make_populated_stale_index(root, tmp_path / "external" / "session.jsonl", independent_raws=1)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    monkeypatch.setattr(startup, "_RAW_PAGE", 1)
    checkpoint = IndexGenerationStore.checkpoint_reconstruction

    def stop_after_checkpoint(
        self: IndexGenerationStore,
        generation: IndexGeneration,
        *,
        raw_id: str,
        source_sequence: int,
        rewind: bool = False,
    ) -> IndexGeneration:
        checkpoint(self, generation, raw_id=raw_id, source_sequence=source_sequence, rewind=rewind)
        raise asyncio.CancelledError()

    with monkeypatch.context() as control:
        control.setattr(IndexGenerationStore, "checkpoint_reconstruction", stop_after_checkpoint)
        with pytest.raises(asyncio.CancelledError):
            _run_startup()
    candidate = _inactive(root)
    if changed == "recipe":
        path = Path(candidate.index_path).parent / "generation.json"
        metadata = json.loads(path.read_text())
        metadata["reconstruction_parser_identity"] = "prior-parser"
        path.write_text(json.dumps(metadata))
    elif changed == "acquisition":
        mutate_fixture_database(
            root / "source.db",
            "UPDATE raw_sessions SET acquired_at_ms=acquired_at_ms+1 WHERE raw_id=?",
            (candidate.reconstruction_raw_id,),
        )
    elif changed == "new-earlier-raw":
        from polylogue.core.enums import Provider
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from tests.infra.archive_templates import run_archive_fixture_write

        def acquire() -> str:
            with ArchiveStore.open_source_tier_acquisition(root) as archive:
                return archive.write_raw_payload(
                    provider=Provider.CODEX,
                    raw_id="!new-before-cursor",
                    source_path="new.jsonl",
                    canonical_source_path="new.jsonl",
                    acquired_at_ms=2,
                    payload=(
                        b'{"type":"session_meta","payload":{"id":"new-before-cursor"}}\n'
                        b'{"type":"response_item","payload":{"type":"message","id":"m-new",'
                        b'"role":"user","content":[{"type":"input_text","text":"new acquired prose"}]}}\n'
                    ),
                )

        assert asyncio.run(run_archive_fixture_write(root, acquire)) < candidate.reconstruction_raw_id
    elif changed in {"interpretation", "pruned-journal", "regressed-journal"}:
        mutate_fixture_database(
            root / "source.db",
            "UPDATE raw_authority_parser_census SET parser_fingerprint='prior-parser' WHERE raw_id=?",
            (candidate.reconstruction_raw_id,),
        )
        if changed in {"pruned-journal", "regressed-journal"}:
            mutate_fixture_database(root / "source.db", "DELETE FROM raw_existence_changes")
        if changed == "regressed-journal":
            # Model lost NORMAL-synchronous Source WAL commits after the FULL
            # candidate and its synchronized metadata have survived.
            mutate_fixture_database(root / "source.db", "UPDATE raw_existence_journal_control SET retained_floor=0")
    elif changed == "assertions":
        mutate_fixture_database(root / "user.db", "UPDATE assertions SET body_text='changed reference custody'")
    else:
        path = Path(candidate.index_path)
        saved = path.with_suffix(".saved")
        path.rename(saved)
        shutil.copyfile(saved, path)
    offered = _finish(root, monkeypatch)
    assert any(candidate.reconstruction_raw_id in scope for scope in offered)
    if changed in {"interpretation", "pruned-journal", "regressed-journal"}:
        assert ArchiveLocation.resolve(root).active_index_path.resolve() == Path(candidate.index_path)
    else:
        assert ArchiveLocation.resolve(root).active_index_path.resolve() != Path(candidate.index_path)
        assert not Path(candidate.index_path).exists()
    if changed == "new-earlier-raw":
        assert any("!new-before-cursor" in scope for scope in offered)
        with closing(open_readonly_connection(ArchiveLocation.resolve(root).active_index_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3


@pytest.mark.parametrize("interrupt", [False, True])
def test_shared_source_phase_refreshes_the_completed_prefix_or_rewinds_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: bool,
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationInspection
    from polylogue.storage.sqlite.archive_tiers import archive, revision_governance
    from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, derived_schema_identity
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from tests.infra.retained_replay import publish_retained_payload

    root = tmp_path / "archive"
    source = tmp_path / "external" / "session.jsonl"
    old, _raw, sessions = make_populated_stale_index(root, source)
    mutate_fixture_database(
        old, "UPDATE schema_identity SET identity=? WHERE tier='index'", (derived_schema_identity(DerivedTier.INDEX),)
    )
    payload = (
        b'{"type":"session_meta","payload":{"id":"retained-reconvergence","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"neutral-message","role":"user",'
        b'"content":[{"type":"input_text","text":"neutral retained prose"}]}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"second-message","role":"assistant",'
        b'"content":[{"type":"output_text","text":"neutral second prose"}]}}\n'
    )
    asyncio.run(
        publish_retained_payload(
            root, provider=Provider.CODEX, payload=payload, source_path=str(source), acquired_at_ms=1
        )
    )
    mutate_fixture_database(old, "UPDATE schema_identity SET identity='prior-runtime' WHERE tier='index'")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    monkeypatch.setattr(startup, "_RAW_PAGE", 1)
    checkpoint = IndexGenerationStore.checkpoint_reconstruction
    source_phase = RawObservationDerivation._apply_source_phase
    completed: list[IndexGeneration] = []
    touched: list[str] = []
    republished: list[tuple[str, ...]] = []
    apply = revision_governance.apply_raw_revision_replay

    def observe_replay(store: Any, plan: Any, *args: Any, **kwargs: Any) -> Any:
        if completed:
            republished.append(tuple(application.raw_id for application in plan.applications))
        return apply(store, plan, *args, **kwargs)

    def arm_after_first_page(
        self: IndexGenerationStore,
        generation: IndexGeneration,
        *,
        raw_id: str,
        source_sequence: int,
        rewind: bool = False,
    ) -> IndexGeneration:
        result = checkpoint(self, generation, raw_id=raw_id, source_sequence=source_sequence, rewind=rewind)
        if not completed:
            completed.append(result)
            # A stale shared prior-member receipt forces the NEXT selected Raw
            # through its canonical Source phase before Index publication.
            with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as conn, conn:
                assert (
                    conn.execute(
                        "UPDATE raw_authority_parser_census SET parser_fingerprint='prior-parser' WHERE raw_id=?",
                        (raw_id,),
                    ).rowcount
                    == 1
                )
        return result

    def cancel_after_source_commit(self: RawObservationDerivation, *args: Any, **kwargs: Any) -> bool:
        result = source_phase(self, *args, **kwargs)
        if completed:
            with closing(open_readonly_connection(root / "source.db")) as conn:
                assert (
                    conn.execute(
                        "SELECT parser_fingerprint FROM raw_authority_parser_census WHERE raw_id=?",
                        (completed[0].reconstruction_raw_id,),
                    ).fetchone()[0]
                    != "prior-parser"
                )
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM raw_existence_changes WHERE sequence>? AND raw_id<=?",
                        (completed[0].reconstruction_source_sequence, completed[0].reconstruction_raw_id),
                    ).fetchone()[0]
                    > 0
                )
            touched.append(completed[0].reconstruction_raw_id)
            if interrupt:
                raise asyncio.CancelledError()
        return result

    with monkeypatch.context() as control:
        control.setattr(archive, "apply_raw_revision_replay", observe_replay)
        control.setattr(IndexGenerationStore, "checkpoint_reconstruction", arm_after_first_page)
        control.setattr(RawObservationDerivation, "_apply_source_phase", cancel_after_source_commit)
        if interrupt:
            with pytest.raises(asyncio.CancelledError):
                _run_startup()
        else:
            _finish(root, control)
    assert touched == [completed[0].reconstruction_raw_id]
    candidate = completed[0]
    if interrupt:
        assert _inactive(root) == candidate
        offered = _finish(root, monkeypatch)
        assert offered[0] == (candidate.reconstruction_raw_id,)
    else:
        assert republished and candidate.reconstruction_raw_id in republished[0]
    assert ArchiveLocation.resolve(root).active_index_path.resolve() == Path(candidate.index_path)
    with closing(open_readonly_connection(Path(candidate.index_path))) as conn:
        assert tuple(row[0] for row in conn.execute("SELECT session_id FROM sessions")) == sessions
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        assert conn.execute("PRAGMA foreign_key_check").fetchone() is None

    inspection = RawObservationInspection(root, index_db_path=Path(candidate.index_path))
    with inspection.read_current() as conn:
        raws = tuple(row[0] for row in conn.execute("SELECT raw_id FROM raw_sessions"))
        assert all(inspection.inspect_current(conn, raw) == "valid" for raw in raws)


def test_initial_binding_with_regressed_source_journal_replays_the_same_empty_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    make_populated_stale_index(root, tmp_path / "external" / "session.jsonl", independent_raws=1)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    monkeypatch.setattr(startup, "_RAW_PAGE", 1)
    begin = IndexGenerationStore.begin_reconstruction

    def stop_after_initial_binding(self: IndexGenerationStore, *args: Any, **kwargs: Any) -> IndexGeneration:
        begin(self, *args, **kwargs)
        raise asyncio.CancelledError()

    with monkeypatch.context() as control:
        control.setattr(IndexGenerationStore, "begin_reconstruction", stop_after_initial_binding)
        with pytest.raises(asyncio.CancelledError):
            _run_startup()
    candidate = _inactive(root)
    assert candidate.reconstruction_raw_id == ""
    assert candidate.reconstruction_source_sequence > 0
    mutate_fixture_database(root / "source.db", "DELETE FROM raw_existence_changes")
    mutate_fixture_database(root / "source.db", "UPDATE raw_existence_journal_control SET retained_floor=0")
    resets: list[IndexGeneration] = []
    checkpoint = IndexGenerationStore.checkpoint_reconstruction

    def observe_checkpoint(self: IndexGenerationStore, generation: IndexGeneration, **kwargs: Any) -> IndexGeneration:
        result = checkpoint(self, generation, **kwargs)
        if kwargs.get("rewind"):
            resets.append(result)
        return result

    monkeypatch.setattr(IndexGenerationStore, "checkpoint_reconstruction", observe_checkpoint)
    offered = _finish(root, monkeypatch)
    assert offered
    assert len(resets) == 1
    assert resets[0].generation_id == candidate.generation_id
    assert resets[0].reconstruction_raw_id == ""
    assert resets[0].reconstruction_source_sequence == 0
    assert ArchiveLocation.resolve(root).active_index_path.resolve() == Path(candidate.index_path)
    with closing(open_readonly_connection(Path(candidate.index_path))) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        assert conn.execute("PRAGMA foreign_key_check").fetchone() is None
