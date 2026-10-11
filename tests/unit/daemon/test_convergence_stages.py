from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import polylogue.daemon.convergence_stages as stages
import polylogue.logging as plog
import polylogue.operations.claude_workflow_convergence as workflow
import polylogue.operations.sinex_convergence as sinex_composition
from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.daemon.convergence_stages import (
    make_default_convergence_stages,
    make_raw_authority_verdict_cache_stage,
)
from polylogue.storage.derived.session import storage as session_storage
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import IDENTITY_INVALIDATION_DEBT_STAGE
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.compute_owner import owned_compute_adapter


class _SessionIdOnly:
    def __init__(self, session_id: str, marker: str) -> None:
        self.session_id = session_id
        self.marker = marker


def test_session_storage_dedupes_records_by_session_id() -> None:
    records = [
        _SessionIdOnly("codex-session:one", "first"),
        _SessionIdOnly("codex-session:one", "second"),
        _SessionIdOnly("codex-session:two", "third"),
    ]

    deduped = session_storage._dedupe_records_by_session(records)

    assert [(record.session_id, record.marker) for record in deduped] == [
        ("codex-session:one", "second"),
        ("codex-session:two", "third"),
    ]


def test_lineage_prefix_recompose_stage_is_registered(tmp_path: Path) -> None:
    """The stage the writer records debt under is registered, and it can act.

    A ``ConvergenceStage`` with no ``check_sessions``/``execute_sessions`` pair
    is SKIPPED for every session subject, and a skipped stage counts as
    converged -- so registering an inert stage under this name would clear the
    backlog by declaring success instead of draining it.

    Anti-vacuity: drop the registration and the membership assertion goes red;
    register a stage without the session pair and the callable assertions do.
    """
    with owned_compute_adapter() as compute:
        stages_by_name = {
            stage.name: stage
            for stage in make_default_convergence_stages(tmp_path / "index.db", compute_adapter=compute)
        }

    stage = stages_by_name[IDENTITY_INVALIDATION_DEBT_STAGE]
    assert callable(stage.check_sessions)
    assert callable(stage.execute_sessions)
    assert stage.whole_archive is False


def test_default_convergence_stages_leave_derived_domains_to_typed_owners(tmp_path: Path) -> None:
    with owned_compute_adapter() as compute:
        stages_by_name = {
            stage.name: stage
            for stage in make_default_convergence_stages(tmp_path / "index.db", compute_adapter=compute)
        }

    assert "raw_parse_recovery" not in stages_by_name
    assert "fts" not in stages_by_name
    assert "fts_readiness" not in stages_by_name
    assert "embed" not in stages_by_name
    assert "derived" not in stages_by_name


def test_raw_authority_verdict_cache_stage_warms_in_bounded_batches_and_reports_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(stages._DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS + 1):
            raw_id = f"full-{index}"
            written_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f"payload-{index}".encode(),
                source_path="session.jsonl",
                canonical_source_path="session.jsonl",
                acquired_at_ms=1,
                raw_id=raw_id,
            )
            archive.bind_raw_revision(
                written_id,
                RawRevisionEnvelope(f"codex:full-{index}", RawRevisionKind.FULL, f"revision-{raw_id}", 0),
            )
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"append-payload",
            source_path="session.jsonl",
            canonical_source_path="session.jsonl",
            acquired_at_ms=1,
            raw_id="append-only",
        )
        archive.bind_raw_revision(
            append_id,
            RawRevisionEnvelope(
                "codex:append",
                RawRevisionKind.APPEND,
                "revision-append",
                0,
                predecessor_source_revision="revision-base",
                predecessor_raw_id="base",
                baseline_raw_id="base",
                append_start_offset=0,
                append_end_offset=1,
            ),
        )

    stage = make_raw_authority_verdict_cache_stage(tmp_path / "index.db")
    assert stage.check_many is not None
    assert stage.execute_many is not None
    assert stage.false_means_pending is True
    path = tmp_path / "source.jsonl"
    # A zero pass budget stops after one bounded batch, which is what the two
    # executions below observe.
    monkeypatch.setattr(stages, "_DAEMON_RAW_AUTHORITY_CACHE_PASS_SECONDS", 0.0)
    # The daemon executes cache-warming stages under its writer lease.
    with write_lease("test.raw-authority-cache", archive_root=tmp_path), plog.capture() as records:
        assert stage.check(path) is True
        assert stage.execute_many((path,)) is False
        assert stage.check(path) is True
        assert stage.execute_many((path,)) is True
        assert stage.check(path) is False

    with sqlite3.connect(tmp_path / "source.db") as conn:
        cached_cohorts = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT logical_source_key FROM raw_authority_verdicts")
        }
    assert len(cached_cohorts) == stages._DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS + 2
    assert "codex:append" in cached_cohorts
    # The first execute_many left cohorts pending, the second did not. Prose
    # reported both at INFO; the span outcome now separates them.
    terminals = [
        r
        for r in records
        if str(r["event"]).startswith("daemon.stage.execute.")
        and not str(r["event"]).endswith(".start")
        and r.get("stage") == "raw_authority_verdict_cache"
    ]
    assert [r["outcome"] for r in terminals] == ["degraded", "ok"]
    assert terminals[0]["reason"] == "cohorts_still_pending"
    assert int(cast(int, terminals[0]["pending"])) > 0
    assert terminals[1]["pending"] == 0
    assert all(int(cast(int, r["cohorts"])) > 0 for r in terminals)

    import polylogue.storage.raw_authority_verdict_cache as cache_module

    def _fail_projection(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("warm cache was recomputed")

    monkeypatch.setattr(cache_module, "project_raw_authority_verdicts", _fail_projection)
    assert stage.execute_many((path,)) is True


def test_raw_authority_verdict_cache_execution_keeps_warming_batches_within_its_budget(tmp_path: Path) -> None:
    """Anti-vacuity: one batch per execution leaves the second batch pending,
    so the single execution below returns False."""
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(2 * stages._DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS + 1):
            raw_id = f"full-{index}"
            written_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f"payload-{index}".encode(),
                source_path="session.jsonl",
                canonical_source_path="session.jsonl",
                acquired_at_ms=1,
                raw_id=raw_id,
            )
            archive.bind_raw_revision(
                written_id,
                RawRevisionEnvelope(f"codex:full-{index}", RawRevisionKind.FULL, f"revision-{raw_id}", 0),
            )

    stage = make_raw_authority_verdict_cache_stage(tmp_path / "index.db")
    assert stage.execute_many is not None
    path = tmp_path / "source.jsonl"
    with write_lease("test.raw-authority-cache", archive_root=tmp_path):
        assert stage.execute_many((path,)) is True
    assert stage.check(path) is False


def test_sinex_stage_uses_configured_source_tier_not_active_index_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sinex.models import PublicationMode

    configured_root = tmp_path / "configured"
    captured: dict[str, object] = {}

    class CapturePublicationService:
        mode = PublicationMode.PRIMARY

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def blocking_object_ids(self, object_ids: object) -> set[str]:
            del object_ids
            return set()

    monkeypatch.setattr(stages, "load_polylogue_config", lambda: SimpleNamespace(sinex_mode="primary"))
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: configured_root)
    monkeypatch.setattr("polylogue.sinex.service.PublicationService", CapturePublicationService)
    monkeypatch.setattr("polylogue.sinex.transport.resolve_configured_transport", lambda: object())

    with owned_compute_adapter() as compute:
        make_default_convergence_stages(tmp_path / "external-generation" / "index.db", compute_adapter=compute)

    assert captured["source_db_path"] == configured_root / "source.db"


@pytest.mark.parametrize("mode", ["off", "mirror", "primary"])
def test_derivation_barrier_exists_only_in_primary_mode_and_reads_configured_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Derivation owners get the primary barrier exactly when the staged routes honor it.

    Anti-vacuity (polylogue-wtfyv): returning ``None`` in primary mode leaves
    every derivation owner unbarriered.
    """
    import polylogue.sinex.service as sinex_service

    configured_root = tmp_path / "configured"
    read: list[tuple[Path, tuple[str, ...]]] = []

    def blocking(source_db_path: Path, object_ids: Sequence[str]) -> set[str]:
        ids = tuple(object_ids)
        read.append((source_db_path, ids))
        return {"held"} & set(ids)

    monkeypatch.setattr(sinex_composition, "load_polylogue_config", lambda: SimpleNamespace(sinex_mode=mode))
    monkeypatch.setattr(sinex_service, "primary_blocking_object_ids", blocking)

    barrier = sinex_composition.configured_derivation_barrier(configured_root)

    if mode != "primary":
        assert barrier is None
        return
    assert barrier is not None
    assert barrier(("held", "free")) == {"held"}
    assert read == [(configured_root / "source.db", ("held", "free"))]


def test_workflow_publication_keeps_configured_root_after_index_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Following the generation parent would lose source authority and the receipt."""
    from contextlib import closing

    from polylogue.analysis import claude_workflow_materializer as materializer
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_active_archive_root(tmp_path)
    generation = tmp_path / ".index-generations" / "promoted"
    generation.mkdir(parents=True)
    initialize_archive_database(generation / "index.db", ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(str(generation / "index.db"), encoding="utf-8")
    roots: list[Path] = []
    summary = SimpleNamespace(
        run_count=1, call_count=1, attempt_count=1, linked_session_count=1, unresolved_call_count=0, gaps=()
    )

    def materialize(root: Path, **_kwargs: object) -> object:
        roots.append(root)
        return summary

    monkeypatch.setattr(materializer, "materialize_claude_workflow_archive", materialize)
    target = tmp_path / "projects" / "demo" / "session.jsonl"
    stage = workflow.make_claude_workflow_stage(tmp_path / "index.db")
    assert stage.execute(target)
    assert roots == [tmp_path]
    with closing(sqlite3.connect(tmp_path / "ops.db")) as conn:
        assert conn.execute(
            "SELECT status FROM daemon_stage_events WHERE event_id = 'claude_workflow:current'"
        ).fetchone() == ("clean",)
    assert not (generation / "ops.db").exists()


@pytest.mark.parametrize(("binding", "outcome"), [(True, "ok"), (False, "degraded"), (None, "empty")])
def test_fts_stage_production_event_preserves_boolean_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: bool | None,
    outcome: str,
) -> None:
    monkeypatch.setattr("polylogue.operations.fts_derivation.publish_fts_readiness_binding", lambda _root: binding)
    stage = stages.make_fts_readiness_binding_stage(tmp_path / "index.db")
    previous_level = plog.set_level("info")
    try:
        with plog.capture() as records:
            result = stage.execute(tmp_path / "neutral-source")
        terminal = [
            record
            for record in records
            if record["event"] == f"daemon.stage.execute.{outcome}" and record.get("outcome") == outcome
        ]
        assert len(terminal) == 1
        assert terminal[0]["bound"] is (binding is True)
        assert result is (True if binding is None else binding)
        assert not any(
            record["event"] == "log.field_rejected" and record.get("source_event") == "daemon.stage.execute"
            for record in records
        )
    finally:
        plog.set_level(previous_level)
