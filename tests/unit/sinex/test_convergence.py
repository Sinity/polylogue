"""Per-subject primary barriers and mirror continuation in the real converger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from polylogue.daemon.convergence import ConvergenceStage, DaemonConverger, StageState
from polylogue.daemon.convergence_stages import make_sinex_publication_stage
from polylogue.sinex.models import PublicationMode, ReceiptState
from polylogue.sinex.service import PublicationService
from polylogue.sinex.transport import LocalReferenceTransport
from tests.unit.sinex._fixtures import publication_payload


def _projection_stage(calls: list[Path]) -> ConvergenceStage:
    def execute(path: Path) -> bool:
        calls.append(path)
        return True

    return ConvergenceStage(
        name="projection",
        description="test projection",
        check=lambda _path: True,
        execute=execute,
    )


def _projection_stage_with_sessions(
    path_calls: list[Path],
    session_calls: list[str],
) -> ConvergenceStage:
    stage = _projection_stage(path_calls)

    def execute_sessions(session_ids: Sequence[str]) -> bool:
        session_calls.extend(session_ids)
        return True

    return ConvergenceStage(
        name=stage.name,
        description=stage.description,
        check=stage.check,
        execute=stage.execute,
        check_sessions=lambda session_ids: set(session_ids),
        execute_sessions=execute_sessions,
    )


def test_primary_barrier_blocks_only_affected_path_and_mirror_does_not() -> None:
    blocked = Path("blocked.json")
    free = Path("free.json")
    projected: list[Path] = []
    primary = ConvergenceStage(
        name="sinex_publication",
        description="primary publication",
        check=lambda _path: False,
        execute=lambda _path: True,
        blocks_following_stages=True,
        barrier_check=lambda path: path == blocked,
    )
    states, _ = DaemonConverger((primary, _projection_stage(projected))).converge_batch((blocked, free))
    assert states[blocked].stages["projection"] is StageState.PENDING
    assert states[free].stages["projection"] is StageState.DONE
    assert projected == [free]

    projected.clear()
    mirror = ConvergenceStage(
        name="sinex_publication",
        description="mirror publication",
        check=lambda _path: False,
        execute=lambda _path: True,
        blocks_following_stages=False,
        barrier_check=lambda _path: True,
    )
    states, _ = DaemonConverger((mirror, _projection_stage(projected))).converge_batch((blocked, free))
    assert states[blocked].stages["projection"] is StageState.DONE
    assert states[free].stages["projection"] is StageState.DONE
    assert set(projected) == {blocked, free}


def test_stage_status_masks_probe_failure() -> None:
    good = ConvergenceStage("good", "", lambda _p: False, lambda _p: True, status=lambda: {"lag": 2})

    def bad_status() -> Mapping[str, object]:
        raise ZeroDivisionError

    bad = ConvergenceStage("bad", "", lambda _p: False, lambda _p: True, status=bad_status)
    assert DaemonConverger((good, bad)).stage_status() == {
        "good": {"lag": 2},
        "bad": {"state": "unavailable"},
    }


def test_real_sinex_stage_blocks_affected_file_and_session_scopes(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_path = Path("blocked.json")
    free_path = Path("free.json")
    blocked_id = "claude-code-session:blocked"
    free_id = "claude-code-session:free"
    service = PublicationService(
        workspace_env["archive_root"] / "source.db",
        PublicationMode.PRIMARY,
        LocalReferenceTransport(fault_fn=lambda _request, _attempt: ReceiptState.RAW_ACCEPTED),
    )
    service.stage_payload(publication_payload(object_id=blocked_id))
    monkeypatch.setattr(
        "polylogue.daemon.convergence_stages._sinex_session_ids_for_paths",
        lambda _db, paths: {path: [blocked_id] if path == blocked_path else [free_id] for path in paths},
    )
    projected: list[Path] = []
    projected_sessions: list[str] = []
    stage = make_sinex_publication_stage(workspace_env["archive_root"] / "index.db", service)
    converger = DaemonConverger((stage, _projection_stage_with_sessions(projected, projected_sessions)))

    blocked_state = converger.converge_file(blocked_path)
    free_state = converger.converge_file(free_path)
    session_states, _timings = converger.converge_sessions((blocked_id, free_id))

    assert blocked_state.stages["sinex_publication"] is StageState.PENDING
    assert blocked_state.stages["projection"] is StageState.PENDING
    assert free_state.stages["projection"] is StageState.DONE
    assert session_states[blocked_id].stages["projection"] is StageState.PENDING
    assert session_states[free_id].stages["projection"] is StageState.DONE
    assert projected == [free_path]
    assert projected_sessions == [free_id]


def test_real_sinex_stage_treats_barrier_probe_failure_as_failure(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = "claude-code-session:barrier-error"
    service = PublicationService(
        workspace_env["archive_root"] / "source.db",
        PublicationMode.PRIMARY,
        LocalReferenceTransport(),
    )
    monkeypatch.setattr(service, "blocking_object_ids", lambda _ids: (_ for _ in ()).throw(RuntimeError("boom")))
    stage = make_sinex_publication_stage(workspace_env["archive_root"] / "index.db", service)
    projected: list[Path] = []
    states, _timings = DaemonConverger((stage, _projection_stage(projected))).converge_sessions((subject,))

    assert states[subject].stages["sinex_publication"] is StageState.FAILED
    assert states[subject].stages["projection"] is StageState.PENDING
    assert states[subject].last_error == "stage sinex_publication barrier check failed"
