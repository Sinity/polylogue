"""Per-subject primary barriers and mirror continuation in the real converger."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage, DaemonConverger, StageState


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
