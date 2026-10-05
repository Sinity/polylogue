"""DaemonConverger.summary() buckets each tracked file exactly once."""

from __future__ import annotations

from pathlib import Path

from polylogue.daemon.convergence import DaemonConverger, FileState, StageState


def test_summary_counts_a_converged_after_failure_file_once() -> None:
    """Anti-vacuity (polylogue-fkn5): counting ``error_count > 0`` as failed puts
    the recovered file in both buckets and drives ``in_progress`` negative."""
    converger = DaemonConverger(stages=())
    recovered = FileState(Path("/synthetic/recovered.jsonl"), stages={"parse": StageState.DONE}, error_count=2)
    failing = FileState(Path("/synthetic/failing.jsonl"), stages={"parse": StageState.FAILED})
    pending = FileState(Path("/synthetic/pending.jsonl"), stages={"parse": StageState.PENDING})
    retrying = FileState(Path("/synthetic/retrying.jsonl"), stages={"parse": StageState.PENDING}, error_count=1)
    converger._file_states = {state.path: state for state in (recovered, failing, pending, retrying)}

    summary = converger.summary()

    # ``retrying`` failed before and is pending now: in progress, not failed.
    assert summary == {"total": 4, "converged": 1, "failed": 1, "in_progress": 2}
    assert summary["converged"] + summary["failed"] + summary["in_progress"] == summary["total"]


class _Facts:
    def __init__(self, *, present: bool, status: str) -> None:
        self.session_present = present
        self.status = status
        self.input_binding: str | None = "b1"
        self.output_binding: str | None = None
        self.profiles = 0


class _SelectedAdapter:
    recipe_version = "r"

    def __init__(self) -> None:
        self.computed: list[str] = []

    def selected_part_facts(self, frame: object, session_id: str) -> _Facts:
        facts = _Facts(present=session_id != "gone", status="missing")
        if session_id == "gone":
            facts.profiles = 1  # an orphaned profile the excess target retires
        return facts

    def selected_frame_is_current(self, frame: object) -> bool:
        return True

    def quiet(self, frame: object, session_id: str) -> bool:
        return False

    def compute(self, frame: object, session_id: str) -> object:
        self.computed.append(session_id)
        raise RuntimeError("stop after the barrier decision")

    def publish(self, frame: object, replacement: object) -> bool:
        raise AssertionError("not reached")


def test_selected_maintenance_waits_for_the_primary_barrier() -> None:
    """Exact-target maintenance honors the barrier its recurring pass honors.

    Anti-vacuity (polylogue-wtfyv review): without the barrier argument the
    held required target is computed; an excess target is retirement and
    must never wait.
    """
    from polylogue.daemon.convergence import SelectedSessionTarget, _converge_selected_session_parts_sync
    from polylogue.daemon.derivation import DerivationFrame

    adapter = _SelectedAdapter()
    frame = DerivationFrame(archive_root="/archive", source_revision="g", recipe_versions={"session_profile": "r"})
    outcomes = _converge_selected_session_parts_sync(
        frame,
        targets=(SelectedSessionTarget("held", "required"), SelectedSessionTarget("gone", "excess")),
        expected_generation="g",
        expected_recipe="r",
        adapter=adapter,  # type: ignore[arg-type]
        adapter_recipe="r",
        stop_requested=lambda: None,
        admission=lambda domain, publish: publish(),  # type: ignore[arg-type]
        barrier=lambda sessions: set(sessions),
    )

    held = outcomes[0]
    assert (held.session_id, held.state, held.reason) == ("held", "pending", "awaits primary publication")
    assert adapter.computed == ["gone"]


def test_selected_maintenance_rechecks_the_barrier_at_publication() -> None:
    """A revision staged during compute blocks publication of the selected target.

    Anti-vacuity: a pre-compute-only check publishes the stale derivation.
    """
    from polylogue.daemon.convergence import SelectedSessionTarget, _converge_selected_session_parts_sync
    from polylogue.daemon.derivation import DerivationFrame

    staged: set[str] = set()
    published: list[object] = []

    class _Adapter(_SelectedAdapter):
        def compute(self, frame: object, session_id: str) -> object:
            staged.add(session_id)
            return type("R", (), {"input_binding": "b1", "close": lambda _self: None})()

        def publish(self, frame: object, replacement: object) -> bool:
            published.append(replacement)
            return True

    frame = DerivationFrame(archive_root="/archive", source_revision="g", recipe_versions={"session_profile": "r"})
    outcomes = _converge_selected_session_parts_sync(
        frame,
        targets=(SelectedSessionTarget("s", "required"),),
        expected_generation="g",
        expected_recipe="r",
        adapter=_Adapter(),  # type: ignore[arg-type]
        adapter_recipe="r",
        stop_requested=lambda: None,
        admission=lambda domain, publish: publish(),  # type: ignore[arg-type]
        barrier=lambda sessions: staged & set(sessions),
    )

    assert published == []
    assert (outcomes[0].state, outcomes[0].reason) == ("pending", "awaits primary publication")
