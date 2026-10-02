"""Every interrupted, refused or moved verification still ends in a true receipt.

The tests replace host observations (Git, processes, signals), never the
receipt writers or the verdict decisions they exercise.
"""

from __future__ import annotations

import io
import json
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

from devtools import pytest_slot, run_tests, verify, verify_runs, why
from devtools.checkout_identity import CheckoutIdentity
from devtools.testmon_provision import TestmonGraphState, TestmonGraphStatus
from devtools.verification_admission import AffectedAdmission


@pytest.fixture
def receipt_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the actual receipt writers; replace only host-dependent observations."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(verify_runs.VERIFY_HISTORY_PATH_ENV, str(tmp_path / "history.jsonl"))
    monkeypatch.setenv(verify_runs.VERIFY_EVIDENCE_PATH_ENV, str(tmp_path / "evidence.jsonl"))
    identity = CheckoutIdentity(tmp_path, "fix/receipt-regression", "a" * 40, "master")
    monkeypatch.setattr(verify_runs, "checkout_identity", lambda _root: identity)
    monkeypatch.setattr(verify_runs, "git_dirty", lambda _root: False)
    graph = TestmonGraphState(TestmonGraphStatus.USABLE, "fixture", recorded_tests=1, source_dependencies=1)
    for module in (run_tests, verify):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "checkout_identity", lambda _root: identity)
        monkeypatch.setattr(module, "git_head", lambda _root: identity.head)
        monkeypatch.setattr(module, "git_worktree_content_sha256", lambda _root: "fixture-content")
        monkeypatch.setattr(module, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(module, "inspect_testmon_graph", lambda _root: graph)
        monkeypatch.setattr(module, "prune_successful_verify_runs", lambda **_kwargs: None)
    monkeypatch.setattr(run_tests, "basetemp_root", lambda _env, **_kwargs: tmp_path / "temporary-tests")
    monkeypatch.setattr(run_tests, "_prepare_nodatacow_parent", lambda _path: None)
    monkeypatch.setattr(verify, "refuse_verify_tier", lambda *_args: None)
    monkeypatch.setattr(verify, "_declared_agentctl_operation", lambda _argv: None)
    monkeypatch.setattr(verify, "sync_testmon_graph", lambda _root: False)
    monkeypatch.setattr(verify, "validate_authority_matrix", lambda: None)
    monkeypatch.setattr(verify, "reconcile_and_record_abandoned_verify_runs", lambda **_kwargs: [])
    monkeypatch.setattr(verify, "_git_changed_paths", lambda _root: frozenset({"polylogue/example.py"}))
    from polylogue.context import failure_seed

    monkeypatch.setattr(failure_seed, "write_failure_seed", lambda **_kwargs: None)
    return tmp_path


def _only_receipt(root: Path) -> dict[str, Any]:
    paths = list((root / verify_runs.VERIFY_RUNS_DIR).glob("*/run.json"))
    assert len(paths) == 1
    receipt: dict[str, Any] = json.loads(paths[0].read_text(encoding="utf-8"))
    return receipt


def test_focused_sigterm_finishes_receipt_and_publications(
    receipt_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a SIGTERM handler the default disposition escapes before finish and history."""
    selected = receipt_workspace / "tests" / "test_target.py"
    selected.parent.mkdir()
    selected.write_text("def test_target(): pass\n", encoding="utf-8")
    previous = signal.getsignal(signal.SIGTERM)

    def default_termination(_number: int) -> None:
        # Make the old default-signal path fail safely rather than killing pytest.
        raise SystemExit(143)

    def interrupt(_label: str, _cmd: list[str], **_kwargs: Any) -> Any:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        pytest.fail("the installed handler returned instead of interrupting")

    monkeypatch.setattr(signal, "raise_signal", default_termination)
    monkeypatch.setattr(run_tests, "_run", interrupt)
    assert run_tests.main([str(selected)]) == 143
    receipt = _only_receipt(receipt_workspace)
    assert receipt["status"] == "failed"
    assert receipt["steps"][0]["termination_reason"] == "sigterm"
    assert receipt["steps"][0]["exit"] == 143
    assert list(verify_runs._iter_history_pinned(receipt_workspace / "history.jsonl"))
    assert verify_runs.read_verification_evidence(receipt_workspace / "evidence.jsonl")
    assert signal.getsignal(signal.SIGTERM) == previous


def test_focused_missing_directory_has_a_path_specific_usage_diagnostic(
    receipt_workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """pytest reports a missing directory as usage error 4; the path diagnostic must cover it."""
    monkeypatch.setattr(run_tests, "_run", lambda *_args, **_kwargs: (4, 0.1, {"diagnosis": "pytest_failed"}))
    assert run_tests.main(["tests/missing-directory"]) == 4
    output = capsys.readouterr().err
    assert "these selection paths do not exist: tests/missing-directory" in output
    assert "every path exists" not in output


def test_focused_verdict_receipt_is_absolute_from_a_nested_invocation(
    receipt_workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A checkout-relative receipt path does not resolve from the caller's nested directory."""
    nested = receipt_workspace / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.setattr(run_tests, "_run", lambda *_args, **_kwargs: (4, 0.1, {"diagnosis": "pytest_failed"}))
    assert run_tests.main(["tests/missing-directory"]) == 4
    output = capsys.readouterr().err
    advertised = Path(output.rsplit("receipt=", 1)[1].split(" checkout=", 1)[0])
    assert advertised.is_absolute()
    assert advertised.is_file()
    assert json.loads(advertised.read_text(encoding="utf-8"))["run_id"] == _only_receipt(receipt_workspace)["run_id"]


@pytest.mark.parametrize("ending", ["a" * 40, "b" * 40, None])
def test_canonical_receipt_never_attributes_an_untested_ending_head(
    receipt_workspace: Path, ending: str | None
) -> None:
    """Preferring final_git_head attributes a moved head (b) that pytest never executed."""
    run = verify_runs.VerifyRun(tier="focused-test", argv=[], git_head="a" * 40, root=receipt_workspace)
    run.record_execution_worktree(
        {
            "git_head": "a" * 40,
            "git_branch": "fix/receipt-regression",
            "git_dirty": False,
            "git_worktree_content_sha256": "fixture-content",
            "capture_source": "slot_start",
        }
    )
    payload = run.finish(exit_code=1, duration_s=0.1, final_git_head=ending)
    evidence = receipt_workspace / "evidence.jsonl"
    verify_runs.append_verification_evidence(payload, path=evidence)
    row = verify_runs.read_verification_evidence(evidence)[0]
    assert row["source_revision"] == ("a" * 40 if ending == "a" * 40 else None)


@pytest.mark.parametrize("phase", ["selection", "assembly"])
def test_verifier_finalizes_interruptions_outside_the_step_loop(
    receipt_workspace: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """A signal during selection or verdict assembly must finish the run, not leave it running.

    Before the fix both phases sat outside the handler, so the receipt stayed
    ``running`` with no history row.
    """
    monkeypatch.setattr(
        verify, "_affected_admission", lambda **_kwargs: AffectedAdmission("admitted", 1, 0.1, "fixture")
    )
    monkeypatch.setattr(verify, "build_verify_steps", lambda **_kwargs: [])
    interrupted = False

    if phase == "selection":
        original_selection = verify_runs.VerifyRun.record_selection

        def interrupt_selection(self: verify_runs.VerifyRun, **kwargs: Any) -> Any:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise verify.VerificationInterrupted(signal.SIGTERM)
            return original_selection(self, **kwargs)

        monkeypatch.setattr(verify_runs.VerifyRun, "record_selection", interrupt_selection)
    else:
        original_aggregate = verify._aggregate_pytest_results

        def interrupt_aggregate(*args: Any, **kwargs: Any) -> Any:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise verify.VerificationInterrupted(signal.SIGTERM)
            return original_aggregate(*args, **kwargs)

        monkeypatch.setattr(verify, "_aggregate_pytest_results", interrupt_aggregate)
    assert verify.main([]) == 143
    assert interrupted
    receipt = _only_receipt(receipt_workspace)
    assert receipt["status"] == "failed"
    assert receipt["diagnosis"] == "verification_interrupted"
    assert receipt["pytest_aggregate"]["termination_reason"] == "sigterm"
    rows = list(verify_runs._iter_history_pinned(receipt_workspace / "history.jsonl"))
    assert [row["run_id"] for row in rows] == [receipt["run_id"]]


@pytest.mark.parametrize("gate_exit", [0, 1])
def test_refused_affected_admission_still_executes_static_gates(
    receipt_workspace: Path, monkeypatch: pytest.MonkeyPatch, gate_exit: int
) -> None:
    """A refused pytest plan still runs the static gates; the early return ran none."""
    monkeypatch.setattr(
        verify, "_affected_admission", lambda **_kwargs: AffectedAdmission("unknown", None, None, "fixture")
    )
    observed: list[str] = []

    def execute(label: str, command: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        observed.append(label)
        assert not label.startswith("pytest")
        return gate_exit, 0.1, {"diagnosis": "gate_passed" if gate_exit == 0 else "gate_semantic_violation"}

    # Keep step construction and _run_steps; replace the gate subprocess boundary.
    monkeypatch.setattr(verify, "_run", execute)
    assert verify.main([]) == (gate_exit or 2)
    assert observed
    assert not any(name.startswith("pytest") for name in observed)
    receipt = _only_receipt(receipt_workspace)
    assert receipt["pytest_aggregate"]["complete_corpus_covered"] is False
    assert receipt["exit_code"] == (gate_exit or 2)
    # The retained exit code is the first failure's, and so is the diagnosis.
    assert receipt["diagnosis"] == ("gate_semantic_violation" if gate_exit else "affected_admission_refused")
    assert receipt["pytest_aggregate"]["admission"]["status"] == "unknown"


def test_queued_interruption_keeps_start_time_worktree_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interruption receipt must keep the provenance _run_launch captured at start."""
    provenance = {"git_head": "a" * 40, "git_branch": "fix/queued", "git_worktree_content_sha256": "start-digest"}
    monkeypatch.setattr(pytest_slot, "_focused_worktree_provenance", lambda *_args: provenance)
    monkeypatch.setattr(pytest_slot, "resize_worker_argument", lambda argv, **_sizing: (argv, None))

    class Process:
        pid = 12345

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
            raise AssertionError("the slot signal handler returned")

    class Sampler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def persist(self) -> None:
            return None

        def stop(self) -> None:
            return None

    def exit_process(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr("devtools.pytest_slot.subprocess.Popen", Process)
    monkeypatch.setattr(pytest_slot, "ProcessGroupMemorySampler", Sampler)
    monkeypatch.setattr("devtools.pytest_slot.os._exit", exit_process)
    launch, log = tmp_path / "launch.json", tmp_path / "slot.log"
    pytest_slot._write_launch(launch, argv=[sys.executable, "-m", "pytest"], cwd=str(tmp_path), env={}, log_path=log)
    with pytest.raises(SystemExit) as ended:
        pytest_slot._run_launch(launch)
    assert ended.value.code == 143
    receipt = json.loads(pytest_slot._slot_result_path(log).read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["worktree_provenance"] == provenance


def test_successful_abandoned_adoption_retries_an_interrupted_publication(
    receipt_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adopted exit-zero receipt must be republished after an interrupted first attempt."""
    run = verify_runs.VerifyRun(tier="quick", argv=[], git_head="a" * 40, root=receipt_workspace)
    run._payload["agentctl_job_id"] = "job-adopted"
    run.write()
    state = receipt_workspace / "agentctl"
    state.mkdir()
    (state / "job-adopted.outcome").write_text('{"exit_code":0,"outcome":"completed"}', encoding="utf-8")
    monkeypatch.setattr(verify_runs, "_process_owns_receipt", lambda *_args, **_kwargs: False)
    history, evidence = receipt_workspace / "history.jsonl", receipt_workspace / "evidence.jsonl"
    original_append = verify_runs.append_verify_history

    def interrupted_publication(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected publication interruption")

    monkeypatch.setattr(verify_runs, "append_verify_history", interrupted_publication)
    arguments: dict[str, Any] = {
        "runs_root": receipt_workspace / verify_runs.VERIFY_RUNS_DIR,
        "state_root": state,
        "history_path": history,
        "evidence_path": evidence,
    }
    first = verify_runs.reconcile_and_record_abandoned_verify_runs(**arguments)
    assert first[0]["status"] == "success"
    assert not list(verify_runs._iter_history_pinned(history))
    monkeypatch.setattr(verify_runs, "append_verify_history", original_append)
    second = verify_runs.reconcile_and_record_abandoned_verify_runs(**arguments)
    assert second and list(verify_runs._iter_history_pinned(history))[0]["run_id"] == run.run_id
    assert len(verify_runs.read_verification_evidence(evidence)) == 1
    verify_runs.reconcile_and_record_abandoned_verify_runs(**arguments)
    assert len(list(verify_runs._iter_history_pinned(history))) == 1


def _full_run(root: Path, run_id: str, finished: str, *, complete: bool = True) -> Path:
    run_dir = root / verify_runs.VERIFY_RUNS_DIR / run_id
    steps = []
    for index, name in enumerate(("parallel", "serial", "storage")):
        step_id = f"0{index + 1}-pytest-{name}"
        report = run_dir / "steps" / step_id / verify_runs.PYTEST_CANONICAL_REPORT_NAME
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "nodeid": f"tests/{run_id}_{name}.py::test_case",
                            "call": {"duration": index + 1},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        steps.append({"name": f"pytest {name}", "step_id": step_id})
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "tier": "all",
                "status": "success" if complete else "running",
                "finished_at": finished,
                "pytest_aggregate": {"complete_corpus_covered": complete},
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_outliers_read_all_lanes_of_one_completed_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every lane comes from one completed receipt; shared shard names and partial runs are ignored."""
    _full_run(tmp_path, "older", "2026-01-01T00:00:00Z")
    _full_run(tmp_path, "selected", "2026-01-02T00:00:00Z")
    _full_run(tmp_path, "partial", "2026-01-03T00:00:00Z", complete=False)
    (tmp_path / run_tests.PYTEST_REPORT_DIR / "last-pytest-poison.json").write_text(
        '{"tests":[{"nodeid":"poison","call":{"duration":999}}]}',
        encoding="utf-8",
    )
    assert run_tests.print_outliers(10, root=tmp_path) == 0
    output = capsys.readouterr().out
    assert "Full-run receipts: 3; tests: 3; serial time: 6.00s" in output
    for lane in ("parallel", "serial", "storage"):
        assert f"selected_{lane}.py" in output
    assert "older" not in output and "partial" not in output and "poison" not in output


def test_outliers_refuse_a_missing_lane_instead_of_falling_back_to_older_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lost lane report refuses instead of summarizing an incomplete run."""
    _full_run(tmp_path, "older", "2026-01-01T00:00:00Z")
    chosen = _full_run(tmp_path, "selected", "2026-01-02T00:00:00Z")
    (chosen / "steps/03-pytest-storage/pytest-report.json").unlink()
    assert run_tests.print_outliers(root=tmp_path) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "incomplete full-run evidence" in output.err


def _passing_step_evidence(step_dir: Path) -> None:
    for name, payload in {
        "pytest-report.json": {"tests": [{"nodeid": "test_target", "outcome": "passed"}]},
        "selection.json": {"selected_count": 1},
        "summary.json": {"exitstatus": 0},
    }.items():
        (step_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (step_dir / "events.jsonl").write_text(
        json.dumps({"event": "collection_finished", "selected_count": 1}) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("artifact", "fault", "error_type"),
    [
        ("pytest-report.json", "directory", "IsADirectoryError"),
        ("selection.json", "directory", "IsADirectoryError"),
        ("summary.json", "directory", "IsADirectoryError"),
        ("events.jsonl", "directory", "IsADirectoryError"),
        ("events", "not_directory", "NotADirectoryError"),
        ("events.jsonl", "utf8", "UnicodeDecodeError"),
        ("events.jsonl", "json", "JSONDecodeError"),
        ("events/gw0.jsonl", "utf8", "UnicodeDecodeError"),
        ("events/gw0.jsonl", "json", "JSONDecodeError"),
        ("pytest-report.json", "json", "JSONDecodeError"),
        ("selection.json", "json", "JSONDecodeError"),
        ("summary.json", "json", "JSONDecodeError"),
        ("summary.json", "object", "ValueError"),
        ("pytest-report.json", "tests_null", "TypeError"),
    ],
)
def test_existing_unreadable_pytest_evidence_fails_the_actual_receipt_owner(
    receipt_workspace: Path, artifact: str, fault: str, error_type: str
) -> None:
    """Removing strict terminal reads or reinstating suppression turns real faults green."""
    run = verify_runs.VerifyRun(tier="focused-test", argv=[], git_head="a" * 40, root=receipt_workspace)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])
    _passing_step_evidence(artifacts.step_dir)
    path = artifacts.step_dir / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    if fault == "directory":
        path.unlink()
        path.mkdir()
    else:
        prefix = b'{"event": "collection_finished"}\n' if artifact.endswith("jsonl") else b""
        path.write_bytes(
            prefix
            + (
                b"\xff"
                if fault == "utf8"
                else b"[]"
                if fault == "object"
                else b'{"tests": null}'
                if fault == "tests_null"
                else b"invalid-json"
            )
        )
    result = run.finish_step(step_id=artifacts.step_id, result={"exit": 0})
    assert result is not None
    assert result["status"] == "failed"
    assert result["exit"] == 1 and result["process_exit"] == 0
    assert result["diagnosis"] == "pytest_evidence_unavailable"
    assert result["evidence_error"]["phase"] == "aggregation"
    assert result["evidence_error"]["type"] == error_type
    assert result["evidence_error"]["message"]
    assert "statistics" not in result
    persisted = _only_receipt(receipt_workspace)["steps"][0]
    assert persisted == result
    assert path.exists(), "the unreadable input remains diagnostic evidence"


@pytest.mark.parametrize("phase", ["statistics_publication", "statistics_mirror"])
def test_statistics_publication_fault_records_failure_before_success(receipt_workspace: Path, phase: str) -> None:
    run = verify_runs.VerifyRun(tier="focused-test", argv=[], git_head="a" * 40, root=receipt_workspace)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])
    _passing_step_evidence(artifacts.step_dir)
    destination = (
        artifacts.statistics_path
        if phase == "statistics_publication"
        else receipt_workspace / verify_runs.CURRENT_STATISTICS_PATH
    )
    destination.mkdir()
    result = run.finish_step(step_id=artifacts.step_id, result={"exit": 0})
    assert result is not None
    assert result["status"] == "failed" and result["process_exit"] == 0 and result["exit"] == 1
    assert result["evidence_error"]["phase"] == phase
    assert result["evidence_error"]["type"] == "IsADirectoryError"
    assert result["statistics"]["outcomes"] == {"passed": 1}
    assert result["statistics"]["ordinary_eligible"] is False
    assert result["statistics"]["ok"] is False
    assert _only_receipt(receipt_workspace)["steps"][0] == result


@pytest.mark.parametrize(
    ("process_exit", "diagnosis"),
    [
        (143, "pytest_interrupted"),
        (130, "verification_interrupted"),
        (125, "focused_test_runner_exception"),
        (125, "pytest_slot_unavailable"),
        (137, "oom_killed"),
        (2, "pytest_failed"),
    ],
)
def test_evidence_fault_preserves_execution_failure_and_explicit_terminal_reason(
    receipt_workspace: Path, process_exit: int, diagnosis: str
) -> None:
    run = verify_runs.VerifyRun(tier="focused-test", argv=[], git_head="a" * 40, root=receipt_workspace)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])
    artifacts.events_merged_path.write_bytes(b"\xff")
    result = run.finish_step(step_id=artifacts.step_id, result={"exit": process_exit, "diagnosis": diagnosis})
    assert result is not None
    assert result["process_exit"] == process_exit and result["exit"] == process_exit
    assert result["diagnosis"] == ("pytest_evidence_unavailable" if diagnosis == "pytest_failed" else diagnosis)
    assert result["evidence_error"]["type"] == "UnicodeDecodeError"
    assert result["status"] == "failed"


def test_focused_caller_publishes_failed_verdict_from_existing_unreadable_ledger(
    receipt_workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A raw zero reaches the real owner; callers must adopt its failed evidence verdict."""
    selected = receipt_workspace / "test_target.py"
    selected.write_text("def test_target(): pass\n", encoding="utf-8")

    def zero_with_unreadable_ledger(_label: str, _cmd: list[str], **kwargs: Any) -> Any:
        step_dir = Path(kwargs["env"]["POLYLOGUE_PYTEST_SUMMARY_PATH"]).parent
        _passing_step_evidence(step_dir)
        (step_dir / "events.jsonl").write_bytes(b'{"event": "collection_finished"}\n\xff')
        return 0, 0.1, {"diagnosis": "pytest_passed"}

    monkeypatch.setattr(run_tests, "_run", zero_with_unreadable_ledger)
    assert run_tests.main([str(selected)]) == 1
    payload = _only_receipt(receipt_workspace)
    assert payload["exit_code"] == 1 and payload["status"] == "failed"
    assert payload["diagnosis"] == "pytest_evidence_unavailable"
    assert payload["pytest_aggregate"]["terminal_green"] is False
    step = payload["steps"][0]
    assert step["process_exit"] == 0 and step["evidence_error"]["type"] == "UnicodeDecodeError"
    history = list(verify_runs._iter_history_pinned(receipt_workspace / "history.jsonl"))
    assert history[-1]["semantic_receipt"]["status"] == "failed"
    historical_step = history[-1]["semantic_receipt"]["steps"][0]
    assert historical_step["process_exit"] == 0
    assert historical_step["evidence_error"] == {"phase": "aggregation", "type": "UnicodeDecodeError"}
    assert "message" not in historical_step["evidence_error"]
    canonical = verify_runs.read_verification_evidence(receipt_workspace / "evidence.jsonl")[-1]
    assert canonical["status"] == "failed"
    assert canonical["steps"][0]["process_exit"] == 0
    assert canonical["steps"][0]["evidence_error"] == {"phase": "aggregation", "type": "UnicodeDecodeError"}
    output = capsys.readouterr().err
    assert "FAILED exit=1 diagnosis=pytest_evidence_unavailable" in output
    stream = io.StringIO()
    why._render(payload, stream)
    assert "UnicodeDecodeError" in stream.getvalue()
    assert "aggregation" in stream.getvalue()


def test_clean_evidence_still_publishes_success(receipt_workspace: Path) -> None:
    run = verify_runs.VerifyRun(tier="focused-test", argv=[], git_head="a" * 40, root=receipt_workspace)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])
    _passing_step_evidence(artifacts.step_dir)
    result = run.finish_step(step_id=artifacts.step_id, result={"exit": 0})
    assert result is not None
    assert result["exit"] == 0 and result["process_exit"] == 0 and result["status"] == "success"
    assert result["statistics"]["ordinary_eligible"] is True
    assert "evidence_error" not in result
    assert json.loads(artifacts.statistics_path.read_text())["ok"] is True
    assert json.loads((receipt_workspace / verify_runs.CURRENT_STATISTICS_PATH).read_text())["ok"] is True
