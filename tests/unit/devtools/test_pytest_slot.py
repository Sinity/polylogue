"""The host's single pytest slot is the pytest pool, entered through agentctl.

Anti-vacuity: deleting the submitting branch in ``devtools.pytest_slot.run_pytest``
makes ``test_outside_the_pool_the_run_is_submitted`` red — the command executes
here and the marker file appears. Widening ``INHERITED_ENVIRONMENT_KEYS`` makes
``test_the_client_environment_carries_only_the_allowed_keys`` red. Dropping
either half of the temporary-directory containment (the ``--basetemp``
argument or the exported TMPDIR) makes
``test_a_submitted_run_contains_its_temporary_trees`` red. Treating a job id
as ownership makes ``test_a_job_id_is_never_slot_ownership`` red. Publishing the
result document only on the timeout path makes
``test_a_queued_run_publishes_its_result_document`` red, and dropping the memory
sampler makes ``test_a_held_run_records_what_it_took`` red -- a run that is
killed leaves the receipt as the only account of what it took.

Every submitting test here resolves ``agentctl`` from a fake that is the whole
PATH, so a green run says nothing about what the workstation has deployed; the
cgroup a slot decision depends on is stubbed for the same reason.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import tomllib

from devtools import cloud_sentinels, pytest_slot, worker_memory
from devtools.pytest_slot import (
    BASETEMP_ROOT_ENV,
    PytestSlotUnavailableError,
    basetemp_root,
    holds_pytest_slot,
    run_pytest,
)
from tests.unit.devtools.cgroups import (
    AGENT_CGROUPS,
    OUTSIDE_CGROUP,
    PYTEST_CGROUPS,
    stub_cgroup,
)

#: An ``agentctl`` whose ``job start`` snapshots the launch file it was handed
#: and whose ``job get`` reports one terminal job. It records every call with
#: the environment it ran in; the record path is derived from the script's own
#: location because a submitting client's environment is scrubbed.
FAKE_AGENTCTL = """import json, os, sys, shutil

with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ)}}) + "\\n")

words = [word for word in sys.argv[1:] if word != "--json"]
verb = " ".join(words[:2])
if verb == "job start":
    shutil.copyfile(sys.argv[-1], {launch_snapshot!r})
    launch = json.loads(open(sys.argv[-1], encoding="utf-8").read())
    with open(launch["log_path"], "wb") as captured:
        captured.write(b"captured output")
    if {receipt!r} is not None:
        # The slot runner publishes its result document next to the launch file.
        with open(sys.argv[-1][: -len(".json")] + ".result.json", "w", encoding="utf-8") as handle:
            handle.write(json.dumps({receipt!r}))
    print(json.dumps({{"job_id": {job_id}, "phase": "queued", "terminal": False}}))
elif verb == "job get":
    print(json.dumps({{"job_id": {job_id}, "phase": {phase!r}, "terminal": True, "exit_code": {exit_code}}}))
sys.exit(0)
"""


def _install_executable(directory: Path, name: str, source: str) -> Path:
    """Install ``source`` as an executable that needs nothing else on PATH.

    The shebang names the interpreter absolutely: these fakes are the entire
    PATH of the run under test, so ``/usr/bin/env python3`` would not resolve.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _install_fake_agentctl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_id: int = 7,
    phase: str = "succeeded",
    exit_code: int | None = 0,
    receipt: dict[str, Any] | None = None,
) -> Path:
    directory = tmp_path / "fakebin"
    script = _install_executable(
        directory,
        "agentctl",
        FAKE_AGENTCTL.format(
            job_id=job_id,
            phase=phase,
            exit_code=exit_code,
            receipt=receipt,
            launch_snapshot=str(tmp_path / "submitted-launch.json"),
        ),
    )
    # The fake is the whole PATH: submitting must resolve its tool from what
    # the test installed, never from whatever the workstation has deployed.
    monkeypatch.setenv("PATH", str(directory))
    monkeypatch.setattr(pytest_slot, "POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(pytest_slot, "OBSERVATION_BACKOFF_MAX_S", 0.0)
    return Path(str(script) + ".calls.jsonl")


def _install_scripted_agentctl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observations: list[dict[str, Any] | str],
    *,
    receipt: dict[str, Any] | None = None,
    cancel_state: str = "removed",
) -> Path:
    source = (
        """import json, os, sys
with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")
words = [word for word in sys.argv[1:] if word != "--json"]
verb = " ".join(words[:2])
if verb == "job start":
    if {receipt!r} is not None:
        with open(sys.argv[-1][:-len(".json")] + ".result.json", "w", encoding="utf-8") as handle:
            handle.write(json.dumps({receipt!r}))
    print(json.dumps({"job_id": 23, "reference": "ref-23", "phase": "queued", "terminal": False}))
elif verb == "job get":
    index_path = sys.argv[0] + ".get-count"
    try:
        index = int(open(index_path, encoding="utf-8").read())
    except OSError:
        index = 0
    open(index_path, "w", encoding="utf-8").write(str(index + 1))
    response = {observations!r}[min(index, len({observations!r}) - 1)]
    if response == "error":
        sys.stderr.write("temporary queue read failure\\n")
        sys.exit(1)
    print(json.dumps(response))
elif verb == "job cancel":
    print(json.dumps({"job_id": 23, "state": __CANCEL_STATE__}))
sys.exit(0)
""".replace("{observations!r}", repr(observations))
        .replace("{receipt!r}", repr(receipt))
        .replace("__CANCEL_STATE__", repr(cancel_state))
    )
    script = _install_executable(tmp_path / "fakebin", "agentctl", source)
    monkeypatch.setenv("PATH", str(script.parent))
    monkeypatch.setattr(pytest_slot, "POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(pytest_slot, "OBSERVATION_BACKOFF_MAX_S", 0.0)
    return Path(str(script) + ".calls.jsonl")


def _calls(record: Path) -> list[dict[str, Any]]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line]


def _verbs(record: Path) -> list[str]:
    """``job <verb>`` plus the job id when the call names one."""
    verbs = []
    for call in _calls(record):
        words = [word for word in call["argv"] if word != "--json"][:3]
        if len(words) == 3 and not words[2].isdigit():
            words.pop()
        verbs.append(" ".join(words))
    return verbs


def _marker_command(marker: Path) -> list[str]:
    """A command that proves it ran by creating ``marker``."""
    return [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]


def _environment(**extra: str) -> dict[str, str]:
    base = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/home/nobody"),
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "XDG_DATA_HOME": "/home/nobody/.local/share",
        "POLYLOGUE_ROOT": "/somewhere",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    base.update(extra)
    return base


def _launch_document(root: Path) -> dict[str, Any]:
    """The launch file captured by the fake agentctl before consumption."""
    document: dict[str, Any] = json.loads((root / "submitted-launch.json").read_text(encoding="utf-8"))
    return document


@pytest.fixture(autouse=True)
def _outside_the_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Launcher tests use a controlled roomy budget, never the outer test job's live headroom."""
    stub_cgroup(OUTSIDE_CGROUP, tmp_path=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _environment: None)

    def roomy_worker_cap(
        requested: int,
        *,
        profile: worker_memory.ChargeProfile,
        max_workers: int | None = None,
        **_kwargs: Any,
    ) -> tuple[int, dict[str, Any]]:
        workers = min(requested, max_workers) if max_workers is not None else requested
        budget = profile.charge_mib(workers) + 1024.0
        basis = profile.admission_estimate(workers, budget)
        basis.update(
            {
                "admission": "admitted",
                "available_mib": budget,
                "basis": "declared_budget",
                "cgroup_available_mib": None,
                "cgroup_directory": "/fixture",
                "host_available_mib": budget,
                "limiting_cgroups": ["/fixture"],
                "narrowed": workers < requested,
                "requested_workers": requested,
                "workers": workers,
            }
        )
        return workers, basis

    monkeypatch.setattr(worker_memory, "memory_bounded_worker_cap", roomy_worker_cap)


_CONTROLLED_ADMISSION_SETUP = """
from devtools import pytest_slot, worker_memory

def roomy_worker_cap(requested, *, profile, max_workers=None, **kwargs):
    workers = min(requested, max_workers) if max_workers is not None else requested
    budget = profile.charge_mib(workers) + 1024.0
    basis = profile.admission_estimate(workers, budget)
    basis.update(dict(
        admission="admitted",
        available_mib=budget,
        basis="declared_budget",
        cgroup_available_mib=None,
        cgroup_directory="/fixture",
        host_available_mib=budget,
        limiting_cgroups=["/fixture"],
        narrowed=workers < requested,
        requested_workers=requested,
        workers=workers,
    ))
    return workers, basis

worker_memory.memory_bounded_worker_cap = roomy_worker_cap
pytest_slot.admission_ledger = lambda _environment: None
"""


def test_outside_the_pool_the_run_is_submitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"

    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert not marker.exists(), "pytest ran directly instead of through the pytest pool"
    assert outcome.slot == "agentctl job 7"
    assert outcome.returncode == 0
    assert outcome.log_path is not None and outcome.log_path.parent == tmp_path / ".cache" / "verify"
    assert outcome.log_path.name.startswith(f"pytest-slot-{os.getpid()}-")
    identity = outcome.log_path.stem.removeprefix("pytest-slot-")
    start = next(call for call in _calls(record) if "start" in call["argv"])
    assert start["argv"] == [
        "--json",
        "job",
        "start",
        str(tmp_path),
        "pytest_focused",
        "--workspace",
        str(tmp_path),
        "--",
        str(tmp_path / ".cache" / "verify" / f"pytest-slot-{identity}.json"),
    ]
    launch = _launch_document(tmp_path)
    assert launch["kind"] == "polylogue.pytest-slot-launch"
    assert launch["argv"][:3] == _marker_command(marker)
    assert launch["working_directory"] == str(tmp_path)
    assert launch["log_path"] == str(outcome.log_path)
    assert _verbs(record) == ["job start", "job get 7"]
    assert not list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json")), "the launch file outlived its run"


def test_a_queued_run_names_its_output_log_when_it_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The captured log is named while the job waits, not only after it ends.

    Anti-vacuity: printing the path only on release leaves a queued run with
    no watchable output, since the child writes nowhere else.
    """
    _install_fake_agentctl(tmp_path, monkeypatch)

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    [waiting] = [line for line in capsys.readouterr().err.splitlines() if "waiting for the host pytest slot" in line]
    assert str(outcome.log_path) in waiting


def test_two_acquisitions_keep_distinct_capture_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rerun cannot replace the first acquisition's captured stdout file."""
    _install_fake_agentctl(tmp_path, monkeypatch)
    first = run_pytest(_marker_command(tmp_path / "first"), cwd=str(tmp_path), env=_environment(), root=tmp_path)
    second = run_pytest(_marker_command(tmp_path / "second"), cwd=str(tmp_path), env=_environment(), root=tmp_path)
    assert first.log_path is not None and second.log_path is not None
    assert first.log_path != second.log_path
    assert first.log_path.exists() and second.log_path.exists()
    assert first.log_path.read_bytes() == second.log_path.read_bytes() == b"captured output"


def test_the_client_environment_carries_only_the_allowed_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    secret_environment = _environment(
        ANTHROPIC_API_KEY="secret",
        AGENTCTL_PRINCIPAL="agent-control",
        SINNIXD_PRINCIPAL="agent-control",
        POLYLOGUE_ARCHIVE_ROOT="/realm/state/polylogue",
    )

    run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=secret_environment, root=tmp_path)

    allowed = set(pytest_slot.INHERITED_ENVIRONMENT_KEYS)
    for call in _calls(record):
        recorded = {key for key in call["env"] if not key.startswith(("PYTHON", "LC_", "LANG"))}
        assert recorded <= allowed, f"agentctl inherited {sorted(recorded - allowed)}"
    launch = _launch_document(tmp_path)
    assert launch["environment"]["ANTHROPIC_API_KEY"] == "secret", "pytest's own environment travels in the launch file"
    assert "POLYLOGUE_ARCHIVE_ROOT" not in launch["environment"], (
        "a queued test must not carry the operator archive path into its launch snapshot"
    )


def test_agentctl_jobs_do_not_inject_the_live_archive_root() -> None:
    descriptor = Path(__file__).resolve().parents[3] / ".agentctl" / "project.toml"
    payload = tomllib.loads(descriptor.read_text(encoding="utf-8"))
    environment = payload["environment"]

    assert "POLYLOGUE_ARCHIVE_ROOT" not in environment.get("inherit", [])
    assert "POLYLOGUE_ARCHIVE_ROOT" in environment.get("unset", [])
    assert "XDG_DATA_HOME" not in environment.get("inherit", [])
    assert "XDG_DATA_HOME" in environment.get("unset", [])
    assert "POLYLOGUE_ARCHIVE_ROOT" not in environment.get("require", [])
    assert "POLYLOGUE_ARCHIVE_ROOT" not in environment.get("values", {})


@pytest.mark.parametrize("prefix", ["AGENTCTL_", "SINNIXD_"])
def test_a_job_id_is_never_slot_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """A lane inherits job identity, a principal and an operation name; none is the slot."""
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    lane_environment = _environment(
        **{
            f"{prefix}JOB_ID": "job-1",
            f"{prefix}OPERATION": "verify_affected",
            f"{prefix}QUEUE_WORKER": "1",
            f"{prefix}PRINCIPAL": "agent-control",
        }
    )

    assert not holds_pytest_slot(lane_environment)
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=lane_environment, root=tmp_path)

    assert not marker.exists()
    assert outcome.slot == "agentctl job 7"
    assert _verbs(record)[0] == "job start"


@pytest.mark.parametrize("cgroup", PYTEST_CGROUPS)
def test_the_pytest_pool_cgroup_holds_the_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cgroup: str) -> None:
    """The slice the runtime placed the job in is ownership, with no environment at all.

    Anti-vacuity: pointing the stub at any other slice makes this red, because
    the run submits instead of executing.
    """
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    stub_cgroup(cgroup, tmp_path=tmp_path, monkeypatch=monkeypatch)
    marker = tmp_path / "pytest-ran"

    assert holds_pytest_slot({})
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert marker.exists()
    assert outcome.slot == "held"
    assert outcome.returncode == 0
    assert _calls(record) == [], "a pytest-pool job must not recursively submit"


@pytest.mark.parametrize("pool_variable", ["AGENTCTL_POOL", "SINNIXD_QUEUE_POOL"])
def test_the_declared_pytest_pool_holds_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pool_variable: str
) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    holder = {pool_variable: "pytest"}

    assert holds_pytest_slot(holder)
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(**holder), root=tmp_path)

    assert marker.exists()
    assert outcome.slot == "held"
    assert _calls(record) == []


def test_explicit_slot_holder_runs_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    marker = tmp_path / "pytest-ran"
    holder = {"POLYLOGUE_PYTEST_SLOT": "held"}

    assert holds_pytest_slot(holder)
    outcome = run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(**holder), root=tmp_path)

    assert marker.exists()
    assert outcome.slot == "held"
    assert outcome.returncode == 0
    assert _calls(record) == [], "a slot holder must not talk to agentctl"


@pytest.mark.parametrize("cgroup", AGENT_CGROUPS)
def test_an_agent_pool_job_submits_its_focused_run_to_the_pytest_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cgroup: str
) -> None:
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    stub_cgroup(cgroup, tmp_path=tmp_path, monkeypatch=monkeypatch)
    marker = tmp_path / "pytest-ran"

    outcome = run_pytest(
        _marker_command(marker),
        cwd=str(tmp_path),
        env=_environment(AGENTCTL_JOB_ID="agent-job", AGENTCTL_POOL="agent"),
        root=tmp_path,
    )

    assert not marker.exists()
    assert outcome.slot == "agentctl job 7"
    start = next(call for call in _calls(record) if "start" in call["argv"])
    assert "pytest_focused" in start["argv"]


def test_a_missing_runtime_refuses_rather_than_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "pytest-ran"
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(PytestSlotUnavailableError) as failure:
        run_pytest(_marker_command(marker), cwd=str(tmp_path), env=_environment(PATH=str(empty)), root=tmp_path)

    assert "`agentctl` is not on PATH" in str(failure.value)
    assert "systemctl --user start pueued" in str(failure.value)
    assert not marker.exists()
    assert not list((tmp_path / ".cache" / "verify").glob("tmp-*")), "a refused run leaves no scratch"


def test_a_failed_job_reports_its_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase="failed", exit_code=1)

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert (outcome.returncode, outcome.slot) == (1, "agentctl job 12")


def test_transient_get_failure_keeps_waiting_on_the_submitted_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _install_scripted_agentctl(
        tmp_path,
        monkeypatch,
        [
            "error",
            {"job_id": 23, "phase": "running", "terminal": False},
            {"job_id": 23, "phase": "succeeded", "terminal": True, "exit_code": 0},
        ],
    )

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert outcome.returncode == 0
    assert _verbs(record) == ["job start", "job get 23", "job get 23", "job get 23"]


def test_read_failures_never_abandon_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of failed reads is a stall to report, not a reason to cancel.

    Anti-vacuity: restore a failure limit and the ten failed reads below end
    in a cancelled job instead of the terminal result.
    """
    record = _install_scripted_agentctl(
        tmp_path,
        monkeypatch,
        ["error"] * 10 + [{"job_id": 23, "phase": "succeeded", "terminal": True, "exit_code": 0}],
    )

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert outcome.returncode == 0
    assert "job cancel 23" not in _verbs(record)
    assert _verbs(record).count("job get 23") == 11


def test_reads_address_the_job_by_its_launch_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``job get`` passes the reference ``job start`` returned.

    Anti-vacuity: read by bare id and a job whose pueue entry was cleaned is
    unreadable ("pueue has no task"), so the wait could never end.
    """
    record = _install_scripted_agentctl(
        tmp_path, monkeypatch, [{"job_id": 23, "phase": "succeeded", "terminal": True, "exit_code": 0}]
    )

    run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    gets = [call["argv"] for call in _calls(record) if "get" in call["argv"]]
    assert gets and all(call[-2:] == ["--reference", "ref-23"] for call in gets)


def test_an_invalid_view_is_waited_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _install_scripted_agentctl(
        tmp_path,
        monkeypatch,
        [{"job_id": 99, "terminal": True}, {"job_id": 23, "phase": "succeeded", "terminal": True, "exit_code": 0}],
    )

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert outcome.returncode == 0
    assert "job cancel 23" not in _verbs(record)


@pytest.mark.parametrize(
    ("phase", "meaning"),
    [
        ("cancelled", "the job was cancelled"),
        ("vanished", "the job vanished"),
        ("slot_occupied", "the pytest pool was occupied"),
        ("refused", "the runtime refused"),
        ("dependency-failed", "depended on failed"),
        ("launch-failed", "could not be launched"),
    ],
)
def test_a_job_that_did_not_run_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, meaning: str
) -> None:
    """The phases are agentctl's own (``run.Outcome`` plus the launch-side ones)."""
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase=phase, exit_code=130)

    with pytest.raises(PytestSlotUnavailableError, match=f"{meaning}.*agentctl job 12 ended '{phase}'"):
        run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)


def test_a_success_receipt_does_not_override_runtime_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {"kind": "polylogue.pytest-slot-result", "status": "success", "exit_code": 0}
    _install_fake_agentctl(
        tmp_path,
        monkeypatch,
        job_id=12,
        phase="cancelled",
        exit_code=None,
        receipt=receipt,
    )

    with pytest.raises(PytestSlotUnavailableError) as failure:
        run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    runtime_evidence = failure.value.runtime_evidence
    assert runtime_evidence is not None
    assert runtime_evidence["phase"] == "cancelled"
    assert runtime_evidence["pytest_slot_receipt"] == receipt


def test_an_unknown_terminal_phase_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase="something-new", exit_code=3)

    with pytest.raises(PytestSlotUnavailableError, match="agentctl job 12 ended 'something-new' \\(exit 3\\)"):
        run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)


def test_a_timed_out_job_reports_the_typed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = {"status": "timed_out", "diagnosis": "pytest_deadline"}
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase="timeout", exit_code=124, receipt=receipt)

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert outcome.returncode == 124
    assert outcome.receipt == receipt


def test_progress_counts_completed_nodeids_not_phase_reports(tmp_path: Path) -> None:
    """Setup/call/teardown reports for one item contribute one completed test."""
    events = tmp_path / "events.jsonl"
    lines = [
        {"event": "test_report", "nodeid": "test_a", "when": phase, "outcome": "passed"}
        for phase in ("setup", "call", "teardown")
    ]
    lines.append({"event": "test_finished", "nodeid": "test_a"})
    events.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    snapshot = pytest_slot._ProgressSnapshot({"POLYLOGUE_PYTEST_EVENTS_PATH": str(events)})
    progress = snapshot()
    assert progress["terminal_count"] == 1
    assert progress["outcomes"] == {"passed": 1}


def test_a_stale_result_document_is_not_reported_as_this_run_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's paths are per pid, and a pid is reused."""
    _install_fake_agentctl(tmp_path, monkeypatch, job_id=12, phase="succeeded", exit_code=0)
    stale = tmp_path / ".cache" / "verify" / f"pytest-slot-{os.getpid()}.result.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"status": "timed_out", "diagnosis": "pytest_deadline"}), encoding="utf-8")

    outcome = run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    assert outcome.returncode == 0
    assert outcome.receipt is None


def test_the_slot_runner_executes_the_launch_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    marker = tmp_path / "pytest-ran"
    log_path = tmp_path / "slot.log"
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    f"import os; open({str(marker)!r},'w').write(os.environ['ONLY_THIS']); print('hi')",
                ],
                "working_directory": str(tmp_path),
                "environment": {"PATH": os.environ["PATH"], "ONLY_THIS": "value"},
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )

    assert pytest_slot.main([str(launch_path)]) == 0

    assert marker.read_text(encoding="utf-8") == "value"
    assert "hi" in log_path.read_text(encoding="utf-8")
    assert not launch_path.exists(), "the launch file carries a resolved environment and must not persist"
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "polylogue.pytest-slot-result"
    assert (result["status"], result["exit_code"]) == ("success", 0)


def test_the_slot_runner_runs_a_selection_through_devtools_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """``agentctl job start polylogue pytest_focused -- <selection>`` is ``devtools test`` in the pool."""
    from devtools import run_tests

    seen: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        seen.append(list(argv))
        return 3

    monkeypatch.setattr(run_tests, "main", fake_main)

    assert pytest_slot.main(["tests/unit/devtools/test_agent_env.py", "-n", "0"]) == 3
    assert seen == [["tests/unit/devtools/test_agent_env.py", "-n", "0"]]
    assert pytest_slot.main([]) == 2


#: How long the backgrounded descendant sleeps before it touches its marker.
#: The reap happens within milliseconds of the SIGTERM, so a check taken
#: before this instant cannot distinguish a reaped descendant from one that is
#: merely still sleeping -- which is exactly how this test used to pass under
#: its own named mutation.
_DESCENDANT_MARKER_DELAY_S = 2.0


@pytest.mark.uses_real_clock("exercises a real signal deadline and process-group reap")
def test_slot_timeout_writes_typed_receipt_and_reaps_child_group(tmp_path: Path) -> None:
    """An external deadline retains progress even when pytest cannot finalize reports.

    Anti-vacuity: removing the signal handler loses the sibling receipt; omitting
    ``start_new_session`` from the ``Popen`` in ``devtools.pytest_slot`` leaves
    the sleeping descendant in the runner's own process group, where
    ``killpg`` never reaches it, and the marker below appears. Both were
    executed against this file: the second one is red here.

    The wait is derived from the descendant's own marker delay rather than
    left at a fixed 0.1 s. Measured, the 0.1 s form is red under that mutation
    too -- but only because the runner's SIGTERM escalation happens to take
    longer than the descendant's 2 s sleep, so ``process.wait()`` returns after
    the marker already exists. That is a property of the reap's escalation
    budget, not of this test; shorten the escalation and the fixed margin
    silently stops proving anything.
    """
    log_path = tmp_path / "pytest-slot-1.log"
    launch_path = tmp_path / "launch.json"
    events_path = tmp_path / "events.jsonl"
    started = tmp_path / "started"
    survivor = tmp_path / "survivor"
    events_path.write_text(
        json.dumps({"event": "test_report", "when": "call", "outcome": "passed"})
        + "\n"
        + json.dumps({"event": "test_finished", "nodeid": "test_a"})
        + "\n",
        encoding="utf-8",
    )
    child = f"touch {started}; (sleep {_DESCENDANT_MARKER_DELAY_S:g}; touch {survivor}) & sleep 30"
    launch_path.write_text(
        json.dumps(
            {
                "argv": ["sh", "-c", child],
                "working_directory": str(tmp_path),
                "environment": {
                    "PATH": os.environ["PATH"],
                    "POLYLOGUE_PYTEST_EVENTS_PATH": str(events_path),
                },
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )
    # This runner is a separate interpreter, so pytest's in-process admission
    # fixture cannot control its resource estimate. Install the same controlled
    # admission setup through Python's normal sitecustomize hook while keeping
    # the real module entrypoint and subprocess lifecycle under test.
    bootstrap = tmp_path / "python-bootstrap"
    bootstrap.mkdir()
    repo_root = Path(__file__).resolve().parents[3]
    (bootstrap / "sitecustomize.py").write_text(
        f"import sys\nsys.path.insert(0, {str(repo_root)!r})\n{_CONTROLLED_ADMISSION_SETUP}",
        encoding="utf-8",
    )
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = os.pathsep.join((str(bootstrap), child_environment.get("PYTHONPATH", "")))
    process = subprocess.Popen(
        [sys.executable, "-m", "devtools.pytest_slot", str(launch_path)],
        env=child_environment,
    )
    try:
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        signalled_at = time.monotonic()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 128 + signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()

    receipt = json.loads(log_path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["diagnosis"] == "pytest_interrupted"
    assert receipt["elapsed_s"] >= 0
    assert receipt["progress"]["terminal_count"] == 1
    # The descendant began its sleep at or before the signal, so waiting past
    # that sleep from the signal instant is a strict upper bound on when a
    # surviving descendant would have written its marker.
    marker_due = signalled_at + _DESCENDANT_MARKER_DELAY_S + 0.5
    while time.monotonic() < marker_due:
        time.sleep(0.05)
    assert not survivor.exists()


def test_a_submitted_run_contains_its_temporary_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither pytest nor the code under test may write to the ambient TMPDIR.

    ``nix develop`` points TMPDIR at a small tmpfs; a corpus run left there
    fills the mount and dies on exhausted file descriptors.
    """
    _install_fake_agentctl(tmp_path, monkeypatch)
    scratch = tmp_path / ".cache" / "verify"

    run_pytest(
        _marker_command(tmp_path / "unused"),
        cwd=str(tmp_path),
        env=_environment(TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
    )

    launch = _launch_document(tmp_path)
    assert Path(launch["environment"]["TMPDIR"]).is_relative_to(scratch)
    argv = launch["argv"]
    basetemp = Path(argv[argv.index("--basetemp") + 1])
    assert basetemp.is_relative_to(scratch)


def test_a_declared_basetemp_is_kept_and_still_anchors_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``devtools test`` names its own basetemp so it can dispose of it."""
    _install_fake_agentctl(tmp_path, monkeypatch)
    declared = tmp_path / ".cache" / "verify" / "tmp-chosen"

    run_pytest(
        [*_marker_command(tmp_path / "unused"), "--basetemp", str(declared)],
        cwd=str(tmp_path),
        env=_environment(TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
    )

    launch = _launch_document(tmp_path)
    assert launch["argv"].count("--basetemp") == 1
    assert launch["argv"][launch["argv"].index("--basetemp") + 1] == str(declared)
    # pytest empties its own basetemp as it starts, so TMPDIR must be beside it.
    tmpdir = Path(launch["environment"]["TMPDIR"])
    assert tmpdir.parent == declared.parent and tmpdir != declared


def test_a_run_holding_the_slot_sees_the_contained_tmpdir(tmp_path: Path) -> None:
    recorded = tmp_path / "tmpdir-seen"
    command = [sys.executable, "-c", f"import os; open({str(recorded)!r},'w').write(os.environ['TMPDIR'])"]

    run_pytest(
        command,
        cwd=str(tmp_path),
        env=_environment(POLYLOGUE_PYTEST_SLOT="held", TMPDIR="/tmp/nix-shell.L3brFS"),
        root=tmp_path,
    )

    seen = Path(recorded.read_text(encoding="utf-8"))
    assert seen.is_relative_to(tmp_path / ".cache" / "verify")


def _scratch_trees(root: Path) -> list[Path]:
    return sorted((root / ".cache" / "verify").glob("tmp-*"))


def test_a_successful_run_removes_both_temporary_trees(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import os, sys; os.makedirs(os.path.join(os.environ['TMPDIR'], 'used')); os.makedirs(sys.argv[2])",
    ]

    outcome = run_pytest(command, cwd=str(tmp_path), env=_environment(POLYLOGUE_PYTEST_SLOT="held"), root=tmp_path)

    assert outcome.returncode == 0
    assert _scratch_trees(tmp_path) == []


def test_a_successful_run_preserves_hard_linked_fixture_mode(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("retained fixture\n", encoding="utf-8")
    fixture.chmod(stat.S_IRUSR)
    command = [
        sys.executable,
        "-c",
        (
            "import os, sys; "
            "scratch = os.environ['TMPDIR']; "
            "nested = os.path.join(scratch, 'used', 'nested'); os.makedirs(nested); "
            "os.link(sys.argv[1], os.path.join(nested, 'fixture.txt')); "
            "basetemp = sys.argv[-1]; os.makedirs(basetemp); "
            "os.chmod(nested, 0o500); os.chmod(os.path.dirname(nested), 0o500); "
            "os.chmod(scratch, 0o500); os.chmod(basetemp, 0o500)"
        ),
        str(fixture),
    ]

    outcome = run_pytest(command, cwd=str(tmp_path), env=_environment(POLYLOGUE_PYTEST_SLOT="held"), root=tmp_path)

    assert outcome.returncode == 0
    assert _scratch_trees(tmp_path) == []
    assert fixture.read_text(encoding="utf-8") == "retained fixture\n"
    assert stat.S_IMODE(fixture.stat().st_mode) == stat.S_IRUSR


def test_a_failed_run_keeps_its_temporary_trees_for_reading(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import os, sys; os.makedirs(os.path.join(os.environ['TMPDIR'], 'used')); os.makedirs(sys.argv[2]); sys.exit(1)",
    ]

    outcome = run_pytest(command, cwd=str(tmp_path), env=_environment(POLYLOGUE_PYTEST_SLOT="held"), root=tmp_path)

    assert outcome.returncode == 1
    assert [tree.name.endswith(".tmpdir") for tree in _scratch_trees(tmp_path)] == [False, True]


def test_an_interrupted_held_run_removes_its_temporary_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: disposing only on a normal return leaves the trees this test finds."""

    def interrupted(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", interrupted)

    with pytest.raises(KeyboardInterrupt):
        run_pytest(
            _marker_command(tmp_path / "unused"),
            cwd=str(tmp_path),
            env=_environment(POLYLOGUE_PYTEST_SLOT="held"),
            root=tmp_path,
        )

    assert _scratch_trees(tmp_path) == []


def test_a_configured_basetemp_root_is_honoured(tmp_path: Path) -> None:
    """A sandbox with no checkout-local scratch names its own root."""
    elsewhere = tmp_path / "sandbox-scratch"

    assert basetemp_root({BASETEMP_ROOT_ENV: str(elsewhere)}, root=tmp_path) == elsewhere


def test_the_leaked_cloud_basetemp_sentinel_is_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`.claude/settings.json` exports the cloud value into workstation sessions."""
    monkeypatch.setattr(cloud_sentinels, "_WORKSTATION_SCRATCH_MOUNT", tmp_path)
    sentinel = cloud_sentinels.CLOUD_SENTINELS[BASETEMP_ROOT_ENV]

    assert basetemp_root({BASETEMP_ROOT_ENV: sentinel}, root=tmp_path) == tmp_path / ".cache" / "verify"


#: An ``agentctl`` whose ``job get`` kills the process waiting on it, the way a
#: session or a wrapper being killed leaves a queued job with no waiter. The
#: cancellation is recorded like every other call; the refusing variant
#: answers ``job cancel`` the way a job id pueue no longer holds is answered.
FAKE_AGENTCTL_KILLS_ITS_WAITER = """import json, os, signal, sys, time

with open(sys.argv[0] + ".calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:]}}) + "\\n")

words = [word for word in sys.argv[1:] if word != "--json"]
verb = " ".join(words[:2])
if verb == "job start":
    print(json.dumps({{"job_id": 11, "phase": "queued", "terminal": False}}))
elif verb == "job get":
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(2)
    print(json.dumps({{"job_id": 11, "phase": "queued", "terminal": False}}))
elif verb == "job cancel":
    if {refuses}:
        sys.stderr.write("agentctl: no such job\\n")
        sys.exit(1)
    print(json.dumps({{"job_id": 11, "state": "removed"}}))
sys.exit(0)
"""

_WAITER = """
import os, sys
sys.path.insert(0, {repo!r})
from devtools import agent_env
agent_env._CGROUP_PATH = type(agent_env._CGROUP_PATH)({cgroup!r})
from devtools.pytest_slot import run_pytest

run_pytest(
    [sys.executable, "-c", "pass"],
    cwd={cwd!r},
    env={{"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}},
    root={root!r},
)
"""


def _killed_waiter(tmp_path: Path, *, refuses: bool) -> tuple[subprocess.CompletedProcess[str], Path]:
    directory = tmp_path / "fakebin"
    agentctl = _install_executable(directory, "agentctl", FAKE_AGENTCTL_KILLS_ITS_WAITER.format(refuses=refuses))
    repo = str(Path(pytest_slot.__file__).resolve().parents[1])
    cgroup = tmp_path / "cgroup"
    cgroup.write_text(OUTSIDE_CGROUP, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _WAITER.format(repo=repo, cwd=str(tmp_path), root=str(tmp_path), cgroup=str(cgroup)),
        ],
        env={
            "PATH": str(directory),
            "HOME": os.environ.get("HOME", "/home/nobody"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed, Path(str(agentctl) + ".calls.jsonl")


def test_a_killed_waiter_reaps_the_job_it_submitted(tmp_path: Path) -> None:
    """A job outlives its waiter, and the pool's parallelism is one.

    Anti-vacuity: dropping the reaping action records no cancellation at all --
    the job stays queued with nothing left to wait on it, which is exactly the
    starvation this reap exists to prevent. Disposing of the scratch trees
    only on a normal return leaves the ``tmp-*`` directories this test finds.
    """
    completed, record = _killed_waiter(tmp_path, refuses=False)

    assert completed.returncode == -int(signal.SIGTERM), completed.stderr
    assert _verbs(record) == ["job start", "job get 11", "job cancel 11"], _verbs(record)
    assert not list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json")), (
        "the launch file carries a resolved environment and must not survive the reap"
    )
    assert _scratch_trees(tmp_path) == [], "a killed waiter leaves no scratch behind"


def test_an_interrupt_right_after_job_start_still_reaps_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signal between ``job start`` and the wait cancels the job it created.

    The interrupt lands while the waiting message is written. Anti-vacuity:
    installing the reaping handler only around the wait lets this interrupt
    unwind with no ``job cancel``, leaving the queued job in the pytest pool.
    """
    record = _install_fake_agentctl(tmp_path, monkeypatch)
    real_stderr = sys.stderr

    class _InterruptOnWait:
        def write(self, text: str) -> int:
            if "waiting for the host pytest slot" in text:
                raise KeyboardInterrupt
            return real_stderr.write(text)

        def flush(self) -> None:
            real_stderr.flush()

    monkeypatch.setattr(sys, "stderr", _InterruptOnWait())

    with pytest.raises(KeyboardInterrupt):
        run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)

    verbs = _verbs(record)
    assert verbs[0] == "job start", verbs
    assert "job cancel 7" in verbs, verbs


def test_a_refused_cancellation_leaves_the_launch_file_for_the_job(tmp_path: Path) -> None:
    """A job agentctl would not stop still reads its launch file when it starts.

    Anti-vacuity: unlinking the launch file regardless of the cancellation's
    outcome empties the glob below, and the surviving job then starts with no
    resolved environment to read.
    """
    completed, record = _killed_waiter(tmp_path, refuses=True)

    assert completed.returncode == -int(signal.SIGTERM), completed.stderr
    assert _verbs(record)[-1] == "job cancel 11"
    surviving = list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json"))
    assert len(surviving) == 1, surviving
    assert _scratch_trees(tmp_path), "a possibly running job retains its referenced temp trees"


def test_terminal_refusal_removes_secret_bearing_launch_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal non-execution phase must not leave the resolved environment on disk."""
    _install_fake_agentctl(tmp_path, monkeypatch, phase="refused", exit_code=None)
    with pytest.raises(PytestSlotUnavailableError):
        run_pytest(_marker_command(tmp_path / "unused"), cwd=str(tmp_path), env=_environment(), root=tmp_path)
    assert not list((tmp_path / ".cache" / "verify").glob("pytest-slot-*.json"))


@pytest.mark.uses_real_clock("measures a real child process group over sampling intervals")
def test_a_held_run_records_what_it_took(tmp_path: Path) -> None:
    """The receipt attributes the run's peak to the processes that took it."""
    command = [
        sys.executable,
        "-c",
        "import time; block = b'x' * (96 * 1024 * 1024); time.sleep(1.2); del block",
    ]

    outcome = run_pytest(command, cwd=str(tmp_path), env=_environment(POLYLOGUE_PYTEST_SLOT="held"), root=tmp_path)

    assert outcome.returncode == 0
    receipt = outcome.receipt
    assert receipt is not None
    assert receipt["kind"] == "polylogue.pytest-slot-result"
    memory = receipt["memory"]
    assert memory["observed_samples"] >= 1
    # The child allocated 96 MiB; the peak is at least that, and it is named.
    assert memory["peak"]["pss_kib"] >= 96 * 1024
    assert memory["processes"][0]["peak_rss_kib"] >= 96 * 1024
    assert memory["host_mem_available_mib"]["minimum"] is not None


def test_a_held_run_with_no_explicit_worker_count_still_receipts_its_admitted_width(tmp_path: Path) -> None:
    """A serial/no-``-n`` run's receipt still names the width and budget it was admitted on.

    (polylogue-k1o3t AC2a) Before this, ``resize_worker_argument`` returned no
    basis at all for a command naming no worker count, so a real receipt --
    ``.cache/verify/runs/*-focused-test-*/run.json`` ``steps[0].pytest_slot_receipt``
    -- carried only ``elapsed_s, exit_code, kind, log_path, memory,
    schema_version, status``: no selected width, no budget, no narrowing
    decision, ever, for the whole class of runs that name no ``-n``. A
    command with no ``-n`` is read as a request for one worker, the width it
    already runs at; the receipt must still name the observed budget it was
    measured against.
    """
    command = [sys.executable, "-c", "print('ok')"]

    outcome = run_pytest(command, cwd=str(tmp_path), env=_environment(POLYLOGUE_PYTEST_SLOT="held"), root=tmp_path)

    assert outcome.returncode == 0
    receipt = outcome.receipt
    assert receipt is not None
    sizing = receipt.get("sizing")
    assert sizing is not None, "the receipt must name the width/budget this run was admitted on"
    assert sizing["workers"] == 1
    assert sizing["requested_workers"] == 1
    assert sizing["narrowed"] is False
    assert sizing["basis"] in {"cgroup_budget", "declared_budget"}


@pytest.mark.uses_real_clock("runs a real child through the slot runner")
def test_a_queued_run_publishes_its_result_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A run that ends normally files the same document a timed-out one does.

    The waiting client reads that file: it is how the width a queued run chose,
    and the peak it then reached, reach the verification receipt at all.
    """
    log_path = tmp_path / "slot.log"
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", "import time; time.sleep(0.7)"],
                "working_directory": str(tmp_path),
                "environment": {"PATH": os.environ["PATH"]},
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )

    assert pytest_slot.main([str(launch_path)]) == 0

    published = json.loads(log_path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert published == json.loads(capsys.readouterr().out)
    assert published["status"] == "success"
    assert published["memory"]["observed_samples"] >= 1
    assert published["memory"]["processes"], "the run's own processes are named"


def test_a_failed_queued_run_keeps_sizing_telemetry_sidecar(tmp_path: Path) -> None:
    """Launch sizing is durable even when the child prevents a final success receipt.

    The child sleeps briefly before exiting. ``ProcessGroupMemorySampler`` takes
    its first sample as soon as its thread is scheduled, with no initial wait,
    but ``subprocess.Popen`` returns (and the child starts running) strictly
    before the sampler is constructed and started -- a child that exits
    immediately can be gone from ``/proc`` before any thread gets scheduled to
    read it. Measured live under host contention (20 concurrent managed pytest
    jobs, agentctl job 362, 2026-09-22): a bare ``raise SystemExit(1)`` child
    completed the whole run in 0.03s and left ``observed_samples: 0`` /
    ``peak.pss_kib: 0`` -- this exact assertion failed for exactly that reason.
    The sleep removes the race without weakening what is asserted.
    """
    log_path = tmp_path / "slot.log"
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", "import time; time.sleep(0.3); raise SystemExit(1)", "-n", "8"],
                "working_directory": str(tmp_path),
                "environment": {"PATH": os.environ["PATH"]},
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )

    assert pytest_slot.main([str(launch_path)]) == 1

    telemetry = json.loads(log_path.with_suffix(".telemetry.json").read_text(encoding="utf-8"))
    assert telemetry["status"] == "failed"
    assert telemetry["kind"] == "polylogue.pytest-slot-telemetry"
    assert telemetry["sizing"]["requested_workers"] == 8
    assert telemetry["memory"]["process_group"] > 0
    assert telemetry["memory"]["peak"]["pss_kib"] > 0


def test_worktree_snapshot_refuses_content_change_during_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The recorded digest and HEAD must describe one stable checkout snapshot."""
    from types import SimpleNamespace

    from devtools import checkout_identity, verify_runs

    identity = SimpleNamespace(head="a" * 40, branch="feature/test")
    monkeypatch.setattr(checkout_identity, "checkout_identity", lambda _root: identity)
    monkeypatch.setattr(checkout_identity, "default_branch_refusal", lambda *_a, **_kw: None)
    monkeypatch.setattr(verify_runs, "git_dirty", lambda _root: False)
    digests = iter(("first", "changed"))
    monkeypatch.setattr(verify_runs, "git_worktree_content_sha256", lambda _root: next(digests))
    with pytest.raises(PytestSlotUnavailableError, match="changed while its content"):
        pytest_slot._focused_worktree_provenance(str(tmp_path), {pytest_slot.WORKTREE_PROVENANCE_ENV: "1"})


def test_the_startup_sweep_reclaims_only_dead_owners(tmp_path: Path) -> None:
    """``tmp-<pid>-*`` trees survive exactly as long as their owning pid does.

    A run killed outright runs no handler at all, so the next run's startup is
    the first moment anything can reclaim its basetemp (evidence: 514 such
    trees, 56.7 GiB, across this host's worktrees). Anti-vacuity: dropping the
    liveness check deletes the live owner's tree out from under a concurrent
    run and turns this red; dropping the sweep leaves the dead one behind.
    """
    from devtools.pytest_slot import sweep_stale_temp_trees

    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True)
    root = tmp_path / "verify"
    live = root / "tmp-4242-a1"
    dead = root / "tmp-4243-b2"
    dead_scratch = root / "tmp-4243-b2.tmpdir"
    unrelated = root / "runs"
    for directory in (live, dead, dead_scratch, unrelated):
        directory.mkdir(parents=True)
    (dead / "fixture").mkdir()

    removed = sweep_stale_temp_trees(root, proc=proc)

    assert sorted(path.name for path in removed) == ["tmp-4243-b2", "tmp-4243-b2.tmpdir"]
    assert live.exists()
    assert unrelated.exists()
    assert not dead.exists()
    assert not dead_scratch.exists()


@pytest.mark.uses_real_clock("waits on a real child process it signals")
@pytest.mark.parametrize("kill_signal", [signal.SIGTERM, signal.SIGKILL])
def test_a_killed_run_leaves_no_temporary_tree(tmp_path: Path, kill_signal: signal.Signals) -> None:
    """A signalled run leaves nothing: the guard on SIGTERM, the sweep on SIGKILL.

    Anti-vacuity: removing the signal handlers from the guard leaves the
    SIGTERM tree behind, and removing the startup sweep leaves the SIGKILL one
    behind -- neither can be reclaimed by the ``finally`` the ordinary route
    relies on.
    """
    from devtools.pytest_slot import sweep_stale_temp_trees

    root = tmp_path / "verify"
    root.mkdir()
    ready = tmp_path / "ready"
    program = (
        "import os, pathlib, sys, time\n"
        "from devtools.pytest_slot import guard_temp_trees\n"
        f"root = pathlib.Path({str(root)!r})\n"
        "tree = root / f'tmp-{os.getpid()}-deadbeef'\n"
        "(tree / 'fixture').mkdir(parents=True)\n"
        "guard_temp_trees(tree)\n"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", program],
        cwd=str(Path(__file__).parents[3]),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[3])},
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "the guarded child never started"
        tree = root / f"tmp-{child.pid}-deadbeef"
        assert tree.exists()
        child.send_signal(kill_signal)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:  # pragma: no cover - only on a stuck child
            child.kill()
            child.wait(timeout=30)

    if kill_signal is signal.SIGKILL:
        assert tree.exists(), "SIGKILL cannot run a handler; the sweep is the mechanism"
        sweep_stale_temp_trees(root)
    assert not tree.exists()
    assert list(root.iterdir()) == []


def test_the_child_reap_leaves_the_majority_of_the_unit_stop_budget() -> None:
    """The reap runs before the outer handler, so its worst case is a tax on it.

    A signalled run has one budget, and the receipt -- not the reap -- is the
    part of it that cannot be recovered afterwards.

    Anti-vacuity: restore the shipped 5 + 5 escalation and the worst case is
    10s of a 15s budget, which is not a minority of it, so this goes red.
    """
    assert pytest_slot.STOP_ESCALATION_BUDGET_S == pytest_slot.STOP_TERM_GRACE_S + pytest_slot.STOP_KILL_GRACE_S
    assert pytest_slot.STOP_ESCALATION_BUDGET_S < pytest_slot.UNIT_STOP_BUDGET_S / 2


#: A signalled waiter whose child refuses SIGTERM, so ``stop()`` must escalate
#: the whole way before the outer handler can write anything. The receipt this
#: writes stands in for verify.py's own terminal receipt: the same handler, in
#: the same place, after the same reap.
_SIGNALLED_HELD_RUN = """
import os, pathlib, signal, sys
sys.path.insert(0, {repo!r})
{setup}
from devtools.pytest_slot import _run_held

receipt = pathlib.Path({receipt!r})


class Interrupted(BaseException):
    pass


def interrupt(signal_number, frame):
    raise Interrupted()


signal.signal(signal.SIGTERM, interrupt)
ignores_sigterm = (
    "import signal, sys, time\\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
    "sys.stderr.write('up\\\\n'); sys.stderr.flush()\\n"
    "time.sleep(120)\\n"
)
try:
    _run_held(
        [sys.executable, "-c", ignores_sigterm],
        cwd={cwd!r},
        env={{"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}},
        stdout=sys.stderr,
        on_exit=lambda: None,
    )
except Interrupted:
    receipt.write_text("terminal")
"""


@pytest.mark.load_sensitive
@pytest.mark.uses_real_clock("spends a real stop budget on a real child process group")
def test_a_signalled_held_run_writes_its_receipt_inside_the_stop_budget(tmp_path: Path) -> None:
    """The interrupted receipt lands before the unit's stop budget runs out.

    Observed 2026-09-20 on a cancelled corpus run: Stopping 03:05:26.885 ->
    Stopped 03:05:36.898, 10.01s of a 15s ``DefaultTimeoutStopUSec`` spent
    escalating SIGTERM/SIGKILL at the child, leaving ~5s for
    finish_interrupted_steps, append_verify_history,
    append_verification_evidence, prune_successful_verify_runs and
    write_failure_seed. The receipt did not land.

    The child here ignores SIGTERM, so the reap takes its full escalation --
    which it does anyway: ``stop()`` runs inside a signal handler nested in
    the blocking ``Popen.wait()`` that holds ``_waitpid_lock``, so each leg
    spins to its own timeout. Measured 4.015s for 2 + 2.

    Anti-vacuity: restore ``wait(timeout=5)`` on both legs and the reap costs
    the incident's 10.01s, which is past the budget below twice over: the
    constant assertion goes red and so does the receipt.
    """
    receipt = tmp_path / "terminal-receipt"
    repo = str(Path(pytest_slot.__file__).resolve().parents[1])
    waiter = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _SIGNALLED_HELD_RUN.format(
                repo=repo, cwd=str(tmp_path), receipt=str(receipt), setup=_CONTROLLED_ADMISSION_SETUP
            ),
        ],
        env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/home/nobody")},
        stderr=subprocess.PIPE,
    )
    stream = waiter.stderr
    assert stream is not None
    try:
        assert stream.readline().strip() == b"up", "the SIGTERM-ignoring child never started"
        waiter.send_signal(signal.SIGTERM)
        # Under half the unit's stop budget, and above the whole escalation
        # with room for the unwind: what is under test is that the reap left
        # the handler above it time to finish. The relationship between the
        # constants is asserted on its own above; this budget is the wall
        # clock, so the failure here is behavioural.
        budget = 6.0
        assert budget < pytest_slot.UNIT_STOP_BUDGET_S / 2
        try:
            waiter.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            waiter.kill()
            waiter.wait(timeout=30)
            pytest.fail(f"the waiter did not unwind within {budget}s of the signal")
    finally:
        if waiter.poll() is None:  # pragma: no cover - only on a stuck waiter
            waiter.kill()
            waiter.wait(timeout=30)
        stream.close()

    assert receipt.read_text() == "terminal", "the outer handler must reach its receipt work"


#: A held run signalled at its deadline, with the caller's disposal wired the
#: way ``run_pytest`` wires it: the scratch telemetry sidecar is removed on the
#: way out. Nothing here installs an outer handler, so the re-raised signal
#: takes the process with the default action -- the shape the corpus operation
#: dies in when AgentCTL reaches its ``timeout_seconds``.
_INTERRUPTED_HELD_RUN = """
import os, pathlib, sys
sys.path.insert(0, {repo!r})
{setup}
from devtools.pytest_slot import _run_held

telemetry = pathlib.Path({telemetry!r})
result = pathlib.Path({result!r})
keep = [False]


def dispose():
    if not keep[0]:
        telemetry.unlink(missing_ok=True)


child = (
    "import sys, time\\n"
    "sys.stderr.write('up\\\\n'); sys.stderr.flush()\\n"
    "time.sleep(120)\\n"
)
_run_held(
    [sys.executable, "-c", child],
    cwd={cwd!r},
    env={{"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}},
    stdout=sys.stderr,
    on_exit=dispose,
    telemetry_path=telemetry,
    result_path=result,
    on_interrupt=lambda: keep.__setitem__(0, True),
)
"""


@pytest.mark.load_sensitive
@pytest.mark.uses_real_clock("signals a real child process group and reads what survived it")
def test_an_interrupted_held_run_preserves_its_receipt(tmp_path: Path) -> None:
    """A terminated held run leaves its width and its measured peak on disk.

    ``verify_all`` and ``verify_affected`` run through this path whenever they
    already hold the pytest pool. Before this, the signal handler stopped the
    child and re-raised, so ``sampler.stop()`` and the receipt return were
    never reached, and the caller's disposal deleted the live telemetry
    sidecar on the way past -- a deadline kill left no evidence at all, which
    is precisely the run whose sizing evidence matters most.

    Anti-vacuity: drop ``on_signal=preserve`` from ``_run_held``'s
    ``_on_exit`` and no result document exists; keep it but let ``dispose``
    run first and the sampler has nothing left to read. The opposite
    direction is pinned by
    ``test_a_completed_held_run_writes_no_interrupted_receipt``, so a handler
    that always writes ``timed_out`` cannot pass either.
    """
    telemetry = tmp_path / "telemetry.json"
    result = tmp_path / "held-run.log"
    repo = str(Path(pytest_slot.__file__).resolve().parents[1])
    waiter = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _INTERRUPTED_HELD_RUN.format(
                repo=repo,
                cwd=str(tmp_path),
                telemetry=str(telemetry),
                result=str(result),
                setup=_CONTROLLED_ADMISSION_SETUP,
            ),
        ],
        env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/home/nobody")},
        stderr=subprocess.PIPE,
    )
    stream = waiter.stderr
    assert stream is not None
    try:
        assert stream.readline().strip() == b"up", "the held child never started"
        waiter.send_signal(signal.SIGTERM)
        waiter.wait(timeout=30)
    finally:
        if waiter.poll() is None:  # pragma: no cover - only on a stuck waiter
            waiter.kill()
            waiter.wait(timeout=30)
        stream.close()

    # The scratch sidecar is gone: the caller's disposal ran, as it does in
    # production. The receipt is what had to outlive it.
    assert telemetry.exists(), "interrupted runs retain periodic telemetry for diagnosis"
    document = json.loads(pytest_slot._slot_result_path(result).read_text(encoding="utf-8"))
    assert document["kind"] == "polylogue.pytest-slot-result"
    assert document["status"] == "interrupted"
    assert document["signal"] == "SIGTERM"
    # The two facts the lost receipt was carrying: the width the run was
    # admitted at, and what it took at that width.
    assert document["sizing"]["workers"] >= 1
    assert document["memory"]["peak"]["rss_kib"] > 0


def test_a_completed_held_run_writes_no_interrupted_receipt(tmp_path: Path) -> None:
    """A run that finishes on its own leaves no interruption behind.

    Without this a handler that unconditionally wrote ``timed_out`` would
    satisfy the test above while making every ordinary run look terminated.
    """
    result = tmp_path / "held-run.log"
    returncode, receipt = pytest_slot._run_held(
        [sys.executable, "-c", "pass"],
        cwd=str(tmp_path),
        env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/home/nobody")},
        stdout=None,
        on_exit=lambda: None,
        telemetry_path=tmp_path / "telemetry.json",
        result_path=result,
    )

    assert returncode == 0
    assert receipt["status"] == "success"
    assert not pytest_slot._slot_result_path(result).exists()


def test_a_held_run_defers_when_admission_has_no_worker_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: None)
    monkeypatch.setattr(pytest_slot, "charge_profile_for", lambda _env: (worker_memory.ChargeProfile(1, 1, 1), 1))
    monkeypatch.setattr(
        pytest_slot,
        "resize_worker_argument",
        lambda argv, **_kwargs: (
            list(argv),
            {"admission": "resource_not_ready", "workers": 0, "requested_workers": 1},
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("resource_not_ready must not launch pytest"),
    )
    result = tmp_path / "held-result.log"

    returncode, receipt = pytest_slot._run_held(
        ["pytest", "tests"],
        cwd=str(tmp_path),
        env={},
        stdout=None,
        on_exit=lambda: None,
        result_path=result,
    )

    assert returncode == 75
    assert receipt["status"] == "deferred"
    assert receipt["diagnosis"] == "resource_not_ready"
    assert receipt["sizing"]["workers"] == 0
    assert json.loads(pytest_slot._slot_result_path(result).read_text(encoding="utf-8")) == receipt


def test_a_queued_slot_defers_before_starting_pytest_when_admission_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: None)
    monkeypatch.setattr(pytest_slot, "charge_profile_for", lambda _env: (worker_memory.ChargeProfile(1, 1, 1), 1))
    monkeypatch.setattr(
        pytest_slot,
        "resize_worker_argument",
        lambda argv, **_kwargs: (
            list(argv),
            {"admission": "resource_not_ready", "workers": 0, "requested_workers": 1},
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("resource_not_ready must not launch pytest"),
    )
    launch = tmp_path / "launch.json"
    log = tmp_path / "pytest.log"
    launch.write_text(
        json.dumps(
            {
                "argv": ["pytest", "tests"],
                "working_directory": str(tmp_path),
                "environment": {},
                "log_path": str(log),
            }
        ),
        encoding="utf-8",
    )

    returncode = pytest_slot._run_launch(launch)

    receipt = json.loads(pytest_slot._slot_result_path(log).read_text(encoding="utf-8"))
    assert returncode == 75
    assert receipt["status"] == "deferred"
    assert receipt["diagnosis"] == "resource_not_ready"
    assert receipt["sizing"]["workers"] == 0


def test_live_reservations_are_released_when_pytest_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools.pytest_memory_admission import AdmissionLedger

    ledger = AdmissionLedger(tmp_path / "admission")
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: ledger)
    monkeypatch.setattr(pytest_slot, "charge_profile_for", lambda _env: (worker_memory.ChargeProfile(1, 1, 1), 1))
    monkeypatch.setattr(
        pytest_slot,
        "resize_worker_argument",
        lambda argv, **_kwargs: (
            list(argv),
            {
                "admission": "admitted",
                "cgroup_directory": "/",
                "limiting_cgroups": ["/"],
                "predicted_charge_mib": 3.0,
                "workers": 1,
                "requested_workers": 1,
            },
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("pytest missing")),
    )

    with pytest.raises(FileNotFoundError, match="pytest missing"):
        pytest_slot._run_held(
            ["pytest", "tests"],
            cwd=str(tmp_path),
            env={},
            stdout=None,
            on_exit=lambda: None,
        )

    assert not ledger._path(os.getpid()).exists()


def test_launch_reservation_is_released_when_telemetry_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools.pytest_memory_admission import AdmissionLedger

    ledger = AdmissionLedger(tmp_path / "admission")
    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: ledger)
    monkeypatch.setattr(pytest_slot, "charge_profile_for", lambda _env: (worker_memory.ChargeProfile(1, 1, 1), 1))
    monkeypatch.setattr(
        pytest_slot,
        "resize_worker_argument",
        lambda argv, **_kwargs: (
            list(argv),
            {
                "admission": "admitted",
                "cgroup_directory": "/",
                "limiting_cgroups": ["/"],
                "predicted_charge_mib": 3.0,
                "workers": 1,
                "requested_workers": 1,
            },
        ),
    )
    monkeypatch.setattr(
        pytest_slot,
        "_persist_telemetry_seed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("telemetry path unavailable")),
    )
    launch = tmp_path / "launch.json"
    launch.write_text(
        json.dumps(
            {
                "argv": ["pytest", "tests"],
                "working_directory": str(tmp_path),
                "environment": {},
                "log_path": str(tmp_path / "pytest.log"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(OSError, match="telemetry path unavailable"):
        pytest_slot._run_launch(launch)

    assert not ledger._path(os.getpid()).exists()


def test_held_launch_reaps_child_if_sampler_construction_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[subprocess.Popen[Any]] = []
    real_popen = subprocess.Popen

    def capture(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: None)
    monkeypatch.setattr(pytest_slot, "charge_profile_for", lambda _env: (worker_memory.ChargeProfile(1, 1, 1), 1))
    monkeypatch.setattr(
        pytest_slot,
        "resize_worker_argument",
        lambda argv, **_kwargs: (list(argv), {"admission": "admitted", "workers": 1}),
    )
    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(
        pytest_slot,
        "ProcessGroupMemorySampler",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sampler construction failed")),
    )

    with pytest.raises(RuntimeError, match="sampler construction failed"):
        pytest_slot._run_held(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(tmp_path),
            env={"PATH": os.environ["PATH"]},
            stdout=None,
            on_exit=lambda: None,
        )

    assert len(started) == 1
    assert started[0].poll() is not None


def test_held_launch_stops_partial_sampler_if_sampler_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[subprocess.Popen[Any]] = []
    stopped: list[bool] = []
    real_popen = subprocess.Popen

    def capture(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    class BrokenSampler:
        def start(self) -> None:
            raise RuntimeError("sampler start failed")

        def stop(self) -> dict[str, Any]:
            stopped.append(True)
            return {}

    monkeypatch.setattr(pytest_slot, "admission_ledger", lambda _env: None)
    monkeypatch.setattr(pytest_slot, "charge_profile_for", lambda _env: (worker_memory.ChargeProfile(1, 1, 1), 1))
    monkeypatch.setattr(
        pytest_slot,
        "resize_worker_argument",
        lambda argv, **_kwargs: (list(argv), {"admission": "admitted", "workers": 1}),
    )
    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(pytest_slot, "ProcessGroupMemorySampler", lambda *_args, **_kwargs: BrokenSampler())

    with pytest.raises(RuntimeError, match="sampler start failed"):
        pytest_slot._run_held(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(tmp_path),
            env={"PATH": os.environ["PATH"]},
            stdout=None,
            on_exit=lambda: None,
        )

    assert len(started) == 1
    assert started[0].poll() is not None
    assert stopped == [True]


def test_queued_launch_restores_signal_handlers_if_ledger_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(
        json.dumps(
            {
                "argv": ["pytest", "tests"],
                "working_directory": str(tmp_path),
                "environment": {},
                "log_path": str(tmp_path / "pytest.log"),
            }
        ),
        encoding="utf-8",
    )
    prior = {number: signal.getsignal(number) for number in pytest_slot.REAPED_SIGNALS}
    monkeypatch.setattr(
        pytest_slot,
        "admission_ledger",
        lambda _env: (_ for _ in ()).throw(RuntimeError("ledger setup failed")),
    )

    with pytest.raises(RuntimeError, match="ledger setup failed"):
        pytest_slot._run_launch(launch)

    assert {number: signal.getsignal(number) for number in pytest_slot.REAPED_SIGNALS} == prior


def test_a_group_left_with_only_zombies_counts_as_reaped() -> None:
    """A process group whose only members are unreaped zombies is gone.

    Anti-vacuity (#5708): decide liveness with ``killpg(pgid, 0)``, which
    succeeds on a zombie, and the reap spends both escalation graces and
    reports the group as surviving -- the signal handler then outlives its
    caller's stop deadline.
    """
    import ctypes

    from devtools.pytest_slot import _group_reaped

    if not Path("/proc").is_dir():
        pytest.skip("zombie membership is read from /proc")
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_child_subreaper = 36
    if libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
        pytest.skip("cannot become a child subreaper here")
    orphan = 0
    try:
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, sys, time\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    os._exit(0)\n"
                "print(pid, flush=True)\n"
                "time.sleep(60)\n",
            ],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        assert leader.stdout is not None
        orphan = int(leader.stdout.readline())
        leader.kill()
        leader.wait(timeout=5)
        # The exited grandchild now belongs to this (never-waiting) subreaper;
        # a liveness check that counts it reports the group as surviving.
        assert _group_reaped(leader.pid)
    finally:
        libc.prctl(pr_set_child_subreaper, 0, 0, 0, 0)
        if orphan:
            with contextlib.suppress(ChildProcessError):
                os.waitpid(orphan, 0)


def test_verify_never_inherits_the_focused_charge_profile() -> None:
    """Broad verification is sized by the corpus model, whatever the caller exported.

    Anti-vacuity (Codex P2, #5708): keep an ambient focused marker and corpus
    workers are admitted at the focused per-worker budget and ceiling.
    """
    from devtools import verify
    from devtools.worker_memory import CHARGE_PROFILE_ENV

    env = {CHARGE_PROFILE_ENV: "focused"}
    verify._normalize_managed_pytest_environment(env)

    assert CHARGE_PROFILE_ENV not in env
