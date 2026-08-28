"""The JavaScript gate must refuse loudly rather than skip a suite it cannot run.

Anti-vacuity: make `_check_package` fall through to `npm test` when
`node_modules` is absent, or let `run_js_tests` return a green result when npm
is missing, and `test_missing_node_modules_is_a_blocked_failure` /
`test_missing_npm_blocks_every_package` go red.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from devtools import repo_root, verify_js_tests
from devtools.command_catalog import COMMANDS


def _package(root: Path, name: str, *, with_lock: bool = True, with_modules: bool = True) -> Path:
    package_dir = root / name
    (package_dir / "src").mkdir(parents=True)
    (package_dir / "package.json").write_text('{"name": "x", "scripts": {"test": "true"}}', encoding="utf-8")
    if with_lock:
        (package_dir / "package-lock.json").write_text("{}", encoding="utf-8")
    if with_modules:
        (package_dir / "node_modules").mkdir()
    return package_dir


def test_missing_node_modules_is_a_blocked_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _package(tmp_path, "webui", with_modules=False)
    monkeypatch.setattr("devtools.verify_js_tests._npm_path", lambda: "/usr/bin/npm")

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("no suite may be executed when its dependencies are absent")

    monkeypatch.setattr("devtools.verify_js_tests._run", _forbidden)

    [result] = verify_js_tests.run_js_tests(root=tmp_path, packages=("webui",), install=False)

    assert result.status == "blocked-deps"
    assert not result.ok
    assert result.remedy == "cd webui && npm ci"


def test_missing_npm_blocks_every_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("devtools.verify_js_tests._npm_path", lambda: None)

    results = verify_js_tests.run_js_tests(root=tmp_path, packages=("browser-extension", "webui"), install=False)

    assert [result.status for result in results] == ["blocked-env", "blocked-env"]
    assert all(not result.ok for result in results)


def test_a_red_suite_fails_the_gate_and_reports_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _package(tmp_path, "webui")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("CIRCLECI", raising=False)
    monkeypatch.setattr("devtools.verify_js_tests.repo_root", lambda: tmp_path)
    monkeypatch.setattr("devtools.verify_js_tests._npm_path", lambda: "/usr/bin/npm")
    monkeypatch.setattr(
        "devtools.verify_js_tests._run",
        lambda *_a, **_k: subprocess.CompletedProcess(["npm", "test"], 1, "1 failed", ""),
    )

    exit_code = verify_js_tests.main(["--json", "--package", "webui"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "red"
    assert payload["packages"][0]["status"] == "red"


def test_install_uses_npm_ci_only_with_a_lockfile(tmp_path: Path) -> None:
    locked = _package(tmp_path, "webui")
    unlocked = _package(tmp_path, "browser-extension", with_lock=False)

    assert verify_js_tests._install_command(locked) == ["npm", "ci"]
    assert verify_js_tests._install_command(unlocked) == ["npm", "install"]


def test_every_declared_package_exists_in_the_repository() -> None:
    root = repo_root()
    for package in verify_js_tests.JS_PACKAGES:
        assert (root / package / "package.json").is_file(), package


def test_extension_workers_never_exceed_available_cpus() -> None:
    # Over-subscribing starves the extension's timing-sensitive backfill tests:
    # 4 workers on a 2-CPU runner failed ~5 runs in 6, 1-2 workers passed 8 of 8.
    assert verify_js_tests.extension_test_workers(2) == 2
    assert verify_js_tests.extension_test_workers(1) == 1
    # Plentiful CPUs keep the suite's own declared default rather than raising it.
    assert verify_js_tests.extension_test_workers(24) == verify_js_tests.DEFAULT_EXTENSION_TEST_WORKERS
    # An unknown CPU count must not become an unbounded worker count.
    assert verify_js_tests.extension_test_workers(None) == 1
    assert verify_js_tests.extension_test_workers(0) == 1


def test_cgroup_quota_beats_the_host_cpu_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A CircleCI medium runner reports 24 host CPUs while its cgroup allows 2.
    quota = tmp_path / "cpu.max"
    quota.write_text("200000 100000", encoding="utf-8")
    monkeypatch.setattr("devtools.verify_js_tests._CGROUP_V2_CPU_MAX", quota)
    monkeypatch.setattr("devtools.verify_js_tests.os.cpu_count", lambda: 24)
    monkeypatch.setattr("devtools.verify_js_tests.os.sched_getaffinity", lambda _pid: set(range(24)))

    assert verify_js_tests.available_cpus() == 2
    assert verify_js_tests.extension_test_workers(verify_js_tests.available_cpus()) == 2


def test_unlimited_cgroup_quota_falls_back_to_the_host_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000", encoding="utf-8")
    monkeypatch.setattr("devtools.verify_js_tests._CGROUP_V2_CPU_MAX", quota)
    monkeypatch.setattr("devtools.verify_js_tests._CGROUP_V1_QUOTA", tmp_path / "absent")
    monkeypatch.setattr("devtools.verify_js_tests.os.cpu_count", lambda: 8)
    monkeypatch.setattr("devtools.verify_js_tests.os.sched_getaffinity", lambda _pid: set(range(8)))

    assert verify_js_tests.available_cpus() == 8


def test_suite_env_pins_the_worker_cap_for_the_extension() -> None:
    env = verify_js_tests._suite_env()
    assert int(env["POLYLOGUE_EXTENSION_TEST_WORKERS"]) >= 1


def test_an_explicit_worker_budget_wins_over_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI states its own budget because container CPU limits are not reliably
    # visible; detection must not override what the runner declares.
    monkeypatch.setenv("POLYLOGUE_EXTENSION_TEST_WORKERS", "2")
    monkeypatch.setattr("devtools.verify_js_tests.os.cpu_count", lambda: 24)

    assert verify_js_tests._suite_env()["POLYLOGUE_EXTENSION_TEST_WORKERS"] == "2"


def test_gate_is_registered_in_the_command_catalog() -> None:
    assert "verify js-tests" in COMMANDS


def test_ci_is_detected_from_the_conventional_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("CIRCLECI", raising=False)
    assert verify_js_tests.ci_environment() is None
    monkeypatch.setenv("CIRCLECI", "true")
    assert verify_js_tests.ci_environment() == "circleci"
    monkeypatch.delenv("CIRCLECI")
    monkeypatch.setenv("CI", "true")
    assert verify_js_tests.ci_environment() == "ci"


def test_ci_names_the_unrun_suites_instead_of_passing_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # CI does not gate these suites, but its log must say so in words rather
    # than report a green the suites never earned.
    _package(tmp_path, "webui", with_modules=False)
    monkeypatch.setenv("CIRCLECI", "true")
    monkeypatch.setattr("devtools.verify_js_tests.repo_root", lambda: tmp_path)
    monkeypatch.setattr("devtools.verify_js_tests._npm_path", lambda: "/usr/bin/npm")

    exit_code = verify_js_tests.main(["--package", "webui"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert verify_js_tests.NOT_RUN_IN_CI in output
    assert "DID NOT RUN" in output
    assert "green" not in output


def test_ci_json_reports_the_named_status_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _package(tmp_path, "webui", with_modules=False)
    monkeypatch.setenv("CIRCLECI", "true")
    monkeypatch.setattr("devtools.verify_js_tests.repo_root", lambda: tmp_path)
    monkeypatch.setattr("devtools.verify_js_tests._npm_path", lambda: "/usr/bin/npm")

    verify_js_tests.main(["--json", "--package", "webui"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == verify_js_tests.NOT_RUN_IN_CI
    assert payload["ci"] == "circleci"
    assert payload["packages"][0]["status"] == "blocked-deps"


def test_a_suite_that_ran_and_failed_stays_red_under_ci(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the cannot-run case is downgraded. A real failure is still a failure.
    _package(tmp_path, "webui")
    monkeypatch.setenv("CIRCLECI", "true")
    monkeypatch.setattr("devtools.verify_js_tests.repo_root", lambda: tmp_path)
    monkeypatch.setattr("devtools.verify_js_tests._npm_path", lambda: "/usr/bin/npm")
    monkeypatch.setattr(
        "devtools.verify_js_tests._run",
        lambda *_a, **_k: subprocess.CompletedProcess(["npm", "test"], 1, "1 failed", ""),
    )

    assert verify_js_tests.main(["--package", "webui"]) == 1


def test_local_absent_dependencies_still_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CI carve-out must not leak into a developer's machine.
    _package(tmp_path, "webui", with_modules=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("CIRCLECI", raising=False)
    monkeypatch.setattr("devtools.verify_js_tests.repo_root", lambda: tmp_path)
    monkeypatch.setattr("devtools.verify_js_tests._npm_path", lambda: "/usr/bin/npm")

    assert verify_js_tests.main(["--package", "webui"]) == 1
