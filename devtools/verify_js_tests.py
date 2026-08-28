"""Run the JavaScript test suites of every Node package in the repository.

The repository ships two Node packages whose suites guard real product
behaviour: the browser extension's capture path and the WebUI's typed
contracts. Neither is reachable from pytest, so without this gate a red
JavaScript suite is invisible to `devtools verify`.

Absent dependencies are provisioned, never skipped: both packages commit a
lockfile, so `npm ci` is deterministic and a gate that installs what it needs
is real in every checkout rather than only where someone ran npm by hand.
A gate that reports green for a suite it did not run is worse than no gate.

CI carries no Node runtime and does not gate these suites (polylogue-7b45d).
There it reports `not-run-in-ci` rather than a pass, so a CI log states that
the JavaScript suites did not run instead of implying they succeeded.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devtools import repo_root
from polylogue.runtime import available_cpus as _available_cpus

#: Packages whose `npm test` script participates in the gate, in run order.
JS_PACKAGES: tuple[str, ...] = ("browser-extension", "webui")

#: The extension suite's own default worker count (see its vitest.config.js).
DEFAULT_EXTENSION_TEST_WORKERS = 4

#: cgroup v2 and v1 CPU quota files, retained as seams for gate tests.
_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def available_cpus() -> int | None:
    """Compatibility wrapper around the shared process CPU budget."""
    affinity = None
    with contextlib.suppress(AttributeError, OSError):
        affinity = len(os.sched_getaffinity(0))
    return _available_cpus(
        cpu_count=os.cpu_count(),
        affinity=affinity,
        v2_path=_CGROUP_V2_CPU_MAX,
        v1_quota_path=_CGROUP_V1_QUOTA,
        v1_period_path=_CGROUP_V1_PERIOD,
    )


def extension_test_workers(cpu_count: int | None) -> int:
    """Workers to request, never more than the CPUs actually available.

    Several extension tests drive real timers and a backfill coordinator whose
    recovery must settle before the case asserts. Over-subscribing workers
    starves them and they fail on timing rather than behaviour, so the gate
    caps the suite's own default at the visible CPU count.
    """
    if not cpu_count or cpu_count < 1:
        return 1
    return max(1, min(DEFAULT_EXTENSION_TEST_WORKERS, cpu_count))


def _suite_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault(
        "POLYLOGUE_EXTENSION_TEST_WORKERS",
        str(extension_test_workers(available_cpus())),
    )
    return env


@dataclass(frozen=True, slots=True)
class PackageResult:
    package: str
    status: str
    returncode: int | None
    output: str
    remedy: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "green"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "package": self.package,
            "status": self.status,
            "returncode": self.returncode,
            "output": self.output,
        }
        if self.remedy is not None:
            payload["remedy"] = self.remedy
        return payload


def _run(command: list[str], *, cwd: Path, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        env=_suite_env(),
        timeout=timeout,
    )


def _npm_path() -> str | None:
    """Seam for locating the npm binary."""
    return shutil.which("npm")


#: Status reported when CI cannot run the suites. Never a pass, never silent.
NOT_RUN_IN_CI = "not-run-in-ci"


def ci_environment() -> str | None:
    """The CI system running this process, or None when run locally."""
    if os.environ.get("CIRCLECI") == "true":
        return "circleci"
    if os.environ.get("CI") in {"true", "1"}:
        return "ci"
    return None


def _install_command(package_dir: Path) -> list[str]:
    """`npm ci` where a lockfile pins the tree, `npm install` otherwise."""
    if (package_dir / "package-lock.json").is_file():
        return ["npm", "ci"]
    return ["npm", "install"]


#: A provisioning run that outlives this is wedged, not slow.
INSTALL_TIMEOUT_SECONDS = 900

#: Written inside node_modules so a tree provisioned from a different lockfile
#: is reinstalled instead of silently tested against stale dependencies.
_STAMP_NAME = ".polylogue-provisioned"


def _lock_fingerprint(package_dir: Path) -> str | None:
    lock = package_dir / "package-lock.json"
    try:
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        return None


def needs_provisioning(package_dir: Path) -> bool:
    """True when node_modules is absent, or was built from another lockfile."""
    if not (package_dir / "node_modules").is_dir():
        return True
    fingerprint = _lock_fingerprint(package_dir)
    if fingerprint is None:
        # No lockfile to compare against; an existing tree is all we can ask for.
        return False
    try:
        return (package_dir / "node_modules" / _STAMP_NAME).read_text().strip() != fingerprint
    except OSError:
        return True


def _provision(package: str, package_dir: Path) -> PackageResult | None:
    """Install dependencies; return a blocked result if that fails."""
    command = _install_command(package_dir)
    print(f"verify js-tests: {package}: installing dependencies ({' '.join(command)})...", flush=True)
    started = time.monotonic()
    try:
        completed = _run(command, cwd=package_dir, timeout=INSTALL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return PackageResult(
            package=package,
            status="blocked-deps",
            returncode=None,
            output=f"{' '.join(command)} exceeded {INSTALL_TIMEOUT_SECONDS}s in {package}.",
            remedy=f"cd {package} && {' '.join(command)}",
        )
    elapsed = time.monotonic() - started
    if completed.returncode == 0:
        print(f"verify js-tests: {package}: dependencies installed in {elapsed:.1f}s", flush=True)
        fingerprint = _lock_fingerprint(package_dir)
        if fingerprint is not None:
            with contextlib.suppress(OSError):
                (package_dir / "node_modules" / _STAMP_NAME).write_text(fingerprint, encoding="utf-8")
        return None
    # Retryable: a failed install is evidence about the environment (network,
    # registry, a lockfile out of step with package.json), never about the
    # product code the suite covers. It still fails the gate -- the suite did
    # not run -- but it is named as a dependency problem, not a red suite.
    return PackageResult(
        package=package,
        status="blocked-deps",
        returncode=completed.returncode,
        output=(completed.stdout + completed.stderr).strip(),
        remedy=f"cd {package} && {' '.join(command)}",
    )


def _check_package(package: str, *, root: Path, install: bool) -> PackageResult:
    package_dir = root / package
    if not (package_dir / "package.json").is_file():
        return PackageResult(
            package=package,
            status="blocked-env",
            returncode=None,
            output=f"{package}/package.json is missing; the package catalog and the tree disagree.",
        )

    if install or needs_provisioning(package_dir):
        blocked = _provision(package, package_dir)
        if blocked is not None:
            return blocked

    completed = _run(["npm", "test"], cwd=package_dir)
    return PackageResult(
        package=package,
        status="green" if completed.returncode == 0 else "red",
        returncode=completed.returncode,
        output=(completed.stdout + completed.stderr).strip(),
    )


def run_js_tests(*, root: Path, packages: tuple[str, ...], install: bool) -> list[PackageResult]:
    if _npm_path() is None:
        return [
            PackageResult(
                package=package,
                status="blocked-env",
                returncode=None,
                output="npm is not on PATH, so no JavaScript suite could run.",
                remedy="install Node.js 22 and npm",
            )
            for package in packages
        ]
    return [_check_package(package, root=root, install=install) for package in packages]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result envelope.")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Reinstall dependencies even when the tree is already current.",
    )
    parser.add_argument(
        "--package",
        action="append",
        choices=JS_PACKAGES,
        help="Limit the run to one package (repeatable). Defaults to every package.",
    )
    args = parser.parse_args(argv)

    packages = tuple(args.package) if args.package else JS_PACKAGES
    env = _suite_env()
    results = run_js_tests(root=repo_root(), packages=packages, install=args.install)
    status = "green" if all(result.ok for result in results) else "red"

    # A suite that genuinely ran and failed stays red everywhere. Only the
    # cannot-run case is downgraded, and only under CI, which does not gate
    # these suites -- and it is downgraded to a named status, not to a pass.
    ci = ci_environment()
    blocked = all(result.status.startswith("blocked-") for result in results)
    if ci is not None and blocked:
        status = NOT_RUN_IN_CI
    payload = {
        "command": "devtools verify js-tests",
        "status": status,
        "ci": ci,
        # Recorded because a starved suite fails on timing, not behaviour, and
        # the budget is the first thing to check when it does.
        "available_cpus": available_cpus(),
        "extension_test_workers": int(env["POLYLOGUE_EXTENSION_TEST_WORKERS"]),
        "packages": [result.to_payload() for result in results],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            if not result.ok:
                print(result.output)
            line = f"verify js-tests: {result.package}: {result.status}"
            if result.remedy is not None:
                line += f" -- remedy: {result.remedy}"
            print(line)
        if status == NOT_RUN_IN_CI:
            print(
                f"verify js-tests: {NOT_RUN_IN_CI}: the JavaScript suites DID NOT RUN "
                f"on {ci}. They are gated locally by `devtools verify --quick` and "
                f"pre-push, not here. This is not a pass -- see polylogue-7b45d."
            )
        print(
            f"verify js-tests: {status} "
            f"(cpus={payload['available_cpus']} "
            f"extension_workers={payload['extension_test_workers']})"
        )
    return 0 if status in {"green", NOT_RUN_IN_CI} else 1


if __name__ == "__main__":
    raise SystemExit(main())
