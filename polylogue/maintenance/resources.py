"""Resource isolation policy for foreground maintenance work."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from polylogue.core.json import JSONDocument, json_document

ResourceMode = Literal["auto", "scope", "process", "off"]

_GUARD_ENV = "POLYLOGUE_RESOURCE_GUARD"
_CONTEXT_ENV = "POLYLOGUE_RESOURCE_CONTEXT"
_SYSTEMD_GUARD = "systemd-scope"


class ResourceWorkload(str, Enum):
    """Known foreground workloads large enough to need resource isolation."""

    DOCTOR_MAINTENANCE = "doctor-maintenance"
    PIPELINE_RUN = "pipeline-run"
    EMBED_BATCH = "embed-batch"


@dataclass(frozen=True, slots=True)
class ResourceBoundaryRequest:
    """Resource-isolation request for one CLI operation."""

    workload: ResourceWorkload
    heavy: bool
    reason: str
    requested_mode: ResourceMode


@dataclass(frozen=True, slots=True)
class ResourceBoundaryReport:
    """Machine-readable result of applying or skipping resource isolation."""

    workload: str
    requested_mode: ResourceMode
    effective_mode: str
    status: str
    detail: str
    heavy: bool

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "workload": self.workload,
                "requested_mode": self.requested_mode,
                "effective_mode": self.effective_mode,
                "status": self.status,
                "detail": self.detail,
                "heavy": self.heavy,
            }
        )


def normalize_resource_mode(value: str | None) -> ResourceMode:
    """Normalize CLI/env resource mode strings."""

    if value is None or value == "":
        return "auto"
    normalized = value.strip().lower()
    if normalized in {"auto", "scope", "process", "off"}:
        return normalized  # type: ignore[return-value]
    msg = f"unknown resource mode: {value!r}"
    raise ValueError(msg)


def doctor_resource_request(
    *,
    repair: bool,
    cleanup: bool,
    preview: bool,
    resource_mode: ResourceMode,
) -> ResourceBoundaryRequest:
    """Classify doctor maintenance as heavy only when it mutates the archive."""

    heavy = (repair or cleanup) and not preview
    reason = "mutating doctor maintenance" if heavy else "doctor preview/readiness only"
    return ResourceBoundaryRequest(
        workload=ResourceWorkload.DOCTOR_MAINTENANCE,
        heavy=heavy,
        reason=reason,
        requested_mode=resource_mode,
    )


def pipeline_resource_request(
    *,
    stage_sequence: tuple[str, ...],
    preview: bool,
    watch: bool,
    embed_batch: bool,
    resource_mode: ResourceMode,
) -> ResourceBoundaryRequest:
    """Classify pipeline stages that can scan or rewrite large archive state."""

    heavy_stages = {"materialize", "render", "site", "index", "publish"}
    heavy = not preview and (watch or embed_batch or bool(heavy_stages.intersection(stage_sequence)))
    if embed_batch:
        workload = ResourceWorkload.EMBED_BATCH
        reason = "batch embedding writes"
    else:
        workload = ResourceWorkload.PIPELINE_RUN
        reason = "heavy pipeline stages" if heavy else "light or preview pipeline stages"
    return ResourceBoundaryRequest(
        workload=workload,
        heavy=heavy,
        reason=reason,
        requested_mode=resource_mode,
    )


def apply_resource_boundary(
    request: ResourceBoundaryRequest,
    *,
    argv: tuple[str, ...] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResourceBoundaryReport:
    """Apply resource isolation for the current process or re-exec in a scope.

    In normal CLI execution, ``auto`` prefers a user systemd transient scope
    so the entire process tree inherits CPU/IO controls. If a scope is not
    available, it falls back to process-local nice/ionice demotion. Unit tests
    intentionally get a skipped report unless they call lower-level helpers
    directly, avoiding accidental priority changes to the pytest process.
    """

    env = environ if environ is not None else os.environ
    if not request.heavy:
        return _report(request, "not_applicable", "skipped", request.reason)
    if request.requested_mode == "off":
        return _report(request, "off", "skipped", "resource isolation disabled by operator")
    if env.get(_GUARD_ENV) == _SYSTEMD_GUARD:
        detail = "running inside Polylogue systemd transient scope"
        return _report(request, "systemd_scope", "applied", detail)
    if env.get("PYTEST_CURRENT_TEST"):
        return _report(request, "test_harness", "skipped", "test harness; no process priority changes applied")

    command_argv = tuple(argv or sys.argv)
    if request.requested_mode in {"auto", "scope"} and _systemd_scope_available(env):
        try:
            _run_in_systemd_scope(request, command_argv, env)
        except ScopeLaunchUnavailableError as exc:
            if request.requested_mode == "scope":
                return _report(request, "unavailable", "unavailable", str(exc))
        else:  # pragma: no cover - _run_in_systemd_scope exits the process
            raise AssertionError("systemd scope launch returned without exiting")

    if request.requested_mode in {"auto", "process"}:
        return _apply_process_demotion(request)
    return _report(request, "unavailable", "unavailable", "no supported resource isolation mechanism was available")


class ScopeLaunchUnavailableError(RuntimeError):
    """Raised when a systemd transient scope cannot be launched."""


def _systemd_scope_available(env: Mapping[str, str]) -> bool:
    if platform.system() != "Linux":
        return False
    if shutil.which("systemd-run") is None:
        return False
    return bool(env.get("XDG_RUNTIME_DIR"))


def _run_in_systemd_scope(
    request: ResourceBoundaryRequest,
    argv: tuple[str, ...],
    env: Mapping[str, str],
) -> None:
    if not argv:
        raise ScopeLaunchUnavailableError("missing process argv for systemd scope re-exec")
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--working-directory={Path.cwd()}",
        "--property=Nice=19",
        "--property=IOSchedulingClass=idle",
        "--property=IOWeight=10",
        "--property=CPUWeight=10",
        f"--setenv={_GUARD_ENV}={_SYSTEMD_GUARD}",
        f"--setenv={_CONTEXT_ENV}={request.workload.value}",
        "--",
        *argv,
    ]
    try:
        completed = subprocess.run(command, env=dict(env), check=False, capture_output=True, text=True)
    except OSError as exc:
        raise ScopeLaunchUnavailableError(f"could not launch systemd transient scope: {type(exc).__name__}") from exc
    if completed.returncode == 0:
        raise SystemExit(0)
    raise ScopeLaunchUnavailableError(
        f"systemd-run exited {completed.returncode}: " + (completed.stderr or completed.stdout or "").strip()[:200]
    )


def _apply_process_demotion(request: ResourceBoundaryRequest) -> ResourceBoundaryReport:
    applied: list[str] = []
    errors: list[str] = []

    try:
        current_nice = os.nice(0)
        increment = max(0, 19 - current_nice)
        if increment:
            os.nice(increment)
        applied.append("nice=19")
    except OSError as exc:
        errors.append(f"nice unavailable: {type(exc).__name__}")

    ionice = shutil.which("ionice")
    if ionice is None:
        errors.append("ionice unavailable")
    else:
        try:
            completed = subprocess.run(
                [ionice, "-c", "3", "-p", str(os.getpid())],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            errors.append(f"ionice failed: {type(exc).__name__}")
        else:
            if completed.returncode == 0:
                applied.append("ionice=idle")
            else:
                errors.append(f"ionice exited {completed.returncode}")

    if applied:
        detail = f"applied process-local demotion ({', '.join(applied)})"
        if errors:
            detail = f"{detail}; {'; '.join(errors)}"
        return _report(request, "process_demoted", "applied", detail)
    return _report(request, "unavailable", "unavailable", "; ".join(errors) or "process demotion unavailable")


def _report(
    request: ResourceBoundaryRequest,
    effective_mode: str,
    status: str,
    detail: str,
) -> ResourceBoundaryReport:
    return ResourceBoundaryReport(
        workload=request.workload.value,
        requested_mode=request.requested_mode,
        effective_mode=effective_mode,
        status=status,
        detail=detail,
        heavy=request.heavy,
    )


__all__ = [
    "ResourceBoundaryReport",
    "ResourceBoundaryRequest",
    "ResourceMode",
    "ResourceWorkload",
    "ScopeLaunchUnavailableError",
    "apply_resource_boundary",
    "doctor_resource_request",
    "normalize_resource_mode",
    "pipeline_resource_request",
]
