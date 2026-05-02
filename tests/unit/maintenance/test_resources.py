"""Resource-boundary policy tests."""

from __future__ import annotations

import subprocess

import pytest

import polylogue.maintenance.resources as resource_module
from polylogue.maintenance.resources import (
    ResourceWorkload,
    apply_resource_boundary,
    doctor_resource_request,
    normalize_resource_mode,
    pipeline_resource_request,
)


def test_doctor_maintenance_classification_distinguishes_preview_from_apply() -> None:
    preview = doctor_resource_request(repair=True, cleanup=False, preview=True, resource_mode="auto")
    apply = doctor_resource_request(repair=True, cleanup=False, preview=False, resource_mode="auto")

    assert preview.workload is ResourceWorkload.DOCTOR_MAINTENANCE
    assert preview.heavy is False
    assert apply.heavy is True


def test_pipeline_classification_marks_bulk_stages_and_embed_batch() -> None:
    parse_only = pipeline_resource_request(
        stage_sequence=("parse",),
        preview=False,
        watch=False,
        embed_batch=False,
        resource_mode="auto",
    )
    materialize = pipeline_resource_request(
        stage_sequence=("parse", "materialize"),
        preview=False,
        watch=False,
        embed_batch=False,
        resource_mode="auto",
    )
    embed_batch = pipeline_resource_request(
        stage_sequence=("embed",),
        preview=False,
        watch=False,
        embed_batch=True,
        resource_mode="auto",
    )

    assert parse_only.heavy is False
    assert materialize.heavy is True
    assert embed_batch.workload is ResourceWorkload.EMBED_BATCH
    assert embed_batch.heavy is True


def test_apply_resource_boundary_reports_test_harness_without_mutating_priority() -> None:
    request = doctor_resource_request(repair=True, cleanup=False, preview=False, resource_mode="auto")

    report = apply_resource_boundary(request, environ={"PYTEST_CURRENT_TEST": "test"})

    assert report.heavy is True
    assert report.effective_mode == "test_harness"
    assert report.status == "skipped"


def test_apply_resource_boundary_reports_explicit_opt_out() -> None:
    request = doctor_resource_request(repair=True, cleanup=False, preview=False, resource_mode="off")

    report = apply_resource_boundary(request, environ={})

    assert report.effective_mode == "off"
    assert report.status == "skipped"
    assert report.to_dict()["requested_mode"] == "off"


def test_scope_mode_reports_systemd_launch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    request = doctor_resource_request(repair=True, cleanup=False, preview=False, resource_mode="scope")

    monkeypatch.setattr(resource_module, "_systemd_scope_available", lambda _env: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args[0], returncode=1, stderr="scope denied"),
    )

    report = apply_resource_boundary(
        request,
        environ={"XDG_RUNTIME_DIR": "/run/user/1000"},
        argv=("polylogue", "doctor", "--repair"),
    )

    assert report.status == "unavailable"
    assert report.effective_mode == "unavailable"
    assert report.detail == "systemd-run exited 1: scope denied"


def test_normalize_resource_mode_rejects_unknown_modes() -> None:
    assert normalize_resource_mode(None) == "auto"

    try:
        normalize_resource_mode("turbo")
    except ValueError as exc:
        assert "unknown resource mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid resource mode to fail")
