"""Recovery read-view handler."""

from __future__ import annotations

from typing import cast

from polylogue.cli.read_views.base import ReadViewInvocation, ReadViewRecoveryOptions, deliver_content
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def run_read_recovery(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render the deterministic recovery digest for one archived session."""

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.shared.helper_support import fail
    from polylogue.cli.shared.machine_errors import success
    from polylogue.insights.transforms import RecoveryReportPreset
    from polylogue.surfaces.payloads import model_json_document

    del request
    assert invocation.session_id is not None
    options = cast(ReadViewRecoveryOptions, invocation.options or ReadViewRecoveryOptions())
    if options.report is not None:
        if options.report == "work-packet" and invocation.output_format == "json":
            packet = run_coroutine_sync(env.polylogue.recovery_work_packet(invocation.session_id))
            if packet is None:
                fail("read", f"Session not found: {invocation.session_id}")
            payload = success({"recovery_work_packet": model_json_document(packet, exclude_none=True)}).to_json()
            deliver_content(env, payload + "\n", destination=invocation.destination, out_path=invocation.out_path)
            return
        rendered_report = run_coroutine_sync(
            env.polylogue.recovery_report(
                invocation.session_id,
                cast(RecoveryReportPreset, options.report),
            )
        )
        if rendered_report is None:
            fail("read", f"Session not found: {invocation.session_id}")
        deliver_content(env, rendered_report, destination=invocation.destination, out_path=invocation.out_path)
        return
    digest = run_coroutine_sync(env.polylogue.recovery_digest(invocation.session_id))
    if digest is None:
        fail("read", f"Session not found: {invocation.session_id}")
    if invocation.output_format == "json":
        payload = success({"recovery": model_json_document(digest, exclude_none=True)}).to_json()
        deliver_content(env, payload + "\n", destination=invocation.destination, out_path=invocation.out_path)
        return
    deliver_content(env, digest.resume_markdown, destination=invocation.destination, out_path=invocation.out_path)


__all__ = ["run_read_recovery"]
