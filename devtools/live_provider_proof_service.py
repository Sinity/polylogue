"""Declared AgentCTL entrypoint for the shared-Chrome provider proof.

The caller selects exact supported conversations from a private JSON file.
The receiver is fixed by this operation. The Node workflow uses
the Sinnix shared-Chrome control boundary, which opens and parks proof-owned
windows in the existing authenticated browser. The receiver binds a free
loopback port, passes it to the Node workflow, and publishes it in the result.
The runtime (agentctl) remains the authority for admission and exact-head binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from devtools.agentctl_service_context import require_declared_operation_context
from devtools.shared_chrome_lock import shared_chrome_extension_lock
from polylogue.browser_capture.models import validate_capture_envelope
from polylogue.core.enums import BlockType
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.browser_capture import (
    _browser_capture_attachment_content,
    _parsed_blocks_for_turn,
    parse,
)

_RECEIVER_PORT_ENV = "POLYLOGUE_LIVE_PROVIDER_RECEIVER_PORT"
_NODE_PROOF_TIMEOUT_S = 120
_MAX_ERROR_MESSAGE = 512

_PROOF_PHASES = {
    "service_context",
    "inputs",
    "chrome_status",
    "extension_load",
    "chrome_connect",
    "extension_startup",
    "popup_open",
    "popup_connect",
    "receiver_snapshot",
    "pause",
    "revision",
    "desktop_snapshot",
    "popup_bind",
    "provider_preflight",
    "permission_grant",
    "receiver_pairing",
    "provider_open",
    "provider_window",
    "provider_wait",
    "capture",
    "summary",
    "unknown",
}
_PROOF_CATEGORIES = {
    "control_failed",
    "window_precondition_failed",
    "window_rules_failed",
    "window_creation_failed",
    "window_navigation_failed",
    "window_compositor_changed",
    "control_timeout",
    "window_refused",
    "window_visibility_refused",
    "window_unknown",
    "shutdown",
    "revision_mismatch",
    "revision_missing",
    "receiver_configuration_failed",
    "receiver_permission_refused",
    "receiver_handshake_failed",
    "configuration_changed",
    "pause_failed",
    "capture_incomplete",
    "cleanup_failed",
    "operation_failed",
}
_NATIVE_PROGRESS_STAGES = {
    "throttle",
    "pending_header",
    "restore",
    "staging",
    "provider_auth",
    "provider_response",
    "body",
    "header",
    "canonical",
    "publication",
}
_BACKGROUND_PROGRESS_STAGES = {"normalize_admission", "native_prepare", "native_assets", "native_finalize"}
_CLEANUP_STATES = {"not_required", "settled", "failed", "unknown"}
_CLEANUP_KEYS = {"receiver", "permission", "mutations", "targets"}
_CAPTURE_CATEGORIES = {
    "native_capture_unavailable",
    "provider_throttle_authority_unavailable",
    "rate_limited",
    "capture_cancelled",
    "capture_rejected",
    "capture_timed_out",
    "unknown",
}
_BRIDGE_FAILURE_CATEGORIES = _CAPTURE_CATEGORIES | {
    "conversation_api_url_not_found",
    "asset_body_stream_unavailable",
    "receiver_unpaired",
    "capture_staging_unavailable",
    "capture_staging_sender_invalid",
    "capture_staging_request_invalid",
    "capture_staging_owner_mismatch",
    "capture_staging_invalid_ref",
    "capture_staging_sequence_mismatch",
    "capture_staging_records_unavailable",
    "native_acquisition_sequence_invalid",
    "capture_response_metadata_invalid",
    "capture_response_metadata_conflict",
    "capture_staging_incomplete",
    "capture_staging_interrupted",
    "capture_staging_missing_bytes",
}
_SUMMARY_CHECKS = {
    "response_ok",
    "identity_matches",
    "native_digest_valid",
    "native_full",
    "turn_count_valid",
    "attachment_count_valid",
    "artifact_present",
    "receiver_request_present",
}


def _capture_evidence_valid(evidence: object) -> bool:
    if not isinstance(evidence, list) or len(evidence) > 2:
        return False
    providers: set[str] = set()
    for entry in evidence:
        if not isinstance(entry, dict) or set(entry) != {"provider", "category", "bridge", "summary"}:
            return False
        provider, category = entry["provider"], entry["category"]
        if (
            not isinstance(provider, str)
            or provider not in {"chatgpt", "claude-ai"}
            or provider in providers
            or not isinstance(category, str)
            or category not in _CAPTURE_CATEGORIES | {"response_empty", "response_invalid", "deferred", "accepted"}
        ):
            return False
        providers.add(provider)
        bridge, summary = entry["bridge"], entry["summary"]
        if (
            not isinstance(bridge, dict)
            or set(bridge) != {"observed", "accepted", "status", "category", "failure_stage"}
            or type(bridge["observed"]) is not bool
            or (bridge["accepted"] is not None and type(bridge["accepted"]) is not bool)
            or (
                bridge["status"] is not None
                and (type(bridge["status"]) is not int or not 100 <= bridge["status"] <= 599)
            )
            or not isinstance(bridge["failure_stage"], str)
            or bridge["failure_stage"] not in {"admission", "provider_fetch", "staging", "unknown"}
            or (
                bridge["failure_stage"] != "unknown"
                and (bridge["accepted"] is not False or not bridge["observed"] or bridge["category"] == "none")
            )
            or not isinstance(bridge["category"], str)
            or bridge["category"] not in _BRIDGE_FAILURE_CATEGORIES | {"none"}
            or not isinstance(summary, dict)
            or set(summary) != _SUMMARY_CHECKS
            or any(type(value) is not bool for value in summary.values())
        ):
            return False
    return True


class ChildProofError(RuntimeError):
    """A strict public diagnostic from the failed child, never its exception text."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__("live provider proof failed")
        self.report = report
        self.receiver_requests: list[dict[str, object]] = []


def child_failure_report(stdout: str) -> dict[str, Any] | None:
    try:
        report = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    if (
        not isinstance(report, dict)
        or set(report) != {"ok", "error", "cleanup", "native_progress", "capture_evidence"}
        or report["ok"] is not False
        or not _capture_evidence_valid(report["capture_evidence"])
    ):
        return None
    progress = report["native_progress"]
    if (
        not isinstance(progress, list)
        or len(progress) > 6
        or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("stage"), str)
            or not (
                (set(entry) == {"stage", "state"} and entry.get("stage") in _NATIVE_PROGRESS_STAGES)
                or (
                    set(entry) == {"stage", "state", "source"}
                    and entry.get("source") == "background_debug_log"
                    and entry.get("stage") in _BACKGROUND_PROGRESS_STAGES
                )
            )
            or not isinstance(entry["stage"], str)
            or not isinstance(entry["state"], str)
            or entry["state"] not in {"BEGIN", "END"}
            for entry in progress
        )
    ):
        return None
    error, cleanup = report["error"], report["cleanup"]
    if (
        not isinstance(error, dict)
        or set(error) != {"phase", "category"}
        or not isinstance(error["phase"], str)
        or error["phase"] not in _PROOF_PHASES
        or not isinstance(error["category"], str)
        or error["category"] not in _PROOF_CATEGORIES
        or not isinstance(cleanup, dict)
        or set(cleanup) != _CLEANUP_KEYS
        or any(not isinstance(state, str) or state not in _CLEANUP_STATES for state in cleanup.values())
    ):
        return None
    return report


def failed_child(
    stdout: str | bytes | None, receiver_requests: list[dict[str, object]], *, timed_out: bool = False
) -> ChildProofError:
    # TimeoutExpired may carry bytes despite Popen(text=True). Only the same
    # strict child report may cross this boundary; stderr and raw faults cannot.
    if isinstance(stdout, bytes):
        try:
            stdout = stdout.decode("utf-8")
        except UnicodeDecodeError:
            stdout = None
    report = child_failure_report(stdout) if isinstance(stdout, str) else None
    if report is None:
        report = {
            "ok": False,
            "native_progress": [],
            "capture_evidence": [],
            "error": {"phase": "unknown", "category": "control_timeout" if timed_out else "operation_failed"},
            "cleanup": dict.fromkeys(_CLEANUP_KEYS, "unknown"),
        }
    error = ChildProofError(report)
    error.receiver_requests = receiver_requests
    return error


def conversation_targets(path: Path) -> list[dict[str, str]]:
    """Validate explicitly selected conversations before any browser mutation."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("proof requires explicit conversation URLs")
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    routes = {"chatgpt.com": ("chatgpt", "c"), "claude.ai": ("claude", "chat")}
    for value in values:
        if not isinstance(value, str):
            raise ValueError("conversation selection must contain URL strings")
        url = urlsplit(value)
        route = routes.get(url.netloc)
        parts = url.path.split("/")
        if (
            url.scheme != "https"
            or route is None
            or url.query
            or url.fragment
            or len(parts) != 3
            or parts[1] != route[1]
            or not parts[2]
            or not all(c.isascii() and (c.isalnum() or c in "-_") for c in parts[2])
        ):
            raise ValueError("proof requires an exact supported conversation URL")
        provider, _prefix = route
        if provider in seen:
            raise ValueError("select one exact conversation per provider")
        seen.add(provider)
        selected.append({"name": provider, "url": value, "nativeId": parts[2]})
    return selected


def verify_captured_artifact(spool: Path, receipt: dict[str, Any]) -> dict[str, object]:
    """Prove actual admitted bytes through the ordinary canonical parser."""
    reference = receipt.get("artifact_ref")
    if not isinstance(reference, str):
        raise ValueError("proof has no admitted artifact")
    target = (spool / reference).resolve()
    if not target.is_relative_to(spool.resolve()) or not target.is_file():
        raise ValueError("proof artifact is outside its private receiver")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != receipt.get("artifact_sha256"):
            raise ValueError("proof artifact receipt digest mismatch")
        handle.seek(0)
        payload = json.load(handle)
    # Selected live conversations exercise the existing parser. This proof
    # does not establish scalar-independent memory use for arbitrary sessions.
    envelope = validate_capture_envelope(payload)
    if envelope.raw_provider_payload is None:
        raise ValueError("proof has no literal native evidence")
    parsed = parse(payload, "live-provider-proof")
    if len(envelope.session.turns) != len(parsed.messages):
        raise ValueError("proof canonical messages were omitted")
    if (
        envelope.session.title != parsed.title
        or envelope.session.created_at != parsed.created_at
        or envelope.session.updated_at != parsed.updated_at
    ):
        raise ValueError("proof canonical session fields disagree with native evidence")
    for ordinal, (turn, message) in enumerate(zip(envelope.session.turns, parsed.messages, strict=True)):
        if (
            turn.provider_turn_id != message.provider_message_id
            or turn.parent_turn_id != message.parent_message_provider_id
            or turn.role != message.role
            or turn.text != message.text
            or turn.timestamp != message.timestamp
            or turn.ordinal != ordinal
            or [block.model_dump(mode="json") for block in _parsed_blocks_for_turn(turn)]
            != [block.model_dump(mode="json") for block in message.blocks]
        ):
            raise ValueError("proof canonical message or block fields disagree with native evidence")
    if len(envelope.session.attachments) != len(parsed.attachments):
        raise ValueError("proof omitted canonical asset receipts")
    for attachment in envelope.session.attachments:
        outcome = attachment.provider_meta.get("asset_acquisition")
        if not isinstance(outcome, dict) or not isinstance(outcome.get("status"), str) or not outcome["status"]:
            raise ValueError("proof has no terminal asset receipt")
        if outcome["status"] == "acquired":
            content = _browser_capture_attachment_content(attachment)
            if (
                not isinstance(content, bytes)
                or attachment.size_bytes != len(content)
                or attachment.provider_meta.get("content_sha256") != hashlib.sha256(content).hexdigest()
            ):
                raise ValueError("proof acquired asset receipt mismatch")
    if (
        parsed.source_name.value != receipt.get("provider")
        or hashlib.sha256(parsed.provider_session_id.encode()).hexdigest() != receipt.get("provider_session_id_sha256")
        or len(parsed.messages) != receipt.get("turn_count")
        or not parsed.messages
        or len(parsed.attachments) != receipt.get("attachment_count")
    ):
        raise ValueError("canonical artifact does not match the selected capture")
    return {
        "message_count": len(parsed.messages),
        "block_count": sum(len(message.blocks) for message in parsed.messages),
        "attachment_count": len(parsed.attachments),
        "attachment_directions": sorted({attachment.direction or "unknown" for attachment in parsed.attachments}),
        "parent_message_count": sum(message.parent_message_provider_id is not None for message in parsed.messages),
        "tool_use_count": sum(
            block.type is BlockType.TOOL_USE for message in parsed.messages for block in message.blocks
        ),
        "tool_result_count": sum(
            block.type is BlockType.TOOL_RESULT for message in parsed.messages for block in message.blocks
        ),
        "tool_error_count": sum(block.is_error is True for message in parsed.messages for block in message.blocks),
        "tool_outcomes": sorted(
            {
                block.tool_outcome.value
                for message in parsed.messages
                for block in message.blocks
                if block.tool_outcome is not None
            }
        ),
        "content_hash": session_content_hash(parsed),
        "artifact_sha256": digest.hexdigest(),
    }


def run_proof(*, conversations_file: Path, repo_root: Path | None = None) -> dict[str, object]:
    """Run the shared-Chrome workflow against a self-bound loopback receiver."""
    targets = conversation_targets(conversations_file)
    with shared_chrome_extension_lock(timeout_s=_NODE_PROOF_TIMEOUT_S):
        return _run_proof_locked(targets=targets, repo_root=repo_root)


def _run_proof_locked(*, targets: list[dict[str, str]], repo_root: Path | None = None) -> dict[str, object]:
    require_declared_operation_context("live_provider_proof")
    # A second production extension would register content scripts on operator
    # tabs before its ambient-capture pause. Until registration itself has owned
    # target authority, do not load it or replace the operator's native host.
    raise ChildProofError(
        {
            "ok": False,
            "error": {"phase": "extension_load", "category": "provider_target_isolation_unavailable"},
            "cleanup": dict.fromkeys(["receiver", "permission", "mutations", "targets"], "not_required"),
            "native_progress": [],
            "capture_evidence": [],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one bounded JSON result.")
    parser.add_argument(
        "--conversations-file", type=Path, required=True, help="Private JSON array of exact conversation URLs."
    )
    arguments = parser.parse_args(argv)
    try:
        payload: dict[str, Any] = run_proof(conversations_file=arguments.conversations_file)
    except ChildProofError as error:
        payload = {**error.report, "receiver_requests": error.receiver_requests}
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        payload = {
            "ok": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:_MAX_ERROR_MESSAGE]
                if isinstance(error, subprocess.TimeoutExpired)
                else "Live proof failed; private capture data omitted.",
            },
        }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
