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
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlsplit

from devtools.agentctl_service_context import require_declared_operation_context, terminate_process_group
from devtools.isolated_environment import isolated_home_environment
from devtools.native_transport_proof import NativeProofCustodyError, scoped_native_host
from devtools.shared_chrome_lock import shared_chrome_extension_lock
from polylogue.browser_capture.models import _CanonicalNativeTurnWitness, validate_capture_envelope
from polylogue.browser_capture.receiver import load_or_mint_receiver_identity, persist_receiver_token
from polylogue.browser_capture.server import BrowserCaptureHandler, make_server
from polylogue.core.enums import BlockType
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.browser_capture import (
    _browser_capture_attachment_content,
    _parsed_blocks_for_turn,
    parse,
)

_SHARED_CHROME_LOCK_TIMEOUT_S = 120
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
    "capture_start",
    "capture_membership",
    "capture_result",
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
    "provider_isolation_refused",
    "automatic_capture_missing",
    "automatic_capture_pending",
    "automatic_capture_start_failed",
    "capture_listener_invalid",
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
    parsed = parse(payload, "live-provider-proof")
    envelope = validate_capture_envelope(payload, native_witness=_CanonicalNativeTurnWitness(parsed.messages))
    if envelope.raw_provider_payload is None:
        raise ValueError("proof has no literal native evidence")
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


def _retain_proof_evidence(scratch: Path, spool: Path, evidence: Path) -> None:
    """Retain the sealed selected receiver bytes and their original bindings."""
    shutil.copytree(spool, evidence / "browser-capture")
    for name, source in (
        ("constructor-scope.json", scratch / "constructor-scope.json"),
        ("owned-capture-result.json", scratch / "owned-capture-result.json"),
        ("owned-capture-diagnostic.json", scratch / "owned-capture-diagnostic.json"),
        ("proof-binding.json", scratch / "native-transport-proof/extension/proof-binding.json"),
        ("owned-scope.json", scratch / "native-transport-proof/extension/owned-scope.json"),
    ):
        if source.is_file():
            shutil.copy2(source, evidence / name)
    with (evidence / "retained-files.jsonl").open("x", encoding="utf-8") as manifest:
        for source in evidence.rglob("*"):
            if not source.is_file() or source == evidence / "retained-files.jsonl":
                continue
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                while chunk := handle.read(64 * 1024):
                    digest.update(chunk)
            manifest.write(
                json.dumps(
                    {
                        "path": str(source.relative_to(evidence)),
                        "bytes": source.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
                + "\n"
            )


def run_proof(
    *, conversations_file: Path, chrome_user_data_dir: Path, evidence_root: Path, repo_root: Path | None = None
) -> dict[str, object]:
    """Capture only declared conversations through an owned-window runtime."""
    targets = conversation_targets(conversations_file)
    with shared_chrome_extension_lock(timeout_s=_SHARED_CHROME_LOCK_TIMEOUT_S):
        return _run_proof_locked(
            targets=targets, chrome_user_data_dir=chrome_user_data_dir, evidence_root=evidence_root, repo_root=repo_root
        )


def _run_proof_locked(
    *, targets: list[dict[str, str]], chrome_user_data_dir: Path, evidence_root: Path, repo_root: Path | None = None
) -> dict[str, object]:
    require_declared_operation_context("live_provider_proof")
    if not targets:
        raise ValueError("proof requires explicitly owned conversation targets")
    evidence_root = evidence_root.absolute()
    evidence_root.mkdir(mode=0o700)
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    scratch = Path(tempfile.mkdtemp(prefix="polylogue-owned-provider-proof-")).resolve()
    spool, archive = scratch / "browser-capture", scratch / "archive"
    spool.mkdir()
    inherited = dict(os.environ)
    environment = isolated_home_environment(
        {key: value for key, value in inherited.items() if not key.startswith("POLYLOGUE_")},
        home=scratch / "home",
    )
    environment.update(POLYLOGUE_ARCHIVE_ROOT=str(archive), TMPDIR=str(scratch), POLYLOGUE_EMBEDDINGS_ENABLED="false")
    scope_keys = {
        key
        for key in environment
        if key.startswith("POLYLOGUE_") or key.startswith("XDG_") or key in {"HOME", "TMPDIR"}
    }
    for key in tuple(os.environ):
        if key.startswith("POLYLOGUE_"):
            os.environ.pop(key)
    os.environ.update({key: environment[key] for key in scope_keys})
    server = None
    thread = None
    thread_started = False
    process: subprocess.Popen[str] | None = None
    receiver_requests: list[dict[str, object]] = []
    settled = False
    retain_custody = False
    evidence_retained = False
    try:
        secret = secrets.token_urlsafe(32)
        persist_receiver_token(secret, archive / "browser-capture-receiver-token")
        identity = load_or_mint_receiver_identity(archive / "browser-capture-receiver-id")
        server = make_server("127.0.0.1", 0, spool_path=spool, archive_root=archive, auth_token=secret)
        server.daemon_threads = False
        server.block_on_close = True

        class ObservedHandler(BrowserCaptureHandler):
            def _finish_observed_request(self, method: str, started_at: float) -> None:
                path = urlsplit(self.path).path
                if path in {"/v1/status", "/v1/receiver/attest", "/v1/archive-state"}:
                    status = getattr(self, "_polylogue_status", None)
                    receiver_requests.append(
                        {
                            "method": method if method in {"GET", "POST", "OPTIONS"} else "unknown",
                            "path": path,
                            "status": status if type(status) is int and 100 <= status <= 599 else None,
                        }
                    )
                super()._finish_observed_request(method, started_at)

        server.RequestHandlerClass = ObservedHandler
        port = int(server.server_address[1])
        endpoint = f"http://127.0.0.1:{port}"
        scope_path = scratch / "constructor-scope.json"
        scope_path.write_text(json.dumps({"receiverUrl": endpoint, "receiverId": identity, "targets": targets}))
        scope_path.chmod(0o600)

        def build(extension: Path, host_name: str) -> dict[str, object]:
            completed = subprocess.run(
                [
                    "node",
                    str(root / "browser-extension/scripts/owned_provider_extension.mjs"),
                    "--destination",
                    str(extension),
                    "--host",
                    host_name,
                    "--scope",
                    str(scope_path),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            binding = json.loads(completed.stdout)
            if (
                not isinstance(binding, dict)
                or binding.get("kind") != "owned-provider-runtime"
                or binding.get("owned_targets_bound") is not False
            ):
                raise ValueError("proof owned runtime constructor binding invalid")
            return binding

        with scoped_native_host(
            repo_root=root,
            scratch=scratch,
            environment=environment,
            chrome_user_data_dir=chrome_user_data_dir,
            build_extension=build,
        ) as host:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            thread_started = True
            try:
                environment["POLYLOGUE_LIVE_PROVIDER_EXTENSION_ROOT"] = host["extension_root"]
                environment["POLYLOGUE_LIVE_PROVIDER_DIAGNOSTIC_PATH"] = str(scratch / "owned-capture-diagnostic.json")
                process = subprocess.Popen(
                    ["node", "scripts/owned_provider_proof.mjs"],
                    cwd=root / "browser-extension",
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    stdout, _stderr = process.communicate()
                except UnicodeError as error:
                    raise failed_child(None, receiver_requests) from error
                if process.returncode != 0:
                    raise failed_child(stdout, receiver_requests)
                try:
                    result = json.loads(stdout)
                except (ValueError, TypeError) as error:
                    raise failed_child(stdout, receiver_requests) from error
                if not isinstance(result, dict) or result.get("ok") is not True:
                    raise failed_child(stdout, receiver_requests)
                descriptor = os.open(scratch / "owned-capture-result.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as retained_result:
                    retained_result.write(stdout)
                receipts = result.get("providers")
                if not isinstance(receipts, dict) or set(receipts) != {
                    urlsplit(target["url"]).hostname for target in targets
                }:
                    raise failed_child(None, receiver_requests)
                isolation = result.get("isolation")
                if (
                    result.get("automatic_capture_enabled") is not True
                    or not isinstance(isolation, dict)
                    or isolation.get("declared_window_count") != len(targets)
                    or isolation.get("admitted_tab_count") != len(targets)
                    or isolation.get("static_content_scripts") is not False
                    or isolation.get("document_bound_effects") is not True
                    or isolation.get("current_window") != "first_declared_owned_window"
                ):
                    raise failed_child(None, receiver_requests)
                try:
                    for target in targets:
                        receipt = receipts[urlsplit(target["url"]).hostname]
                        if (
                            not isinstance(receipt, dict)
                            or receipt.get("provider") != {"chatgpt": "chatgpt", "claude": "claude-ai"}[target["name"]]
                            or receipt.get("provider_session_id_sha256")
                            != hashlib.sha256(target["nativeId"].encode()).hexdigest()
                        ):
                            raise ValueError("capture receipt differs from declared owned target")
                    providers = {name: verify_captured_artifact(spool, receipt) for name, receipt in receipts.items()}
                    extension = result["extension"]
                    binding = result["proof_binding"]
                    if (
                        not isinstance(binding, dict)
                        or binding.get("extension_id") != host["extension_id"]
                        or binding.get("host_name") != host["host_name"]
                    ):
                        raise ValueError("owned native host binding differs")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    failure = failed_child(None, receiver_requests)
                    failure.report["error"] = {"phase": "summary", "category": "capture_incomplete"}
                    raise failure from error
                return {
                    "ok": True,
                    "ports": {"browser_capture": port},
                    "providers": providers,
                    "extension": extension,
                    "proof_binding": binding,
                    "isolation": isolation,
                    "automatic_capture_enabled": True,
                    "archive_convergence": "not_exercised",
                    "retained_evidence": str(evidence_root),
                    "receiver_requests": receiver_requests,
                }
            finally:
                if process is not None:
                    terminate_process_group(process)
                server.shutdown()
                server.server_close()
                thread.join()
                settled = True
                retain_custody = True
                _retain_proof_evidence(scratch, spool, evidence_root)
                evidence_retained = True
                retain_custody = False
    except NativeProofCustodyError:
        retain_custody = True
        raise
    finally:
        if not settled:
            if process is not None:
                terminate_process_group(process)
            if server is not None:
                if thread_started:
                    server.shutdown()
                server.server_close()
            if thread_started and thread is not None:
                thread.join()
            settled = True
        if settled:
            try:
                if not evidence_retained and not (evidence_root / "browser-capture").exists():
                    _retain_proof_evidence(scratch, spool, evidence_root)
                    evidence_retained = True
            except OSError:
                retain_custody = True
                raise
            finally:
                os.environ.clear()
                os.environ.update(inherited)
            if not retain_custody:
                shutil.rmtree(scratch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one bounded JSON result.")
    parser.add_argument(
        "--conversations-file", type=Path, required=True, help="Private JSON array of exact conversation URLs."
    )
    parser.add_argument(
        "--chrome-user-data-dir",
        type=Path,
        required=True,
        help="Actual running Chrome user-data directory for the independently named native host.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Nonexistent private per-run directory retaining selected acquired bytes and bindings.",
    )
    arguments = parser.parse_args(argv)
    try:
        payload: dict[str, Any] = run_proof(
            conversations_file=arguments.conversations_file,
            chrome_user_data_dir=arguments.chrome_user_data_dir,
            evidence_root=arguments.evidence_root,
        )
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
