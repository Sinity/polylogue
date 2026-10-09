from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from polylogue.browser_capture.models import (
    BROWSER_CAPTURE_API_SCHEMA,
    BrowserCaptureAcceptedPayload,
    BrowserCaptureArchiveStatePayload,
    BrowserCaptureCapabilitiesPayload,
    BrowserCaptureEnvelope,
    BrowserCaptureErrorPayload,
    BrowserCaptureReceiverStatusPayload,
)
from polylogue.browser_capture.receiver import (
    BrowserCaptureReceiverConfig,
    capture_artifact_path,
    capture_artifact_ref,
    existing_capture_state,
    receiver_identity,
    receiver_status_payload,
    summarize_capture_envelope,
    write_capture_envelope,
)
from polylogue.browser_capture.route_contracts import (
    BROWSER_CAPTURE_ROUTE_CONTRACTS,
    browser_capture_route_contract_for,
)
from polylogue.browser_capture.server import (
    make_server,
    mission_control_archive_facts,
)
from polylogue.daemon.commands import main as daemon_cli
from polylogue.paths import browser_capture_receiver_identity_path

_EXTENSION_ORIGIN = "chrome-extension://polylogue-test"
_CHATGPT_ORIGIN = "https://chatgpt.com"


def _payload(provider: str = "chatgpt", session_id: str = "conv-123") -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-123",
            "page_title": "ChatGPT - Work plan",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
            "extension_instance_id": "test-extension-instance",
        },
        "session": {
            "provider": provider,
            "provider_session_id": session_id,
            "title": "Work plan",
            "turns": [{"provider_turn_id": "u1", "role": "user", "text": "Draft"}],
        },
    }


@contextmanager
def _running_receiver(
    tmp_path: Path,
    *,
    archive_root: Path | None = None,
    auth_token: str | None = None,
    extra_origins: tuple[str, ...] = (),
) -> Iterator[tuple[str, int]]:
    server = make_server(
        "127.0.0.1",
        0,
        spool_path=tmp_path,
        archive_root=archive_root,
        auth_token=auth_token,
        extra_origins=extra_origins,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _seed_browser_capture_archive(
    archive_root: Path,
    *,
    native_id: str = "conv-123",
    raw_id: str = "raw-capture",
    message_count: int = 1,
    parse_error: str | None = None,
    validation_status: str | None = None,
    validation_error: str | None = None,
    parsed_at_ms: int | None = None,
    validated_at_ms: int | None = None,
    updated_at_ms: int | None = None,
) -> None:
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                parse_error TEXT,
                validation_status TEXT,
                validation_error TEXT,
                parsed_at_ms INTEGER,
                validated_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, parse_error, validation_status, validation_error, parsed_at_ms, validated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "chatgpt-export",
                native_id,
                f"browser-capture/chatgpt/{native_id}.json",
                parse_error,
                validation_status,
                validation_error,
                parsed_at_ms,
                validated_at_ms,
            ),
        )
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                origin TEXT NOT NULL,
                session_id TEXT GENERATED ALWAYS AS (origin || ':' || native_id) STORED UNIQUE,
                raw_id TEXT,
                native_id TEXT,
                message_count INTEGER,
                updated_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (origin, raw_id, native_id, message_count, updated_at_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("chatgpt-export", raw_id, native_id, message_count, updated_at_ms),
        )


def _request(
    host: str, port: int, method: str, path: str, *, body: object | str | bytes | None = None, origin: str
) -> HTTPResponse:
    conn = HTTPConnection(host, port)
    headers = {"Origin": origin}
    payload: str | bytes | None
    if isinstance(body, (bytes, str)):
        payload = body
        headers["Content-Type"] = "application/json"
    elif body is not None:
        payload = json.dumps(body)
        headers["Content-Type"] = "application/json"
    else:
        payload = None
    conn.request(method, path, body=payload, headers=headers)
    return conn.getresponse()


def test_capture_artifact_path_is_deterministic(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload(session_id="c/with spaces"))

    first = capture_artifact_path(envelope, tmp_path)
    second = capture_artifact_path(envelope, tmp_path)
    artifact_ref = capture_artifact_ref(envelope, tmp_path)

    assert first == second
    assert first.parent.name == "chatgpt"
    assert "with-spaces" in first.name
    assert artifact_ref == first.relative_to(tmp_path).as_posix()
    assert Path(artifact_ref).is_absolute() is False


def test_write_capture_envelope_replaces_same_artifact(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())

    first = write_capture_envelope(envelope, spool_path=tmp_path)
    second = write_capture_envelope(envelope, spool_path=tmp_path)

    assert first.path == second.path
    assert first.replaced is False
    assert second.replaced is True
    assert json.loads(first.path.read_text(encoding="utf-8"))["session"]["provider_session_id"] == "conv-123"


@pytest.mark.parametrize("reverse", [False, True])
def test_same_instance_equal_clock_acquisition_order_preserves_newest_snapshot(tmp_path: Path, reverse: bool) -> None:
    snapshots = []
    for sequence, text in [(1, "Earlier synthetic revision"), (2, "Later synthetic revision")]:
        payload = _payload()
        cast(dict[str, object], payload["provenance"])["acquisition_sequence"] = sequence
        session = cast(dict[str, object], payload["session"])
        session["updated_at"] = "2026-04-24T00:00:00+00:00"
        session["turns"] = [{"provider_turn_id": "u1", "role": "user", "text": text}]
        snapshots.append(BrowserCaptureEnvelope.model_validate(payload))
    first, second = reversed(snapshots) if reverse else snapshots
    write_capture_envelope(first, spool_path=tmp_path)
    result = write_capture_envelope(second, spool_path=tmp_path)
    assert result.convergence.value == ("superseded" if reverse else "publish")
    retained = json.loads(result.path.read_text(encoding="utf-8"))
    assert retained["session"]["turns"][0]["text"] == "Later synthetic revision"


def test_acquisition_counters_do_not_compare_independent_instances_or_change_fingerprints(tmp_path: Path) -> None:
    earlier = BrowserCaptureEnvelope.model_validate(_payload())
    first = earlier.model_copy(update={"provenance": earlier.provenance.model_copy(update={"acquisition_sequence": 1})})
    second = earlier.model_copy(
        update={
            "provenance": earlier.provenance.model_copy(
                update={"acquisition_sequence": 999, "extension_instance_id": "independent-instance"}
            )
        }
    )
    assert summarize_capture_envelope(first).dedup_content_hash == summarize_capture_envelope(second).dedup_content_hash
    second = second.model_copy(
        update={
            "session": second.session.model_copy(
                update={
                    "turns": [second.session.turns[0].model_copy(update={"text": "Independent synthetic revision"})]
                }
            )
        }
    )
    write_capture_envelope(first, spool_path=tmp_path)
    result = write_capture_envelope(second, spool_path=tmp_path)
    assert result.convergence.value == "superseded"
    assert json.loads(result.path.read_text(encoding="utf-8"))["session"]["turns"][0]["text"] == "Draft"


@pytest.mark.parametrize("sequence", [0, -1, True, 1.5, 2**53])
def test_capture_observation_refuses_unrepresentable_or_noninteger_sequences(sequence: object) -> None:
    payload = _payload()
    cast(dict[str, object], payload["provenance"])["acquisition_sequence"] = sequence
    with pytest.raises(ValidationError):
        BrowserCaptureEnvelope.model_validate(payload)


def test_capture_observation_requires_instance_and_keeps_missing_historical_proof() -> None:
    payload = _payload()
    provenance = cast(dict[str, object], payload["provenance"])
    provenance.pop("extension_instance_id")
    assert BrowserCaptureEnvelope.model_validate(payload).provenance.acquisition_sequence is None
    provenance["acquisition_sequence"] = 1
    with pytest.raises(ValidationError):
        BrowserCaptureEnvelope.model_validate(payload)


def test_duplicate_acquisition_witness_does_not_replace_newer_provider_revision(tmp_path: Path) -> None:
    baseline = BrowserCaptureEnvelope.model_validate(_payload())
    current = baseline.model_copy(
        update={
            "provenance": baseline.provenance.model_copy(update={"acquisition_sequence": 1}),
            "session": baseline.session.model_copy(update={"updated_at": "2026-01-02T00:00:00Z"}),
        }
    )
    accepted = write_capture_envelope(current, spool_path=tmp_path)
    original = accepted.path.read_bytes()
    older = current.model_copy(
        update={
            "provenance": current.provenance.model_copy(update={"acquisition_sequence": 3}),
            "session": current.session.model_copy(update={"updated_at": "2026-01-01T00:00:00Z"}),
        }
    )
    duplicate = write_capture_envelope(older, spool_path=tmp_path)
    assert duplicate.convergence.value == "superseded"
    assert duplicate.path.read_bytes() == original


def test_duplicate_publication_durably_advances_witness_before_delayed_revision_after_restart(tmp_path: Path) -> None:
    baseline = BrowserCaptureEnvelope.model_validate(_payload())
    first = baseline.model_copy(
        update={
            "provenance": baseline.provenance.model_copy(update={"acquisition_sequence": 1}),
            "session": baseline.session.model_copy(
                update={"turns": [baseline.session.turns[0].model_copy(update={"text": "Synthetic alpha"})]}
            ),
        }
    )
    write_capture_envelope(first, spool_path=tmp_path)
    latest = first.model_copy(update={"provenance": first.provenance.model_copy(update={"acquisition_sequence": 3})})
    duplicate = write_capture_envelope(latest, spool_path=tmp_path)
    assert duplicate.convergence.value == "duplicate" and duplicate.deduplicated
    assert json.loads(duplicate.path.read_text(encoding="utf-8"))["provenance"]["acquisition_sequence"] == 3
    delayed = first.model_copy(
        update={
            "provenance": first.provenance.model_copy(update={"acquisition_sequence": 2}),
            "session": first.session.model_copy(
                update={"turns": [first.session.turns[0].model_copy(update={"text": "Synthetic beta"})]}
            ),
        }
    )
    restarted = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from polylogue.browser_capture.receiver import write_capture_envelope_bytes; "
            "result = write_capture_envelope_bytes(sys.argv[1].encode(), spool_path=Path(sys.argv[2])); "
            "print(result.convergence.value)",
            delayed.model_dump_json(exclude_none=True),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert restarted.stdout.strip() == "superseded"
    assert json.loads(duplicate.path.read_text(encoding="utf-8"))["session"]["turns"][0]["text"] == "Synthetic alpha"
    inode = duplicate.path.stat().st_ino
    cached = write_capture_envelope(latest, spool_path=tmp_path)
    assert cached.convergence.value == "duplicate"
    assert cached.path.stat().st_ino == inode


def test_native_envelope_spool_preserves_mapping_order_and_ordinary_parser_identity(tmp_path: Path) -> None:
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.parsers.browser_capture import parse as parse_capture
    from polylogue.sources.parsers.chatgpt import parse as parse_native

    fixture = Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-conversation-v1.json"
    native = json.loads(fixture.read_text(encoding="utf-8"))
    assert list(native["mapping"]) != sorted(native["mapping"])
    payload = _payload(session_id=native["conversation_id"])
    cast(dict[str, object], payload["session"])["title"] = native["title"]
    payload["raw_provider_payload"] = native
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    canonical_order = {**native, "mapping": dict(sorted(native["mapping"].items()))}
    reordered = envelope.model_copy(update={"raw_provider_payload": canonical_order})
    assert (
        summarize_capture_envelope(envelope).dedup_content_hash
        == summarize_capture_envelope(reordered).dedup_content_hash
    )
    published = write_capture_envelope(envelope, spool_path=tmp_path)
    retained = json.loads(published.path.read_text(encoding="utf-8"))
    assert list(retained["raw_provider_payload"]["mapping"]) == list(native["mapping"])
    expected = parse_native(native, native["conversation_id"])
    actual = parse_capture(retained, native["conversation_id"])
    assert [message.model_dump(mode="json") for message in actual.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert session_content_hash(actual) == session_content_hash(expected)


def test_claude_native_spool_preserves_graph_order_and_missing_id_attachment_custody(tmp_path: Path) -> None:
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.parsers.browser_capture import parse as parse_capture
    from polylogue.sources.parsers.claude.ai_parser import parse_ai

    fixture = Path(__file__).parents[2] / "fixtures" / "claude-ai" / "native-attachment-order.json"
    native = json.loads(fixture.read_text(encoding="utf-8"))
    assert all(not message.get("uuid") and not message.get("id") for message in native["chat_messages"])
    payload = _payload(provider="claude-ai", session_id=native["uuid"])
    expected = parse_ai(native, native["uuid"])
    session = cast(dict[str, object], payload["session"])
    session["title"] = native["name"]
    session["turns"] = [
        {
            "provider_turn_id": message.provider_message_id or "",
            "ordinal": ordinal,
            "role": message.role.value,
            "text": message.text,
        }
        for ordinal, message in enumerate(expected.messages)
    ]
    payload["raw_provider_payload"] = native
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    published = write_capture_envelope(envelope, spool_path=tmp_path)
    retained = json.loads(published.path.read_text(encoding="utf-8"))
    assert retained["raw_provider_payload"] == native
    expected = parse_ai(native, native["uuid"])
    assert expected.messages and expected.attachments
    assert [message.text for message in expected.messages] == [
        "Earlier synthetic message",
        "Later synthetic message",
    ]
    assert [message.text for message in expected.messages] != [message["text"] for message in native["chat_messages"]]
    actual = parse_capture(retained, native["uuid"])
    assert [message.model_dump(mode="json") for message in actual.messages] == [
        message.model_dump(mode="json") for message in expected.messages
    ]
    assert [attachment.model_dump(mode="json") for attachment in actual.attachments] == [
        attachment.model_dump(mode="json") for attachment in expected.attachments
    ]
    assert session_content_hash(actual) == session_content_hash(expected)


def test_capture_admission_preserves_a_backlog_larger_than_twenty_thousand(tmp_path: Path) -> None:
    # Populate valid durable envelopes without paying an fsync per fixture.
    # Admission still runs the actual production stage/reservation/publish path.
    first_path = None
    first_bytes = None
    for index in range(20_001):
        envelope = BrowserCaptureEnvelope.model_validate(_payload(session_id=f"backlog-{index}"))
        path = capture_artifact_path(summarize_capture_envelope(envelope), tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = envelope.model_dump_json().encode("utf-8")
        path.write_bytes(raw)
        if index == 0:
            first_path, first_bytes = path, raw
    result = write_capture_envelope(
        BrowserCaptureEnvelope.model_validate(_payload(session_id="after-backlog")), spool_path=tmp_path
    )
    assert result.path.is_file()
    assert first_path is not None and first_path.read_bytes() == first_bytes
    assert sum(1 for _ in tmp_path.rglob("*.json")) == 20_002


def test_concurrent_distinct_captures_are_all_preserved(tmp_path: Path) -> None:
    outcomes: list[object] = []
    lock = Lock()

    def write(index: int) -> None:
        result: object
        try:
            result = write_capture_envelope(
                BrowserCaptureEnvelope.model_validate(_payload(session_id=f"conv-race-{index}")),
                spool_path=tmp_path,
            )
        except Exception as exc:
            result = exc
        with lock:
            outcomes.append(result)

    threads = [Thread(target=write, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outcomes) == 20
    assert not any(isinstance(result, Exception) for result in outcomes)
    assert len(list(tmp_path.rglob("*.json"))) == 20


def test_existing_capture_state_reports_written_artifact(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)

    state = existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    typed = BrowserCaptureArchiveStatePayload.model_validate(state)

    assert typed.captured is False
    assert typed.spooled is True
    assert typed.state == "spooled_only"
    assert typed.lifecycle == "spooled_only"
    assert typed.provider == "chatgpt"
    assert typed.artifact_ref == capture_artifact_ref(envelope, tmp_path)
    assert Path(typed.artifact_ref).is_absolute() is False


def test_receiver_identity_is_stable_across_restarts_and_hides_pairing_secret(tmp_path: Path) -> None:
    token = "pairing-secret-that-must-never-leak"
    first = BrowserCaptureReceiverConfig(spool_path=tmp_path / "spool", auth_token=token)
    restarted = BrowserCaptureReceiverConfig(spool_path=tmp_path / "moved-spool", auth_token=token)

    first_id = receiver_identity(first)
    restarted_id = receiver_identity(restarted)

    assert first_id == restarted_id
    assert first_id.startswith("rx-")
    assert len(first_id) == 23
    assert token not in first_id
    assert str(first.spool_path) not in first_id


def test_receiver_identity_survives_token_rotation(tmp_path: Path) -> None:
    """polylogue-jlme.5 AC2: identity is persisted independently of the bearer token.

    Before this fix, ``receiver_identity`` hashed ``config.auth_token``
    directly, so an ordinary token rotation silently minted a *different*
    receiver identity and forced every paired browser profile through an
    unnecessary re-pair. Rotation must now be invisible to identity.
    """
    before = BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token="token-before-rotation")
    after = BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token="token-after-rotation")

    assert receiver_identity(before) == receiver_identity(after)


def test_receiver_identity_is_minted_once_and_persisted_on_disk(tmp_path: Path) -> None:
    """The identity is a real minted value read back from disk, not merely a
    process-memoized pure function of config -- a fresh config instance
    (simulating a daemon restart) must still recover the same value."""
    identity_path = browser_capture_receiver_identity_path()
    assert not identity_path.exists()

    config = BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token="some-token")
    minted = receiver_identity(config)

    assert identity_path.exists()
    assert identity_path.read_text(encoding="utf-8").strip() == minted
    # A brand-new config object for the "same receiver" (same archive root)
    # must read the persisted value back rather than minting a new one.
    assert receiver_identity(BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token="different-token")) == minted


def test_no_auth_receiver_identity_is_also_persisted_independently_of_spool(tmp_path: Path) -> None:
    """The no-auth escape hatch previously derived identity from the resolved
    spool path (the only stable-ish signal available without a token). It now
    shares the same archive-root-scoped persisted identity as the
    authenticated path, so it is unaffected by spool relocation too."""
    first = BrowserCaptureReceiverConfig(spool_path=tmp_path / "same" / ".." / "spool")
    same_resolved_spool = BrowserCaptureReceiverConfig(spool_path=tmp_path / "spool")
    relocated_spool = BrowserCaptureReceiverConfig(spool_path=tmp_path / "other")

    assert receiver_identity(first) == receiver_identity(same_resolved_spool)
    assert receiver_identity(first) == receiver_identity(relocated_spool)


def test_receiver_status_payload_advertises_versioned_stable_identity(tmp_path: Path) -> None:
    config = BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token="stable-pairing-token")

    payload = receiver_status_payload(config)
    typed = BrowserCaptureReceiverStatusPayload.model_validate(payload)

    assert typed.api_schema == BROWSER_CAPTURE_API_SCHEMA
    assert typed.receiver_id == receiver_identity(config)
    assert typed.auth_required is True
    assert "stable-pairing-token" not in json.dumps(payload)


def test_receiver_status_route_exposes_pairing_contract(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(host, port, "GET", "/v1/status", origin=_EXTENSION_ORIGIN)
        payload = json.loads(response.read())

    typed = BrowserCaptureReceiverStatusPayload.model_validate(payload)
    assert response.status == HTTPStatus.OK
    assert typed.api_schema == BROWSER_CAPTURE_API_SCHEMA
    assert typed.receiver_id.startswith("rx-")


def test_browser_capture_status_daemon_cli_json(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.config import resolve_runtime_config

    with _running_receiver(cli_workspace["archive_root"] / "browser-capture") as (host, port):
        runtime = resolve_runtime_config(cli_overrides={"browser_capture_host": host, "browser_capture_port": port})
        monkeypatch.setattr("polylogue.config.resolve_runtime_config", lambda: runtime)
        result = CliRunner().invoke(
            daemon_cli, ["browser-capture", "status", "--format", "json"], catch_exceptions=False
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    typed = BrowserCaptureReceiverStatusPayload.model_validate(payload)
    assert payload["ok"] is True
    assert typed.receiver == "polylogue-browser-capture"
    assert typed.spool_ready is True
    assert typed.auth_required is False
    assert typed.allowed_origins == ["chrome-extension://*"]
    assert typed.spool_path.endswith("browser-capture")


def test_browser_capture_token_show_mints_and_persists(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()

    first = runner.invoke(daemon_cli, ["browser-capture", "token", "show", "--format", "json"], catch_exceptions=False)
    second = runner.invoke(daemon_cli, ["browser-capture", "token", "show", "--format", "json"], catch_exceptions=False)

    assert first.exit_code == 0
    assert second.exit_code == 0
    first_token = json.loads(first.output)["token"]
    second_token = json.loads(second.output)["token"]
    assert first_token == second_token
    assert len(first_token) > 20


def test_browser_capture_token_show_rotate_changes_the_token(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()

    original = json.loads(
        runner.invoke(
            daemon_cli, ["browser-capture", "token", "show", "--format", "json"], catch_exceptions=False
        ).output
    )["token"]
    rotated = json.loads(
        runner.invoke(
            daemon_cli, ["browser-capture", "token", "show", "--rotate", "--format", "json"], catch_exceptions=False
        ).output
    )["token"]
    reloaded = json.loads(
        runner.invoke(
            daemon_cli, ["browser-capture", "token", "show", "--format", "json"], catch_exceptions=False
        ).output
    )["token"]

    assert rotated != original
    assert reloaded == rotated


class _StubServer:
    def __init__(self, spool_path: Path) -> None:
        self.config = SimpleNamespace(spool_path=spool_path)

    def serve_forever(self) -> None:
        return None

    def server_close(self) -> None:
        return None


def test_browser_capture_serve_allow_no_auth_env_var_matches_the_flag(cli_workspace: dict[str, Path]) -> None:
    """The --allow-no-auth flag documents itself as equivalent to setting
    POLYLOGUE_BROWSER_CAPTURE_ALLOW_NO_AUTH=1 -- prove the env var alone
    (no flag) actually produces the same no-auth server construction, and
    that plain default invocation (neither) still auto-mints a token."""
    runner = CliRunner()

    def _invoke(extra_args: list[str], *, env: dict[str, str] | None = None) -> object | None:
        captured: dict[str, object] = {}

        def _fake_make_server(*_args: object, **kwargs: object) -> _StubServer:
            captured.update(kwargs)
            return _StubServer(cli_workspace["archive_root"] / "browser-capture")

        with patch("polylogue.daemon.browser_capture.make_server", side_effect=_fake_make_server):
            result = runner.invoke(
                daemon_cli, ["browser-capture", "serve", *extra_args], env=env, catch_exceptions=False
            )
        assert result.exit_code == 0
        return captured.get("auth_token")

    default_token = _invoke([])
    env_token = _invoke([], env={"POLYLOGUE_BROWSER_CAPTURE_ALLOW_NO_AUTH": "1"})
    flag_token = _invoke(["--allow-no-auth"])

    assert isinstance(default_token, str) and len(default_token) > 20
    assert env_token is None
    assert flag_token is None


@pytest.mark.parametrize(
    ("auth_args", "expected_token"),
    [(["--auth-token", "explicit-token"], "explicit-token"), (["--allow-no-auth"], None)],
)
def test_browser_action_uses_the_selected_receiver_auth_identity(
    tmp_path: Path,
    auth_args: list[str],
    expected_token: str | None,
) -> None:
    result = CliRunner().invoke(
        daemon_cli,
        [
            "browser-capture",
            "action",
            "--provider",
            "chatgpt",
            "--text",
            "harmless draft",
            "--model-slug",
            "gpt-5-6-pro",
            "--model-label",
            "GPT-5.6 Sol",
            "--effort-label",
            "Pro",
            "--format",
            "json",
            *auth_args,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # ``--allow-no-auth`` logs its loud warning to stderr; ``output`` interleaves
    # both streams, and the JSON contract is stdout's.
    payload = json.loads(result.stdout)
    expected = receiver_identity(BrowserCaptureReceiverConfig(spool_path=tmp_path, auth_token=expected_token))
    assert payload["receiver_id"] == expected


def test_browser_capture_route_contracts_cover_receiver_boundary() -> None:
    concrete_routes = {(contract.method, contract.pattern) for contract in BROWSER_CAPTURE_ROUTE_CONTRACTS}

    assert ("GET", "/v1/status") in concrete_routes
    assert ("GET", "/v1/browser-captures/capabilities") in concrete_routes
    assert ("GET", "/v1/archive-state") in concrete_routes
    assert ("GET", "/v1/mission-control") in concrete_routes
    assert ("POST", "/v1/browser-captures") in concrete_routes
    assert browser_capture_route_contract_for("POST", "/v1/browser-captures") is not None


def test_receiver_rejects_web_origins_by_default(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(host, port, "GET", "/v1/status", origin=_CHATGPT_ORIGIN)
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.FORBIDDEN
    assert response.getheader("X-Request-ID")
    assert error.error == "origin_not_allowed"


def test_receiver_declares_durable_browser_backfill_ack_contract(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(host, port, "GET", "/v1/browser-captures/capabilities", origin=_EXTENSION_ORIGIN)
        body = BrowserCaptureCapabilitiesPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.OK
    assert response.getheader("X-Request-ID")
    assert body.durable_ack_fields == ("receiver_request_id", "content_hash", "submitted_content_hash", "outcome")
    assert body.assertion_candidates is True


def test_receiver_rejects_extra_web_origin_without_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="web origins require"):
        make_server("127.0.0.1", 0, spool_path=tmp_path, extra_origins=(_CHATGPT_ORIGIN,))


def test_receiver_accepts_extension_capture_and_reports_typed_dto(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=_payload(),
            origin=_EXTENSION_ORIGIN,
        )
        body = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.ACCEPTED
    assert response.getheader("X-Request-ID")
    assert body.ok is True
    assert body.receiver == "polylogue-browser-capture"
    assert body.source == "browser-extension"
    assert body.schema_version == 1
    assert body.capture_id == "chatgpt:conv-123"
    assert body.artifact_ref == capture_artifact_ref(BrowserCaptureEnvelope.model_validate(_payload()), tmp_path)
    request_body = json.dumps(_payload()).encode()
    assert body.content_hash == hashlib.sha256(request_body).hexdigest()
    assert Path(body.artifact_ref).is_absolute() is False
    assert (tmp_path / body.artifact_ref).exists()


def test_receiver_rejects_capture_without_extension_instance_id(tmp_path: Path) -> None:
    payload = _payload()
    cast(dict[str, object], payload["provenance"]).pop("extension_instance_id")

    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=payload,
            origin=_EXTENSION_ORIGIN,
        )
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.BAD_REQUEST
    assert error.error == "missing_extension_instance_id"
    assert not list(tmp_path.rglob("*.json"))


def test_receiver_ack_hashes_exact_javascript_request_bytes(tmp_path: Path) -> None:
    request_body = (
        '{"polylogue_capture_kind":"browser_llm_session","schema_version":1,'
        '"provenance":{"source_url":"https://chatgpt.com/c/conv-123",'
        '"captured_at":"2026-04-24T00:00:00.000Z","adapter_name":"chatgpt-backfill-native-v1",'
        '"extension_instance_id":"test-extension-instance"},'
        '"session":{"provider":"chatgpt","provider_session_id":"conv-123",'
        '"turns":[{"provider_turn_id":"u1","role":"user","text":"żółć 😀"}]},'
        '"provider_meta":{"number":1.25,"nested":{"b":2,"a":1}}}'
    ).encode()
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=request_body,
            origin=_EXTENSION_ORIGIN,
        )
        accepted = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))

    assert accepted.content_hash == hashlib.sha256(request_body).hexdigest()


def test_receiver_admits_distinct_captures_without_discarding_existing_evidence(tmp_path: Path) -> None:
    prior = write_capture_envelope(
        BrowserCaptureEnvelope.model_validate(_payload(session_id="conv-existing")), spool_path=tmp_path
    )
    original = prior.path.read_bytes()
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=_payload(session_id="conv-new"),
            origin=_EXTENSION_ORIGIN,
        )
        accepted = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))
    assert response.status == HTTPStatus.ACCEPTED
    assert accepted.ok
    assert prior.path.read_bytes() == original
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_physical_publication_exhaustion_refuses_without_retiring_resident_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = write_capture_envelope(
        BrowserCaptureEnvelope.model_validate(_payload(session_id="conv-existing")), spool_path=tmp_path
    )
    original = prior.path.read_bytes()

    def exhausted_publication(_source: object, _target: object) -> None:
        raise OSError(errno.ENOSPC, "synthetic filesystem capacity exhausted")

    monkeypatch.setattr(os, "replace", exhausted_publication)
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=_payload(session_id="conv-new"),
            origin=_EXTENSION_ORIGIN,
        )
        refused = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))
    assert response.status == HTTPStatus.INSUFFICIENT_STORAGE
    assert refused.error == "spool_storage_exhausted"
    assert prior.path.read_bytes() == original
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_duplicate_retry_completes_a_previously_failed_directory_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import stat

    real_fsync = os.fsync
    failed = False
    directory_syncs = 0
    target = capture_artifact_path(BrowserCaptureEnvelope.model_validate(_payload()), tmp_path)

    def interrupted_sync(descriptor: int) -> None:
        nonlocal failed, directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            # Reach the actual post-rename barrier, after new-ancestor sync.
            if not failed and target.is_file():
                failed = True
                raise OSError(errno.EIO, "synthetic publication sync interruption")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", interrupted_sync)
    with _running_receiver(tmp_path) as (host, port):
        first = _request(host, port, "POST", "/v1/browser-captures", body=_payload(), origin=_EXTENSION_ORIGIN)
        assert first.status == HTTPStatus.INTERNAL_SERVER_ERROR
        first.read()
        path = target
        inode = path.stat().st_ino
        syncs_before_retry = directory_syncs
        retried = _request(host, port, "POST", "/v1/browser-captures", body=_payload(), origin=_EXTENSION_ORIGIN)
        accepted = BrowserCaptureAcceptedPayload.model_validate(json.loads(retried.read()))
    assert retried.status == HTTPStatus.ACCEPTED
    assert accepted.deduplicated is True
    assert directory_syncs > syncs_before_retry
    assert path.stat().st_ino == inode


def test_orphan_inspection_requires_pairing_even_on_an_unauthenticated_receiver(tmp_path: Path) -> None:
    root = tmp_path / "backfill-checkpoints"
    root.mkdir()
    raw = b'{"checkpoint":{"jobs":[]}}'
    path = root / "retained.json"
    path.write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "GET",
            f"/v1/capture-jobs/orphans/{digest}/payload?client_protocol=1",
            origin=_EXTENSION_ORIGIN,
        )
        refused = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))
    assert response.status == HTTPStatus.UNAUTHORIZED
    assert refused.error == "orphan_inspection_auth_required"
    assert path.read_bytes() == raw


def test_receiver_does_not_double_prefix_prefixed_capture_id(tmp_path: Path) -> None:
    payload = _payload()
    payload["capture_id"] = "chatgpt:chatgpt:conv-123"

    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=payload,
            origin=_EXTENSION_ORIGIN,
        )
        accepted = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))
        state_response = _request(
            host,
            port,
            "GET",
            "/v1/archive-state?provider=chatgpt&provider_session_id=conv-123",
            origin=_EXTENSION_ORIGIN,
        )
        state = BrowserCaptureArchiveStatePayload.model_validate(json.loads(state_response.read()))

    assert accepted.capture_id == "chatgpt:conv-123"
    assert state.capture_id == "chatgpt:conv-123"
    assert state.state == "spooled_only"
    assert state.lifecycle == "spooled_only"
    assert state.captured is False
    assert state.spooled is True


def test_mission_control_reports_uncaptured_without_archive_facts(tmp_path: Path) -> None:
    """An unindexed conversation must project 'uncaptured', never a zero cost."""
    with _running_receiver(tmp_path, archive_root=tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "GET",
            "/v1/mission-control?provider=chatgpt&provider_session_id=conv-123",
            origin=_EXTENSION_ORIGIN,
        )
        projection = json.loads(response.read())

    assert response.status == HTTPStatus.OK
    assert projection["status"] == "uncaptured"
    assert projection["archive"] == {"status": "uncaptured", "session_id": None, "ref": None}
    assert projection["cost"] == {"status": "unknown", "total_usd": None, "provenance": []}
    assert projection["assertions"] == {"status": "unknown", "items": []}


def test_mission_control_resolves_the_canonical_session_for_an_archived_capture(tmp_path: Path) -> None:
    """Layer 2 reads canonical identity from the archive, not from the request."""
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(tmp_path)

    with _running_receiver(tmp_path, archive_root=tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "GET",
            "/v1/mission-control?provider=chatgpt&provider_session_id=conv-123",
            origin=_EXTENSION_ORIGIN,
        )
        projection = json.loads(response.read())

    assert projection["archive"]["status"] == "available"
    assert projection["archive"]["session_id"] == "chatgpt-export:conv-123"
    assert projection["archive"]["ref"] == "session:chatgpt-export:conv-123"
    # This fixture archive carries raw + index rows but no derived insight
    # tables, so the projection must degrade explicitly rather than invent a
    # zero cost. The productive read path is covered by
    # test_mission_control_archive_facts_read_a_real_archive.
    assert projection["status"] == "unknown"
    assert projection["reason"] == "archive_projection_unavailable"
    assert projection["cost"] == {"status": "unknown", "total_usd": None, "provenance": []}
    assert projection["assertions"] == {"status": "unknown", "items": []}


def test_mission_control_resolves_the_default_archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production receiver is built with no explicit archive root.

    `run_daemon_services` calls `make_server(...)` without `archive_root`, so
    `_mission_control` saw `None` and degraded every projection to
    `archive_projection_unavailable` -- cost and assertions were permanently
    unavailable in production while every test that passed a temporary root
    saw them. `existing_capture_state` above already resolves the same
    default, which is why the indexed session was still found.

    Anti-vacuity: drop the `or default_archive_root()` fallback and the
    handler never calls `mission_control_archive_facts` -- `roots` stays empty
    and the projection degrades.
    """
    import polylogue.browser_capture.server as server_mod

    write_capture_envelope(BrowserCaptureEnvelope.model_validate(_payload()), spool_path=tmp_path)
    _seed_browser_capture_archive(tmp_path)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))

    roots: list[Path] = []

    def _facts(root: Path, session_id: str) -> tuple[dict[str, object], dict[str, object]]:
        del session_id
        roots.append(root)
        return (
            {"status": "available", "total_usd": 1.5, "provenance": []},
            {"status": "available", "items": []},
        )

    monkeypatch.setattr(server_mod, "mission_control_archive_facts", _facts)

    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "GET",
            "/v1/mission-control?provider=chatgpt&provider_session_id=conv-123",
            origin=_EXTENSION_ORIGIN,
        )
        projection = json.loads(response.read())

    assert roots == [tmp_path]
    assert projection["status"] == "available"
    assert projection["cost"] == {"status": "available", "total_usd": 1.5, "provenance": []}


def test_mission_control_archive_facts_read_a_real_archive(tmp_path: Path) -> None:
    """A schema-complete archive answers the projection instead of degrading.

    Anti-vacuity: if either read route is renamed or the facade construction
    stops matching the six-tier layout, the helper swallows the error and
    returns None, and this fails.
    """
    root = tmp_path / "archive"
    from tests.infra.archive_templates import bootstrap_ready_archive_root

    bootstrap_ready_archive_root(root)

    facts = mission_control_archive_facts(root, "chatgpt:conv-123")

    assert facts is not None
    cost, assertions = facts
    assert cost == {"status": "unknown", "total_usd": None, "provenance": []}
    assert assertions["status"] == "available"
    assert assertions["items"] == []


def test_mission_control_reads_only_judged_session_assertions_in_one_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: candidates must not reach this projection, and it must not issue one read per message."""

    import polylogue

    calls: list[tuple[str, tuple[str, ...]]] = []

    class FakePolylogue:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def list_session_cost_insights(self, _query: object) -> list[object]:
            return []

        async def list_assertion_claim_payloads(
            self,
            *,
            statuses: tuple[str, ...],
            limit: int,
            session_id: str | None = None,
        ) -> list[object]:
            calls.append((str(session_id), statuses))
            return []

    monkeypatch.setattr(polylogue, "Polylogue", FakePolylogue)
    result = mission_control_archive_facts(tmp_path, "chatgpt:conversation")

    assert result is not None
    assert calls == [
        ("chatgpt:conversation", ("active",)),
    ]


def test_mission_control_archive_facts_degrade_on_an_unreadable_archive(tmp_path: Path) -> None:
    assert mission_control_archive_facts(tmp_path, "chatgpt:missing") is None


def test_mission_control_rejects_a_request_without_provider_and_session(tmp_path: Path) -> None:
    with _running_receiver(tmp_path, archive_root=tmp_path) as (host, port):
        response = _request(host, port, "GET", "/v1/mission-control", origin=_EXTENSION_ORIGIN)
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.BAD_REQUEST
    assert error.error == "missing_provider_or_session"


def test_receiver_archive_state_reports_missing_without_spool_or_archive(tmp_path: Path) -> None:
    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "missing"
    assert state.lifecycle == "missing"
    assert state.captured is False
    assert state.spooled is False
    assert state.raw_row_exists is False
    assert state.indexed_session_exists is False
    assert Path(state.artifact_ref).is_absolute() is False


def test_receiver_archive_state_tolerates_invalid_active_index_pointer(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    (tmp_path / ".index-active-pointer").write_text("not-an-index.db\n", encoding="utf-8")

    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "spooled_only"
    assert state.indexed_session_exists is False


def test_receiver_archive_state_requires_indexed_messages(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(tmp_path, message_count=0)

    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "ingest_pending"
    assert state.captured is False
    assert state.spooled is True
    assert state.raw_row_exists is True
    assert state.raw_id == "raw-capture"
    assert state.indexed_session_exists is True
    assert state.indexed_message_count == 0


def test_receiver_archive_state_reports_archived_only_with_raw_index_and_messages(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(tmp_path)

    with _running_receiver(tmp_path, archive_root=tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "GET",
            "/v1/archive-state?provider=chatgpt&provider_session_id=conv-123",
            origin=_EXTENSION_ORIGIN,
        )
        state = BrowserCaptureArchiveStatePayload.model_validate(json.loads(response.read()))

    assert state.state == "archived"
    assert state.lifecycle == "archived"
    assert state.captured is True
    assert state.raw_row_exists is True
    assert state.indexed_session_exists is True
    assert state.indexed_session_id == "chatgpt-export:conv-123"
    assert state.indexed_message_count == 1


def test_receiver_archive_state_reports_stale_when_spool_is_newer(tmp_path: Path) -> None:
    payload = _payload()
    payload["session"] = {
        **cast(dict[str, object], payload["session"]),
        "updated_at": "2026-04-24T00:00:00+00:00",
    }
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(tmp_path, updated_at_ms=1_744_588_800_000)

    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "stale"
    assert state.lifecycle == "stale"
    assert state.captured is False
    assert state.spooled is True
    assert state.raw_row_exists is True
    assert state.indexed_session_exists is True
    assert state.indexed_message_count == 1


def test_receiver_archive_state_surfaces_raw_failure(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(tmp_path, parse_error="bad payload")

    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "failed"
    assert state.captured is False
    assert state.latest_failure == "bad payload"
    assert state.failure_source == "raw_parse"


def test_receiver_uses_active_index_and_ignores_historical_validation_failure(tmp_path: Path) -> None:
    """The public state reads the promoted generation, not its stale shadow."""
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(
        tmp_path,
        validation_status="failed",
        parsed_at_ms=1,
        validated_at_ms=0,
        message_count=0,
    )
    active_index = tmp_path / "generations" / "active" / "index.db"
    active_index.parent.mkdir(parents=True)
    with sqlite3.connect(active_index) as conn:
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT, raw_id TEXT, native_id TEXT, message_count INTEGER, updated_at_ms INTEGER)"
        )
        conn.execute("INSERT INTO sessions VALUES ('chatgpt-export:conv-123', 'raw-capture', 'conv-123', 1, NULL)")
    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")

    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "archived"
    assert state.latest_failure is None
    assert state.indexed_message_count == 1


def test_receiver_surfaces_validation_failure_newer_than_parse(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(
        tmp_path,
        validation_status="failed",
        parsed_at_ms=1,
        validated_at_ms=2,
        validation_error="strict validation rejected current bytes",
    )

    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "failed"
    assert state.latest_failure == "strict validation rejected current bytes"
    assert state.failure_source == "raw_validation"


def test_receiver_surfaces_indeterminate_raw_state_order_without_choosing_validation(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, spool_path=tmp_path)
    _seed_browser_capture_archive(
        tmp_path,
        validation_status="failed",
        parsed_at_ms=1,
        validated_at_ms=1,
        validation_error="equal-time failure",
    )

    state = BrowserCaptureArchiveStatePayload.model_validate(
        existing_capture_state("chatgpt", "conv-123", spool_path=tmp_path, archive_root=tmp_path)
    )

    assert state.state == "failed"
    assert state.latest_failure == "raw validation and parse timestamps are indeterminate"
    assert state.failure_source == "raw_state_order"


def test_receiver_echoes_safe_request_id_header(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "GET",
            "/v1/status",
            headers={
                "Origin": _EXTENSION_ORIGIN,
                "X-Request-ID": "dev-loop/request 123",
            },
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.OK
    assert response.getheader("X-Request-ID") == "dev-looprequest123"


@pytest.mark.parametrize(
    ("raw_body", "expected_error"),
    [
        ("{not-json", "invalid_json"),
        ({"polylogue_capture_kind": "browser_llm_session", "schema_version": 1}, "invalid_payload"),
        (
            {
                **_payload(),
                "session": {"provider": "chatgpt", "provider_session_id": "conv-123", "turns": []},
            },
            "invalid_payload",
        ),
    ],
    ids=["invalid-json", "missing-session", "empty-turns"],
)
def test_receiver_rejects_malformed_capture_payloads(
    tmp_path: Path,
    raw_body: object | str,
    expected_error: str,
) -> None:
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/browser-captures",
            body=raw_body,
            origin=_EXTENSION_ORIGIN,
        )
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.BAD_REQUEST
    assert error.error == expected_error
    assert list(tmp_path.rglob("*.json")) == []


def test_receiver_streams_captures_past_the_control_bound_byte_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture is staged in bounded reads and published as the exact body.

    Anti-vacuity: routing captures back through ``_read_json_body`` refuses
    this body (it exceeds the shrunken control bound), and a whole-body read
    of the capture makes the largest recorded read the body size instead of
    the chunk.
    """
    import polylogue.browser_capture.capture_stream as capture_stream
    import polylogue.browser_capture.server as server

    chunk = 64
    monkeypatch.setattr(server, "MAX_CONTROL_BODY_BYTES", 256)
    monkeypatch.setattr(capture_stream, "CAPTURE_READ_CHUNK_BYTES", chunk)
    reads: list[int] = []
    original_stage = capture_stream.stage_capture_body

    def recording_stage(read: Callable[[int], bytes], length: int, *, spool_root: Path) -> object:
        def recording_read(size: int) -> bytes:
            reads.append(size)
            return read(size)

        return original_stage(recording_read, length, spool_root=spool_root)

    monkeypatch.setattr(server, "stage_capture_body", recording_stage)
    payload = _payload()
    session = cast(dict[str, object], payload["session"])
    session["turns"] = [
        {"provider_turn_id": f"u{index}", "role": "user", "text": "long turn " * 40} for index in range(20)
    ]
    raw = json.dumps(payload, indent=2).encode("utf-8")
    assert len(raw) > 256 * 10

    with _running_receiver(tmp_path) as (host, port):
        response = _request(host, port, "POST", "/v1/browser-captures", body=raw, origin=_EXTENSION_ORIGIN)
        accepted = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.ACCEPTED
    assert max(reads) == chunk
    assert sum(reads) == len(raw)
    assert accepted.content_hash == hashlib.sha256(raw).hexdigest()
    assert (tmp_path / accepted.artifact_ref).read_bytes() == raw
    assert accepted.provider_session_id == "conv-123"
    assert not list(tmp_path.rglob(".*.tmp"))


def test_receiver_refuses_a_body_shorter_than_its_declared_length(tmp_path: Path) -> None:
    """A truncated upload is refused and leaves neither artifact nor staging file."""
    raw = json.dumps(_payload()).encode("utf-8")
    with _running_receiver(tmp_path) as (host, port):
        conn = HTTPConnection(host, port)
        conn.putrequest("POST", "/v1/browser-captures")
        conn.putheader("Origin", _EXTENSION_ORIGIN)
        conn.putheader("Content-Length", str(len(raw) + 100))
        conn.endheaders()
        assert conn.sock is not None
        conn.sock.sendall(raw)
        conn.sock.shutdown(socket.SHUT_WR)
        response = conn.getresponse()
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))
        conn.close()

    assert response.status == HTTPStatus.BAD_REQUEST
    assert error.error == "incomplete_body"
    assert list(tmp_path.rglob("*.json")) == []
    assert not list(tmp_path.rglob(".*.tmp"))


def test_receiver_auth_allows_extension_contract_preflight_without_bypassing_bearer(tmp_path: Path) -> None:
    requested_headers = {
        "authorization",
        "content-type",
        "x-request-id",
        "x-polylogue-client-protocol",
        "x-polylogue-extension-contract",
    }
    with _running_receiver(tmp_path, auth_token="secret") as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "OPTIONS",
            "/v1/browser-captures",
            headers={
                "Origin": _EXTENSION_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": ", ".join(sorted(requested_headers)),
            },
        )
        response = conn.getresponse()
        allow_headers = response.getheader("Access-Control-Allow-Headers")
        assert response.status == HTTPStatus.NO_CONTENT
        assert response.getheader("Access-Control-Allow-Origin") == _EXTENSION_ORIGIN
        assert allow_headers is not None
        assert requested_headers <= {header.strip().lower() for header in allow_headers.split(",")}
        response.read()
        headers = {
            "Origin": _EXTENSION_ORIGIN,
            "Content-Type": "application/json",
            "X-Polylogue-Extension-Contract": "1",
        }
        conn.request("POST", "/v1/browser-captures", body=json.dumps(_payload()), headers=headers)
        refused = conn.getresponse()
        assert refused.status == HTTPStatus.UNAUTHORIZED
        refused.read()
        conn.request(
            "POST",
            "/v1/browser-captures",
            body=json.dumps(_payload()),
            headers={**headers, "Authorization": "Bearer secret"},
        )
        accepted = conn.getresponse()
        assert accepted.status == HTTPStatus.ACCEPTED
        assert json.loads(accepted.read())["artifact_ref"]
        conn.close()


def test_receiver_preflight_grants_private_network_access_when_requested(tmp_path: Path) -> None:
    with _running_receiver(tmp_path, auth_token="secret", extra_origins=(_CHATGPT_ORIGIN,)) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "OPTIONS",
            "/v1/browser-captures",
            headers={
                "Origin": _CHATGPT_ORIGIN,
                "Access-Control-Request-Headers": "authorization, content-type, x-request-id",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        response = conn.getresponse()
        allow_private_network = response.getheader("Access-Control-Allow-Private-Network")
        response.read()
        conn.close()

    assert response.status == HTTPStatus.NO_CONTENT
    assert allow_private_network == "true"


def test_receiver_preflight_omits_private_network_header_when_not_requested(tmp_path: Path) -> None:
    with _running_receiver(tmp_path, auth_token="secret", extra_origins=(_CHATGPT_ORIGIN,)) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "OPTIONS",
            "/v1/browser-captures",
            headers={
                "Origin": _CHATGPT_ORIGIN,
                "Access-Control-Request-Headers": "authorization, content-type, x-request-id",
            },
        )
        response = conn.getresponse()
        allow_private_network = response.getheader("Access-Control-Allow-Private-Network")
        response.read()
        conn.close()

    assert response.status == HTTPStatus.NO_CONTENT
    assert allow_private_network is None


def test_receiver_allows_extra_web_origin_only_with_token(tmp_path: Path) -> None:
    with _running_receiver(tmp_path, auth_token="secret", extra_origins=(_CHATGPT_ORIGIN,)) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/browser-captures",
            body=json.dumps(_payload()),
            headers={
                "Content-Type": "application/json",
                "Origin": _CHATGPT_ORIGIN,
                "Authorization": "Bearer secret",
            },
        )
        response = conn.getresponse()
        body = BrowserCaptureAcceptedPayload.model_validate(json.loads(response.read()))
        conn.close()

    assert response.status == HTTPStatus.ACCEPTED
    assert body.ok is True
    assert Path(body.artifact_ref).is_absolute() is False


def test_receiver_rejects_wrong_token(tmp_path: Path) -> None:
    with _running_receiver(tmp_path, auth_token="secret", extra_origins=(_CHATGPT_ORIGIN,)) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/browser-captures",
            body=json.dumps(_payload()),
            headers={
                "Content-Type": "application/json",
                "Origin": _CHATGPT_ORIGIN,
                "Authorization": "Bearer wrong-token",
            },
        )
        response = conn.getresponse()
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))
        conn.close()

    assert response.status == HTTPStatus.UNAUTHORIZED
    assert error.error == "unauthorized"


@pytest.mark.parametrize("fallocate_errno", [errno.ENOSPC, errno.EOPNOTSUPP])
def test_capture_space_is_reserved_before_the_body_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallocate_errno: int
) -> None:
    """A body the spool filesystem cannot hold is refused before any byte is read.

    Covers both reservation routes: ``posix_fallocate`` reporting ENOSPC, and
    a filesystem without allocation falling back to free space. Anti-vacuity:
    staging without a reservation reads and writes the body first, so
    ``reads`` is non-empty, and a check that ignores the incoming length
    admits it into the one free byte.
    """
    import os

    import polylogue.browser_capture.capture_stream as capture_stream

    def failing_fallocate(fd: int, offset: int, length: int) -> None:
        raise OSError(fallocate_errno, os.strerror(fallocate_errno))

    monkeypatch.setattr(os, "posix_fallocate", failing_fallocate, raising=False)
    monkeypatch.setattr(capture_stream, "_available_bytes", lambda _directory: 1)
    reads: list[int] = []

    def read(size: int) -> bytes:
        reads.append(size)
        return b"x" * size

    with pytest.raises(capture_stream.SpoolStorageExhaustedError) as refused:
        capture_stream.stage_capture_body(read, 4096, spool_root=tmp_path)

    assert refused.value.requested_bytes == 4096
    assert reads == []
    assert list((tmp_path / capture_stream.STAGING_DIRNAME).iterdir()) == []


def test_receiver_answers_an_unreservable_capture_with_retryable_pressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP route maps the physical refusal to 507 and publishes nothing.

    Anti-vacuity: an untyped refusal surfaces as ``write_failed`` (500).
    """
    import os

    def full_disk(fd: int, offset: int, length: int) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(os, "posix_fallocate", full_disk, raising=False)
    raw = json.dumps(_payload()).encode("utf-8")
    with _running_receiver(tmp_path) as (host, port):
        response = _request(host, port, "POST", "/v1/browser-captures", body=raw, origin=_EXTENSION_ORIGIN)
        error = BrowserCaptureErrorPayload.model_validate(json.loads(response.read()))

    assert response.status == HTTPStatus.INSUFFICIENT_STORAGE
    assert error.error == "spool_storage_exhausted"
    assert not list(tmp_path.rglob("*.json"))


def _post_capture_raw(
    host: str, port: int, *, content_length: str, body: bytes, end_body: bool = False
) -> tuple[int, dict[str, object]]:
    """Send a capture request whose declared length the body need not match."""
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(
            (
                "POST /v1/browser-captures HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Origin: {_EXTENSION_ORIGIN}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {content_length}\r\n\r\n"
            ).encode("ascii")
            + body
        )
        if end_body:
            sock.shutdown(socket.SHUT_WR)
        response = HTTPResponse(sock)
        response.begin()
        return response.status, json.loads(response.read())


def test_receiver_answers_an_unrepresentable_body_length_with_the_physical_refusal(tmp_path: Path) -> None:
    """A length past what a file offset can hold is the typed 507, before any read.

    Anti-vacuity: ``posix_fallocate`` raises ``OverflowError`` for it, which
    escapes the staging route's ``OSError`` handling and drops the response.
    """
    import polylogue.browser_capture.capture_stream as capture_stream

    with _running_receiver(tmp_path) as (host, port):
        status, body = _post_capture_raw(host, port, content_length=str(2**63), body=b"{}")

    assert status == HTTPStatus.INSUFFICIENT_STORAGE
    assert body["error"] == "spool_storage_exhausted"
    assert list((tmp_path / capture_stream.STAGING_DIRNAME).iterdir()) == []


@pytest.mark.parametrize("content_length", ["+2", "0_2"])
def test_receiver_refuses_a_content_length_that_is_not_ascii_digits(tmp_path: Path, content_length: str) -> None:
    """``Content-Length`` is ``1*DIGIT``; a spelling ``int`` merely tolerates is refused.

    Anti-vacuity: ``int()`` reads ``+2`` and ``0_2`` as 2, so the two body
    bytes would be staged as a valid length.
    """
    with _running_receiver(tmp_path) as (host, port):
        status, body = _post_capture_raw(host, port, content_length=content_length, body=b"{}")

    assert status == HTTPStatus.BAD_REQUEST
    assert body["error"] == "invalid_content_length"


def test_receiver_releases_the_reservation_after_producer_cancellation(tmp_path: Path) -> None:
    """Closing the producer ends its physical read and releases partial staging.

    Anti-vacuity: a partial stage surviving EOF would retain reserved space
    after the producer has settled. A slow open producer is not cancellation.
    """
    import polylogue.browser_capture.capture_stream as capture_stream

    with _running_receiver(tmp_path) as (host, port):
        status, body = _post_capture_raw(host, port, content_length="4096", body=b'{"polylogue', end_body=True)

    assert status == HTTPStatus.BAD_REQUEST
    assert body["error"] == "incomplete_body"
    assert list((tmp_path / capture_stream.STAGING_DIRNAME).iterdir()) == []


def test_receiver_startup_reaps_abandoned_staging_but_not_live_uploads(tmp_path: Path) -> None:
    """A staging file no upload holds is removed at startup; a held one stays.

    Anti-vacuity: without the startup reap the abandoned file survives, and a
    reap that ignored the upload lock would delete the live upload's file.
    """
    import io

    import polylogue.browser_capture.capture_stream as capture_stream

    staging = tmp_path / capture_stream.STAGING_DIRNAME
    staging.mkdir(parents=True)
    abandoned = staging / ".capture-abandoned.tmp"
    abandoned.write_bytes(b"half an upload")
    live = capture_stream.stage_capture_body(io.BytesIO(b"{}").read, 2, spool_root=tmp_path)
    try:
        server = make_server("127.0.0.1", 0, spool_path=tmp_path)
        server.server_close()
        assert not abandoned.exists()
        assert live.path.exists()
    finally:
        live.discard()


@pytest.mark.uses_real_clock("starts the real UDS operation stack; wall-clock events bound its writer handoff")
def test_selected_message_candidate_persists_through_the_daemon_actuator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A browser selection save reaches ``user.db`` with its canonical message target.

    Anti-vacuity: restore the bare ``message:<provider id>`` ref in the receiver,
    drop message-ref support from ``resolve_assertion_candidate_refs``, pass the
    capture evidence ref as the assertion scope, or read the accepted mutation
    envelope as a failure, and the request fails before an assertion row is
    written; stop forwarding ``evidence_refs`` and the row loses the capture
    locator.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.daemon_operations import running_daemon_operations
    from tests.infra.live_ingest import write_index_session

    archive_root = tmp_path / "archive"
    message_ref = "chatgpt-export:conv-123:n:turn-1"
    spool = tmp_path / "spool"
    write_capture_envelope(BrowserCaptureEnvelope.model_validate(_payload()), spool_path=spool)
    # An outer write lease names its archive root; custody admits only an
    # existing root directory.
    archive_root.mkdir(parents=True, exist_ok=True)
    with write_lease("test.capture-intelligence", archive_root=archive_root), ArchiveStore(archive_root) as archive:
        write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="conv-123",
                messages=[ParsedMessage(provider_message_id="turn-1", role=Role.USER, text="selected message")],
            ),
        )
    with running_daemon_operations(archive_root) as stack:
        monkeypatch.setattr("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: stack.socket_path)
        monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_a, **_k: None)
        with _running_receiver(tmp_path / "spool", archive_root=stack.archive_root) as (host, port):
            response = _request(
                host,
                port,
                "POST",
                "/v1/assertion-candidates",
                body={
                    "body_text": "remember this turn",
                    "kind": "lesson",
                    # The capture artifact's evidence ref, as the receiver returns it.
                    "evidence_refs": ["chatgpt/conv-123.json#message:turn-1"],
                    "target_ref": message_ref,
                    "source_observation": {
                        "fidelity": "native",
                        "origin": "chatgpt-export",
                        "provider_conversation_id": "conv-123",
                        "provider_message_id": "turn-1",
                    },
                    "idempotency_key": "selection-1",
                    "context_policy": {"inject": False},
                },
                origin=_EXTENSION_ORIGIN,
            )
            body = json.loads(response.read())
        assert response.status == HTTPStatus.ACCEPTED, body
        with _running_receiver(spool, archive_root=stack.archive_root) as (host, port):

            def panel() -> dict[str, object]:
                response = _request(
                    host,
                    port,
                    "GET",
                    "/v1/mission-control?provider=chatgpt&provider_session_id=conv-123",
                    origin=_EXTENSION_ORIGIN,
                )
                assert response.status == HTTPStatus.OK
                return cast(dict[str, object], json.loads(response.read()))

            before = panel()
            assert before["assertions"] == {"status": "available", "items": []}, before
            with sqlite3.connect(stack.archive_root / "user.db") as conn:
                candidate_id = conn.execute(
                    "SELECT assertion_id FROM assertions WHERE status = 'candidate'"
                ).fetchone()[0]
            judgment = stack.client.operation_to_completion(
                "mutation.facade.judge_assertion_candidate",
                {
                    "candidate_ref": f"assertion:{candidate_id}",
                    "decision": "accept",
                    "reason": "synthetic verified selection",
                },
                archive_root=str(stack.archive_root),
            )
            assert judgment is not None and judgment["outcome"] == "completed"
            after = panel()
            assert after["status"] == "available"
            claims = cast(dict[str, object], after["assertions"])
            assert claims["status"] == "available"
            items = cast(list[dict[str, object]], claims["items"])
            assert len(items) == 1
            assert items[0]["target_ref"] == f"message:{message_ref}"
            assert items[0]["status"] == "active"
        with sqlite3.connect(stack.archive_root / "user.db") as conn:
            rows = conn.execute(
                "SELECT target_ref, scope_ref, body_text, evidence_refs_json FROM assertions "
                "WHERE key = 'terminal-note' AND status = 'accepted'"
            ).fetchall()
    assert [row[:3] for row in rows] == [
        (f"message:{message_ref}", "session:chatgpt-export:conv-123", "remember this turn")
    ]
    assert "chatgpt/conv-123.json#message:turn-1" in json.loads(rows[0][3])


@pytest.mark.parametrize("missing", ["origin", "provider_conversation_id"])
def test_selection_without_native_session_coordinates_is_refused(tmp_path: Path, missing: str) -> None:
    """A native observation must name its session before refs are built from it.

    Anti-vacuity: build the refs from ``observation.get(...)`` without checking
    and ``None:None:n:<id>`` passes the target equality check.
    """
    observation: dict[str, object] = {
        "fidelity": "native",
        "origin": "chatgpt-export",
        "provider_conversation_id": "conv-123",
        "provider_message_id": "turn-1",
    }
    del observation[missing]
    with _running_receiver(tmp_path) as (host, port):
        response = _request(
            host,
            port,
            "POST",
            "/v1/assertion-candidates",
            body={
                "body_text": "remember this turn",
                "kind": "lesson",
                "evidence_refs": ["chatgpt/conv-123.json#message:turn-1"],
                "target_ref": "None:None:n:turn-1",
                "source_observation": observation,
                "context_policy": {"inject": False},
            },
            origin=_EXTENSION_ORIGIN,
        )
        body = json.loads(response.read())
    assert response.status == HTTPStatus.BAD_REQUEST
    assert body["error"] == "exact_message_evidence_required"


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("fault_directory", [None, "root", "provider"])
def test_receiver_settles_ancestor_barriers_before_http_ack_and_duplicate_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool, fault_directory: str | None
) -> None:
    """Failed post-rename sync cannot become an acknowledged duplicate on retry."""
    import os
    import stat

    provider_dir = tmp_path / "chatgpt"
    if preexisting:
        provider_dir.mkdir()
    events: list[Path | str] = []
    real_sync, real_replace = os.fsync, os.replace
    fault = None if fault_directory is None else {"root": tmp_path, "provider": provider_dir}[fault_directory]
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    target = capture_artifact_path(envelope, tmp_path)

    def sync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory = Path(os.readlink(f"/proc/self/fd/{fd}"))
            events.append(directory)
            if directory == fault:
                raise OSError(errno.EIO, "injected directory barrier failure")
        else:
            events.append("file")
        real_sync(fd)

    def replace(source: object, destination: object) -> None:
        if Path(destination) == target:  # type: ignore[arg-type]
            assert events == ["file", tmp_path]
            events.append("publish")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    with _running_receiver(tmp_path) as (host, port):

        def post() -> tuple[int, dict[str, object]]:
            response = _request(host, port, "POST", "/v1/browser-captures", body=_payload(), origin=_EXTENSION_ORIGIN)
            return response.status, json.loads(response.read())

        if fault is not None:
            status, body = post()
            assert status == HTTPStatus.INTERNAL_SERVER_ERROR
            assert body["error"] == "write_failed"
            assert target.exists() is (fault_directory == "provider")
            fault = None
            events.clear()
        status, body = post()
        assert status == HTTPStatus.ACCEPTED
        assert body["ok"] is True
        if fault_directory == "provider":
            assert body["deduplicated"] is True
            assert events == ["file", tmp_path, provider_dir]
        else:
            assert events == ["file", tmp_path, "publish", provider_dir]
        assert target.exists()


@pytest.mark.parametrize("outcome", ["accepted", "noop", "superseded"])
def test_http_receipt_preserves_resident_identity_through_backfill_retirement(tmp_path: Path, outcome: str) -> None:
    import copy
    import subprocess

    retained = _payload()
    retained["capture_id"] = "resident-capture"
    retained_session = cast(dict[str, object], retained["session"])
    retained_session["updated_at"] = "2026-04-24T00:05:00Z"
    cast(list[dict[str, object]], retained_session["turns"])[0]["identity_observation"] = {
        "origin": "chatgpt-export",
        "provider_conversation_id": "conv-123",
        "provider_message_id": "u1",
        "adapter_name": "chatgpt-dom-v1",
        "fidelity": "native",
    }
    incoming = copy.deepcopy(retained)
    incoming["capture_id"] = "incoming-capture"
    incoming_session = cast(dict[str, object], incoming["session"])
    if outcome != "noop":
        incoming_session["turns"] = [
            {
                "provider_turn_id": "other-turn",
                "role": "user",
                "text": "Different",
                "identity_observation": {
                    "origin": "chatgpt-export",
                    "provider_conversation_id": "conv-123",
                    "provider_message_id": "other-turn",
                    "adapter_name": "chatgpt-dom-v1",
                    "fidelity": "native",
                },
            }
        ]
        incoming_session["updated_at"] = "2026-04-24T00:01:00Z" if outcome == "superseded" else "2026-04-24T00:06:00Z"
    cast(dict[str, object], incoming["provenance"])["captured_at"] = "2026-04-24T00:06:00Z"
    with _running_receiver(tmp_path) as (host, port):
        first = _request(host, port, "POST", "/v1/browser-captures", body=retained, origin=_EXTENSION_ORIGIN)
        first_receipt = json.loads(first.read())
        assert first.status == HTTPStatus.ACCEPTED
        artifact = tmp_path / first_receipt["artifact_ref"]
        retained_bytes = artifact.read_bytes()
        completed = subprocess.run(
            ["node", str(Path(__file__).parents[2] / "infra/browser_receiver_ack.mjs")],
            input=json.dumps({"endpoint": f"http://{host}:{port}/v1/browser-captures", "capture": incoming}),
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(completed.stdout)
    receipt = result["item"]["receiver_receipt"]
    assert receipt["outcome"] == outcome
    assert receipt["content_hash"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert result["submissions"] == 1
    assert result["staged_files"] == 0
    assert result["item"]["envelope"] is None
    assert result["job"]["status"] == "complete"
    if outcome == "superseded":
        assert artifact.read_bytes() == retained_bytes
        assert receipt["capture_id"] == "chatgpt:resident-capture"
        assert receipt["submitted_content_hash"] != receipt["content_hash"]
        assert [identity["message_ref"].rsplit(":", 1)[-1] for identity in receipt["accepted_identities"]] == ["u1"]
        assert result["item"]["state"] == "superseded"
        assert result["item"]["content_hash"] is None
        assert result["revision"] is None
        assert result["job"]["progress"]["complete"] == 0
        assert result["job"]["progress"]["superseded"] == 1
    else:
        assert result["item"]["state"] == "complete"
        assert result["revision"]["receiver_content_hash"] == receipt["content_hash"]
        assert result["item"]["content_hash"] == receipt["content_hash"]
        if outcome == "noop":
            assert artifact.read_bytes() == retained_bytes
            assert receipt["capture_id"] == "chatgpt:resident-capture"
            assert receipt["submitted_content_hash"] != receipt["content_hash"]
        else:
            assert receipt["content_hash"] == receipt["submitted_content_hash"]
            assert receipt["capture_id"] == "chatgpt:incoming-capture"
