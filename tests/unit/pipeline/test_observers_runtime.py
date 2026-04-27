# mypy: disable-error-code="no-untyped-def,call-arg,arg-type,attr-defined"

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from polylogue.pipeline import observers as observers_module


def _result(*, conversations: int, new: int, changed: int):
    return MagicMock(
        counts={
            "conversations": conversations,
            "new_conversations": new,
            "changed_conversations": changed,
        },
        drift={"new": {"conversations": new}, "changed": {"conversations": changed}},
    )


def test_run_observer_base_methods_are_noops() -> None:
    observer = observers_module.RunObserver()

    observer.on_progress(1, "progress")
    observer.on_completed(_result(conversations=1, new=1, changed=0))
    observer.on_idle(_result(conversations=0, new=0, changed=0))
    observer.on_error(RuntimeError("boom"))


def test_composite_observer_dispatches_progress_idle_and_error_and_logs_failures() -> None:
    good = MagicMock()
    bad = MagicMock()
    bad.on_progress.side_effect = RuntimeError("bad progress")
    bad.on_idle.side_effect = RuntimeError("bad idle")
    bad.on_error.side_effect = RuntimeError("bad error")
    composite = observers_module.CompositeObserver([bad, good])

    with patch("polylogue.pipeline.observers.logger.exception") as log_exception:
        composite.on_progress(1, "step")
        composite.on_idle(_result(conversations=0, new=0, changed=0))
        composite.on_error(RuntimeError("boom"))

    good.on_progress.assert_called_once_with(1, "step")
    good.on_idle.assert_called_once()
    good.on_error.assert_called_once()
    assert log_exception.call_count == 3


def test_notification_observer_includes_new_and_changed_detail_parts() -> None:
    with patch("subprocess.run") as run:
        observers_module.NotificationObserver().on_completed(_result(conversations=3, new=2, changed=1))

    assert "2 new, 1 changed" in run.call_args.args[0][2]


def test_notification_observer_covers_changed_only_and_idle_paths() -> None:
    with patch("subprocess.run") as run:
        observers_module.NotificationObserver().on_completed(_result(conversations=1, new=0, changed=1))

    assert "(1 changed)" in run.call_args.args[0][2]

    with patch("subprocess.run") as run:
        observers_module.NotificationObserver().on_completed(_result(conversations=0, new=0, changed=0))

    run.assert_not_called()


def test_webhook_request_target_and_post_webhook_cover_http_https_and_hostname_errors() -> None:
    public_ip = "93.184.216.34"
    with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=[(2, 1, 6, "", (public_ip, 443))]):
        scheme, host, validated_ip, port, path = observers_module._webhook_request_target(
            "https://example.com/hook?x=1"
        )

    assert (scheme, host, validated_ip, port, path) == ("https", "example.com", public_ip, 443, "/hook?x=1")

    try:
        observers_module._webhook_request_target("http:///missing-host")
    except ValueError as exc:
        assert "hostname" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing hostname")

    response = MagicMock()
    http_connection = MagicMock()
    http_connection.getresponse.return_value = response
    pinned_https = MagicMock()
    pinned_https.getresponse.return_value = response

    with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=[(2, 1, 6, "", (public_ip, 80))]):
        with patch("polylogue.pipeline.observers.http.client.HTTPConnection", return_value=http_connection):
            observers_module._post_webhook("http://example.com/hook", b"{}")
    http_connection.request.assert_called_once()
    # HTTP path connects to the validated IP and restores the Host header.
    request_kwargs = http_connection.request.call_args.kwargs
    assert request_kwargs["headers"]["Host"] == "example.com"
    http_connection.close.assert_called_once()

    with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=[(2, 1, 6, "", (public_ip, 443))]):
        with patch("polylogue.pipeline.observers._PinnedHTTPSConnection", return_value=pinned_https):
            with patch("polylogue.pipeline.observers.ssl.create_default_context", return_value="ctx"):
                observers_module._post_webhook("https://example.com/hook", b"{}")
    pinned_https.request.assert_called_once()
    pinned_https.close.assert_called_once()


def test_validate_webhook_url_rejects_bad_scheme_host_resolution_and_private_ips() -> None:
    with pytest.raises(ValueError, match="http or https"):
        observers_module._validate_webhook_url("ftp://example.com/hook")

    with pytest.raises(ValueError, match="hostname"):
        observers_module._validate_webhook_url("http:///missing-host")

    with patch("polylogue.pipeline.observers.socket.getaddrinfo", side_effect=socket.gaierror(-2, "no host")):
        with pytest.raises(ValueError, match="Cannot resolve webhook hostname"):
            observers_module._validate_webhook_url("https://missing.example/hook")

    with patch(
        "polylogue.pipeline.observers.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("127.0.0.1", 443))],
    ):
        with pytest.raises(ValueError, match="private/reserved address"):
            observers_module._validate_webhook_url("https://localhost/hook")


def test_webhook_observer_logs_valueerror_and_exec_validation_covers_parse_edges() -> None:
    with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 80))]):
        observer = observers_module.WebhookObserver("http://example.com/hook")

    with patch("polylogue.pipeline.observers._post_webhook", side_effect=ValueError("blocked")):
        with patch("polylogue.pipeline.observers.logger.warning") as warning:
            observer.on_completed(_result(conversations=1, new=1, changed=0))
    assert warning.call_args.args[0] == "Webhook blocked for %s: %s"
    assert warning.call_args.args[1] == "http://example.com/hook"
    assert str(warning.call_args.args[2]) == "blocked"

    with patch("polylogue.pipeline.observers.shlex.split", side_effect=ValueError("unterminated")):
        try:
            observers_module._validate_exec_command('echo "unterminated')
        except ValueError as exc:
            assert "Cannot parse exec command" in str(exc)
        else:
            raise AssertionError("expected parse failure")

    with patch("polylogue.pipeline.observers.shlex.split", return_value=[]):
        try:
            observers_module._validate_exec_command("echo")
        except ValueError as exc:
            assert "empty argument list" in str(exc)
        else:
            raise AssertionError("expected empty argv failure")


def test_webhook_observer_skips_idle_and_logs_generic_failures() -> None:
    with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 80))]):
        observer = observers_module.WebhookObserver("http://example.com/hook")

    with patch("polylogue.pipeline.observers._post_webhook") as post_webhook:
        observer.on_completed(_result(conversations=0, new=0, changed=0))

    post_webhook.assert_not_called()

    with patch("polylogue.pipeline.observers._post_webhook", side_effect=RuntimeError("boom")):
        with patch("polylogue.pipeline.observers.logger.warning") as warning:
            observer.on_completed(_result(conversations=1, new=0, changed=1))

    assert warning.call_args.args[0] == "Webhook failed for %s: %s"
    assert warning.call_args.args[1] == "http://example.com/hook"
    assert str(warning.call_args.args[2]) == "boom"


def test_validate_exec_command_rejects_empty_and_unsafe_and_exec_observer_runs_safe_command() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        observers_module._validate_exec_command("  ")

    with pytest.raises(ValueError, match="unsafe shell metacharacters"):
        observers_module._validate_exec_command("echo hello; rm -rf /")

    assert observers_module._validate_exec_command("echo hello") == ["echo", "hello"]

    with patch("subprocess.run") as run:
        observers_module.ExecObserver("echo hello").on_completed(_result(conversations=2, new=1, changed=1))

    run.assert_called_once()
    assert run.call_args.args[0] == ["echo", "hello"]
    assert run.call_args.kwargs["env"]["POLYLOGUE_ACTIVITY_COUNT"] == "2"
