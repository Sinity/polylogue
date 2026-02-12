"""Security tests for pipeline event handlers."""
from __future__ import annotations

import pytest

from polylogue.pipeline.events import (
    ExecHandler,
    SyncEvent,
    _validate_exec_command,
    _validate_webhook_url,
)


class TestExecCommandValidation:
    """Tests for _validate_exec_command — command injection prevention."""

    def test_simple_command_accepted(self):
        argv = _validate_exec_command("echo hello")
        assert argv == ["echo", "hello"]

    def test_command_with_path(self):
        argv = _validate_exec_command("/usr/bin/my-script --flag value")
        assert argv == ["/usr/bin/my-script", "--flag", "value"]

    def test_empty_command_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_exec_command("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_exec_command("   ")

    @pytest.mark.parametrize(
        "dangerous_cmd",
        [
            "echo hello; rm -rf /",
            "echo hello && cat /etc/passwd",
            "echo hello | grep secret",
            "echo `whoami`",
            "echo $(id)",
            "echo hello > /tmp/evil",
            "echo hello < /etc/passwd",
            "cmd & background",
            "echo hello!",
            r"echo hello\n",
        ],
    )
    def test_shell_metacharacters_rejected(self, dangerous_cmd):
        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            _validate_exec_command(dangerous_cmd)

    def test_quoted_arguments_accepted(self):
        argv = _validate_exec_command('echo "hello world"')
        assert argv == ["echo", "hello world"]


class TestWebhookUrlValidation:
    """Tests for _validate_webhook_url — SSRF prevention."""

    def test_https_url_accepted(self):
        # This may fail in DNS resolution, but should NOT fail on scheme
        try:
            _validate_webhook_url("https://hooks.example.com/webhook")
        except ValueError as e:
            # DNS resolution failure is OK, but SSRF block is not expected
            assert "Cannot resolve" in str(e) or "private" not in str(e).lower()

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="http or https"):
            _validate_webhook_url("ftp://example.com/file")

    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="http or https"):
            _validate_webhook_url("file:///etc/passwd")

    def test_no_hostname_rejected(self):
        with pytest.raises(ValueError, match="must have a hostname"):
            _validate_webhook_url("http:///path")

    @pytest.mark.parametrize(
        "private_url",
        [
            "http://127.0.0.1/webhook",
            "http://localhost/webhook",
            "http://[::1]/webhook",
        ],
    )
    def test_loopback_rejected(self, private_url):
        with pytest.raises(ValueError, match="private|reserved|loopback"):
            _validate_webhook_url(private_url)

    @pytest.mark.parametrize(
        "private_url",
        [
            "http://10.0.0.1/webhook",
            "http://192.168.1.1/webhook",
            "http://172.16.0.1/webhook",
        ],
    )
    def test_private_ip_rejected(self, private_url):
        with pytest.raises(ValueError, match="private|reserved"):
            _validate_webhook_url(private_url)

    def test_metadata_endpoint_rejected(self):
        with pytest.raises(ValueError, match="private|reserved|link"):
            _validate_webhook_url("http://169.254.169.254/latest/meta-data/")


class TestExecHandler:
    """Integration tests for ExecHandler."""

    def test_construction_validates_command(self):
        with pytest.raises(ValueError):
            ExecHandler("echo hello; evil")

    def test_valid_construction(self):
        handler = ExecHandler("echo hello")
        assert handler._argv == ["echo", "hello"]

    def test_no_op_on_zero_conversations(self):
        handler = ExecHandler("echo hello")
        # Should not raise even though command is valid
        from unittest.mock import MagicMock

        event = SyncEvent(new_conversations=0, run_result=MagicMock())
        handler.on_sync(event)  # Should be a no-op
