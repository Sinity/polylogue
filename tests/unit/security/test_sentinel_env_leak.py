"""Runtime sentinel test: secret-shaped env vars never appear in output.

Companion to ``test_no_secret_leak_in_logs.py``. The static scan catches
*direct* interpolation of names like ``api_key`` into log calls. This
runtime test guards against the harder case: a future error handler that
serializes ``os.environ`` or a config object into a traceback / panic
message.

Strategy:

1. Set a known sentinel value on a secret-shaped env var
   (``POLYLOGUE_TEST_SENTINEL_TOKEN``).
2. Invoke the CLI with both well-formed and malformed commands that
   exercise the error path (unknown command, bad flag, missing arg).
3. Assert the sentinel value never appears in stdout or stderr.

If a regression ever causes the daemon, CLI, or MCP layer to dump env
contents into a user-facing error, this test fails loudly.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli

_SENTINEL_NAME = "POLYLOGUE_TEST_SENTINEL_TOKEN"
_SENTINEL_VALUE = "tk-ZQXWVPLR-do-not-leak-7f3a9e21-Bearer"


@pytest.fixture
def runner_with_sentinel(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv(_SENTINEL_NAME, _SENTINEL_VALUE)
    # Also seed conventional secret env names that careless serializers
    # might pick up. The sentinel value is identical so a single output
    # scan catches any of them.
    for name in (
        "POLYLOGUE_API_KEY",
        "POLYLOGUE_AUTH_TOKEN",
        "VOYAGE_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.setenv(name, _SENTINEL_VALUE)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    return CliRunner()


_COMMANDS_TO_PROBE: tuple[tuple[str, ...], ...] = (
    ("--help",),
    ("--version",),
    ("run", "--definitely-not-a-flag"),
    ("run", "no-such-subcommand"),
    ("stats", "--definitely-not-a-flag"),
    ("list", "--limit", "not-an-integer"),
    ("count", "--definitely-not-a-flag"),
    ("auth", "--help"),
    ("schema", "--help"),
    ("import", "--help"),
    ("tags", "--help"),
)


@pytest.mark.parametrize("argv", _COMMANDS_TO_PROBE)
def test_cli_output_never_contains_sentinel(runner_with_sentinel: CliRunner, argv: tuple[str, ...]) -> None:
    result = runner_with_sentinel.invoke(cli, list(argv), catch_exceptions=False)
    combined = (result.output or "") + (
        result.stderr_bytes.decode("utf-8", errors="replace") if hasattr(result, "stderr_bytes") else ""
    )
    assert _SENTINEL_VALUE not in combined, (
        f"Sentinel value leaked into CLI output for argv={argv!r}.\nOutput excerpt: {combined[:500]!r}"
    )


def test_logger_repr_of_environ_does_not_leak(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A logger that accidentally repr()'s os.environ would dump the sentinel."""
    monkeypatch.setenv(_SENTINEL_NAME, _SENTINEL_VALUE)
    import logging

    from polylogue.logging import get_logger

    logger = get_logger("polylogue.test_sentinel_env_leak")
    with caplog.at_level(logging.DEBUG, logger="polylogue.test_sentinel_env_leak"):
        # Simulate the normal application code path: log structured context,
        # never raw environ. This line is the *positive* control — it must
        # not contain the sentinel even though the env is set.
        logger.info("running with arg count=%d", 3)
    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert _SENTINEL_VALUE not in combined, (
        "Default logger surface emitted the sentinel environment value; "
        "investigate whether a structured-logging extension is serializing "
        "os.environ."
    )
