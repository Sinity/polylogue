"""Helper utilities for CLI testing."""

from pathlib import Path
from typing import Any

from click.testing import CliRunner, Result


def invoke_command(
    command: Any,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    workspace: Path | None = None,
    plain_mode: bool = True,
) -> Result:
    """
    Invoke a Click command with isolated environment.

    Args:
        command: Click command to invoke
        args: Command line arguments (default: [])
        env: Environment variables (default: {})
        workspace: Workspace directory path (sets POLYLOGUE_ARCHIVE_ROOT)
        plain_mode: Enable plain output mode (POLYLOGUE_FORCE_PLAIN=1)

    Returns:
        Click Result object

    Example:
        >>> result = invoke_command(run_cmd, ["--preview"], workspace=tmp_path)
        >>> assert result.exit_code == 0
    """
    runner = CliRunner()
    args = args or []
    env = dict(env) if env else {}

    # Set workspace if provided
    if workspace:
        env["POLYLOGUE_ARCHIVE_ROOT"] = str(workspace)

    # Force plain output for predictable testing
    if plain_mode:
        env["POLYLOGUE_FORCE_PLAIN"] = "1"

    return runner.invoke(command, args, env=env, catch_exceptions=False)


def assert_cli_success(result: Result, expected_output: str | None = None) -> None:
    """
    Assert CLI command succeeded.

    Args:
        result: Click Result from invoke_command
        expected_output: Optional substring expected in output

    Raises:
        AssertionError: If command failed or output doesn't match

    Example:
        >>> result = invoke_command(config_cmd, ["show"])
        >>> assert_cli_success(result, "archive_root")
    """
    if result.exit_code != 0:
        raise AssertionError(
            f"Command failed with exit code {result.exit_code}\n"
            f"Output: {result.output}\n"
            f"Stderr: {result.stderr if hasattr(result, 'stderr') else 'N/A'}"
        )

    if expected_output and expected_output not in result.output:
        raise AssertionError(
            f"Expected output '{expected_output}' not found in:\n{result.output}"
        )


def assert_cli_error(result: Result, expected_message: str | None = None, expected_code: int = 1) -> None:
    """
    Assert CLI command failed with expected error.

    Args:
        result: Click Result from invoke_command
        expected_message: Optional substring expected in error output
        expected_code: Expected exit code (default: 1)

    Raises:
        AssertionError: If command succeeded or error doesn't match

    Example:
        >>> result = invoke_command(run_cmd, ["--invalid"])
        >>> assert_cli_error(result, "Invalid option")
    """
    if result.exit_code == 0:
        raise AssertionError(
            f"Expected command to fail, but it succeeded\nOutput: {result.output}"
        )

    if result.exit_code != expected_code:
        raise AssertionError(
            f"Expected exit code {expected_code}, got {result.exit_code}\n"
            f"Output: {result.output}"
        )

    if expected_message:
        combined_output = result.output + (result.stderr if hasattr(result, "stderr") else "")
        if expected_message not in combined_output:
            raise AssertionError(
                f"Expected error message '{expected_message}' not found in:\n{combined_output}"
            )
