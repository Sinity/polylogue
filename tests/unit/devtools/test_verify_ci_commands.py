from __future__ import annotations

from pathlib import Path

from devtools.verify_ci_commands import validate_ci_commands


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_current_ci_devtools_commands_match_the_catalog() -> None:
    from devtools import repo_root

    assert validate_ci_commands(repo_root()) == ()


def test_unknown_github_run_command_fails_without_scanning_prose(tmp_path: Path) -> None:
    _write(
        tmp_path,
        ".github/workflows/ci.yml",
        """
name: devtools imaginary prose is not executable
jobs:
  test:
    steps:
      - name: devtools also-imaginary
        run: uv run devtools verify definitely-unknown
""",
    )

    assert validate_ci_commands(tmp_path) == (
        ".github/workflows/ci.yml: unknown devtools command 'verify definitely-unknown'",
    )


def test_circle_command_mapping_is_validated(tmp_path: Path) -> None:
    _write(
        tmp_path,
        ".circleci/config.yml",
        """
version: 2.1
jobs:
  gate:
    steps:
      - run:
          name: broken
          command: uv run devtools nonexistent-command
""",
    )

    assert validate_ci_commands(tmp_path) == (".circleci/config.yml: unknown devtools command 'nonexistent-command'",)


def test_invalid_workflow_yaml_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, ".github/workflows/bad.yml", ": invalid: [")

    (error,) = validate_ci_commands(tmp_path)
    assert error.startswith(".github/workflows/bad.yml: invalid YAML:")
