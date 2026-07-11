"""Focused mutation tests for the registered pytest timeout policy command."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.command_catalog import COMMANDS


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_tree(
    tmp_path: Path,
    *,
    test_source: str = "def test_ok():\n    pass\n",
    devtools_source: str = "COMMAND = ['pytest']\n",
    manifest: str = "",
) -> Path:
    _write(
        tmp_path,
        "pyproject.toml",
        "[tool.pytest.ini_options]\ntimeout = 300\n",
    )
    _write(tmp_path, "tests/unit/test_target.py", test_source)
    _write(tmp_path, "devtools/managed_command.py", devtools_source)
    _write(tmp_path, "devtools/pytest_timeout_overrides.toml", manifest)
    return tmp_path


def _run_registered(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Exercise the catalog-resolved production command, not a test-only helper."""
    command = COMMANDS["verify pytest-timeout-overrides"].resolve_main()
    rc = command(["--root", str(root)])
    return rc, capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import pytest\n\n@pytest.mark.timeout(0)\ndef test_target():\n    pass\n", "must be positive"),
        ("import pytest\n\n@pytest.mark.timeout(-1)\ndef test_target():\n    pass\n", "must be positive"),
        (
            "import pytest\n\nLIMIT = 30\n@pytest.mark.timeout(LIMIT)\ndef test_target():\n    pass\n",
            "dynamic or malformed",
        ),
        ("import pytest\n\n@pytest.mark.timeout()\ndef test_target():\n    pass\n", "malformed"),
        ("import pytest\n\n@pytest.mark.timeout(None)\ndef test_target():\n    pass\n", "unbounded"),
        ("import pytest\n\n@pytest.mark.timeout\ndef test_target():\n    pass\n", "missing a timeout value"),
        ("from pytest import mark\n\n@mark.timeout(None)\ndef test_target():\n    pass\n", "unbounded"),
        ("import pytest as pt\n\n@pt.mark.timeout(0)\ndef test_target():\n    pass\n", "must be positive"),
        ("import pytest\n\n@pytest.mark.timeout(None)\nclass TestTarget:\n    pass\n", "unbounded"),
        ("import pytest\n\npytestmark = pytest.mark.timeout(None)\ndef test_target():\n    pass\n", "unbounded"),
        (
            "import pytest\n\npytestmark = []\npytestmark += [pytest.mark.timeout(None)]\ndef test_target():\n    pass\n",
            "unbounded",
        ),
        (
            "import pytest\n\nCASE = pytest.param(1, marks=pytest.mark.timeout(None))\ndef test_target():\n    pass\n",
            "unbounded",
        ),
        (
            "import pytest as pt\n\nLIMIT = 30\nCASE = pt.param(1, marks=[pt.mark.timeout(LIMIT)])\ndef test_target():\n    pass\n",
            "dynamic or malformed",
        ),
    ],
)
def test_registered_verifier_rejects_invalid_decorator_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
    expected: str,
) -> None:
    """Anti-vacuity: deleting decorator AST parsing makes this production-command test fail."""
    root = _make_tree(tmp_path, test_source=source)

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert expected in output


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\n\nCASE = pytest.param(1, marks=pytest.mark.timeout(30))\ndef test_target():\n    pass\n",
        "from pytest import mark, param\n\nCASE = param(1, marks=[mark.timeout(30), mark.slow])\ndef test_target():\n    pass\n",
    ],
)
def test_registered_verifier_accepts_bounded_pytest_param_marks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
) -> None:
    """Anti-vacuity: removing pytest.param marks scanning makes this production-command test fail."""
    root = _make_tree(tmp_path, test_source=source)

    rc, output = _run_registered(root, capsys)

    assert rc == 0, output


def test_registered_verifier_rejects_module_timeout_mark_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: removing safe marker-alias flattening makes this production-command test fail."""
    root = _make_tree(
        tmp_path,
        test_source=(
            "import pytest\n\nTIMEOUT_MARKS = [pytest.mark.timeout(None)]\npytestmark = TIMEOUT_MARKS\n"
            "def test_target():\n    pass\n"
        ),
    )

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert "unbounded" in output


def test_registered_verifier_rejects_incremental_timeout_mark_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: bypassing alias flattening for pytestmark += makes this production-command test fail."""
    root = _make_tree(
        tmp_path,
        test_source=(
            "import pytest\n\nEXTRA_MARKS = [pytest.mark.timeout(None)]\npytestmark = []\n"
            "pytestmark += EXTRA_MARKS\ndef test_target():\n    pass\n"
        ),
    )

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert "unbounded" in output


def test_registered_verifier_rejects_parameter_timeout_mark_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: removing alias flattening for pytest.param marks makes this production-command test fail."""
    root = _make_tree(
        tmp_path,
        test_source=(
            "import pytest\n\nCASE_MARK = pytest.mark.timeout(0)\n"
            "CASE = pytest.param(1, marks=CASE_MARK)\ndef test_target():\n    pass\n"
        ),
    )

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert "must be positive" in output


def test_registered_verifier_rejects_cyclic_marker_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: removing cycle detection makes this production-command test fail instead of terminating safely."""
    root = _make_tree(
        tmp_path,
        test_source="FIRST = SECOND\nSECOND = FIRST\npytestmark = FIRST\ndef test_target():\n    pass\n",
    )

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert "cyclic pytest marker alias" in output


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\n\nMARKS = [pytest.mark.timeout(30)]\npytestmark = MARKS\ndef test_target():\n    pass\n",
        (
            "import pytest\n\nEXTRA_MARKS = [pytest.mark.timeout(30)]\npytestmark = []\n"
            "pytestmark += EXTRA_MARKS\ndef test_target():\n    pass\n"
        ),
        (
            "import pytest\n\nCASE_MARK = pytest.mark.timeout(30)\n"
            "CASE = pytest.param(1, marks=CASE_MARK)\ndef test_target():\n    pass\n"
        ),
    ],
)
def test_registered_verifier_accepts_bounded_marker_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
) -> None:
    """Anti-vacuity: rejecting safe list/tuple marker aliases makes this production-command test fail."""
    root = _make_tree(tmp_path, test_source=source)

    rc, output = _run_registered(root, capsys)

    assert rc == 0, output


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("COMMAND = ['pytest', '--timeout=0']\n", "must be positive"),
        ("COMMAND = ['pytest', '--timeout=-1']\n", "must be positive"),
        ("LIMIT = '30'\nCOMMAND = ['pytest', '--timeout', LIMIT]\n", "dynamic or malformed"),
        ("LIMIT = 0\nCOMMAND = ['pytest', f'--timeout={LIMIT}']\n", "dynamic or malformed"),
        (
            "LIMIT = 0\nTIMEOUT_ARG = f'--timeout={LIMIT}'\nCOMMAND = ['pytest', TIMEOUT_ARG]\n",
            "dynamic or malformed",
        ),
        (
            "LIMIT = 0\nTIMEOUT_ARGS = [f'--timeout={LIMIT}']\nexecution = pytest_execution(*TIMEOUT_ARGS)\n",
            "dynamic or malformed",
        ),
        ("TIMEOUT_ARGS = ['--timeout=0']\nexecution = pytest_execution(*TIMEOUT_ARGS)\n", "must be positive"),
        ("COMMAND = ['pytest'] + ['--timeout=0']\n", "must be positive"),
        ("COMMAND = ['pytest', *['--timeout=0']]\n", "must be positive"),
        ("COMMAND = ['pytest', '--timeout=forever']\n", "malformed"),
        ("COMMAND = ['pytest', '--timeout']\n", "unbounded"),
        ("execution = pytest_execution('--timeout=')\n", "unbounded"),
    ],
)
def test_registered_verifier_rejects_invalid_managed_command_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
    expected: str,
) -> None:
    """Anti-vacuity: deleting command-literal AST parsing makes this production-command test fail."""
    root = _make_tree(tmp_path, devtools_source=source)

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert expected in output


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\n\n@pytest.mark.timeout(30, method='thread')\ndef test_target():\n    pass\n",
        "import pytest\n\n@pytest.mark.timeout(timeout=30, func_only=True)\ndef test_target():\n    pass\n",
    ],
)
def test_registered_verifier_accepts_supported_static_decorator_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
) -> None:
    """Anti-vacuity: rejecting every keyword option makes this production-command test fail."""
    root = _make_tree(tmp_path, test_source=source)

    rc, output = _run_registered(root, capsys)

    assert rc == 0, output


def test_registered_verifier_scans_nested_devtools_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: replacing the recursive devtools scan with glob() makes this test fail."""
    root = _make_tree(tmp_path)
    _write(root, "devtools/nested/managed_command.py", "COMMAND = ['pytest', '--timeout=0']\n")

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert "must be positive" in output


def test_registered_verifier_requires_exact_rationale_manifest_for_longer_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: removing above-default reconciliation makes this production-command test fail."""
    root = _make_tree(tmp_path, devtools_source="COMMAND = ['pytest', '--timeout=600']\n")

    missing_rc, missing_output = _run_registered(root, capsys)
    _write(
        root,
        "devtools/pytest_timeout_overrides.toml",
        "[[exception]]\npath = 'devtools/managed_command.py'\nvalue = 600\nrationale = 'Bounded full diagnostic.'\n",
    )
    present_rc, present_output = _run_registered(root, capsys)

    assert missing_rc == 1
    assert "without a manifest rationale" in missing_output
    assert present_rc == 0, present_output


def test_registered_verifier_rejects_stale_or_rationaleless_manifest_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anti-vacuity: removing stale-entry or rationale validation makes this production-command test fail."""
    root = _make_tree(
        tmp_path,
        manifest=(
            "[[exception]]\npath = 'devtools/retired.py'\nvalue = 600\nrationale = 'No longer live.'\n"
            "\n[[exception]]\npath = 'devtools/managed_command.py'\nvalue = 700\nrationale = ''\n"
        ),
    )

    rc, output = _run_registered(root, capsys)

    assert rc == 1
    assert "rationale must be non-empty" in output
    assert "stale timeout override manifest entry" in output


def test_committed_registered_verifier_is_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """Anti-vacuity: deleting real-surface scanning or the live 600s manifest entry makes this test fail."""
    command = COMMANDS["verify pytest-timeout-overrides"].resolve_main()

    rc = command([])

    assert rc == 0, capsys.readouterr().out
