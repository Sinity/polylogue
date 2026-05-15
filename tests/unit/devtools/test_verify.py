from __future__ import annotations

import sys

from devtools.verify import build_verify_steps


def test_quick_verify_omits_pytest() -> None:
    steps = build_verify_steps(quick=True, lab=False, skip_slow=False)

    labels = [label for label, _command in steps]
    assert labels == [
        "ruff format",
        "ruff check",
        "mypy",
        "render-all",
        "verify-topology",
        "verify-layering",
        "verify-file-budgets",
        "verify-test-ownership",
        "verify-schema-roundtrip",
        "verify-cross-cuts",
        "verify-suppressions",
        "verify-manifests",
        "verify-witness-lifecycle",
        "verify-lane-assertions",
        "proof-pack check",
    ]


def test_full_verify_includes_pytest() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    labels = [label for label, _command in steps]
    assert labels[-1] == "pytest"


def test_lab_verify_delegates_to_lab_scenario() -> None:
    steps = build_verify_steps(quick=True, lab=True, skip_slow=False)

    labels = [label for label, _command in steps]
    assert "lab scenario" in labels
    assert "verify-slos" in labels
    lab_step = next(step for step in steps if step[0] == "lab scenario")
    assert lab_step == (
        "lab scenario",
        [sys.executable, "-m", "devtools", "lab-scenario", "run", "archive-smoke", "--tier", "0"],
    )
