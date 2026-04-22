"""Compatibility wrapper for showcase baseline verification.

Routine callers should prefer:

    devtools lab-scenario verify-baselines

This module remains as the stable generated-surface check entrypoint and
delegates the baseline semantics to the verification-lab scenario surface.
"""

from __future__ import annotations

import sys

from devtools import lab_scenario
from polylogue.showcase.exercises import Exercise

BASELINE_DIR = lab_scenario.BASELINE_DIR


def _sync_baseline_dir() -> None:
    lab_scenario.BASELINE_DIR = BASELINE_DIR


def get_tier_0_exercises() -> list[Exercise]:
    return lab_scenario.get_tier_0_exercises()


def run_tier_0() -> dict[str, str]:
    return lab_scenario.run_tier_0()


def load_baselines() -> dict[str, str]:
    _sync_baseline_dir()
    return lab_scenario.load_baselines()


def save_baselines(outputs: dict[str, str]) -> None:
    _sync_baseline_dir()
    lab_scenario.save_baselines(outputs)


def compare_outputs(
    current: dict[str, str],
    baselines: dict[str, str],
) -> list[str]:
    return lab_scenario.compare_outputs(current, baselines)


def verify_showcase_baselines(*, update: bool) -> int:
    _sync_baseline_dir()
    return lab_scenario.verify_showcase_baselines(update=update)


def main(argv: list[str] | None = None) -> int:
    _sync_baseline_dir()
    args = argv or []
    return lab_scenario.main(["verify-baselines", *args])


__all__ = [
    "BASELINE_DIR",
    "compare_outputs",
    "get_tier_0_exercises",
    "load_baselines",
    "main",
    "run_tier_0",
    "save_baselines",
    "verify_showcase_baselines",
]


if __name__ == "__main__":
    sys.exit(main())
