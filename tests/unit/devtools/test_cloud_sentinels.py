"""Contracts for the cloud-sandbox environment sentinels.

`.claude/settings.json` is applied in every Claude Code session, so its
cloud-only `env` block leaks onto this workstation. The property under test is
that a sentinel is recognised by its exact CLOUD VALUE and declined only off a
sandbox -- never by variable name, so a deliberate operator override always
survives.
"""

from __future__ import annotations

import pytest

from devtools import cloud_sentinels
from devtools.cloud_sentinels import CLOUD_SENTINELS, INDISTINGUISHABLE_SENTINELS, cloud_sentinel_declined


@pytest.fixture
def on_workstation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_sentinels, "running_in_cloud_sandbox", lambda: False)


@pytest.fixture
def in_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_sentinels, "running_in_cloud_sandbox", lambda: True)


@pytest.mark.usefixtures("on_workstation")
def test_every_distinguishable_sentinel_is_declined_on_a_workstation() -> None:
    for name, value in CLOUD_SENTINELS.items():
        if name in INDISTINGUISHABLE_SENTINELS:
            continue
        assert cloud_sentinel_declined(name, value), f"{name} leaks onto the workstation"


@pytest.mark.usefixtures("on_workstation")
def test_an_indistinguishable_sentinel_is_never_declined() -> None:
    """POLYLOGUE_FORCE_PLAIN=1 is what anyone wanting plain output writes, and
    this repo's demo, proof and lab-scenario paths set exactly that deliberately.
    Declining it by value would override them, so the leak stays open and
    cosmetic rather than being closed incorrectly."""
    for name in INDISTINGUISHABLE_SENTINELS:
        assert not cloud_sentinel_declined(name, CLOUD_SENTINELS[name])


@pytest.mark.usefixtures("in_sandbox")
def test_no_sentinel_is_declined_inside_a_sandbox() -> None:
    """In a real sandbox these values are exactly right."""
    for name, value in CLOUD_SENTINELS.items():
        assert not cloud_sentinel_declined(name, value)


@pytest.mark.usefixtures("on_workstation")
def test_a_deliberate_override_is_never_mistaken_for_the_sentinel() -> None:
    """Recognition is by value, not by name -- otherwise the scrub would silently
    override an operator who meant it."""
    assert not cloud_sentinel_declined("POLYLOGUE_PYTEST_WORKERS", "12")
    assert not cloud_sentinel_declined("POLYLOGUE_ARCHIVE_ROOT", "/realm/state/polylogue")
    assert not cloud_sentinel_declined("HYPOTHESIS_PROFILE", "thorough")


@pytest.mark.usefixtures("on_workstation")
def test_unknown_variables_and_absent_values_are_left_alone() -> None:
    assert not cloud_sentinel_declined("SOMETHING_ELSE", "1")
    assert not cloud_sentinel_declined("POLYLOGUE_PYTEST_WORKERS", None)
    assert not cloud_sentinel_declined("POLYLOGUE_PYTEST_WORKERS", "")


def test_the_enumeration_matches_the_settings_file_that_causes_the_leak() -> None:
    """The set is only useful if it is complete. If settings.json grows a new
    cloud value, this fails rather than letting a fifth leak sit unnoticed while
    four are closed."""
    import json
    from pathlib import Path

    settings = json.loads((Path(__file__).resolve().parents[3] / ".claude/settings.json").read_text(encoding="utf-8"))

    assert dict(settings.get("env") or {}) == CLOUD_SENTINELS
