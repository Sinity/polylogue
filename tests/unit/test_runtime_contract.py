from __future__ import annotations

import sys
from pathlib import Path

import pytest

import polylogue.runtime as runtime_mod
from devtools.runtime_census import collect_census
from devtools.runtime_census import report as census_report
from devtools.runtime_contract import main as runtime_check
from polylogue.runtime import (
    ExtensionProbe,
    RuntimeContractError,
    probe_extensions,
    require_free_threaded_runtime,
    runtime_identity,
)


def test_runtime_identity_is_positive_on_the_managed_interpreter() -> None:
    identity = runtime_identity()
    assert identity.implementation == "cpython"
    assert identity.version[:2] >= (3, 14)
    assert identity.free_threaded is True


def test_gil_enabled_substitution_fails_before_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)
    with pytest.raises(RuntimeContractError, match="requires CPython 3.14 free-threading"):
        require_free_threaded_runtime(consumer="test")


def test_missing_free_threading_probe_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sys, "_is_gil_enabled", raising=False)
    with pytest.raises(RuntimeContractError, match="probe .* unavailable"):
        require_free_threaded_runtime(consumer="test")


def test_incompatible_extension_substitution_fails_before_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_mod,
        "probe_extensions",
        lambda: (ExtensionProbe("fake_extension", True, "fake.so", ("fake.cpython-313-x86_64-linux-gnu.so",), False),),
    )
    with pytest.raises(RuntimeContractError, match="fake_extension"):
        require_free_threaded_runtime(consumer="test")


def test_required_extensions_are_importable() -> None:
    probes = probe_extensions()
    assert probes
    assert all(probe.safe for probe in probes)


def test_runtime_check_is_managed_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert runtime_check(["--json"]) == 0
    assert '"free_threaded": true' in capsys.readouterr().out


def test_runtime_census_classifies_every_discovered_item() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = census_report(root)
    item_count = payload["item_count"]
    assert isinstance(item_count, int)
    assert item_count > 0
    assert payload["unexplained_count"] == 0
    assert payload["pass"] is True
    assert any(item.disposition == "transactionally-serialized" for item in collect_census(root))
