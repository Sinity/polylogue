from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import live_provider_profile_provision_service as provision


def _source_profile(root: Path) -> Path:
    (root / "Default" / "Network").mkdir(parents=True)
    (root / "Local State").write_text('{"profile": "state"}', encoding="utf-8")
    (root / "Default" / "Network" / "Cookies").write_text("cookies", encoding="utf-8")
    (root / "Default" / "Preferences").write_text("preferences", encoding="utf-8")
    (root / "Default" / "SingletonLock").write_text("ignored", encoding="utf-8")
    return root


def test_profile_provision_reports_missing_source_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"missing required path\(s\): Local State, Default"):
        provision._copy_selected_profile(source_root=tmp_path / "absent", destination_root=tmp_path / "destination")


def test_profile_provision_copies_fixed_profile_without_live_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_profile(tmp_path / "current-chrome")
    destination = tmp_path / "proof-profile"
    monkeypatch.setattr(provision, "_SOURCE_ROOT", source)
    monkeypatch.setattr(provision, "_DESTINATION_ROOT", destination)
    monkeypatch.setattr(provision, "require_declared_operation_context", lambda operation: f"unit-{operation}")

    assert provision.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["profile"]["files"] == 3
    assert (destination / "Local State").read_text(encoding="utf-8") == '{"profile": "state"}'
    assert (destination / "Default" / "Network" / "Cookies").read_text(encoding="utf-8") == "cookies"
    assert not (destination / "Default" / "SingletonLock").exists()
