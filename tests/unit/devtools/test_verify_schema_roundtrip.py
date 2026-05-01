from __future__ import annotations

import json

import pytest

from devtools import verify_schema_roundtrip


def test_verify_schema_roundtrip_all_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_schema_roundtrip.main(["--all", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["provider_count"] >= 1
    assert payload["package_count"] >= 1
    assert payload["element_count"] >= 1


def test_verify_schema_roundtrip_requires_scope() -> None:
    with pytest.raises(SystemExit):
        verify_schema_roundtrip.main(["--json"])
