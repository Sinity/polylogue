"""Import preflight truthfulness tests for unsupported/degraded shapes."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from polylogue.sources.import_preflight import ImportPreflightStatus, preflight_import_source
from polylogue.types import Provider


def _chatgpt_payload() -> dict[str, object]:
    return {
        "title": "Supported ChatGPT fixture",
        "mapping": {
            "root": {
                "id": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
                "children": [],
            }
        },
    }


def test_preflight_accepts_supported_json_file(tmp_path: Path) -> None:
    source = tmp_path / "chatgpt.json"
    source.write_text(json.dumps(_chatgpt_payload()))

    result = preflight_import_source(source)

    assert result.status is ImportPreflightStatus.SUPPORTED
    assert result.admissible is True
    assert result.supported_count == 1
    assert result.providers == (Provider.CHATGPT,)
    assert result.error_code == ""


def test_preflight_rejects_unknown_json_shape(tmp_path: Path) -> None:
    source = tmp_path / "unknown.json"
    source.write_text(json.dumps({"not": "an export"}))

    result = preflight_import_source(source)

    assert result.status is ImportPreflightStatus.UNSUPPORTED
    assert result.admissible is False
    assert result.error_code == "unsupported_import_source"
    assert result.unsupported_count == 1
    assert "unsupported" in result.summary()


def test_preflight_rejects_malformed_json(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text('{"mapping": ')

    result = preflight_import_source(source)

    assert result.status is ImportPreflightStatus.MALFORMED
    assert result.admissible is False
    assert result.error_code == "malformed_import_source"
    assert result.malformed_count == 1


def test_preflight_accepts_supported_zip_member(tmp_path: Path) -> None:
    source = tmp_path / "export.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("conversations.json", json.dumps(_chatgpt_payload()))

    result = preflight_import_source(source)

    assert result.status is ImportPreflightStatus.SUPPORTED
    assert result.supported_count == 1
    assert result.providers == (Provider.CHATGPT,)
    assert result.samples == ("export.zip:conversations.json: chatgpt",)


def test_preflight_marks_mixed_directory_as_degraded(tmp_path: Path) -> None:
    source = tmp_path / "fixture-world"
    source.mkdir()
    (source / "chatgpt.json").write_text(json.dumps(_chatgpt_payload()))
    (source / "unknown.json").write_text(json.dumps({"not": "an export"}))

    result = preflight_import_source(source)

    assert result.status is ImportPreflightStatus.DEGRADED
    assert result.admissible is True
    assert result.supported_count == 1
    assert result.unsupported_count == 1
    assert "degraded" in result.summary()


def test_preflight_rejects_zip_without_parseable_members(tmp_path: Path) -> None:
    source = tmp_path / "notes.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("README.txt", "not an export")

    result = preflight_import_source(source)

    assert result.status is ImportPreflightStatus.UNSUPPORTED
    assert result.admissible is False
    assert result.error_code == "unsupported_import_source"
