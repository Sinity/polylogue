from __future__ import annotations

import json

import pytest

from devtools import verify_position_derived_identity as lint


def test_scan_flags_inline_fstring_identity() -> None:
    fixture_source = """
def parse(records):
    for index, record in enumerate(records):
        provider_message_id = f"msg-{index}"
"""
    findings = lint._scan_module(fixture_source, rel_path="fixture.py")
    assert len(findings) == 1
    assert findings[0].field == "provider_message_id"
    assert findings[0].qualname == "fixture.py:parse:provider_message_id"


def test_scan_flags_keyword_argument_construction() -> None:
    fixture_source = """
def build(records):
    for index, record in enumerate(records):
        messages.append(ParsedMessage(provider_message_id=f"msg-{index}", role=role))
"""
    findings = lint._scan_module(fixture_source, rel_path="fixture.py")
    assert len(findings) == 1


def test_scan_follows_bare_name_to_local_binding() -> None:
    """The dominant real-codebase shape: build a local var first (``msg_id =
    ... or f"msg-{idx}"``), then pass the bare name as the identity kwarg a
    few lines later (chatgpt.py/base_support.py/drive.py's actual shape)."""
    fixture_source = """
def build(records):
    for idx, record in enumerate(records):
        msg_id = record.get("id") or f"msg-{idx}"
        messages.append(ParsedMessage(provider_message_id=msg_id, role=role))
"""
    findings = lint._scan_module(fixture_source, rel_path="fixture.py")
    assert len(findings) == 1


def test_scan_ignores_provider_native_identity() -> None:
    """A real provider-native id (no index fallback at all) is not flagged."""
    fixture_source = """
def build(records):
    for index, record in enumerate(records):
        provider_message_id = record["uuid"]
"""
    assert lint._scan_module(fixture_source, rel_path="fixture.py") == []


def test_scan_ignores_index_used_for_unrelated_purpose() -> None:
    """An ``index`` variable used for something other than an in-scope
    identity field name must not be flagged -- only IDENTITY_FIELD_NAMES
    trips this lint."""
    fixture_source = """
def build(records):
    for index, record in enumerate(records):
        log.info(f"processing record {index}")
        display_label = f"item-{index}"
"""
    assert lint._scan_module(fixture_source, rel_path="fixture.py") == []


def test_scan_deduplicates_multiple_findings_in_one_function_with_ordinal() -> None:
    fixture_source = """
def build_a(records):
    for index, record in enumerate(records):
        provider_message_id = f"a-{index}"

def build_a_second(records):
    for index, record in enumerate(records):
        provider_message_id = f"b-{index}"
"""
    findings = lint._scan_module(fixture_source, rel_path="fixture.py")
    assert {f.qualname for f in findings} == {
        "fixture.py:build_a:provider_message_id",
        "fixture.py:build_a_second:provider_message_id",
    }


def test_real_parsers_directory_is_clean_without_acknowledgements(capsys: pytest.CaptureFixture[str]) -> None:
    """A clean parser audit has no manifest; new findings must fail or use
    a temporary ``--ack`` entry with a tracked follow-up."""
    assert not lint.MANIFEST_PATH.exists()
    assert lint.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["unacknowledged"] == []
    assert payload["stale"] == []


def test_main_reports_unacknowledged_finding_and_nonzero_exit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_finding = lint.PositionIdentityFinding(
        qualname="fixture.py:build:provider_message_id",
        path="fixture.py",
        lineno=3,
        field="provider_message_id",
    )
    monkeypatch.setattr(lint, "collect_position_derived_identity", lambda: {fake_finding.qualname: fake_finding})
    monkeypatch.setattr(lint, "load_manifest", lambda: {})
    assert lint.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["unacknowledged"] == [fake_finding.qualname]


def test_ack_validation_rejects_short_reason_and_bad_ref() -> None:
    assert lint._validate_ack("too short", "polylogue-abcd") is not None
    assert lint._validate_ack("a sufficiently long justification here", "not-a-valid-ref") is not None
    assert lint._validate_ack("a sufficiently long justification here", "polylogue-abcd") is None
    assert lint._validate_ack("a sufficiently long justification here", "#123") is None
