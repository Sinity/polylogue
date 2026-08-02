from __future__ import annotations

import json

import pytest

from devtools import verify_raw_payload_hash_purity as lint


def test_scan_flags_the_historical_codex_header_splice_pattern() -> None:
    """polylogue-u19l's actual bug: a json-serialized header literal spliced
    ahead of the captured payload before hashing/storing it."""
    fixture_source = """
def _append_payload_for_provider(path, source_name, payload):
    session_meta = json_dumps({"type": "session_meta", "payload": {"id": identity}}).encode()
    return session_meta + b"\\n" + payload
"""
    violations = lint.scan_source_for_payload_concatenation(fixture_source, path="fixture.py")
    assert len(violations) == 1
    assert violations[0].path == "fixture.py"
    assert "concatenated" in violations[0].detail


def test_scan_flags_reversed_operand_order() -> None:
    """The captured reference can appear on either side of the ``+``."""
    fixture_source = """
def build(payload):
    return payload + b"-trailer"
"""
    violations = lint.scan_source_for_payload_concatenation(fixture_source, path="fixture.py")
    assert len(violations) == 1


def test_scan_ignores_bare_reference_passthrough() -> None:
    """The fixed shape: identity carried as a sidecar return value, payload
    bytes returned untouched -- no concatenation at all."""
    fixture_source = """
def _append_payload_for_provider(path, source_name, payload):
    identity = self._existing_provider_session_id(path)
    return payload, identity
"""
    assert lint.scan_source_for_payload_concatenation(fixture_source, path="fixture.py") == []


def test_scan_ignores_two_literals_joined() -> None:
    """Two literals concatenated (e.g. building a fixed log-message prefix)
    is not the splice-onto-captured-bytes hazard this lint targets."""
    fixture_source = """
def log_prefix():
    return "codex_append_identity_resolved" + "_as_sidecar_hint"
"""
    assert lint.scan_source_for_payload_concatenation(fixture_source, path="fixture.py") == []


def test_scan_ignores_two_references_joined() -> None:
    """Two already-captured buffers joined together (not a synthesize-and-
    splice) is not flagged."""
    fixture_source = """
def combine(prefix_bytes, tail_bytes):
    return prefix_bytes + tail_bytes
"""
    assert lint.scan_source_for_payload_concatenation(fixture_source, path="fixture.py") == []


def test_scan_flags_fstring_splice() -> None:
    fixture_source = """
def build(payload, identity):
    return f"id:{identity}\\n".encode() + payload
"""
    violations = lint.scan_source_for_payload_concatenation(fixture_source, path="fixture.py")
    assert len(violations) == 1


def test_real_write_path_modules_currently_pass(capsys: pytest.CaptureFixture[str]) -> None:
    """The real raw-capture write path (post polylogue-u19l fix, PR #3539)
    must currently be clean -- this locks the fix in as a regression test."""
    assert lint.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["violations"] == []
    assert payload["modules_scanned"] == list(lint.WRITE_PATH_MODULES)


def test_main_reports_violation_and_nonzero_exit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lint,
        "_collect_write_path_violations",
        lambda: [
            lint.HashPurityViolation(
                path="polylogue/sources/live/batch.py",
                lineno=1,
                col_offset=0,
                detail="literal/serialized value concatenated onto a bare reference before hashing",
            )
        ],
    )
    assert lint.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert len(payload["violations"]) == 1
