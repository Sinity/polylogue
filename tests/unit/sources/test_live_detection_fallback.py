"""Live provider detection separates a crash fallback from a shape fallback.

Both outcomes store the source's fallback provider, so only the evidence can
say that detection crashed (polylogue-fkqxx).

Anti-vacuity: return ``None`` evidence from the ``except`` branches of
``detect_provider_from_path_evidence`` and the undecodable document
reads as an ordinary shape fallback.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.sources.live.batch_support import detect_provider_from_path_evidence


def test_undecodable_document_reports_the_crash(tmp_path: Path) -> None:
    document = tmp_path / "capture.json"
    document.write_bytes(b'{"unterminated": [1, 2')

    provider, crash = detect_provider_from_path_evidence(document, Provider.CHATGPT, json_document=True)

    assert provider is Provider.CHATGPT
    assert crash is not None


def test_unclaimed_document_is_a_shape_fallback(tmp_path: Path) -> None:
    document = tmp_path / "capture.json"
    document.write_bytes(b'{"neutral": "value"}')

    provider, crash = detect_provider_from_path_evidence(document, Provider.CHATGPT, json_document=True)

    assert provider is Provider.CHATGPT
    assert crash is None


def test_undecodable_jsonl_reports_the_crash(tmp_path: Path) -> None:
    """A JSONL file with no decodable record is a crash fallback, not a shape one."""
    stream = tmp_path / "session.jsonl"
    stream.write_bytes(b"{not json\n{also not json\n")

    provider, crash = detect_provider_from_path_evidence(stream, Provider.CODEX)

    assert provider is Provider.CODEX
    assert crash is not None


def test_jsonl_without_a_failed_record_is_a_shape_fallback(tmp_path: Path) -> None:
    """Empty, blank and large valid records are shape outcomes, not failures."""
    from polylogue.archive.raw_payload.decode import JSONL_RECORD_INSPECTION_BYTES

    oversized = b'{"type": "note", "text": "' + b"x" * (JSONL_RECORD_INSPECTION_BYTES + 1) + b'"}\n'
    for name, content in (
        ("empty.jsonl", b""),
        ("blank.jsonl", b"\n  \n\t\n"),
        ("oversized.jsonl", oversized * 2),
    ):
        stream = tmp_path / name
        stream.write_bytes(content)

        provider, crash = detect_provider_from_path_evidence(stream, Provider.CODEX)

        assert provider is Provider.CODEX
        assert crash is None
