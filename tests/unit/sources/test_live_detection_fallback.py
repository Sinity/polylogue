"""Live provider detection separates a crash fallback from a shape fallback.

Both outcomes store the source's fallback provider, so only the evidence can
say that detection crashed (polylogue-fkqxx).

Anti-vacuity: return ``None`` evidence from the ``except`` branches of
``detect_provider_from_path_sample_evidence`` and the undecodable document
reads as an ordinary shape fallback.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.sources.live.batch_support import detect_provider_from_path_sample_evidence


def test_undecodable_document_reports_the_crash(tmp_path: Path) -> None:
    document = tmp_path / "capture.json"
    document.write_bytes(b'{"unterminated": [1, 2')

    provider, crash = detect_provider_from_path_sample_evidence(document, Provider.CHATGPT, json_document=True)

    assert provider is Provider.CHATGPT
    assert crash is not None


def test_unclaimed_document_is_a_shape_fallback(tmp_path: Path) -> None:
    document = tmp_path / "capture.json"
    document.write_bytes(b'{"neutral": "value"}')

    provider, crash = detect_provider_from_path_sample_evidence(document, Provider.CHATGPT, json_document=True)

    assert provider is Provider.CHATGPT
    assert crash is None
