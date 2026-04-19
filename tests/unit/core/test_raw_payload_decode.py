from __future__ import annotations

from pathlib import Path
from typing import cast

from polylogue.lib.raw_payload_decode import JSONValue, build_raw_payload_envelope, sample_jsonl_payload


def _as_dict(sample: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(sample, dict)
    return sample


def test_sample_jsonl_payload_accepts_lone_surrogates_via_stdlib_fallback(tmp_path: Path) -> None:
    path = tmp_path / "surrogate.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"text":"broken \\udce2 surrogate"}\n{"ok": 2}\n')

    samples, malformed = sample_jsonl_payload(path, max_samples=8, jsonl_dict_only=True)

    dict_samples = [_as_dict(sample) for sample in samples]

    assert malformed == 0
    assert [cast(int | None, sample.get("ok")) for sample in dict_samples if "ok" in sample] == [1, 2]
    assert any(sample.get("text") == "broken \udce2 surrogate" for sample in dict_samples)


def test_build_raw_payload_envelope_reports_first_bad_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\n{"broken": \n{"ok": 2}\n', encoding="utf-8")

    envelope = build_raw_payload_envelope(
        path,
        source_path=str(path),
        fallback_provider="codex",
        jsonl_dict_only=True,
    )

    assert envelope.malformed_jsonl_lines == 1
    assert envelope.malformed_jsonl_detail is not None
    assert "line 2" in envelope.malformed_jsonl_detail
