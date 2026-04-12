from __future__ import annotations

from pathlib import Path

from polylogue.lib.raw_payload_decode import build_raw_payload_envelope, sample_jsonl_payload


def test_sample_jsonl_payload_accepts_lone_surrogates_via_stdlib_fallback(tmp_path: Path) -> None:
    path = tmp_path / "surrogate.jsonl"
    path.write_bytes(
        b'{"ok": 1}\n'
        b'{"text":"broken \\udce2 surrogate"}\n'
        b'{"ok": 2}\n'
    )

    samples, malformed = sample_jsonl_payload(path, max_samples=8, jsonl_dict_only=True)

    assert malformed == 0
    assert [sample.get("ok") for sample in samples if "ok" in sample] == [1, 2]
    assert any(sample.get("text") == "broken \udce2 surrogate" for sample in samples)


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
