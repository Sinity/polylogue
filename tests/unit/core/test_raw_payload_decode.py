from __future__ import annotations

from pathlib import Path

from polylogue.archive.raw_payload.decode import (
    JSONValue,
    _sample_jsonl_payload_with_detail,
    build_raw_payload_envelope,
    sample_jsonl_payload,
)


def _as_dict(sample: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(sample, dict)
    return sample


def _optional_int(value: JSONValue) -> int | None:
    assert value is None or isinstance(value, int)
    return value


def test_sample_jsonl_payload_accepts_lone_surrogates_via_stdlib_fallback(tmp_path: Path) -> None:
    path = tmp_path / "surrogate.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"text":"broken \\udce2 surrogate"}\n{"ok": 2}\n')

    samples, malformed = sample_jsonl_payload(path, max_samples=8, jsonl_dict_only=True)

    dict_samples = [_as_dict(sample) for sample in samples]

    assert malformed == 0
    assert [_optional_int(sample.get("ok")) for sample in dict_samples if "ok" in sample] == [1, 2]
    assert any(sample.get("text") == "broken \udce2 surrogate" for sample in dict_samples)


def test_raw_json_payload_preserves_utf8_encoded_lone_surrogates(tmp_path: Path) -> None:
    """Historical provider bytes may encode a lone UTF-16 surrogate directly."""
    path = tmp_path / "surrogate.json"
    path.write_bytes(b'{"text":"broken \xed\xa0\x80 provider value"}')

    envelope = build_raw_payload_envelope(
        path,
        source_path=str(path),
        fallback_provider="hermes",
    )

    payload = _as_dict(envelope.payload)
    assert payload["text"] == "broken \ud800 provider value"


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


def test_jsonl_sampling_can_stop_after_bounded_prefix(tmp_path: Path) -> None:
    path = tmp_path / "bounded.jsonl"
    path.write_text('{"ok": 1}\n{"ok": 2}\n{"broken": \n', encoding="utf-8")

    samples, malformed, detail = _sample_jsonl_payload_with_detail(
        path,
        max_samples=2,
        jsonl_dict_only=True,
        scan_full=False,
    )

    assert [_as_dict(sample)["ok"] for sample in samples] == [1, 2]
    assert malformed == 0
    assert detail is None
