from __future__ import annotations

from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.services import acquisition_streams
from polylogue.sources.parsers.base import RawConversationData
from polylogue.types import Provider


async def test_iter_raw_record_stream_logs_make_raw_record_value_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def _raw_stream(*args, **kwargs):
        del args, kwargs
        yield RawConversationData(
            raw_bytes=b'{"id":"broken"}',
            source_path=str(tmp_path / "broken.json"),
            provider_hint=Provider.CHATGPT,
        )

    def _raise(*args, **kwargs):
        del args, kwargs
        raise ValueError("boom")

    warnings: list[tuple[str, dict[str, object]]] = []

    def _record_warning(event: str, **kwargs: object) -> None:
        warnings.append((event, kwargs))

    monkeypatch.setattr(acquisition_streams, "iter_source_raw_stream", _raw_stream)
    monkeypatch.setattr(acquisition_streams, "make_raw_record", _raise)
    monkeypatch.setattr(acquisition_streams.logger, "warning", _record_warning)

    items = [item async for item in acquisition_streams.iter_raw_record_stream(Source(name="chatgpt", path=tmp_path))]

    assert items == []
    assert warnings == [
        (
            "Skipping raw payload",
            {
                "source": "chatgpt",
                "path": str(tmp_path / "broken.json"),
                "error": "boom",
            },
        )
    ]
